#!/usr/bin/env python3
"""
매칭만 수행하는 스크립트 (2D 매칭 + RANSAC + 선택적 3D 매칭, 후처리 제외).
Source/Target를 PLY로 줄 경우 config의 image_size만 사용하고, 가상 intrinsic으로 역투영해
이미지·depth를 생성한 뒤 매칭에 사용.
"""

import sys
import os
import glob
import argparse
import shutil
import warnings
import logging
from typing import Optional

import yaml
import copy
import numpy as np
import open3d as o3d

# torchvision 경고 숨기기
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.matchers.matcher import Matcher
from core.utils.image_utils import read_image
from core.utils.logger_utils import setup_logger
from core.utils.pcd_utils import create_point_cloud_from_depth_image
from core.utils.geometry_utils import (
    project_pcd_to_image,
    project_pcd_to_depth_image,
)
from core.utils.io_utils import create_camera_from_yaml_config


def build_intrinsic_from_config(config):
    """config의 camera_intrinsics로 3x3 intrinsic 행렬 생성."""
    ci = config.get("camera_intrinsics", {})
    return np.array(
        [
            [ci.get("fx", 0), 0, ci.get("cx", 0)],
            [0, ci.get("fy", 0), ci.get("cy", 0)],
            [0, 0, 1],
        ]
    )


def build_virtual_intrinsic(width, height):
    """이미지 크기만으로 가상 pinhole intrinsic 생성 (fx=fy=max(w,h), cx=w/2, cy=h/2)."""
    f = float(max(width, height))
    cx, cy = width / 2.0, height / 2.0
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


def save_matching_results_yaml(
    output_path: str,
    base_name: str,
    matches: dict,
    filtered_matches: dict,
    result_3d: Optional[dict],
) -> None:
    """주요 매칭 결과를 YAML 파일로 저장."""
    data = {
        "base_name": base_name,
        "matches_2d": {
            "total": len(matches["keypoints0"]),
            "filtered": len(filtered_matches["filtered_kpts0"]),
        },
    }
    if result_3d is not None:
        data["pose_estimation"] = {
            "method": result_3d.get("pose_estimation_method", "unknown"),
            "num_inliers": result_3d.get("num_inliers"),
            "fitness": result_3d.get("fitness"),
            "inlier_rmse": result_3d.get("inlier_rmse"),
        }
        if "chamfer_distance" in result_3d:
            data["chamfer_distance"] = float(result_3d["chamfer_distance"])
        # 4x4 transformation matrix (list of lists for YAML)
        T = result_3d["transformation"]
        data["transformation"] = (
            T.tolist() if hasattr(T, "tolist") else [[float(x) for x in row] for row in T]
        )
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _pcd_to_images(pcd, config):
    """PointCloud를 가상 intrinsic으로 역투영해 (color_image, depth_image) 반환."""
    if not pcd.has_points():
        return None, None
    points_3d = np.asarray(pcd.points)
    colors = None
    if pcd.has_colors():
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)

    ims = config.get("image_size", {})
    w, h = ims.get("width", 640), ims.get("height", 480)
    image_size = (w, h)
    K = build_virtual_intrinsic(w, h)
    depth_max = config.get("depth_max", 5000.0)
    depth_range = (0.1, float(depth_max))

    depth_image = project_pcd_to_depth_image(
        points_3d,
        intrinsic_matrix=K,
        image_size=image_size,
        depth_range=depth_range,
    )
    color_image = project_pcd_to_image(
        points_3d,
        colors=colors,
        intrinsic_matrix=K,
        image_size=image_size,
        depth_range=depth_range,
    )
    return color_image, depth_image


def ply_to_images(ply_path, config):
    """
    PLY를 가상 intrinsic + config의 image_size로 역투영해
    컬러 이미지와 depth 이미지를 생성.
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    return _pcd_to_images(pcd, config)


def ply_merge_to_images(ply_paths, config):
    """
    여러 PLY를 합쳐 하나의 PointCloud로 만든 뒤 역투영.
    ply_paths: 경로 리스트 (예: ["a.ply", "b.ply"])
    """
    pcd = o3d.io.read_point_cloud(ply_paths[0])
    for p in ply_paths[1:]:
        pcd += o3d.io.read_point_cloud(p)
    return _pcd_to_images(pcd, config)


def main():
    """메인 함수"""

    parser = argparse.ArgumentParser(
        description="매칭만 수행 (2D + RANSAC + 선택적 3D 매칭, 후처리 제외)"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Configuration file path (YAML)",
    )
    parser.add_argument(
        "--template_param_path",
        type=str,
        default=None,
        help="Template parameter file path (YAML). --target_ply 와 --source_ply 둘 다 줄 때는 생략 가능 (config 가상 intrinsic 사용)",
    )
    parser.add_argument(
        "--source_ply",
        type=str,
        default=None,
        help='Source PLY. ",": 같은 target에 대해 source만 순차 교체 (런타임 중 재로드). "+": 여러 PLY 합쳐서 한 source로 사용',
    )
    parser.add_argument(
        "--target_ply",
        type=str,
        default=None,
        help="Target PLY path. 지정 시 가상 intrinsic으로 역투영해 target 이미지/depth 생성 (한 번만 실행)",
    )

    args = parser.parse_args()

    # Load configuration file (YAML)
    try:
        with open(args.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config_path}")
        return
    except yaml.YAMLError as e:
        print(f"YAML file parsing failed: {e}")
        return

    # template_param: 파일에서 로드하거나, target_ply+source_ply 둘 다 있을 때는 config로 생성
    if args.template_param_path is not None:
        try:
            with open(args.template_param_path, "r", encoding="utf-8") as f:
                template_param = yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Template parameter file not found: {args.template_param_path}")
            return
        except yaml.YAMLError as e:
            print(f"YAML file parsing failed: {e}")
            return
    elif args.target_ply and args.source_ply:
        # 둘 다 PLY → 같은 가상 intrinsic만 쓰므로 config로 최소 template 생성
        ims = config.get("image_size", {})
        w, h = ims.get("width", 640), ims.get("height", 480)
        f = float(max(w, h))
        template_param = {
            "camera_intrinsics": {"fx": f, "fy": f, "cx": w / 2.0, "cy": h / 2.0},
            "camera_distortions": {"k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0},
            "image_size": {"width": w, "height": h},
            "path_match_source": "",
            "selected_points": {"L": {"x": 0, "y": 0, "z": 1000}, "R": {"x": 100, "y": 0, "z": 1000}, "U": {"x": 50, "y": 50, "z": 1000}},
        }
    else:
        print("--template_param_path 가 필요합니다. (--target_ply, --source_ply 둘 다 주면 생략 가능)")
        return

    # 로거 설정
    if config.get("debug_mode", False):
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = setup_logger(__name__)

    # Matcher 인스턴스 생성
    matcher = Matcher(config=config, template_param=template_param)
    matcher.init_config(config=config, template_param=template_param)

    input_dir = config.get("input_dir", "datasets")
    depth_files = glob.glob(os.path.join(input_dir, "*_depth.tif"))
    if not args.target_ply and not depth_files:
        logger.warning(f"No *_depth.tif files found in {input_dir}. Use --target_ply or set input_dir.")
        return

    ims = config.get("image_size", {})
    w, h = ims.get("width", 640), ims.get("height", 480)
    virtual_camera_config = {
        "camera_intrinsics": {"fx": float(max(w, h)), "fy": float(max(w, h)), "cx": w / 2.0, "cy": h / 2.0},
        "camera_distortions": {"k1": 0, "k2": 0, "p1": 0, "p2": 0, "k3": 0},
        "image_size": {"width": w, "height": h},
    }

    # Source: PLY 역투영 또는 이미지 경로
    # ",": 같은 target에 대해 source만 순차 교체 (런타임 중 한 번씩 로드). "+": 여러 PLY 합쳐서 한 source
    source_image = None
    source_depth = None
    source_ply_paths = None
    sequence_sources = False  # True면 target_ply 블록에서 source를 하나씩 로드하며 매칭
    if args.source_ply:
        if "," in args.source_ply and "+" not in args.source_ply:
            source_paths = [p.strip() for p in args.source_ply.split(",") if p.strip()]
            sequence_sources = len(source_paths) > 1
        elif "+" in args.source_ply:
            source_paths = [p.strip() for p in args.source_ply.split("+") if p.strip()]
        else:
            source_paths = [args.source_ply.strip()]
        if not source_paths:
            logger.error("Source PLY path가 비어 있습니다.")
            return
        if not all(os.path.isfile(p) for p in source_paths):
            logger.error(f"Source PLY not found: {source_paths}")
            return
        source_ply_paths = source_paths
        # 순차 모드(,) + target_ply 이면 source는 루프 안에서 로드
        if not (sequence_sources and args.target_ply):
            if len(source_paths) == 1:
                logger.info(
                    f"Building source from PLY: {source_paths[0]} (가상 intrinsic, image_size={w}x{h})"
                )
                source_image, source_depth = ply_to_images(source_paths[0], config)
            else:
                logger.info(
                    f"Building source from merged PLYs: {source_paths} (가상 intrinsic, image_size={w}x{h})"
                )
                source_image, source_depth = ply_merge_to_images(source_paths, config)
            if source_image is None or source_depth is None:
                logger.error("Failed to project PLY to source image/depth")
                return
            matcher.camera_source = create_camera_from_yaml_config(virtual_camera_config)
    else:
        path_match_source = (
            template_param.get("path_match_source")
            or template_param.get("matching_model", {}).get("path_match_source")
        )
        if path_match_source is None:
            logger.error("path_match_source를 찾을 수 없습니다. template_param 또는 --source_ply 사용.")
            return
        intrinsic_matrix = build_intrinsic_from_config(config)
        image_size_dict = config.get("image_size", {})
        source_image = read_image(
            path_match_source,
            width=image_size_dict.get("width"),
            height=image_size_dict.get("height"),
            intrinsic_matrix=intrinsic_matrix,
        )
        if source_image is None:
            logger.error(f"Source image not found: {path_match_source}")
            return

    output_dir = config.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)

    def run_one(
        target_texture,
        target_depth,
        target_texture_path,
        target_depth_path,
        base_name,
        src_image=None,
        src_depth=None,
        src_ply_paths=None,
    ):
        """한 번의 매칭 실행 + 변환 PLY 저장. src_* 미지정 시 기존 source_image/depth 사용."""
        s_img = src_image if src_image is not None else source_image
        s_dep = src_depth if src_depth is not None else source_depth
        pcd_paths = src_ply_paths if src_ply_paths is not None else source_ply_paths
        matches, filtered_matches, result_3d = matcher.run_pipeline_matching_only(
            target_texture=target_texture,
            target_depth=target_depth,
            source_image=s_img,
            source_depth=s_dep,
            target_texture_path=target_texture_path,
            target_depth_path=target_depth_path,
            output_dir=output_dir,
        )
        if result_3d is not None:
            try:
                if pcd_paths:
                    pcd_source = o3d.io.read_point_cloud(pcd_paths[0])
                    for p in pcd_paths[1:]:
                        pcd_source += o3d.io.read_point_cloud(p)
                else:
                    src_for_pcd = s_img
                    if matcher.config.get("image_undistortion", False):
                        src_for_pcd = matcher.camera_source.undistort_image(s_img)
                    depth_src = (
                        src_for_pcd[:, :, 0]
                        if len(src_for_pcd.shape) == 3
                        else src_for_pcd
                    )
                    pcd_source = create_point_cloud_from_depth_image(
                        depth_src,
                        matcher.camera_source.get_intrinsic_matrix(),
                        texture_image=None,
                    )
                pcd_transformed = copy.deepcopy(pcd_source)
                pcd_transformed.transform(result_3d["transformation"])
                ply_path = os.path.join(output_dir, f"{base_name}_source_transformed.ply")
                pts32 = np.asarray(pcd_transformed.points, dtype=np.float32)
                pcd_transformed.points = o3d.utility.Vector3dVector(pts32.astype(np.float64))
                if pcd_transformed.has_colors():
                    cols = np.asarray(pcd_transformed.colors, dtype=np.float32)
                    pcd_transformed.colors = o3d.utility.Vector3dVector(cols.astype(np.float64))

                o3d.io.write_point_cloud(ply_path, pcd_transformed)
                logger.info(f"Saved transformed source PLY: {ply_path}")
            except Exception as e:
                logger.warning(f"Failed to save transformed source PLY: {e}")
        # 주요 결과 YAML 저장
        result_yaml_path = os.path.join(output_dir, f"{base_name}_result.yaml")
        try:
            save_matching_results_yaml(
                result_yaml_path,
                base_name,
                matches,
                filtered_matches,
                result_3d,
            )
            logger.info(f"Saved matching results: {result_yaml_path}")
        except Exception as e:
            logger.warning(f"Failed to save result YAML: {e}")

        chamfer_str = ""
        if result_3d is not None and "chamfer_distance" in result_3d:
            chamfer_str = f", chamfer_distance={result_3d['chamfer_distance']:.4f}"
        logger.info(
            f"Success - {base_name}: "
            f"matches={len(matches['keypoints0'])}, "
            f"filtered={len(filtered_matches['filtered_kpts0'])}, "
            f"3d_result={'OK' if result_3d is not None else 'N/A'}"
            f"{chamfer_str}"
        )
        return matches, filtered_matches, result_3d


    if args.target_ply:
        # Target PLY 1회 로드
        if not os.path.isfile(args.target_ply):
            logger.error(f"Target PLY not found: {args.target_ply}")
            return
        logger.info(
            f"Building target from PLY: {args.target_ply} (가상 intrinsic, image_size={w}x{h})"
        )
        target_texture, target_depth = ply_to_images(args.target_ply, config)
        if target_texture is None or target_depth is None:
            logger.error("Failed to project PLY to target image/depth")
            return
        matcher.camera_target = create_camera_from_yaml_config(virtual_camera_config)
        target_stem = os.path.splitext(os.path.basename(args.target_ply))[0]

        if sequence_sources:
            # 런타임 중 source만 바꿔가며 매칭 (target는 한 번만 로드)
            for src_path in source_ply_paths:
                source_stem = os.path.splitext(os.path.basename(src_path))[0]
                base_name = f"{target_stem}_{source_stem}"
                logger.info(
                    f"Loading source: {src_path} (가상 intrinsic, image_size={w}x{h})"
                )
                src_image, src_depth = ply_to_images(src_path, config)
                if src_image is None or src_depth is None:
                    logger.error(f"Failed to project PLY to source image/depth: {src_path}")
                    continue
                matcher.camera_source = create_camera_from_yaml_config(
                    virtual_camera_config
                )
                fake_path = f"{base_name}.ply"
                try:
                    target_ply_out = os.path.join(output_dir, f"{base_name}_target.ply")
                    shutil.copy2(args.target_ply, target_ply_out)
                    logger.info(f"Saved target PLY: {target_ply_out}")
                    run_one(
                        target_texture,
                        target_depth,
                        fake_path,
                        fake_path,
                        base_name,
                        src_image=src_image,
                        src_depth=src_depth,
                        src_ply_paths=[src_path],
                    )
                except Exception as e:
                    logger.error(f"Failed - {base_name}: {e}")
        else:
            # 기존: target·source 한 쌍만 실행
            source_stems = [
                os.path.splitext(os.path.basename(p))[0] for p in source_ply_paths
            ]
            source_stem = "_".join(source_stems)
            base_name = f"{target_stem}_{source_stem}"
            fake_path = f"{base_name}.ply"
            try:
                target_ply_out = os.path.join(output_dir, f"{base_name}_target.ply")
                shutil.copy2(args.target_ply, target_ply_out)
                logger.info(f"Saved target PLY: {target_ply_out}")
                run_one(
                    target_texture, target_depth,
                    fake_path, fake_path,
                    base_name,
                )
            except Exception as e:
                logger.error(f"Failed - {base_name}: {e}")
    else:
        for depth_file in depth_files:
            base_name = os.path.basename(depth_file).replace("_depth.tif", "")
            texture_file = os.path.join(input_dir, f"{base_name}_texture.png")

            logger.info(f"Processing: {depth_file}")

            if not os.path.exists(texture_file):
                logger.warning(f"Texture file not found: {texture_file}. Skipping.")
                continue

            try:
                target_texture = read_image(texture_file)
                target_depth = read_image(depth_file)
                run_one(
                    target_texture, target_depth,
                    texture_file, depth_file,
                    base_name,
                    source_image=source_image,
                    source_depth=None,
                    source_ply_paths=None,
                )
            except Exception as e:
                logger.error(f"Failed - {base_name}: {e}")
                continue

    matcher.cleanup()
    logger.info("Execution completed")


if __name__ == "__main__":
    main()
