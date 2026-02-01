#!/usr/bin/env python3
"""
Depth 이미지와 selected_points를 사용하여 PCD를 생성하고 프로젝션하는 스크립트
"""

import sys
import logging
from pathlib import Path
import argparse
import yaml
import numpy as np
import cv2
import open3d as o3d

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils.logger_utils import get_logger
from core.utils.image_utils import read_image

logger = get_logger(__name__, level=logging.INFO)
from core.utils.camera_utils import create_default_camera, undistort_image
from core.utils.io_utils import create_camera_from_yaml_config
from core.utils.pcd_utils import (
    create_point_cloud_from_depth_image,
    compute_plane_normal,
    clip_pointcloud_by_depth,
    add_3d_points_to_pcd,
)
from core.utils.geometry_utils import (
    project_open3d_pcd_to_image,
    create_transform_matrix_from_vectors,
)


def load_selected_points(selected_points):
    """YAML 파일에서 selected_points 로드"""

    def get_point(selected_points, point_name):
        return np.array([
            selected_points[point_name]["x"],
            selected_points[point_name]["y"],
            selected_points[point_name].get("z", 0),
        ])
    
    return get_point(selected_points, "L"), get_point(selected_points, "R"), get_point(selected_points, "U")


def load_camera_config(config_path):
    """Config 파일에서 카메라 설정 로드"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    camera_intrinsics = config.get("camera_intrinsics", {})
    camera_distortions = config.get("camera_distortions", {})
    image_size = config.get("image_size", {})
    
    return {
        "intrinsics": camera_intrinsics,
        "distortions": camera_distortions,
        "image_size": image_size,
    }

def load_teaching_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "selected_points" not in config:
        raise ValueError("selected_points가 YAML 파일에 없습니다.")
    
    selected_points = config["selected_points"]

    if "path_match_source" not in config:
        raise ValueError("path_match_source가 YAML 파일에 없습니다.")
    path_match_source = config["path_match_source"]

    return path_match_source, selected_points


def find_depth_and_result_in_folder(folder: Path):
    """
    폴더 내에서 *_match_texture_result.yaml과 같은 prefix의 *_match_depth.tif 쌍을 찾습니다.
    예: 20260127_000428_match_texture_result.yaml -> 20260127_000428_match_depth.tif

    Returns:
        (depth_path, result_yaml_path) 또는 None (쌍이 없으면)
    """
    result_yamls = list(folder.glob("*_match_texture_result.yaml"))
    if not result_yamls:
        return None
    for result_path in sorted(result_yamls):
        # 20260127_000428_match_texture_result.yaml -> 20260127_000428
        prefix = result_path.stem.replace("_match_texture_result", "")
        depth_path = folder / f"{prefix}_match_depth.tif"
        if depth_path.exists():
            return depth_path, result_path
    return None


def load_selected_points_from_result_yaml(yaml_path):
    """
    *_match_texture_result.yaml에서 selected_points 로드.
    - selected_points (L/R/U) 또는 matching_model.selected_points 지원
    - transformed_points_3d (pointL/pointR/pointU) 형식이면 L/R/U로 변환하여 반환
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    selected_points = config.get("selected_points") or config.get("matching_model", {}).get("selected_points")
    if selected_points:
        return selected_points

    # transformed_points_3d (pointL, pointR, pointU) 형식
    t3 = config.get("transformed_points_3d")
    if t3 and "pointL" in t3 and "pointR" in t3 and "pointU" in t3:
        return {
            "L": {"x": t3["pointL"]["x"], "y": t3["pointL"]["y"], "z": t3["pointL"].get("z", 0)},
            "R": {"x": t3["pointR"]["x"], "y": t3["pointR"]["y"], "z": t3["pointR"].get("z", 0)},
            "U": {"x": t3["pointU"]["x"], "y": t3["pointU"]["y"], "z": t3["pointU"].get("z", 0)},
        }

    raise ValueError(
        f"selected_points 또는 transformed_points_3d(pointL/pointR/pointU)가 YAML에 없습니다: {yaml_path}"
    )


def can_process_folder(folder: Path) -> bool:
    """폴더에 depth+result 쌍 또는 teaching config가 있으면 True."""
    if find_depth_and_result_in_folder(folder) is not None:
        return True
    config_candidates = list(folder.glob("*matching_model.config.yml")) or list(folder.glob("*.yml"))
    config_candidates = [p for p in config_candidates if "_match_texture_result" not in p.name]
    return bool(config_candidates)


def collect_processable_subfolders(root: Path):
    """상위 폴더 아래에서 처리 가능한 하위 폴더를 재귀적으로 수집 (제너레이터)."""
    root = root.resolve()
    for path in sorted(root.rglob("*")):
        if path.is_dir() and can_process_folder(path):
            yield path


def _input_prefix(depth_image_path: Path) -> str:
    """입력 depth 파일명에서 prefix 추출 (예: 20260127_000428_match_depth.tif → 20260127_000428)."""
    stem = depth_image_path.stem
    if "_match_depth" in stem:
        return stem.replace("_match_depth", "")
    return stem


def process_one_folder(
    depth_image_path: Path,
    path_match_source: str,
    selected_points: dict,
    output_dir: Path,
    args,
    camera=None,
):
    """한 폴더에 대해 depth → PCD → 변환 → 프로젝션 이미지 저장까지 수행. camera가 None이면 이미지 크기로 기본 카메라 생성."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _input_prefix(depth_image_path)

    depth_image_path_str = str(depth_image_path)
    logger.info(f"Depth 이미지 로드 중: {depth_image_path_str}")
    depth_image = read_image(depth_image_path_str)
    if depth_image is None:
        logger.error(f"Depth 이미지를 로드할 수 없습니다: {depth_image_path_str}")
        return False

    if len(depth_image.shape) == 3:
        depth_image = depth_image[:, :, 0]

    result1_3d, result2_3d, result3_3d = load_selected_points(selected_points)
    plane_normal = compute_plane_normal(result1_3d, result2_3d, result3_3d)
    center_point_3d = (result1_3d + result2_3d + result3_3d) / 3

    if camera is None:
        camera = create_default_camera(image_size=(depth_image.shape[1], depth_image.shape[0]))

    texture_image = None
    if args.texture_image:
        texture_image = read_image(args.texture_image)

    depth_for_pcd = (
        undistort_image(depth_image, camera.get_intrinsic_matrix(), camera.get_distortion_coeffs())
        if args.undistort
        else depth_image
    )
    pcd = create_point_cloud_from_depth_image(
        depth_image=depth_for_pcd,
        intrinsic=camera.get_intrinsic_matrix(),
        texture_image=texture_image,
    )

    # matcher.py와 동일: 포인트는 /1000 (mm→m), 추가 후 PCD 전체 1000 스케일
    def get_scaled_point(point, scale):
        return np.array(point) / scale

    scaledL_3d = get_scaled_point(result1_3d, 1000.0)
    scaledR_3d = get_scaled_point(result2_3d, 1000.0)
    scaledU_3d = get_scaled_point(result3_3d, 1000.0)
    num_original = len(pcd.points)
    logger.info("L/R/U 포인트 PCD에 추가 (add_3d_points_to_pcd)")
    pcd = add_3d_points_to_pcd(pcd, [scaledL_3d, scaledR_3d, scaledU_3d])
    pcd.scale(1000.0, center=[0, 0, 0])

    num_points = len(pcd.points)
    colors = np.ones((num_points, 3), dtype=np.float64)  # 기본 흰색
    colors[num_original:] = [0.0, 1.0, 0.0]  # L/R/U 포인트는 초록색
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd_path = output_dir / f"{prefix}_pointcloud_transformed.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    logger.info(f"PCD 저장: {pcd_path}")

    lr_diff = result2_3d - result1_3d
    lr_distance = np.linalg.norm(lr_diff)
    camera_distance = lr_distance * 3.0
    camera_pos = center_point_3d + plane_normal * camera_distance
    front_vector = -center_point_3d + camera_pos
    front_vector = front_vector / np.linalg.norm(front_vector)
    lr_vector = lr_diff / lr_distance
    up_vector = np.cross(front_vector, lr_vector)
    up_vector = up_vector / np.linalg.norm(up_vector)

    camera_transform = create_transform_matrix_from_vectors(
        right_vector=lr_vector,
        up_vector=up_vector,
        front_vector=front_vector,
        position=camera_pos,
    )
    transform_inv = np.linalg.inv(camera_transform)

    def transform_point(point_3d):
        point_homogeneous = np.append(point_3d, 1.0)
        return (transform_inv @ point_homogeneous)[:3]

    result1_3d_transformed = transform_point(result1_3d)
    result2_3d_transformed = transform_point(result2_3d)
    result3_3d_transformed = transform_point(result3_3d)

    transformed_path_match_source = f"{prefix}_transformed.png"

    transformed_points_path = output_dir / f"{prefix}_transformed.matcher.teaching.param.yaml"
    logger.info(f"변환된 포인트 YAML 저장: {transformed_points_path}")
    with open(transformed_points_path, "w", encoding="utf-8") as f:
        f.write(f'path_match_source: "{transformed_path_match_source}"\n\n')
        f.write("selected_points:\n")
        f.write(f'  L:\n    x: {result1_3d_transformed[0]}\n    y: {result1_3d_transformed[1]}\n    z: {result1_3d_transformed[2]}\n')
        f.write(f'  R:\n    x: {result2_3d_transformed[0]}\n    y: {result2_3d_transformed[1]}\n    z: {result2_3d_transformed[2]}\n')
        f.write(f'  U:\n    x: {result3_3d_transformed[0]}\n    y: {result3_3d_transformed[1]}\n    z: {result3_3d_transformed[2]}\n')
        f.write(f'camera_transform: {transform_inv.tolist()}\n')

    pcd_transformed = pcd.transform(transform_inv)

    # 포인트는 이미 add_3d_points_to_pcd로 첫 PCD에 그려져 있어 변환 시 함께 변환됨
    pcd_clipped = clip_pointcloud_by_depth(
        pcd_transformed,
        near_z=0.0,
        far_z=float(camera_distance + camera_distance * 0.1),
    )
    pcd_clipped.transform(camera_transform)
    clipped_path = output_dir / f"{prefix}_clipped.ply"
    o3d.io.write_point_cloud(str(clipped_path), pcd_clipped)
    logger.info(f"클리핑된 PCD 저장: {clipped_path}")

    if texture_image is not None:
        image_size = (texture_image.shape[1], texture_image.shape[0])
    elif camera is not None and getattr(camera, "image_size", None):
        image_size = camera.image_size  # (width, height) from global config
    else:
        image_size = (depth_image.shape[1], depth_image.shape[0])
    front_view_image = project_open3d_pcd_to_image(
        pcd=pcd_clipped,
        intrinsic_matrix=camera.get_intrinsic_matrix(),
        image_size=image_size,
    )
    if args.contrast > 0:
        front_view_image = cv2.convertScaleAbs(front_view_image, alpha=args.contrast)

    output_image_path = output_dir / transformed_path_match_source
    cv2.imwrite(str(output_image_path), cv2.cvtColor(front_view_image, cv2.COLOR_RGB2BGR))
    logger.info(f"프로젝션 이미지 저장: {output_image_path}")
    return True


def _resolve_inputs(parser, args, input_folder, teaching_config_path):
    """입력(teaching_config 또는 input_folder)으로부터 depth_image_path, path_match_source, selected_points 해석."""
    depth_image_path = None
    path_match_source = None
    selected_points = None

    if teaching_config_path is not None and input_folder is None:
        path_match_source, selected_points = load_teaching_config(teaching_config_path)
        depth_image_path = Path(path_match_source)
        return depth_image_path, path_match_source, selected_points

    if input_folder is None:
        return None, None, None

    pair = find_depth_and_result_in_folder(input_folder)
    if pair is not None:
        depth_image_path, result_yaml_path = pair
        selected_points = load_selected_points_from_result_yaml(result_yaml_path)
        path_match_source = depth_image_path.name
        return depth_image_path, path_match_source, selected_points

    config_candidates = list(input_folder.glob("*matching_model.config.yml")) or list(input_folder.glob("*.yml"))
    config_candidates = [p for p in config_candidates if "_match_texture_result" not in p.name]
    if not config_candidates:
        parser.error(
            f"--input_folder 내에 depth+result 쌍 또는 teaching config가 없습니다: {input_folder}"
        )
    # 배치 시에는 teaching_config_path가 None으로 넘어오므로 폴더 내 첫 config 사용
    if teaching_config_path is not None:
        p = Path(teaching_config_path)
        path_to_load = teaching_config_path if p.is_absolute() else str(input_folder / p)
    else:
        path_to_load = str(config_candidates[0])
    path_match_source, selected_points = load_teaching_config(path_to_load)
    depth_image_path = input_folder / path_match_source
    return depth_image_path, path_match_source, selected_points


def main():
    parser = argparse.ArgumentParser(
        description="Depth 이미지와 selected_points를 사용하여 PCD 프로젝션"
    )

    parser.add_argument(
        "--teaching_config",
        type=str,
        default=None,
        help="Teaching config YAML 파일 경로 (--input_folder 사용 시 폴더 내 config 자동 탐색)",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default=None,
        help="입력 폴더 (depth 이미지·teaching config 기준 경로). 설정 시 path_match_source는 이 폴더 기준으로 해석",
    )
    # parser.add_argument(
    #     "--config",
    #     type=str,
    #     required=True,
    #     help="카메라 설정 YAML 파일 경로",
    # )
    parser.add_argument(
        "--texture_image",
        type=str,
        default=None,
        help="Texture 이미지 경로 (선택사항)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="출력 디렉토리",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=10.0,
        help="이미지 대비 조정 (기본값: 10.0)",
    )
    parser.add_argument(
        "--undistort",
        action="store_true",
        help="Depth 이미지 언디스토션 적용 (기본값: 미적용)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="--input_folder 지정 시, 하위 폴더를 재귀 탐색하여 처리 가능한 폴더마다 결과 생성",
    )
    parser.add_argument(
        "--camera_config",
        type=str,
        default=None,
        help="카메라 intrinsic YAML 경로 (camera_intrinsics, camera_distortions, image_size). 지정 시 전역으로 사용",
    )

    args = parser.parse_args()

    # 글로벌 카메라( intrinsic ) 로드 (선택)
    global_camera = None
    if args.camera_config:
        with open(args.camera_config, "r", encoding="utf-8") as f:
            camera_config = yaml.safe_load(f)
        if "camera_intrinsics" in camera_config and "camera_distortions" in camera_config:
            global_camera = create_camera_from_yaml_config(camera_config)
            logger.info(f"글로벌 카메라 사용: {args.camera_config}")

    input_folder = Path(args.input_folder) if args.input_folder else None
    teaching_config_path = args.teaching_config
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch and input_folder is not None:
        # 배치: 상위 폴더 하위를 재귀 탐색하여 처리 가능한 폴더마다 결과 생성
        subfolders = list(collect_processable_subfolders(input_folder))
        if not subfolders:
            parser.error(f"--input_folder 내에 처리 가능한 하위 폴더가 없습니다: {input_folder}")
        logger.info(f"배치 처리: {len(subfolders)}개 폴더")
        root = input_folder.resolve()
        ok, fail = 0, 0
        for subfolder in subfolders:
            rel = subfolder.relative_to(root)
            try:
                depth_image_path, path_match_source, selected_points = _resolve_inputs(
                    parser, args, subfolder, None
                )
                if depth_image_path is None:
                    fail += 1
                    continue
                logger.info(f"[{rel}] 처리 중...")
                if process_one_folder(depth_image_path, path_match_source, selected_points, output_dir, args, camera=global_camera):
                    ok += 1
                else:
                    fail += 1
            except Exception as e:
                logger.warning(f"[{rel}] 오류: {e}")
                fail += 1
        logger.info(f"배치 완료: 성공 {ok}, 실패 {fail}")
        return

    # 단일 입력
    if input_folder is None and teaching_config_path is None:
        parser.error("--teaching_config 또는 --input_folder 중 하나는 반드시 지정해야 합니다.")

    depth_image_path, path_match_source, selected_points = _resolve_inputs(
        parser, args, input_folder, teaching_config_path
    )
    if depth_image_path is None:
        parser.error("입력을 해석할 수 없습니다.")
    process_one_folder(depth_image_path, path_match_source, selected_points, output_dir, args, camera=global_camera)
    logger.info("완료!")


if __name__ == "__main__":
    main()

# python scripts/project_pcd_with_points.py --input_folder datasets\NX4_0127 --output output_NX4_0127 --batch --camera_config  configs/NX4/matcher_config.yaml --input_folder datasets/NX4_0127 --batch
# 
