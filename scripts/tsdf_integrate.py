#!/usr/bin/env python3
"""
TSDF (Truncated Signed Distance Function) Volume Integration Module
여러 RGBD 이미지를 통합하여 3D 재구성을 수행.
매칭 결과 YAML의 transformation과 PLY에서 생성한 RGB/depth를 사용.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml

# 프로젝트 루트
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.utils.geometry_utils import project_pcd_to_depth_image, project_pcd_to_image


def build_virtual_intrinsic(width: int, height: int) -> np.ndarray:
    """이미지 크기만으로 가상 pinhole intrinsic 생성."""
    f = float(max(width, height))
    cx, cy = width / 2.0, height / 2.0
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])


def ply_to_rgbd(ply_path: str, config: dict, depth_scale: float = 1000.0) -> tuple:
    """
    PLY를 역투영해 RGBD 이미지 생성.

    Args:
        ply_path: PLY 파일 경로
        config: matcher config (image_size, depth_max 등)
        depth_scale: depth 단위 → meter 변환 (mm면 1000.0)

    Returns:
        (rgbd: o3d.geometry.RGBDImage, intrinsic: o3d.camera.PinholeCameraIntrinsic)
        또는 (None, None)
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        return None, None

    points_3d = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None

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

    # Open3D RGBD 생성 (depth: mm → meter via depth_scale)
    depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))
    color_o3d = o3d.geometry.Image(np.ascontiguousarray(color_image))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_o3d,
        depth=depth_o3d,
        depth_scale=depth_scale,
        depth_trunc=depth_max / depth_scale,
        convert_rgb_to_intensity=False,
    )
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=w, height=h,
        fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
    )
    print(f"Intrinsic: {intrinsic.intrinsic_matrix}")
    return rgbd, intrinsic


def load_pose_from_yaml(yaml_path: str, pose_scale: float = 1.0) -> np.ndarray:
    """YAML에서 transformation (4x4) 로드.
    pose_scale: translation 스케일 (매칭이 mm 단위면 0.001로 해서 meter로 변환)
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "transformation" not in data:
        raise ValueError(f"No 'transformation' in {yaml_path}")
    T = np.array(data["transformation"], dtype=np.float64)
    if pose_scale != 1.0:
        T = T.copy()
        T[:3, 3] *= pose_scale
    return T


def save_poses_yaml(camera_names: list, poses_list: list, out_path: str) -> None:
    """TSDF 통합에 사용한 카메라 이름·포즈를 한 YAML로 저장. 포즈는 이미 meter 단위."""
    data = {
        "reference": camera_names[0] if camera_names else "",
        "poses_meter": True,
        "cameras": [
            {"name": name, "transformation": T.tolist()}
            for name, T in zip(camera_names, poses_list)
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def parse_result_basenames(result_dir: str) -> list:
    """*_result.yaml 파일에서 (base, target, source, yaml_path) 추출.
    base_name 형식: {target_stem}_{source_stem} (예: 5_cp_6_cp, 2_cp_5_cp)
    """
    result_dir = Path(result_dir)
    pairs = []
    for p in result_dir.glob("*_result.yaml"):
        base = p.stem.replace("_result", "")
        # target_source 형식: 마지막 '_' 기준으로 분리 불가. 숫자_cp_숫자_cp 패턴 가정
        parts = base.split("_")
        if len(parts) >= 4:  # e.g. 5_cp_6_cp, 2_cp_5_cp
            target = "_".join(parts[:2])
            source = "_".join(parts[2:4])
            pairs.append((base, target, source, str(p)))
    return pairs


class TSDFVolumeIntegrator:
    """TSDF Volume을 사용한 3D 재구성 클래스"""

    def __init__(
        self,
        voxel_length=0.0005,
        sdf_trunc=0.02,
        volume_unit_resolution=32,
        depth_sampling_stride=1,
    ):
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            volume_unit_resolution=volume_unit_resolution,
            depth_sampling_stride=depth_sampling_stride,
        )

    def integrate_frame(
        self,
        rgbd: o3d.geometry.RGBDImage,
        intrinsic: o3d.camera.PinholeCameraIntrinsic,
        extrinsic: np.ndarray,
    ):
        """단일 RGBD 프레임을 volume에 통합"""
        self.volume.integrate(rgbd, intrinsic, extrinsic)

    def extract_mesh(self):
        """volume으로부터 triangle mesh 추출"""
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def extract_point_cloud(self):
        """volume으로부터 point cloud 추출"""
        return self.volume.extract_point_cloud()

    def reset(self):
        """volume 초기화"""
        self.volume.reset()


def integrate_rgbd_frames(
    rgbd_list,
    intrinsics_list,
    poses_list,
    voxel_length=0.0005,
    sdf_trunc=0.02,
    volume_unit_resolution=32,
    depth_sampling_stride=1,
):
    """
    여러 RGBD 프레임을 통합하여 3D 모델 생성

    Args:
        rgbd_list: RGBD 이미지 리스트
        intrinsics_list: 카메라 내부 파라미터 리스트
        poses_list: 카메라 포즈 리스트 (camera-to-world, 4x4)
        voxel_length: voxel 크기 (meter). 작을수록 해상도 상승, 메모리 증가
        sdf_trunc: truncation distance (meter)
        volume_unit_resolution: 볼륨 단위당 해상도
        depth_sampling_stride: 1=전체 픽셀, 2/4=서브샘플 (작을수록 고해상도)

    Returns:
        mesh, pcd
    """
    integrator = TSDFVolumeIntegrator(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        volume_unit_resolution=volume_unit_resolution,
        depth_sampling_stride=depth_sampling_stride,
    )

    for i, (rgbd, intrinsic, pose) in enumerate(zip(rgbd_list, intrinsics_list, poses_list)):
        print(f"Integrating frame {i} into the volume...")
        integrator.integrate_frame(rgbd, intrinsic, np.linalg.inv(pose))

    mesh = integrator.extract_mesh()
    pcd = integrator.extract_point_cloud()
    return mesh, pcd


def save_reconstruction(mesh, pcd, output_dir: str, name: str, flip_yz: bool = True):
    """재구성 결과 저장"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if flip_yz:
        flip_transform = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        mesh.transform(flip_transform)
        pcd.transform(flip_transform)

    # PLY 저장 시 "clamped color" 경고 방지: vertex color를 [0,1]로 클램핑
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    mesh_path = output_dir / f"{name}_mesh.ply"
    pcd_path = output_dir / f"{name}_pcd.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    print(f"[INFO] Mesh saved: {mesh_path}")
    print(f"[INFO] PCD saved: {pcd_path}")
    return str(mesh_path), str(pcd_path)


def main():
    parser = argparse.ArgumentParser(
        description="매칭 결과 YAML + PLY로 TSDF 통합하여 mesh/pcd 생성"
    )
    parser.add_argument("--config_path", type=str, required=True, help="matcher config YAML")
    parser.add_argument("--ply_dir", type=str, required=True, help="PLY 파일 디렉토리")
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="result YAML 디렉토리 (기본: config의 output_dir)",
    )
    parser.add_argument(
        "--result_files",
        type=str,
        nargs="+",
        default=None,
        help="사용할 result YAML 파일들 (예: 5_cp_6_cp_result.yaml 5_cp_8_cp_result.yaml)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="기준 프레임 (target) PLY stem. 예: 5_cp. result에서 target으로 쓰인 것만 사용",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="출력 디렉토리")
    parser.add_argument("--output_name", type=str, default="tsdf_fused", help="출력 파일명 prefix")
    parser.add_argument(
        "--voxel_length",
        type=float,
        default=0.0005,
        help="voxel 크기 (m). 작을수록 고해상도, 메모리 증가. 예: 0.0005=0.5mm",
    )
    parser.add_argument(
        "--sdf_trunc",
        type=float,
        default=0.02,
        help="truncation distance (m). 작을수록 날카로운 디테일 유지",
    )
    parser.add_argument(
        "--depth_sampling_stride",
        type=int,
        default=1,
        help="depth 서브샘플 (1=전체픽셀, 2/4=저해상도 빠름)",
    )
    parser.add_argument(
        "--volume_unit_resolution",
        type=int,
        default=32,
        help="ScalableTSDF 볼륨 단위 해상도 (32=고해상도)",
    )
    parser.add_argument(
        "--pose_scale",
        type=float,
        default=0.001,
        help="YAML pose의 translation 스케일 (mm→m: 0.001). 이미 meter면 1.0",
    )
    parser.add_argument("--no_flip_yz", action="store_true", help="Y/Z 축 뒤집기 비활성화")

    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    result_dir = args.result_dir or config.get("output_dir", "output")
    output_dir = args.output_dir or result_dir
    ply_dir = Path(args.ply_dir)

    # 사용할 (target, source, yaml_path) 수집
    if args.result_files:
        pairs = []
        for rf in args.result_files:
            p = Path(rf)
            if not p.is_absolute():
                p = Path(result_dir) / p
            if not p.exists():
                print(f"[WARN] Skip (not found): {p}")
                continue
            base = p.stem.replace("_result", "")
            parts = base.split("_")
            if len(parts) >= 4:
                target = "_".join(parts[:2])
                source = "_".join(parts[2:4])
                pairs.append((base, target, source, str(p)))
    else:
        pairs = parse_result_basenames(result_dir)
        if args.reference:
            pairs = [x for x in pairs if x[1] == args.reference]

    if not pairs:
        print("[ERROR] No result YAML found.")
        return 1

    # reference = 첫 target
    ref = pairs[0][1]
    rgbd_list = []
    intrinsics_list = []
    poses_list = []
    camera_names = []

    # Frame 0: reference (identity)
    ref_ply = ply_dir / f"{ref}.ply"
    if not ref_ply.exists():
        print(f"[ERROR] Reference PLY not found: {ref_ply}")
        return 1
    rgbd, intrinsic = ply_to_rgbd(str(ref_ply), config)
    if rgbd is None:
        print(f"[ERROR] Failed to create RGBD from {ref_ply}")
        return 1
    rgbd_list.append(rgbd)
    intrinsics_list.append(intrinsic)
    poses_list.append(np.eye(4))
    camera_names.append(ref)
    print(f"Frame 0 (reference): {ref}")

    # Frames 1..: source with pose from YAML
    for base, target, source, yaml_path in pairs:
        if target != ref:
            continue
        src_ply = ply_dir / f"{source}.ply"
        if not src_ply.exists():
            print(f"[WARN] Skip {source}: PLY not found")
            continue
        T = load_pose_from_yaml(yaml_path, pose_scale=args.pose_scale)
        rgbd, intrinsic = ply_to_rgbd(str(src_ply), config)
        if rgbd is None:
            print(f"[WARN] Skip {source}: RGBD creation failed")
            continue
        rgbd_list.append(rgbd)
        intrinsics_list.append(intrinsic)
        poses_list.append(T)
        camera_names.append(source)
        print(f"Frame {len(poses_list) - 1}: {source} (pose from {Path(yaml_path).name})")

    if len(rgbd_list) < 2:
        print("[WARN] Only 1 frame. Consider adding more result YAMLs.")

    mesh, pcd = integrate_rgbd_frames(
        rgbd_list, intrinsics_list, poses_list,
        voxel_length=args.voxel_length,
        sdf_trunc=args.sdf_trunc,
        volume_unit_resolution=args.volume_unit_resolution,
        depth_sampling_stride=args.depth_sampling_stride,
    )
    save_reconstruction(
        mesh, pcd, output_dir, args.output_name,
        flip_yz=not args.no_flip_yz,
    )

    # 통합에 사용한 포즈를 한 번에 YAML로 저장 (rerun_viewer 등에서 동일 포즈 사용)
    poses_yaml_path = Path(output_dir) / f"{args.output_name}_poses.yaml"
    save_poses_yaml(camera_names, poses_list, str(poses_yaml_path))
    print(f"[INFO] Poses saved: {poses_yaml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
