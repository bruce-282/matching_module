#!/usr/bin/env python3
"""
Depth 이미지와 selected_points를 사용하여 PCD를 생성하고 프로젝션하는 스크립트
"""

import sys
from pathlib import Path
import argparse
import yaml
import numpy as np
import cv2
import open3d as o3d

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils.image_utils import read_image
from core.utils.camera_utils import create_default_camera, undistort_image
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


def main():
    parser = argparse.ArgumentParser(
        description="Depth 이미지와 selected_points를 사용하여 PCD 프로젝션"
    )

    parser.add_argument(
        "--teaching_config",
        type=str,
        required=True,
        help="Teaching config YAML 파일 경로",
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

    args = parser.parse_args()

    # 출력 디렉토리 생성
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Depth 이미지 로드
    path_match_source, selected_points = load_teaching_config(args.teaching_config)


    # 소스 이미지 로드
    # source_image = read_image(path_match_source)
    # if source_image is None:
    #     print(f"오류: 소스 이미지를 로드할 수 없습니다: {path_match_source}")
    #     return

    #print(f"Depth 이미지 로드 중: {args.depth_image}")
    depth_image = read_image(path_match_source)
    if depth_image is None:
        print(f"오류: Depth 이미지를 로드할 수 없습니다: {path_match_source}")
        return

    # Depth 이미지가 3차원인 경우 첫 번째 채널만 사용
    if len(depth_image.shape) == 3:
        depth_image = depth_image[:, :, 0]

    print(f"Depth 이미지 shape: {depth_image.shape}")

    # Selected points 로드

    result1_3d, result2_3d, result3_3d = load_selected_points(selected_points)
    print(f"Point L: {result1_3d}")
    print(f"Point R: {result2_3d}")
    print(f"Point U: {result3_3d}")

    # Plane normal 계산
    plane_normal = compute_plane_normal(result1_3d, result2_3d, result3_3d)
    print(f"Plane normal: {plane_normal}")

    # Center point 계산
    center_point_3d = (result1_3d + result2_3d + result3_3d) / 3
    print(f"Center point: {center_point_3d}")

    # 카메라 설정 로드

    
    # Camera 객체 생성
    camera = create_default_camera( image_size=(depth_image.shape[1], depth_image.shape[0]))

    # Texture 이미지 로드 (선택사항)
    texture_image = None
    if args.texture_image:
        print(f"Texture 이미지 로드 중: {args.texture_image}")
        texture_image = read_image(args.texture_image)
        if texture_image is None:
            print(f"경고: Texture 이미지를 로드할 수 없습니다: {args.texture_image}")

    undistorted_depth_image = undistort_image(depth_image, camera.get_intrinsic_matrix(), camera.get_distortion_coeffs())
    # Point cloud 생성
    print("Point cloud 생성 중...")
    pcd = create_point_cloud_from_depth_image(
        depth_image=undistorted_depth_image,
        intrinsic=camera.get_intrinsic_matrix(),
        texture_image=texture_image,
    )
    pcd.scale(1000.0, center=[0, 0, 0])
    # 모든 포인트에 하얀색 설정
    num_points = len(pcd.points)
    white_colors = np.ones((num_points, 3), dtype=np.float64)  # [1, 1, 1] for all points
    pcd.colors = o3d.utility.Vector3dVector(white_colors)

    # PCD 저장
    pcd_path = output_dir / "pointcloud_transformed.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    print(f"Point cloud 저장: {pcd_path}")

    # Camera 위치 계산 및 변환
    print("Camera 위치 계산 중...")
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
    print(f"Camera transform: {camera_transform}")
    
    # 세 포인트에 camera_transform 적용
    transform_inv = np.linalg.inv(camera_transform)
    
    def transform_point(point_3d):
        """3D 포인트를 homogeneous 좌표로 변환 후 transform 적용"""
        point_homogeneous = np.append(point_3d, 1.0)  # [x, y, z, 1]
        transformed_homogeneous = transform_inv @ point_homogeneous
        return transformed_homogeneous[:3]  # [x, y, z]
    
    result1_3d_transformed = transform_point(result1_3d)
    result2_3d_transformed = transform_point(result2_3d)
    result3_3d_transformed = transform_point(result3_3d)
    
    print(f"변환 전 - Point L: {result1_3d}, Point R: {result2_3d}, Point U: {result3_3d}")
    print(f"변환 후 - Point L: {result1_3d_transformed}, Point R: {result2_3d_transformed}, Point U: {result3_3d_transformed}")
    
    # path_match_source 파일명에 _transformed 추가
    path_obj = Path(path_match_source)
    transformed_path_match_source = str(
         f"{path_obj.stem}_transformed{path_obj.suffix}"
    )
    
    # 변환된 포인트를 selected_points 형식으로 저장
    transformed_points_path = output_dir / "transformed.matcher.teaching.param.yaml"
    with open(transformed_points_path, "w", encoding="utf-8") as f:
        # path_match_source를 따옴표로 감싸서 저장
        f.write(f'path_match_source: "{transformed_path_match_source}"\n\n')
        f.write("selected_points:\n")
        f.write(f'  L:\n')
        f.write(f'    x: {result1_3d_transformed[0]}\n')
        f.write(f'    y: {result1_3d_transformed[1]}\n')
        f.write(f'    z: {result1_3d_transformed[2]}\n')
        f.write(f'  R:\n')
        f.write(f'    x: {result2_3d_transformed[0]}\n')
        f.write(f'    y: {result2_3d_transformed[1]}\n')
        f.write(f'    z: {result2_3d_transformed[2]}\n')
        f.write(f'  U:\n')
        f.write(f'    x: {result3_3d_transformed[0]}\n')
        f.write(f'    y: {result3_3d_transformed[1]}\n')
        f.write(f'    z: {result3_3d_transformed[2]}\n')
     
        f.write(f'camera_transform: {transform_inv.tolist()}\n')

    
    # PCD 변환
    pcd_transformed = pcd.transform(transform_inv)

    def get_scaled_point(point, scale):
        point_3d = np.array(point) / scale
        return point_3d


    pcd_transformed = add_3d_points_to_pcd(pcd_transformed, [get_scaled_point(result1_3d_transformed, 1000.0), get_scaled_point(result2_3d_transformed, 1000.0), get_scaled_point(result3_3d_transformed, 1000.0)])

    # PCD 클리핑
    pcd_clipped = clip_pointcloud_by_depth(
        pcd_transformed,
        near_z=0.0,
        far_z=float(camera_distance + camera_distance * 0.1),
    )

    pcd_clipped.transform(camera_transform)
    o3d.io.write_point_cloud(str(output_dir / "clipped.ply"), pcd_clipped)

    # 이미지 크기 결정
    if texture_image is not None:
        image_size = (texture_image.shape[1], texture_image.shape[0])
    else:
        image_size = (depth_image.shape[1], depth_image.shape[0])

    # PCD를 이미지로 프로젝션
    print("PCD를 이미지로 프로젝션 중...")
    front_view_image = project_open3d_pcd_to_image(
        pcd=pcd_clipped,
        intrinsic_matrix=camera.get_intrinsic_matrix(),
        image_size=image_size,  # (width, height)
    )

    # 대비 조정
    if args.contrast > 0:
        front_view_image = cv2.convertScaleAbs(
            front_view_image, alpha=args.contrast
        )

    # 결과 이미지 저장
    output_image_path = output_dir / f"{transformed_path_match_source}"
    cv2.imwrite(
        str(output_image_path),
        cv2.cvtColor(front_view_image, cv2.COLOR_RGB2BGR),
    )
    print(f"프로젝션 이미지 저장: {output_image_path}")

    print("완료!")


if __name__ == "__main__":
    main()

