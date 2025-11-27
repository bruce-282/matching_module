#!/usr/bin/env python3
"""
Point Cloud 생성 스크립트

Depth와 Color 이미지를 받아서 Open3D를 이용해 Point Cloud를 생성하고 저장하는 스크립트

사용법:
    # 기본 사용
    python create_pointcloud.py --depth_image scripts/Depth.png --color_image scripts/Color.png --output output/pointcloud.ply

    # 폴더 처리
    python create_pointcloud.py --depth_folder datasets/depth/ --color_folder datasets/color/ --output_folder output/
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import logging
import sys
import open3d as o3d
from typing import Union, Tuple, Optional

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# 카메라 내부 파라미터 (사용자가 직접 수정)
CAMERA_INTRINSICS = {
    "fx": 2344.06988494,  # Focal length X
    "fy": 2344.40009342502,  # Focal length Y
    "cx": 989.06314625513,  # Principal point X
    "cy": 807.02989528271,  # Principal point Y
    "width": 2064,  # Image width
    "height": 1544,  # Image height
}

# camera_intrinsics:
#   fx: 2344.06988494
#   fy: 2344.40009342502
#   cx: 989.06314625513
#   cy: 807.02989528271
# camera_distortions:
#   k1: -0.24331290305526787
#   k2: 0.13922919417642093
#   p1: 0.0005252878633098153
#   p2: -0.0010237886757940777
#   k3: -0.01443719970450923


def detect_image_format(image_path: Path) -> str:
    """
    파일 시그니처로 실제 이미지 형식 감지

    Args:
        image_path: 이미지 파일 경로

    Returns:
        str: 실제 이미지 형식 ('tiff', 'jpeg', 'png', 'exr')
    """
    try:
        with open(image_path, "rb") as f:
            header = f.read(16)

        # 파일 시그니처 확인
        if header.startswith(b"\x49\x49\x2a\x00") or header.startswith(
            b"\x4d\x4d\x00\x2a"
        ):
            return "tiff"
        elif header.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        elif header.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        elif header.startswith(b"\x76\x2f\x31\x01"):
            return "exr"
        else:
            # 확장자로 추정
            return image_path.suffix.lower().lstrip(".")

    except Exception:
        return image_path.suffix.lower().lstrip(".")


def load_image(image_path: Path) -> np.ndarray:
    """
    이미지 로드 (TIFF, PNG, EXR, JPEG 지원)

    Args:
        image_path: 이미지 파일 경로

    Returns:
        np.ndarray: 이미지 데이터
    """
    try:
        # 실제 파일 형식 감지
        actual_format = detect_image_format(image_path)
        logger.info(f"파일 형식 감지: {image_path.name} -> {actual_format}")

        if actual_format == "exr":
            # EXR 파일 읽기
            image = cv2.imread(
                str(image_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED
            )
            if image is None:
                raise ValueError(f"EXR 파일을 읽을 수 없습니다: {image_path}")

            # 다중 채널인 경우 첫 번째 채널만 사용
            if len(image.shape) == 3:
                image = image[:, :, 0]

            # Float32로 변환
            if image.dtype != np.float32:
                image = image.astype(np.float32)

        elif actual_format == "tiff":
            # TIFF 파일 읽기
            import tifffile

            image = tifffile.imread(str(image_path))
            if len(image.shape) == 3:
                image = image[:, :, 0]  # 첫 번째 채널만 사용
            image = image.astype(np.float32)

        else:
            # PNG, JPG 등 일반 이미지 (OpenCV로 읽기)
            if "depth" in image_path.name.lower():
                # Depth 이미지는 grayscale로 읽기
                image = cv2.imread(
                    str(image_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE
                )
            else:
                # Color 이미지는 color로 읽기
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

            # Float32로 변환
            image = image.astype(np.float32)

        logger.info(
            f"이미지 로드 완료: {image_path.name} - shape: {image.shape}, dtype: {image.dtype}"
        )
        return image

    except Exception as e:
        logger.error(f"이미지 로드 실패: {image_path}. 오류: {e}")
        raise


def resize_depth_to_color(
    depth_image: np.ndarray, color_image: np.ndarray
) -> np.ndarray:
    """
    Depth 이미지를 Color 이미지 크기로 리사이징

    Args:
        depth_image: Depth 이미지
        color_image: Color 이미지

    Returns:
        np.ndarray: 리사이징된 depth 이미지
    """
    color_height, color_width = color_image.shape[:2]
    depth_height, depth_width = depth_image.shape[:2]

    logger.info(f"Depth 크기: {depth_width}x{depth_height}")
    logger.info(f"Color 크기: {color_width}x{color_height}")

    if (depth_width, depth_height) == (color_width, color_height):
        logger.info("이미지 크기가 동일합니다.")
        return depth_image

    # INTER_NEAREST 사용 (depth 값 보존)
    resized_depth = cv2.resize(
        depth_image, (color_width, color_height), interpolation=cv2.INTER_NEAREST
    )

    logger.info(f"Depth 리사이징 완료: {color_width}x{color_height}")
    return resized_depth


def create_point_cloud(
    depth_image: np.ndarray,
    color_image: np.ndarray,
    intrinsics: dict,
    depth_scale: float = 1000.0,
) -> o3d.geometry.PointCloud:
    """
    Depth와 Color 이미지로부터 Point Cloud 생성

    Args:
        depth_image: Depth 이미지 (mm 단위) - 이미 color 크기로 리사이징됨
        color_image: Color 이미지 (BGR)
        intrinsics: 카메라 내부 파라미터
        depth_scale: Depth 스케일링 팩터 (mm 단위면 1.0, m 단위면 1000.0)

    Returns:
        o3d.geometry.PointCloud: 생성된 Point Cloud
    """
    # Color 이미지의 해상도 사용 (depth는 이미 리사이징됨)
    color_height, color_width = color_image.shape[:2]

    # 카메라 내부 파라미터 매트릭스 생성
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    # Open3D 카메라 내부 파라미터 (color 이미지 해상도 사용)
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=color_width, height=color_height, fx=fx, fy=fy, cx=cx, cy=cy
    )

    # Depth 이미지 정규화 (Open3D는 mm 단위를 기대)
    depth_normalized = depth_image / depth_scale

    # # Color 이미지 정규화 (0-1 범위)
    # if color_image.dtype != np.uint8:
    #     color_normalized = color_image / 255.0
    # else:
    #     color_normalized = color_image.astype(np.float32) / 255.0

    # BGR to RGB 변환
    if len(color_image.shape) == 3:
        color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    else:
        # Grayscale인 경우 RGB로 변환
        color_rgb = np.stack([color_image] * 3, axis=-1)

    # Open3D 이미지 생성
    depth_o3d = o3d.geometry.Image(depth_normalized.astype(np.float32))
    color_o3d = o3d.geometry.Image((color_rgb).astype(np.uint8))

    # RGBD 이미지 생성
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=1.0,  # 이미 정규화됨
        depth_trunc=10.0,  # 10m 이상은 제거
        convert_rgb_to_intensity=False,
    )

    # Point Cloud 생성
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

    # 좌표계 변환 (Open3D는 Y축이 위쪽, Z축이 앞쪽)
    # 일반적인 카메라 좌표계로 변환
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    logger.info(f"Point Cloud 생성 완료: {len(pcd.points)}개 포인트")
    return pcd


def save_point_cloud(pcd: o3d.geometry.PointCloud, output_path: Path) -> bool:
    """
    Point Cloud 저장

    Args:
        pcd: Point Cloud 객체
        output_path: 저장 경로

    Returns:
        bool: 성공 여부
    """
    try:
        # 출력 디렉토리 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Point Cloud 저장
        success = o3d.io.write_point_cloud(str(output_path), pcd)

        if success:
            logger.info(f"Point Cloud 저장 완료: {output_path}")
            return True
        else:
            logger.error(f"Point Cloud 저장 실패: {output_path}")
            return False

    except Exception as e:
        logger.error(f"Point Cloud 저장 중 오류: {e}")
        return False


def process_single_pair(
    depth_path: Path,
    color_path: Path,
    output_path: Path,
    intrinsics: dict,
    depth_scale: float = 1000.0,
) -> bool:
    """
    단일 depth-color 쌍 처리

    Args:
        depth_path: Depth 이미지 경로
        color_path: Color 이미지 경로
        output_path: 출력 Point Cloud 경로
        intrinsics: 카메라 내부 파라미터
        depth_scale: Depth 스케일링 팩터

    Returns:
        bool: 성공 여부
    """
    try:
        logger.info(f"처리 중: {depth_path.name} + {color_path.name}")

        # 이미지 로드
        depth_image = load_image(depth_path)
        color_image = load_image(color_path)

        # Depth를 Color 크기로 리사이징
        depth_resized = resize_depth_to_color(depth_image, color_image)

        # Point Cloud 생성
        pcd = create_point_cloud(depth_resized, color_image, intrinsics, depth_scale)
        pcd.scale(1000.0, center=[0, 0, 0])

        # Point Cloud 저장
        success = save_point_cloud(pcd, output_path)

        return success

    except Exception as e:
        logger.error(f"처리 실패: {depth_path.name} + {color_path.name}. 오류: {e}")
        return False


def process_batch(
    depth_folder: Path,
    color_folder: Path,
    output_folder: Path,
    intrinsics: dict,
    depth_scale: float = 1000.0,
) -> bool:
    """
    배치 처리

    Args:
        depth_folder: Depth 이미지 폴더
        color_folder: Color 이미지 폴더
        output_folder: 출력 폴더
        intrinsics: 카메라 내부 파라미터
        depth_scale: Depth 스케일링 팩터

    Returns:
        bool: 성공 여부
    """
    try:
        # 지원되는 이미지 확장자
        image_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr"]

        # Depth 파일 찾기
        depth_files = []
        for ext in image_extensions:
            depth_files.extend(depth_folder.glob(f"*{ext}"))
            depth_files.extend(depth_folder.glob(f"*{ext.upper()}"))

        # Color 파일 찾기
        color_files = []
        for ext in image_extensions:
            color_files.extend(color_folder.glob(f"*{ext}"))
            color_files.extend(color_folder.glob(f"*{ext.upper()}"))

        if not depth_files:
            logger.error(f"Depth 파일을 찾을 수 없습니다: {depth_folder}")
            return False

        if not color_files:
            logger.error(f"Color 파일을 찾을 수 없습니다: {color_folder}")
            return False

        logger.info(f"Depth 파일: {len(depth_files)}개")
        logger.info(f"Color 파일: {len(color_files)}개")

        # 출력 폴더 생성
        output_folder.mkdir(parents=True, exist_ok=True)

        # 각 쌍 처리
        success_count = 0
        for depth_file in depth_files:
            # 대응하는 color 파일 찾기 (파일명 기반)
            depth_stem = depth_file.stem
            color_file = None

            for cf in color_files:
                if (
                    cf.stem == depth_stem
                    or depth_stem in cf.stem
                    or cf.stem in depth_stem
                ):
                    color_file = cf
                    break

            if color_file is None:
                logger.warning(
                    f"대응하는 color 파일을 찾을 수 없습니다: {depth_file.name}"
                )
                continue

            # 출력 파일명
            output_file = output_folder / f"{depth_stem}_pointcloud.ply"

            # 처리
            if process_single_pair(
                depth_file, color_file, output_file, intrinsics, depth_scale
            ):
                success_count += 1

        logger.info(f"배치 처리 완료: {success_count}개 성공")
        return success_count > 0

    except Exception as e:
        logger.error(f"배치 처리 중 오류: {e}")
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Point Cloud 생성 도구")

    # 단일 파일 처리
    parser.add_argument("--depth_image", type=str, help="Depth 이미지 파일 경로")
    parser.add_argument("--color_image", type=str, help="Color 이미지 파일 경로")
    parser.add_argument("--output", type=str, help="출력 Point Cloud 파일 경로 (.ply)")

    # 배치 처리
    parser.add_argument("--depth_folder", type=str, help="Depth 이미지 폴더 경로")
    parser.add_argument("--color_folder", type=str, help="Color 이미지 폴더 경로")
    parser.add_argument("--output_folder", type=str, help="출력 폴더 경로")

    # 공통 옵션
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1000.0,
        help="Depth 스케일링 팩터 (mm 단위면 1.0, m 단위면 1000.0)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")

    args = parser.parse_args()

    # 로그 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 카메라 내부 파라미터 출력
    logger.info("카메라 내부 파라미터:")
    for key, value in CAMERA_INTRINSICS.items():
        logger.info(f"  {key}: {value}")

    # 처리 모드 결정
    if args.depth_image and args.color_image and args.output:
        # 단일 파일 처리
        depth_path = Path(args.depth_image)
        color_path = Path(args.color_image)
        output_path = Path(args.output)

        if not depth_path.exists():
            logger.error(f"Depth 파일이 존재하지 않습니다: {depth_path}")
            return 1

        if not color_path.exists():
            logger.error(f"Color 파일이 존재하지 않습니다: {color_path}")
            return 1

        success = process_single_pair(
            depth_path, color_path, output_path, CAMERA_INTRINSICS, args.depth_scale
        )

    elif args.depth_folder and args.color_folder and args.output_folder:
        # 배치 처리
        depth_folder = Path(args.depth_folder)
        color_folder = Path(args.color_folder)
        output_folder = Path(args.output_folder)

        if not depth_folder.exists():
            logger.error(f"Depth 폴더가 존재하지 않습니다: {depth_folder}")
            return 1

        if not color_folder.exists():
            logger.error(f"Color 폴더가 존재하지 않습니다: {color_folder}")
            return 1

        success = process_batch(
            depth_folder,
            color_folder,
            output_folder,
            CAMERA_INTRINSICS,
            args.depth_scale,
        )

    else:
        logger.error("단일 파일 처리 또는 배치 처리 옵션을 모두 제공해주세요.")
        logger.error("단일 파일: --depth_image, --color_image, --output")
        logger.error("배치 처리: --depth_folder, --color_folder, --output_folder")
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
