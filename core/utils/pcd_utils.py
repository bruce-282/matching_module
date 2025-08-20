"""
포인트 클라우드 유틸리티 함수들
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class PointCloudToImageConverter:
    """포인트 클라우드를 이미지로 변환하는 클래스"""

    def __init__(self, width: int, height: int, intrinsic_matrix: np.ndarray):
        """
        초기화

        Args:
            width: 이미지 너비
            height: 이미지 높이
            intrinsic_matrix: 카메라 내부 파라미터 행렬 (3x3)
        """
        self.width = width
        self.height = height
        self.intrinsic_matrix = intrinsic_matrix

    def _point_cloud_to_rgb_image(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """
        포인트 클라우드를 RGB 이미지로 변환

        Args:
            pcd: Open3D 포인트 클라우드 객체

        Returns:
            RGB 이미지 (height, width, 3)

        Raises:
            ValueError: 변환 실패 시
        """
        if not pcd.has_points() or not pcd.has_colors():
            raise ValueError("Point cloud must have both points and colors.")

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) * 255  # Convert colors to 0-255 range

        if points.shape[0] != colors.shape[0]:
            raise ValueError(
                f"Number of points ({points.shape[0]}) and colors ({colors.shape[0]}) do not match."
            )

        rgb_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        K = self.intrinsic_matrix

        try:
            for i, point in enumerate(points):
                x, y, z = point
                if z != 0:
                    u = int((K[0, 0] * x + K[0, 2] * z) / z)
                    v = int((K[1, 1] * y + K[1, 2] * z) / z)
                    if 0 <= u < self.width and 0 <= v < self.height:
                        rgb_image[v, u] = colors[i]
        except Exception as e:
            raise ValueError(f"Failed to project points to image: {e}")

        return rgb_image


def load_ply_as_image(
    ply_path: str,
    width: int = 1920,
    height: int = 1080,
    intrinsic_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    PLY 파일을 이미지로 변환

    Args:
        ply_path: PLY 파일 경로
        width: 이미지 너비 (기본값: 1920)
        height: 이미지 높이 (기본값: 1080)
        intrinsic_matrix: 카메라 내부 파라미터 행렬 (기본값: None, 표준 카메라 사용)

        Returns:
        RGB 이미지 (height, width, 3)

    Raises:
        FileNotFoundError: PLY 파일을 찾을 수 없을 때
        ValueError: 변환 실패 시
    """
    ply_path = Path(ply_path)

    if not ply_path.exists():
        raise FileNotFoundError(f"PLY 파일을 찾을 수 없습니다: {ply_path}")

    # 기본 카메라 내부 파라미터 (표준 카메라)
    if intrinsic_matrix is None:
        intrinsic_matrix = np.array(
            [[width, 0, width / 2], [0, height, height / 2], [0, 0, 1]]
        )

    try:
        # PLY 파일 로드
        logger.info(f"PLY 파일을 로드하는 중: {ply_path}")
        pcd = o3d.io.read_point_cloud(str(ply_path))

        if not pcd.has_points():
            raise ValueError("PLY 파일에 포인트가 없습니다.")

        if not pcd.has_colors():
            logger.warning(
                "PLY 파일에 색상 정보가 없습니다. 기본 색상(흰색)을 사용합니다."
            )
            # 기본 색상 추가 (흰색)
            colors = np.ones((len(pcd.points), 3), dtype=np.float32)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # 변환기 생성 및 이미지 변환
        converter = PointCloudToImageConverter(width, height, intrinsic_matrix)
        rgb_image = converter._point_cloud_to_rgb_image(pcd)

        logger.info(f"PLY 파일을 이미지로 변환 완료: {rgb_image.shape}")
        return rgb_image

    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        else:
            raise ValueError(f"PLY 파일 처리 중 오류 발생: {e}")


def is_ply_file(file_path: str) -> bool:
    """
    파일이 PLY 형식인지 확인

    Args:
        file_path: 파일 경로

    Returns:
        PLY 파일이면 True, 아니면 False
    """
    return Path(file_path).suffix.lower() == ".ply"


def get_image_from_file(
    file_path: str,
    width: int = 1920,
    height: int = 1080,
    intrinsic_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    파일 경로에 따라 적절한 방법으로 이미지를 로드

    Args:
        file_path: 이미지 또는 PLY 파일 경로
        width: PLY 변환 시 이미지 너비
        height: PLY 변환 시 이미지 높이
        intrinsic_matrix: PLY 변환 시 카메라 내부 파라미터

    Returns:
        RGB 이미지 (height, width, 3)
    """
    file_path = Path(file_path)

    if is_ply_file(str(file_path)):
        logger.info(f"PLY 파일을 이미지로 변환: {file_path}")
        return load_ply_as_image(str(file_path), width, height, intrinsic_matrix)
    else:
        # 일반 이미지 파일 로드
        from .image_utils import read_image

        logger.info(f"일반 이미지 파일 로드: {file_path}")
        return read_image(str(file_path))
