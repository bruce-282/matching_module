import cv2
import numpy as np
import open3d as o3d
from typing import Tuple, Optional, List
from .logger_utils import get_logger

logger = get_logger(__name__)


class Camera:
    def __init__(
        self, K: np.ndarray, dist_coeffs: np.ndarray, image_size: Tuple[int, int]
    ):
        """
        Camera 클래스 초기화

        Args:
            K: 카메라 내부 파라미터 (3x3)
            dist_coeffs: 왜곡 계수 [k1, k2, p1, p2, k3, ...]
            image_size: (width, height)
        """
        self.K = K
        self.dist_coeffs = dist_coeffs
        self.image_size = image_size

        # Undistortion 맵 미리 계산 (효율성)
        self.compute_undistort_maps()

    def compute_undistort_maps(self):
        """Undistortion 맵 미리 계산"""
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist_coeffs, None, self.K, self.image_size, cv2.CV_32FC1
        )
        logger.debug("Undistortion map calculation completed")

    def undistort_image(
        self, image: np.ndarray, interpolation: int = cv2.INTER_NEAREST
    ) -> np.ndarray:
        """
        이미지 undistortion

        Args:
            image: 왜곡된 이미지
            interpolation: 보간법 (기본값: cv2.INTER_LINEAR)

        Returns:
            undistorted_image: undistorted 이미지
        """
        if self.map1 is not None and self.map2 is not None:
            undistorted_image = cv2.remap(image, self.map1, self.map2, interpolation)
        else:
            undistorted_image = image
            logger.warning("Undistortion map not calculated. Undistortion correction not applied")
        return undistorted_image.astype(image.dtype)

    def undistort_depth_image(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Depth image undistortion (INTER_NEAREST used)

        Args:
            depth_image: Distorted depth image

        Returns:
            undistorted_depth: undistorted depth 이미지
        """
        undistorted_depth = cv2.remap(
            depth_image, self.map1, self.map2, cv2.INTER_NEAREST
        )
        return undistorted_depth

    def undistort_points(self, points_2d: np.ndarray) -> np.ndarray:
        """
        2D 포인트들 undistortion

        Args:
            points_2d: 왜곡된 2D 포인트들 (N, 2)

        Returns:
            undistorted_points: undistorted 2D 포인트들 (N, 2)
        """
        points_2d_reshaped = points_2d.reshape(-1, 1, 2).astype(np.float32)
        undistorted_points = cv2.undistortPoints(
            points_2d_reshaped, self.K, self.dist_coeffs, P=self.K
        )
        return undistorted_points.reshape(-1, 2)

    def get_intrinsic_matrix(self) -> np.ndarray:
        """카메라 내부 파라미터 반환"""
        return self.K.copy()

    def get_distortion_coeffs(self) -> np.ndarray:
        """왜곡 계수 반환"""
        return self.dist_coeffs.copy()


def create_default_camera(image_size: Tuple[int, int] = (2064, 1544)) -> Camera:
    """
    기본 카메라 설정으로 Camera 객체 생성

    Args:
        image_size: (width, height)

    Returns:
        camera: Camera 객체
    """

    K = np.array(
        [
            [2344.0698849413925, 0.0, 989.06314625513],
            [0.0, 2344.400093425026, 807.02989528271],
            [0.0, 0.0, 1.0],
        ]
    )

    dist_coeffs = np.array(
        [
            -0.24331290305526787,
            0.13922919417642093,
            0.0005252878633098153,
            -0.0010237886757940777,
            -0.01443719970450923,
        ]
    )

    return Camera(K, dist_coeffs, image_size)


def undistort_image(
    image: np.ndarray,
    K: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    이미지 undistortion (편의 함수)

    Args:
        image: 왜곡된 이미지
        K: 카메라 내부 파라미터 (None이면 기본값 사용)
        dist_coeffs: 왜곡 계수 (None이면 기본값 사용)

    Returns:
        undistorted_image: undistorted 이미지
    """
    if K is None or dist_coeffs is None:
        # 기본값 사용
        camera = create_default_camera((image.shape[1], image.shape[0]))
    else:
        camera = Camera(K, dist_coeffs, (image.shape[1], image.shape[0]))

    return camera.undistort_image(image)
