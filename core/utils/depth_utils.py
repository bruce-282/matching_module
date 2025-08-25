#!/usr/bin/env python3
"""
Depth map 관련 유틸리티 함수들
"""
import numpy as np
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

DEFAULT_DEPTH_MAP_WIDTH = 2064
DEFAULT_DEPTH_MAP_HEIGHT = 1544


def point_cloud_to_depth_map(
    points: np.ndarray, colors: Optional[np.ndarray] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    포인트 클라우드를 depth map으로 변환

    Args:
        points: 3D 포인트 배열 (N, 3)
        colors: 색상 배열 (N, 3) - 선택사항

    Returns:
        depth_map: depth map 이미지 (H, W)
        intrinsic: 카메라 내부 파라미터 (3, 3)
    """
    try:
        # 포인트 클라우드의 경계 계산
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        # 이미지 크기 설정 (기본값)
        img_width, img_height = DEFAULT_DEPTH_MAP_WIDTH, DEFAULT_DEPTH_MAP_HEIGHT

        # 카메라 내부 파라미터 (기본값)
        fx = img_width
        fy = img_height
        cx, cy = img_width / 2, img_height / 2  # principal point
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # Depth map 초기화
        depth_map = np.zeros((img_height, img_width), dtype=np.float32)
        depth_count = np.zeros((img_height, img_width), dtype=np.int32)

        # 각 포인트를 이미지 평면에 투영
        for i, point in enumerate(points):
            x, y, z = point

            if z <= 0:  # 카메라 뒤의 포인트는 무시
                continue

            # 3D to 2D projection
            u = int((x * fx / z) + cx)
            v = int((y * fy / z) + cy)

            # 이미지 경계 확인
            if 0 <= u < img_width and 0 <= v < img_height:
                # 가장 가까운 depth 값 저장 (최소값 사용)
                if depth_count[v, u] == 0 or z < depth_map[v, u]:
                    depth_map[v, u] = z
                depth_count[v, u] += 1

        # 유효한 depth가 있는 픽셀만 사용
        valid_mask = depth_count > 0

        if np.sum(valid_mask) == 0:
            logger.warning("유효한 depth map을 생성할 수 없습니다.")
            return None, None

        logger.info(f"Depth map 생성 완료: {np.sum(valid_mask)} 유효 픽셀")
        return depth_map, intrinsic

    except Exception as e:
        logger.error(f"Depth map 생성 중 오류: {e}")
        return None, None


def find_3d_from_2d_depthmap_robust(
    depth_image: np.ndarray,
    intrinsic: np.ndarray,
    pixel_2d: Tuple[int, int],
    radius: int = 3,
) -> Optional[np.ndarray]:
    """
    주변 픽셀의 평균을 사용한 robust한 3D 포인트 계산

    Args:
        depth_image: depth map 이미지 (H, W)
        intrinsic: 카메라 내부 파라미터 (3, 3)
        pixel_2d: 2D 픽셀 좌표 (u, v)
        radius: 주변 픽셀 반지름

    Returns:
        3D 포인트 좌표 [x, y, z] 또는 None
    """
    u, v = pixel_2d
    h, w = depth_image.shape

    # 주변 픽셀들의 depth 값 수집
    depths = []

    for dv in range(max(0, v - radius), min(h, v + radius + 1)):
        for du in range(max(0, u - radius), min(w, u + radius + 1)):
            # 원형 마스크 적용
            dist = np.sqrt((du - u) ** 2 + (dv - v) ** 2)
            if dist <= radius:
                z = depth_image[dv, du]
                if z > 0:  # 유효한 depth만
                    depths.append(z)

    if len(depths) == 0:
        return None

    # 중앙값 사용 (outlier에 강함)
    z = np.median(depths)

    # Back-projection
    # fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    # cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # x = (u - cx) * z / fx
    # y = (v - cy) * z / fy
    x = u
    y = v
    return np.array([x, y, z])


def get_pixels_in_radius(
    depth_image: np.ndarray, pixel_2d: Tuple[int, int], radius: int
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    주어진 픽셀 주변 radius 내의 depth 값들 수집

    Args:
        depth_image: depth map 이미지 (H, W)
        pixel_2d: 2D 픽셀 좌표 (u, v)
        radius: 주변 픽셀 반지름

    Returns:
        depths: depth 값 배열
        positions: 픽셀 위치 리스트 [(u, v), ...]
    """
    u, v = pixel_2d
    h, w = depth_image.shape

    depths = []
    positions = []

    for dv in range(max(0, v - radius), min(h, v + radius + 1)):
        for du in range(max(0, u - radius), min(w, u + radius + 1)):
            # 원형 마스크 적용
            dist = np.sqrt((du - u) ** 2 + (dv - v) ** 2)
            if dist <= radius:
                d = depth_image[dv, du]
                if d > 0:  # 유효한 depth만
                    depths.append(d)
                    positions.append((du, dv))

    return np.array(depths), positions


def find_3d_from_2d_depthmap(
    depth_image: np.ndarray, intrinsic: np.ndarray, pixel_2d: Tuple[int, int]
) -> Optional[np.ndarray]:
    """
    Depth map에서 직접 lookup - O(1) 복잡도

    Args:
        depth_image: depth map 이미지 (H, W)
        intrinsic: 카메라 내부 파라미터 (3, 3)
        pixel_2d: 2D 픽셀 좌표 (u, v)

    Returns:
        3D 포인트 좌표 [x, y, z] 또는 None
    """
    u, v = pixel_2d

    # 직접 depth 값 읽기
    z = depth_image[v, u]

    if z > 0:
        # Back-projection
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        return np.array([x, y, z])

    return None


def depth_estimation_mad(
    depth_image: np.ndarray, pixel_2d: Tuple[int, int], radius: int = 3, k: float = 2.5
) -> Optional[dict]:
    """
    MAD를 사용한 outlier 제거 후 평균
    가장 로버스트한 방법 중 하나

    Args:
        depth_image: depth map 이미지 (H, W)
        pixel_2d: 2D 픽셀 좌표 (u, v)
        radius: 주변 픽셀 반지름
        k: MAD 배수 (기본값: 2.5)

    Returns:
        통계 정보 딕셔너리 또는 None
    """
    depths, _ = get_pixels_in_radius(depth_image, pixel_2d, radius)

    if len(depths) == 0:
        return None

    # Median과 MAD 계산
    median = np.median(depths)
    mad = np.median(np.abs(depths - median))

    # MAD가 0인 경우 처리
    if mad == 0:
        mad = 1.4826 * np.std(depths)  # Gaussian 가정

    # Outlier 제거 (k*MAD 이상 떨어진 값)
    lower = median - k * mad
    upper = median + k * mad
    inliers = depths[(depths >= lower) & (depths <= upper)]

    if len(inliers) == 0:
        return {"mean": median, "median": median, "std": 0, "inlier_ratio": 0}

    # 평균 또는 중앙값 반환
    result = {
        "mean": np.mean(inliers),
        "median": np.median(inliers),
        "std": np.std(inliers),
        "inlier_ratio": len(inliers) / len(depths),
    }

    return result


def depth_estimation_histogram(
    depth_image: np.ndarray,
    pixel_2d: Tuple[int, int],
    radius: int = 3,
    n_bins: int = 50,
) -> Optional[dict]:
    """
    히스토그램의 최빈 구간 주변 값들 평균
    다중 평면이 있을 때 유용

    Args:
        depth_image: depth map 이미지 (H, W)
        pixel_2d: 2D 픽셀 좌표 (u, v)
        radius: 주변 픽셀 반지름
        n_bins: 히스토그램 bin 수

    Returns:
        통계 정보 딕셔너리 또는 None
    """
    depths, _ = get_pixels_in_radius(depth_image, pixel_2d, radius)

    if len(depths) == 0:
        return None

    # 히스토그램 생성
    hist, bin_edges = np.histogram(depths, bins=n_bins)

    # 최빈 구간 찾기
    mode_bin = np.argmax(hist)

    # 최빈 구간 ± 1 구간의 값들 선택
    window = 1  # 주변 구간 포함
    lower_bin = max(0, mode_bin - window)
    upper_bin = min(len(bin_edges) - 1, mode_bin + window + 1)

    lower_bound = bin_edges[lower_bin]
    upper_bound = bin_edges[upper_bin]

    # 해당 구간 내 값들 평균
    mode_depths = depths[(depths >= lower_bound) & (depths <= upper_bound)]

    if len(mode_depths) == 0:
        return {
            "mean": np.median(depths),
            "mode_range": (0, 0),
            "mode_count": 0,
            "total_count": len(depths),
            "confidence": 0,
        }

    result = {
        "mean": np.mean(mode_depths),
        "mode_range": (lower_bound, upper_bound),
        "mode_count": len(mode_depths),
        "total_count": len(depths),
        "confidence": len(mode_depths) / len(depths),
    }

    return result
