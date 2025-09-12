"""
시각화 유틸리티
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List

from core.utils.logger_utils import get_logger

logger = get_logger(__name__)


def visualize_matches(
    image0_origin: np.ndarray,
    image1_origin: np.ndarray,
    keypoints0,
    keypoints1,
    confidence,
    output_path: Path,
    confidence_threshold: float = 0.5,
    circle_radius: int = 5,
    line_thickness: int = 1,
    circle_color: Tuple[int, int, int] = (0, 255, 0),  # 녹색
    line_color: Tuple[int, int, int] = (255, 0, 0),  # 파란색
):
    """매칭 결과를 시각화합니다."""
    # 이미지 로드

    if image0_origin is None or image1_origin is None:
        raise ValueError(
            f"이미지를 로드할 수 없습니다: {image0_origin}, {image1_origin}"
        )

    # BGR to RGB 변환
    img0_rgb = cv2.cvtColor(image0_origin, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(image1_origin, cv2.COLOR_BGR2RGB)

    # 이미지 크기 조정 (높이 맞춤)
    h0, w0 = img0_rgb.shape[:2]
    h1, w1 = img1_rgb.shape[:2]

    # 두 이미지의 높이를 맞춤
    max_height = max(h0, h1)
    scale0 = 1.0
    scale1 = 1.0

    if h0 != max_height:
        scale0 = max_height / h0
        new_w0 = int(w0 * scale0)
        img0_rgb = cv2.resize(img0_rgb, (new_w0, max_height))
        w0 = new_w0
    if h1 != max_height:
        scale1 = max_height / h1
        new_w1 = int(w1 * scale1)
        img1_rgb = cv2.resize(img1_rgb, (new_w1, max_height))
        w1 = new_w1

    # 이미지 연결
    combined_img = np.hstack([img0_rgb, img1_rgb])

    # 키포인트 그리기 (스케일링 적용)
    match_count = 0
    for i, (kp0, kp1, conf) in enumerate(zip(keypoints0, keypoints1, confidence)):
        if conf > confidence_threshold:
            match_count += 1
            # 첫 번째 이미지의 키포인트 (스케일링 적용)
            x0, y0 = int(kp0[0] * scale0), int(kp0[1] * scale0)
            cv2.circle(combined_img, (x0, y0), circle_radius, circle_color, -1)

            # 두 번째 이미지의 키포인트 (스케일링 적용, x 좌표에 w0 더함)
            x1, y1 = int(kp1[0] * scale1) + w0, int(kp1[1] * scale1)
            cv2.circle(combined_img, (x1, y1), circle_radius, circle_color, -1)

            # 매칭 선 그리기
            cv2.line(combined_img, (x0, y0), (x1, y1), line_color, line_thickness)

    # 결과 저장
    combined_img_bgr = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, combined_img_bgr)
    logger.info(f"Matching result is saved to {output_path}")
    logger.info(f"Total {match_count} matches are visualized.")

    return combined_img_bgr


def visualize_keypoints(
    image_path,
    keypoints,
    output_path="keypoints_visualization.png",
    circle_radius=5,
    color=(0, 255, 0),
    thickness=-1,
):
    """키포인트를 시각화합니다."""
    # 이미지 로드
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Image loading failed: {image_path}")

    # 키포인트 그리기
    for kp in keypoints:
        x, y = int(kp[0]), int(kp[1])
        cv2.circle(img, (x, y), circle_radius, color, thickness)

    # 결과 저장
    cv2.imwrite(output_path, img)
    logger.info(f"Key points visualization is saved to {output_path}")
    logger.info(f"Total {len(keypoints)} key points are visualized.")

    return img


def warp_images(
    img0: np.ndarray,
    img1: np.ndarray,
    homography: np.ndarray,
    pointL_pos: Dict[str, float],
    pointR_pos: Dict[str, float],
    pointU_pos: Dict[str, float],
    point_radius: int = 10,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Warps images using homography transformation and creates overlapped visualization.

    Args:
        img0: numpy array representing the first image (target).
        img1: numpy array representing the second image (source).
        homography: 3x3 homography matrix for transformation.
        pointL_pos: Point L position configuration
        pointR_pos: Point R position configuration
        pointU_pos: Point U position configuration
        point_radius: radius for drawing points

    Returns:
        A tuple containing (overlapped_image, warped_image).
    """
    h0, w0, _ = img0.shape
    h1, w1, _ = img1.shape
    
    # Homography inverse transformation
    H_inv = np.linalg.inv(homography)
    warped_image = cv2.warpPerspective(img1, H_inv, (w0, h0))

    # Transform 2D points using homography
    point1_coords = np.array(
        [[w1 * pointL_pos["x_ratio"], h1 * pointL_pos["y_ratio"], 1]],
        dtype=np.float32,
    )
    transformed_point1 = H_inv @ point1_coords.T
    transformed_point1 = transformed_point1 / transformed_point1[2]  # Normalize

    point2_coords = np.array(
        [[w1 * pointR_pos["x_ratio"], h1 * pointR_pos["y_ratio"], 1]],
        dtype=np.float32,
    )
    transformed_point2 = H_inv @ point2_coords.T
    transformed_point2 = transformed_point2 / transformed_point2[2]  # Normalize

    point3_coords = np.array(
        [[w1 * pointU_pos["x_ratio"], h1 * pointU_pos["y_ratio"], 1]],
        dtype=np.float32,
    )
    transformed_point3 = H_inv @ point3_coords.T
    transformed_point3 = transformed_point3 / transformed_point3[2]  # Normalize

    # Create overlapped image by blending warped image onto target image
    warped_gray = cv2.cvtColor(warped_image, cv2.COLOR_RGB2GRAY)
    mask = warped_gray > 0

    overlapped_image = img0.copy().astype(np.float32)
    warped_float = warped_image.astype(np.float32)

    # Alpha blending: 0.7 for original, 0.3 for warped
    alpha = 0.7
    overlapped_image[mask] = (
        alpha * overlapped_image[mask] + (1 - alpha) * warped_float[mask]
    )

    # Draw red circles for the transformed points
    points = [
        (int(transformed_point1[0][0]), int(transformed_point1[1][0])),
        (int(transformed_point2[0][0]), int(transformed_point2[1][0])),
        (int(transformed_point3[0][0]), int(transformed_point3[1][0])),
    ]

    for x, y in points:
        if 0 <= x < overlapped_image.shape[1] and 0 <= y < overlapped_image.shape[0]:
            cv2.circle(overlapped_image, (x, y), point_radius, (255, 0, 0), -1)

    # Add red tint to overlapping areas for better visibility
    red_overlay = np.zeros_like(overlapped_image)
    red_overlay[mask] = [50, 0, 0]  # Red tint
    overlapped_image = np.clip(overlapped_image + red_overlay, 0, 255)
    overlapped_image = overlapped_image.astype(np.uint8)

    return overlapped_image, warped_image