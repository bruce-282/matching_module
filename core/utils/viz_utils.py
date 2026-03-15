"""
시각화 유틸리티
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List

from .logger_utils import get_logger

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
    contrast: int = 0,
    display_scale: float = 1.0,
):
    """매칭 결과를 시각화합니다. display_scale > 1 이면 결과용 이미지를 키워서 마커가 상대적으로 작게 보이게 함."""
    if image0_origin is None or image1_origin is None:
        raise ValueError(
            f"이미지를 로드할 수 없습니다: {image0_origin}, {image1_origin}"
        )

    # PLY 역투영(project_pcd_to_image), read_image는 RGB 반환. BGR 변환 생략.
    img0_rgb = np.asarray(image0_origin).copy()
    img1_rgb = np.asarray(image1_origin).copy()

    h0, w0 = img0_rgb.shape[:2]
    h1, w1 = img1_rgb.shape[:2]

    # 결과용 이미지 확대 (마커가 상대적으로 작게 보이도록)
    if display_scale != 1.0 and display_scale > 0:
        new_w0, new_h0 = int(w0 * display_scale), int(h0 * display_scale)
        new_w1, new_h1 = int(w1 * display_scale), int(h1 * display_scale)
        img0_rgb = cv2.resize(img0_rgb, (new_w0, new_h0), interpolation=cv2.INTER_LINEAR)
        img1_rgb = cv2.resize(img1_rgb, (new_w1, new_h1), interpolation=cv2.INTER_LINEAR)
        w0, h0, w1, h1 = new_w0, new_h0, new_w1, new_h1
        keypoints0 = np.asarray(keypoints0, dtype=np.float64) * display_scale
        keypoints1 = np.asarray(keypoints1, dtype=np.float64) * display_scale

    # 이미지 크기 조정 (높이 맞춤)

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

    # 선을 그릴 오버레이 이미지 (알파 블렌딩용)
    overlay = combined_img.copy()

    # 연한 빨간색 (투명 효과를 위한 색상)
    line_color_alpha = (255, 0, 0)
    alpha = 0.15  # 투명도 (0.0 = 완전 투명, 1.0 = 불투명)

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

            # 매칭 선 그리기 (오버레이에)
            cv2.line(overlay, (x0, y0), (x1, y1), line_color_alpha, line_thickness)

    # 오버레이 블렌딩
    combined_img = cv2.addWeighted(overlay, alpha, combined_img, 1 - alpha, 0)

    # 결과 저장
    # 밝기 조정 (더 밝게)
    if contrast > 0:
        combined_img = cv2.convertScaleAbs(combined_img, alpha=contrast)
    else:
        combined_img = combined_img
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


def visualize_3d_correspondences(
    ref_corr: np.ndarray,
    src_corr: np.ndarray,
    output_path: str,
    *,
    max_points: int = 500,
    line_step: int = 10,
    projection: str = "xy",
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 150,
    show_both_xy_xz: bool = True,
    color_by_depth: bool = True,
) -> None:
    """
    3D correspondence를 2D 평면에 투영해 시각화 (REF=target, SRC=source).
    Z(depth) 기반 컬러맵으로 모양 파악 용이. XY·XZ 두 평면 동시 출력 가능.

    Args:
        ref_corr: Target 쪽 대응점 (N, 3)
        src_corr: Source 쪽 대응점 (N, 3)
        output_path: 저장 경로 (.png 등)
        max_points: 스캐터에 쓸 최대 점 수 (초과 시 랜덤 샘플)
        line_step: 연결선 그릴 때 간격 (step 마다 한 줄)
        projection: "xz" | "xy" | "yz" (show_both_xy_xz=False일 때만 사용)
        title: 그래프 제목
        figsize: figure 크기
        dpi: 저장 해상도
        show_both_xy_xz: True면 XY·XZ 각각 별도 파일로 저장 (_xy.png, _xz.png)
        color_by_depth: True면 Z값으로 컬러맵 (모양 파악용)
    """
    # matplotlib import 전에 DEBUG 로그 억제 (data path, CONFIGDIR, CACHEDIR 등)
    for _name in ("matplotlib", "matplotlib.font_manager", "matplotlib.pyplot"):
        logging.getLogger(_name).setLevel(logging.WARNING)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref_corr = np.asarray(ref_corr)
    src_corr = np.asarray(src_corr)
    n = len(ref_corr)
    if n != len(src_corr):
        raise ValueError("ref_corr and src_corr must have same length")

    if n == 0:
        logger.warning("visualize_3d_correspondences: no points, skip")
        return

    corr_idx = (
        np.random.choice(n, min(max_points, n), replace=False)
        if n > max_points
        else np.arange(n)
    )
    line_indices = corr_idx[::line_step] if line_step > 0 else corr_idx

    # Z(depth) 기반 컬러
    if color_by_depth:
        z_all = np.concatenate([ref_corr[:, 2], src_corr[:, 2]])
        vmin, vmax = z_all.min(), z_all.max()
        if vmax - vmin < 1e-6:
            vmin, vmax = vmin - 1, vmax + 1
        c_ref = ref_corr[corr_idx, 2]
        c_src = src_corr[corr_idx, 2]
        cmap = "viridis"
    else:
        c_ref, c_src = "blue", "red"
        cmap = None
        vmin = vmax = None

    def draw_projection(ax, xi: int, yi: int, xlabel: str, ylabel: str, proj_name: str):
        if color_by_depth:
            s0 = ax.scatter(
                ref_corr[corr_idx, xi],
                ref_corr[corr_idx, yi],
                c=c_ref,
                s=5,
                alpha=0.8,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                label="REF (target)",
            )
            s1 = ax.scatter(
                src_corr[corr_idx, xi],
                src_corr[corr_idx, yi],
                c=c_src,
                s=5,
                alpha=0.8,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                label="SRC (source)",
            )
        else:
            ax.scatter(
                ref_corr[corr_idx, xi],
                ref_corr[corr_idx, yi],
                c="blue",
                s=5,
                alpha=0.8,
                label="REF (target)",
            )
            ax.scatter(
                src_corr[corr_idx, xi],
                src_corr[corr_idx, yi],
                c="red",
                s=5,
                alpha=0.8,
                label="SRC (source)",
            )
        for i in line_indices:
            ax.plot(
                [ref_corr[i, xi], src_corr[i, xi]],
                [ref_corr[i, yi], src_corr[i, yi]],
                "g-",
                alpha=0.2,
                linewidth=0.5,
            )
        ax.set_title(f"{proj_name} ({n:,} pairs)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(markerscale=2, fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    if show_both_xy_xz:
        # XY·XZ 각각 별도 파일로 저장
        out_path = Path(output_path)
        base, ext = out_path.parent / out_path.stem, out_path.suffix
        paths = [f"{base}_xy{ext}", f"{base}_xz{ext}"]
        for path, (xi, yi, xl, yl, name) in [
            (paths[0], (0, 1, "X", "Y", "XY (top-down)")),
            (paths[1], (0, 2, "X", "Z", "XZ (side)")),
        ]:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            draw_projection(ax, xi, yi, xl, yl, name)
            if color_by_depth and ax.collections:
                from matplotlib.cm import ScalarMappable
                from matplotlib.colors import Normalize
                sm = ScalarMappable(cmap=plt.cm.viridis, norm=Normalize(vmin=vmin, vmax=vmax))
                sm.set_array([])
                fig.colorbar(sm, ax=ax, label="Z (depth)")
            fig.tight_layout()
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"3D correspondence plot saved: {path}")
        return
    else:
        axes = projection.lower()
        if axes == "xz":
            xi, yi = 0, 2
            xlabel, ylabel = "X", "Z"
        elif axes == "xy":
            xi, yi = 0, 1
            xlabel, ylabel = "X", "Y"
        elif axes == "yz":
            xi, yi = 1, 2
            xlabel, ylabel = "Y", "Z"
        else:
            raise ValueError('projection must be "xz", "xy", or "yz"')
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        draw_projection(ax, xi, yi, xlabel, ylabel, projection.upper())
        if color_by_depth and ax.collections:
            fig.colorbar(ax.collections[0], ax=ax, label="Z (depth)")
        if title:
            ax.set_title(title)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"3D correspondence plot saved: {output_path}")
