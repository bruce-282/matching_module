"""
RANSAC 유틸리티 함수들
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# # RANSAC 관련 상수들
DEFAULT_MIN_NUM_MATCHES = 4


# RANSAC 메서드 매핑
ransac_zoo = {
    "CV2_RANSAC": cv2.RANSAC,
    "CV2_USAC_MAGSAC": cv2.USAC_MAGSAC,
    "CV2_USAC_DEFAULT": cv2.USAC_DEFAULT,
    "CV2_USAC_FM_8PTS": cv2.USAC_FM_8PTS,
    "CV2_USAC_PROSAC": cv2.USAC_PROSAC,
    "CV2_USAC_FAST": cv2.USAC_FAST,
    "CV2_USAC_ACCURATE": cv2.USAC_ACCURATE,
    "CV2_USAC_PARALLEL": cv2.USAC_PARALLEL,
}


def _filter_matches_opencv(
    kp0: np.ndarray,
    kp1: np.ndarray,
    method: int = cv2.RANSAC,
    reproj_threshold: float = 3.0,
    confidence: float = 0.99,
    max_iter: int = 2000,
    geometry_type: str = "Homography",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters matches between two sets of keypoints using OpenCV's findHomography.

    Args:
        kp0 (np.ndarray): Array of keypoints from the first image.
        kp1 (np.ndarray): Array of keypoints from the second image.
        method (int, optional): RANSAC method. Defaults to "cv2.RANSAC".
        reproj_threshold (float, optional): RANSAC reprojection threshold. Defaults to 3.0.
        confidence (float, optional): RANSAC confidence. Defaults to 0.99.
        max_iter (int, optional): RANSAC maximum iterations. Defaults to 2000.
        geometry_type (str, optional): Type of geometry. Defaults to "Homography".

    Returns:
        Tuple[np.ndarray, np.ndarray]: Homography matrix and mask.
    """
    if geometry_type == "Homography":
        try:
            M, mask = cv2.findHomography(
                kp0,
                kp1,
                method=method,
                ransacReprojThreshold=reproj_threshold,
                confidence=confidence,
                maxIters=max_iter,
            )
        except cv2.error:
            logger.error("compute findHomography error, len(kp0): {}".format(len(kp0)))
            return None, None
    elif geometry_type == "Fundamental":
        try:
            M, mask = cv2.findFundamentalMat(
                kp0,
                kp1,
                method=method,
                ransacReprojThreshold=reproj_threshold,
                confidence=confidence,
                maxIters=max_iter,
            )
        except cv2.error:
            logger.error(
                "compute findFundamentalMat error, len(kp0): {}".format(len(kp0))
            )
            return None, None
    mask = np.array(mask.ravel().astype("bool"), dtype="bool")
    return M, mask


def proc_ransac_matches(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    ransac_method: str,
    ransac_reproj_threshold: float = 3.0,
    ransac_confidence: float = 0.99,
    ransac_max_iter: int = 2000,
    geometry_type: str = "Homography",
):
    if ransac_method.startswith("CV2"):
        logger.info(f"ransac_method: {ransac_method}, geometry_type: {geometry_type}")
        return _filter_matches_opencv(
            mkpts0,
            mkpts1,
            ransac_zoo[ransac_method],
            ransac_reproj_threshold,
            ransac_confidence,
            ransac_max_iter,
            geometry_type,
        )
    else:
        raise NotImplementedError


def set_null_pred(feature_type: str, pred: Dict[str, Any]) -> Dict[str, Any]:
    """Set null prediction when no matches are found."""
    if feature_type == "KEYPOINT":
        pred["mmkeypoints0_orig"] = np.array([])
        pred["mmkeypoints1_orig"] = np.array([])
        pred["mmconf"] = np.array([])
    elif feature_type == "LINE":
        pred["mline_keypoints0_orig"] = np.array([])
        pred["mline_keypoints1_orig"] = np.array([])
    pred["H"] = np.eye(3)
    pred["geom_info"] = {}
    return pred


def filter_matches(
    pred: Dict[str, Any],
    ransac_method: str,
    ransac_reproj_threshold: float,
    ransac_confidence: float,
    ransac_max_iter: int,
    geometry_type: str,
):
    """
    Filter matches using RANSAC. If keypoints are available, filter by keypoints.
    If lines are available, filter by lines. If both keypoints and lines are
    available, filter by keypoints.

    Args:
        pred (Dict[str, Any]): dict of matches, including original keypoints.
        ransac_method (str, optional): RANSAC method. Defaults to DEFAULT_RANSAC_METHOD.
        ransac_reproj_threshold (float, optional): RANSAC reprojection threshold. Defaults to DEFAULT_RANSAC_REPROJ_THRESHOLD.
        ransac_confidence (float, optional): RANSAC confidence. Defaults to DEFAULT_RANSAC_CONFIDENCE.
        ransac_max_iter (int, optional): RANSAC maximum iterations. Defaults to DEFAULT_RANSAC_MAX_ITER.

    Returns:
        Dict[str, Any]: filtered matches.
    """
    mkpts0: Optional[np.ndarray] = None
    mkpts1: Optional[np.ndarray] = None
    feature_type: Optional[str] = None
    if "mkeypoints0_orig" in pred.keys() and "mkeypoints1_orig" in pred.keys():
        mkpts0 = pred["mkeypoints0_orig"]
        mkpts1 = pred["mkeypoints1_orig"]
        feature_type = "KEYPOINT"
    elif (
        "line_keypoints0_orig" in pred.keys() and "line_keypoints1_orig" in pred.keys()
    ):
        mkpts0 = pred["line_keypoints0_orig"]
        mkpts1 = pred["line_keypoints1_orig"]
        feature_type = "LINE"
    else:
        return set_null_pred(feature_type, pred)
    if mkpts0 is None or mkpts0 is None:
        return set_null_pred(feature_type, pred)
    if ransac_method not in ransac_zoo.keys():
        ransac_method = "CV2_USAC_DEFAULT"

    if len(mkpts0) < DEFAULT_MIN_NUM_MATCHES:
        return set_null_pred(feature_type, pred)

    geom_info = compute_geometry(
        pred,
        ransac_method=ransac_method,
        ransac_reproj_threshold=ransac_reproj_threshold,
        ransac_confidence=ransac_confidence,
        ransac_max_iter=ransac_max_iter,
        geometry_type=geometry_type,
    )

    if "Homography" in geom_info.keys():
        mask = geom_info["mask_h"]
        pred["H"] = np.array(geom_info["Homography"])
    elif "Fundamental" in geom_info.keys():
        mask = geom_info["mask_f"]
        pred["F"] = np.array(geom_info["Fundamental"])
    else:
        set_null_pred(feature_type, pred)

    if feature_type == "KEYPOINT":
        pred["mmkeypoints0_orig"] = mkpts0[mask]
        pred["mmkeypoints1_orig"] = mkpts1[mask]
        pred["mmconf"] = pred["mconf"][mask]
    elif feature_type == "LINE":
        pred["mline_keypoints0_orig"] = mkpts0[mask]
        pred["mline_keypoints1_orig"] = mkpts1[mask]

    # do not show mask
    geom_info.pop("mask_h", None)
    geom_info.pop("mask_f", None)
    pred["geom_info"] = geom_info
    return pred


def compute_geometry(
    pred: Dict[str, Any],
    ransac_method: str,
    ransac_reproj_threshold: float,
    ransac_confidence: float,
    ransac_max_iter: int,
    geometry_type: str,
) -> Dict[str, List[float]]:
    """
    Compute geometric information of matches, including Fundamental matrix,
    Homography matrix, and rectification matrices (if available).

    Args:
        pred (Dict[str, Any]): dict of matches, including original keypoints.
        ransac_method (str, optional): RANSAC method. Defaults to DEFAULT_RANSAC_METHOD.
        ransac_reproj_threshold (float, optional): RANSAC reprojection threshold. Defaults to DEFAULT_RANSAC_REPROJ_THRESHOLD.
        ransac_confidence (float, optional): RANSAC confidence. Defaults to DEFAULT_RANSAC_CONFIDENCE.
        ransac_max_iter (int, optional): RANSAC maximum iterations. Defaults to DEFAULT_RANSAC_MAX_ITER.

    Returns:
        Dict[str, List[float]]: geometric information in form of a dict.
    """
    mkpts0: Optional[np.ndarray] = None
    mkpts1: Optional[np.ndarray] = None

    if "mkeypoints0_orig" in pred.keys() and "mkeypoints1_orig" in pred.keys():
        mkpts0 = pred["mkeypoints0_orig"]
        mkpts1 = pred["mkeypoints1_orig"]
    elif (
        "line_keypoints0_orig" in pred.keys() and "line_keypoints1_orig" in pred.keys()
    ):
        mkpts0 = pred["line_keypoints0_orig"]
        mkpts1 = pred["line_keypoints1_orig"]

    if mkpts0 is not None and mkpts1 is not None:
        if len(mkpts0) < 2 * DEFAULT_MIN_NUM_MATCHES:
            return {}
        geo_info: Dict[str, List[float]] = {}

        if geometry_type == "Fundamental":
            F, mask_f = proc_ransac_matches(
                mkpts0,
                mkpts1,
                ransac_method,
                ransac_reproj_threshold,
                ransac_confidence,
                ransac_max_iter,
                geometry_type="Fundamental",
            )

            if F is not None:
                geo_info["Fundamental"] = F.tolist()
                geo_info["mask_f"] = mask_f

                return geo_info

        else:
            H, mask_h = proc_ransac_matches(
                mkpts0,
                mkpts1,
                ransac_method,
                ransac_reproj_threshold,
                ransac_confidence,
                ransac_max_iter,
                geometry_type="Homography",
            )

            if H is not None:
                geo_info["Homography"] = H.tolist()
                geo_info["mask_h"] = mask_h

            return geo_info
    else:
        return {}


def save_points_to_yaml(
    image_path: Path,
    image_size: Tuple[int, int],
    point1_2d: Tuple[int, int],
    point2_2d: Tuple[int, int],
    point3_2d: Tuple[int, int],
    output_path: Optional[Path] = None,
    point1_3d: Optional[np.ndarray] = None,
    point2_3d: Optional[np.ndarray] = None,
    point3_3d: Optional[np.ndarray] = None,
    plane_normal: Optional[np.ndarray] = None,
) -> None:
    """포인트 위치를 YAML 파일로 저장합니다."""

    # source 이미지 이름으로 yaml 파일 생성
    yaml_filename = f"{image_path.stem}_result.yaml"

    if output_path is not None:
        yaml_path = output_path / yaml_filename
    else:
        yaml_path = image_path.parent / yaml_filename

    # YAML 데이터 구조
    points_data = {
        "source_image": image_path.name,
        "image_size": {
            "width": int(image_size[1]),
            "height": int(image_size[0]),
        },
        "transformed_points": {
            "pointL": {"x": int(point1_2d[0]), "y": int(point1_2d[1])},
            "pointR": {"x": int(point2_2d[0]), "y": int(point2_2d[1])},
            "pointU": {"x": int(point3_2d[0]), "y": int(point3_2d[1])},
        },
    }

    # 3D 정보가 있는 경우 추가
    if point1_3d is not None and point2_3d is not None:
        points_data["transformed_points_3d"] = {
            "pointL": {
                "x": float(point1_3d[0] if point1_3d is not None else 0),
                "y": float(point1_3d[1] if point1_3d is not None else 0),
                "z": float(point1_3d[2] if point1_3d is not None else 0),
            },
            "pointR": {
                "x": float(point2_3d[0] if point2_3d is not None else 0),
                "y": float(point2_3d[1]),
                "z": float(point2_3d[2]),
            },
            "pointU": {
                "x": float(point3_3d[0] if point3_3d is not None else 0),
                "y": float(point3_3d[1] if point3_3d is not None else 0),
                "z": float(point3_3d[2] if point3_3d is not None else 0),
            },
            "plane_normal": {
                "x": float(plane_normal[0] if plane_normal is not None else 0),
                "y": float(plane_normal[1] if plane_normal is not None else 0),
                "z": float(plane_normal[2] if plane_normal is not None else 0),
            },
        }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            points_data,
            f,
            default_flow_style=False,
            allow_unicode=True,
            indent=2,
        )

    print(f"포인트 위치가 {yaml_path}에 저장되었습니다.")


def wrap_images(
    img0: np.ndarray,
    img1: np.ndarray,
    geo_info: Optional[Dict[str, List[float]]],
    geom_type: str,
    offset_pointL: Tuple[float, float] = (0.5, 0.92),
    offset_pointR: Tuple[float, float] = (1.4, 0.92),
    offset_pointU: Tuple[float, float] = (0.9, 0.1),
    point_radius: int = 10,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Wraps the images based on the geometric transformation used to align them.

    Args:
        img0: numpy array representing the first image.
        img1: numpy array representing the second image.
        geo_info: dictionary containing the geometric transformation information.
        geom_type: type of geometric transformation used to align the images.

    Returns:
        A tuple containing a base64 encoded image string and a dictionary with the transformation matrix.
    """
    h0, w0, _ = img0.shape
    h1, w1, _ = img1.shape
    if geo_info is not None and len(geo_info) != 0:
        # rectified_image0 = img0
        rectified_image1 = None
        if "Homography" not in geo_info:
            logger.warning(f"{geom_type} not exist, maybe too less matches")
            return None, None

        H = np.array(geo_info["Homography"])

        title: List[str] = []
        if geom_type == "Homography":
            H_inv = np.linalg.inv(H)
            rectified_image1 = cv2.warpPerspective(img1, H_inv, (w0, h0))

            # Apply H_inv transformation to 2D points using user-defined ratios
            offset_point1_coords = np.array(
                [
                    [
                        w1 * offset_pointL[0],
                        h1 * offset_pointL[1],
                        1,
                    ]
                ],
                dtype=np.float32,
            )  # Homogeneous coordinates for first point

            transformed_point = H_inv @ offset_point1_coords.T
            transformed_point = (
                transformed_point / transformed_point[2]
            )  # Normalize homogeneous coordinates

            offset_point2_coords = np.array(
                [
                    [
                        w1 * offset_pointR[0],
                        h1 * offset_pointR[1],
                        1,
                    ]
                ],
                dtype=np.float32,
            )
            transformed_point_2 = H_inv @ offset_point2_coords.T
            transformed_point_2 = (
                transformed_point_2 / transformed_point_2[2]
            )  # Normalize homogeneous coordinates

            offset_point3_coords = np.array(
                [
                    [
                        w1 * offset_pointU[0],
                        h1 * offset_pointU[1],
                        1,
                    ]
                ],
                dtype=np.float32,
            )
            transformed_point_3 = H_inv @ offset_point3_coords.T
            transformed_point_3 = (
                transformed_point_3 / transformed_point_3[2]
            )  # Normalize homogeneous coordinates

        else:
            print("Error: Unknown geometry type")

        # Create overlapped image by blending warped_image onto input_image0
        overlapped_image = None
        if rectified_image1 is not None:
            # For Homography: blend warped_image1 onto original img0
            if geom_type == "Homography":
                # Create a mask for non-zero pixels in warped image
                warped_gray = cv2.cvtColor(rectified_image1, cv2.COLOR_RGB2GRAY)
                mask = warped_gray > 0

                # Blend images where warped image has content
                overlapped_image = img0.copy().astype(np.float32)
                warped_float = rectified_image1.astype(np.float32)

                # Alpha blending: 0.7 for original, 0.3 for warped
                alpha = 0.7
                overlapped_image[mask] = (
                    alpha * overlapped_image[mask] + (1 - alpha) * warped_float[mask]
                )

                # Add red circles for the transformed points
                # 포인트 반지름 설정

                # Draw red circle for first transformed point
                x1, y1 = int(transformed_point[0][0]), int(transformed_point[1][0])
                if (
                    0 <= x1 < overlapped_image.shape[1]
                    and 0 <= y1 < overlapped_image.shape[0]
                ):
                    cv2.circle(
                        overlapped_image, (x1, y1), point_radius, (255, 0, 0), -1
                    )  # 빨간색 원

                # Draw red circle for second transformed point
                x2, y2 = int(transformed_point_2[0][0]), int(transformed_point_2[1][0])
                if (
                    0 <= x2 < overlapped_image.shape[1]
                    and 0 <= y2 < overlapped_image.shape[0]
                ):
                    cv2.circle(
                        overlapped_image, (x2, y2), point_radius, (255, 0, 0), -1
                    )  # 빨간색 원

                # Draw red circle for third transformed point
                x3, y3 = int(transformed_point_3[0][0]), int(transformed_point_3[1][0])
                if (
                    0 <= x3 < overlapped_image.shape[1]
                    and 0 <= y3 < overlapped_image.shape[0]
                ):
                    cv2.circle(
                        overlapped_image, (x3, y3), point_radius, (255, 0, 0), -1
                    )  # 빨간색 원

                # Add red tint to overlapping areas for better visibility
                # Create a red overlay for overlapping regions
                red_overlay = np.zeros_like(overlapped_image)
                red_overlay[mask] = [50, 0, 0]  # Red tint
                overlapped_image = np.clip(overlapped_image + red_overlay, 0, 255)
                overlapped_image = overlapped_image.astype(np.uint8)

        return overlapped_image, rectified_image1
    else:
        return None, None


def generate_warp_images(
    input_image0: np.ndarray,
    input_image1: np.ndarray,
    matches_info: Dict[str, Any],
    choice: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Changes the estimate of the geometric transformation used to align the images.

    Args:
        input_image0: First input image.
        input_image1: Second input image.
        matches_info: Dictionary containing information about the matches.
        choice: Type of geometric transformation to use ('Homography' or 'Fundamental') or 'No' to disable.

    Returns:
        A tuple containing the updated images and the warpped images.
    """
    if (
        matches_info is None
        or len(matches_info) < 1
        or "geom_info" not in matches_info.keys()
    ):
        return None, None
    geom_info = matches_info["geom_info"]
    warped_image = None
    if choice != "No":
        wrapped_image_pair, warped_image = wrap_images(
            input_image0, input_image1, geom_info, choice
        )
        return wrapped_image_pair, warped_image
    else:
        return None, None


def scale_keypoints(kpts, scale):
    if np.any(scale != 1.0):
        kpts *= kpts.new_tensor(scale)
    return kpts
