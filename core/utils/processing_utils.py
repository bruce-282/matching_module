"""
RANSAC 유틸리티 함수들
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import yaml
import open3d as o3d
from .logger_utils import get_logger
from typing import Dict, List, Optional, Tuple, Any

logger = get_logger(__name__)

# # RANSAC related constants
DEFAULT_MIN_NUM_MATCHES = 4


# RANSAC method mapping
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
    geometry_type: str = "Homography",
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

    if geometry_type not in ["Homography", "Fundamental"]:
        logger.warning(f"unsupported geometry_type: {geometry_type}, default 'Homography' used")
        geometry_type = "Homography"

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

        M, mask = proc_ransac_matches(
                mkpts0,
                mkpts1,
                ransac_method,
                ransac_reproj_threshold,
                ransac_confidence,
                ransac_max_iter,
                geometry_type=geometry_type,
        )

        if M is None: return {}

        if geometry_type == "Fundamental":
            geo_info["Fundamental"] = M.tolist()
            geo_info["mask_f"] = mask

        else: 
            geo_info["Homography"] = M.tolist()
            geo_info["mask_h"] = mask
        
        
        return geo_info

    else:
        return {}


def solve_rigid_transform_between_points(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    SVD 기반 3D point correspondence로 rigid transformation 계산
    
    Args:
        points1: 첫 번째 point cloud (Nx3)
        points2: 두 번째 point cloud (Nx3)
    
    Returns:
        4x4 transformation matrix
    """
    assert points1.shape[1] == 3 and points1.shape[0] >= 3, "points1 must be Nx3 with at least 3 points"
    assert points2.shape[1] == 3 and points2.shape[0] >= 3, "points2 must be Nx3 with at least 3 points"
    assert points1.shape[0] == points2.shape[0], "points1 and points2 must have same number of points"
    
    # 4x4 identity matrix 초기화
    pose = np.eye(4, dtype=np.float32)
    
    try:
        # Centroid 계산
        mean1 = np.mean(points1, axis=0)
        mean2 = np.mean(points2, axis=0)
        
        # Centered points
        P = points1 - mean1
        Q = points2 - mean2
        
        # Cross-covariance matrix
        S = P.T @ Q
        assert S.shape == (3, 3), "Cross-covariance matrix must be 3x3"
        
        # SVD decomposition
        U, _, Vt = np.linalg.svd(S)
        R = Vt.T @ U.T
        
        # Rotation matrix 검증
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6):
            logger.warning("Invalid rotation matrix detected, returning identity")
            return pose
        
        # Reflection 방지
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Translation 계산
        t = mean2 - R @ mean1
        
        # 4x4 transformation matrix 구성
        pose[:3, :3] = R
        pose[:3, 3] = t
        
        # Finite 값 검증
        if not np.all(np.isfinite(pose)):
            logger.warning("Non-finite values in transformation matrix, returning identity")
            return np.eye(4, dtype=np.float32)
        
        logger.info(f"SVD-based rigid transform computed successfully")
        logger.debug(f"Rotation matrix:\n{R}")
        logger.debug(f"Translation: {t}")
        
        return pose
        
    except Exception as e:
        logger.error(f"Error in solve_rigid_transform_between_points: {e}")
        return np.eye(4, dtype=np.float32)


def registration_ransac_based_on_correspondence(pcd_source: o3d.geometry.PointCloud, 
                                               pcd_target: o3d.geometry.PointCloud,
                                               correspondences: List[List[int]],
                                               max_correspondence_distance: float = 0.1, # 0.01m
                                               ransac_n: int = 3,
                                               max_iterations: int = 5000,
                                               confidence: float = 0.95) -> np.ndarray or None:
    """
    Known correspondences를 사용한 RANSAC registration
    
    Args:
        pcd_source: Source point cloud
        pcd_target: Target point cloud
        correspondences: Correspondence list [[i, i], [j, j], ...]
        max_correspondence_distance: Maximum correspondence distance
        ransac_n: Number of points for RANSAC
        max_iterations: Maximum RANSAC iterations
        confidence: RANSAC confidence
    
    Returns:
        RegistrationResult (transformation, correspondence_set 등) 또는 None
    """
    try:
        # Correspondence vector 생성
        corres_vector = o3d.utility.Vector2iVector(correspondences)
        
        # RANSAC registration 실행
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            source=pcd_source, 
            target=pcd_target,
            corres=corres_vector,
            max_correspondence_distance=max_correspondence_distance,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
            ransac_n=ransac_n,
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iterations, confidence)
        )
        
        logger.info(f"[3D RANSAC] registration completed - Fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}")
        logger.debug(f"[3D RANSAC] converged: {result.fitness > 0.1}")
        logger.debug(f"[3D RANSAC] Inlier correspondences: {len(result.correspondence_set)} pairs")
        logger.debug(f"[3D RANSAC] transformation : {result.transformation}")
        return result  # RegistrationResult (transformation, correspondence_set 등)
        
    except Exception as e:
        logger.error(f"Error in registration_ransac_based_on_correspondence: {e}")
        # 실패 시 identity transformation 반환
        return None


def chamfer_distance(
    ref_points: np.ndarray,
    src_points: np.ndarray,
    device: Optional[torch.device] = None,
    symmetric: bool = True,
) -> float:
    """
    Chamfer Distance (GPU 가능).
    ref=target, src=source. symmetric=True면 양방향 평균.

    Args:
        ref_points: (N, 3) reference points
        src_points: (M, 3) source points
        device: torch device (None이면 cuda 또는 cpu)
        symmetric: True면 ref→src, src→ref 양방향 평균

    Returns:
        mean chamfer distance
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=torch.float32, device=device)
        return x.to(device)

    ref = _to_tensor(ref_points)
    src = _to_tensor(src_points)

    def _one_way(a, b):
        dist = torch.cdist(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
        return torch.mean(torch.min(dist, dim=0)[0]).item()

    if symmetric:
        cd = (_one_way(ref, src) + _one_way(src, ref)) / 2.0
    else:
        cd = _one_way(ref, src)
    return cd
