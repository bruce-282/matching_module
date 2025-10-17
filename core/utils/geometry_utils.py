"""
Geometry utility functions for coordinate transformations and homography calculations.
"""

import numpy as np
from typing import Tuple, Optional, Union


def se3_to_homography(se3_matrix: np.ndarray, 
                     K1: np.ndarray, 
                     K2: np.ndarray, 
                     depth: Optional[float] = None) -> np.ndarray:
    """
    Convert 4x4 SE(3) transformation matrix to homography matrix.
    
    Args:
        se3_matrix: 4x4 SE(3) transformation matrix (T_2_1: transform from frame 1 to frame 2)
        K1: 3x3 intrinsic matrix of camera 1
        K2: 3x3 intrinsic matrix of camera 2  
        depth: Plane depth for homography calculation (if None, assumes planar scene)
        
    Returns:
        3x3 homography matrix H such that x2 = H * x1
    """
    # Extract rotation and translation
    R = se3_matrix[:3, :3]  # 3x3 rotation matrix
    t = se3_matrix[:3, 3]   # 3x1 translation vector
    
    if depth is not None:
        # For planar scene at specific depth
        # H = K2 * (R + t * n^T / d) * K1^(-1)
        # where n = [0, 0, 1]^T (normal to plane) and d is depth
        n = np.array([0, 0, 1])  # plane normal
        H = K2 @ (R + np.outer(t, n) / depth) @ np.linalg.inv(K1)
    else:
        # For general homography (assumes planar scene)
        # H = K2 * R * K1^(-1) + K2 * t * n^T * K1^(-1) / d
        # For simplicity, assume d=1 (unit depth)
        n = np.array([0, 0, 1])
        H = K2 @ R @ np.linalg.inv(K1) + K2 @ np.outer(t, n) @ np.linalg.inv(K1)
    
    return H


def homography_to_se3(H: np.ndarray, 
                     K1: np.ndarray, 
                     K2: np.ndarray,
                     depth: float = 1.0) -> np.ndarray:
    """
    Convert homography matrix to 4x4 SE(3) transformation matrix.
    
    Args:
        H: 3x3 homography matrix
        K1: 3x3 intrinsic matrix of camera 1
        K2: 3x3 intrinsic matrix of camera 2
        depth: Plane depth for homography calculation
        
    Returns:
        4x4 SE(3) transformation matrix
    """
    # H = K2 * (R + t * n^T / d) * K1^(-1)
    # Solve for R and t
    
    # First, compute the essential matrix part
    E_part = np.linalg.inv(K2) @ H @ K1
    
    # For planar scene, E_part = R + t * n^T / d
    # where n = [0, 0, 1]^T
    
    # Extract rotation and translation
    n = np.array([0, 0, 1])
    
    # R = E_part - t * n^T / d
    # We need to solve this system
    
    # For simplicity, assume t_z = 0 (translation in image plane)
    # This is a common assumption for planar homography
    
    # Extract rotation matrix (approximate)
    R = E_part.copy()
    R[:, 2] = 0  # Remove translation component from last column
    
    # Ensure R is a valid rotation matrix using SVD
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        R = -R
    
    # Extract translation
    t = (E_part - R) @ n * depth
    
    # Construct SE(3) matrix
    se3_matrix = np.eye(4)
    se3_matrix[:3, :3] = R
    se3_matrix[:3, 3] = t
    
    return se3_matrix


def decompose_homography(H: np.ndarray, 
                        K1: np.ndarray, 
                        K2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose homography matrix into rotation, translation, and plane normal.
    
    Args:
        H: 3x3 homography matrix
        K1: 3x3 intrinsic matrix of camera 1
        K2: 3x3 intrinsic matrix of camera 2
        
    Returns:
        Tuple of (R, t, n) where:
        - R: 3x3 rotation matrix
        - t: 3x1 translation vector
        - n: 3x1 plane normal vector
    """
    # H = K2 * (R + t * n^T / d) * K1^(-1)
    E_part = np.linalg.inv(K2) @ H @ K1
    
    # For planar scene, assume n = [0, 0, 1]^T
    n = np.array([0, 0, 1])
    
    # Extract rotation and translation
    # This is an approximation - in practice, you might need more sophisticated methods
    R = E_part.copy()
    R[:, 2] = 0  # Remove translation component
    
    # Ensure R is a valid rotation matrix
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    
    if np.linalg.det(R) < 0:
        R = -R
    
    # Extract translation (approximate)
    t = (E_part - R) @ n
    
    return R, t, n


def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Apply homography transformation to 2D points.
    
    Args:
        points: Nx2 array of 2D points
        H: 3x3 homography matrix
        
    Returns:
        Nx2 array of transformed 2D points
    """
    # Convert to homogeneous coordinates
    points_homo = np.column_stack([points, np.ones(points.shape[0])])
    
    # Apply homography
    transformed_homo = (H @ points_homo.T).T
    
    # Convert back to Cartesian coordinates
    transformed_points = transformed_homo[:, :2] / transformed_homo[:, 2:3]
    
    return transformed_points


def compute_homography_from_correspondences(points1: np.ndarray, 
                                          points2: np.ndarray) -> np.ndarray:
    """
    Compute homography matrix from point correspondences using DLT algorithm.
    
    Args:
        points1: Nx2 array of points in first image
        points2: Nx2 array of corresponding points in second image
        
    Returns:
        3x3 homography matrix
    """
    assert points1.shape[0] >= 4, "At least 4 point correspondences are required"
    assert points1.shape == points2.shape, "Point arrays must have the same shape"
    
    n = points1.shape[0]
    
    # Build the A matrix for DLT
    A = np.zeros((2 * n, 9))
    
    for i in range(n):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        
        # Two equations per correspondence
        A[2*i, :] = [-x1, -y1, -1, 0, 0, 0, x2*x1, x2*y1, x2]
        A[2*i+1, :] = [0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2]
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1, :].reshape(3, 3)
    
    # Normalize
    H = H / H[2, 2]
    
    return H


def validate_homography(H: np.ndarray, 
                       points1: np.ndarray, 
                       points2: np.ndarray, 
                       threshold: float = 1.0) -> Tuple[bool, float]:
    """
    Validate homography by computing reprojection error.
    
    Args:
        H: 3x3 homography matrix
        points1: Nx2 array of points in first image
        points2: Nx2 array of corresponding points in second image
        threshold: Maximum allowed reprojection error
        
    Returns:
        Tuple of (is_valid, mean_error)
    """
    # Apply homography to points1
    transformed_points = apply_homography(points1, H)
    
    # Compute reprojection error
    errors = np.linalg.norm(transformed_points - points2, axis=1)
    mean_error = np.mean(errors)
    
    is_valid = mean_error < threshold
    
    return is_valid, mean_error


def normalize_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize 2D points for better numerical stability in homography computation.
    
    Args:
        points: Nx2 array of 2D points
        
    Returns:
        Tuple of (normalized_points, T) where T is the normalization matrix
    """
    # Compute centroid
    centroid = np.mean(points, axis=0)
    
    # Compute scale
    centered_points = points - centroid
    distances = np.linalg.norm(centered_points, axis=1)
    scale = np.sqrt(2) / np.mean(distances)
    
    # Create normalization matrix
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    
    # Apply normalization
    points_homo = np.column_stack([points, np.ones(points.shape[0])])
    normalized_homo = (T @ points_homo.T).T
    normalized_points = normalized_homo[:, :2]
    
    return normalized_points, T


def robust_homography_estimation(points1: np.ndarray, 
                                points2: np.ndarray,
                                threshold: float = 1.0,
                                max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate homography using RANSAC for robustness against outliers.
    
    Args:
        points1: Nx2 array of points in first image
        points2: Nx2 array of corresponding points in second image
        threshold: Inlier threshold for RANSAC
        max_iterations: Maximum number of RANSAC iterations
        
    Returns:
        Tuple of (best_H, inlier_mask)
    """
    assert points1.shape[0] >= 4, "At least 4 point correspondences are required"
    assert points1.shape == points2.shape, "Point arrays must have the same shape"
    
    n = points1.shape[0]
    best_H = None
    best_inliers = 0
    best_inlier_mask = None
    
    for _ in range(max_iterations):
        # Randomly select 4 points
        indices = np.random.choice(n, 4, replace=False)
        sample1 = points1[indices]
        sample2 = points2[indices]
        
        try:
            # Compute homography from sample
            H = compute_homography_from_correspondences(sample1, sample2)
            
            # Count inliers
            transformed_points = apply_homography(points1, H)
            errors = np.linalg.norm(transformed_points - points2, axis=1)
            inlier_mask = errors < threshold
            num_inliers = np.sum(inlier_mask)
            
            if num_inliers > best_inliers:
                best_inliers = num_inliers
                best_H = H
                best_inlier_mask = inlier_mask
                
        except np.linalg.LinAlgError:
            # Skip if homography computation fails
            continue
    
    if best_H is None:
        raise ValueError("Failed to estimate homography")
    
    return best_H, best_inlier_mask


def project_3d_point_to_2d(point_3d: np.ndarray, 
                          intrinsic_matrix: np.ndarray) -> np.ndarray:
    """
    Project single 3D point to 2D image coordinates.
    
    Args:
        point_3d: 3D point [x, y, z]
        intrinsic_matrix: 3x3 camera intrinsic matrix
        
    Returns:
        2D point [u, v]
    """
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    u = fx * point_3d[0] / point_3d[2] + cx
    v = fy * point_3d[1] / point_3d[2] + cy
    
    return np.array([u, v])


def project_pcd_to_image(points_3d: np.ndarray, 
                        colors: Optional[np.ndarray] = None,
                        intrinsic_matrix: np.ndarray = None,
                        image_size: Tuple[int, int] = (640, 480),
                        depth_range: Tuple[float, float] = (0.1, 5000.0)) -> np.ndarray:
    """
    Project 3D point cloud to 2D image.
    
    Args:
        points_3d: Nx3 array of 3D points
        colors: Nx3 array of RGB colors (0-255) for each point, optional
        intrinsic_matrix: 3x3 camera intrinsic matrix
        image_size: (width, height) of output image
        depth_range: (min_depth, max_depth) for depth filtering
        
    Returns:
        2D image as numpy array (HxWx3)
    """
    if intrinsic_matrix is None:
        # Default intrinsic matrix
        intrinsic_matrix = np.array([
            [500, 0, image_size[0]/2],
            [0, 500, image_size[1]/2],
            [0, 0, 1]
        ])
    
    # Filter points by depth
    depths = points_3d[:, 2]
    valid_mask = (depths >= depth_range[0]) & (depths <= depth_range[1])
    valid_points = points_3d[valid_mask]
    
    if len(valid_points) == 0:
        # Return empty image if no valid points
        return np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    
    # Project 3D points to 2D
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    # Perspective projection
    x_2d = (fx * valid_points[:, 0] / valid_points[:, 2] + cx).astype(int)
    y_2d = (fy * valid_points[:, 1] / valid_points[:, 2] + cy).astype(int)
    
    # Filter points within image bounds
    valid_2d_mask = (x_2d >= 0) & (x_2d < image_size[0]) & (y_2d >= 0) & (y_2d < image_size[1])
    x_2d = x_2d[valid_2d_mask]
    y_2d = y_2d[valid_2d_mask]
    valid_points = valid_points[valid_2d_mask]
    
    if len(x_2d) == 0:
        return np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    
    # Create image
    image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    
    if colors is not None:
        # Use provided colors
        valid_colors = colors[valid_mask][valid_2d_mask]
        for i, (x, y) in enumerate(zip(x_2d, y_2d)):
            image[y, x] = valid_colors[i]
    else:
        # Use depth-based coloring
        depths = valid_points[:, 2]
        depth_normalized = (depths - depths.min()) / (depths.max() - depths.min() + 1e-8)
        
        # Create colormap (blue to red based on depth)
        colors_2d = np.zeros((len(depths), 3), dtype=np.uint8)
        colors_2d[:, 0] = (depth_normalized * 255).astype(np.uint8)  # Red channel
        colors_2d[:, 2] = ((1 - depth_normalized) * 255).astype(np.uint8)  # Blue channel
        
        for i, (x, y) in enumerate(zip(x_2d, y_2d)):
            image[y, x] = colors_2d[i]
    
    return image


def project_pcd_to_depth_image(points_3d: np.ndarray,
                              intrinsic_matrix: np.ndarray = None,
                              image_size: Tuple[int, int] = (640, 480),
                              depth_range: Tuple[float, float] = (0.1, 100.0)) -> np.ndarray:
    """
    Project 3D point cloud to depth image.
    
    Args:
        points_3d: Nx3 array of 3D points
        intrinsic_matrix: 3x3 camera intrinsic matrix
        image_size: (width, height) of output image
        depth_range: (min_depth, max_depth) for depth filtering
        
    Returns:
        Depth image as numpy array (HxW)
    """
    if intrinsic_matrix is None:
        # Default intrinsic matrix
        intrinsic_matrix = np.array([
            [500, 0, image_size[0]/2],
            [0, 500, image_size[1]/2],
            [0, 0, 1]
        ])
    
    # Filter points by depth
    depths = points_3d[:, 2]
    valid_mask = (depths >= depth_range[0]) & (depths <= depth_range[1])
    valid_points = points_3d[valid_mask]
    
    if len(valid_points) == 0:
        # Return empty depth image
        return np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    
    # Project 3D points to 2D
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    # Perspective projection
    x_2d = (fx * valid_points[:, 0] / valid_points[:, 2] + cx).astype(int)
    y_2d = (fy * valid_points[:, 1] / valid_points[:, 2] + cy).astype(int)
    
    # Filter points within image bounds
    valid_2d_mask = (x_2d >= 0) & (x_2d < image_size[0]) & (y_2d >= 0) & (y_2d < image_size[1])
    x_2d = x_2d[valid_2d_mask]
    y_2d = y_2d[valid_2d_mask]
    valid_depths = valid_points[valid_2d_mask, 2]
    
    if len(x_2d) == 0:
        return np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    
    # Create depth image
    depth_image = np.zeros((image_size[1], image_size[0]), dtype=np.float32)
    
    # For overlapping points, use minimum depth (closest point)
    for x, y, depth in zip(x_2d, y_2d, valid_depths):
        if depth_image[y, x] == 0 or depth < depth_image[y, x]:
            depth_image[y, x] = depth
    
    return depth_image


def project_open3d_pcd_to_image(pcd, 
                         intrinsic_matrix: np.ndarray = None,
                         image_size: Tuple[int, int] = (640, 480),
                         depth_range: Tuple[float, float] = (0.0, 5000.0)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project point cloud to 2D image.
    
    Args:
        pcd: Open3D point cloud object
        intrinsic_matrix: 3x3 camera intrinsic matrix
        image_size: (width, height) of output image
        depth_range: (min_depth, max_depth) for depth filtering
        
    Returns:
        2D image as numpy array (HxWx3)
    """
    # Extract points and colors from PCD
    points_3d = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255 if pcd.has_colors() else None
    
    # Create color projection
    image = project_pcd_to_image(
        points_3d, 
        colors, 
        intrinsic_matrix, 
        image_size,
        depth_range=depth_range
    )
    
    
    return image


def create_transform_matrix_from_vectors(
    right_vector: np.ndarray,
    up_vector: np.ndarray, 
    front_vector: np.ndarray,
    position: np.ndarray
) -> np.ndarray:
    """
    벡터들로부터 4x4 변환 행렬을 생성합니다.
    
    Args:
        right_vector: 우측 방향 벡터 (정규화된 단위 벡터)
        up_vector: 상단 방향 벡터 (정규화된 단위 벡터)
        front_vector: 전방 방향 벡터 (정규화된 단위 벡터)
        position: 위치 벡터 (3D 좌표)
        
    Returns:
        4x4 변환 행렬 (homogeneous transformation matrix)
    """
    # 회전 행렬 구성 (카메라 좌표계)
    rotation_matrix = np.column_stack([right_vector, -up_vector, -front_vector])
    
    # 4x4 변환 행렬 생성
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = position
    
    return transform_matrix
    
