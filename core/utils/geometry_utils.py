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


def euler_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    XYZ Euler 각(라디안)으로부터 3x3 회전 행렬을 생성합니다.

    회전은 R = Rx @ Ry @ Rz 로 합성됩니다. 이는 safe_zones 의 euler 를 생성하는
    프론트엔드(Three.js)의 기본 회전 순서 'XYZ'(Matrix4.makeRotationFromEuler)와
    동일한 규약이며, 임의 각도에 대해 행렬이 정확히 일치함을 확인하였다.
    규약상 intrinsic XYZ (= extrinsic ZYX, = scipy Rotation.from_euler('XYZ'))
    이며, extrinsic XYZ(Rz·Ry·Rx)와는 다르다.

    Args:
        rx: X축 회전 각 (radian)
        ry: Y축 회전 각 (radian)
        rz: Z축 회전 각 (radian)

    Returns:
        3x3 회전 행렬
    """
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rx @ Ry @ Rz


def is_point_in_safe_zone(
    point_3d: np.ndarray,
    zone_min: np.ndarray,
    zone_max: np.ndarray,
    euler: np.ndarray,
) -> bool:
    """
    3D 포인트가 방향이 있는 큐보이드(safe zone, OBB) 안에 있는지 판정합니다.

    큐보이드는 두 대각 꼭짓점(min, max)과 중심점 기준 XYZ Euler 회전으로 정의됩니다.
    (min/max 는 회전된 박스의 월드 대각 꼭짓점이므로 half 는 역회전 후 산출한다 —
    자세한 내용은 obb_from_min_max_euler 참고.)
        center, half, R = obb_from_min_max_euler(min, max, euler)
    포인트를 큐보이드 로컬 좌표계로 변환(p_local = R^T @ (point - center)) 한 뒤,
    모든 축에서 |p_local[i]| <= half[i] 이면 내부로 판정합니다.

    Args:
        point_3d: 검사할 3D 포인트 [x, y, z]
        zone_min: 큐보이드 꼭짓점 [x, y, z]
        zone_max: 반대편 큐보이드 꼭짓점 [x, y, z]
        euler: 중심점 기준 회전 [rx, ry, rz] (radian)

    Returns:
        포인트가 큐보이드 내부에 있으면 True, 아니면 False
    """
    center, half, R = obb_from_min_max_euler(zone_min, zone_max, euler)
    return is_point_in_obb(point_3d, center, half, R)


def obb_from_min_max_euler(
    zone_min: np.ndarray, zone_max: np.ndarray, euler: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(min, max, euler) 표현의 OBB 를 (center, half, R) 표현으로 변환합니다.

    *** 중요 ***: 프론트엔드(Three.js)가 저장하는 min/max 는 회전이 적용된 박스의
    **월드 대각 꼭짓점**이지, 로컬 축에 정렬된 값이 아니다. 따라서 half 를
    ``|max - min| / 2`` 로 곧장 구하면 euler != 0 인 경우 크기가 틀린다(축이 섞여
    회전각만큼 왜곡됨). min/max 를 먼저 로컬 프레임으로 역회전(Rᵀ)해 축 정렬값으로
    되돌린 뒤 half 를 산출해야 한다(프론트 복원 로직과 동일). euler == 0 이면
    Rᵀ = I 라 종전과 동일하다.

    Returns:
        center: 큐보이드 중심 [x, y, z]
        half: 각 축 반쪽 크기 [x, y, z]
        R: 중심점 기준 3x3 회전 행렬 (euler 로부터)
    """
    zone_min = np.asarray(zone_min, dtype=float)
    zone_max = np.asarray(zone_max, dtype=float)

    center = (zone_min + zone_max) / 2.0
    R = euler_to_rotation_matrix(float(euler[0]), float(euler[1]), float(euler[2]))
    local_min = R.T @ (zone_min - center)
    local_max = R.T @ (zone_max - center)
    half = np.abs(local_max - local_min) / 2.0
    return center, half, R


def is_point_in_obb(
    point_3d: np.ndarray,
    center: np.ndarray,
    half: np.ndarray,
    R: np.ndarray,
) -> bool:
    """3D 포인트가 (center, half, R) 로 표현된 방향 큐보이드(OBB) 안에 있는지 판정합니다.

    포인트를 큐보이드 로컬 좌표계로 변환(p_local = R^T @ (point - center)) 한 뒤,
    모든 축에서 |p_local[i]| <= half[i] 이면 내부로 판정합니다. (center/half/R 가 어떤
    프레임에 있든, 포인트가 같은 프레임에 있으면 그대로 적용됩니다.)
    """
    point_3d = np.asarray(point_3d, dtype=float)
    center = np.asarray(center, dtype=float)
    half = np.asarray(half, dtype=float)
    R = np.asarray(R, dtype=float)

    p_local = R.T @ (point_3d - center)
    return bool(np.all(np.abs(p_local) <= half))


def closest_point_on_obb(
    point_3d: np.ndarray,
    center: np.ndarray,
    half: np.ndarray,
    R: np.ndarray,
) -> np.ndarray:
    """OBB(또는 그 내부) 표면에서 point_3d 에 가장 가까운 점을 반환합니다.

    포인트를 로컬 좌표로 역회전한 뒤 각 축을 [-half, half] 로 clamp 하고, 다시
    월드(또는 입력) 프레임으로 복원한다. 포인트가 OBB 내부면 입력점이 그대로 반환된다.
    위반 화살표(표면 최근접점 -> 위반 anchor) 시각화에 쓰인다.
    """
    point_3d = np.asarray(point_3d, dtype=float)
    center = np.asarray(center, dtype=float)
    half = np.asarray(half, dtype=float)
    R = np.asarray(R, dtype=float)

    p_local = R.T @ (point_3d - center)
    clamped = np.clip(p_local, -half, half)
    return R @ clamped + center


def obb_violation_vector(
    point_3d: np.ndarray,
    center: np.ndarray,
    half: np.ndarray,
    R: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """OBB 표면 최근접점에서 point_3d 로 향하는 (벡터, 거리) 를 반환합니다.

    포인트가 OBB 내부면 (영벡터, 0.0). 거리는 mm 단위(입력 좌표 단위와 동일).
    """
    closest = closest_point_on_obb(point_3d, center, half, R)
    vec = np.asarray(point_3d, dtype=float) - closest
    return vec, float(np.linalg.norm(vec))


def transform_safe_zone(
    zone_min: np.ndarray,
    zone_max: np.ndarray,
    euler: np.ndarray,
    transform_4x4: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(min, max, euler) 로 정의된 safe zone(OBB)을 4x4 강체변환으로 다른 프레임으로
    옮긴 (center, half, R) 표현을 반환합니다.

    강체변환 T = [[R_T, t_T], [0, 1]] 에 대해:
        center' = R_T @ center + t_T
        R'      = R_T @ R          (방향 합성)
        half    = half             (강체변환은 크기 보존)

    Args:
        zone_min, zone_max: 원 프레임에서의 큐보이드 대각 꼭짓점
        euler: 원 프레임에서의 중심점 기준 회전 [rx, ry, rz] (radian)
        transform_4x4: 원 프레임 -> 목표 프레임 4x4 변환

    Returns:
        목표 프레임에서의 (center', half, R')
    """
    center, half, R = obb_from_min_max_euler(zone_min, zone_max, euler)
    T = np.asarray(transform_4x4, dtype=float)
    R_T = T[:3, :3]
    t_T = T[:3, 3]

    center_t = R_T @ center + t_T
    R_t = R_T @ R
    return center_t, half, R_t


def as_4x4_matrix(value) -> np.ndarray:
    """list-of-lists(4x4) 또는 flat(16개) 값을 (4, 4) 동차변환 행렬로 변환합니다.

    Args:
        value: 4x4 중첩 리스트, 길이 16 의 1차원 시퀀스, 또는 (4,4)/(16,) ndarray.

    Returns:
        (4, 4) float ndarray

    Raises:
        ValueError: 4x4(또는 16개 원소)로 해석할 수 없는 경우.
    """
    arr = np.asarray(value, dtype=float)
    if arr.shape == (4, 4):
        return arr
    if arr.size == 16:
        return arr.reshape(4, 4)
    raise ValueError(
        f"4x4 변환 행렬(또는 16개 원소)이 필요하지만 shape {arr.shape} 입니다."
    )


def transform_point_3d(point_3d: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    """4x4 동차변환을 3D 포인트에 적용합니다.

    Args:
        point_3d: 변환할 3D 포인트 [x, y, z]
        transform_4x4: (4, 4) 동차변환 행렬

    Returns:
        변환된 3D 포인트 [x', y', z']
    """
    point_3d = np.asarray(point_3d, dtype=float)
    T = np.asarray(transform_4x4, dtype=float)
    p_homo = np.append(point_3d[:3], 1.0)
    return (T @ p_homo)[:3]


