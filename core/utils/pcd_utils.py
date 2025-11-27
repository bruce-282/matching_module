"""
포인트 클라우드 유틸리티 함수들
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, List

from .logger_utils import get_logger
logger = get_logger(__name__)



class PointCloudToImageConverter:
    """포인트 클라우드를 이미지로 변환하는 클래스"""

    def __init__(self, width: int, height: int, intrinsic_matrix: np.ndarray):
        """
        Initialize

        Args:
            width: image width
            height: image height
            intrinsic_matrix: camera intrinsic matrix (3x3)
        """
        self.width = width
        self.height = height
        self.intrinsic_matrix = intrinsic_matrix

    def _point_cloud_to_rgb_image(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Convert point cloud to RGB image

        Args:
            pcd: Open3D point cloud object

        Returns:
            RGB image (height, width, 3)

        Raises:
            ValueError: conversion failed
        """
        if not pcd.has_points() or not pcd.has_colors():
            raise ValueError("Point cloud must have both points and colors.")

        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) * 255  # convert colors to 0-255 range

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


def compute_plane_normal(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """3D points to plane normal vector
    Args:
        p1: 3D point
        p2: 3D point
        p3: 3D point
    Returns:
        plane normal vector
    """

    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)

    normal = normal / np.linalg.norm(normal)

    return normal


def normal_to_angles(normal: np.ndarray) -> tuple:
    """Normal 벡터를 각도로 변환
    
    Args:
        normal: 정규화된 법선 벡터 [x, y, z]
        
    Returns:
        tuple: (horizontal_deg, vertical_deg)
            - horizontal_deg: 수평각 (도) - 수평면에서의 방향
            - vertical_deg: 수직각 (도) - 수직면에서의 방향
    """
    if normal is None:
        return None, None
        
    x, y, z = normal
    
    # Horizontal (수평각): 수평면에서의 방향
    horizontal_rad = np.arctan2(y, x)  # -π ~ π radian
    horizontal_deg = np.degrees(horizontal_rad)  # -180° ~ 180°
    
    # Vertical (수직각): 수직면에서의 방향  
    vertical_rad = np.arccos(z)  # 0 ~ π radian
    vertical_deg = np.degrees(vertical_rad)  # 0° ~ 180°
    
    return horizontal_deg, vertical_deg


def load_ply_as_image(
    ply_path: Path,
    width: int = 1920,
    height: int = 1080,
    intrinsic_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert PLY file to image

    Args:
        ply_path: PLY file path
        width: image width (default: 1920)
        height: image height (default: 1080)
        intrinsic_matrix: camera intrinsic matrix (default: None, standard camera)

        Returns:
        RGB image (height, width, 3)

    Raises:
        FileNotFoundError: PLY file not found
        ValueError: conversion failed
    """

    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    if intrinsic_matrix is None:
        intrinsic_matrix = np.array(
            [[width, 0, width / 2], [0, height, height / 2], [0, 0, 1]]
        )

    try:
        logger.info(f"Loading PLY file: {ply_path}")
        pcd = o3d.io.read_point_cloud(str(ply_path))

        if not pcd.has_points():
            raise ValueError("PLY file does not have points.")

        if not pcd.has_colors():
            logger.warning(
                "PLY file does not have color information. Using default color (white)."
            )
            colors = np.ones((len(pcd.points), 3), dtype=np.float32)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        converter = PointCloudToImageConverter(width, height, intrinsic_matrix)
        rgb_image = converter._point_cloud_to_rgb_image(pcd)

        logger.info(f"PLY file to image conversion completed: {rgb_image.shape}")
        return rgb_image

    except Exception as e:
        if isinstance(e, ValueError):
            raise e
        else:
            raise ValueError(f"PLY file processing error: {e}")


def is_ply_file(file_path: str) -> bool:
    """
    Check if the file is a PLY file

    Args:
        file_path: file path

    Returns:
        True if the file is a PLY file, False otherwise
    """
    return Path(file_path).suffix.lower() == ".ply"


def visualize_normal_on_pointcloud(
    pcd: o3d.geometry.PointCloud,
    normal_vector: np.ndarray,
    center_point_3d: np.ndarray,
    normal_length: float = 0.1,
    normal_color: List[float] = [1.0, 0.0, 0.0],  # red
    center_color: List[float] = [0.0, 1.0, 0.0],  # green
) -> o3d.geometry.PointCloud:
    """
    Visualize normal vector on point cloud

    Args:
        pcd: Open3D point cloud
        normal_vector: 3D normal vector [x, y, z]
        center_point_3d: normal vector's starting point (3D coordinates)
        normal_length: normal vector's length
        normal_color: normal vector's color [r, g, b]
        center_color: center point's color [r, g, b]

    Returns:
        point cloud with normal vector
    """
    # normalize normal vector
    normal_norm = normal_vector / np.linalg.norm(normal_vector)

    # calculate end point of normal vector
    end_point = center_point_3d + normal_norm * normal_length

    # represent normal vector as a line
    normal_line = o3d.geometry.LineSet()
    normal_line.points = o3d.utility.Vector3dVector([center_point_3d, end_point])
    normal_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    normal_line.colors = o3d.utility.Vector3dVector([normal_color])

    # represent center point as a sphere
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    center_sphere.translate(center_point_3d)
    center_sphere.paint_uniform_color(center_color)


    return pcd, normal_line, center_sphere


def save_pointcloud_with_normal(
    pcd: o3d.geometry.PointCloud,
    normal_vector: np.ndarray,
    center_point_3d: np.ndarray,
    output_path: str,
    normal_length: float = 0.1,
) -> None:
    """
    Save point cloud with normal vector

    Args:
        pcd: Open3D point cloud
        normal_vector: 3D normal vector
        center_point_3d: normal vector's starting point
        output_path: path to save the file
        normal_length: normal vector's length
    """
    # normalize normal vector
    normal_norm = normal_vector / np.linalg.norm(normal_vector)

    # calculate end point of normal vector
    end_point = center_point_3d + normal_norm * normal_length

    # represent normal vector as a line
    normal_line = o3d.geometry.LineSet()
    normal_line.points = o3d.utility.Vector3dVector([center_point_3d, end_point])
    normal_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    normal_line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])  # red

    # represent center point as a sphere
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    center_sphere.translate(center_point_3d)
    center_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # green

    # combine all geometries into a mesh
    combined_mesh = o3d.geometry.TriangleMesh()

    # convert point cloud to mesh (optional)
    # combined_mesh += pcd

    # add normal vector and center point
    combined_mesh += center_sphere

    # save as PLY file
    o3d.io.write_triangle_mesh(output_path, combined_mesh)

    # save normal vector information to a separate text file
    normal_info_path = output_path.replace(".ply", "_normal_info.txt")
    with open(normal_info_path, "w") as f:
        f.write(f"Normal Vector: {normal_vector}\n")
        f.write(f"Center Point: {center_point_3d}\n")
        f.write(f"Normal Length: {normal_length}\n")
        f.write(f"End Point: {end_point}\n")


def create_point_cloud_from_depth_image(
    depth_image,  # np.ndarray 또는 o3d.geometry.Image
    intrinsic: np.ndarray,
    texture_image: Optional[np.ndarray] = None,  # texture 이미지 추가
):
    """
    Create a point cloud from a depth image using Open3D.
    Parameters:
        - depth_image: A numpy array or Open3D Image containing the depth image.
        - intrinsic: An Open3D camera intrinsic object.
        - scale: A scaling factor to adjust the depth values.
    Returns:
        - A point cloud object.
    """
    # depth_image 타입 확인 및 변환
    if isinstance(depth_image, o3d.geometry.Image):
        # Open3D Image → numpy array로 변환
        depth_array = np.asarray(depth_image)
    else:
        depth_array = depth_image


    h, w = depth_array.shape[:2]

    depth_scaled = (depth_array).astype(np.float32)
    depth_o3d = o3d.geometry.Image(depth_scaled)


    if texture_image is not None:
    
        if len(texture_image.shape) == 2:
            color_array = cv2.cvtColor(texture_image, cv2.COLOR_GRAY2RGB)
        else:
            color_array = texture_image
    else:
        color_array = np.zeros((h, w, 3), dtype=np.uint8)

    # C-style buffer로 변환 (Open3D 요구사항)
    color_array = np.ascontiguousarray(color_array)
    color_o3d = o3d.geometry.Image(color_array)

    # RGBD 이미지 생성
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_o3d,
        depth=depth_o3d,
        convert_rgb_to_intensity=False,
    )

    # Point cloud 생성
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=depth_array.shape[1],
        height=depth_array.shape[0],
        intrinsic_matrix=intrinsic,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, o3d_intrinsic)

    return pcd

def add_normal_line_to_pcd(
    pcd,
    position: np.ndarray,
    normal: np.ndarray,
    line_length=0.1,
    num_points=50,
    line_color=[0.0, 1.0, 0.0],
):
    """
    Represent normal direction line as points and add to PCD

    Args:
        pcd: Open3D point cloud
        position: normal vector's starting point
        normal: normal vector
        line_length: length of the line
        num_points: number of points on the line
        line_color: color of the line
    """
    # set position

    origin = np.array(position)

    # Normal vector
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)

    # 선분 상의 점들 생성
    t = np.linspace(0, line_length, num_points)
    line_points = origin + normal * t[:, np.newaxis]

    # 화살표 머리 부분 강조 (끝부분에 더 많은 점)
    # head_points = origin + normal * line_length
    # head_cloud = np.random.normal(head_points, 0.01, (20, 3))

    # 모든 점 결합
    arrow_points = np.vstack([line_points])

    # Point Cloud 생성
    arrow_pcd = o3d.geometry.PointCloud()
    arrow_pcd.points = o3d.utility.Vector3dVector(arrow_points)

    # 색상 설정
    colors = np.tile(line_color, (len(arrow_points), 1))
    arrow_pcd.colors = o3d.utility.Vector3dVector(colors)

    # 병합
    combined_pcd = pcd + arrow_pcd

    return combined_pcd


def add_3d_points_to_pcd(
    pcd,
    points_3d: List[np.ndarray],
    point_color: List[float] = [0.0, 1.0, 0.0],  # 더 진한 녹색  
):
    """
    3D points to PCD with red points

    Args:
        pcd: Open3D point cloud
        points_3d: 3D points
        point_color: color of the points
        point_size: size of the points
    """
    # create small spheres around each 3D point
    all_point_clouds = []

    for point_3d in points_3d:
        # create points on the surface of the sphere 
        phi = np.linspace(0, 2 * np.pi, 80)  
        theta = np.linspace(0, np.pi, 40)    
        phi_grid, theta_grid = np.meshgrid(phi, theta)

        # small sphere radius
        radius = 0.02

        # sphere surface coordinates
        x = point_3d[0] + radius * np.sin(theta_grid) * np.cos(phi_grid)
        y = point_3d[1] + radius * np.sin(theta_grid) * np.sin(phi_grid)
        z = point_3d[2] + radius * np.cos(theta_grid)

        # 점들로 변환
        sphere_points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])

        # Point Cloud 생성
        sphere_pcd = o3d.geometry.PointCloud()
        sphere_pcd.points = o3d.utility.Vector3dVector(sphere_points)

        # 색상 설정
        colors = np.tile(point_color, (len(sphere_points), 1))
        sphere_pcd.colors = o3d.utility.Vector3dVector(colors)

        all_point_clouds.append(sphere_pcd)

    # merge all point clouds
    combined_pcd = pcd
    for sphere_pcd in all_point_clouds:
        combined_pcd = combined_pcd + sphere_pcd

    return combined_pcd


def rotation_matrix_from_vectors(vec1, vec2):
    """Calculate rotation matrix between two vectors
    Args:
        vec1: vector 1
        vec2: vector 2
    Returns:
        rotation matrix
    """
    a = np.array(vec1) / np.linalg.norm(vec1)
    b = np.array(vec2) / np.linalg.norm(vec2)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s == 0:  # parallel case
        if c > 0:
            return np.eye(3)
        else:
            # 180 degree rotation
            return -np.eye(3)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))
    return R


def clip_pointcloud_by_depth(
    pcd: o3d.geometry.PointCloud, 
    near_z: float, 
    far_z: float
) -> o3d.geometry.PointCloud:
    """
    포인트 클라우드를 depth 범위로 클리핑합니다.
    
    Args:
        pcd: 입력 포인트 클라우드
        near_z: 최소 depth 값
        far_z: 최대 depth 값
        
    Returns:
        클리핑된 포인트 클라우드
    """
    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()
    
    # X, Y는 원래 범위 유지, Z만 depth 범위로 제한
    aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=[min_bound[0], min_bound[1], near_z],
        max_bound=[max_bound[0], max_bound[1], far_z],
    )
    
    return pcd.crop(aabb)
