"""
포인트 클라우드 유틸리티 함수들
"""

import numpy as np
import open3d as o3d
from pathlib import Path
import logging
from typing import Optional, Tuple, Dict, Any, List

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


def compute_plane_normal(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """3개 점으로 평면의 법선 벡터 계산"""
    # None 값 검증
    if p1 is None or p2 is None or p3 is None:
        logger.warning("3D 포인트 중 하나 이상이 None입니다.")
        return None

    # 두 벡터 생성
    v1 = p2 - p1
    v2 = p3 - p1

    # 외적(cross product)으로 법선 구하기
    normal = np.cross(v1, v2)

    # 정규화 (단위 벡터로)
    normal = normal / np.linalg.norm(normal)

    return normal


def load_ply_as_image(
    ply_path: Path,
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


# def get_image_from_file(
#     file_path: str,
#     width: int = 1920,
#     height: int = 1080,
#     intrinsic_matrix: Optional[np.ndarray] = None,
# ) -> np.ndarray:
#     """
#     파일 경로에 따라 적절한 방법으로 이미지를 로드

#     Args:
#         file_path: 이미지 또는 PLY 파일 경로
#         width: PLY 변환 시 이미지 너비
#         height: PLY 변환 시 이미지 높이
#         intrinsic_matrix: PLY 변환 시 카메라 내부 파라미터

#     Returns:
#         RGB 이미지 (height, width, 3)
#     """
#     file_path = Path(file_path)

#     if is_ply_file(str(file_path)):
#         logger.info(f"PLY 파일을 이미지로 변환: {file_path}")
#         return load_ply_as_image(file_path, width, height, intrinsic_matrix)
#     else:
#         # 일반 이미지 파일 로드
#         from .image_utils import read_image

#         logger.info(f"일반 이미지 파일 로드: {file_path}")
#         return read_image(str(file_path))


def visualize_normal_on_pointcloud(
    pcd: o3d.geometry.PointCloud,
    normal_vector: np.ndarray,
    center_point_3d: np.ndarray,
    normal_length: float = 0.1,
    normal_color: List[float] = [1.0, 0.0, 0.0],  # 빨강
    center_color: List[float] = [0.0, 1.0, 0.0],  # 초록
) -> o3d.geometry.PointCloud:
    """
    포인트 클라우드에 normal 벡터를 시각화합니다.

    Args:
        pcd: Open3D 포인트 클라우드
        normal_vector: 3D 법선 벡터 [x, y, z]
        center_point_3d: normal 벡터의 시작점 (3D 좌표)
        normal_length: normal 벡터의 길이
        normal_color: normal 벡터의 색상 [r, g, b]
        center_color: 중심점의 색상 [r, g, b]

    Returns:
        normal 벡터가 추가된 포인트 클라우드
    """
    # normal 벡터를 정규화
    normal_norm = normal_vector / np.linalg.norm(normal_vector)

    # normal 벡터의 끝점 계산
    end_point = center_point_3d + normal_norm * normal_length

    # normal 벡터를 선으로 표현
    normal_line = o3d.geometry.LineSet()
    normal_line.points = o3d.utility.Vector3dVector([center_point_3d, end_point])
    normal_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    normal_line.colors = o3d.utility.Vector3dVector([normal_color])

    # 중심점을 구체로 표현
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    center_sphere.translate(center_point_3d)
    center_sphere.paint_uniform_color(center_color)

    # 포인트 클라우드에 normal 벡터와 중심점 추가
    # (Open3D에서는 PointCloud에 직접 선을 추가할 수 없으므로 별도로 시각화)
    return pcd, normal_line, center_sphere


def save_pointcloud_with_normal(
    pcd: o3d.geometry.PointCloud,
    normal_vector: np.ndarray,
    center_point_3d: np.ndarray,
    output_path: str,
    normal_length: float = 0.1,
) -> None:
    """
    포인트 클라우드와 normal 벡터를 함께 저장합니다.

    Args:
        pcd: Open3D 포인트 클라우드
        normal_vector: 3D 법선 벡터
        center_point_3d: normal 벡터의 시작점
        output_path: 저장할 파일 경로
        normal_length: normal 벡터의 길이
    """
    # normal 벡터를 정규화
    normal_norm = normal_vector / np.linalg.norm(normal_vector)

    # normal 벡터의 끝점 계산
    end_point = center_point_3d + normal_norm * normal_length

    # normal 벡터를 선으로 표현
    normal_line = o3d.geometry.LineSet()
    normal_line.points = o3d.utility.Vector3dVector([center_point_3d, end_point])
    normal_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    normal_line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])  # 빨강

    # 중심점을 구체로 표현
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    center_sphere.translate(center_point_3d)
    center_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # 초록

    # 모든 geometry를 하나의 메시로 결합
    combined_mesh = o3d.geometry.TriangleMesh()

    # 포인트 클라우드를 메시로 변환 (옵션)
    # combined_mesh += pcd

    # normal 벡터와 중심점 추가
    combined_mesh += center_sphere

    # PLY 파일로 저장
    o3d.io.write_triangle_mesh(output_path, combined_mesh)

    # normal 벡터 정보를 별도 텍스트 파일로 저장
    normal_info_path = output_path.replace(".ply", "_normal_info.txt")
    with open(normal_info_path, "w") as f:
        f.write(f"Normal Vector: {normal_vector}\n")
        f.write(f"Center Point: {center_point_3d}\n")
        f.write(f"Normal Length: {normal_length}\n")
        f.write(f"End Point: {end_point}\n")


def create_point_cloud_from_depth_image(
    depth_image,  # np.ndarray 또는 o3d.geometry.Image
    plane_normal: np.ndarray,
    center_point_3d: np.ndarray,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    point1_3d: Optional[np.ndarray] = None,
    point2_3d: Optional[np.ndarray] = None,
    point3_3d: Optional[np.ndarray] = None,
    scale: float = 1.0,
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
        # 이미 numpy array
        depth_array = depth_image

    # 크기 가져오기
    h, w = depth_array.shape[:2]
    # scale 적용 및 Open3D Image 생성
    depth_scaled = (depth_array / scale).astype(np.float32)
    depth_o3d = o3d.geometry.Image(depth_scaled)

    # Color 이미지 생성 (검은색)
    color_array = np.zeros((h, w, 3), dtype=np.uint8)
    color_o3d = o3d.geometry.Image(color_array)

    # RGBD 이미지 생성
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_o3d,
        depth=depth_o3d,
        convert_rgb_to_intensity=False,
    )

    # Point cloud 생성
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # print(f"center_point_3d: {center_point_3d}")
    def get_point_3d(point, intrinsic):

        fx = intrinsic.intrinsic_matrix[0, 0]
        fy = intrinsic.intrinsic_matrix[1, 1]
        cx = intrinsic.intrinsic_matrix[0, 2]
        cy = intrinsic.intrinsic_matrix[1, 2]

        # 3D 좌표 계산
        z = point[2]
        x = (point[0] - cx) * z / fx
        y = (point[1] - cy) * z / fy
        point_3d = np.array([x, y, z]) / 1000.0
        return point_3d

    point1_3d = get_point_3d(point1_3d, intrinsic)
    point2_3d = get_point_3d(point2_3d, intrinsic)
    point3_3d = get_point_3d(point3_3d, intrinsic)

    center_point_3d = (point1_3d + point2_3d + point3_3d) / 3
    # print(f"center_point_3d 2: {center_point_3d}")
    # print(f"pcd.get_center(): {pcd.get_center()}")
    pcd = add_normal_line_to_pcd(pcd, center_point_3d, plane_normal)

    # 3개의 3D 포인트에 빨간색 점 추가
    if point1_3d is not None and point2_3d is not None and point3_3d is not None:
        pcd = add_3d_points_to_pcd(pcd, [point1_3d, point2_3d, point3_3d])

    # pcd = draw_normal_at_center(pcd, center_point_3d, plane_normal)  # type: ignore
    return pcd


def draw_normal_at_center(
    pcd, point, normal: np.ndarray, arrow_length=1.0, arrow_color=[1, 0, 0]
):
    """
    PCD 중심점에 Normal 화살표 그리기
    """
    # PCD 중심점
    center = point

    # Normal
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)

    # 화살표 생성 (Open3D 내장 함수)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.02,
        cone_radius=0.04,
        cylinder_height=arrow_length * 0.8,
        cone_height=arrow_length * 0.2,
    )

    # Z축을 normal 방향으로 회전
    R = rotation_matrix_from_vectors([0, 0, 1], normal)
    arrow.rotate(R, center=[0, 0, 0])
    arrow.translate(center)
    arrow.paint_uniform_color(arrow_color)

    o3d.visualization.draw_geometries(
        [pcd, arrow], window_name="PCD with Single Normal", width=800, height=600
    )

    return arrow


def add_normal_line_to_pcd(
    pcd,
    position: np.ndarray,
    normal: np.ndarray,
    line_length=0.1,
    num_points=50,
    line_color=[1, 0, 0],
):
    """
    Normal 방향 선분을 점들로 표현하여 PCD에 추가
    """
    # 위치 설정

    origin = np.array(position)

    # Normal 벡터
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
    point_color: List[float] = [1, 0, 0],  # 빨간색
    point_size: int = 80,
):
    """
    3D 포인트들을 PCD에 빨간색 점으로 추가
    """
    # 각 3D 포인트 주변에 작은 구체 모양의 점들 생성
    all_point_clouds = []

    for point_3d in points_3d:
        # 구체 표면의 점들 생성
        phi = np.linspace(0, 2 * np.pi, 20)
        theta = np.linspace(0, np.pi, 10)
        phi_grid, theta_grid = np.meshgrid(phi, theta)

        # 작은 구체 반지름
        radius = 0.01

        # 구체 표면 좌표
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

    # 모든 포인트 클라우드 병합
    combined_pcd = pcd
    for sphere_pcd in all_point_clouds:
        combined_pcd = combined_pcd + sphere_pcd

    return combined_pcd


def rotation_matrix_from_vectors(vec1, vec2):
    """두 벡터 사이의 회전 행렬 계산"""
    a = np.array(vec1) / np.linalg.norm(vec1)
    b = np.array(vec2) / np.linalg.norm(vec2)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s == 0:  # 평행한 경우
        if c > 0:
            return np.eye(3)
        else:
            # 180도 회전
            return -np.eye(3)

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))
    return R
