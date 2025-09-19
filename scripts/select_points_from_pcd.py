#!/usr/bin/env python3
"""
Open3D를 사용해서 포인트 클라우드 또는 TIF 파일에서 마우스로 L, R, U 포인트를 선택하고 YAML로 저장하는 스크립트
Windows 환경 호환 버전

지원 파일 형식:
- 포인트 클라우드: .ply, .pcd, .xyz 등
- 깊이 맵: .tif, .tiff

사용 예시:
1. 포인트 클라우드 파일:
   python select_points_from_pcd.py --input_path data.ply

2. TIF 파일 (기본 카메라 파라미터):
   python select_points_from_pcd.py --input_path depth.tif

3. TIF 파일 (사용자 정의 카메라 파라미터):
   python select_points_from_pcd.py --input_path depth.tif --camera_intrinsic 525 525 320 240

4. TIF 파일 (깊이 스케일 조정):
   python select_points_from_pcd.py --input_path depth.tif --depth_scale 1000
"""

import sys
import argparse
import yaml
import numpy as np
import open3d as o3d
from pathlib import Path


# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def tif_to_pointcloud(tif_path, camera_intrinsic=None, depth_scale=1000.0):
    """
    TIF 파일을 포인트 클라우드로 변환

    Args:
        tif_path: TIF 파일 경로
        camera_intrinsic: 카메라 내부 파라미터 (fx, fy, cx, cy)
        depth_scale: 깊이 값 스케일링 팩터 (기본값: 1000.0)

    Returns:
        o3d.geometry.PointCloud: 변환된 포인트 클라우드
    """
    try:
        # tifffile로 TIF 파일 로드
        import tifffile

        depth_array = tifffile.imread(str(tif_path))

        # float32로 변환
        depth_array = depth_array.astype(np.float32)

        # 다차원 배열인 경우 첫 번째 채널만 사용
        if len(depth_array.shape) > 2:
            print(f"다차원 배열 감지: {depth_array.shape}, 첫 번째 채널 사용")
            depth_array = (
                depth_array[:, :, 0] if depth_array.shape[2] > 0 else depth_array[0]
            )

        print(f"TIF 파일 로드 완료: {depth_array.shape}")
        print(f"깊이 값 범위: {depth_array.min():.2f} ~ {depth_array.max():.2f}")

        # Open3D Image 객체로 변환
        height, width = depth_array.shape
        depth_image = o3d.geometry.Image(depth_array.astype(np.uint16))

        # 카메라 내부 파라미터 설정
        if camera_intrinsic is None:
            fx = fy = width  # 기본 focal length
            cx = width / 2.0
            cy = height / 2.0
        else:
            fx, fy, cx, cy = camera_intrinsic

        # 카메라 내부 파라미터 객체 생성
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        # 깊이 이미지를 포인트 클라우드로 변환
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_image, intrinsic, depth_scale=depth_scale
        )
        pcd.scale(1000.0, center=[0, 0, 0])

        print(f"포인트 클라우드 생성 완료: {len(pcd.points)}개 포인트")

        return pcd

    except Exception as e:
        print(f"TIF 파일 처리 오류: {e}")
        return None


def load_pointcloud_or_tif(file_path, camera_intrinsic=None, depth_scale=1000.0):
    """
    파일 확장자에 따라 포인트 클라우드 또는 TIF 파일을 로드

    Args:
        file_path: 파일 경로
        camera_intrinsic: 카메라 내부 파라미터 (TIF 파일용)
        depth_scale: 깊이 값 스케일링 팩터 (TIF 파일용)

    Returns:
        o3d.geometry.PointCloud: 로드된 포인트 클라우드
    """
    file_path = Path(file_path)

    if file_path.suffix.lower() in [".tif", ".tiff"]:
        print(f"TIF 파일로 인식: {file_path}")
        return tif_to_pointcloud(file_path, camera_intrinsic, depth_scale)
    else:
        print(f"포인트 클라우드 파일로 인식: {file_path}")
        try:
            pcd = o3d.io.read_point_cloud(str(file_path))
            if len(pcd.points) == 0:
                print(f"오류: 포인트 클라우드가 비어있습니다: {file_path}")
                return None
            return pcd
        except Exception as e:
            print(f"오류: 포인트 클라우드 로드 실패: {e}")
            return None


class PointPickingVisualization:
    """포인트 피킹을 위한 시각화 클래스"""

    def __init__(self, pcd, output_path):
        self.pcd = pcd
        self.output_path = output_path
        self.picked_points = []
        self.point_names = ["L", "R", "U"]
        self.picked_coords = {}

        # 색상이 없으면 회색으로 초기화
        if not self.pcd.has_colors():
            gray = np.ones((len(self.pcd.points), 3)) * 0.5
            self.pcd.colors = o3d.utility.Vector3dVector(gray)

        # 선택된 포인트를 표시할 구체들
        self.picked_spheres = []

        # Visualizer 생성
        self.vis = o3d.visualization.Visualizer()

    def pick_points(self):
        """포인트 선택 시작"""
        self.vis.create_window(
            "포인트 선택 - Shift+Click으로 L, R, U 순서대로 선택",
            width=1200,
            height=800,
        )
        self.vis.add_geometry(self.pcd)

        # 뷰 설정
        self.vis.reset_view_point(True)

        # 콜백 함수 등록
        self.vis.register_animation_callback(self.animation_callback)

        print("\n=== 포인트 선택 방법 (Windows) ===")
        print("1. Shift + 마우스 왼쪽 클릭으로 포인트를 선택하세요")
        print("2. L, R, U 순서대로 3개 포인트를 선택하세요")
        print("3. 선택된 포인트는 빨간 구체로 표시됩니다")
        print("4. 3개 포인트 선택 후 Q를 눌러 창을 닫으세요")
        print("5. ESC 키로 취소할 수 있습니다")
        print("=====================================\n")

        # pick_points 메소드 사용
        self.vis.run()
        picked = self.vis.get_picked_points()

        # 선택된 포인트 처리
        if len(picked) > 0:
            points_array = np.asarray(self.pcd.points)
            for i, idx in enumerate(picked[:3]):  # 최대 3개만
                if i < len(self.point_names):
                    point_name = self.point_names[i]
                    coord = points_array[idx]
                    self.picked_coords[point_name] = {
                        "x": float(coord[0]),
                        "y": float(coord[1]),
                        "z": float(coord[2]),
                    }
                    print(
                        f"{point_name} 포인트: ({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f})"
                    )

                    # 선택된 포인트에 빨간 구체 추가
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                    sphere.translate(coord)
                    sphere.paint_uniform_color([1, 0, 0])  # 빨간색
                    self.vis.add_geometry(sphere)

        self.vis.destroy_window()

        return self.picked_coords

    def animation_callback(self, vis):
        """애니메이션 콜백 (업데이트용)"""
        return False


def pick_points_interactive(pcd_path, output_path):
    """인터랙티브 포인트 선택 (대체 방법)"""

    # 포인트 클라우드 로드
    pcd = o3d.io.read_point_cloud(str(pcd_path))

    print(f"포인트 클라우드 로드 완료: {len(pcd.points)}개 포인트")
    print("\n=== 인터랙티브 포인트 선택 ===")
    print("Shift + 왼쪽 클릭으로 포인트를 선택하세요")
    print("L, R, U 순서대로 3개를 선택한 후 Q를 눌러 종료하세요")
    print("================================\n")

    # 색상이 없으면 회색으로 설정
    if not pcd.has_colors():
        colors = np.ones((len(pcd.points), 3)) * 0.5
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # 시각화 및 포인트 선택
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("포인트 선택", width=1200, height=800)
    vis.add_geometry(pcd)

    picked_points = []
    point_names = ["L", "R", "U"]
    spheres = []

    def pick_points_callback(vis):
        """포인트 선택 콜백"""
        nonlocal picked_points, spheres

        # get_picked_points를 사용한 포인트 선택
        picked = vis.get_picked_points()

        if len(picked) > len(picked_points):
            # 새로 선택된 포인트
            new_picks = picked[len(picked_points) :]
            points_array = np.asarray(pcd.points)

            for idx in new_picks:
                if len(picked_points) < 3:
                    point_name = point_names[len(picked_points)]
                    coord = points_array[idx]
                    picked_points.append(
                        {"name": point_name, "coord": coord, "index": idx}
                    )

                    print(
                        f"{point_name} 포인트 선택: ({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f})"
                    )

                    # 빨간 구체로 표시
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                    sphere.translate(coord)
                    sphere.paint_uniform_color([1, 0, 0])
                    vis.add_geometry(sphere, reset_bounding_box=False)
                    spheres.append(sphere)

                    if len(picked_points) >= 3:
                        print("\n3개 포인트 모두 선택됨. Q를 눌러 종료하세요.")

        return False

    # 키보드 콜백 등록 (S 키로 선택 모드 활성화)
    vis.register_key_callback(ord("S"), pick_points_callback)

    vis.run()
    vis.destroy_window()

    return picked_points


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="포인트 클라우드 또는 TIF 파일에서 L, R, U 포인트 선택 및 저장"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="입력 파일 경로 (.ply, .pcd, .xyz, .tif, .tiff 등)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="selected_points.yaml",
        help="출력 YAML 파일 경로 (기본값: selected_points.yaml)",
    )

    parser.add_argument(
        "--camera_intrinsic",
        type=float,
        nargs=4,
        metavar=("FX", "FY", "CX", "CY"),
        default=[2344.069, 2344.4, 989.063, 807.029],
        help="카메라 내부 파라미터 (TIF 파일용): fx fy cx cy",
    )
    parser.add_argument(
        "--depth_scale",
        type=float,
        default=1000.0,
        help="깊이 값 스케일링 팩터 (TIF 파일용, 기본값: 1000.0)",
    )

    args = parser.parse_args()

    # 파일 경로 검증
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"오류: 입력 파일을 찾을 수 없습니다: {input_path}")
        return

    # 출력 디렉토리 생성
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 카메라 내부 파라미터 설정
    camera_intrinsic = args.camera_intrinsic

    # 포인트 클라우드 또는 TIF 파일 로드
    pcd = load_pointcloud_or_tif(input_path, camera_intrinsic, args.depth_scale)

    if pcd is None:
        print("오류: 포인트 클라우드 로드 실패")
        return

    print(f"포인트 클라우드 로드 완료: {len(pcd.points)}개 포인트")

    selected_points = {}

    try:
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(
            "Shift+Click으로 L, R, U 포인트 선택",
            width=1200,
            height=800,
        )
        vis.add_geometry(pcd)

        print("\n=== 포인트 선택 방법 ===")
        print("1. Shift + 왼쪽 마우스 클릭으로 포인트를 선택하세요")
        print("2. L, R, U 순서대로 3개 포인트를 선택하세요")
        print("3. 선택 후 Q를 눌러 창을 닫으세요")
        print("=======================\n")

        vis.run()
        vis.destroy_window()

        # 선택된 포인트들 가져오기
        picked_points = vis.get_picked_points()

        if len(picked_points) < 3:
            print(
                f"경고: 3개 포인트가 모두 선택되지 않았습니다. (현재: {len(picked_points)}개)"
            )
            return

        # 포인트 좌표 추출
        points_array = np.asarray(pcd.points)

        for i, point_idx in enumerate(picked_points[:3]):
            if point_idx < len(points_array):
                point = points_array[point_idx]
                point_type = ["L", "R", "U"][i]
                selected_points[point_type] = {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                }
                print(f"{point_type}: ({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f})")

    except Exception as e:
        print(f"VisualizerWithEditing 오류: {e}")
        return

    # YAML 파일 저장
    if selected_points:
        try:
            yaml_data = {
                "selected_points": selected_points,
                "input_path": str(input_path),
                "description": "선택된 L, R, U 포인트의 3D 좌표",
                "file_type": input_path.suffix.lower(),
            }

            # TIF 파일인 경우 추가 정보 저장
            if input_path.suffix.lower() in [".tif", ".tiff"]:
                yaml_data.update(
                    {
                        "camera_intrinsic": camera_intrinsic,
                        "depth_scale": args.depth_scale,
                    }
                )

            with open(output_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    yaml_data, f, default_flow_style=False, allow_unicode=True, indent=2
                )

            print(f"\n포인트 좌표가 저장되었습니다: {output_path}")
            print("저장된 좌표:")
            for point_type, coords in selected_points.items():
                print(
                    f"  {point_type}: ({coords['x']:.4f}, {coords['y']:.4f}, {coords['z']:.4f})"
                )

        except Exception as e:
            print(f"오류: YAML 파일 저장 실패: {e}")


if __name__ == "__main__":
    main()
