#!/usr/bin/env python3
"""
Open3D를 사용해서 포인트 클라우드에서 마우스로 L, R, U 포인트를 선택하고 YAML로 저장하는 스크립트
Windows 환경 호환 버전
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
        description="포인트 클라우드에서 L, R, U 포인트 선택 및 저장"
    )
    parser.add_argument(
        "--pcd_path",
        type=str,
        required=True,
        help="포인트 클라우드 파일 경로 (.ply, .pcd, .xyz 등)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="selected_points.yaml",
        help="출력 YAML 파일 경로 (기본값: selected_points.yaml)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["edit", "manual"],
        default="edit",
        help="선택 방법: edit(VisualizerWithEditing) 또는 manual(수동 좌표 입력)",
    )

    args = parser.parse_args()

    # 파일 경로 검증
    pcd_path = Path(args.pcd_path)
    if not pcd_path.exists():
        print(f"오류: 포인트 클라우드 파일을 찾을 수 없습니다: {pcd_path}")
        return

    # 출력 디렉토리 생성
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 포인트 클라우드 로드
    try:
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        if len(pcd.points) == 0:
            print(f"오류: 포인트 클라우드가 비어있습니다: {pcd_path}")
            return

        print(f"포인트 클라우드 로드 완료: {len(pcd.points)}개 포인트")
    except Exception as e:
        print(f"오류: 포인트 클라우드 로드 실패: {e}")
        return

    selected_points = {}

    if args.method == "edit":
        # VisualizerWithEditing 사용 (기본)
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
                print("--method manual 옵션으로 수동 입력을 시도해보세요.")
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
                    print(
                        f"{point_type}: ({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f})"
                    )

        except Exception as e:
            print(f"VisualizerWithEditing 오류: {e}")
            print("--method manual 옵션으로 수동 입력을 시도해보세요.")
            return

    else:  # manual method
        # 수동 좌표 입력 방법
        print("\n=== 수동 좌표 입력 모드 ===")
        print("포인트 클라우드를 확인하고 좌표를 직접 입력하세요.")

        # 시각화 (참조용)
        vis = o3d.visualization.Visualizer()
        vis.create_window("포인트 클라우드 (참조용)", width=1200, height=800)
        vis.add_geometry(pcd)

        # 좌표축 추가
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        vis.add_geometry(axes)

        print("\n포인트 클라우드가 표시되었습니다.")
        print("창을 보면서 대략적인 좌표를 확인하세요.")
        print("준비되면 창을 닫고 좌표를 입력하세요. (Q를 눌러 닫기)\n")

        vis.run()
        vis.destroy_window()

        # 좌표 입력
        for point_name in ["L", "R", "U"]:
            print(f"\n{point_name} 포인트 좌표 입력:")
            try:
                x = float(input(f"  X 좌표: "))
                y = float(input(f"  Y 좌표: "))
                z = float(input(f"  Z 좌표: "))

                selected_points[point_name] = {
                    "x": x,
                    "y": y,
                    "z": z,
                }
                print(f"  입력된 좌표: ({x:.4f}, {y:.4f}, {z:.4f})")
            except ValueError:
                print("오류: 유효한 숫자를 입력하세요.")
                return

    # YAML 파일 저장
    if selected_points:
        try:
            yaml_data = {
                "selected_points": selected_points,
                "pointcloud_path": str(pcd_path),
                "description": "선택된 L, R, U 포인트의 3D 좌표",
                "selection_method": args.method,
            }

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
