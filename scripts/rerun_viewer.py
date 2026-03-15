#!/usr/bin/env python3
"""
Rerun으로 TSDF 결과 + 카메라 포즈 시각화.
Mesh/Point cloud와 각 뷰의 카메라 포즈를 한 화면에서 확인할 수 있음.

사용 전: pip install rerun-sdk
실행 예: python scripts/rerun_viewer.py --mesh output/SL_ch_260315/5_cp_fused_mesh.ply
         --result_dir output/SL_ch_260315 --reference 5_cp --ply_dir datasets/SL_ch_260315
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 프로젝트 루트
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import rerun as rr  # type: ignore
except ImportError:
    print("rerun-sdk가 필요합니다: pip install rerun-sdk")
    sys.exit(1)

try:
    import open3d as o3d
except ImportError:
    o3d = None


def _pose_to_rerun_transform(T_world_from_camera: np.ndarray):
    """카메라-to-world 4x4 pose → Rerun world-to-camera transform (parent=world, child=camera)."""
    R = T_world_from_camera[:3, :3]
    t = T_world_from_camera[:3, 3]
    # world → camera: inv(T) = [R.T | -R.T@t]
    R_w2c = R.T
    t_w2c = -R.T @ t
    return t_w2c, R_w2c


class RerunViewer:
    """Rerun으로 mesh/pcd + 카메라 포즈 시각화."""

    def __init__(self, app_id: str = "tsdf_cameras", spawn: bool = True, save_path: str | None = None):
        if save_path:
            rr.init(app_id, spawn=False)
            if hasattr(rr, "save"):
                rr.save(save_path)
            self._save_path = save_path
        else:
            rr.init(app_id, spawn=spawn)
            self._save_path = None
        # 시간 시퀀스 (선택, API 버전에 따라 없을 수 있음)
        if hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence("frame", 0)

    def log_mesh(
        self,
        mesh_path: str,
        entity_path: str = "world/reconstruction",
        flip_yz: bool = True,
    ) -> None:
        """Open3D mesh를 로드해 Rerun에 로깅."""
        if o3d is None:
            raise RuntimeError("Open3D required for mesh loading")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if not mesh.has_vertices():
            print(f"[WARN] Empty mesh: {mesh_path}")
            return

        if flip_yz:
            flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)
            mesh.transform(flip)

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        tris = np.asarray(mesh.triangles, dtype=np.uint32)
        rr.log(
            entity_path,
            rr.Mesh3D(
                vertex_positions=verts,
                triangle_indices=tris,
                vertex_colors=np.asarray(mesh.vertex_colors, dtype=np.uint8) if mesh.has_vertex_colors() else None,
            ),
        )

    def log_point_cloud(
        self,
        pcd_path: str,
        entity_path: str = "world/reconstruction_pcd",
        flip_yz: bool = True,
    ) -> None:
        """Point cloud PLY를 로드해 Rerun에 로깅."""
        if o3d is None:
            raise RuntimeError("Open3D required for pcd loading")
        pcd = o3d.io.read_point_cloud(pcd_path)
        if not pcd.has_points():
            print(f"[WARN] Empty pcd: {pcd_path}")
            return

        if flip_yz:
            flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float64)
            pcd.transform(flip)

        pts = np.asarray(pcd.points, dtype=np.float32)
        colors = None
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        rr.log(entity_path, rr.Points3D(positions=pts, colors=colors, radii=0.002))

    def log_camera_pose(
        self,
        T_world_from_camera: np.ndarray,
        name: str,
        entity_path: str = "world/cameras",
        scale: float = 0.1,
    ) -> None:
        """단일 카메라 포즈 (4x4, camera-to-world) 로깅. 좌표축으로 프러스텀 표시."""
        t_w2c, R_w2c = _pose_to_rerun_transform(T_world_from_camera)
        path = f"{entity_path}/{name}"

        # Rerun은 parent→child 변환: world → camera 이므로 inv(pose) 사용
        rr.log(
            path,
            rr.Transform3D(
                translation=t_w2c.tolist(),
                mat3x3=R_w2c.tolist(),
            ),
        )
        # 카메라 좌표축 (camera 프레임에서 X,Y,Z 방향 화살표)
        rr.log(
            f"{path}/axes",
            rr.Arrows3D(
                origins=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                vectors=[[scale, 0, 0], [0, scale, 0], [0, 0, scale]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

    def _ply_to_color_image(self, ply_path: str, config: dict) -> tuple[np.ndarray, tuple[int, int, float, float, float, float]] | None:
        """PLY를 역투영해 RGB 이미지 생성. (image, (w, h, fx, fy, cx, cy)) 또는 None."""
        if o3d is None:
            return None
        pcd = o3d.io.read_point_cloud(ply_path)
        if not pcd.has_points():
            return None
        from core.utils.geometry_utils import project_pcd_to_image

        ims = config.get("image_size", {})
        w, h = ims.get("width", 640), ims.get("height", 480)
        f = float(max(w, h))
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        depth_max = config.get("depth_max", 5000.0)
        depth_range = (0.1, float(depth_max))
        points_3d = np.asarray(pcd.points)
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
        color_image = project_pcd_to_image(
            points_3d, colors=colors, intrinsic_matrix=K, image_size=(w, h), depth_range=depth_range
        )
        return (color_image, (w, h, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])))

    def log_camera_image(
        self,
        name: str,
        entity_path: str,
        ply_path: str,
        config: dict,
    ) -> bool:
        """카메라 프레임에 해당하는 RGB 이미지 + Pinhole 로깅 (Rerun 2D 뷰에 표시)."""
        out = self._ply_to_color_image(ply_path, config)
        if out is None:
            return False
        color_image, (w, h, fx, fy, cx, cy) = out
        path = f"{entity_path}/{name}/image"
        rr.log(path, rr.Pinhole(image_from_camera=[[fx, 0, cx], [0, fy, cy], [0, 0, 1]], width=w, height=h))
        rr.log(path, rr.Image(color_image))
        return True

    def log_cameras_from_yaml(
        self,
        result_dir: str,
        reference: str,
        pose_scale: float = 0.001,
        entity_path: str = "world/cameras",
    ) -> tuple[list[str], list[np.ndarray]]:
        """result_dir 내 *_result.yaml에서 reference가 target인 것만 읽어 카메라 포즈 로깅.
        reference 프레임은 identity, 나머지는 해당 YAML의 transformation (camera-to-world).
        Returns:
            (camera_names, poses): poses[i]는 names[i]의 4x4 camera-to-world (meter).
        """
        import yaml

        result_dir = Path(result_dir)
        pairs = []
        for p in sorted(result_dir.glob("*_result.yaml")):
            base = p.stem.replace("_result", "")
            parts = base.split("_")
            if len(parts) >= 4:
                target = "_".join(parts[:2])
                source = "_".join(parts[2:4])
                if target == reference:
                    pairs.append((source, str(p)))

        names = [reference] + [s for s, _ in pairs]
        poses = [np.eye(4)]  # reference = identity

        self.log_camera_pose(np.eye(4), reference, entity_path=entity_path)

        for source, yaml_path in pairs:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if "transformation" not in data:
                continue
            T = np.array(data["transformation"], dtype=np.float64)
            if pose_scale != 1.0:
                T = T.copy()
                T[:3, 3] *= pose_scale
            self.log_camera_pose(T, source, entity_path=entity_path)
            poses.append(T.copy())

        return names, poses

    def log_cameras_from_poses_yaml(
        self,
        poses_yaml_path: str | Path,
        entity_path: str = "world/cameras",
    ) -> tuple[list[str], list[np.ndarray]]:
        """tsdf_integrate가 저장한 한 번에 포즈 YAML에서 읽어 카메라 포즈 로깅.
        (reference, poses_meter, cameras: [{name, transformation}])
        Returns:
            (camera_names, poses): poses는 이미 meter 단위면 scale 없음.
        """
        import yaml

        with open(poses_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cameras = data.get("cameras", [])
        if not cameras:
            return [], []
        pose_scale = 1.0 if data.get("poses_meter", True) else 0.001
        names = []
        poses = []
        for c in cameras:
            name = c.get("name")
            T = np.array(c["transformation"], dtype=np.float64)
            if pose_scale != 1.0:
                T = T.copy()
                T[:3, 3] *= pose_scale
            names.append(name)
            poses.append(T)
            self.log_camera_pose(T, name, entity_path=entity_path)
        return names, poses

    def log_world_points_for_projection(
        self,
        fused_pcd_path: str | Path,
        entity_path: str = "world/projection_points",
        flip_yz: bool = True,
        subsample: int = 10,
    ) -> None:
        """월드(기준 프레임) 좌표에 Points3D 로깅. 5_cp 뷰에서는 이걸로 투영됨."""
        if o3d is None:
            return
        pcd = o3d.io.read_point_cloud(str(fused_pcd_path))
        if not pcd.has_points():
            return
        pts = np.asarray(pcd.points, dtype=np.float32)
        if flip_yz:
            flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
            pts = pts @ flip.T
        pts = pts[::subsample]
        colors = None
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)[::subsample]
        rr.log(entity_path, rr.Points3D(positions=pts, colors=colors, radii=0.002))

    def _world_to_image_2d(
        self,
        points_world: np.ndarray,
        T_world_from_cam: np.ndarray,
        K: np.ndarray,
        image_size: tuple[int, int],
        depth_min: float = 0.01,
        depth_max: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """월드(기준) 좌표 포인트를 해당 카메라 상대포즈로 옮긴 뒤 2D (u,v)로 투영.
        Returns:
            (uv, valid_mask): uv는 전체 포인트의 (u,v), valid_mask는 이미지 안·앞쪽 bool mask.
        """
        R = T_world_from_cam[:3, :3]
        t = T_world_from_cam[:3, 3]
        # 월드 -> 해당 카메라: P_cam = R.T @ (P_world - t)
        pts_cam = (points_world - t) @ R
        depth = pts_cam[:, 2]
        valid = (depth >= depth_min) & (depth <= depth_max)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        w, h = image_size
        u = fx * pts_cam[:, 0] / depth + cx
        v = fy * pts_cam[:, 1] / depth + cy
        valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        uv = np.stack([u, v], axis=1)
        return uv, valid

    def log_camera_frames_with_images(
        self,
        camera_names: list[str],
        camera_poses: list[np.ndarray],
        ply_dir: str | Path,
        config: dict,
        entity_path: str = "world/cameras",
        fused_pcd_path: str | Path | None = None,
        flip_yz: bool = True,
        points_subsample: int = 10,
    ) -> None:
        """각 카메라별 RGB 이미지 + 상대포즈 적용 후 2D 포인트 로깅.
        포인트는 월드(기준)에 있고, 비기준 프레임 이미지에는 해당 카메라의 상대포즈로 변환한 뒤 투영.

        FIX: projection용 포인트는 원본 좌표계(flip 없음)로 유지해야 함.
             YAML의 pose는 원본 좌표계 기준이므로 flip된 포인트와 섞으면
             reference(identity)에서만 우연히 맞고 나머지에서는 좌표계 불일치 발생.
        """
        ply_dir = Path(ply_dir)
        ims = config.get("image_size", {})
        w, h = ims.get("width", 640), ims.get("height", 480)
        f = float(max(w, h))
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        image_size = (w, h)

        # fused_pcd는 save_reconstruction에서 flip_yz가 적용된 상태로 저장됨 (Y,Z 반전).
        # poses_yaml은 원본 좌표계(flip 전) 기준으로 저장됨.
        # pinhole projection은 Z-forward(양수)를 가정하므로, fused_pcd를 un-flip해서
        # 원본 좌표계로 복원한 뒤 projection해야 함. (flip은 self-inverse)
        points_world_proj = None
        colors_world_proj = None
        if fused_pcd_path and Path(fused_pcd_path).exists() and o3d is not None:
            pcd = o3d.io.read_point_cloud(str(fused_pcd_path))
            if pcd.has_points():
                pts = np.asarray(pcd.points, dtype=np.float64)
                # un-flip: 저장 시 적용된 flip_yz를 되돌림 (flip은 self-inverse)
                if flip_yz:
                    F = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
                    pts = pts @ F.T
                points_world_proj = pts[::points_subsample].copy()
                if pcd.has_colors():
                    colors_world_proj = (np.asarray(pcd.colors) * 255).astype(np.uint8)[::points_subsample]
                print(f"[INFO] Projection source: fused_pcd (un-flipped), "
                      f"{len(points_world_proj)} points, "
                      f"z=[{points_world_proj[:,2].min():.3f}, {points_world_proj[:,2].max():.3f}]")

        for i, name in enumerate(camera_names):
            if hasattr(rr, "set_time_sequence"):
                rr.set_time_sequence("frame", i)
            ply_path = ply_dir / f"{name}.ply"
            if not ply_path.exists():
                continue
            out = self._ply_to_color_image(str(ply_path), config)
            if out is None:
                continue
            color_image, (_, _, fx, fy, cx, cy) = out
            camera_image_path = f"{entity_path}/{name}/image"
            rr.log(
                camera_image_path,
                rr.Pinhole(image_from_camera=[[fx, 0, cx], [0, fy, cy], [0, 0, 1]], width=w, height=h),
            )
            rr.log(camera_image_path, rr.Image(color_image))

            # projection: world 포인트 → 해당 카메라 좌표계 → 2D (+ 원본 색상)
            if points_world_proj is not None and i < len(camera_poses):
                T = camera_poses[i]

                uv_all, valid_mask = self._world_to_image_2d(points_world_proj, T, K, image_size)
                uv = uv_all[valid_mask]
                n_valid = len(uv)

                print(f"[DEBUG] {name}: {n_valid} valid points projected")

                if n_valid > 0:
                    # 원본 PLY 색상이 있으면 함께 로깅
                    pt_colors = None
                    if colors_world_proj is not None:
                        pt_colors = colors_world_proj[valid_mask]
                    rr.log(camera_image_path, rr.Points2D(
                        positions=uv.astype(np.float32),
                        colors=pt_colors,
                        radii=1.5,
                    ))
            print(f"[INFO] Logged frame {i}: {name}")


def main():
    parser = argparse.ArgumentParser(description="Rerun으로 mesh + 카메라 포즈 시각화")
    parser.add_argument("--mesh", type=str, default=None, help="fused mesh PLY 경로")
    parser.add_argument("--pcd", type=str, default=None, help="fused point cloud PLY 경로")
    parser.add_argument(
        "--poses_yaml",
        type=str,
        default=None,
        help="tsdf_integrate가 저장한 포즈 한 번에 YAML (예: output/SL_ch_260315/5_cp_fused_poses.yaml). 있으면 result_dir 대신 사용",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="result YAML 디렉토리 (poses_yaml 없을 때만, *_result.yaml에서 포즈 읽음)",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="5_cp",
        help="기준 프레임 이름 (target)",
    )
    parser.add_argument(
        "--ply_dir",
        type=str,
        default=None,
        help="프레임 PLY 디렉토리 (카메라별 이미지 로깅 시 필요, 예: datasets/SL_ch_260315)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="matcher config YAML (image_size 등, 카메라 이미지 생성용)",
    )
    parser.add_argument("--pose_scale", type=float, default=0.001, help="YAML pose translation 스케일 (mm→m)")
    parser.add_argument("--no_flip_yz", action="store_true", help="mesh/pcd YZ 뒤집기 비활성화")
    parser.add_argument("--no_spawn", action="store_true", help="뷰어 실행 안 함 (--save_rrd와 함께 사용)")
    parser.add_argument("--save_rrd", type=str, default=None, help=".rrd 파일로 저장 (GPU 오류 시 사용 후 rerun file.rrd 로 열기)")
    args = parser.parse_args()

    save_path = args.save_rrd
    spawn = not args.no_spawn and not save_path
    if save_path and not save_path.endswith(".rrd"):
        save_path = save_path + ".rrd"

    viewer = RerunViewer(app_id="tsdf_cameras", spawn=spawn, save_path=save_path)

    flip_yz = not args.no_flip_yz
    if args.mesh and Path(args.mesh).exists():
        viewer.log_mesh(args.mesh, flip_yz=flip_yz)
    if args.pcd and Path(args.pcd).exists():
        viewer.log_point_cloud(args.pcd, flip_yz=flip_yz)
        viewer.log_world_points_for_projection(args.pcd, flip_yz=flip_yz, subsample=10)

    camera_names = []
    camera_poses = []
    if args.poses_yaml and Path(args.poses_yaml).exists():
        camera_names, camera_poses = viewer.log_cameras_from_poses_yaml(args.poses_yaml)
        print(f"[INFO] Logged cameras from poses YAML: {camera_names}")
    elif args.result_dir and Path(args.result_dir).exists():
        camera_names, camera_poses = viewer.log_cameras_from_yaml(
            args.result_dir,
            args.reference,
            pose_scale=args.pose_scale,
        )
        print(f"[INFO] Logged cameras from result_dir: {camera_names}")

    if camera_names and args.ply_dir and Path(args.ply_dir).exists() and args.config_path and Path(args.config_path).exists():
        import yaml
        with open(args.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        viewer.log_camera_frames_with_images(
            camera_names,
            camera_poses,
            args.ply_dir,
            config,
            fused_pcd_path=args.pcd,
            flip_yz=flip_yz,
            points_subsample=10,
        )
        print("[INFO] Logged per-frame images + 2D points (비기준 프레임은 상대포즈 적용 후 투영)")

    if save_path:
        print(f"[INFO] Saved to {save_path}. Open with: rerun {save_path}")
    elif not args.no_spawn and not save_path:
        print("[INFO] Rerun viewer running. Close the window or Ctrl+C to exit.")


if __name__ == "__main__":
    main()