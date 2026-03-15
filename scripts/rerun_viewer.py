#!/usr/bin/env python3
"""
Rerun으로 TSDF 결과 + 카메라 포즈 시각화.

Pose convention:
  - matching: T @ P_source = P_target (source→target, c2w)
  - target=reference(5_cp) → T는 camera→world 변환
  - PLY 좌표: mm 단위, pose translation: meter 단위
  - fused_pcd: world(mm) + flip_yz 적용 상태로 저장
  - projection: un-flip → c2w의 역변환(w2c) → pinhole
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import rerun as rr
except ImportError:
    print("rerun-sdk가 필요합니다: pip install rerun-sdk")
    sys.exit(1)

try:
    import open3d as o3d
except ImportError:
    o3d = None


def _pose_to_rerun_transform(T_c2w: np.ndarray):
    R = T_c2w[:3, :3]
    t = T_c2w[:3, 3]
    return -R.T @ t, R.T


class RerunViewer:

    def __init__(self, app_id: str = "tsdf_cameras", spawn: bool = True, save_path: str | None = None):
        if save_path:
            rr.init(app_id, spawn=False)
            if hasattr(rr, "save"):
                rr.save(save_path)
        else:
            rr.init(app_id, spawn=spawn)
        self._save_path = save_path
        if hasattr(rr, "set_time_sequence"):
            rr.set_time_sequence("frame", 0)

    def log_mesh(self, mesh_path: str, entity_path: str = "world/reconstruction", flip_yz: bool = True):
        if o3d is None:
            raise RuntimeError("Open3D required")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if not mesh.has_vertices():
            return
        if flip_yz:
            mesh.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        rr.log(entity_path, rr.Mesh3D(
            vertex_positions=np.asarray(mesh.vertices, dtype=np.float32),
            triangle_indices=np.asarray(mesh.triangles, dtype=np.uint32),
            vertex_colors=(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8) if mesh.has_vertex_colors() else None,
        ))

    def log_point_cloud(self, pcd_path: str, entity_path: str = "world/reconstruction_pcd", flip_yz: bool = True):
        if o3d is None:
            raise RuntimeError("Open3D required")
        pcd = o3d.io.read_point_cloud(pcd_path)
        if not pcd.has_points():
            return
        if flip_yz:
            pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        pts = np.asarray(pcd.points, dtype=np.float32)
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
        rr.log(entity_path, rr.Points3D(positions=pts, colors=colors, radii=0.002))

    def log_camera_pose(self, T_c2w: np.ndarray, name: str, entity_path: str = "world/cameras", scale: float = 0.1):
        R = T_c2w[:3, :3]
        t = T_c2w[:3, 3]
        path = f"{entity_path}/{name}"
        rr.log(path, rr.Transform3D(
            translation=t.tolist(),
            mat3x3=R.tolist(),
        ))
        rr.log(f"{path}/axes", rr.Arrows3D(
            origins=[[0,0,0],[0,0,0],[0,0,0]],
            vectors=[[scale,0,0],[0,scale,0],[0,0,scale]],
            colors=[[255,0,0],[0,255,0],[0,0,255]],
        ))

    def _ply_to_color_image(self, ply_path: str, config: dict):
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
        points_3d = np.asarray(pcd.points)
        colors = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
        color_image = project_pcd_to_image(
            points_3d, colors=colors, intrinsic_matrix=K,
            image_size=(w, h), depth_range=(0.1, float(depth_max)),
        )
        return (color_image, (w, h, float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2])))

    def log_cameras_from_yaml(self, result_dir: str, reference: str, pose_scale: float = 0.001, entity_path: str = "world/cameras"):
        import yaml
        result_dir = Path(result_dir)
        pairs = []
        for p in sorted(result_dir.glob("*_result.yaml")):
            base = p.stem.replace("_result", "")
            parts = base.split("_")
            if len(parts) >= 4:
                target, source = "_".join(parts[:2]), "_".join(parts[2:4])
                if target == reference:
                    pairs.append((source, str(p)))
        names = [reference] + [s for s, _ in pairs]
        poses = [np.eye(4)]
        self.log_camera_pose(np.eye(4), reference, entity_path=entity_path)
        for source, yaml_path in pairs:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if "transformation" not in data:
                continue
            T = np.array(data["transformation"], dtype=np.float64)
            if pose_scale != 1.0:
                T = T.copy(); T[:3, 3] *= pose_scale
            self.log_camera_pose(T, source, entity_path=entity_path)
            poses.append(T.copy())
        return names, poses

    def log_cameras_from_poses_yaml(self, poses_yaml_path: str | Path, entity_path: str = "world/cameras"):
        import yaml
        with open(poses_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cameras = data.get("cameras", [])
        if not cameras:
            return [], []
        pose_scale = 1.0 if data.get("poses_meter", True) else 0.001
        names, poses = [], []
        for c in cameras:
            name = c.get("name")
            T = np.array(c["transformation"], dtype=np.float64)
            if pose_scale != 1.0:
                T = T.copy(); T[:3, 3] *= pose_scale
            names.append(name)
            poses.append(T)
            self.log_camera_pose(T, name, entity_path=entity_path)
        return names, poses

    # ─── projection ───

    @staticmethod
    def _project_to_camera(
        points_world: np.ndarray,
        T_c2w: np.ndarray,
        K: np.ndarray,
        image_size: tuple[int, int],
        depth_min: float = 0.01,
        depth_max: float = 100.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """World 포인트를 카메라 이미지에 projection.

        fused_pcd(meter)와 pose translation(meter)이 동일 단위.
        c2w 역변환: P_cam = R^T @ (P_world - t)
        """
        R = T_c2w[:3, :3]
        t = T_c2w[:3, 3]  # 둘 다 meter, 변환 불필요

        # c2w 역변환: P_cam = R^T @ (P_world - t)
        # row-vector: (P - t) @ R
        pts_cam = (points_world - t) @ R

        depth = pts_cam[:, 2]
        valid = (depth >= depth_min) & (depth <= depth_max)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        w, h = image_size

        safe_depth = np.where(depth > 0, depth, 1.0)
        u = fx * pts_cam[:, 0] / safe_depth + cx
        v = fy * pts_cam[:, 1] / safe_depth + cy
        valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)

        return np.stack([u, v], axis=1), valid

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
        """각 카메라별 PLY → RGB 이미지 생성 후 Rerun에 로깅."""
        ply_dir = Path(ply_dir)
        ims = config.get("image_size", {})
        w, h = ims.get("width", 640), ims.get("height", 480)

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
            cam_path = f"{entity_path}/{name}/image"
            rr.log(cam_path, rr.Pinhole(
                image_from_camera=[[fx, 0, cx], [0, fy, cy], [0, 0, 1]], width=w, height=h,
                image_plane_distance=0.3,
            ))
            rr.log(cam_path, rr.Image(color_image))
            print(f"[INFO] Frame {i}: {name}")


def main():
    parser = argparse.ArgumentParser(description="Rerun으로 mesh + 카메라 포즈 시각화")
    parser.add_argument("--mesh", type=str, default=None)
    parser.add_argument("--pcd", type=str, default=None)
    parser.add_argument("--poses_yaml", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--reference", type=str, default="5_cp")
    parser.add_argument("--ply_dir", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--pose_scale", type=float, default=0.001)
    parser.add_argument("--no_flip_yz", action="store_true")
    parser.add_argument("--no_spawn", action="store_true")
    parser.add_argument("--save_rrd", type=str, default=None)
    args = parser.parse_args()

    save_path = args.save_rrd
    spawn = not args.no_spawn and not save_path
    if save_path and not save_path.endswith(".rrd"):
        save_path += ".rrd"

    viewer = RerunViewer(app_id="tsdf_cameras", spawn=spawn, save_path=save_path)
    flip_yz = not args.no_flip_yz

    if args.mesh and Path(args.mesh).exists():
        viewer.log_mesh(args.mesh, flip_yz=flip_yz)
    if args.pcd and Path(args.pcd).exists():
        viewer.log_point_cloud(args.pcd, flip_yz=flip_yz)

    camera_names, camera_poses = [], []
    if args.poses_yaml and Path(args.poses_yaml).exists():
        camera_names, camera_poses = viewer.log_cameras_from_poses_yaml(args.poses_yaml)
        print(f"[INFO] Cameras: {camera_names}")
    elif args.result_dir and Path(args.result_dir).exists():
        camera_names, camera_poses = viewer.log_cameras_from_yaml(
            args.result_dir, args.reference, pose_scale=args.pose_scale,
        )
        print(f"[INFO] Cameras: {camera_names}")

    if camera_names and args.ply_dir and Path(args.ply_dir).exists() and args.config_path and Path(args.config_path).exists():
        import yaml
        with open(args.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        viewer.log_camera_frames_with_images(
            camera_names, camera_poses, args.ply_dir, config,
            fused_pcd_path=args.pcd, flip_yz=flip_yz, points_subsample=10,
        )

    if save_path:
        print(f"[INFO] Saved: {save_path}")
    elif not args.no_spawn:
        print("[INFO] Rerun viewer running.")


if __name__ == "__main__":
    main()