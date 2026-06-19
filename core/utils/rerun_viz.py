"""Rerun 기반 safe zone 디버그 시각화 (독립 실행 스크립트).

파이프라인에 끼워넣지 않는다 — 모듈이 남긴 **결과 파일들만** 가지고 사용자가 직접
실행하는 디버깅 도구다. 파이프라인이 저장한 ``{stem}_result.json`` 과 (있으면) 입력
depth tif 를 읽어 rerun 으로 로깅하고 ``.rrd`` 파일을 남긴다.

그리는 것:
  - 배경 점군 (입력 depth tif backproject, texture 색) — 회색/원색 배경
  - anchor L/R/U (Points3D, 색 구분) — result.json 의 정확한 좌표
  - safe zone OBB (Boxes3D) — 통과 anchor 는 초록, 위반 anchor 는 빨강
  - 위반 화살표 (OBB 표면 최근접점 -> 위반 anchor, 거리 라벨)

좌표 프레임 (두 탭):
  - camera frame: 매칭(runtime) 카메라 기준. anchor·점군만(원본 좌표).
  - robot frame : anchor·점군 -> T_runtime, safe_zone -> T_teach 로 변환해 같은
    로봇 프레임에 모아 비교. teaching/runtime 카메라 위치도 3축으로 표시.

설치:
    pip install -e .[viz]          # rerun-sdk

실행:
    # result.json 직접 지정
    python -m core.utils.rerun_viz path/to/<stem>_result.json

    # 디렉터리 안의 *_result.json 전부
    python -m core.utils.rerun_viz path/to/output_dir --glob

    # 즉시 뷰어 띄우기 (GUI 되는 환경에서)
    python -m core.utils.rerun_viz <stem>_result.json --spawn

기본은 ``{stem}.rrd`` 저장. 나중에 ``rerun {stem}.rrd`` 로 열어본다.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

# 패키지 내부 헬퍼 — safe zone OBB 기하. (이 레포의 geometry_utils API 에 맞춰 호출)
from core.utils.geometry_utils import (
    obb_from_min_max_euler,
    is_point_in_obb,
    obb_violation_vector,
    closest_point_on_obb,
    transform_point_3d,
    transform_safe_zone,
)


# anchor 색 (RGB).
_ANCHOR_COLORS = {"L": (60, 160, 255), "R": (255, 180, 60), "U": (180, 120, 255)}
_OK_COLOR = (60, 200, 90)       # 통과 OBB / anchor
_NG_COLOR = (230, 50, 50)       # 위반 OBB / anchor
_ANCHOR_RADIUS = 14.0           # anchor 구 반경 (mm) — 전부 통일


def _zone_to_obb(zone: Dict[str, Any]):
    """safe zone dict({min,max,euler}) → (center, half, R). euler 없으면 0."""
    euler = zone.get("euler", [0.0, 0.0, 0.0])
    return obb_from_min_max_euler(zone["min"], zone["max"], euler)


def _transform_zone(zone: Dict[str, Any], T: np.ndarray):
    """safe zone dict 를 4x4 T 로 변환한 (center, half, R)."""
    euler = zone.get("euler", [0.0, 0.0, 0.0])
    return transform_safe_zone(zone["min"], zone["max"], euler, T)


def _parse_calibration_matrix(value):
    """camera_calibration 값 → 4x4 numpy. 중첩 리스트/길이16 모두. 형식 틀리면 None."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.size == 16:
        return arr.reshape(4, 4)
    return None


def _require_rerun():
    try:
        import rerun as rr  # noqa: F401
    except ImportError:
        sys.exit(
            "rerun 이 설치되어 있지 않습니다. `pip install -e .[viz]` 로 설치하세요."
        )
    return rr


def _backproject_depth(depth_path: str, K: np.ndarray, texture_path: Optional[str]):
    """입력 depth tif(mm) + K → 배경 점군 (anchor sphere 없는 깨끗한 점군).

    depth tif 는 3채널(grayscale 복제)일 수 있어 채널 0 사용. texture 가 있으면 픽셀
    색을 입힌다. 좌표/단위는 파이프라인과 동일(mm, 카메라 프레임).
    """
    try:
        import tifffile
    except ImportError:
        return None, None
    z = tifffile.imread(depth_path).astype(np.float32)
    if z.ndim == 3:
        z = z[..., 0]
    H, W = z.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    m = z > 0
    K = np.asarray(K, dtype=np.float64)
    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x[m], y[m], z[m]], axis=1)

    cols = None
    if texture_path and os.path.isfile(texture_path):
        try:
            import cv2
            tex = cv2.imread(texture_path, cv2.IMREAD_COLOR)
            if tex is not None:
                tex = cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
                if tex.shape[:2] == z.shape:
                    cols = tex[m]
        except ImportError:
            pass
    return pts, cols


def _load_point_cloud(json_path: str, result: Dict[str, Any]):
    """배경 점군 (points, colors, source) 반환. 없으면 (None, None, None).

    우선순위:
      1. 입력 depth tif backproject (anchor sphere 없는 깨끗한 점군) — result.json 의
         ``inputs.depth`` + ``camera_target_K``. texture 있으면 색도 입힘.
      2. fallback: ``{stem}_with_anchor.ply`` (anchor sphere 섞임, 최후 수단).
    """
    K = result.get("camera_target_K")
    inputs = result.get("inputs") or {}
    depth_path = inputs.get("depth")
    texture_path = inputs.get("texture")

    # 1) 입력 depth backproject (선호).
    if K is not None and depth_path and os.path.isfile(depth_path):
        pts, cols = _backproject_depth(depth_path, np.array(K), texture_path)
        if pts is not None:
            return pts, cols, depth_path

    # 2) fallback: PLY (_with_anchor — sphere 섞임).
    stem = result.get("frame_stem") or os.path.splitext(os.path.basename(json_path))[0]
    if stem.endswith("_result"):
        stem = stem[: -len("_result")]
    base = os.path.dirname(os.path.abspath(json_path))
    candidates = [
        os.path.join(base, f"{stem}_with_anchor.ply"),
    ]
    ply = next((p for p in candidates if os.path.isfile(p)), None)
    if ply is None:
        return None, None, None
    try:
        import open3d as o3d
    except ImportError:
        return None, None, ply
    pcd = o3d.io.read_point_cloud(ply)
    pts = np.asarray(pcd.points)
    cols = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
    return pts, cols, ply


def visualize(json_path: str, *, spawn: bool = False, save: bool = True) -> Optional[str]:
    """result.json 하나를 rerun 으로 로깅. 저장 시 ``.rrd`` 경로 반환."""
    rr = _require_rerun()

    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    stem = result.get("frame_stem") or os.path.splitext(os.path.basename(json_path))[0]
    rec_name = f"safe_zone/{stem}"
    rr.init(rec_name, spawn=spawn)

    rrd_path = None
    if save and not spawn:
        out_dir = os.path.dirname(os.path.abspath(json_path))
        rrd_path = os.path.join(out_dir, f"{stem}.rrd")
        rr.save(rrd_path)

    # 좌표계: 카메라 광학 규약(RDF = X-Right, Y-Down, Z-Forward). depth backproject
    # 점군의 자연스러운 시점이라 뷰어에서 보기 편하다. (데이터 값 변환 아님 — 뷰 힌트만.)
    rr.log("world", rr.ViewCoordinates.RDF, static=True)

    pts, cols, ply = _load_point_cloud(json_path, result)
    if pts is not None and len(pts) > 400_000:
        idx = np.linspace(0, len(pts) - 1, 400_000).astype(int)
        pts = pts[idx]
        cols = cols[idx] if cols is not None else None

    violation = result.get("safe_zone_violation") or {}
    bad_point = violation.get("point")
    anchors = result.get("anchors") or {}
    safe_zones = result.get("safe_zones") or {}
    # 캘리브레이션: teaching(safe_zone 출처) / runtime(anchor·점군 출처) 구분.
    # 둘 다 없으면 camera_calibration 으로 fallback (self-match 호환).
    # (numpy 배열은 `or` 로 fallback 불가 — 명시적 None 체크.)
    _T_fallback = _parse_calibration_matrix(result.get("camera_calibration"))
    T_teach = _parse_calibration_matrix(result.get("T_teach_cam_to_base"))
    if T_teach is None:
        T_teach = _T_fallback
    T_runtime = _parse_calibration_matrix(result.get("T_runtime_cam_to_base"))
    if T_runtime is None:
        T_runtime = _T_fallback

    # ── camera frame 탭: 매칭(runtime) 카메라 기준. anchor·점군 만(원본 좌표). ──
    # safe_zone 은 teaching 카메라 좌표라 runtime 카메라 프레임에 섞으면 의미 없음 → draw_zone=False.
    _log_frame(rr, "world/cam", anchors, safe_zones, bad_point, violation,
               pts, cols, t_anchor=None, t_zone=None, draw_zone=False)
    _log_axes(rr, "world/cam/origin", "cam")

    # ── robot frame 탭: anchor·점군→T_runtime, safe_zone→T_teach 로 모아 비교. ──
    have_robot = T_runtime is not None and T_teach is not None
    if have_robot:
        _log_frame(rr, "world/robot", anchors, safe_zones, bad_point, violation,
                   pts, cols, t_anchor=T_runtime, t_zone=T_teach, draw_zone=True)
        _log_axes(rr, "world/robot/origin", "robot")
        # teaching / runtime 카메라 위치 (robot frame 에서 = 각 T 의 translation).
        _log_camera_markers(rr, "world/robot", T_teach, T_runtime)

    # ── 메타 텍스트 (rerun 뷰어 폰트가 한글 글리프 없어 영문으로만) ──────
    status = "OK" if result.get("ok") else f"FAIL: {result.get('failed_at')}"
    summary = [f"frame: {stem}", f"status: {status}",
               f"frames: cam" + ("  +  robot" if have_robot else "  (no camera_calibration)")]
    if result.get("error_detail"):
        summary.append(f"detail: {result['error_detail']}")
    if violation:
        summary.append(f"violation: point {violation.get('point')} @ {violation.get('position')}")
    for grp in (["world/cam", "world/robot"] if have_robot else ["world/cam"]):
        rr.log(f"{grp}/info", rr.TextDocument("\n".join(summary)), static=True)

    # ── Blueprint: robot / cam 탭. 각 탭은 자기 origin + 초기 카메라 시점. ──
    # robot frame 점군은 runtime 카메라로 변환되므로 blueprint 카메라 시점도 T_runtime 기준.
    _send_tabbed_blueprint(rr, pts, T_runtime, have_robot)

    print(f"[rerun] {stem}: {status}  [{'cam+robot' if have_robot else 'cam'}]"
          + (f"  → {rrd_path}" if rrd_path else "  (spawned)"))
    return rrd_path


def _log_camera_markers(rr, prefix, T_teach, T_runtime):
    """robot frame 에서 teaching / runtime 카메라 위치 + **3축 좌표계** 표시.

    카메라 위치(robot frame) = T 의 translation, 자세 = T[:3,:3].
    origin 축과 동일하게 XYZ(R/G/B) 화살표로 그려 카메라 방향까지 한눈에.
    위치 점은 어느 카메라인지 색 구분(teach=청록, runtime=주황). self-match 면 겹친다.
    """
    L = 100.0
    specs = [("teach", T_teach, (0, 200, 255)), ("runtime", T_runtime, (255, 120, 0))]
    for name, T, color in specs:
        if T is None:
            continue
        pos = T[:3, 3]
        R = T[:3, :3]
        # 카메라 위치 마커 (어느 카메라인지 색 구분).
        rr.log(f"{prefix}/camera/{name}",
               rr.Points3D([pos.tolist()], colors=[color], radii=[18.0],
                           labels=[f"cam:{name}"]), static=True)
        # 카메라 3축 (origin 축과 동일 컨벤션: X=빨강, Y=초록, Z=파랑).
        axes = (R @ (np.eye(3) * L)).T   # 각 행이 회전된 X/Y/Z 축 벡터
        rr.log(f"{prefix}/camera/{name}/axes",
               rr.Arrows3D(origins=[pos.tolist()] * 3, vectors=axes.tolist(),
                           colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
                           labels=[f"{name} X", f"{name} Y", f"{name} Z"]),
               static=True)


def _xf(p, T):
    """점 p 를 T 로 변환 (T 가 None 이면 그대로)."""
    return p if T is None else transform_point_3d(np.asarray(p, dtype=np.float64), T)


def _send_tabbed_blueprint(rr, pts, T_cam_to_base, have_robot):
    """robot frame / camera frame 탭 Blueprint 전송. 각 탭은 자기 origin + 초기 시점.

    - camera frame 탭: world/cam. 카메라 원점이 (0,0,0) 이라 점군 중심을 보게 함.
    - robot frame 탭: world/robot. extrinsic 으로 카메라 위치(T[:3,3])를 알므로 그걸 eye 로,
      점군 쪽을 look_target 으로 → 실제 촬영 시점 재현.
    """
    try:
        import rerun.blueprint as rrb
    except Exception:
        return

    cam_center = pts.mean(0) if (pts is not None and len(pts)) else np.array([0.0, 0.0, 1500.0])

    # EyeControls3D 는 rerun 의 unstable API (>=0.24 부근 추가) — 없는 버전(예: 0.23)
    # 에서는 초기 카메라 시점 지정을 생략하고 기본 Spatial3DView 로 폴백한다.
    # (뷰어에서 더블클릭으로 회전중심을 직접 지정하면 된다.)
    _has_eye = hasattr(rrb, "EyeControls3D") and hasattr(rrb, "Eye3DKind")

    def view3d(origin, name, eye, target):
        if not _has_eye:
            return rrb.Spatial3DView(origin=origin, name=name)
        return rrb.Spatial3DView(
            origin=origin, name=name,
            eye_controls=rrb.EyeControls3D(
                kind=rrb.Eye3DKind.Orbital,
                position=np.asarray(eye, dtype=float).tolist(),
                look_target=np.asarray(target, dtype=float).tolist(),
            ),
        )

    # camera frame 탭: 카메라 원점 뒤쪽에서 점군 바라봄.
    cam_view = view3d("/world/cam", "camera frame",
                      eye=[cam_center[0], cam_center[1], 0.0], target=cam_center)

    if have_robot:
        eye = T_cam_to_base[:3, 3]                                   # robot frame 에서의 카메라 위치
        robot_pts = (T_cam_to_base[:3, :3] @ pts.T + T_cam_to_base[:3, 3:4]).T \
            if (pts is not None and len(pts)) else None
        target = robot_pts.mean(0) if robot_pts is not None else \
            eye + T_cam_to_base[:3, :3] @ np.array([0.0, 0.0, 1500.0])
        robot_view = view3d("/world/robot", "robot frame", eye=eye, target=target)
        blueprint = rrb.Blueprint(rrb.Tabs(robot_view, cam_view, active_tab=0))
    else:
        blueprint = rrb.Blueprint(cam_view)

    try:
        rr.send_blueprint(blueprint)
    except Exception:
        pass


def _log_axes(rr, path: str, prefix: str = ""):
    """원점 + XYZ 축 화살표 (X=빨강, Y=초록, Z=파랑, 100mm). 두 프레임 동일하게."""
    L = 100.0
    p = (prefix + " ") if prefix else ""
    rr.log(path, rr.Arrows3D(
        origins=[[0, 0, 0]] * 3,
        vectors=[[L, 0, 0], [0, L, 0], [0, 0, L]],
        colors=[(255, 0, 0), (0, 255, 0), (0, 0, 255)],
        labels=[f"{p}X", f"{p}Y", f"{p}Z"],
    ), static=True)


def _log_frame(rr, prefix, anchors, safe_zones, bad_point, violation,
               pts, cols, *, t_anchor, t_zone, draw_zone=True):
    """한 프레임(prefix)에 로깅 — anchor/점군 과 safe_zone 에 **다른 변환** 적용.

    좌표계 출처가 다르다:
      - 점군·anchor·위반화살표 = **매칭(runtime) 카메라** 기준 → t_anchor 적용
      - safe_zone           = **템플릿(teaching) 카메라** 기준 → t_zone 적용
    robot frame 탭은 각각 T_runtime / T_teach 로 변환해 같은 robot frame 에 모은다.
    camera frame 탭은 t_anchor=None(원본), draw_zone=False (zone 은 다른 카메라 좌표라
    같은 프레임에 못 섞음).

    inside 판정(OBB 색)은 두 변환을 반영한 좌표로 — anchor(t_anchor) vs zone(t_zone).
    """
    Ta, Tz = t_anchor, t_zone

    def to_a(p):  # anchor/점군 변환
        return _xf(p, Ta) if Ta is not None else np.asarray(p, dtype=np.float64)

    # 배경 점군 (runtime 카메라 기준)
    if pts is not None:
        pts_f = pts if Ta is None else (Ta[:3, :3] @ pts.T + Ta[:3, 3:4]).T
        kwargs = {"radii": 0.6}
        if cols is not None:
            kwargs["colors"] = cols
        rr.log(f"{prefix}/cloud", rr.Points3D(pts_f, **kwargs), static=True)

    # anchor 점
    for name in ("L", "R", "U"):
        a = anchors.get(name)
        if a is None:
            continue
        is_bad = name == bad_point
        color = _NG_COLOR if is_bad else _ANCHOR_COLORS.get(name, (200, 200, 200))
        rr.log(f"{prefix}/anchor/{name}",
               rr.Points3D([to_a(a).tolist()], colors=[color], radii=[_ANCHOR_RADIUS],
                           labels=[f"{name}{' x' if is_bad else ''}"]),
               static=True)
    if bad_point and anchors.get(bad_point) is None and violation.get("position"):
        rr.log(f"{prefix}/anchor/{bad_point}",
               rr.Points3D([to_a(violation["position"]).tolist()],
                           colors=[_NG_COLOR], radii=[_ANCHOR_RADIUS],
                           labels=[f"{bad_point} x"]),
               static=True)

    if not draw_zone:
        return

    # safe zone OBB (teaching 카메라 기준 → t_zone)
    for name in ("L", "R"):
        zone = safe_zones.get(name)
        if zone is None:
            continue
        if Tz is None:
            center, half, R = _zone_to_obb(zone)
        else:
            center, half, R = _transform_zone(zone, Tz)
        # inside 판정: anchor(robot frame, t_anchor) vs zone(robot frame, 위 center/half/R).
        a = anchors.get(name)
        if a is not None:
            inside = is_point_in_obb(to_a(a), center, half, R)
            zcolor = _OK_COLOR if inside else _NG_COLOR
        else:
            zcolor = (150, 150, 150)
        # OBB 회전: rerun(>=0.23) + numpy<2.0 조합에서 Boxes3D 의 quaternion/axis-angle
        # 배치 직렬화가 깨진다(asarray(copy=) — numpy 2.0 전용). 회전을 Transform3D 의
        # mat3x3 으로 주고 박스는 로컬 축정렬(center 원점)로 그리면 회전이 정상 적용되고
        # 경고도 사라진다. 같은 엔티티에 Transform3D + Boxes3D 를 함께 로깅.
        path = f"{prefix}/safe_zone/{name}"
        rr.log(path, rr.Transform3D(translation=np.asarray(center).tolist(),
                                    mat3x3=np.asarray(R).tolist()), static=True)
        rr.log(path, rr.Boxes3D(centers=[[0.0, 0.0, 0.0]],
                                half_sizes=[np.asarray(half).tolist()],
                                colors=[zcolor], labels=[f"safe_zone {name}"],
                                fill_mode="majorwireframe"),
               static=True)

    # 위반 화살표 (zone 표면 최근접점 → 위반 anchor). 양쪽 같은 robot frame 에서.
    if bad_point and safe_zones.get(bad_point) is not None:
        bad_pos = anchors.get(bad_point) or violation.get("position")
        if bad_pos is not None:
            if Tz is None:
                zc, zh, zR = _zone_to_obb(safe_zones[bad_point])
            else:
                zc, zh, zR = _transform_zone(safe_zones[bad_point], Tz)
            p_anchor = to_a(bad_pos)
            dist = obb_violation_vector(p_anchor, zc, zh, zR)[1]
            closest = closest_point_on_obb(p_anchor, zc, zh, zR)
            if dist > 1e-6:
                rr.log(f"{prefix}/violation_arrow",
                       rr.Arrows3D(origins=[closest.tolist()],
                                   vectors=[(p_anchor - closest).tolist()],
                                   colors=[_NG_COLOR], labels=[f"{dist:.1f}mm"]),
                       static=True)


def _collect_json_paths(target: str, use_glob: bool) -> List[str]:
    if os.path.isdir(target) or use_glob:
        d = target if os.path.isdir(target) else os.path.dirname(target) or "."
        return sorted(glob.glob(os.path.join(d, "*_result.json")))
    return [target]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Rerun safe zone 디버그 시각화 (모듈 결과 파일 기반)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("target", help="{stem}_result.json 경로 또는 디렉터리")
    p.add_argument("--glob", action="store_true",
                   help="디렉터리 안의 *_result.json 전부 처리")
    p.add_argument("--spawn", action="store_true",
                   help="rrd 저장 대신 rerun 뷰어 즉시 띄움 (GUI 필요)")
    args = p.parse_args(argv)

    paths = _collect_json_paths(args.target, args.glob)
    if not paths:
        print(f"처리할 *_result.json 을 찾지 못했습니다: {args.target}", file=sys.stderr)
        return 1

    for jp in paths:
        if not os.path.isfile(jp):
            print(f"[skip] 파일 없음: {jp}", file=sys.stderr)
            continue
        visualize(jp, spawn=args.spawn, save=not args.spawn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
