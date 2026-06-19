# Safe Zone & Rerun Viz 작업 정리

> 세션 작업 로그 (2026-06-15 ~ 06-18). 브랜치: `feature/bruce/safe-zone-check-viz`
> 대상 모듈: `crp_matching` (autochecker matching)

---

## 1. 한 일 요약

1. **Safe zone OBB half 계산 버그 수정** (실제 검사 판정에 영향)
2. **rerun 기반 디버그 시각화 도입** (camera frame / robot frame 탭)
3. **잔차(residual PCD) 제거 → 3D correspondence plot 으로 대체** (open3d EGL segfault 회피)
4. **3D RANSAC 파라미터 튜닝** (`max_correspondence_distance` 0.8 → 3.0mm)
5. **NX4 self-match 셋업** + 전용 실행 스크립트

---

## 2. 핵심 버그 수정: Safe zone OBB half

### 증상
회전된 safe zone OBB 가 점군 대비 방향·크기가 뒤틀려 보임. safe zone 검사(`is_point_in_obb`)도 틀린 판정.

### 원인
프론트엔드(Three.js `SafeZoneBox`)가 저장하는 `min`/`max` 는 **회전된 박스의 월드 대각 꼭짓점**인데,
`safe_zone_to_obb` 가 이를 축 정렬로 오해해 `half = |max - min| / 2` 로 계산.
`euler != 0` 이면 half 가 틀림 (예: z축 25 vs 실제 80).

### 수정
`min`/`max` 를 euler 로 **역회전(Rᵀ)한 뒤** half 산출 (프론트 복원 로직과 동일):

```
center = (min + max) / 2
R      = euler_to_rotation_matrix(euler)          # 로컬→월드
half   = |Rᵀ·(max - center) - Rᵀ·(min - center)| / 2
```

euler=0 이면 종전과 동일. → `src/crp_matching/utils/geometry.py: safe_zone_to_obb`

### 검증
NX4 데이터에서 half 가 정확히 `[30, 45, 80]` (프론트 DEFAULT_SIZE 와 일치) 복원,
selected_points 로컬좌표 `[0, 35, 15]` 로 zone 안에 정확히 들어감.

---

## 3. 좌표 프레임 / extrinsic 정리 (중요)

### safe zone 검사는 robot base 절대위치 기준이어야 함
로봇은 base 좌표계로 움직이므로, 카메라가 어디 있든 anchor 의 **robot base 절대위치**가
safe zone 안에 있어야 한다. 카메라가 움직이면 카메라 프레임 anchor 는 달라져도 robot base
anchor 는 같은 절대위치로 모인다.

### 출처가 다른 두 변환
- **safe_zone** = 템플릿(teaching) 카메라 기준 → `T_teach` 로 robot base 변환
- **anchor / ply** = 매칭(runtime) 카메라 기준 → `T_runtime` 로 robot base 변환

`T_cam_to_base` 의미: **base 좌표계에서 본 cam pose** = **cam→base 점 변환**.
`T_cam_to_base @ p_cam = p_base` (역행렬 미적용, 그대로 곱함).

> 키 이름: `camera_calibration` → **`T_cam_to_base`** 로 통일.
> result.json 엔 `T_teach_cam_to_base` / `T_runtime_cam_to_base` 구분 저장.

### self-match 의 함정 (디버깅으로 규명)
같은 이미지(self-match)인데 NG 가 났던 진짜 원인:
- **target 카메라 intrinsic 불일치** — NX4 데이터에 Tucson cx=989 를 써서 32px 어긋남
- → pose 에 ~1° 회전 + Y 평행이동 → 먼 anchor(z≈1574)에서 ~60mm 증폭 → zone 밖
- intrinsic 을 NX4(cx=1021)로 맞추니 pose ≈ identity(0.03°), anchor = selected_points(오차 0.3mm), OK

---

## 4. Rerun 디버그 시각화

`src/crp_matching/utils/rerun_viz.py` — 모듈 결과 파일(result.json + depth)만으로 `.rrd` 생성.
open3d EGL 렌더가 WSL 에서 segfault 나는 문제를 우회 + 인터랙티브 3D.

### 탭 구성 (Tabs blueprint)
- **camera frame** (`world/cam`): 매칭(runtime) 카메라 기준. anchor·ply 만 (원본 좌표).
- **robot frame** (`world/robot`): anchor·ply→`T_runtime`, safe_zone→`T_teach` 로 변환해 같은
  robot frame 에 모아 비교. teaching/runtime 카메라 위치도 3축으로 표시.

### 각 프레임 내용
- 원점 3축 (X=빨강, Y=초록, Z=파랑)
- 배경 점군 (depth backproject + texture 색)
- anchor L/R/U (크기 통일 14, 위반은 빨강)
- safe zone OBB (통과 초록 / 위반 빨강, `fill_mode="majorwireframe"`)
- 위반 화살표 (OBB 표면 최근접점 → 위반 anchor, 거리 라벨)

### 초기 카메라 시점
extrinsic 으로 카메라 위치를 알므로 (`T_cam_to_base[:3,3]`), 그 위치를 eye 로 → 실제 촬영 시점 재현.

### 주의 사항
- rerun 뷰어 폰트가 한글 글리프 없음 → `world/info` 텍스트는 영문만.
- 좌표값 변환 아니라 뷰 방향만 — 데이터는 그대로.

---

## 5. 잔차 → Correspondence Plot 교체

### 제거: residual PCD
`viz/residual_pcd.py` 삭제. open3d EGL OffscreenRenderer 라 WSL 에서 segfault(core dump),
점군 색만으론 정합 품질 안 보임.

### 추가: 3D correspondence plot (matplotlib)
`pipeline` 의 pose 직후 `visualize_3d_correspondences` 호출 (`config.corr_plot_path` 있을 때만).
- **XY (top-down) 한 장**, Y축 반전 (카메라 Y-down 이라 안 뒤집으면 물체가 뒤집혀 보임)
- 선 없음 (`line_step=0`) — 좋은 매칭이면 점이 겹쳐 선 무의미
- 배경에 target/source 점군 옅게 → 부품 형체 맥락
- 제목에 inlier 수 + rmse + fitness
- open3d 무관 → WSL 안전

---

## 6. 3D RANSAC 파라미터 튜닝

### 증상
2D RANSAC(homography) 후엔 매칭 고루 분포하는데, **3D RANSAC inlier 가 한쪽으로 쏠림**.

### 원인 / 수정
`max_correspondence_distance=0.8mm` 가 너무 빡빡 (inlier 56%). depth 정밀도 노이즈가 큰
영역이 다 outlier 로 버려져 평평한 부분만 살아남음.

| dist | inlier 비율 |
|------|------|
| 0.8mm | 56% (쏠림) |
| 1.5mm | 79% |
| **3.0mm** | **96%** (고루 분포) |

→ `max_correspondence_distance` **0.8 → 3.0mm**, `ransac_n` 10 → 7 (NX4 config).
depth 정밀도 여유 반영. pose 회전 0.01~0.03° 로 안정 유지.

### 품질 지표 (RANSAC registration 결과)
- **fitness** = inlier 비율 (inlier수/전체). "얼마나 많이 맞나" (넓이).
- **inlier_rmse** = inlier 평균 오차(mm). "얼마나 정확히 맞나" (정밀도).
- dist 키우면 fitness↑ rmse↑ (트레이드오프).
- **chamfer** 는 rmse 와 거의 중복(추세 동일) → 표시 제거. (pose identity 게이트용으로만 내부 계산)
  - chamfer 비용 우려는 기우 — inlier(수천개)에 torch ~2ms, 이미 매번 계산하던 값.

### upsample_res 실험 (효과 미미)
`864×1152` → `1152×1536` 올려도 inlier/rmse 거의 동일 (RANSAC 무작위성 범위). VRAM 6.8GB 여유.
→ **864×1152 유지**. 정합 품질은 dist 튜닝이 훨씬 컸음.

---

## 7. 실행 스크립트 / 데이터 셋업

### 루트 스크립트
- `run_match.sh` — 매칭 실행 (template/target/config 환경변수, self-match 지원, SAVE 레벨)
- `run_viz.sh` — rerun viz 실행 (.rrd 생성, open3d 무관)
- 매칭/viz 분리 → open3d segfault 격리

### NX4 self-match 셋업 (`data/NX4_selfmatch/`, gitignore)
- `matcher.teaching.param.yaml` — template (T_cam_to_base 4x4, NX4 intrinsic, safe_zones)
- `matcher.config.yaml` — target 카메라 intrinsic 을 NX4 로 맞춤 (cx 1021), 3.0mm
- hood.6.tif/png 심볼릭 링크 (self-match: source=target)

### 폴더 전용 스크립트 (`data/013107_HUA 686654_NX4/`)
- `match.sh` / `viz.sh` — 그 폴더 데이터 전용 (NX4 template/config 사용)

### runner 변경 (`tests/runner/module.py`)
- `{stem}_result.json` 저장 (open3d viz 전에 — segfault 나도 json 안전)
- teaching/runtime calib 구분 저장
- 성공 로그에 `rmse / fitness / inlier` 표시

---

## 8. 검증 결과

### NX4 template + 013107 target (진짜 다른 view 5장, 3.0mm)
5장 전부 OK. anchor L 이 selected_point `[141.65, 134.53, 1574.18]` 근처 ±몇 mm 로 일관.

| frame | inlier | rmse | fitness |
|-------|--------|------|---------|
| 013107 | 2775 | 1.92mm | 0.64 |
| 013337 | 3646 | 1.97mm | 0.80 |
| 013747 | 3300 | 2.08mm | 0.76 |
| 013837 | 2997 | 1.92mm | 0.67 |
| 014107 | 3401 | 1.87mm | 0.78 |

### 테스트
- `tests/test_safe_zone.py` (단위, geometry/OBB)
- `tests/test_safe_zone_scenarios.py` (시나리오, OK/NG/skip/frame)
- `tests/test_rerun_viz.py` (quaternion, rrd 생성, robot frame calib)
- 전부 통과.

---

## 9. 커밋 (브랜치 `feature/bruce/safe-zone-check-viz`)

이전 세션 (dev/0.x 머지 + 태그 hmma-0.1.0-rc2):
- `feat(safe-zone)`: anchor OBB safe zone 검사 추가 (SAFE_ZONE_VIOLATION)
- `build(fused-local-corr)`: PyPI prebuilt 대신 GitHub source 빌드 pin

이번 세션:
- `fix(safe-zone)`: OBB half 계산 버그 수정 (회전된 박스 월드 min/max 역회전)
- `feat(viz)`: rerun 기반 safe zone 디버그 시각화 (cam/robot frame 탭)
- `refactor(viz)`: residual PCD 제거, 3D correspondence plot 으로 대체

> 미커밋: chamfer→rmse/fitness 교체, correspondence plot 개선(배경/XY-only/Y반전),
> 3.0mm 튜닝(NX4 config 는 gitignore). 커밋 정리 필요.

---

## 10. 변경 파일

**구현**
- `src/crp_matching/utils/geometry.py` — safe_zone_to_obb(half 수정), closest_point_on_obb, obb_violation_vector
- `src/crp_matching/core/pipeline.py` — check_safe_zones, T_cam_to_base 인자, correspondence plot, rmse/fitness
- `src/crp_matching/module/error_handler.py` — ErrorCode.SAFE_ZONE_VIOLATION (504)
- `src/crp_matching/module/module.py` — failed_at→ErrorCode 매핑, T_cam_to_base 배선
- `src/crp_matching/module/protocol.py` — ReqMatch.T_cam_to_base
- `src/crp_matching/utils/rerun_viz.py` (신규) — rerun 시각화
- `src/crp_matching/viz/_recon_viz.py` — visualize_3d_correspondences (배경/XY/Y반전)
- `src/crp_matching/viz/residual_pcd.py` (삭제)

**스크립트 / 테스트**
- `run_match.sh`, `run_viz.sh` (신규)
- `tests/runner/module.py` — result.json, calib 구분, rmse/fitness 로그
- `tests/test_rerun_viz.py` (신규), `tests/test_safe_zone*.py`

**설정**
- `configs/Tucson/matcher.config.yaml`, `matcher.teaching.param.yaml` — T_cam_to_base, safe_zones
- `pyproject.toml` — optional extra `[viz]` = rerun-sdk

---

## 11. 핵심 코드 (다른 레포 반영용)

> 의존성: `numpy` 만. `Tuple` 은 `typing` 에서. euler 규약은 Three.js 'XYZ' = scipy intrinsic 'XYZ'.

### 11.1 geometry — OBB 기하

```python
import numpy as np
from typing import Tuple, Optional, Dict, Any


def euler_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Three.js 'XYZ' (intrinsic) = R = Rx·Ry·Rz = scipy from_euler('XYZ')."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return Rx @ Ry @ Rz


def safe_zone_to_obb(zone: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """safe zone {min,max,euler} → (center, half, R).

    *** 핵심 버그 수정 ***: min/max 는 '회전된 박스의 월드 대각 꼭짓점'이므로
    half 는 |max-min|/2 가 아니라 로컬 프레임으로 역회전(Rᵀ) 후 구해야 한다.
    """
    mn = np.asarray(zone["min"], dtype=np.float64)
    mx = np.asarray(zone["max"], dtype=np.float64)
    center = (mn + mx) / 2.0
    rx, ry, rz = zone.get("euler", [0.0, 0.0, 0.0])
    R = euler_to_rotation_matrix(rx, ry, rz)
    local_min = R.T @ (mn - center)
    local_max = R.T @ (mx - center)
    half = np.abs(local_max - local_min) / 2.0          # ← 역회전 후 크기
    return center, half, R


def is_point_in_obb(point, center, half, R) -> bool:
    """p_local = Rᵀ·(point-center); 각 축 |p_local| ≤ half 면 내부."""
    p = np.asarray(point, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    p_local = np.asarray(R, dtype=np.float64).T @ p
    return bool(np.all(np.abs(p_local) <= np.asarray(half, dtype=np.float64)))


def transform_point_3d(point, T) -> np.ndarray:
    """4x4 T 로 점 변환. T @ [x,y,z,1]."""
    p = np.append(np.asarray(point, dtype=np.float64), 1.0)
    return (np.asarray(T, dtype=np.float64) @ p)[:3]


def transform_safe_zone(zone, T) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OBB 를 4x4 T 로 변환: center'=R_T·center+t_T, R'=R_T·R, half 불변."""
    center, half, R = safe_zone_to_obb(zone)
    T = np.asarray(T, dtype=np.float64)
    R_T, t_T = T[:3, :3], T[:3, 3]
    return R_T @ center + t_T, half, R_T @ R


def closest_point_on_obb(point, center, half, R) -> np.ndarray:
    """OBB 표면(또는 내부) 최근접점 — 로컬 clamp 후 월드 복원. (위반 화살표용)"""
    R = np.asarray(R, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    half = np.asarray(half, dtype=np.float64)
    p_local = R.T @ (np.asarray(point, dtype=np.float64) - center)
    return R @ np.clip(p_local, -half, half) + center


def obb_violation_vector(point, center, half, R) -> Tuple[np.ndarray, float]:
    """(vector, distance) — 최근접 표면점→point 벡터와 거리(mm). 내부면 0."""
    closest = closest_point_on_obb(point, center, half, R)
    vec = np.asarray(point, dtype=np.float64) - closest
    return vec, float(np.linalg.norm(vec))
```

### 11.2 check_safe_zones (검사 본체)

```python
def check_safe_zones(
    *,
    anchors: Dict[str, np.ndarray],          # {"L": (3,), "R": (3,)} 카메라 프레임 anchor
    safe_zones: Optional[Dict[str, Any]],    # {"L": {min,max,euler}, "R": {...}} or None
    teaching_calibration: Optional[np.ndarray] = None,   # T_teach (cam→base) — safe_zone 용
    runtime_calibration: Optional[np.ndarray] = None,    # T_runtime (cam→base) — anchor 용
) -> Optional[Dict[str, Any]]:
    """위반 시 {"point": "L"|"R", "position": [x,y,z]} 반환, 통과/skip 시 None.

    두 calib 모두 있으면 robot base 프레임에서 검사 (anchor→runtime, zone→teaching).
    하나라도 없으면 카메라 프레임 직접 비교 (하위 호환). 검사 대상은 L/R 뿐 (U 제외).
    """
    if not safe_zones:
        return None
    use_robot_frame = (teaching_calibration is not None
                       and runtime_calibration is not None)
    for name in ("L", "R"):
        zone = safe_zones.get(name)
        point_cam = anchors.get(name)
        if zone is None or point_cam is None:
            continue
        if use_robot_frame:
            point = transform_point_3d(point_cam, runtime_calibration)
            center, half, R = transform_safe_zone(zone, teaching_calibration)
        else:
            point = np.asarray(point_cam, dtype=np.float64)
            center, half, R = safe_zone_to_obb(zone)
        if not is_point_in_obb(point, center, half, R):
            position = [float(v) for v in np.asarray(point_cam, dtype=np.float64)]
            return {"point": name, "position": position}   # 카메라 프레임 좌표로 보고
    return None
```

### 11.3 pipeline 통합 (pose 적용된 anchor 검사)

3D 경로에서 anchor(L/R/U) 확정 + depth correction 후, return 직전에:

```python
safe_zones = template_param.get("safe_zones")
if safe_zones:
    teaching_calib = _parse_calibration_matrix(template_param.get("T_cam_to_base"))
    runtime_calib = _parse_calibration_matrix(
        T_cam_to_base if T_cam_to_base is not None
        else config.get("T_cam_to_base"))        # 인자 우선, 없으면 config fallback
    violation = check_safe_zones(
        anchors={"L": result1_3d, "R": result2_3d},
        safe_zones=safe_zones,
        teaching_calibration=teaching_calib,
        runtime_calibration=runtime_calib,
    )
    if violation is not None:
        # 실패 처리 — MatchResult/PipelineResult 에 error_code=SAFE_ZONE_VIOLATION 으로.
        # (디버그용으로 위반 시점 L/R/U anchor 도 violation 에 같이 담아두면 좋음)
        return _fail(error_code="SAFE_ZONE_VIOLATION",
                     detail=f"point {violation['point']} {violation['position']} "
                            f"is outside its safe zone",
                     safe_zone_violation=violation)


def _parse_calibration_matrix(value):
    """T_cam_to_base → 4x4 numpy. 중첩 리스트/길이16 모두. 형식 틀리면 None."""
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.size == 16:
        return arr.reshape(4, 4)
    return None
```

### 11.4 ErrorCode

```python
# error_handler.py — Pose/depth errors (500-599)
SAFE_ZONE_VIOLATION = (504, "Anchor outside its safe zone")
# 숫자코드 MMMSEEE: severity 3 → 30504
```

### 11.5 config / template 스키마

```yaml
# matcher.teaching.param.yaml (matching_model 아래)
matching_model:
  selected_points: {L: {x,y,z}, R: {...}, U: {...}}
  # teaching 카메라→robot base 4x4 (T_cam_to_base @ p_cam = p_base, 역행렬 미적용)
  T_cam_to_base: [[...4x4...]]
  # safe zone (L/R OBB) — min/max 는 '회전된 박스의 월드 대각 꼭짓점', euler 는 Three.js 'XYZ'
  safe_zones:
    L: {min: [x,y,z], max: [x,y,z], euler: [rx,ry,rz]}
    R: {min: [...], max: [...], euler: [...]}

# matcher.config.yaml (top-level) — runtime fallback
T_cam_to_base: [[...4x4...]]
```

### 반영 시 체크리스트 (백엔드/실제 레포)
1. **`camera_calibration` → `T_cam_to_base`** 키 통일 (teaching template + matcher config).
2. `safe_zones` / `T_cam_to_base` 는 teaching yaml 의 **`matching_model:` 아래**에 nesting.
3. `safe_zone_to_obb` 의 half 는 **역회전 후** 계산 (euler!=0 이면 |max-min|/2 는 틀림).
4. `check_safe_zones` 결과로 분기 — `SAFE_ZONE_VIOLATION`(504)로 실패 반환.
5. req_match 인자 `T_cam_to_base` (runtime), 미전달 시 config fallback.
6. 운영(teaching≠runtime 카메라)에선 두 calib 다 줘야 robot base 절대위치 검사가 됨.

---

## 12. Rerun 디버그 시각화 코드

> **전체 코드: `docs/code_ref/rerun_viz.py`** (491줄, numpy + rerun-sdk + 선택적 tifffile/cv2/open3d).
> 아래는 이해·이식에 필요한 핵심 함수만 발췌. 디버그 전용이라 백엔드 필수 아님.

### 12.1 입력 / 출력
- **입력**: runner 가 저장한 `{stem}_result.json` (anchors, safe_zones, T_teach/T_runtime_cam_to_base,
  camera_target_K, inputs.depth/texture). 점군은 depth tif backproject.
- **출력**: `{stem}.rrd` (rerun 뷰어로 열기). camera frame / robot frame 두 탭.

### 12.2 핵심: 회전행렬 → quaternion (rerun Boxes3D 용)
rerun `Boxes3D` 는 회전행렬을 직접 안 받고 quaternion(xyzw)만 받음:

```python
def _rotation_matrix_to_quaternion_xyzw(R):
    try:
        from scipy.spatial.transform import Rotation
        return Rotation.from_matrix(R).as_quat().tolist()   # scipy: xyzw
    except Exception:
        ...  # 순수 numpy fallback (Shepperd's method)
```

### 12.3 핵심: 두 출처를 각각 변환해 한 프레임에 로깅
**anchor/ply 는 runtime calib, safe_zone 은 teaching calib** 로 변환 (출처가 다름):

```python
def _log_frame(rr, prefix, anchors, safe_zones, bad_point, violation,
               pts, cols, *, t_anchor, t_zone, draw_zone=True):
    """t_anchor: 점군·anchor·위반화살표 변환 / t_zone: safe_zone 변환. None 이면 원본(카메라 프레임)."""
    def to_a(p):
        return transform_point_3d(p, t_anchor) if t_anchor is not None else np.asarray(p, float)

    # 점군 (runtime 카메라 기준)
    if pts is not None:
        pts_f = pts if t_anchor is None else (t_anchor[:3,:3] @ pts.T + t_anchor[:3,3:4]).T
        rr.log(f"{prefix}/cloud", rr.Points3D(pts_f, colors=cols, radii=0.6), static=True)

    # anchor 점 (위반은 빨강)
    for name in ("L","R","U"):
        a = anchors.get(name)
        if a is None: continue
        color = _NG_COLOR if name==bad_point else _ANCHOR_COLORS[name]
        rr.log(f"{prefix}/anchor/{name}",
               rr.Points3D([to_a(a).tolist()], colors=[color], radii=[14.0]), static=True)

    if not draw_zone:   # camera frame 탭: zone 은 다른 카메라 좌표라 생략
        return

    # safe zone OBB (teaching 기준) + inside 판정으로 색
    for name in ("L","R"):
        zone = safe_zones.get(name)
        if zone is None: continue
        center, half, R = (transform_safe_zone(zone, t_zone) if t_zone is not None
                           else safe_zone_to_obb(zone))
        quat = _rotation_matrix_to_quaternion_xyzw(R)
        a = anchors.get(name)
        inside = a is not None and is_point_in_obb(to_a(a), center, half, R)
        zcolor = _OK_COLOR if inside else _NG_COLOR
        rr.log(f"{prefix}/safe_zone/{name}",
               rr.Boxes3D(centers=[center.tolist()], half_sizes=[half.tolist()],
                          quaternions=[rr.Quaternion(xyzw=quat)], colors=[zcolor],
                          fill_mode="majorwireframe"), static=True)
    # 위반 화살표: closest_point_on_obb → 위반 anchor (생략, code_ref 참고)
```

### 12.4 핵심: 탭 + 초기 카메라 시점 (Blueprint)
extrinsic 으로 카메라 위치를 알므로(`T[:3,3]`) 그걸 eye 로 → 실제 촬영 시점:

```python
import rerun.blueprint as rrb
robot_view = rrb.Spatial3DView(origin="/world/robot", name="robot frame",
    eye_controls=rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital,
        position=T_cam_to_base[:3,3].tolist(),       # robot frame 에서의 카메라 위치
        look_target=robot_pts.mean(0).tolist()))     # 점군 중심
cam_view = rrb.Spatial3DView(origin="/world/cam", name="camera frame", ...)
rr.send_blueprint(rrb.Blueprint(rrb.Tabs(robot_view, cam_view, active_tab=0)))
```

### 12.5 visualize() 흐름
```python
def visualize(json_path, *, spawn=False, save=True):
    result = json.load(...)
    rr.init(f"safe_zone/{stem}", spawn=spawn)
    if save: rr.save(f"{stem}.rrd")
    rr.log("world", rr.ViewCoordinates.RDF, static=True)   # 카메라 광학 규약
    pts, cols, _ = _load_point_cloud(...)                  # depth backproject
    T_teach   = _parse_calibration_matrix(result["T_teach_cam_to_base"])    or fallback
    T_runtime = _parse_calibration_matrix(result["T_runtime_cam_to_base"])  or fallback
    # camera frame 탭 (zone 생략) + robot frame 탭 (anchor→runtime, zone→teach)
    _log_frame(rr, "world/cam",   ..., t_anchor=None,      t_zone=None,    draw_zone=False)
    _log_axes(rr, "world/cam/origin", "cam")
    if T_runtime is not None and T_teach is not None:
        _log_frame(rr, "world/robot", ..., t_anchor=T_runtime, t_zone=T_teach, draw_zone=True)
        _log_axes(rr, "world/robot/origin", "robot")
        _log_camera_markers(rr, "world/robot", T_teach, T_runtime)   # 카메라 위치 3축
    _send_tabbed_blueprint(rr, pts, T_runtime, have_robot)
```

### 12.6 3D correspondence plot (matplotlib — 잔차 대체)
`viz/_recon_viz.py: visualize_3d_correspondences`. pose 직후 호출. XY(top-down) 한 장,
Y축 반전(카메라 Y-down), 선 없음, 배경 점군 옅게, 제목에 inlier/rmse/fitness.

```python
# pipeline pose 직후
visualize_3d_correspondences(
    pose.viz_ref_corr, pose.viz_src_corr, corr_plot_path,
    max_points=n_corr, line_step=0,           # 전부, 선 없음
    bg_ref=tgt_pts, bg_src=src_pts,           # 배경 점군 (depth backproject)
    title=f"3D correspondences ({n} inliers, rmse={rmse:.2f}mm fitness={fit:.2f})")
# 함수 내부: yi==1(Y축)이면 ax.invert_yaxis() — 카메라 Y-down 보정
```

### 12.7 실행
```bash
uv sync --extra viz                                    # rerun-sdk 설치
uv run python -m crp_matching.utils.rerun_viz <result.json>     # → {stem}.rrd
uv run python -m crp_matching.utils.rerun_viz <dir> --glob      # 디렉터리 전부
# .rrd 는 rerun 0.33 뷰어로 열기. WSL 은 Windows rerun 으로 파일 열기.
```

### rerun viz 주의
- rerun 뷰어 폰트에 한글 글리프 없음 → 텍스트는 영문만.
- `ViewCoordinates.RDF` (카메라 광학) — 뷰 힌트일 뿐 데이터 변환 아님.
- `EyeControls3D` 는 unstable API (rerun 0.33) — 버전 바뀌면 무시될 수 있음 (그땐 뷰어 더블클릭으로 회전중심 지정).
- 점군은 cam/robot 양 프레임에 복제됨 (rerun 이 같은 데이터를 두 절대좌표로 참조 못 함) → 파일 ~2배.
