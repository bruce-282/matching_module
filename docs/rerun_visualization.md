# Rerun 디버그 시각화 (Safe Zone / Anchor)

매칭 결과(anchor, safe zone, pose)를 **rerun** 으로 3D 시각화하는 디버깅 도구의 전체
프로세스 문서입니다. 파이프라인에는 끼어들지 않고, 매칭이 남긴 **결과 파일만** 가지고
사용자가 따로 실행합니다 (운영엔 불필요한 optional 기능).

구현: [`core/utils/rerun_viz.py`](../core/utils/rerun_viz.py)

---

## 1. 목적

- 매칭으로 추정한 anchor(L/R/U)와 safe zone(OBB)을 **3D 로 눈으로** 확인 — pose 가
  제대로 맞았는지, anchor 가 zone 안에 들어오는지.
- open3d EGL 오프스크린 렌더가 WSL 등에서 segfault 나는 문제를 우회하고, 인터랙티브
  3D 뷰 + `.rrd` 파일로 보존/공유.

---

## 2. 설치

rerun 은 optional extra(`viz`) 로 분리되어 있습니다 (운영 설치엔 불필요).

```bash
pip install -e .[viz]        # rerun-sdk
# uv 사용 시:
uv sync --extra viz
```

rerun 미설치 상태에서 실행하면 친절한 안내와 함께 종료하므로, 본체 동작엔 영향이 없습니다.

---

## 3. 입력 — `{stem}_result.json`

매칭 파이프라인(`run_pipeline`)은 `save_essential != "none"` 일 때 캡처마다
`{stem}_result.json` 을 저장합니다. rerun 시각화는 이 파일을 입력으로 받습니다.
(safe zone 위반으로 실패한 경우에도 위반 정보를 담아 저장됩니다.)

| 키 | 내용 |
|----|------|
| `frame_stem` | 프레임 이름 |
| `ok` / `failed_at` / `error_detail` | 성공 여부 / 실패 단계 / 상세 |
| `anchors` | `{L, R, U: [x,y,z]}` — **매칭(현재) 카메라 프레임** anchor 3D |
| `safe_zones` | `{L, R: {min, max, euler}}` — **템플릿(teaching) 카메라 프레임** OBB |
| `safe_zone_violation` | 위반 시 `{point, position}` (없으면 null) |
| `camera_target_K` | 타겟 카메라 intrinsic 3x3 (점군 backproject 용) |
| `T_teach_cam_to_base` | teaching 카메라 → robot base (4x4) |
| `T_runtime_cam_to_base` | 현재(매칭) 카메라 → robot base (4x4) |
| `inputs.depth` / `inputs.texture` | 배경 점군 backproject 용 입력 경로 |

> 점군은 별도 PLY 가 아니라 `inputs.depth`(depth tif)를 `camera_target_K` 로
> backproject 해서 만들고, `inputs.texture` 가 있으면 색을 입힙니다.

---

## 4. 실행

### 4.1 직접 실행
```bash
# result.json 하나
python -m core.utils.rerun_viz path/to/<stem>_result.json

# 디렉터리 안 *_result.json 전부
python -m core.utils.rerun_viz path/to/output_dir --glob

# 뷰어 즉시 띄움 (GUI 되는 환경)
python -m core.utils.rerun_viz <stem>_result.json --spawn
```
기본은 `{stem}.rrd` 저장 (`--spawn` 이면 저장 대신 뷰어 실행).

### 4.2 매칭과 함께 (`VIZ=1`)
매칭 실행 스크립트에서 `VIZ=1` 을 주면 매칭 후 자동으로 `.rrd` 를 생성합니다.
```bash
VIZ=1 bash scripts/shell/run_match.sh
```

### 4.3 뷰어로 열기
```bash
# .rrd 를 rerun web viewer 로 (기존 rerun 프로세스 정리 후)
bash scripts/shell/view_rerun.sh <stem>.rrd
# 인자 없으면 output/ 의 가장 최근 .rrd 를 연다
```
WSL 등 GUI 가 까다로우면 Windows 쪽 rerun 으로 `.rrd` 파일을 열어도 됩니다.

---

## 5. 두 프레임 탭

`.rrd` 는 **두 개의 3D 뷰 탭**으로 구성됩니다. 좌표 출처가 다른 데이터를 각각
올바른 변환으로 모읍니다.

| 탭 | 내용 | extrinsic |
|----|------|-----------|
| **camera frame** (`world/cam`) | 매칭(현재) 카메라 기준. anchor·점군만 (원본 좌표). | 미적용 |
| **robot frame** (`world/robot`) | anchor·점군 → `T_runtime`, safe_zone → `T_teach` 로 **로봇 베이스 좌표계**에 모아 비교. | 적용 |

- `camera frame` 에는 safe zone 을 **안 그립니다** — zone 은 teaching 카메라 좌표라
  현재 카메라 프레임에 섞으면 의미가 없기 때문(`draw_zone=False`).
- `robot frame` 은 두 캘리브레이션(`T_teach`, `T_runtime`)이 **모두 있을 때만**
  활성화됩니다. 없으면 camera frame 탭만 표시.
- 기본 활성 탭은 robot frame.

자세한 좌표 변환 근거(왜 anchor 는 runtime, zone 은 teaching 으로 변환하는지)는
[safe_zone_check.md](safe_zone_check.md) 참고.

---

## 6. 그려지는 요소

| 요소 | 표현 | 색 / 크기 |
|------|------|-----------|
| 배경 점군 | `inputs.depth` backproject (+texture 색) | radius 0.6, 최대 40만 점 |
| anchor L/R/U | `Points3D` | L=파랑·R=주황·U=보라, 위반 anchor=빨강, radius 14 |
| safe zone OBB | `Transform3D(mat3x3)` + 축정렬 `Boxes3D` (majorwireframe) | 통과=초록 / 위반=빨강 |
| 위반 화살표 | OBB 표면 최근접점 → 위반 anchor (`Arrows3D`) | 빨강, 거리(mm) 라벨 |
| 카메라 마커 | teaching / runtime 카메라 위치 + 3축 (robot frame) | teach=청록 / runtime=주황 |
| 좌표축 | 원점·카메라 마커 축 / safe zone 로컬 축 | `_AXIS_LEN`(300mm) / `_ZONE_AXIS_LEN`(100mm) |

> 축 길이는 [`rerun_viz.py`](../core/utils/rerun_viz.py) 상단 `_AXIS_LEN`(씬 스케일,
> 로봇 베이스 ~m) / `_ZONE_AXIS_LEN`(OBB 박스 비례) 상수로 조정합니다.

---

## 7. 나중에 재생성

`.rrd` 는 매칭을 다시 돌릴 필요 없이 **`{stem}_result.json` + `inputs` 가 가리키는
depth/texture 파일**만 있으면 4단계 명령으로 언제든 다시 만듭니다. 그래서 `.rrd` 자체는
보관/커밋하지 않아도 됩니다.

- `inputs` 경로가 깨지면(파일 이동·다른 머신) **배경 점군만** 빠지고, anchor·safe
  zone·위반 화살표는 그대로 그려집니다.

---

## 8. 주의 사항 / 트러블슈팅

- **rerun 버전 / numpy 호환**: rerun 0.23 + numpy < 2.0 조합에서 `Boxes3D` 의 회전
  배치(quaternion/axis-angle) 직렬화가 깨집니다(`asarray(copy=)` — numpy 2.0 전용).
  그래서 OBB 회전은 `Boxes3D` 의 quaternion 대신 **`Transform3D(mat3x3)` + 축정렬
  박스**로 그립니다. (회전 정상 적용 + 경고 없음.)
- **초기 카메라 시점**: `EyeControls3D` 는 rerun 의 unstable API(>=0.24 부근 추가)라,
  없는 버전(예: 0.23)에서는 기본 `Spatial3DView` 로 graceful fallback 합니다(뷰어에서
  더블클릭으로 회전중심 지정).
- **한글 글리프**: rerun 뷰어 폰트에 한글이 없어 텍스트 라벨은 영문으로만 표기합니다.
- **좌표계 힌트**: `ViewCoordinates.RDF`(카메라 광학: X-Right, Y-Down, Z-Forward) —
  뷰 힌트일 뿐 데이터 값 변환이 아닙니다.
- **`.rrd` 크기**: 점군이 cam/robot 두 프레임에 복제되어 파일이 ~2배가 됩니다(rerun 이
  같은 데이터를 두 절대좌표로 참조하지 못하므로). 점군은 40만 점으로 다운샘플됩니다.
- **점군이 안 보임**: `save_essential` 이 `none` 이면 result.json 이 안 생깁니다.
  또한 `inputs.depth` 경로가 유효해야 점군이 backproject 됩니다.
