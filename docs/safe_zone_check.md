# Safe Zone Check Process

매칭으로 추정한 anchor 포인트가 **유효한 영역(safe zone)** 안에 있는지 검사하는
**안전장치(safety guard)** 입니다.

## 목적

매칭/포즈 추정이 잘못되면 anchor 결과(`result_3d`)가 엉뚱한 위치로 날아갈 수 있습니다.
이를 모르고 로봇이 그 좌표로 이동하면 이상한 곳으로 가서 충돌할 수 있습니다.
Safe zone check는 **포즈가 적용된 최종 anchor 결과가 미리 정의된 유효 영역을
벗어나면 매칭 실패로 처리**하여, 로봇이 비정상 위치로 가는 것을 막습니다.

## 동작 로직

Safe zone은 각 anchor 포인트(`L`, `R`)마다 정의된 **방향이 있는 큐보이드
(OBB, Oriented Bounding Box)** 입니다.

- `min`, `max`: 큐보이드의 두 대각 꼭짓점 `[x, y, z]`
- `euler`: 큐보이드 **중심점 기준 회전** `[rx, ry, rz]` (radian)

판정은 다음과 같이 계산합니다.

```
center  = (min + max) / 2          # 큐보이드 중심
half    = |max - min| / 2          # 각 축 반쪽 크기
R       = euler_to_rotation_matrix(rx, ry, rz)
p_local = Rᵀ · (point - center)    # 포인트를 큐보이드 로컬 좌표계로 변환
safe    = (|p_local.x| ≤ half.x) AND (|p_local.y| ≤ half.y) AND (|p_local.z| ≤ half.z)
```

### 핵심 포인트

- **매칭 transform을 적용하지 않습니다.** Safe zone은 기준 템플릿/월드 좌표
  (고정 카메라 공간)에 고정된 "유효 영역"이고, 포즈가 이미 적용된 `result_3d`를
  **있는 그대로** 그 영역과 비교합니다.
- **검사 대상은 `L`, `R` 두 포인트뿐입니다.** `U` 포인트는 safe zone이 없어 검사하지
  않습니다.
- **하위 호환**: `template_param`에 `safe_zones`가 없으면 검사를 건너뜁니다.

### Euler 회전 규약

`euler`는 **Three.js 의 기본 회전 순서 `'XYZ'`** (프론트엔드에서 safe zone을
생성하는 규약) 와 동일하게 해석합니다.

```
R = Rx · Ry · Rz        (intrinsic XYZ  =  extrinsic ZYX  =  scipy Rotation.from_euler('XYZ'))
```

> 주의: extrinsic XYZ(`Rz · Ry · Rx`)와는 다른 행렬입니다. 구현은 Three.js
> `Matrix4.makeRotationFromEuler('XYZ')` 및 scipy intrinsic `'XYZ'`와 임의 각도에서
> 행렬이 정확히 일치함을 확인했습니다.

## 설정 (template param)

`matcher.teaching.param.yaml` 의 `safe_zones` 섹션에 정의합니다. (최상위 또는
`matching_model` 하위 모두 인식)

```yaml
safe_zones:
  L:
    min: [34.7676, 105.0314, 1552.3996]
    max: [181.7779, -9.7333, 1502.2293]
    euler: [2.7066, 0.5931, -0.1091]
  R:
    min: [514.7950, 295.5452, 1820.6679]
    max: [661.8052, 180.7805, 1770.4976]
    euler: [2.7066, 0.5931, -0.1091]
```

## 결과

검사는 파이프라인에서 3D anchor 결과가 확정된 직후 수행됩니다.

| 상황 | 동작 |
|------|------|
| 모든 검사 포인트가 zone **안** | 통과 → 파이프라인 계속 진행, `Safe zone check passed for point L/R.` (DEBUG) |
| 어느 한 포인트라도 zone **밖** | `Exception` 발생 → depth 계산 실패와 동일하게 **매칭 실패 처리** |
| `safe_zones` 미설정 / transform 없음 | 검사 skip (DEBUG 로그) |

zone을 벗어나면 `run_matcher.py`에서 다음과 같이 실패로 기록됩니다.

```
ERROR  {base_name} - Safe zone check failed: point L [x y z] is outside its safe zone
ERROR  ❌ Matching failed - {base_name}
```

## 구현 위치

- `core/utils/geometry_utils.py`
  - `euler_to_rotation_matrix(rx, ry, rz)` — Three.js XYZ(intrinsic) 회전행렬
  - `is_point_in_safe_zone(point, min, max, euler)` — OBB 내부 판정
- `core/matchers/matcher.py`
  - `Matcher.check_safe_zones(result1_3d, result2_3d)` — L/R 검사, 실패 시 raise
  - `run_pipeline` 내 anchor 3D 확정 직후 호출
