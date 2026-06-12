# Error Codes

매칭 모듈의 실패는 사내 표준 `ErrorCode` (`core/matchers/error_handler.py`) 로 구분된다.
reconstruction_module 의 범주 패턴(0/100/200/300/400/500)을 그대로 따른다.

## 숫자 코드 인코딩: `MMMSEEE`

| 자리 | 의미 | 현재 값 |
|------|------|------|
| `MMM` | 사내 모듈 번호 (3자리) | `000` (`error_handler.MODULE_NAME`) |
| `S` | severity (1자리) | raise 시점 기본 `3` (`results.DEFAULT_SEVERITY`) |
| `EEE` | `ErrorCode.num` | 아래 표 참고 |

예: `MODULE_NAME="000"`, severity `3`, `SAFE_ZONE_VIOLATION`(504) → **`30504`**.

```python
ErrorCode.SAFE_ZONE_VIOLATION.code_str("3")   # -> 30504
```

> 인코딩은 `code = mod*100000 + severity*10000 + num` 이다(`ErrorCode.code_str`).
> `MODULE_NAME` 만 바꾸면 prefix 가 전부 갱신된다 — 예: `"009"` 이면
> `SAFE_ZONE_VIOLATION` → `900000 + 30000 + 504 = 930504`.

## 전체 코드 표

`코드(예)` 열은 `MODULE_NAME="000"`, severity `3` 기준 `MMMSEEE` 값이다.

### General (0–99)
| `ErrorCode` | num | 기본 메시지 | 코드(예) |
|------|----|------|------|
| `UNKNOWN` | 0 | Unknown cause of error | `30000` |
| `NOT_INITIALIZED` | 1 | Module not initialized | `30001` |
| `INVALID_PARAM` | 5 | Invalid parameter | `30005` |
| `NOT_IMPLEMENTED` | 8 | Not implemented | `30008` |

### System (100–199)
| `ErrorCode` | num | 기본 메시지 | 코드(예) |
|------|----|------|------|
| `SYSTEM_RESET_FAILED` | 101 | Module reset failed | `30101` |
| `ENGINE_INIT_FAILED` | 102 | Matcher engine initialization failed | `30102` |

### File I/O (200–299)
| `ErrorCode` | num | 기본 메시지 | 코드(예) |
|------|----|------|------|
| `FILE_NOT_FOUND` | 201 | File not found | `30201` |
| `FILE_LOAD_FAILED` | 202 | Failed to load file | `30202` |
| `SAVE_FAILED` | 203 | Failed to save output | `30203` |

### Template (300–399)
| `ErrorCode` | num | 기본 메시지 | 코드(예) |
|------|----|------|------|
| `TEMPLATE_NOT_LOADED` | 301 | Template not loaded — call req_load_config first | `30301` |
| `TEMPLATE_INVALID` | 302 | Template parameters invalid | `30302` |

### Matching (400–499)
| `ErrorCode` | num | 기본 메시지 | 코드(예) |
|------|----|------|------|
| `MATCH_FAILED` | 401 | Feature matching failed | `30401` |
| `INSUFFICIENT_MATCHES` | 402 | Not enough matches found | `30402` |
| `HOMOGRAPHY_FAILED` | 403 | Homography estimation failed | `30403` |

### Pose / Depth (500–599)
| `ErrorCode` | num | 기본 메시지 | 코드(예) |
|------|----|------|------|
| `POSE_ESTIMATION_FAILED` | 501 | Object pose estimation failed | `30501` |
| `DEPTH_CALCULATION_FAILED` | 502 | Anchor depth calculation failed | `30502` |
| `DEPTH_OUT_OF_RANGE` | 503 | Anchor depth outside stable range | `30503` |
| `SAFE_ZONE_VIOLATION` | 504 | Anchor outside safe zone | `30504` |

## `run_pipeline` 이 실제로 반환하는 코드

`Matcher.run_pipeline` 은 파이프라인 처리 중의 실패를 예외 대신 `MatchResult` 로 반환한다
(아래 참고. 초기 인자/config 검증 단계의 비정상 입력은 예외가 날 수 있음).
**현재 파이프라인이 실패 시 사용하는 코드**는 다음 4개다.

| `error_code` | 발생 조건 | `details` |
|------|------|------|
| `SAFE_ZONE_VIOLATION` | anchor L/R 이 safe zone 을 벗어남 | `{"point", "position"}` |
| `DEPTH_CALCULATION_FAILED` | anchor depth 계산 결과 없음(None) | `{"depths"}` (해당 시) |
| `DEPTH_OUT_OF_RANGE` | depth 가 안정 범위(`stable_depth_range`)를 초과 | `{"point", "depth_diff", "stable_range"}` |
| `INVALID_PARAM` | 필수 config/파라미터 키 누락 (예: `target_depth_path`, `enable_3d_matching`) | — |
| `MATCH_FAILED` | 그 외 매칭 실패 (2D/3D 매칭·필터링, 예기치 못한 오류 등) | — |

> 설정 키 누락은 `KeyError` 를 잡아 `INVALID_PARAM` 으로 분류하며, 누락된 키 이름을
> `error_message` 에 담는다(예: `missing required config/parameter key: 'enable_3d_matching'`).

> 나머지 코드(`NOT_INITIALIZED`, `FILE_NOT_FOUND`, `ENGINE_INIT_FAILED`,
> `TEMPLATE_*` 등)는 사내 표준 스킴과 요청 단위(req_init / req_load_config / req_reset)
> 핸들러·`MatchingErrorDefinitions`(crp_core 연동)를 위해 **정의되어 있으나**, 현재
> `run_pipeline` 자체에서는 사용하지 않는다.

## 결과 / 예외 표현

- `core/matchers/results.py` `MatchResult` — `run_pipeline` 의 공개 반환 타입.
  실패 시 `success=False`, `error_code`(`ErrorCode`), `error_message`, `severity`,
  `details` 가 채워진다. `res.code`(숫자 `MMMSEEE`), `res.to_error()`(→ `MatchingError`)
  브리지 제공.
- `core/matchers/error_handler.py` `MatchingError` — 사내 표준 예외(`error_code` +
  `severity` + 숫자 `code`). crp_core 응답/로깅용.
- `core/matchers/errors.py` `MatcherError` — 파이프라인 **내부** 신호용 예외
  (`ErrorCode` + `details`). `run_pipeline` 경계에서 `MatchResult` 로 변환되며 외부로
  새지 않는다.

### 호출 측 처리 예시 (in-process 백엔드)

```python
from core.matchers.error_handler import ErrorCode

res = matcher.run_pipeline(...)
if not res.success:
    if res.error_code == ErrorCode.SAFE_ZONE_VIOLATION:
        bad, pos = res.details["point"], res.details["position"]
    # 사내 표준 예외/숫자코드가 필요하면:
    #   raise res.to_error()        # MatchingError
    #   log(res.code)               # 30504
    return
use(res.point_l, res.point_r, res.point_u, res.plane_normal)
```

## 새 코드 추가

1. `error_handler.py` `ErrorCode` 의 적절한 범주에 `NAME = (num, "메시지")` 추가.
2. 발생 지점에서 `raise MatcherError(ErrorCode.NAME, msg, details=...)` (내부) 하면
   `run_pipeline` 경계에서 자동으로 실패 `MatchResult` 로 변환된다.
3. 이 문서 표에 한 줄 추가.

## 관련 문서

- [Safe Zone Check Process](safe_zone_check.md) — safe zone 검사 동작/좌표 프레임
