# Changelog

`matching_module` (HMMA AutoChecker hood 매칭) 의 릴리즈 이력입니다.

- 태그는 접두사 없이 `MAJOR.MINOR.PATCH` 입니다 (예: `0.1.1`). 이전 릴리즈 후보는 `hmma-` 접두사를
  썼습니다 (`hmma-0.1.0-rc5` 등).
- 릴리즈는 `master` 병합 커밋에 태그합니다.

---

## [0.1.1] — 2026-10-28

`hmma-0.1.0-rc5` 에 **크래시·무응답 진단 로그**를 더한 버전입니다. 매칭 결과·`run_pipeline` 인자와 반환값
(`MatchResult`)·에러코드는 rc5 와 같습니다. tail 매칭(`autochecker_matching_module` 1.0.1)과 같은 기능입니다.

### Added

- **로그 파일.** `Matcher` 로그가 콘솔과 함께 날짜별 파일
  `~/.cmes/CRP_Project/logs/YYYY-MM/hood_matching_YYYYMMDD.log` 에도 남습니다. 라이브러리로 호출할 때
  호출 프로세스가 stdout 을 모으지 않으면 로그가 사라지던 문제를 막습니다. 경로는 `MATCHING_LOG_DIR` 로 바꿉니다.
- **예외 시 VRAM/RAM 상태 기록.** `run_matching` / `run_pipeline` / warmup 에서 예외가 나면 torch(alloc·reserved·peak),
  디바이스 VRAM, 프로세스 RSS, 시스템 RAM 을 한 줄로 남깁니다. CUDA OOM 이면 GPU 를 쓰는 프로세스 목록도 남기고
  torch 캐시를 반환합니다. `*` 가 붙은 pid 가 매칭을 돌린 프로세스입니다. 매칭 forward 의 OOM 은 `run_matching` 에서
  잡혀 결과가 "2D matching failed" 로만 보이므로, 그 자리에서 기록합니다.

  ```
  ERROR  matcher.py:230: [RES] CUDA OOM | torch alloc 1.35G rsv 1.44G peak 1.36G (rsv 1.44G) | gpu0 31.29G/31.84G util 0% 42C | rss 2.76G (peak 4.50G) | ram avail 50.46G/62.18G swap 0.00G
  ERROR  matcher.py:232: [RES] gpu procs: 2173112:python=22.32G, 2169948:python=6.39G, 2172879*:python=2.04G, 2006:gnome-remote-desktop-daemon=0.49G
  ```
- **무응답(hang) 감시.** `Matcher` 초기화(모델 로드·warmup·init_config)와 `run_pipeline` 이
  `MATCHING_HANG_TIMEOUT_SEC`(기본 60초) 안에 끝나지 않으면, 멈춘 위치의 스택과 VRAM 상태를 남깁니다.
  타임아웃 5초 뒤에는 전체 스레드 스택도 한 번 더 남깁니다. 정상적으로 끝나면 아무것도 남기지 않습니다.

  ```
  ERROR  matcher.py:239: run_pipeline 응답 없음 (60s 경과) — 처리 스레드 스택:
    File ".../core/matchers/matcher.py", line 1506, in run_pipeline
      matches = self.run_matching(target_clipped, source_image)
  ERROR  matcher.py:240: [RES] run_pipeline hang | torch alloc ... | gpu0 ... | rss ... | ram avail ...
  ```
- **segfault / abort 시 스택 덤프.** 프로세스가 죽는 순간 전체 스레드 스택을 같은 로그 파일에 직접 씁니다
  (`Fatal Python error: Segmentation fault`). 이 줄에는 타임스탬프가 붙지 않으니 바로 위 줄의 시각을 보세요.

### 호환성

- `run_pipeline` 인자·반환값(`MatchResult`)·에러코드, config/template 스키마, 매칭 결과값: 변경 없음.
- 의존성 추가·변경 없음 (표준 라이브러리 + 기존 torch, `nvidia-smi`). 코드만 교체하면 됩니다.
- 패키지 버전(`pyproject.toml`) `0.1.0` → `0.1.1`.
- faulthandler 는 프로세스 전역 설정입니다. 라이브러리로 호출하면 호출 프로세스의 크래시 덤프도 이 로그 파일로 갑니다.

### 설정

- `MATCHING_LOG_DIR` (환경변수, 선택): 로그 디렉터리. 기본 `~/.cmes/CRP_Project/logs`.
- `MATCHING_HANG_TIMEOUT_SEC` (환경변수, 선택): 무응답 판정 시간(초). 기본 `60`, `0` 이면 끔.

### 장애 시 확인 방법

로그 위치: `~/.cmes/CRP_Project/logs/YYYY-MM/hood_matching_YYYYMMDD.log`

```bash
grep -F "[RES]" <log>              # 예외 시점 VRAM/RAM, OOM 시 GPU 점유 프로세스
grep "응답 없음" <log>              # 무응답(hang) — 멈춘 위치 스택
grep "Fatal Python error" <log>    # 프로세스 크래시 (segfault/abort)
# 위 셋 다 없는데 프로세스가 죽었다면 OS 강제 종료(OOM-kill)·GPU 드라이버 오류 확인.
# 재부팅했어도 보이도록 부팅과 상관없이 날짜로 조회한다 (journalctl -k 는 현재 부팅만 봄).
sudo journalctl _TRANSPORT=kernel --since "YYYY-MM-DD" | grep -iE "out of memory|oom-kill|killed process|NVRM: Xid"
```

### 알려진 제한

- OS OOM-kill(SIGKILL)은 프로세스 안에서 기록할 수 없습니다 — 위 `journalctl` 로 확인합니다.
- 자정 이후 `run_pipeline` 호출 없이 죽으면 segfault 덤프는 전날 로그 파일에 남습니다.
- CUDA OOM 은 `run_matching` 과 `run_pipeline` 에서 한 번씩, `[RES]` 가 두 줄 남습니다.
- 현장 PC 에 `/var/log/journal` 이 없으면 커널 로그(journal)가 메모리에만 있어 재부팅하면 사라집니다.

### 검증

NX5a HEV hood.2 템플릿 + `NX5_cycle_data/144530_NHD_574151` 캡처로 라이브러리 호출(`Matcher` + `run_pipeline`)을 돌렸습니다.

- 정상: success, 로그 파일에 기존 Matcher 로그 전체 기록, 진단 로그 추가 없음.
- 실제 유발: 다른 프로세스가 VRAM 22.3G 점유 → CUDA OOM 기록 + `MATCH_FAILED` 정상 반환 /
  매칭 정지 → 응답 없음 + 스택 기록 / SIGSEGV → 크래시 스택 기록.
