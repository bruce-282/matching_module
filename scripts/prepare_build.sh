#!/usr/bin/env bash
# 로컬 / CI 공통: 서브모듈 초기화 + (선택) RoMa 가중치 다운로드 후 python -m build 등을 실행하기 전에 호출.
#
#   ./scripts/prepare_build.sh
#   DOWNLOAD_ROMA_WEIGHTS=1 ./scripts/prepare_build.sh   # 휠에 .pth 넣을 때 (용량·시간 증가)
#   SKIP_SUBMODULES=1 ./scripts/prepare_build.sh       # CI에서 checkout이 이미 recursive인 경우 생략 가능(재실행은 무해)
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${SKIP_SUBMODULES:-0}" != "1" ]] && [[ -d "${ROOT}/.git" ]]; then
  git submodule update --init --recursive
fi

if [[ "${DOWNLOAD_ROMA_WEIGHTS:-0}" == "1" ]]; then
  W="${ROOT}/third_party/RoMa/romatch/weights"
  if [[ -f "${W}/download.sh" ]]; then
    (cd "${W}" && bash ./download.sh)
  else
    echo "prepare_build: RoMa weights download.sh not found at ${W}" >&2
    exit 1
  fi
fi

echo "prepare_build: done."
