#!/usr/bin/env bash
# 로컬 또는 CI 스크립트에서: 서브모듈이 없으면 아무 것도 하지 않고 성공으로 끝남.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
git submodule update --init --recursive
