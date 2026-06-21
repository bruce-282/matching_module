#!/usr/bin/env bash
#
# .rrd 파일을 Rerun web viewer 로 연다 (기존 rerun 프로세스 정리 후).
#
# 사용법:
#   bash scripts/shell/view_rerun.sh
#   bash scripts/shell/view_rerun.sh output/20260619_003308_match_texture.rrd
#   RRD=output/foo.rrd bash scripts/shell/view_rerun.sh
#
# 인자/RRD 미지정 시 output/ 아래 가장 최근 .rrd 를 연다.
set -euo pipefail

cd "$(dirname "$0")/../.."

RRD="${1:-${RRD:-}}"

if [ -z "$RRD" ]; then
  shopt -s nullglob
  candidates=(output/*.rrd)
  if [ ${#candidates[@]} -eq 0 ]; then
    echo "[view_rerun] no .rrd under output/ — run VIZ=1 bash scripts/shell/run_nx4_hood6_match.sh first" >&2
    exit 1
  fi
  RRD="$(ls -t "${candidates[@]}" | head -1)"
fi

if [ ! -f "$RRD" ]; then
  echo "[view_rerun] file not found: $RRD" >&2
  exit 1
fi

echo "[view_rerun] stop stale rerun processes..."
# -x: rerun 바이너리만 (view_rerun.sh 는 -f rerun 에 걸리지 않게)
pkill -x rerun 2>/dev/null || true
sleep 1

echo "[view_rerun] open web viewer: $RRD"
exec rerun "$RRD" --web-viewer
