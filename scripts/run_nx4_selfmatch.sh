#!/usr/bin/env bash
#
# NX4 hood.6 self-match 실행 스크립트 (이 레포 run_matcher.py).
#
#   - target depth   : datasets/NX4_selfmatch/hood6_match_depth.tif   (-> configs/NX4/hood.6.tif)
#   - target texture : datasets/NX4_selfmatch/hood6_match_texture.png (-> configs/NX4/hood.6.png)
#   - source         : datasets/NX4_selfmatch/hood6_source.tif        (-> configs/NX4/hood.6.tif)
#   - config/template: configs/NX4/matcher.selfmatch.{config,teaching.param}.yaml
#
# 사용법:
#   bash scripts/run_nx4_selfmatch.sh            # 매칭만
#   VIZ=1 bash scripts/run_nx4_selfmatch.sh      # 매칭 후 rerun .rrd 생성 (pip install -e .[viz])
#
# 환경변수로 경로 오버라이드 가능: CONFIG, TEMPLATE, PY
set -euo pipefail

# 레포 루트로 이동 (스크립트 위치 기준)
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-configs/NX4/matcher.selfmatch.config.yaml}"
TEMPLATE="${TEMPLATE:-configs/NX4/matcher.selfmatch.teaching.param.yaml}"

# 파이썬 인터프리터: .venv 우선, 없으면 python
PY="${PY:-}"
if [ -z "$PY" ]; then
  if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python"; fi
fi

echo "[run_nx4_selfmatch] python   : $PY"
echo "[run_nx4_selfmatch] config   : $CONFIG"
echo "[run_nx4_selfmatch] template : $TEMPLATE"

"$PY" run_matcher.py \
  --config_path "$CONFIG" \
  --template_param_path "$TEMPLATE"

# output_dir 은 config 에서 읽어 rerun viz 대상 디렉터리로 사용.
OUT_DIR="$("$PY" - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
print(cfg.get("output_dir", "output"))
PY
)"

if [ "${VIZ:-0}" = "1" ]; then
  echo "[run_nx4_selfmatch] rerun viz: $OUT_DIR (*_result.json -> .rrd)"
  "$PY" -m core.utils.rerun_viz "$OUT_DIR" --glob
fi

echo "[run_nx4_selfmatch] done. outputs -> $OUT_DIR"
