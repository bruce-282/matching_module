#!/usr/bin/env bash
#
# NX4 hood.6 매칭 실행 스크립트 (이 레포 run_matcher.py).
#
# 기본 config : configs/NX4/matcher_config.yaml        (input_dir 등 런타임 설정)
# 기본 template: configs/NX4/hood.6.matching_model.config.yml
#
# 사용법:
#   bash scripts/shell/run_nx4_hood6_match.sh            # 매칭만
#   VIZ=1 bash scripts/shell/run_nx4_hood6_match.sh      # 매칭 후 rerun .rrd 생성 (pip install -e .[viz])
#
# 환경변수로 경로 오버라이드 가능: CONFIG, TEMPLATE, PY, PYTORCH_CUDA_ALLOC_CONF
set -euo pipefail

# 레포 루트로 이동 (scripts/shell/ 기준)
cd "$(dirname "$0")/../.."

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG="${CONFIG:-configs/NX4/matcher_config.yaml}"
TEMPLATE="${TEMPLATE:-/home/cmes/cmes_ws/src/autochecker_backend/util/pose_estimation/matching_module/configs/NX4/hood.6.matching_model.config.yml}"

PY="${PY:-/home/cmes/cmes_ws/cmes/bin/python}"

echo "[run_nx4_hood6_match] python   : $PY"
echo "[run_nx4_hood6_match] cuda alloc: $PYTORCH_CUDA_ALLOC_CONF"
echo "[run_nx4_hood6_match] config   : $CONFIG"
echo "[run_nx4_hood6_match] template : $TEMPLATE"

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
  echo "[run_nx4_hood6_match] rerun viz: $OUT_DIR (*_result.json -> .rrd)"
  "$PY" -m core.utils.rerun_viz "$OUT_DIR" --glob
fi

echo "[run_nx4_hood6_match] done. outputs -> $OUT_DIR"
