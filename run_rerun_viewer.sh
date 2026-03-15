#!/usr/bin/env bash
# Rerun 뷰어 전용 - mesh + 카메라 포즈 시각화
# 프로젝트 루트에서 실행: bash datasets/SL_ch_260315/run_rerun_viewer.sh
# (TSDF 결과 5_cp_fused_mesh.ply 등이 있어야 함)

set -e
cd "$(dirname "$0")/../.."
OUTPUT_DIR="output/SL_ch_260315"
RESULT_DIR="output/SL_ch_260315"

echo "=== Rerun 뷰어 (mesh + 카메라 포즈) ==="
# GPU 오류 시: --no_spawn --save_rrd "$OUTPUT_DIR/5_cp_fused.rrd" 추가 후, rerun 5_cp_fused.rrd 로 열기
python scripts/rerun_viewer.py \
  --mesh "$OUTPUT_DIR/5_cp_fused_mesh.ply" \
  --pcd "$OUTPUT_DIR/5_cp_fused_pcd.ply" \
  --result_dir "$RESULT_DIR" \
  --reference "5_cp"
