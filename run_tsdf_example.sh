#!/usr/bin/env bash
# TSDF 통합 예제 - 매칭 결과 YAML + PLY로 mesh/pcd 생성
# 프로젝트 루트에서 실행: bash datasets/SL_ch_260315/run_tsdf_example.sh

set -e
cd "$(dirname "$0")/../.."
CONFIG="configs/SL_ch_260315/matcher_roma_v2.config.config.yaml"
PLY_DIR="datasets/SL_ch_260315"
RESULT_DIR="output/SL_ch_260315"
OUTPUT_DIR="output/SL_ch_260315"

echo "=== TSDF 통합 (5_cp 기준, 6_cp 8_cp 10_cp 통합) ==="
python scripts/tsdf_integrate.py \
  --config_path "$CONFIG" \
  --ply_dir "$PLY_DIR" \
  --result_dir "$RESULT_DIR" \
  --reference "5_cp" \
  --output_dir "$OUTPUT_DIR" \
  --output_name "5_cp_fused" \
  --voxel_length 0.0005 \
  --sdf_trunc 0.02 \
  --depth_sampling_stride 1 \
  --volume_unit_resolution 32

echo ""
echo "결과: $OUTPUT_DIR/5_cp_fused_mesh.ply, 5_cp_fused_pcd.ply"
echo "뷰어: bash datasets/SL_ch_260315/run_rerun_viewer.sh"
