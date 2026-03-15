#!/usr/bin/env bash
# SL_ch_260315 PLY로 매칭만 수행 (target/source 모두 PLY → 가상 intrinsic 역투영)
# 프로젝트 루트에서 실행: bash datasets/SL_ch_260315/run_example.sh

set -e
cd "$(dirname "$0")/../.."
CONFIG="configs/SL_ch_260315/matcher.config.yaml"
DATA="datasets/SL_ch_260315"
OUTPUT="output/SL_ch_260315"

echo "=== Target: 5_cp.ply, Source: 6_cp.ply ==="
python run_matching_only.py \
  --config_path "$CONFIG" \
  --target_ply "$DATA/5_cp.ply" \
  --source_ply "$DATA/6_cp.ply"

echo ""
echo "=== Target: 5_cp.ply, Source: 8_cp.ply ==="
python run_matching_only.py \
  --config_path "$CONFIG" \
  --target_ply "$DATA/5_cp.ply" \
  --source_ply "$DATA/8_cp.ply"

echo ""

echo "=== Target: 5_cp.ply, Source: 10_cp.ply ==="
python run_matching_only.py \
  --config_path "$CONFIG" \
  --target_ply "$DATA/5_cp.ply" \
  --source_ply "$DATA/10_cp.ply"

echo ""
echo "=== Target: 8_cp.ply, Source: 12_cp.ply ==="
python run_matching_only.py \
  --config_path "$CONFIG" \
  --target_ply "$DATA/8_cp.ply" \
  --source_ply "$DATA/12_cp.ply"

echo ""
echo "결과: $OUTPUT/ 에 매칭 시각화 및 PLY, YAML 등 저장됨"

echo ""
echo "=== TSDF 통합 (5_cp 기준) ==="
python scripts/tsdf_integrate.py \
  --config_path "$CONFIG" \
  --ply_dir "$DATA" \
  --result_dir "$OUTPUT" \
  --reference "5_cp" \
  --output_dir "$OUTPUT" \
  --output_name "5_cp_fused" \
  --voxel_length 0.0005 \
  --sdf_trunc 0.02 \
  --depth_sampling_stride 1 \
  --volume_unit_resolution 32

echo ""
echo "TSDF 결과: $OUTPUT/5_cp_fused_mesh.ply, 5_cp_fused_pcd.ply"
