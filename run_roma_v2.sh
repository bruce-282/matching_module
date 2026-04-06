#!/usr/bin/env bash
# SL_ch_260315 PLY로 RoMaV2 모델 매칭만 수행 (target/source 모두 PLY → 가상 intrinsic 역투영)
# 프로젝트 루트에서 실행: bash run_roma_v2.sh

set -e
cd "$(dirname "$0")"
CONFIG="configs/SL_ch_260315/matcher_roma_v2.config.yaml"
DATA="datasets/SL_ch_260315"
OUTPUT="output/SL_ch_260315"

# echo "=== RoMaV2 모델로 매칭 ==="
# echo "=== Target: 5_cp.ply, Source: 6_cp.ply ==="
# python run_matching_only.py \
#   --config_path "$CONFIG" \
#   --target_ply "$DATA/5_cp.ply" \
#   --source_ply "$DATA/6_cp.ply"

# echo ""
# echo "=== Target: 5_cp.ply, Source: 8_cp.ply ==="
# python run_matching_only.py \
#   --config_path "$CONFIG" \
#   --target_ply "$DATA/5_cp.ply" \
#   --source_ply "$DATA/8_cp.ply"

# echo ""
# echo "=== Target: 5_cp.ply, Source: 10_cp.ply ==="
# python run_matching_only.py \
#   --config_path "$CONFIG" \
#   --target_ply "$DATA/5_cp.ply" \
#   --source_ply "$DATA/10_cp.ply"

# Target 8_cp 한 번 로드, Source만 12_cp → 5_cp 순차 로드하며 매칭 (런타임 중 한 프로세스)
# "," 로 구분하면 같은 실행 안에서 source만 바꿔가며 매칭
echo ""
echo "=== Target: 8_cp.ply, Sources: 12_cp → 5_cp (순차) ==="
python run_matching_only.py \
  --config_path "$CONFIG" \
  --target_ply "$DATA/8_cp.ply" \
  --source_ply "$DATA/12_cp.ply,$DATA/5_cp.ply"

# echo ""
# echo "결과: $OUTPUT/ 에 매칭 시각화 및 PLY, YAML 등 저장됨"

# echo ""
# echo "=== TSDF 통합 (5_cp 기준) ==="
python scripts/tsdf_integrate.py \
  --config_path "$CONFIG" \
  --ply_dir "$DATA" \
  --result_dir "$OUTPUT" \
  --reference "8_cp" \
  --output_dir "$OUTPUT" \
  --output_name "8_cp_fused" \
  --voxel_length 0.0005 \
  --sdf_trunc 0.02 \
  --depth_sampling_stride 1 \
  --volume_unit_resolution 32

# echo ""
# echo "TSDF 결과: $OUTPUT/5_cp_fused_mesh.ply, 5_cp_fused_pcd.ply"
# echo "뷰어: bash run_rerun_viewer.sh"
