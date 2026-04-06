@echo off
REM RoMaV2 모델로 매칭 실행
REM 사용법:
REM   run_roma_v2.bat --template_param_path configs/xxx/template_param.yaml
REM   run_roma_v2.bat --source_ply a.ply --target_ply b.ply

set CONFIG=configs/Default/matcher_roma_v2.config.yaml

python run_matching_only.py --config_path %CONFIG% %*
