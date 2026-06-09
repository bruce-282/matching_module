@echo off
REM nike_jackie RGB 쌍 매칭 (RoMaV2, 2D만)
cd /d "%~dp0"

python run_nike_jackie.py ^
  --config_path configs/nike_jackie/matcher.config.yaml ^
  --param_path configs/nike_jackie/nike_jackie.param.yaml

echo 결과: output/nike_jackie/
