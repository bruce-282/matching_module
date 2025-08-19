#!/bin/bash

# 기존 PyTorch 제거 (선택사항)
# pip uninstall torch torchvision torchaudio -y

# CUDA 12.8 버전 PyTorch 설치
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

python -c "import torch; print(f'PyTorch 버전: {torch.__version__}'); print(f'CUDA 사용 가능: {torch.cuda.is_available()}'); print(f'CUDA 버전: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
ㅔㅛ쇄쇄