#!/usr/bin/env python3
"""
이미지에 포인트를 그리는 스크립트
"""

import sys
import argparse
import yaml
import cv2
import numpy as np
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.utils.image_utils import read_image

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="이미지에 포인트 그리기")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="설정 파일 경로 (YAML)",
    )

    args = parser.parse_args()
    
    # 설정 파일 로드 (YAML)
    try:
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"오류: 설정 파일을 찾을 수 없습니다: {args.config_path}")
        return
    except yaml.YAMLError as e:
        print(f"오류: YAML 파일 파싱 실패: {e}")
        return

    # 필요한 설정값들 추출
    source_image_path = config.get("source_image_path")
    pointL_pos = config.get("pointL_pos", {"x_ratio": 0.5, "y_ratio": 0.89})
    pointR_pos = config.get("pointR_pos", {"x_ratio": 1.2, "y_ratio": 0.88})
    pointU_pos = config.get("pointU_pos", {"x_ratio": 0.8, "y_ratio": 0.4})
    point_radius = config.get("point_radius", 25)
    output_dir = config.get("output_dir", "output")
    
    if not source_image_path:
        print("오류: source_image_path가 설정 파일에 없습니다.")
        return

    # 이미지 로드
    try:
        img = read_image(source_image_path)
        if img is None:
            print(f"오류: 이미지를 로드할 수 없습니다: {source_image_path}")
            return
    except Exception as e:
        print(f"오류: 이미지 로드 실패: {e}")
        return

    # 이미지 크기
    h, w = img.shape[:2]
    
    # 포인트 좌표 계산 (상대 좌표를 절대 좌표로 변환)
    points = [
        (int(pointL_pos["x_ratio"] * w), int(pointL_pos["y_ratio"] * h)),
        (int(pointR_pos["x_ratio"] * w), int(pointR_pos["y_ratio"] * h)), 
        (int(pointU_pos["x_ratio"] * w), int(pointU_pos["y_ratio"] * h))
    ]
    
    # 원이 이미지 범위를 벗어날 경우를 위한 패딩 계산
    min_x = min(point[0] for point in points) - point_radius
    max_x = max(point[0] for point in points) + point_radius
    min_y = min(point[1] for point in points) - point_radius
    max_y = max(point[1] for point in points) + point_radius
    
    # 패딩이 필요한지 확인 (정수로 변환)
    pad_left = int(max(0, -min_x))
    pad_right = int(max(0, max_x - w))
    pad_top = int(max(0, -min_y))
    pad_bottom = int(max(0, max_y - h))
    
    # 패딩이 필요한 경우 이미지에 패딩 추가
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        img_padded = cv2.copyMakeBorder(
            img, 
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, 
            value=(0, 0, 0)  # 검은색 패딩
        )
        # 패딩된 좌표로 포인트 위치 조정
        adjusted_points = [
            (point[0] + pad_left, point[1] + pad_top) for point in points
        ]
    else:
        img_padded = img.copy()
        adjusted_points = points
    
    # 원 그리기
    cv2.circle(img_padded, adjusted_points[0], point_radius, (255, 0, 0), -1)  # 파란색
    cv2.circle(img_padded, adjusted_points[1], point_radius, (0, 255, 0), -1)  # 초록색
    cv2.circle(img_padded, adjusted_points[2], point_radius, (0, 0, 255), -1)  # 빨간색
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 파일명 생성
    source_name = Path(source_image_path).stem
    output_file = output_path / f"{source_name}_with_points.png"
    
    # 이미지 저장
    try:
        cv2.imwrite(str(output_file), img_padded)
        print(f"포인트가 그려진 이미지 저장: {output_file}")
        print(f"포인트 위치:")
        print(f"  L (파란색): {adjusted_points[0]}")
        print(f"  R (초록색): {adjusted_points[1]}")
        print(f"  U (빨간색): {adjusted_points[2]}")
    except Exception as e:
        print(f"오류: 이미지 저장 실패: {e}")

if __name__ == "__main__":
    main()
