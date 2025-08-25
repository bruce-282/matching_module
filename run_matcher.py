#!/usr/bin/env python3
"""
통합 매처 스크립트 - Roma 매칭 + RANSAC 필터링
"""

import sys
from pathlib import Path
import argparse
import warnings
import logging

# torchvision 경고 숨기기
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.matchers.matcher import Matcher


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Roma 매칭 + RANSAC 필터링")
    parser.add_argument(
        "--source",
        type=str,
        default="datasets/source.png",
        help="첫 번째 이미지 경로",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="datasets/target.png",
        help="두 번째 이미지 경로",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="결과 저장 디렉토리",
    )
    parser.add_argument(
        "--max_keypoints",
        type=int,
        default=2000,
        help="최대 키포인트 수",
    )
    parser.add_argument(
        "--ransac_method",
        type=str,
        default="CV2_USAC_MAGSAC",
        help="RANSAC 메서드",
    )
    parser.add_argument(
        "--ransac_reproj_threshold",
        type=float,
        default=10.0,
        help="RANSAC 재투영 임계값",
    )
    parser.add_argument(
        "--ransac_confidence",
        type=float,
        default=0.9999,
        help="RANSAC 신뢰도",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드 활성화 (파일 저장)",
    )
    parser.add_argument(
        "--offset_point1_x",
        type=float,
        default=0.5,
        help="첫 번째 포인트 X 좌표 비율 (0.0 ~ 1.0)",
    )
    parser.add_argument(
        "--offset_point1_y",
        type=float,
        default=0.925,
        help="첫 번째 포인트 Y 좌표 비율 (0.0 ~ 1.0)",
    )
    parser.add_argument(
        "--offset_point2_x",
        type=float,
        default=1.4,
        help="두 번째 포인트 X 좌표 비율 (0.0 ~ 1.0)",
    )
    parser.add_argument(
        "--offset_point2_y",
        type=float,
        default=0.937,
        help="두 번째 포인트 Y 좌표 비율 (0.0 ~ 1.0)",
    )
    parser.add_argument(
        "--point_radius",
        type=int,
        default=10,
        help="포인트 반지름",
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=2100.0,
        help="Depth map 최대 값 (기본값: 2000.0)",
    )
    parser.add_argument(
        "--depth_unit",
        type=str,
        default="mm",
        choices=["m", "mm"],
        help="Depth map 단위 (기본값: mm)",
    )

    args = parser.parse_args()

    # 로그 레벨 설정
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Matcher 설정
    config = {
        "target_image_path": args.target,
        "source_image_path": args.source,
        "output_dir": args.output_dir,
        "max_keypoints": args.max_keypoints,
        "ransac_method": args.ransac_method,
        "ransac_reproj_threshold": args.ransac_reproj_threshold,
        "ransac_confidence": args.ransac_confidence,
        "debug_mode": args.debug,
        "offset_point1": (args.offset_point1_x, args.offset_point1_y),
        "offset_point2": (args.offset_point2_x, args.offset_point2_y),
        "point_radius": args.point_radius,
        "depth_max": args.depth_max,
    }

    # Matcher 인스턴스 생성
    matcher = Matcher(config)

    # 파이프라인 실행
    matches_result, ransac_result = matcher.run_pipeline(
        target_image_path=args.target,
        source_image_path=args.source,
        output_dir=args.output_dir,
    )

    if matches_result is not None:
        print("\n=== 실행 완료 ===")
        print(f"매칭 결과: {len(matches_result['keypoints0'])} 개 키포인트")
        if ransac_result is not None:
            print(
                f"RANSAC 필터링 결과: {len(ransac_result['filtered_kpts0'])} 개 키포인트"
            )
        else:
            print("RANSAC 필터링 실패")
    else:
        print("실행 실패")


if __name__ == "__main__":
    main()
