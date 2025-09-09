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
        "--target_texture",
        type=str,
        default="",
        help="Target texture 이미지 경로 (매칭용)",
    )
    parser.add_argument(
        "--target_depth",
        type=str,
        default="datasets/target_depth.tif",
        help="Target depth 이미지 경로 (depth 계산용)",
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
        default=25.0,
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
        "--offset_pointL_x",
        type=float,
        default=0.5,
        help="왼쪽 포인트 X 좌표 비율 (0.0 ~ 1.0)",
    )
    parser.add_argument(
        "--offset_pointL_y",
        type=float,
        default=0.89,
        help="왼쪽 포인트 Y 좌표 비율 (0.0 ~ 1.0)",
    )
    parser.add_argument(
        "--offset_pointR_x",
        type=float,
        default=1.38,
        help="오른쪽 포인트 X 좌표 비율 (0.0 ~ 1.0)",
    )
    parser.add_argument(
        "--offset_pointR_y",
        type=float,
        default=0.89,
        help="오른쪽 포인트 Y 좌표 비율 (0.0 ~ 1.0)",
    )
    parser.add_argument(
        "--offset_pointU_x",
        type=float,
        default=0.9,
        help="위쪽 포인트 X 좌표 비율 (0.0 ~ 1.0)",
    )
    parser.add_argument(
        "--offset_pointU_y",
        type=float,
        default=0.6,
        help="위쪽 포인트 Y 좌표 비율 (0.0 ~ 1.0)",
    )
    parser.add_argument(
        "--point_radius",
        type=int,
        default=25,
        help="포인트 반지름",
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=2000.0,
        help="Depth map 최대 값 (기본값: 2000.0)",
    )
    parser.add_argument(
        "--camera_config",
        type=str,
        help="카메라 설정 파일 경로 (JSON)",
    )
    parser.add_argument(
        "--image_undistortion",
        type=bool,
        default=True,
        help="이미지 왜곡 보정 활성화",
    )

    args = parser.parse_args()

    # 로그 레벨 설정
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Matcher 설정
    config = {
        "target_texture_path": args.target_texture,
        "target_depth_path": args.target_depth,
        "source_image_path": args.source,
        "output_dir": args.output_dir,
        "max_keypoints": args.max_keypoints,
        "ransac_method": args.ransac_method,
        "ransac_reproj_threshold": args.ransac_reproj_threshold,
        "ransac_confidence": args.ransac_confidence,
        "debug_mode": args.debug,
        "offset_pointL": (args.offset_pointL_x, args.offset_pointL_y),
        "offset_pointR": (args.offset_pointR_x, args.offset_pointR_y),
        "offset_pointU": (args.offset_pointU_x, args.offset_pointU_y),
        "point_radius": args.point_radius,
        "depth_max": args.depth_max,
        "camera_config_path": args.camera_config,
        "image_undistortion": args.image_undistortion,
    }

    # Matcher 인스턴스 생성
    matcher = Matcher(config)

    # 파이프라인 실행
    matches_result, ransac_result = matcher.run_pipeline(
        target_texture_path=args.target_texture,
        target_depth_path=args.target_depth,
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

# python run_matcher.py --source datasets/source5.png --target_depth datasets/20250909_152152_match_depth.tif --target_texture datasets/20250909_152152_match_texture.png   --depth_max 2100.0 --camera_config configs/photoneo_camera_config.json --debug
