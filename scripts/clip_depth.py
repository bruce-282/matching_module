#!/usr/bin/env python3
"""
Depth Clipping 스크립트

TIFF depth 이미지에서 depth_max 값을 넘는 픽셀을 0으로 만들어주는 스크립트

사용법:
    python clip_depth.py --image_file_path datasets/3_IMG_DepthMap_undistorted.tif --output_folder output --depth_max 2300.0
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import logging
import sys
import tifffile
from typing import Union

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clip_depth_image(
    input_path: Union[str, Path], 
    output_path: Union[str, Path], 
    depth_max: float
) -> bool:
    """
    단일 depth 이미지 clipping

    Args:
        input_path: 입력 TIFF 이미지 경로
        output_path: 출력 이미지 경로
        depth_max: 최대 depth 값 (이 값을 넘으면 0으로 설정)

    Returns:
        bool: 성공 여부
    """
    try:
        input_path = Path(input_path)
        output_path = Path(output_path)

        # 출력 디렉토리 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # TIFF 이미지 로드
        logger.info(f"TIFF 이미지 로드 중: {input_path}")
        
        try:
            # tifffile로 읽기 (원본 데이터 타입 보존)
            image = tifffile.imread(str(input_path))
            logger.info(f"이미지 로드 완료 - shape: {image.shape}, dtype: {image.dtype}")
            logger.info(f"원본 depth 범위: min={np.min(image):.3f}, max={np.max(image):.3f}")
            
        except Exception as e:
            logger.error(f"TIFF 이미지를 로드할 수 없습니다: {input_path}. 오류: {e}")
            return False

        # Depth clipping 수행
        logger.info(f"Depth clipping 수행 중... (depth_max: {depth_max})")
        
        # 원본 이미지 복사
        clipped_image = image.copy()
        
        # depth_max를 넘는 픽셀을 0으로 설정
        mask = clipped_image > depth_max
        clipped_count = np.sum(mask)
        clipped_image[mask] = 0.0
        
        logger.info(f"Clipping 완료 - {clipped_count}개 픽셀을 0으로 설정")
        logger.info(f"Clipped depth 범위: min={np.min(clipped_image):.3f}, max={np.max(clipped_image):.3f}")

        # 이미지 저장
        logger.info(f"이미지 저장 중: {output_path}")
        
        try:
            # tifffile로 저장 (원본 데이터 타입 보존)
            tifffile.imwrite(str(output_path), clipped_image)
            logger.info(f"Depth clipping 완료: {input_path} -> {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"이미지 저장 실패: {output_path}. 오류: {e}")
            return False

    except Exception as e:
        logger.error(f"Depth clipping 중 오류 발생: {e}")
        return False


def clip_depth_batch(
    input_path: Union[str, Path], 
    output_folder: Union[str, Path], 
    depth_max: float
) -> bool:
    """
    이미지 파일 또는 폴더 내 모든 TIFF 이미지 depth clipping

    Args:
        input_path: 입력 이미지 파일 또는 폴더 경로
        output_folder: 출력 폴더 경로
        depth_max: 최대 depth 값

    Returns:
        bool: 성공 여부
    """
    try:
        input_path = Path(input_path)
        output_folder = Path(output_folder)

        # 출력 폴더 생성
        output_folder.mkdir(parents=True, exist_ok=True)

        # TIFF 파일 찾기
        tiff_files = []
        
        if input_path.is_file():
            # 단일 파일인 경우
            if input_path.suffix.lower() in [".tif", ".tiff"]:
                tiff_files = [input_path]
            else:
                logger.warning(f"TIFF 파일이 아닙니다: {input_path}")
                return False
        elif input_path.is_dir():
            # 폴더인 경우
            tiff_files.extend(input_path.glob("*.tif"))
            tiff_files.extend(input_path.glob("*.tiff"))
            tiff_files.extend(input_path.glob("*.TIF"))
            tiff_files.extend(input_path.glob("*.TIFF"))
        else:
            logger.error(f"입력 경로가 존재하지 않습니다: {input_path}")
            return False

        if not tiff_files:
            logger.warning(f"TIFF 파일을 찾을 수 없습니다: {input_path}")
            return False

        logger.info(f"총 {len(tiff_files)}개의 TIFF 파일을 찾았습니다.")

        # 각 이미지 처리
        success_count = 0
        for tiff_file in tiff_files:
            # 출력 파일명 생성
            if input_path.is_file():
                output_file = output_folder / f"{tiff_file.stem}_clipped{tiff_file.suffix}"
            else:
                output_file = output_folder / f"clipped_{tiff_file.name}"
        
            if clip_depth_image(tiff_file, output_file, depth_max):
                success_count += 1

        logger.info(f"배치 처리 완료: {success_count}/{len(tiff_files)} 성공")
        return success_count > 0

    except Exception as e:
        logger.error(f"배치 처리 중 오류 발생: {e}")
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Depth Clipping 도구")
    parser.add_argument(
        "--image_file_path",
        type=str,
        required=False,
        default="datasets/6_IMG_DepthMap_undistorted.tif",
        help="입력 TIFF 이미지 파일 또는 폴더 경로",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=False,
        default="output",
        help="출력 폴더 경로",
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        required=False,
        default=1900.0,
        help="최대 depth 값 (이 값을 넘으면 0으로 설정)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")

    args = parser.parse_args()

    # 로그 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 입력 경로 확인
    input_path = Path(args.image_file_path)
    if not input_path.exists():
        logger.error(f"입력 경로가 존재하지 않습니다: {input_path}")
        return 1

    # 처리 수행
    success = clip_depth_batch(input_path, args.output_folder, args.depth_max)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
