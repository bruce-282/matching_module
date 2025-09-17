#!/usr/bin/env python3
"""
Depth Clipping 스크립트

TIFF/EXR depth 이미지에서 depth_max 값을 넘는 픽셀을 0으로 만들어주는 스크립트

사용법:
    # TIFF 파일 (TIFF로 저장)
    python clip_depth.py --image_file_path datasets/3_IMG_DepthMap_undistorted.tif --output_folder output --depth_max 2300.0
    
    # EXR 파일 (TIFF로 변환 저장)
    python clip_depth.py --image_file_path scripts/20250821_13454975_depth32.exr --output_folder output --depth_max 2300.0
    
    # EXR 파일을 PNG로 저장
    python clip_depth.py --image_file_path scripts/20250821_13454975_depth32.exr --output_folder output --depth_max 2300.0 --output_format png
    
    # 폴더 처리 (모든 파일을 PNG로 저장)
    python clip_depth.py --image_file_path datasets/ --output_folder output --depth_max 2300.0 --output_format png
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import logging
import sys
import tifffile
import os
from typing import Union

# OpenCV EXR 지원 활성화
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# EXR 지원 확인 (OpenCV)
EXR_AVAILABLE = True  # OpenCV는 기본적으로 EXR을 지원

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_exr_image(file_path: Path) -> np.ndarray:
    """
    EXR 이미지 읽기 (OpenCV 사용)
    
    Args:
        file_path: EXR 파일 경로
        
    Returns:
        np.ndarray: 이미지 데이터
    """
    try:
        # OpenCV로 EXR 파일 읽기 (IMREAD_ANYDEPTH | IMREAD_UNCHANGED)
        image = cv2.imread(str(file_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        
        if image is None:
            raise ValueError(f"EXR 파일을 읽을 수 없습니다: {file_path}")
        
        logger.info(f"EXR 파일 읽기 성공 - shape: {image.shape}, dtype: {image.dtype}")
        
        # 다중 채널인 경우 첫 번째 채널만 사용 (depth는 보통 단일 채널)
        if len(image.shape) == 3:
            logger.info(f"EXR 다중 채널 감지: {image.shape[2]}개 채널, 첫 번째 채널 사용")
            image = image[:, :, 0]
        
        # Float32로 변환
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        return image
        
    except Exception as e:
        logger.error(f"EXR 파일 읽기 실패: {e}")
        raise


def write_exr_image(file_path: Path, image: np.ndarray) -> bool:
    """
    EXR 이미지 저장 (OpenCV 사용)
    
    Args:
        file_path: 저장할 EXR 파일 경로
        image: 이미지 데이터
        
    Returns:
        bool: 성공 여부
    """
    try:
        # Float32로 변환
        image_float = image.astype(np.float32)
        
        # OpenCV로 EXR 파일 저장
        success = cv2.imwrite(str(file_path), image_float)
        
        if success:
            logger.info(f"EXR 파일 저장 성공: {file_path}")
            return True
        else:
            logger.error(f"EXR 파일 저장 실패: {file_path}")
            return False
        
    except Exception as e:
        logger.error(f"EXR 파일 저장 중 오류: {e}")
        return False


def clip_depth_image(
    input_path: Union[str, Path], 
    output_path: Union[str, Path], 
    depth_max: float,
    output_format: str = "auto"
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

        # 이미지 로드
        logger.info(f"이미지 로드 중: {input_path}")
        
        try:
            if input_path.suffix.lower() == '.exr':
                # EXR 파일 읽기 (OpenCV 사용)
                image = read_exr_image(input_path)
            else:
                # TIFF 파일 읽기 (원본 데이터 타입 보존)
                image = tifffile.imread(str(input_path))
            
            logger.info(f"이미지 로드 완료 - shape: {image.shape}, dtype: {image.dtype}")
            
            # 상세한 depth 통계
            valid_pixels = image[image > 0]  # 0이 아닌 픽셀만
            if len(valid_pixels) > 0:
                logger.info(f"원본 depth 통계:")
                logger.info(f"  - 전체 픽셀: {image.size:,}개")
                logger.info(f"  - 유효 픽셀: {len(valid_pixels):,}개 ({len(valid_pixels)/image.size*100:.1f}%)")
                logger.info(f"  - Min depth: {np.min(valid_pixels):.3f} mm")
                logger.info(f"  - Max depth: {np.max(valid_pixels):.3f} mm")
                logger.info(f"  - Mean depth: {np.mean(valid_pixels):.3f} mm")
                logger.info(f"  - Median depth: {np.median(valid_pixels):.3f} mm")
            else:
                logger.warning("유효한 depth 픽셀이 없습니다 (모든 픽셀이 0)")
            
        except Exception as e:
            logger.error(f"이미지를 로드할 수 없습니다: {input_path}. 오류: {e}")
            return False

        # Depth clipping 수행
        logger.info(f"Depth clipping 수행 중... (depth_max: {depth_max})")
        
        # 원본 이미지 복사
        clipped_image = image.copy()
        
        # depth_max를 넘는 픽셀을 0으로 설정
        mask = clipped_image > depth_max
        clipped_count = np.sum(mask)
        clipped_image[mask] = 0.0
        
        logger.info(f"Clipping 완료 - {clipped_count:,}개 픽셀을 0으로 설정 ({clipped_count/image.size*100:.1f}%)")
        
        # Clipping 후 상세한 depth 통계
        valid_pixels_after = clipped_image[clipped_image > 0]
        if len(valid_pixels_after) > 0:
            logger.info(f"Clipped depth 통계:")
            logger.info(f"  - 유효 픽셀: {len(valid_pixels_after):,}개 ({len(valid_pixels_after)/image.size*100:.1f}%)")
            logger.info(f"  - Min depth: {np.min(valid_pixels_after):.3f} mm")
            logger.info(f"  - Max depth: {np.max(valid_pixels_after):.3f} mm")
            logger.info(f"  - Mean depth: {np.mean(valid_pixels_after):.3f} mm")
        else:
            logger.warning("Clipping 후 유효한 depth 픽셀이 없습니다")

        # 출력 형식 결정
        if output_format == "auto":
            # 입력 파일 형식에 따라 결정
            if input_path.suffix.lower() == '.exr':
                output_format = "tiff"  # EXR은 TIFF로 변환
            else:
                output_format = input_path.suffix.lower().lstrip('.')
        
        # 출력 파일 경로 수정
        if output_format != "auto":
            output_path = output_path.with_suffix(f'.{output_format}')
        
        # 이미지 저장
        logger.info(f"이미지 저장 중: {output_path} (형식: {output_format})")
        
        try:
            if output_format == "exr":
                # EXR 파일 저장 (OpenCV 사용)
                success = write_exr_image(output_path, clipped_image)
                if success:
                    logger.info(f"EXR Depth clipping 완료: {input_path} -> {output_path}")
                    return True
                else:
                    return False
            elif output_format in ["tif", "tiff"]:
                # TIFF 파일 저장 (원본 데이터 타입 보존)
                tifffile.imwrite(str(output_path), clipped_image)
                logger.info(f"TIFF Depth clipping 완료: {input_path} -> {output_path}")
                return True
            elif output_format == "png":
                # PNG 파일 저장 (16비트로 변환)
                # depth 값을 16비트 범위로 정규화
                if np.max(clipped_image) > 0:
                    # 0-65535 범위로 정규화
                    png_image = ((clipped_image / np.max(clipped_image)) * 65535).astype(np.uint16)
                else:
                    png_image = clipped_image.astype(np.uint16)
                
                cv2.imwrite(str(output_path), png_image)
                logger.info(f"PNG Depth clipping 완료: {input_path} -> {output_path}")
                logger.info(f"PNG 저장 시 depth 범위: {np.min(png_image)} - {np.max(png_image)} (16-bit)")
                return True
            else:
                logger.error(f"지원되지 않는 출력 형식: {output_format}")
                return False
            
        except Exception as e:
            logger.error(f"이미지 저장 실패: {output_path}. 오류: {e}")
            return False

    except Exception as e:
        logger.error(f"Depth clipping 중 오류 발생: {e}")
        return False


def clip_depth_batch(
    input_path: Union[str, Path], 
    output_folder: Union[str, Path], 
    depth_max: float,
    output_format: str = "auto"
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

        # 지원되는 이미지 파일 찾기
        image_files = []
        supported_extensions = [".tif", ".tiff", ".exr"]
        
        if input_path.is_file():
            # 단일 파일인 경우
            if input_path.suffix.lower() in supported_extensions:
                image_files = [input_path]
            else:
                logger.warning(f"지원되지 않는 파일 형식입니다: {input_path}")
                logger.info(f"지원되는 형식: {supported_extensions}")
                return False
        elif input_path.is_dir():
            # 폴더인 경우
            for ext in supported_extensions:
                image_files.extend(input_path.glob(f"*{ext}"))
                image_files.extend(input_path.glob(f"*{ext.upper()}"))
        else:
            logger.error(f"입력 경로가 존재하지 않습니다: {input_path}")
            return False

        if not image_files:
            logger.warning(f"지원되는 이미지 파일을 찾을 수 없습니다: {input_path}")
            logger.info(f"지원되는 형식: {supported_extensions}")
            return False

        logger.info(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")

        # 각 이미지 처리
        success_count = 0
        for image_file in image_files:
            # 출력 파일명 생성
            if input_path.is_file():
                output_file = output_folder / f"{image_file.stem}_clipped{image_file.suffix}"
            else:
                output_file = output_folder / f"clipped_{image_file.name}"
        
            if clip_depth_image(image_file, output_file, depth_max, output_format):
                success_count += 1

        logger.info(f"배치 처리 완료: {success_count}/{len(image_files)} 성공")
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
        default="scripts/20250821_13454975_depth32.exr",
        help="입력 이미지 파일 또는 폴더 경로 (지원 형식: .tif, .tiff, .exr)",
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
        default=2900.0,
        help="최대 depth 값 (이 값을 넘으면 0으로 설정)",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        required=False,
        default="png",
        choices=["auto", "tiff", "png", "exr"],
        help="출력 형식 (auto: 입력 형식에 따라 자동 결정, EXR은 TIFF로 변환)",
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
    success = clip_depth_batch(input_path, args.output_folder, args.depth_max, args.output_format)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
