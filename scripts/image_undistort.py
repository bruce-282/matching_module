#!/usr/bin/env python3
"""
이미지 Undistortion 스크립트

사용법:
    python undistort.py --input input_image.jpg --output output_image.jpg
    python undistort.py --input input_folder --output output_folder
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import logging
import sys
from typing import Union, List
from PIL import Image

# 프로젝트 모듈 import
sys.path.append(str(Path(__file__).parent.parent))
from core.utils import create_default_camera

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def undistort_single_image(
    input_path: Union[str, Path], output_path: Union[str, Path], camera=None
) -> bool:
    """
    단일 이미지 undistortion

    Args:
        input_path: 입력 이미지 경로
        output_path: 출력 이미지 경로
        camera: Camera 객체 (None이면 기본 카메라 사용)

    Returns:
        bool: 성공 여부
    """
    try:
        input_path = Path(input_path)
        output_path = Path(output_path)

        # 출력 경로가 디렉토리인 경우 파일명 생성
        if output_path.is_dir() or not output_path.suffix:
            # 입력 파일명에 _undistorted 추가
            output_filename = f"{input_path.stem}_undistorted{input_path.suffix}"
            output_path = output_path / output_filename
        
        # 출력 디렉토리 생성
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 이미지 로드
        logger.info(f"이미지 로드 중: {input_path}")

        # TIFF 파일인 경우 PIL 사용, 그 외에는 OpenCV 사용
        if input_path.suffix.lower() in [".tif", ".tiff"]:
            try:
                pil_image = Image.open(str(input_path))
                image = np.array(pil_image)

                # 단일 채널인 경우 3채널로 확장
                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
                elif len(image.shape) == 3 and image.shape[2] == 1:
                    image = np.concatenate([image] * 3, axis=-1)

                # PIL은 RGB, OpenCV는 BGR이므로 변환
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = image[:, :, ::-1]  # RGB to BGR

            except Exception as e:
                logger.error(
                    f"TIFF 이미지를 로드할 수 없습니다: {input_path}. 오류: {e}"
                )
                return False
        else:
            image = cv2.imread(str(input_path))
            if image is None:
                logger.error(f"이미지를 로드할 수 없습니다: {input_path}")
                return False

        # Camera 객체 생성 (없으면)
        if camera is None:
            camera = create_default_camera((image.shape[1], image.shape[0]))

        # Undistortion 수행
        logger.info("Undistortion 수행 중...")
        undistorted_image = camera.undistort_image(image)

        # 이미지 저장
        logger.info(f"이미지 저장 중: {output_path}")
        
        # TIFF 파일인 경우 압축 옵션 사용
        if output_path.suffix.lower() in [".tif", ".tiff"]:
            # OpenCV로 TIFF 무손실 압축 저장
            success = cv2.imwrite(str(output_path), undistorted_image, [cv2.IMWRITE_TIFF_COMPRESSION, 0])
        else:
            # 일반 이미지는 OpenCV 사용
            success = cv2.imwrite(str(output_path), undistorted_image)

        if success:
            logger.info(f"Undistortion 완료: {input_path} -> {output_path}")
            return True
        else:
            logger.error(f"이미지 저장 실패: {output_path}")
            return False

    except Exception as e:
        logger.error(f"Undistortion 중 오류 발생: {e}")
        return False


def undistort_batch_images(
    input_folder: Union[str, Path], output_folder: Union[str, Path], camera=None
) -> bool:
    """
    폴더 내 모든 이미지 undistortion

    Args:
        input_folder: 입력 폴더 경로
        output_folder: 출력 폴더 경로
        camera: Camera 객체 (None이면 기본 카메라 사용)

    Returns:
        bool: 성공 여부
    """
    try:
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)

        # 출력 폴더 생성
        output_folder.mkdir(parents=True, exist_ok=True)

        # 지원하는 이미지 확장자
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        # 이미지 파일 찾기
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_folder.glob(f"*{ext}"))
            image_files.extend(input_folder.glob(f"*{ext.upper()}"))

        if not image_files:
            logger.warning(f"입력 폴더에 이미지 파일이 없습니다: {input_folder}")
            return False

        logger.info(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.")

        # 각 이미지 처리
        success_count = 0
        for image_file in image_files:
            output_file = output_folder / f"undistorted_{image_file.name}"

            if undistort_single_image(image_file, output_file, camera):
                success_count += 1

        logger.info(f"배치 처리 완료: {success_count}/{len(image_files)} 성공")
        return success_count > 0

    except Exception as e:
        logger.error(f"배치 처리 중 오류 발생: {e}")
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="이미지 Undistortion 도구")
    parser.add_argument(
        "--input", "-i", required=True, help="입력 이미지 파일 또는 폴더 경로"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="출력 이미지 파일 또는 폴더 경로"
    )
    parser.add_argument(
        "--camera-params", "-c", help="카메라 파라미터 파일 (JSON/YAML) - 미구현"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그 출력")

    args = parser.parse_args()

    # 로그 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 입력 경로 확인
    if not input_path.exists():
        logger.error(f"입력 경로가 존재하지 않습니다: {input_path}")
        return 1

    # Camera 객체 생성 (기본값 사용)
    camera = None
    if input_path.is_file():
        # 단일 파일인 경우 이미지 크기를 알 수 없으므로 나중에 생성
        pass
    else:
        # 폴더인 경우 첫 번째 이미지로 카메라 객체 생성
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        first_image = None
        for ext in image_extensions:
            first_image = next(input_path.glob(f"*{ext}"), None)
            if first_image:
                break

        if first_image:
            image = cv2.imread(str(first_image))
            if image is not None:
                camera = create_default_camera((image.shape[1], image.shape[0]))

    # 처리 수행
    success = False
    if input_path.is_file():
        # 단일 파일 처리
        success = undistort_single_image(input_path, output_path, camera)
    else:
        # 폴더 처리
        success = undistort_batch_images(input_path, output_path, camera)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
