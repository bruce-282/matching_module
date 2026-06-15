#!/usr/bin/env python3
"""
통합 매처 스크립트 - Roma 매칭 + RANSAC 필터링
"""

import sys
from pathlib import Path
import argparse
import warnings
import logging
import yaml
import time

# torchvision 경고 숨기기
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.matchers.matcher import Matcher
from core.utils.image_utils import read_image


def main():
    """메인 함수"""

    parser = argparse.ArgumentParser(description="Roma 매칭 + RANSAC 필터링")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Configuration file path (YAML)",
    )
    parser.add_argument(
        "--template_param_path",
        type=str,
        required=True,
        help="Template parameter file path (YAML)",
    )

    args = parser.parse_args()

    # Load configuration file (YAML)
    try:
        with open(args.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config_path}")
        return
    except yaml.YAMLError as e:
        print(f"YAML file parsing failed: {e}")
        return

    try:
        with open(args.template_param_path, "r", encoding="utf-8") as f:
            template_param = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Template parameter file not found: {args.template_param_path}")
        return
    except yaml.YAMLError as e:
        print(f"YAML file parsing failed: {e}")
        return
    # 로거 설정
    from core.utils.logger_utils import setup_logger

    # Set log level
    if config.get("debug_mode", False):
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = setup_logger(__name__)

    # Matcher 인스턴스 생성
    matcher = Matcher(config=config, template_param=template_param)

    matcher.init_config(config=config, template_param=template_param)
    # 파이프라인 실행
    # 폴더에서 모든 depth.tif와 texture.png 쌍 찾기
    import glob
    import os
    import numpy as np

    input_dir = config.get("input_dir", "datasets")

    # 입력 폴더에서 모든 depth.tif 파일 찾기
    depth_files = glob.glob(os.path.join(input_dir, "*_depth.tif"))

    if not depth_files:
        logger.warning(f"Warning: No *_depth.tif files found in {input_dir}")
        return

    # 필수 카메라 설정 검증 (누락 시 명확히 안내하고 종료)
    camera_intrinsics = config.get("camera_intrinsics")
    if not isinstance(camera_intrinsics, dict):
        logger.error("config에 'camera_intrinsics'(fx, fy, cx, cy)가 없습니다.")
        return
    missing_keys = [k for k in ("fx", "fy", "cx", "cy") if k not in camera_intrinsics]
    if missing_keys:
        logger.error(f"'camera_intrinsics'에 다음 키가 없습니다: {missing_keys}")
        return

    image_size = config.get("image_size")
    if not isinstance(image_size, dict) or "width" not in image_size or "height" not in image_size:
        logger.error("config에 'image_size'(width, height)가 없습니다.")
        return

    intrinsic_matrix = np.array(
        [
            [camera_intrinsics["fx"], 0, camera_intrinsics["cx"]],
            [0, camera_intrinsics["fy"], camera_intrinsics["cy"]],
            [0, 0, 1],
        ]
    )
    # path_match_source 찾기: template_param 최상위 -> template_param.matching_model -> config.matching_model
    path_match_source = (
        template_param.get("path_match_source")
        or template_param.get("matching_model", {}).get("path_match_source")
    )
    
    if path_match_source is None:
        logger.error("path_match_source를 찾을 수 없습니다. template_param 또는 config를 확인하세요.")
        return
    
    source_image = read_image(
        path_match_source,
        width=image_size["width"],
        height=image_size["height"],
        intrinsic_matrix=intrinsic_matrix,
    )
    if source_image is None:
        logger.error(
            f"Source image not found: {path_match_source}"
        )
        return

    for depth_file in depth_files:
        # Extract folder name from file name
        base_name = os.path.basename(depth_file).replace("_depth.tif", "")
        texture_file = os.path.join(input_dir, f"{base_name}_texture.png")

        logger.info(f"Depth file: {depth_file}")
        logger.info(f"Texture file: {texture_file}")

        # Check if texture file exists
        if not os.path.exists(texture_file):
            logger.warning(f"Warning: {texture_file} file not found. Skipping.")
            continue

        # 각 쌍에 대해 별도 출력 디렉토리 생성
        output_dir = config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"   output_dir: {output_dir}")

        try:
            # 이미지 미리 로드 (undistortion 제외)
            logger.info(f"Loading images...")
            target_texture = read_image(texture_file)
            target_depth = read_image(depth_file)

            # logger.info(
            #     f"     target_texture shape/dtype: {target_texture.shape, target_texture.dtype}"
            # )
            # logger.info(
            #     f"     target_depth shape/dtype: {target_depth.shape, target_depth.dtype}"
            # )
            # logger.info(
            #     f"     source_image shape/dtype: {source_image.shape, source_image.dtype}"
            # )

            time_start = time.time()
            result = matcher.run_pipeline(
                target_texture=target_texture,
                target_depth=target_depth,
                source_image=source_image,
                target_texture_path=texture_file,
                target_depth_path=depth_file,
                output_dir=output_dir,
            )
            time_end = time.time()
            logger.info(f"Total matching time: {time_end - time_start:.3f} seconds")
        except Exception as e:
            # run_pipeline 은 도메인 실패를 결과 객체로 반환하므로, 여기로 오는 것은
            # 이미지 로드 등 파이프라인 밖의 예기치 못한 오류뿐이다.
            logger.error(f"{base_name} - {e}")
            continue

        # 결과 출력 (예외 없이 결과 객체의 success 로 분기)
        if result.success:
            logger.info(
                f"Matching success - {base_name} \n Point L: {result.point_l} \n Point R: {result.point_r} \n Point U: {result.point_u} \n Plane Normal: {result.plane_normal}"
            )
        else:
            logger.error(
                f"❌ Matching failed - {base_name} "
                f"[{result.code}] {result.error_code.name}: {result.error_message}"
            )

    # 메모리 정리
    matcher.cleanup()
    logger.info("Execution completed")


if __name__ == "__main__":
    main()

# python run_matcher.py --config_path configs/MX5_ICE/matcher_config.yaml --template_param_path configs/MX5_ICE/matcher.teac 
