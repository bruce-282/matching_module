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


def collect_subfolders_with_depth(root: Path):
    """root 아래에서 *_depth.tif가 하나라도 있는 디렉터리만 수집 (중복 없이)."""
    seen = set()
    root = root.resolve()
    for path in root.rglob("*_depth.tif"):
        if path.is_file():
            seen.add(path.parent)
    return sorted(seen)


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
    parser.add_argument(
        "--input_folder",
        type=str,
        default=None,
        help="상위 입력 폴더. 지정 시 --batch와 함께 사용해 하위 폴더를 탐색합니다 (config input_dir override).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="--input_folder 지정 시, 하위 폴더를 재귀 탐색하여 *_depth.tif가 있는 폴더마다 매칭 실행.",
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

    # 배치 모드: 상위 폴더 지정 시 하위 폴더 탐색
    input_root = Path(args.input_folder) if args.input_folder else Path(config.get("input_dir", "datasets"))
    input_root = input_root.resolve()
    base_output_dir = config.get("output_dir", "output")

    if args.batch:
        subfolders = collect_subfolders_with_depth(input_root)
        if not subfolders:
            logger.warning(f"No subfolders with *_depth.tif found under {input_root}")
            return
        logger.info(f"Batch mode: {len(subfolders)} subfolders to process under {input_root}")
        input_dirs_and_outputs = [
            (str(sf), str(Path(base_output_dir) / sf.relative_to(input_root)))
            for sf in subfolders
        ]
    else:
        input_dirs_and_outputs = [(str(input_root), base_output_dir)]

    intrinsic_matrix = np.array(
        [
            [
                config.get("camera_intrinsics").get("fx"),
                0,
                config.get("camera_intrinsics").get("cx"),
            ],
            [
                0,
                config.get("camera_intrinsics").get("fy"),
                config.get("camera_intrinsics").get("cy"),
            ],
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
        width=config.get("image_size").get("width"),
        height=config.get("image_size").get("height"),
        intrinsic_matrix=intrinsic_matrix,
    )
    if source_image is None:
        logger.error(
            f"Source image not found: {path_match_source}"
        )
        return

    for input_dir, output_dir in input_dirs_and_outputs:
        depth_files = glob.glob(os.path.join(input_dir, "*_depth.tif"))
        if not depth_files:
            logger.warning(f"No *_depth.tif in {input_dir}, skipping.")
            continue

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Input dir: {input_dir} -> output_dir: {output_dir}")

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

            try:
                # 이미지 미리 로드 (undistortion 제외)
                logger.info(f"Loading images...")
                target_texture = read_image(texture_file)
                target_depth = read_image(depth_file)

                time_start = time.time()
                result1_3d, result2_3d, result3_3d, plane_normal = matcher.run_pipeline(
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
                logger.error(f"{base_name} - {e}")
                continue

            # 결과 출력
            if all(
                x is not None for x in [result1_3d, result2_3d, result3_3d, plane_normal]
            ):
                logger.info(
                    f"Matching success - {base_name} \n Point L: {result1_3d} \n Point R: {result2_3d} \n Point U: {result3_3d} \n Plane Normal: {plane_normal}"
                )
            else:
                logger.error(f"❌ Matching failed - {base_name}")

    # 메모리 정리
    matcher.cleanup()
    logger.info("Execution completed")


if __name__ == "__main__":
    main()

# python run_matcher.py --config_path configs/MX5_ICE/matcher_config.yaml --template_param_path configs/MX5_ICE/matcher.teac 
