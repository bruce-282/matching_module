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

    # 로그 레벨 설정
    if config.get("debug_mode", False):
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Matcher 인스턴스 생성
    matcher = Matcher(config)

    # 파이프라인 실행
    # 폴더에서 모든 depth.tif와 texture.png 쌍 찾기
    import glob
    import os

    input_dir = config.get("input_dir", "datasets")
    
    # 입력 폴더에서 모든 depth.tif 파일 찾기
    depth_files = glob.glob(os.path.join(input_dir, "*_depth.tif"))

    if not depth_files:
        print(f"경고: {input_dir}에서 *_depth.tif 파일을 찾을 수 없습니다.")
        return

    print(f"발견된 depth 파일: {len(depth_files)}개")

    # 각 depth 파일에 대해 매칭 실행
    matches_result = None
    ransac_result = None

    for depth_file in depth_files:
        # 파일명에서 폴더이름 추출
        base_name = os.path.basename(depth_file).replace("_depth.tif", "")
        texture_file = os.path.join(input_dir, f"{base_name}_texture.png")

        print(f"\n파일 검색 중:")
        print(f"  Depth 파일: {depth_file}")
        print(f"  Base name: {base_name}")
        print(f"  Texture 파일: {texture_file}")
        print(f"  Texture 파일 존재: {os.path.exists(texture_file)}")

        # texture 파일이 존재하는지 확인
        if not os.path.exists(texture_file):
            print(f"경고: {texture_file} 파일이 없습니다. 건너뜁니다.")
            continue

        print(f"\n처리 중: {base_name}")
        print(f"  Depth: {depth_file}")
        print(f"  Texture: {texture_file}")

        # 각 쌍에 대해 별도 출력 디렉토리 생성
        output_dir = config.get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)

        try:
            # 이미지 미리 로드 (undistortion 제외)
            print(f"  이미지 로딩 중...")
            from core.utils.image_utils import read_image
            target_texture = read_image(texture_file)
            target_depth = read_image(depth_file)
            source_image = read_image(config.get("source_image_path", "datasets/source.png"))
            
            print(f"  run_pipeline 호출:")
            print(f"    target_texture: {target_texture.shape}")
            print(f"    target_depth: {target_depth.shape}")
            print(f"    source_image: {source_image.shape}")
            print(f"    output_dir: {output_dir}")

            result1_3d, result2_3d, result3_3d, plane_normal = matcher.run_pipeline(
                target_texture=target_texture,
                target_depth=target_depth,
                source_image=source_image,
                target_texture_path=texture_file,
                target_depth_path=depth_file,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"Error: {base_name} - {e}")
            continue

        # 결과 출력
        if all(x is not None for x in [result1_3d, result2_3d, result3_3d, plane_normal]):
            print(f"✅ 매칭 성공 - {base_name}")
            print(f"   Point L: {result1_3d}")
            print(f"   Point R: {result2_3d}")
            print(f"   Point U: {result3_3d}")
            print(f"   Plane Normal: {plane_normal}")
        else:
            print(f"❌ 매칭 실패 - {base_name}")
    
    # 메모리 정리
    matcher.cleanup()
    print("실행 완료")

if __name__ == "__main__":
    main()

# python run_matcher.py --config_path configs/matcher_config.json
