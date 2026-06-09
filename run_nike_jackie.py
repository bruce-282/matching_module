#!/usr/bin/env python3
"""
nike_jackie RGB 이미지 쌍 매칭 (depth 없음).
datasets/nike_jackie/matcher.config.yaml 의 nike_jackie.source / targets 사용.
같은 폴더에 이미지·YAML 두면 파일명만 적으면 됨.
"""

from __future__ import annotations

import os
import sys
import argparse
import warnings
import yaml
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.matchers.matcher import Matcher
from core.utils.image_utils import read_image
from core.utils.logger_utils import setup_logger


def _resolve_asset_path(path_str: str, base_dir: str) -> str:
    """절대 경로 / datasets/... / 그 외(파일명만) → base_dir 기준."""
    if not path_str:
        return path_str
    if os.path.isabs(path_str):
        return os.path.normpath(path_str)
    norm = path_str.replace("\\", "/").lstrip("./")
    if norm.startswith("datasets/"):
        return os.path.normpath(os.path.join(project_root, norm))
    return os.path.normpath(os.path.join(base_dir, path_str))


def _depth_proxy_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """실제 depth 대신 그레이스케일 프록시 (process_depth_map / 파이프라인 호환)."""
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    return np.mean(rgb.astype(np.float32), axis=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="nike_jackie 2D 매칭")
    parser.add_argument(
        "--config_path",
        type=str,
        default="datasets/nike_jackie/matcher.config.yaml",
        help="matcher 설정 YAML (데이터셋 폴더에 두는 것 권장)",
    )
    parser.add_argument(
        "--param_path",
        type=str,
        default="datasets/nike_jackie/matching.param.yaml",
        help="template param YAML",
    )
    args = parser.parse_args()

    os.chdir(project_root)

    config_path_abs = os.path.abspath(os.path.join(project_root, args.config_path))
    param_path_abs = os.path.abspath(os.path.join(project_root, args.param_path))
    config_dir = os.path.dirname(config_path_abs)
    param_dir = os.path.dirname(param_path_abs)

    with open(config_path_abs, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

    nj = full_config.pop("nike_jackie", None)
    if not nj:
        print("matcher.config.yaml 에 nike_jackie: 섹션이 필요합니다.")
        sys.exit(1)

    source_rel = nj.get("source")
    targets = nj.get("targets") or []
    if not source_rel or not targets:
        print("nike_jackie.source 및 nike_jackie.targets 가 필요합니다.")
        sys.exit(1)

    with open(param_path_abs, "r", encoding="utf-8") as f:
        template_param = yaml.safe_load(f)
    if template_param.get("path_match_source"):
        template_param["path_match_source"] = _resolve_asset_path(
            template_param["path_match_source"], param_dir
        )

    logger = setup_logger(__name__)
    matcher = Matcher(config=full_config, template_param=template_param)
    matcher.init_config(config=full_config, template_param=template_param)

    output_dir = full_config.get("output_dir", "output/nike_jackie")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    source_path = _resolve_asset_path(source_rel, config_dir)
    if not os.path.isfile(source_path):
        logger.error(f"Source 이미지 없음: {source_path}")
        sys.exit(1)

    w = full_config.get("image_size", {}).get("width")
    h = full_config.get("image_size", {}).get("height")
    intrinsic = None
    ci = full_config.get("camera_intrinsics")
    if ci:
        intrinsic = np.array(
            [
                [ci.get("fx", 0), 0, ci.get("cx", 0)],
                [0, ci.get("fy", 0), ci.get("cy", 0)],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    source_image = read_image(source_path, width=w, height=h, intrinsic_matrix=intrinsic)
    if source_image is None:
        logger.error("Source 로드 실패")
        sys.exit(1)

    for target_rel in targets:
        target_path = _resolve_asset_path(target_rel, config_dir)
        if not os.path.isfile(target_path):
            logger.warning(f"건너뜀 (파일 없음): {target_path}")
            continue

        base = os.path.splitext(os.path.basename(target_path))[0]
        logger.info(f"매칭: target={target_path} vs source={source_path}")

        target_texture = read_image(target_path, width=w, height=h, intrinsic_matrix=intrinsic)
        if target_texture is None:
            logger.error(f"Target 로드 실패: {target_path}")
            continue

        target_depth = _depth_proxy_from_rgb(target_texture)

        try:
            # 성공 시: filtered_matches = run_ransac_filtering 결과 (Homography면 "homography" 키 포함)
            matches, filtered_matches, result_3d = matcher.run_pipeline_matching_only(
                target_texture=target_texture,
                target_depth=target_depth,
                source_image=source_image,
                source_depth=None,
                target_texture_path=target_path,
                target_depth_path=target_path,
                output_dir=output_dir,
            )

            # visualize_results는 self.output_path 사용 — 파이프라인과 동일 출력 폴더로 고정
            matcher.output_path = Path(output_dir)

            matcher.visualize_results(
                target_texture=target_texture,
                target_depth=target_depth,
                source_image=source_image,
                plane_normal=None,
                result3d=None,
                ransac_result=filtered_matches,
                camera=None,
                result_image_name=base,
            )
            logger.info(f"완료: {base} → {output_dir}/")
        except Exception as e:
            logger.error(f"실패 {base}: {e}")

    matcher.cleanup()
    logger.info("전체 완료")


if __name__ == "__main__":
    main()
