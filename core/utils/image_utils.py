"""
이미지 로딩 및 전처리 유틸리티
"""

from pathlib import Path
import torch
from PIL import Image
import numpy as np
import cv2
import os

import logging

logging.getLogger("PIL").setLevel(logging.WARNING)


def process_depth_map(
    depth_image: np.ndarray,
    depth_max: float = 1500.0,
) -> np.ndarray:
    """
    Depth map을 처리하여 8비트 이미지로 변환합니다.

    Args:
        depth_image: 원본 depth map (float32/float64)
        depth_max: 최대 depth 값, depth_unit에 따라 설정

    Returns:
        처리된 8비트 depth map (uint8)
    """

    # Depth max 값보다 큰 값은 0으로 설정
    depth_image[depth_image > depth_max] = 0.0

    # 정규화 (0-255 범위로 변환)
    if depth_max > 0:
        normalized_depth = (depth_image / depth_max) * 255.0
    else:
        normalized_depth = np.zeros_like(depth_image)

    normalized_depth[normalized_depth > 0] = 255.0

    # 8비트로 변환 (이미 0으로 설정된 값들은 그대로 0)
    depth_8bit = normalized_depth.astype(np.uint8)

    return depth_8bit


def read_image(
    path,
    grayscale=False,
    depth_max: float = 1700.0,
):
    """이미지 또는 PLY 파일을 읽어서 numpy 배열로 반환합니다."""
    path = Path(path)

    # PLY 파일인지 확인
    if path.suffix.lower() == ".ply":
        from .pcd_utils import load_ply_as_image

        return load_ply_as_image(path)

    try:
        # TIFF 파일 처리 (32비트 depth map 지원)
        if path.suffix.lower() in [".tif", ".tiff"]:
            pil_image = Image.open(str(path))
            image = np.array(pil_image)

            # 단일 채널인 경우 3채널로 확장
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=-1)

        # 일반 이미지 파일 처리
        else:
            mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
            image = cv2.imread(str(path), mode)
            if image is None:
                raise ValueError(f"Cannot read image {path}.")
            if not grayscale and len(image.shape) == 3:
                image = image[:, :, ::-1]  # BGR to RGB

        return image

    except Exception as e:
        raise ValueError(f"Cannot read image {path}. Error: {e}")


def load_image(image_path):
    """이미지를 로드하고 전처리합니다."""
    if isinstance(image_path, str):
        image_path = Path(image_path)

    # PIL로 이미지 로드
    image = Image.open(image_path).convert("RGB")

    # numpy 배열로 변환
    image_np = np.array(image)

    # PyTorch 텐서로 변환 (C, H, W) 형태
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

    return image_tensor


# def resize_image(image, target_size):
#     """이미지를 지정된 크기로 리사이즈합니다."""
#     if isinstance(image, torch.Tensor):
#         # PyTorch 텐서인 경우 PIL로 변환 후 리사이즈
#         image_np = image.permute(1, 2, 0).numpy()
#         image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
#         resized_pil = image_pil.resize(target_size, Image.Resampling.LANCZOS)
#         resized_np = np.array(resized_pil)
#         return torch.from_numpy(resized_np).permute(2, 0, 1).float() / 255.0
#     else:
#         # PIL 이미지인 경우
#         resized = image.resize(target_size, Image.Resampling.LANCZOS)
#         return resized


def resize_image(image, size, interp="cv2_area"):
    if interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(Image, interp[len("pil_") :].upper())
        resized = Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized


def normalize_image(image):
    """이미지를 정규화합니다."""
    if isinstance(image, torch.Tensor):
        return (image - image.mean()) / image.std()
    else:
        image_np = np.array(image)
        return (image_np - image_np.mean()) / image_np.std()
