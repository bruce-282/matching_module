"""
이미지 로딩 및 전처리 유틸리티
"""

from pathlib import Path
import torch
from PIL import Image
import numpy as np
import cv2


def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image


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
