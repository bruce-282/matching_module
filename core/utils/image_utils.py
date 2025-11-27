"""
이미지 로딩 및 전처리 유틸리티
"""

from pathlib import Path
import torch
from PIL import Image
import numpy as np
import cv2
import logging

from .logger_utils import get_logger

logger = get_logger(__name__)

logging.getLogger("PIL").setLevel(logging.WARNING)


def process_depth_map(
    depth_image: np.ndarray,
    texture_image: np.ndarray = None,
    depth_max: float = 1500.0,
) -> np.ndarray:
    """
    Process depth map and convert to 8-bit image.
    depth_max보다 큰 값은 texture 값으로 대체합니다.

    Args:
        depth_image: 원본 depth map (float32/float64)
        depth_max: 최대 depth 값
        texture_image: texture 이미지 (depth_max 초과 시 사용)

    Returns:
        처리된 8비트 depth map (uint8)
    """

    # Depth max 값보다 큰 값은 texture 값으로 대체
    if texture_image is not None:

        processed = texture_image.copy()
        mask = (depth_image > depth_max) | (depth_image == 0.0)
        processed[mask] = 0
    else:
        # texture 이미지가 없으면 기존 방식 (0으로 설정)
        processed = depth_image.copy()
        processed[depth_image > depth_max] = 0.0

        if depth_max > 0:
            normalized_depth = (processed / depth_max) * 255.0
        else:
            normalized_depth = np.zeros_like(processed)

        normalized_depth[normalized_depth > 0] = 255.0

        # 8비트로 변환
        processed = normalized_depth.astype(np.uint8)

    return processed


def read_image(
    path,
    grayscale=False,
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
            # tifffile을 사용하여 원본 데이터 타입과 값 보존
            import tifffile

            image = tifffile.imread(str(path))

            # float32로 변환하여 원본 값 보존
            image = image.astype(np.float32)

            # 단일 채널인 경우 3채널로 확장
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                image = np.concatenate([image] * 3, axis=-1)

            # logger.debug(
            #     f"TIFF - tifffile.imread: shape={image.shape}, dtype={image.dtype}"
            # )
            # logger.debug(f"TIFF - min={np.min(image)}, max={np.max(image)}")

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


def apply_roi_mask(image: np.ndarray, roi: list, inplace: bool = False) -> np.ndarray:
    """
    ROI 영역 외의 부분을 0으로 설정합니다.
    이미지 크기는 유지하고 ROI 영역만 유효하게 만듭니다.

    Args:
        image: 입력 이미지 (H, W) 또는 (H, W, C)
        roi: ROI 영역 [x1, y1, x2, y2] 형식
        inplace: True이면 원본 이미지를 수정, False이면 복사본 반환

    Returns:
        ROI 영역 외의 부분이 0으로 설정된 이미지
    """
    if roi is None or len(roi) != 4:
        return image

    x1, y1, x2, y2 = roi[0], roi[1], roi[2], roi[3]

    if not inplace:
        image = image.copy()

    # ROI 영역 외의 부분을 0으로 설정
    image[:y1, :] = 0  # 위쪽 영역
    image[y2:, :] = 0  # 아래쪽 영역
    image[:, :x1] = 0  # 왼쪽 영역
    image[:, x2:] = 0  # 오른쪽 영역

    return image
