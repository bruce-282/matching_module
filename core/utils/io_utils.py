"""
입출력 유틸리티 함수들
"""

import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from core.utils.logger_utils import get_logger

logger = get_logger(__name__)


def load_camera_config(config_path: str) -> Dict[str, Any]:
    """
    카메라 설정 JSON 파일을 로드합니다.

    Args:
        config_path: 설정 파일 경로

    Returns:
        카메라 설정 딕셔너리

    Raises:
        FileNotFoundError: 파일을 찾을 수 없을 때
        json.JSONDecodeError: JSON 파싱 오류
        KeyError: 필수 키가 없을 때
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        logger.info(f"Camera configuration file loaded: {config_path}")
        return config

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"JSON parsing error: {e}")
    except Exception as e:
        raise Exception(f"Configuration file load error: {e}")


def extract_camera_params(
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Extract camera parameters, distortion coefficients, and image size from the camera configuration.

    Args:
        config: Camera configuration dictionary

    Returns:
        (K, dist_coeffs, image_size): Camera parameters, distortion coefficients, and image size

    Raises:
        KeyError: Required key not found
        ValueError: Parameter format is incorrect
    """
    try:
        # Extract scanner information from the sensors section
        if "sensores" not in config:
            raise KeyError("'sensores' key not found in the configuration file.")

        sensors = config["sensores"]
        if "scanner" not in sensors:
            raise KeyError("'scanner' section not found in the configuration file.")

        scanner = sensors["scanner"]

        # Extract intrinsic matrix (3x3)
        if "intrinsic_matrix" not in scanner:
            raise KeyError("'intrinsic_matrix' not found in the configuration file.")

        intrinsic_list = scanner["intrinsic_matrix"]
        if len(intrinsic_list) != 9:
            raise ValueError(   
                f"Intrinsic matrix must have 9 elements. Current: {len(intrinsic_list)}"
            )

        K = np.array(intrinsic_list, dtype=np.float64).reshape(3, 3)

        # 왜곡 계수
        if "distortion_coefficients" not in scanner:
            raise KeyError("'distortion_coefficients'가 설정 파일에 없습니다.")

        dist_list = scanner["distortion_coefficients"]
        # OpenCV는 최대 14개 왜곡 계수를 지원하지만, 일반적으로 5개만 사용
        dist_coeffs = np.array(dist_list[:5], dtype=np.float64)

        # 이미지 크기
        if "resolution" not in scanner:
            raise KeyError("'resolution'이 설정 파일에 없습니다.")

        resolution = scanner["resolution"]
        if "width" not in resolution or "height" not in resolution:
            raise KeyError("'width' 또는 'height'가 resolution 섹션에 없습니다.")

        image_size = (resolution["width"], resolution["height"])

        logger.info(f"Camera parameters extracted:")
        logger.info(f"  Image size: {image_size}")
        logger.info(f"  Intrinsic matrix: {K.shape}")
        logger.info(f"  Distortion coefficients: {dist_coeffs.shape}")

        return K, dist_coeffs, image_size

    except Exception as e:
        if isinstance(e, (KeyError, ValueError)):
            raise e
        else:
            raise Exception(f"Camera parameter extraction error: {e}")


def create_camera_from_config(config_path: str):
    """
    Create a Camera object from the configuration file.

    Args:
        config_path: Camera configuration file path

    Returns:
        Camera object

    Raises:
        FileNotFoundError: Configuration file not found
        Exception: Other errors
    """
    from .camera_utils import Camera

    try:
        # Load configuration file
        config = load_camera_config(config_path)

        # Extract camera parameters
        K, dist_coeffs, image_size = extract_camera_params(config)

        # Create Camera object
        camera = Camera(K, dist_coeffs, image_size)

        logger.info(f"Camera object created from configuration file: {config_path}")
        return camera

    except Exception as e:
        logger.error(f"Camera object creation failed: {e}")
        raise e


def create_camera_from_yaml_config(config: Dict[str, Any]):
    """
    YAML 설정에서 Camera 객체를 생성합니다.

    Args:
        config: YAML 설정 딕셔너리 (camera_intrinsics, camera_distortions 포함)

    Returns:
        Camera 객체

    Raises:
        KeyError: 필수 키가 없을 때
        ValueError: 파라미터 형식이 잘못되었을 때
    """
    from .camera_utils import Camera

    try:
        # 카메라 내부 파라미터 확인
        if "camera_intrinsics" not in config:
            raise KeyError("'camera_intrinsics' key not found in the configuration.")
        
        if "camera_distortions" not in config:
            raise KeyError("'camera_distortions' key not found in the configuration.")

        intrinsics = config["camera_intrinsics"]
        distortions = config["camera_distortions"]

        # 내부 파라미터 매트릭스 생성
        intrinsic_matrix = np.array([
            [intrinsics["fx"], 0, intrinsics["cx"]],
            [0, intrinsics["fy"], intrinsics["cy"]],
            [0, 0, 1]
        ], dtype=np.float32)

        # 왜곡 계수 배열 생성
        distortion_coeffs = np.array([
            distortions["k1"], distortions["k2"], 
            distortions["p1"], distortions["p2"], 
            distortions["k3"]
        ], dtype=np.float32)

        # 이미지 크기 추출 (기본값 제공)
        if "image_size" in config:
            image_size = (config["image_size"]["width"], config["image_size"]["height"])
        else:
            raise KeyError("'image_size'가 설정에 없습니다.")

        # Camera 객체 생성
        camera = Camera(intrinsic_matrix, distortion_coeffs, image_size)

        logger.info("Camera object created from YAML configuration")
        return camera

    except Exception as e:
        logger.error(f"Camera object creation failed from YAML configuration: {e}")
        raise e
