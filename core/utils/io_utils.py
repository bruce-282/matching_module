"""
입출력 유틸리티 함수들
"""

import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


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
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        logger.info(f"카메라 설정 파일 로드 완료: {config_path}")
        return config

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"JSON 파싱 오류: {e}")
    except Exception as e:
        raise Exception(f"설정 파일 로드 중 오류 발생: {e}")


def extract_camera_params(
    config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    카메라 설정에서 내부 파라미터, 왜곡 계수, 이미지 크기를 추출합니다.

    Args:
        config: 카메라 설정 딕셔너리

    Returns:
        (K, dist_coeffs, image_size): 카메라 내부 파라미터, 왜곡 계수, 이미지 크기

    Raises:
        KeyError: 필수 키가 없을 때
        ValueError: 파라미터 형식이 잘못되었을 때
    """
    try:
        # sensors 섹션에서 scanner 정보 추출
        if "sensores" not in config:
            raise KeyError("'sensores' 키가 설정 파일에 없습니다.")

        sensors = config["sensores"]
        if "scanner" not in sensors:
            raise KeyError("'scanner' 섹션이 설정 파일에 없습니다.")

        scanner = sensors["scanner"]

        # 내부 파라미터 행렬 (3x3)
        if "intrinsic_matrix" not in scanner:
            raise KeyError("'intrinsic_matrix'가 설정 파일에 없습니다.")

        intrinsic_list = scanner["intrinsic_matrix"]
        if len(intrinsic_list) != 9:
            raise ValueError(
                f"내부 파라미터 행렬은 9개 요소여야 합니다. 현재: {len(intrinsic_list)}"
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

        logger.info(f"카메라 파라미터 추출 완료:")
        logger.info(f"  이미지 크기: {image_size}")
        logger.info(f"  내부 파라미터 행렬: {K.shape}")
        logger.info(f"  왜곡 계수: {dist_coeffs.shape}")

        return K, dist_coeffs, image_size

    except Exception as e:
        if isinstance(e, (KeyError, ValueError)):
            raise e
        else:
            raise Exception(f"카메라 파라미터 추출 중 오류 발생: {e}")


def create_camera_from_config(config_path: str):
    """
    설정 파일에서 Camera 객체를 생성합니다.

    Args:
        config_path: 카메라 설정 파일 경로

    Returns:
        Camera 객체

    Raises:
        FileNotFoundError: 설정 파일을 찾을 수 없을 때
        Exception: 기타 오류
    """
    from .camera_utils import Camera

    try:
        # 설정 파일 로드
        config = load_camera_config(config_path)

        # 카메라 파라미터 추출
        K, dist_coeffs, image_size = extract_camera_params(config)

        # Camera 객체 생성
        camera = Camera(K, dist_coeffs, image_size)

        logger.info(f"설정 파일에서 Camera 객체 생성 완료: {config_path}")
        return camera

    except Exception as e:
        logger.error(f"Camera 객체 생성 실패: {e}")
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
            raise KeyError("'camera_intrinsics' 키가 설정에 없습니다.")
        
        if "camera_distortions" not in config:
            raise KeyError("'camera_distortions' 키가 설정에 없습니다.")

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

        logger.info("YAML 설정에서 Camera 객체 생성 완료")
        return camera

    except Exception as e:
        logger.error(f"YAML 설정에서 Camera 객체 생성 실패: {e}")
        raise e
