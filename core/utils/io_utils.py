"""
입출력 유틸리티 함수들
"""

import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from .logger_utils import get_logger

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
        config: YAML 설정 딕셔너리 (camera_intrinsics, camera_distortions, image_size 포함)

    Returns:
        Camera 객체

    Raises:
        ValueError: 필수 키가 없거나 파라미터 형식이 잘못되었을 때
        Exception: 예상치 못한 오류가 발생했을 때
    """
    from .camera_utils import Camera

    try:
        # 카메라 내부 파라미터 확인
        if "camera_intrinsics" not in config:
            raise ValueError("'camera_intrinsics' key not found in the configuration.")
        
        if "camera_distortions" not in config:
            raise ValueError("'camera_distortions' key not found in the configuration.")

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

        # 이미지 크기 추출
        if "image_size" not in config:
            image_size = (2064, 1544)
        else:
            image_size = (config["image_size"]["width"], config["image_size"]["height"])
        
        # Camera 객체 생성
        camera = Camera(intrinsic_matrix, distortion_coeffs, image_size)
        return camera

    except ValueError as e:
        # 입력 검증 오류는 로깅 후 재발생
        logger.error(f"Configuration validation failed: {e}")
        raise e
    except Exception as e:
        # 예상치 못한 오류는 로깅 후 재발생
        logger.error(f"Unexpected error during camera creation: {e}")
        raise e


def load_photoneo_camera_config(config_path: str) -> Dict[str, Any]:
    """
    Photoneo 카메라 설정 JSON 파일을 로드하고 create_camera_from_yaml_config 형식으로 변환합니다.

    Args:
        config_path: Photoneo 카메라 설정 JSON 파일 경로

    Returns:
        create_camera_from_yaml_config가 기대하는 형식의 딕셔너리:
        {
            "camera_intrinsics": {"fx": float, "fy": float, "cx": float, "cy": float},
            "camera_distortions": {"k1": float, "k2": float, "p1": float, "p2": float, "k3": float},
            "image_size": {"width": int, "height": int}
        }

    Raises:
        FileNotFoundError: 파일을 찾을 수 없을 때
        KeyError: 필수 키가 없을 때
        ValueError: 파라미터 형식이 잘못되었을 때
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Photoneo camera configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # sensores.scanner 섹션 추출
        if "sensores" not in config:
            raise KeyError("'sensores' key not found in the Photoneo configuration file.")
        
        sensors = config["sensores"]
        if "scanner" not in sensors:
            raise KeyError("'scanner' section not found in the Photoneo configuration file.")
        
        scanner = sensors["scanner"]

        # intrinsic_matrix 추출 (9개 요소: [fx, 0, cx, 0, fy, cy, 0, 0, 1])
        if "intrinsic_matrix" not in scanner:
            raise KeyError("'intrinsic_matrix' not found in the scanner section.")
        
        intrinsic_list = scanner["intrinsic_matrix"]
        if len(intrinsic_list) != 9:
            raise ValueError(
                f"Intrinsic matrix must have 9 elements. Current: {len(intrinsic_list)}"
            )
        
        # intrinsic_matrix는 row-major 형식: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        fx = float(intrinsic_list[0])
        fy = float(intrinsic_list[4])
        cx = float(intrinsic_list[2])
        cy = float(intrinsic_list[5])

        # distortion_coefficients 추출
        if "distortion_coefficients" not in scanner:
            raise KeyError("'distortion_coefficients' not found in the scanner section.")
        
        dist_list = scanner["distortion_coefficients"]
        if len(dist_list) < 5:
            raise ValueError(
                f"Distortion coefficients must have at least 5 elements. Current: {len(dist_list)}"
            )
        
        k1 = float(dist_list[0])
        k2 = float(dist_list[1])
        p1 = float(dist_list[2])
        p2 = float(dist_list[3])
        k3 = float(dist_list[4])

        # resolution 추출
        if "resolution" not in scanner:
            raise KeyError("'resolution' not found in the scanner section.")
        
        resolution = scanner["resolution"]
        if "width" not in resolution or "height" not in resolution:
            raise KeyError("'width' or 'height' not found in the resolution section.")
        
        width = int(resolution["width"])
        height = int(resolution["height"])

        # create_camera_from_yaml_config 형식으로 변환
        result = {
            "camera_intrinsics": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            },
            "camera_distortions": {
                "k1": k1,
                "k2": k2,
                "p1": p1,
                "p2": p2,
                "k3": k3,
            },
            "image_size": {
                "width": width,
                "height": height,
            },
        }

        logger.info(f"Photoneo camera configuration loaded: {config_path}")
        logger.debug(f"  Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        logger.debug(f"  Distortions: k1={k1}, k2={k2}, p1={p1}, p2={p2}, k3={k3}")
        logger.debug(f"  Image size: {width}x{height}")

        return result

    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error: {e}")
    except (KeyError, ValueError) as e:
        raise e
    except Exception as e:
        raise Exception(f"Photoneo camera configuration load error: {e}")


def save_points_to_yaml(
    image_size: Tuple[int, int],
    result_3d_points: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    plane_normal: Optional[np.ndarray] = None,
    normal_angles: Optional[Tuple[float, float]] = None,
    image_name: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> None:
    """
    포인트 위치를 YAML 파일로 저장합니다.
    
    Args:
        image_path: 소스 이미지 경로
        image_size: 이미지 크기 (height, width)
        point1_3d: Point L의 3D 좌표 (선택사항)
        point2_3d: Point R의 3D 좌표 (선택사항)
        point3_3d: Point U의 3D 좌표 (선택사항)
        plane_normal: 평면 법선 벡터 (선택사항)
        output_path: 출력 디렉토리 경로 (선택사항)
    """



    # YAML 데이터 구조
    points_data = {
        "source_image": image_name,
        "image_size": {
            "width": int(image_size[1]),
            "height": int(image_size[0]),
        },
    }

    point1_3d, point2_3d, point3_3d = result_3d_points
    # 3D 정보가 있는 경우 추가

    points_data["transformed_points_3d"] = {
            "pointL": {
                "x": float(point1_3d[0] if point1_3d is not None else 0),
                "y": float(point1_3d[1] if point1_3d is not None else 0),
                "z": float(point1_3d[2] if point1_3d is not None else 0),
            },
            "pointR": {
                "x": float(point2_3d[0] if point2_3d is not None else 0),
                "y": float(point2_3d[1] if point2_3d is not None else 0),
                "z": float(point2_3d[2]),
            },
            "pointU": {
                "x": float(point3_3d[0] if point3_3d is not None else 0),
                "y": float(point3_3d[1] if point3_3d is not None else 0),
                "z": float(point3_3d[2] if point3_3d is not None else 0),
            },
            "plane_normal": {
                "x": float(plane_normal[0] if plane_normal is not None else 0),
                "y": float(plane_normal[1] if plane_normal is not None else 0),
                "z": float(plane_normal[2] if plane_normal is not None else 0),
            },
            "normal_angles": {
                "horizontal": float(normal_angles[0] if normal_angles is not None else 0),
                "vertical": float(normal_angles[1] if normal_angles is not None else 0),
            },
            "point_unit": "mm",
            "normal_unit": "deg",
    }

    # source 이미지 이름으로 yaml 파일 생성
    yaml_path = output_path / f"{image_name}_result.yaml"

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            points_data,
            f,
            default_flow_style=False,
            allow_unicode=True,
            indent=2,
        )

    logger.info(f"Point information is saved to {yaml_path}")
