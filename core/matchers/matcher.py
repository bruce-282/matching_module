#!/usr/bin/env python3
"""
통합 매처 클래스 - Roma 매칭 + RANSAC 필터링
"""

from re import L, T
import sys
from pathlib import Path
import numpy as np
import cv2
import time
import torch
import torchvision.transforms.functional as F
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any

# torchvision 경고 숨기기
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# 로거 설정
logger = logging.getLogger(__name__)

# 이미지 변환 관련 상수들
DEFAULT_OFFSET_POINT1_X_RATIO = 0.5  # 이미지 너비의 비율 (0.0 ~ 1.0)
DEFAULT_OFFSET_POINT1_Y_RATIO = 0.92  # 이미지 높이의 비율 (0.0 ~ 1.0)
DEFAULT_OFFSET_POINT2_X_RATIO = 1.4  # 이미지 너비의 비율 (0.0 ~ 1.0)
DEFAULT_OFFSET_POINT2_Y_RATIO = 0.92  # 이미지 높이의 비율 (0.0 ~ 1.0)
DEFAULT_POINT_RADIUS = 10  # 포인트 반지름

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from core.matchers.roma import Roma
from core.utils.image_utils import resize_image, read_image
from core.utils.viz_utils import visualize_matches
from core.utils.processing_utils import filter_matches, wrap_images
from core.utils.pcd_utils import get_image_from_file


class Matcher:
    """통합 이미지 매칭 클래스"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Matcher 클래스 초기화

        Args:
            config: 설정 딕셔너리
        """
        # 기본 설정
        self.default_config = {
            # 출력 설정
            "output_dir": "output",
            # Roma 매칭 설정
            "max_keypoints": 2000,
            "match_threshold": 0.2,
            "model_name": "minima_roma.pth",
            # RANSAC 설정
            "ransac_method": "CV2_USAC_MAGSAC",
            "ransac_reproj_threshold": 8.0,
            "ransac_confidence": 0.9999,
            "ransac_max_iter": 300000,
            "min_num_matches": 4,
            "geometry_type": "Homography",
            # 시각화 설정
            "confidence_threshold": 0.5,
            # 이미지 resize 설정
            "resize_width": 640,
            "resize_height": 480,
            "resize_max": 1024,
            "dfactor": 8,
            # 기타 설정
            "force_resize": True,
            "threshold": 0.1,
            "max_features": 2000,
            "keypoint_threshold": 0.01,
            "enable_ransac": True,
            # 디버그 모드
            "debug_mode": False,
            # 이미지 변환 포인트 설정
            "offset_point1": (
                DEFAULT_OFFSET_POINT1_X_RATIO,
                DEFAULT_OFFSET_POINT1_Y_RATIO,
            ),
            "offset_point2": (
                DEFAULT_OFFSET_POINT2_X_RATIO,
                DEFAULT_OFFSET_POINT2_Y_RATIO,
            ),
            "point_radius": DEFAULT_POINT_RADIUS,
        }

        # 사용자 설정으로 기본 설정 업데이트
        if config:
            self.default_config.update(config)
            logger.debug(f"사용자 설정: {self.default_config}")

        self.config = self.default_config

        # 디바이스 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 시간 측정을 위한 변수들
        self.model_init_time = 0.0
        self.matching_time = 0.0

        # Roma 모델 초기화
        logger.debug("Roma 모델을 초기화하는 중...")
        init_start_time = time.time()
        conf = Roma.default_conf.copy()
        conf["max_keypoints"] = self.config["max_keypoints"]
        conf["match_threshold"] = self.config["match_threshold"]
        conf["model_name"] = self.config["model_name"]
        self.roma_model = Roma(conf)
        self.model_init_time = time.time() - init_start_time
        logger.info(f"모델 초기화 완료 (소요시간: {self.model_init_time:.3f}초)")

    def scale_keypoints(self, kpts: torch.Tensor, scale: np.ndarray) -> torch.Tensor:
        """키포인트 스케일링"""
        if np.any(scale != 1.0):
            kpts *= kpts.new_tensor(scale)
        return kpts

    def preprocess(
        self,
        image: np.ndarray,
        resize_max: int = 0,
        force_resize: bool = False,
        grayscale: bool = False,
        dfactor: int = 8,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        이미지 전처리

        Args:
            image: 입력 이미지 (NumPy 배열)
            resize_max: 최대 크기
            force_resize: 강제 리사이즈 크기
            grayscale: 그레이스케일 여부
            dfactor: 다운샘플링 팩터

        Returns:
            전처리된 이미지 텐서와 스케일 정보
        """
        image = image.astype(np.float32, copy=False)
        size = image.shape[:2][::-1]
        scale = np.array([1.0, 1.0])
        logger.debug(f"resize_max {resize_max}")
        logger.debug(f"force_resize {force_resize}")
        if resize_max:
            scale = resize_max / max(size)
            logger.debug(f"resize_max:size {size} scale {scale}")
            if scale < 1.0:
                size_new = tuple(int(round(x * scale)) for x in size)
                image = resize_image(image, size_new, "cv2_area")
                scale = np.array(size) / np.array(size_new)
                logger.debug(f"size {size} size_new {size_new} scale {scale}")
        if force_resize:
            size = image.shape[:2][::-1]
            image = resize_image(
                image,
                (self.config["resize_width"], self.config["resize_height"]),
                "cv2_area",
            )
            size_new = (self.config["resize_width"], self.config["resize_height"])
            scale = np.array(size) / np.array(size_new)
            logger.debug(f"force_resize:size {size} size_new {size_new} scale {scale}")

        if grayscale:
            assert image.ndim == 2, image.shape
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW

        image = torch.from_numpy(image / 255.0).float()

        # assure that the size is divisible by dfactor
        size_new = tuple(
            map(
                lambda x: int(x // dfactor * dfactor),
                image.shape[-2:],
            )
        )
        image = F.resize(image, size=size_new, antialias=True)
        scale = np.array(size) / np.array(size_new)[::-1]

        return image, scale

    def run_roma_matching(
        self,
        image0_origin: np.ndarray,
        image1_origin: np.ndarray,
        max_keypoints: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Roma 모델을 사용하여 이미지 매칭을 수행

        Args:
            image0_path: 첫 번째 이미지 경로
            image1_path: 두 번째 이미지 경로
            max_keypoints: 최대 키포인트 수

        Returns:
            매칭 결과 딕셔너리
        """
        logger.debug("=== Roma 매칭 시작 ===")

        # 전처리
        image0, scale0 = self.preprocess(
            image0_origin,
            resize_max=self.config["resize_max"],
            force_resize=self.config["force_resize"],
        )
        image1, scale1 = self.preprocess(
            image1_origin,
            resize_max=self.config["resize_max"],
            force_resize=self.config["force_resize"],
        )

        # 원본 이미지 크기와 전처리 후 크기 출력
        logger.debug(f"원본 이미지0 크기: {image0_origin.shape}")
        logger.debug(f"원본 이미지1 크기: {image1_origin.shape}")
        logger.debug(f"전처리 후 이미지0 크기: {image0.shape}")
        logger.debug(f"전처리 후 이미지1 크기: {image1.shape}")
        logger.debug(f"스케일0: {scale0}")
        logger.debug(f"스케일1: {scale1}")

        image0 = image0.to(self.device)[None]
        image1 = image1.to(self.device)[None]

        logger.debug(f"최종 입력 이미지 크기: {image0.shape}, {image1.shape}")

        # 매칭 실행
        logger.debug("이미지 매칭을 실행하는 중...")
        matching_start_time = time.time()
        data = {"image0": image0, "image1": image1}
        result = self.roma_model(data)
        self.matching_time = time.time() - matching_start_time

        # 스케일 계산
        s0 = np.array(image0_origin.shape[:2][::-1]) / np.array(image0.shape[-2:][::-1])
        s1 = np.array(image1_origin.shape[:2][::-1]) / np.array(image1.shape[-2:][::-1])

        confidence = result["mconf"]

        # 키포인트 스케일링
        keypoints0 = self.scale_keypoints(result["keypoints0"] + 0.5, s0) - 0.5
        keypoints1 = self.scale_keypoints(result["keypoints1"] + 0.5, s1) - 0.5

        logger.info(f"Roma 매칭 완료! (매칭 시간: {self.matching_time:.3f}초)")
        logger.debug(f"매칭된 키포인트 수: {len(keypoints0)}")
        logger.debug(f"평균 신뢰도: {torch.mean(confidence).item():.3f}")
        logger.debug(f"최고 신뢰도: {torch.max(confidence).item():.3f}")
        logger.debug(f"최저 신뢰도: {torch.min(confidence).item():.3f}")

        return {
            "keypoints0": keypoints0.cpu().numpy(),
            "keypoints1": keypoints1.cpu().numpy(),
            "confidence": confidence.cpu().numpy(),
            "image0": image0.squeeze().cpu().numpy(),
            "image1": image1.squeeze().cpu().numpy(),
            "image0_orig": image0_origin,
            "image1_orig": image1_origin,
            "scale0": s0,
            "scale1": s1,
        }

    def run_ransac_filtering(
        self,
        matches_result: Dict[str, Any],
        ransac_method: Optional[str] = None,
        ransac_reproj_threshold: Optional[float] = None,
        ransac_confidence: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        RANSAC 필터링 수행

        Args:
            matches_result: Roma 매칭 결과
            ransac_method: RANSAC 메서드
            ransac_reproj_threshold: RANSAC 재투영 임계값
            ransac_confidence: RANSAC 신뢰도

        Returns:
            RANSAC 필터링 결과 또는 None
        """
        logger.debug("\n=== RANSAC 필터링 시작 ===")

        # 설정값 가져오기
        ransac_method = ransac_method or self.config["ransac_method"]
        ransac_reproj_threshold = (
            ransac_reproj_threshold or self.config["ransac_reproj_threshold"]
        )
        ransac_confidence = ransac_confidence or self.config["ransac_confidence"]

        # Roma 결과를 RANSAC 입력 형식으로 변환
        pred = {
            "mkeypoints0_orig": matches_result["keypoints0"],
            "mkeypoints1_orig": matches_result["keypoints1"],
            "mconf": matches_result["confidence"],
            "image0_orig": matches_result["image0_orig"] * 255,
            "image1_orig": matches_result["image1_orig"] * 255,
        }

        # RANSAC 필터링 수행
        start_time = time.time()
        filtered_pred = filter_matches(
            pred,
            ransac_method=ransac_method,
            ransac_reproj_threshold=ransac_reproj_threshold,
            ransac_confidence=ransac_confidence,
            ransac_max_iter=self.config["ransac_max_iter"],
            geometry_type=self.config["geometry_type"],
        )
        filter_time = time.time() - start_time

        # 결과 분석
        if "mmkeypoints0_orig" in filtered_pred:
            filtered_kpts0 = filtered_pred["mmkeypoints0_orig"]
            filtered_kpts1 = filtered_pred["mmkeypoints1_orig"]
            filtered_conf = filtered_pred["mmconf"]

            logger.info(f"RANSAC 필터링 완료! (소요시간: {filter_time:.3f}초)")
        logger.debug(f"필터링 후 매칭 수: {len(filtered_kpts0)}")
        logger.debug(
            f"필터링 비율: {len(filtered_kpts0)/len(pred['mkeypoints0_orig'])*100:.1f}%"
        )

        if len(filtered_conf) > 0:
            logger.debug(f"필터링 후 평균 신뢰도: {np.mean(filtered_conf):.3f}")
            logger.debug(f"필터링 후 최고 신뢰도: {np.max(filtered_conf):.3f}")

        # Homography 행렬 정보
        if "H" in filtered_pred:
            H = filtered_pred["H"]

            logger.debug(f"Homography H:\n{H}")

            return {
                "filtered_kpts0": filtered_kpts0,
                "filtered_kpts1": filtered_kpts1,
                "filtered_conf": filtered_conf,
                "homography": filtered_pred.get("H", None),
                "geom_info": filtered_pred.get("geom_info", {}),
                "filter_time": filter_time,
            }
        else:
            logger.warning("RANSAC 필터링 실패 - 충분한 매칭이 없습니다.")
            return None

    def visualize_results(
        self,
        image0_origin: np.ndarray,
        image1_origin: np.ndarray,
        image_path: Path,
        matches_result: Dict[str, Any],
        ransac_result: Optional[Dict[str, Any]],
        output_dir: Optional[str] = None,
    ) -> None:
        """
        결과 시각화

        Args:
            image0_path: 첫 번째 이미지 경로
            image1_path: 두 번째 이미지 경로
            matches_result: Roma 매칭 결과
            ransac_result: RANSAC 필터링 결과
            output_dir: 출력 디렉토리
        """
        logger.debug("\n=== 결과 시각화 ===")

        output_dir = output_dir or self.config["output_dir"]
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        image0_name = Path(image_path).stem
        # 1. 원본 Roma 매칭 결과 시각화 (디버그 모드에서만)
        if self.config["debug_mode"]:
            logger.debug("원본 Roma 매칭 결과 시각화...")

            visualize_matches(
                image0_origin,
                image1_origin,
                matches_result["keypoints0"],
                matches_result["keypoints1"],
                matches_result["confidence"],
                str(output_path / f"{image0_name}_matches_original.png"),
                confidence_threshold=self.config["confidence_threshold"],
            )

        # 2. RANSAC 필터링 후 결과 시각화 (디버그 모드에서만)
        if ransac_result and self.config["debug_mode"]:
            logger.debug("RANSAC 필터링 후 결과 시각화...")
            visualize_matches(
                image0_origin,
                image1_origin,
                ransac_result["filtered_kpts0"],
                ransac_result["filtered_kpts1"],
                ransac_result["filtered_conf"],
                str(output_path / f"{image0_name}_matches_ransac_filtered.png"),
                confidence_threshold=self.config["confidence_threshold"],
            )

            # 디버그 모드에서만 이미지 변환 및 시각화
            if ransac_result["homography"] is not None and self.config["debug_mode"]:
                logger.debug("이미지 변환 결과 생성...")
                img0 = image0_origin
                img1 = image1_origin

                if img0 is not None and img1 is not None:
                    # 이미지 변환 및 오버레이
                    warp_result = wrap_images(
                        img0,
                        img1,
                        ransac_result["geom_info"],
                        "Homography",
                        offset_point1=self.config["offset_point1"],
                        offset_point2=self.config["offset_point2"],
                        point_radius=self.config["point_radius"],
                    )

                    if warp_result[0] is not None:
                        # 변환된 이미지를 파일로 저장
                        output_file = str(
                            output_path / f"{image0_name}_warped_overlapped.png"
                        )
                        cv2.imwrite(
                            output_file, cv2.cvtColor(warp_result[0], cv2.COLOR_RGB2BGR)
                        )
                        logger.debug(f"변환된 이미지 저장: {output_file}")
                    else:
                        logger.warning("이미지 변환 실패")
                else:
                    logger.error("이미지 로드 실패")

    def calculate_points(
        self,
        image0_origin: np.ndarray,
        image1_origin: np.ndarray,
        ransac_result: Dict[str, Any],
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        RANSAC 결과를 바탕으로 포인트 위치를 계산

        Args:
            image0_path: 첫 번째 이미지 경로
            image1_path: 두 번째 이미지 경로
            ransac_result: RANSAC 필터링 결과

        Returns:
            계산된 포인트 좌표 (x1, y1, x2, y2) 또는 None
        """
        if ransac_result["homography"] is not None:
            img0 = image0_origin
            img1 = image1_origin

            if img0 is not None and img1 is not None:
                # 포인트 위치 계산을 위한 간단한 변환
                h0, w0, _ = img0.shape
                h1, w1, _ = img1.shape
                H = np.array(ransac_result["geom_info"]["Homography"])
                H_inv = np.linalg.inv(H)

                # 포인트 변환 계산
                offset_point1_coords = np.array(
                    [
                        [
                            w1 * self.config["offset_point1"][0],
                            h1 * self.config["offset_point1"][1],
                            1,
                        ]
                    ],
                    dtype=np.float32,
                )
                transformed_point = H_inv @ offset_point1_coords.T
                transformed_point = transformed_point / transformed_point[2]

                offset_point2_coords = np.array(
                    [
                        [
                            w1 * self.config["offset_point2"][0],
                            h1 * self.config["offset_point2"][1],
                            1,
                        ]
                    ],
                    dtype=np.float32,
                )
                transformed_point_2 = H_inv @ offset_point2_coords.T
                transformed_point_2 = transformed_point_2 / transformed_point_2[2]

                x1, y1 = int(transformed_point[0][0]), int(transformed_point[1][0])
                x2, y2 = int(transformed_point_2[0][0]), int(transformed_point_2[1][0])

                return x1, y1, x2, y2
            else:
                logger.error("이미지 로드 실패")
                return None
        else:
            logger.warning("Homography가 계산되지 않아 포인트를 계산할 수 없습니다.")
            return None

    def run_pipeline(
        self,
        image0_path: Optional[str] = None,
        image1_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        전체 파이프라인 실행

        Args:
            image0_path: 첫 번째 이미지 경로
            image1_path: 두 번째 이미지 경로
            output_dir: 출력 디렉토리

        Returns:
            Roma 매칭 결과와 RANSAC 필터링 결과
        """
        # 경로 설정
        image0_path = image0_path or self.config["image0_path"]
        image1_path = image1_path or self.config["image1_path"]
        output_dir = output_dir or self.config["output_dir"]

        logger.debug("=== Roma 매칭 + RANSAC 필터링 시작 ===")
        logger.debug(f"이미지0: {image0_path}")
        logger.debug(f"이미지1: {image1_path}")
        logger.debug(f"출력 디렉토리: {output_dir}")
        logger.debug(f"최대 키포인트: {self.config['max_keypoints']}")
        logger.debug(f"RANSAC 재투영 임계값: {self.config['ransac_reproj_threshold']}")
        logger.debug(f"RANSAC 신뢰도: {self.config['ransac_confidence']}")
        logger.debug(f"디버그 모드: {self.config['debug_mode']}")

        try:
            # 1. Roma 매칭
            # 경로 설정

            # 이미지 로드
            # PLY 파일인지 확인
            if Path(image0_path).suffix.lower() == ".ply":

                image0_origin = get_image_from_file(
                    image0_path, width=2064, height=1544
                )
            else:
                image0_origin = read_image(image0_path)

            if Path(image1_path).suffix.lower() == ".ply":
                image1_origin = get_image_from_file(
                    image1_path, width=2064, height=1544
                )
            else:
                image1_origin = read_image(image1_path)

            matches_result = self.run_roma_matching(
                image0_origin, image1_origin, self.config["max_keypoints"]
            )

            # 2. RANSAC 필터링
            ransac_result = self.run_ransac_filtering(matches_result)

            # 3. 포인트 계산 및 YAML 파일 저장 (RANSAC 성공 시)
            if ransac_result is not None:
                output_path = Path(output_dir)
                output_path.mkdir(exist_ok=True)

                # 포인트 계산
                points = self.calculate_points(
                    image0_origin, image1_origin, ransac_result
                )

                # YAML 파일 저장
                if points is not None:
                    x1, y1, x2, y2 = points
                    from core.utils.processing_utils import save_points_to_yaml

                    save_points_to_yaml(
                        Path(image0_path),
                        image0_origin.shape[:2],
                        x1,
                        y1,
                        x2,
                        y2,
                        output_path,
                    )
                    logger.info(f"포인트 위치가 YAML 파일로 저장되었습니다.")

            # 4. 결과 시각화
            self.visualize_results(
                image0_origin,
                image1_origin,
                Path(image0_path),
                matches_result,
                ransac_result,
                output_dir,
            )

            # 전체 시간 요약
            total_time = self.model_init_time + self.matching_time
            if ransac_result:
                total_time += ransac_result.get("filter_time", 0.0)

            logger.info("\n=== 파이프라인 완료 ===")
            logger.info(f"모델 초기화 시간: {self.model_init_time:.3f}초")
            logger.info(f"매칭 시간: {self.matching_time:.3f}초")
            if ransac_result:
                logger.info(
                    f"RANSAC 필터링 시간: {ransac_result.get('filter_time', 0.0):.3f}초"
                )
            logger.info(f"총 소요 시간: {total_time:.3f}초")

            return matches_result, ransac_result

        except Exception as e:
            logger.error(f"오류가 발생했습니다: {e}")
            import traceback

            traceback.print_exc()
            return None, None
