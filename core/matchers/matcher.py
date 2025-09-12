#!/usr/bin/env python3
"""
통합 매처 클래스 - 매칭 + RANSAC 필터링
"""

from re import L, S, T
import sys
from pathlib import Path
import numpy as np
from ..utils.pcd_utils import compute_plane_normal
import cv2
import time
import torch
import torchvision.transforms.functional as F
import warnings
import logging
import open3d as o3d
from typing import Dict, List, Optional, Tuple, Any

# torchvision 경고 숨기기
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# 로거 설정
from core.utils.logger_utils import setup_logger

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .models.roma import Roma
from ..utils.image_utils import resize_image, process_depth_map
from ..utils.viz_utils import visualize_matches, warp_images
from ..utils.processing_utils import filter_matches
from ..utils.io_utils import save_points_to_yaml
from ..utils.pcd_utils import create_point_cloud_from_depth_image
from ..utils.camera_utils import Camera
from ..utils.pcd_utils import is_ply_file
from ..utils.depth_utils import (
    point_cloud_to_depth_map,
    find_depth_from_2d_robust,
)


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
            # 매칭 설정
            "max_keypoints": 2000,
            "match_threshold": 0.2,
            "model_name": "minima_roma.pth",
            # RANSAC 설정
            "ransac_method": "CV2_USAC_MAGSAC",
            "ransac_reproj_threshold": 8.0,
            "ransac_confidence": 0.9999,
            "ransac_max_iter": 300000,
            "min_num_matches": 4,
            "geometry_type": "Homography", # Homography or Fundamental
            # 시각화 설정
            "confidence_threshold": 0.5,
            # 이미지 resize 설정
            "resize_width": 1024,
            "resize_height": 768,
            "resize_max": 640,
            "dfactor": 8,
            # 기타 설정
            "force_resize": False,
     
            "debug_mode": False,
            # 이미지 변환 포인트 설정
            "pointL_pos": {
                "x_ratio": 0.5,
                "y_ratio": 0.89,
            },
            "pointR_pos": {
                "x_ratio": 1.38,
                "y_ratio": 0.89,
            },
            "pointU_pos": {
                "x_ratio": 0.9,
                "y_ratio": 0.6,
            },
            "point_radius": 25,
            "depth_max": 2100.0,
        }
        if config.get("debug_mode", False):
            self.logger = setup_logger(__name__, logging.DEBUG)
        else:
            self.logger = setup_logger(__name__, logging.INFO)
        # 사용자 설정으로 기본 설정 업데이트
        if config:
            self.default_config.update(config)
            self.logger.debug(f"User Parameters: {self.default_config}")

        self.config = self.default_config

        # 디바이스 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 시간 측정을 위한 변수들
        self.model_init_time = 0.0
        self.matching_time = 0.0

        # 모델 초기화
        init_start_time = time.time()
        conf = Roma.default_conf.copy()
        conf["max_keypoints"] = self.config["max_keypoints"]
        conf["match_threshold"] = self.config["match_threshold"]
        conf["model_name"] = self.config["model_name"]
        self.model = Roma(conf)

        model_init_time = time.time() - init_start_time
        self.logger.info(f"Model initialization completed (time: {model_init_time:.3f} seconds)")
        self.camera = None
        # Camera 객체 생성 및 이미지 undistortion
        # YAML 설정에서 카메라 파라미터 직접 읽기
        if "camera_intrinsics" in self.config and "camera_distortions" in self.config:
            try:
                from ..utils.io_utils import create_camera_from_yaml_config

                self.camera = create_camera_from_yaml_config(self.config)
                self.logger.info("Camera created from YAML configuration")
            except Exception as e:
                self.logger.error(f"YAML camera configuration load failed: {e}")
                raise e
        else:
            self.logger.error(
                "YAML 설정에 camera_intrinsics 또는 camera_distortions가 없습니다."
            )
            raise ValueError("Camera configuration file not found")

    def scale_keypoints(self, kpts: torch.Tensor, scale: np.ndarray) -> torch.Tensor:
        """
        Scale keypoints
        
        Args:
            kpts: Keypoints to scale
            scale: Scale factor

        Returns:
            Scaled keypoints
        """
        if np.any(scale != 1.0):
            kpts *= kpts.new_tensor(scale)
        return kpts

    def _preprocess(
        self,
        image: np.ndarray,
        resize_max: int = 0,
        force_resize: bool = False,
        grayscale: bool = False,
        dfactor: int = 8,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Image preprocessing

        Args:
            image: 입력 이미지 (NumPy 배열)
            resize_max: Maximum size
            force_resize: Force resize size
            grayscale: Grayscale
            dfactor: Downsampling factor

        Returns:
            Preprocessed image tensor and scale information

        """
        image = image.astype(np.float32, copy=False)
        size = image.shape[:2][::-1]
        scale = np.array([1.0, 1.0])
        # logger.debug(f"resize_max {resize_max}")
        # logger.debug(f"force_resize {force_resize}")
        if resize_max:
            scale = resize_max / max(size)
            #logger.debug(f"resize_max:size {size} scale {scale}")
            if scale < 1.0:
                size_new = tuple(int(round(x * scale)) for x in size)
                image = resize_image(image, size_new, "cv2_area")
                scale = np.array(size) / np.array(size_new)
        if force_resize:
            size = image.shape[:2][::-1]
            image = resize_image(
                image,
                (self.config["resize_width"], self.config["resize_height"]),
                "cv2_area",
            )
            size_new = (self.config["resize_width"], self.config["resize_height"])
            scale = np.array(size) / np.array(size_new)
            #logger.debug(f"force_resize:size {size} size_new {size_new} scale {scale}")

        if grayscale:
            assert image.ndim == 2, image.shape
            image = image[None]
        elif image.ndim == 3:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        else:
            # 2차원 이미지인 경우 (그레이스케일)
            image = image[None]  # HxW to 1xHxW

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

    def run_matching(
        self,
        image0_origin: np.ndarray,
        image1_origin: np.ndarray,
    ) -> Dict[str, Any]:
        """
        이미지 매칭을 수행

        Args:
            image0_origin: First image (NumPy array)
            image1_origin: Second image (NumPy array)

        Returns:
            Matching result dictionary
        """
        self.logger.debug("=== Matching started ===")

        # 전처리
        image0, scale0 = self._preprocess(
            image0_origin,
            resize_max=self.config["resize_max"],
            force_resize=self.config["force_resize"],
        )
        image1, scale1 = self._preprocess(
            image1_origin,
            resize_max=self.config["resize_max"],
            force_resize=self.config["force_resize"],
        )

        # 원본 이미지 크기와 전처리 후 크기 출력
        self.logger.debug(f"original image0 size: {image0_origin.shape}")
        self.logger.debug(f"original image1 size: {image1_origin.shape}")
        self.logger.debug(f"preprocessed image0 size: {image0.shape}")
        self.logger.debug(f"preprocessed image1 size: {image1.shape}")
        self.logger.debug(f"scale0: {scale0}")
        self.logger.debug(f"scale1: {scale1}")

        image0 = image0.to(self.device)[None]
        image1 = image1.to(self.device)[None]



        # 매칭 실행

        matching_start_time = time.time()
        data = {"image0": image0, "image1": image1}
        result = self.model(data)
        self.matching_time = time.time() - matching_start_time

        # 스케일 계산
        s0 = np.array(image0_origin.shape[:2][::-1]) / np.array(image0.shape[-2:][::-1])
        s1 = np.array(image1_origin.shape[:2][::-1]) / np.array(image1.shape[-2:][::-1])

        confidence = result["mconf"]

        # 키포인트 스케일링 (효율성 개선: 중복 연산 제거)
        kpts0_shifted = result["keypoints0"] + 0.5
        kpts1_shifted = result["keypoints1"] + 0.5
        keypoints0 = self.scale_keypoints(kpts0_shifted, s0) - 0.5
        keypoints1 = self.scale_keypoints(kpts1_shifted, s1) - 0.5

        self.logger.info(f"Matching completed! (matching time: {self.matching_time:.3f} seconds)")
        
        # 디버그 모드일 때만 상세 통계 계산 (효율성 개선)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"number of matches: {len(keypoints0)}")
            # GPU에서 한 번에 통계 계산 후 CPU로 전송
            conf_stats = torch.stack([torch.mean(confidence), torch.max(confidence), torch.min(confidence)])
            conf_stats_cpu = conf_stats.cpu().numpy()
            self.logger.debug(f"confidence stats - avg: {conf_stats_cpu[0]:.3f}, max: {conf_stats_cpu[1]:.3f}, min: {conf_stats_cpu[2]:.3f}")

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
            matches_result: 매칭 결과
            ransac_method: RANSAC 메서드
            ransac_reproj_threshold: RANSAC 재투영 임계값
            ransac_confidence: RANSAC 신뢰도

        Returns:
            RANSAC 필터링 결과 및 geometry info 또는 None
        """

        # 설정값 가져오기
        ransac_method = ransac_method or self.config["ransac_method"]
        ransac_reproj_threshold = (
            ransac_reproj_threshold or self.config["ransac_reproj_threshold"]
        )
        ransac_confidence = ransac_confidence or self.config["ransac_confidence"]

        # 결과를 RANSAC 입력 형식으로 변환
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
        # self.logger.debug(f"filtered_pred: {filtered_pred}")
        # logger.debug(f"pred: {pred}")
        filter_time = time.time() - start_time

        if "mmkeypoints0_orig" in filtered_pred:
            filtered_kpts0 = filtered_pred["mmkeypoints0_orig"]
            filtered_kpts1 = filtered_pred["mmkeypoints1_orig"]
            filtered_conf = filtered_pred["mmconf"]

        self.logger.info(f"RANSAC filtering completed! (time: {filter_time:.3f} seconds)")
        
        # 디버그 모드일 때만 상세 통계 계산 (효율성 개선)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"number of matches after filtering: {len(filtered_kpts0)}")
            # 중복 계산 방지
            original_count = len(pred['mkeypoints0_orig'])
            filtered_count = len(filtered_kpts0)
            filtered_ratio = (original_count - filtered_count) / original_count * 100
            self.logger.debug(f"matches filtered out: {filtered_ratio:.1f}% ({original_count - filtered_count}/{original_count})")

            if len(filtered_conf) > 0:
                self.logger.debug(f"average confidence after filtering: {np.mean(filtered_conf):.3f}")
                self.logger.debug(f"max confidence after filtering: {np.max(filtered_conf):.3f}")

        if "Homography" in filtered_pred["geom_info"]:
            H = filtered_pred["geom_info"]["Homography"]
            geom_info = filtered_pred["geom_info"]

            return {
                "filtered_kpts0": filtered_kpts0,
                "filtered_kpts1": filtered_kpts1,
                "filtered_conf": filtered_conf,
                "homography": H,
                "geom_info": geom_info,
                "filter_time": filter_time,
            }
        elif "Fundamental" in filtered_pred["geom_info"]:
            F = filtered_pred["geom_info"]["Fundamental"]
            geom_info = filtered_pred["geom_info"]


            return {
                "filtered_kpts0": filtered_kpts0,
                "filtered_kpts1": filtered_kpts1,
                "filtered_conf": filtered_conf,
                "fundamental": F,
                "geom_info": geom_info,
                "filter_time": filter_time,
            }
        else:
            self.logger.warning("RANSAC filtering failed - not enough matches")
            return None

    def visualize_results(
        self,
        target_texture: np.ndarray = None,
        target_depth: np.ndarray = None,
        source_image: np.ndarray = None,
        plane_normal: np.ndarray = None,
        result3d: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
        result_image_name: str = "",
        matches_result: Dict[str, Any] = None,
        ransac_result: Optional[Dict[str, Any]] = None,
        camera: Camera = None,
    ) -> None:
        """
        결과 시각화

        Args:
            image0_path: 첫 번째 이미지 경로
            image1_path: 두 번째 이미지 경로
            matches_result: 매칭 결과
            ransac_result: RANSAC 필터링 결과
            output_dir: 출력 디렉토리
        """


        if target_texture is not None:
            target_image = target_texture
        else:
            target_image = target_depth

        visualize_matches(
            target_image,
            source_image,
            matches_result["keypoints0"],
            matches_result["keypoints1"],
            matches_result["confidence"],
            str(self.output_path / f"{result_image_name}_matches_original.png"),
            confidence_threshold=self.config["confidence_threshold"],
        )

        # 2. RANSAC 필터링 후 결과 시각화 (디버그 모드에서만)
        if ransac_result:

            visualize_matches(
                target_image,
                source_image,
                ransac_result["filtered_kpts0"],
                ransac_result["filtered_kpts1"],
                ransac_result["filtered_conf"],
                str(
                    self.output_path
                    / f"{result_image_name}_matches_ransac_filtered.png"
                ),
                confidence_threshold=self.config["confidence_threshold"],
            )

            if ransac_result["homography"] is not None:

                if target_image is not None and source_image is not None:
                    # 이미지 변환 및 오버레이
                    warp_result = warp_images(
                        target_image,
                        source_image,
                        ransac_result["homography"],
                        pointL_pos=self.config["pointL_pos"],
                        pointR_pos=self.config["pointR_pos"],
                        pointU_pos=self.config["pointU_pos"],
                        point_radius=self.config["point_radius"],
                    )

                    result1_3d, result2_3d, result3_3d = result3d

                    center_point_3d = (result1_3d + result2_3d + result3_3d) / 3

                    pcd = create_point_cloud_from_depth_image(
                        target_depth,  # depth 이미지
                        plane_normal,
                        center_point_3d,
                        camera.get_intrinsic_matrix(),
                        result1_3d,
                        result2_3d,
                        result3_3d,
                        texture_image=(
                            target_texture if target_texture is not None else None
                        ),  # texture 이미지 (source 이미지 사용)
                    )
                    pcd.scale(1000.0, center=[0, 0, 0])

                    o3d.io.write_point_cloud(
                        str(self.output_path / f"{result_image_name}_with_normal.ply"),
                        pcd,
                    )
                    self.logger.debug(
                        f"PLY file saved: {self.output_path / f'{result_image_name}_with_normal.ply'}"
                    )

                    if warp_result[0] is not None:

                        output_file = str(
                            self.output_path
                            / f"{result_image_name}_warped_overlapped.png"
                        )
                        cv2.imwrite(
                            output_file, cv2.cvtColor(warp_result[0], cv2.COLOR_RGB2BGR)
                        )
                        self.logger.debug(f"Transformed image saved: {output_file}")
                    else:
                        self.logger.warning("Image transformation failed")
                else:
                    self.logger.error("Image loading failed")

    def calculate_anchor_points(
        self,
        source_image_shape: Optional[Tuple[int, int]] = None,
        ransac_result: Dict[str, Any] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        RANSAC 결과를 바탕으로 포인트 위치를 계산

        Args:
            source_image_shape: 소스 이미지 크기
            ransac_result: RANSAC 필터링 결과

        Returns:
            계산된 포인트 좌표 (point1_2d, point2_2d, point3_2d) 또는 None
        """

        if ransac_result is not None and "homography" in ransac_result:
 

            if source_image_shape is not None:
                # 포인트 위치 계산을 위한 간단한 변환
                h, w = source_image_shape

                if "Homography" in ransac_result["geom_info"]:
                    H = np.array(ransac_result["geom_info"]["Homography"])
                    transform_matrix = np.linalg.inv(H)
                elif "Fundamental" in ransac_result["geom_info"]:
                    F = np.array(ransac_result["geom_info"]["Fundamental"])
                    E = K2.T @ F @ K1
   
                    transform_matrix = np.linalg.inv(E)
                else:
                    self.logger.error("Homography or Fundamental matrix not found in ransac_result")

                # 포인트 변환 계산
                pointL_coords = np.array(
                    [
                        [
                            w * self.config["pointL_pos"]["x_ratio"],
                            h * self.config["pointL_pos"]["y_ratio"],
                            1,
                        ]
                    ],
                    dtype=np.float32,
                )
                transformed_point = transform_matrix @ pointL_coords.T
                transformed_point = transformed_point / transformed_point[2]

                pointR_coords = np.array(
                    [
                        [
                            w * self.config["pointR_pos"]["x_ratio"],
                            h * self.config["pointR_pos"]["y_ratio"],
                            1,
                        ]
                    ],
                    dtype=np.float32,
                )
                transformed_point_2 = transform_matrix @ pointR_coords.T
                transformed_point_2 = transformed_point_2 / transformed_point_2[2]

                pointU_coords = np.array(
                    [
                        [
                            w * self.config["pointU_pos"]["x_ratio"],
                            h * self.config["pointU_pos"]["y_ratio"],
                            1,
                        ]
                    ],
                    dtype=np.float32,
                )
                transformed_point_3 = transform_matrix @ pointU_coords.T
                transformed_point_3 = transformed_point_3 / transformed_point_3[2]

                x1, y1 = int(transformed_point[0][0]), int(transformed_point[1][0])
                x2, y2 = int(transformed_point_2[0][0]), int(transformed_point_2[1][0])
                x3, y3 = int(transformed_point_3[0][0]), int(transformed_point_3[1][0])

                # 2D 포인트 정보 (원본 좌표계)
                point1_2d = np.array([x1, y1])
                point2_2d = np.array([x2, y2])
                point3_2d = np.array([x3, y3])

                return point1_2d, point2_2d, point3_2d
            else:
                self.logger.error("source image shape not found in ransac_result")
                return None
        else:
            self.logger.warning(
                "Homography or Fundamental matrix not found in ransac_result, cannot calculate points"
            )
            return None

    def calculate_anchor_depth(
        self,
        target_depth_path: str,
        target_depth_origin: np.ndarray,
        point1_2d: np.ndarray,
        point2_2d: np.ndarray,
        point3_2d: np.ndarray,
        radius: int = 10,
    ) -> Optional[Tuple[float, float, float]]:
        """
        2D 포인트에서 3D 포인트 정보를 계산 (PLY 파일인 경우에만)

        Args:
            target_image_path: 첫 번째 이미지 경로
            point1_2d: 첫 번째 2D 포인트 [x, y]
            point2_2d: 두 번째 2D 포인트 [x, y]
            point3_2d: 세 번째 2D 포인트 [x, y]
            radius: 주변 픽셀 반지름 (기본값: 10)

        Returns:
            3D 포인트 정보 (point1_3d, point2_3d, point3_3d) 또는 None
        """

        # PLY 파일이 아닌 경우 depth map 처리
        if not is_ply_file(target_depth_path):
            # depth_image가 3차원인 경우 첫 번째 채널만 사용
            if len(target_depth_origin.shape) == 3:
                target_depth_origin = target_depth_origin[:, :, 0]

            z1 = find_depth_from_2d_robust(
                target_depth_origin, (point1_2d[0], point1_2d[1]), radius
            )
            z2 = find_depth_from_2d_robust(
                target_depth_origin, (point2_2d[0], point2_2d[1]), radius
            )
            z3 = find_depth_from_2d_robust(
                target_depth_origin, (point3_2d[0], point3_2d[1]), radius
            )

            # z값만 반환
            if z1 is not None and z2 is not None and z3 is not None:
                return z1, z2, z3
            return None

        try:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(target_depth_path)

            if not pcd.has_points():
                self.logger.warning("PLY file has no points")
                return None

            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None

            # 포인트 클라우드를 depth map으로 변환
            depth_image, intrinsic = point_cloud_to_depth_map(points, colors)

            if depth_image is None:
                self.logger.error("Depth map creation failed")
                return None

            # 2D 포인트를 정수 좌표로 변환
            u1, v1 = int(point1_2d[0]), int(point1_2d[1])
            u2, v2 = int(point2_2d[0]), int(point2_2d[1])
            u3, v3 = int(point3_2d[0]), int(point3_2d[1])

            # depth 값 계산 (주변 픽셀 평균 사용)
            z1 = find_depth_from_2d_robust(depth_image, (u1, v1), radius)
            z2 = find_depth_from_2d_robust(depth_image, (u2, v2), radius)
            z3 = find_depth_from_2d_robust(depth_image, (u3, v3), radius)

            if z1 is not None and z2 is not None and z3 is not None:
                self.logger.debug(f"Depth calculation completed: {z1:.1f}, {z2:.1f}, {z3:.1f}")
                return z1, z2, z3
            else:
                self.logger.error("Depth calculation failed")
                return None

        except Exception as e:
            self.logger.error(f"3D point calculation failed: {e}")
            return None

    def _save_failed_matches(
        self,
        target_clipped,
        source_image,
        matches_result,
        ransac_result,
        output_path,
        target_texture_path,
    ):
        """실패한 매칭 결과를 시각화하여 저장"""

        base_name = Path(target_texture_path).stem
        visualize_matches(
            target_clipped,
            source_image,
            matches_result["keypoints0"],
            matches_result["keypoints1"],
            matches_result["confidence"],
            str(output_path / f"{base_name}_failed_matches_original.png"),
            confidence_threshold=self.config["confidence_threshold"],
        )
        visualize_matches(
            target_clipped,
            source_image,
            ransac_result["filtered_kpts0"],
            ransac_result["filtered_kpts1"],
            ransac_result["filtered_conf"],
            str(output_path / f"{base_name}_failed_matches_ransac_filtered.png"),
            confidence_threshold=self.config["confidence_threshold"],
        )

    def _backproject_to_3d(self, point):
        """2D 포인트를 3D로 변환"""

        intrinsic = self.camera.get_intrinsic_matrix()

        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        z = point[2]
        x = (point[0] - cx) * z / fx
        y = (point[1] - cy) * z / fy
        return np.array([x, y, z])

    def run_pipeline(
        self,
        target_texture: Optional[np.ndarray] = None,
        target_depth: Optional[np.ndarray] = None,
        source_image: Optional[np.ndarray] = None,
        target_texture_path: Optional[str] = None,
        target_depth_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        전체 파이프라인 실행

        Args:
            target_texture: Target texture 이미지 (매칭용) - 이미 로드된 이미지
            target_depth: Target depth 이미지 (depth 계산용) - 이미 로드된 이미지
            source_image: Source 이미지 - 이미 로드된 이미지
            target_texture_path: Target texture 이미지 경로 (debug mode 사용 시 사용)
            target_depth_path: Target depth 이미지 경로 (debug mode 사용 시 사용)
            output_dir: 출력 디렉토리

        Returns:
            Tuple[result1_3d, result2_3d, result3_3d, plane_normal]
        """
        # 경로 설정
        if target_texture_path is None or target_texture is None:
            target_texture_path = target_depth_path
        target_depth_path = target_depth_path or self.config["target_depth_path"]
        output_dir = output_dir or self.config["output_dir"]


        self.logger.debug(f"Target texture: {target_texture_path}")
        self.logger.debug(f"Target depth: {target_depth_path}")
        self.logger.debug(f"Output directory: {output_dir}")


        # for output path
        self.target_texture_name = Path(target_texture_path).stem
        self.target_depth_name = Path(target_depth_path).stem
        self.output_path = Path(output_dir)
        self.output_path.mkdir(exist_ok=True)

        try:
            if self.config["image_undistortion"]:
                target_depth = self.camera.undistort_image(target_depth)

            if target_texture is not None:
                texture_exist = True
                target_image = target_texture
                if self.config["image_undistortion"]:
                    target_image = self.camera.undistort_image(target_image)
                target_clipped = process_depth_map(
                    depth_image=target_depth,
                    texture_image=target_image,
                    depth_max=self.config["depth_max"],
                )
            else:
                texture_exist = False
                target_image = target_depth
                target_clipped = process_depth_map(
                    depth_image=target_depth,
                    depth_max=self.config["depth_max"],
                )

            matches = self.run_matching(target_clipped, source_image)

            filtered_matches = self.run_ransac_filtering(matches)

            if self.config["geometry_type"] == "Fundamental":
                self.run_3d_matching(filtered_matches)
                

            if filtered_matches is None:
                return None, None, None, None

            result_points_2d = self.calculate_anchor_points(
                source_image_shape=source_image.shape[:2],
                ransac_result=filtered_matches,
            )
            if result_points_2d is None:
                self.logger.error("2D points calculation failed")
                return None, None, None, None

            result1_2d, result2_2d, result3_2d = result_points_2d
            # Depth 계산
            depth_result = self.calculate_anchor_depth(
                target_depth_path,
                target_depth,
                result1_2d,
                result2_2d,
                result3_2d,
                radius=self.config["point_radius"],
            )

            if depth_result is None:
                self._save_failed_matches(
                    target_clipped,
                    source_image,
                    matches,
                    filtered_matches,
                    self.output_path,
                    target_texture_path,
                )
                self.logger.error("Depth 계산에 실패했습니다.")
                return None, None, None, None

            z1, z2, z3 = depth_result
            self.logger.debug(
                f"Depth information: pointL: {z1:.1f}mm, pointR: {z2:.1f}mm, pointU: {z3:.1f}mm"
            )

            result1_3d, result2_3d, result3_3d = self.calculate_3d_points(
                np.array([result1_2d[0], result1_2d[1], z1]),   
                np.array([result2_2d[0], result2_2d[1], z2]), 
                np.array([result3_2d[0], result3_2d[1], z3])
            )
            self.logger.debug(
                f"3D points: pointL: {result1_3d}, pointR: {result2_3d}, pointU: {result3_3d}"
            )

            plane_normal = compute_plane_normal(result1_3d, result2_3d, result3_3d)

            self.logger.debug(f"Plane normal: {plane_normal}")

            # 4. 결과 시각화
            if self.config["debug_mode"]:

                save_points_to_yaml(
                    Path(target_depth_path),
                    target_depth.shape[:2],
                    result1_2d,
                    result2_2d,
                    result3_2d,
                    result1_3d,
                    result2_3d,
                    result3_3d,
                    plane_normal,
                    self.output_path,
                )
                self.logger.info("Points information is saved to YAML file.")

                self.visualize_results(
                    target_texture=target_clipped if texture_exist else None,
                    target_depth=target_depth,
                    source_image=source_image,
                    plane_normal=plane_normal,
                    result3d=(result1_3d, result2_3d, result3_3d),
                    result_image_name=(
                        self.target_texture_name
                        if texture_exist
                        else self.target_depth_name
                    ),
                    matches_result=matches,
                    ransac_result=filtered_matches,
                    camera=self.camera,
                )

            # 전체 시간 요약
            total_time = self.matching_time
            if filtered_matches:
                total_time += filtered_matches.get("filter_time", 0.0)

            self.logger.info("\n=== Pipeline completed ===")

            self.logger.info(f"Total matching time: {total_time:.3f} seconds")

            return result1_3d, result2_3d, result3_3d, plane_normal

        except Exception as e:
            self.logger.error(f"Error occurred: {e}")
            import traceback

            traceback.print_exc()
            return None, None, None, None

    def calculate_3d_points(self, result1_3d: np.ndarray, result2_3d: np.ndarray, result3_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2D 좌표와 깊이 정보를 이용하여 3D 포인트를 계산합니다.  
        
        Args:
            result1_3d (np.ndarray): Point L의 3D 좌표 [x, y, z]
            result2_3d (np.ndarray): Point R의 3D 좌표 [x, y, z]  
            result3_3d (np.ndarray): Point U의 3D 좌표 [x, y, z]
            
        Returns:
            tuple: (backprojected1_3d, backprojected2_3d, backprojected3_3d)
                - backprojected1_3d: Point L의 역투영된 3D 좌표
                - backprojected2_3d: Point R의 역투영된 3D 좌표
                - backprojected3_3d: Point U의 역투영된 3D 좌표
        """
        backprojected1_3d = self._backproject_to_3d(result1_3d)
        backprojected2_3d = self._backproject_to_3d(result2_3d)
        backprojected3_3d = self._backproject_to_3d(result3_3d)

        return backprojected1_3d, backprojected2_3d, backprojected3_3d

    def cleanup(self):
        """메모리 정리"""
        self.logger.debug("Memory cleanup started...")

        # 1. 모델 정리
        if hasattr(self, "model") and self.model is not None:
            self.logger.debug("Model memory cleanup in progress...")
            del self.model
            self.model = None

        # 2. 카메라 객체 정리
        if hasattr(self, "camera") and self.camera is not None:
            self.logger.debug("Camera object cleanup in progress...")
            del self.camera
            self.camera = None

        # 3. 설정 정리
        if hasattr(self, "config"):
            self.logger.debug("Configuration cleanup in progress...")
            del self.config
            self.config = None

        # 4. PyTorch 메모리 정리
        if torch.cuda.is_available():
            self.logger.debug("CUDA cache cleanup in progress...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # 5. Python 가비지 컬렉션 강제 실행
        import gc

        gc.collect()

        self.logger.debug("Memory cleanup completed")
