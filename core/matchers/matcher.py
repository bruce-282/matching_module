#!/usr/bin/env python3
"""
통합 매처 클래스 - 매칭 + RANSAC 필터링
"""

from re import L, S, T
import sys
from pathlib import Path
import numpy as np
import time
import torch
import torchvision.transforms.functional as F
import warnings
import logging
import open3d as o3d
import copy
import cv2
from typing import Dict, List, Optional, Tuple, Any

# torchvision 경고 숨기기
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 로거 설정
from core.utils.logger_utils import setup_logger
from .models.roma import Roma
from ..utils.image_utils import resize_image, process_depth_map
from ..utils.viz_utils import visualize_matches, warp_images
from ..utils.processing_utils import (
    filter_matches,
    registration_ransac_based_on_correspondence,
    solve_rigid_transform_between_points,
)
from ..utils.io_utils import save_points_to_yaml
from ..utils.pcd_utils import (
    create_point_cloud_from_depth_image,
    normal_to_angles,
    compute_plane_normal,
    is_ply_file,
)
from ..utils.camera_utils import Camera
from ..utils.depth_utils import (
    point_cloud_to_depth_map,
    find_depth_from_2d_robust,
)
from ..utils.geometry_utils import project_pcd_to_images, project_3d_point_to_2d


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
            "geometry_type": "Homography",  # Homography or Fundamental
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
            "save_essential": "2d",
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
            # 3D 매칭 설정
            "pose_estimation_method": "ransac",
            "stable_depth_range": 50.0,
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
        self.logger.info(
            f"Model initialization completed (time: {model_init_time:.3f} seconds)"
        )
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
            # logger.debug(f"resize_max:size {size} scale {scale}")
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
            # logger.debug(f"force_resize:size {size} size_new {size_new} scale {scale}")

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
        target_image: np.ndarray,
        source_image: np.ndarray,
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
            target_image,
            resize_max=self.config["resize_max"],
            force_resize=self.config["force_resize"],
        )
        image1, scale1 = self._preprocess(
            source_image,
            resize_max=self.config["resize_max"],
            force_resize=self.config["force_resize"],
        )

        # 원본 이미지 크기와 전처리 후 크기 출력
        self.logger.debug(f"original image0 size: {target_image.shape}")
        self.logger.debug(f"original image1 size: {source_image.shape}")
        self.logger.debug(f"preprocessed image0 size: {image0.shape}")
        self.logger.debug(f"preprocessed image1 size: {image1.shape}")
        self.logger.debug(f"scale0: {scale0}")
        self.logger.debug(f"scale1: {scale1}")

        image0 = image0.to(self.device)[None]
        image1 = image1.to(self.device)[None]

        # 매칭 실행
        try:
            matching_start_time = time.time()
            data = {"image0": image0, "image1": image1}
            result = self.model(data)
            self.matching_time = time.time() - matching_start_time

            # 스케일 계산
            s0 = np.array(target_image.shape[:2][::-1]) / np.array(
                image0.shape[-2:][::-1]
            )
            s1 = np.array(source_image.shape[:2][::-1]) / np.array(
                image1.shape[-2:][::-1]
            )

            confidence = result["mconf"]

            kpts0_shifted = result["keypoints0"] + 0.5
            kpts1_shifted = result["keypoints1"] + 0.5
            keypoints0 = self.scale_keypoints(kpts0_shifted, s0) - 0.5
            keypoints1 = self.scale_keypoints(kpts1_shifted, s1) - 0.5

            self.logger.info(
                f"Matching completed! (matching time: {self.matching_time:.3f} seconds)"
            )

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"number of matches: {len(keypoints0)}")
                # GPU에서 한 번에 통계 계산 후 CPU로 전송
                conf_stats = torch.stack(
                    [
                        torch.mean(confidence),
                        torch.max(confidence),
                        torch.min(confidence),
                    ]
                )
                conf_stats_cpu = conf_stats.cpu().numpy()
                self.logger.debug(
                    f"confidence stats - avg: {conf_stats_cpu[0]:.3f}, max: {conf_stats_cpu[1]:.3f}, min: {conf_stats_cpu[2]:.3f}"
                )
        except Exception as e:
            self.logger.error(f"Error occurred: {e}")
            return None

        return {
            "keypoints0": keypoints0.cpu().numpy(),
            "keypoints1": keypoints1.cpu().numpy(),
            "confidence": confidence.cpu().numpy(),
            "image0": image0.squeeze().cpu().numpy(),
            "image1": image1.squeeze().cpu().numpy(),
            "image0_orig": target_image,
            "image1_orig": source_image,
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

        if "mmkeypoints0_orig" in filtered_pred:
            filtered_kpts0 = filtered_pred["mmkeypoints0_orig"]
            filtered_kpts1 = filtered_pred["mmkeypoints1_orig"]
            filtered_conf = filtered_pred["mmconf"]

        # 디버그 모드일 때만 상세 통계 계산 (효율성 개선)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"number of matches after filtering: {len(filtered_kpts0)}"
            )
            # 중복 계산 방지
            original_count = len(pred["mkeypoints0_orig"])
            filtered_count = len(filtered_kpts0)
            filtered_ratio = (original_count - filtered_count) / original_count * 100
            self.logger.debug(
                f"matches filtered out: {filtered_ratio:.1f}% ({original_count - filtered_count}/{original_count})"
            )

        if len(filtered_conf) > 0:
            self.logger.debug(
                f"average confidence after filtering: {np.mean(filtered_conf):.3f}"
            )
            self.logger.debug(
                f"max confidence after filtering: {np.max(filtered_conf):.3f}"
            )

        if "Homography" in filtered_pred["geom_info"]:
            H = filtered_pred["geom_info"]["Homography"]
            geom_info = filtered_pred["geom_info"]

            return {
                "filtered_kpts0": filtered_kpts0,
                "filtered_kpts1": filtered_kpts1,
                "filtered_conf": filtered_conf,
                "homography": H,
                "geom_info": geom_info,
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
        ransac_result: Optional[Dict[str, Any]] = None,
        camera: Camera = None,
        result_image_name: str = "",
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

        if source_image is None:
            self.logger.error("Source image not found")
            return

        if (
            ransac_result["homography"]
            and not self.config["enable_3d_matching"]
            and (
                self.config["save_essential"] == "all"
                or self.config["save_essential"] == "2d"
            )
        ):

            try:
                warp_result = warp_images(
                    target_image,
                    source_image,
                    ransac_result["homography"],
                    pointL_pos=self.config["pointL_pos"],
                    pointR_pos=self.config["pointR_pos"],
                    pointU_pos=self.config["pointU_pos"],
                    point_radius=self.config["point_radius"],
                )
                output_file = str(
                    self.output_path / f"{result_image_name}_warped_overlapped.png"
                )
                cv2.imwrite(
                    output_file, cv2.cvtColor(warp_result[0], cv2.COLOR_RGB2BGR)
                )
                self.logger.debug(f"warped image saved: {output_file}")

            except Exception as e:
                self.logger.error(f"image warping failed: {e}")
                return

        result1_3d, result2_3d, result3_3d = result3d
        center_point_3d = (result1_3d + result2_3d + result3_3d) / 3

        if (
            self.config["save_essential"] == "all"
            or self.config["save_essential"] == "3d"
        ):
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

            pcd_path = str(self.output_path / f"{result_image_name}_with_anchor.ply")
            o3d.io.write_point_cloud(
                pcd_path,
                pcd,
            )
            self.logger.debug(f"PLY file saved: {pcd_path}")
        if self.config["save_essential"] == "all" or (
            self.config["save_essential"] == "2d" and "pcd" in locals()
        ):
            # Project PCD to 2D images
            try:
                color_image = project_pcd_to_images(
                    pcd=pcd,
                    intrinsic_matrix=camera.get_intrinsic_matrix(),
                    image_size=(
                        target_image.shape[1],
                        target_image.shape[0],
                    ),  # (width, height)
                )

                # Save color projection
                color_path = str(
                    self.output_path / f"{result_image_name}_with_anchor.png"
                )
                cv2.imwrite(color_path, cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
                self.logger.debug(f"PCD color projection saved: {color_path}")

            except Exception as e:
                self.logger.error(f"Failed to project PCD to 2D: {e}")

    def calculate_anchor_points(
        self,
        source_image_shape: Optional[Tuple[int, int]] = None,
        ransac_result: Dict[str, Any] = None,
        transform_matrix: np.ndarray = None,
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
                else:
                    self.logger.error("No valid transformation matrix found")
                    return None

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
        target_depth: np.ndarray,
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
            if len(target_depth.shape) == 3:
                target_depth = target_depth[:, :, 0]

            z1 = find_depth_from_2d_robust(
                target_depth, (int(point1_2d[0]), int(point1_2d[1])), radius
            )
            z2 = find_depth_from_2d_robust(
                target_depth, (int(point2_2d[0]), int(point2_2d[1])), radius
            )
            z3 = find_depth_from_2d_robust(
                target_depth, (int(point3_2d[0]), int(point3_2d[1])), radius
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
            depth_image = point_cloud_to_depth_map(points, self.camera)

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
                self.logger.debug(
                    f"Depth calculation completed: {z1:.1f}, {z2:.1f}, {z3:.1f}"
                )
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
        if self.config["save_essential"] != "none":
            self.output_path.mkdir(exist_ok=True)

        result1_3d = None
        result2_3d = None
        result3_3d = None
        plane_normal = None

        try:
            if self.config["image_undistortion"]:
                target_depth = self.camera.undistort_image(target_depth)

            if target_texture is not None:
                target_image = target_texture
                if self.config["image_undistortion"]:
                    target_image = self.camera.undistort_image(target_image)
                target_clipped = process_depth_map(
                    depth_image=target_depth,
                    texture_image=target_image,
                    depth_max=self.config["depth_max"],
                )
            else:
                target_image = target_depth
                target_clipped = process_depth_map(
                    depth_image=target_depth,
                    depth_max=self.config["depth_max"],
                )
            time_start = time.time()
            matches = self.run_matching(target_clipped, source_image)
            time_end = time.time()
            self.logger.info(f"Matching time: {time_end - time_start:.3f} seconds")

            if matches is None:
                self.logger.error("2D matching failed")
                raise Exception("2D matching failed")

            if (
                self.config["save_essential"] == "all"
                or self.config["save_essential"] == "2d"
            ):
                visualize_matches(
                    target_clipped,
                    source_image,
                    matches["keypoints0"],
                    matches["keypoints1"],
                    matches["confidence"],
                    str(
                        self.output_path
                        / f"{self.target_texture_name}_matches_original.png"
                    ),
                    confidence_threshold=self.config["confidence_threshold"],
                )
            time_start = time.time()
            filtered_matches = self.run_ransac_filtering(matches)
            time_end = time.time()
            self.logger.info(
                f"RANSAC filtering time: {time_end - time_start:.3f} seconds"
            )

            if filtered_matches is None:
                self.logger.error("2D filtering failed")
                raise Exception("2D filtering failed")

            if (
                self.config["save_essential"] == "all"
                or self.config["save_essential"] == "2d"
            ):
                visualize_matches(
                    target_clipped,
                    source_image,
                    filtered_matches["filtered_kpts0"],
                    filtered_matches["filtered_kpts1"],
                    filtered_matches["filtered_conf"],
                    str(
                        self.output_path
                        / f"{self.target_texture_name}_matches_ransac_filtered.png"
                    ),
                    confidence_threshold=self.config["confidence_threshold"],
                )

            if self.config["enable_3d_matching"]:
                time_start = time.time()
                source_depth = source_image  # 원본 depth 이미지 사용 (RGB 변환 전)
                result = self.run_matching_3d(
                    filtered_matches, target_depth, source_depth
                )
                time_end = time.time()
                self.logger.info(
                    f"3D matching time: {time_end - time_start:.3f} seconds"
                )
                if result is None:
                    raise Exception("3D matching failed")

                selected_points = self.config.get("selected_points", {})
                if selected_points is None:
                    raise Exception("Selected points are not set")

                result1_3d = np.array(
                    [
                        selected_points["L"]["x"],
                        selected_points["L"]["y"],
                        selected_points["L"]["z"],
                    ]
                )
                result2_3d = np.array(
                    [
                        selected_points["R"]["x"],
                        selected_points["R"]["y"],
                        selected_points["R"]["z"],
                    ]
                )
                result3_3d = np.array(
                    [
                        selected_points["U"]["x"],
                        selected_points["U"]["y"],
                        selected_points["U"]["z"],
                    ]
                )

                transform_matrix = result["transformation"]

                # 3D 포인트를 homogeneous coordinate로 변환 (4x1) 후 변환 적용
                def apply_transform_3d(point_3d, transform_4x4):

                    point_homo = np.append(point_3d, 1.0)
                    transformed_homo = transform_4x4 @ point_homo
                    return transformed_homo[:3]

                result1_3d = apply_transform_3d(result1_3d, transform_matrix)
                result2_3d = apply_transform_3d(result2_3d, transform_matrix)
                result3_3d = apply_transform_3d(result3_3d, transform_matrix)

                # Project 3D points to 2D and convert to integer coordinates
                point1_2d = project_3d_point_to_2d(
                    result1_3d, self.camera.get_intrinsic_matrix()
                ).astype(int)
                point2_2d = project_3d_point_to_2d(
                    result2_3d, self.camera.get_intrinsic_matrix()
                ).astype(int)
                point3_2d = project_3d_point_to_2d(
                    result3_3d, self.camera.get_intrinsic_matrix()
                ).astype(int)

                z_depthmap = self.calculate_anchor_depth(
                    target_depth_path=target_depth_path,
                    target_depth=target_depth,
                    point1_2d=point1_2d,
                    point2_2d=point2_2d,
                    point3_2d=point3_2d,
                    radius=self.config["point_radius"],
                )
                if z_depthmap is None:
                    self.logger.warning(
                        f"Out of stable depth range: calculate anchor depth failed"
                    )
                    return None, None, None, None
                # Check depth stability for all points
                points_3d = [result1_3d, result2_3d, result3_3d]
                point_names = ["pointL", "pointR", "pointU"]

                for point_3d, depth, name in zip(points_3d, z_depthmap, point_names):
                    depth_diff = abs(point_3d[2] - depth)
                    if depth_diff > self.config["stable_depth_range"]:
                        self.logger.warning(
                            f"Out of stable depth range {name}: depth difference {depth_diff:.1f}mm > {self.config['stable_depth_range']}mm"
                        )
                        return None, None, None, None
                    self.logger.debug(
                        f"Stable depth range {name}: depth difference {depth_diff:.1f}mm <= {self.config['stable_depth_range']}mm"
                    )

            else:
                time_start = time.time()
                result_points_2d = self.calculate_anchor_points(
                    source_image_shape=source_image.shape[:2],
                    ransac_result=filtered_matches,
                )
                if result_points_2d is None:
                    raise Exception("2D points calculation failed")

                result1_2d, result2_2d, result3_2d = result_points_2d
                # Depth 계산
                depth_result = self.calculate_anchor_depth(
                    target_depth_path=target_depth_path,
                    target_depth=target_depth,
                    point1_2d=result1_2d,
                    point2_2d=result2_2d,
                    point3_2d=result3_2d,
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
                    raise Exception("Depth calculation failed")

                z1, z2, z3 = depth_result

                result1_3d, result2_3d, result3_3d = self.calculate_3d_points(
                    np.array([result1_2d[0], result1_2d[1], z1]),
                    np.array([result2_2d[0], result2_2d[1], z2]),
                    np.array([result3_2d[0], result3_2d[1], z3]),
                )
                time_end = time.time()
                self.logger.info(
                    f"3D points calculation time: {time_end - time_start:.3f} seconds"
                )

            self.logger.debug(
                f"3D points: pointL: {result1_3d}, pointR: {result2_3d}, pointU: {result3_3d}"
            )

            plane_normal = compute_plane_normal(result1_3d, result2_3d, result3_3d)

            self.logger.debug(f"Plane normal: {plane_normal}")

            # 4. 결과 시각화
            if self.config["save_essential"] != "none":

                result_3d_points = (result1_3d, result2_3d, result3_3d)

                normal_angles = normal_to_angles(plane_normal)

                self.logger.debug(
                    f"horizontal_deg: {normal_angles[0]:.1f}°, vertical_deg: {normal_angles[1]:.1f}°"
                )

                save_points_to_yaml(
                    target_depth.shape[:2],
                    result_3d_points,
                    plane_normal,
                    normal_angles,
                    self.target_texture_name,
                    self.output_path,
                )
                self.logger.info("Points information is saved to YAML file.")

                self.visualize_results(
                    target_texture=target_clipped,
                    target_depth=target_depth,
                    source_image=source_image,
                    plane_normal=plane_normal,
                    result3d=result_3d_points,
                    ransac_result=filtered_matches,
                    camera=self.camera,
                    result_image_name=self.target_texture_name,
                )

            self.logger.info("\n=== Pipeline completed ===")

            return result1_3d, result2_3d, result3_3d, plane_normal

        except Exception as e:
            raise Exception(f"{e}")

    def calculate_3d_points(
        self, result1_3d: np.ndarray, result2_3d: np.ndarray, result3_3d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def run_matching_3d(
        self,
        filtered_matches: Dict[str, Any],
        depth_target: np.ndarray,
        depth_source: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """
        Fundamental Matrix를 이용한 3D 매칭 및 GICP 정합

        Args:
            filtered_matches: RANSAC 필터링된 매칭 결과
            depth_target: 타겟 이미지의 depth 이미지 (필수)
            depth_source: 소스 이미지의 depth 이미지 (필수)

        Returns:
            3D 매칭 결과 딕셔너리 또는 None
        """
        try:
            # # 1. Fundamental Matrix 추출
            # if "Fundamental" not in filtered_matches.get("geom_info", {}):
            #     self.logger.error("Fundamental matrix not found in filtered matches")
            #     return None

            # F = np.array(filtered_matches["geom_info"]["Fundamental"])
            # self.logger.debug(f"Fundamental matrix: {F}")

            # E = K.T @ F @ K
            # self.logger.debug(f"Essential matrix: {E}")

            keypoints0 = filtered_matches["filtered_kpts0"].astype(np.int32)
            keypoints1 = filtered_matches["filtered_kpts1"].astype(np.int32)

            # if target_depth_scale > 0 and source_depth_scale > 0:
            #     # 8-bit → 원본 복원 (원본 범위: 0-2472.207763671875)
            #     # 8-bit (0-255) → 원본 depth 범위로 복원
            #     original_max = 2002.207763671875  # 원본 최대값
            #     original_min = 0.0  # 원본 최소값

            #     depth_source_restored = depth_source.copy().astype(np.float32)
            #     valid_mask = depth_source > 0

            #     # 8-bit 정규화 복원: (pixel_value / 255.0) * (original_range) + original_min
            #     depth_source_restored[valid_mask] = (depth_source[valid_mask] / 255.0) * (original_max - original_min) + original_min
            #     depth_source = depth_source_restored

            #     self.logger.info(f"Applied 8-bit to original depth restoration")
            #     self.logger.debug(f"8-bit range: 0-255 → Original range: {original_min:.2f}-{original_max:.2f}")
            #     self.logger.debug(f"Target depth scale: {target_depth_scale:.2f}, Source depth scale: {source_depth_scale:.2f}")
            # else:
            #     self.logger.warning("Could not compute depth scale factor, using original depth values")

            # self.logger.info("Using depth images for 3D point extraction")

            # 매칭된 키포인트에서 유효한 depth 값만 추출
            valid_indices = []
            points_3d_target = []
            points_3d_source = []

            # if depth_target.dtype == np.uint8:
            #     depth_target = depth_target.astype(np.float32)
            # if depth_source.dtype == np.uint8:
            #     depth_source = depth_source.astype(np.float32)

            valid_count = 0
            boundary_fail_count = 0
            zero_depth_count = 0

            # for문 밖에서 미리 처리 - depth 이미지 채널 정규화
            if len(depth_target.shape) == 3:
                depth_target_2d = depth_target[:, :, 0]  # 첫 번째 채널만 사용
            else:
                depth_target_2d = depth_target

            if len(depth_source.shape) == 3:
                depth_source_2d = depth_source[:, :, 0]  # 첫 번째 채널만 사용
            else:
                depth_source_2d = depth_source

            for i, (pt0, pt1) in enumerate(zip(keypoints0, keypoints1)):
                x0, y0 = int(pt0[0]), int(pt0[1])
                x1, y1 = int(pt1[0]), int(pt1[1])

                # 이미지 경계 확인
                if (
                    0 <= x0 < depth_target_2d.shape[1]
                    and 0 <= y0 < depth_target_2d.shape[0]
                    and 0 <= x1 < depth_source_2d.shape[1]
                    and 0 <= y1 < depth_source_2d.shape[0]
                ):

                    # 정규화된 2D depth 이미지에서 값 추출
                    d0 = depth_target_2d[y0, x0]
                    d1 = depth_source_2d[y1, x1]

                    valid_count += 1
                    # 2D → 3D 변환
                    z0 = d0
                    z1 = d1

                    x3d_0 = (x0 - self.camera.K[0, 2]) * z0 / self.camera.K[0, 0]
                    y3d_0 = (y0 - self.camera.K[1, 2]) * z0 / self.camera.K[1, 1]

                    x3d_1 = (x1 - self.camera.K[0, 2]) * z1 / self.camera.K[0, 0]
                    y3d_1 = (y1 - self.camera.K[1, 2]) * z1 / self.camera.K[1, 1]

                    points_3d_target.append([x3d_0, y3d_0, z0])
                    points_3d_source.append([x3d_1, y3d_1, z1])
                    valid_indices.append(i)

                else:
                    boundary_fail_count += 1

            # 통계 정보 로깅
            self.logger.debug(f"Point processing stats:")
            self.logger.debug(f"  - Total points: {len(keypoints0)}")
            self.logger.debug(f"  - Valid 3D points: {valid_count}")
            self.logger.debug(f"  - Boundary failures: {boundary_fail_count}")
            self.logger.debug(f"  - Zero depth: {zero_depth_count}")

            if len(points_3d_target) < 4:
                self.logger.error(
                    f"Insufficient valid 3D points: {len(points_3d_target)}"
                )
                return None

            points_3d_target = np.array(points_3d_target)
            points_3d_source = np.array(points_3d_source)

            correspondences = [[i, i] for i in valid_indices]
            # correspondences = o3d.utility.Vector2iVector(correspondences)
            pcd_target = o3d.geometry.PointCloud()
            pcd_source = o3d.geometry.PointCloud()

            pcd_target.points = o3d.utility.Vector3dVector(points_3d_target)
            pcd_source.points = o3d.utility.Vector3dVector(points_3d_source)

            # RANSAC registration with known correspondences
            if self.config["pose_estimation_method"] == "ransac":
                # Config에서 RANSAC 파라미터 가져오기
                ransac_3d_config = self.config.get("ransac_3d", {})
                pose = registration_ransac_based_on_correspondence(
                    pcd_source,
                    pcd_target,
                    correspondences,
                    max_correspondence_distance=ransac_3d_config.get(
                        "max_correspondence_distance", 0.1
                    ),
                    ransac_n=ransac_3d_config.get("ransac_n", 7),
                    max_iterations=ransac_3d_config.get("max_iterations", 5000),
                    confidence=ransac_3d_config.get("confidence", 0.9999),
                )
            elif self.config["pose_estimation_method"] == "svd":
                # Open3D PointCloud에서 numpy array로 변환
                points_source_np = np.asarray(pcd_source.points)
                points_target_np = np.asarray(pcd_target.points)

                pose = solve_rigid_transform_between_points(
                    points_source_np, points_target_np
                )

            elif self.config["pose_estimation_method"] == "teaserpp":
                # TEASER++ registration
                try:
                    from core.matchers.models.teaserpp import Teaserpp

                    # TEASER++ 모델 초기화
                    teaserpp_conf = {
                        "noise_bound": self.config.get("teaserpp_noise_bound", 0.1)
                    }
                    teaserpp_model = Teaserpp(teaserpp_conf)

                    # Point cloud를 numpy array로 변환
                    points_source_np = np.asarray(pcd_source.points)
                    points_target_np = np.asarray(pcd_target.points)

                    self.logger.info(
                        f"TEASER++ Registration - Source: {len(points_source_np)}, Target: {len(points_target_np)}"
                    )

                    # TEASER++ 실행
                    result = teaserpp_model.register(points_source_np, points_target_np)

                    if result["success"]:
                        pose = result["transformation"]
                        self.logger.info(
                            f"TEASER++ registration successful - Inliers: {result['num_inliers']}"
                        )

                except Exception as e:
                    self.logger.error(f"TEASER++ registration error: {e}")

            else:
                raise ValueError(
                    f"Invalid method: {self.config['pose_estimation_method']}"
                )

            # 공통 pose 검증
            self.logger.info(f"Initial pose result:")
            self.logger.info(f"  Rotation det: {np.linalg.det(pose[:3, :3]):.6f}")
            self.logger.info(f"  Translation: {pose[:3, 3]}")
            self.logger.info(f"  Translation norm: {np.linalg.norm(pose[:3, 3]):.3f}")

            if pose is None:
                self.logger.error("Intial pose estimation failed")
                return None

            # 6. Open3D Point Cloud 생성

            pcd_source_transformed = copy.deepcopy(pcd_source)

            pcd_source_transformed.transform(pose)
            # 색상 설정 (시각화용)
            # pcd_target.paint_uniform_color([1, 0, 0])  # 빨간색 (target)
            # pcd_source.paint_uniform_color([0, 1, 0])  # 초록색 (source)
            # pcd_source_transformed.paint_uniform_color([0, 0, 1])  # 파란색 (source transformed)

            # 두 point cloud 합치기

            # combined_pcd = pcd0 + pcd1_transformed

            # # 저장
            # output_path = Path(self.config.get("output_dir", "output"))
            # output_path.mkdir(exist_ok=True)

            # # 파일명 생성
            # pcd_filename_transformed = f"{self.target_texture_name}_pcd_before_icp_transformed_source.ply"
            # pcd_filename_target = f"{self.target_texture_name}_pcd_before_icp_target.ply"
            # pcd_filename_source = f"{self.target_texture_name}_pcd_before_icp_source.ply"
            # pcd_path_transformed = output_path / pcd_filename_transformed
            # pcd_path_target = output_path / pcd_filename_target
            # pcd_path_source = output_path / pcd_filename_source

            # # Before ICP 저장 (원본 상태)
            # o3d.io.write_point_cloud(str(pcd_path_transformed), pcd_source_transformed.paint_uniform_color([0, 0, 1]))
            # self.logger.info(f"Combined point cloud (before ICP) saved: {pcd_path_transformed}")
            # o3d.io.write_point_cloud(str(pcd_path_target), pcd_target.paint_uniform_color([1, 0, 0]))
            # self.logger.info(f"Combined point cloud (before ICP) saved: {pcd_path_target}")
            # o3d.io.write_point_cloud(str(pcd_path_source), pcd_source.paint_uniform_color([0, 1, 0]))
            # self.logger.info(f"Combined point cloud (before ICP) saved: {pcd_path_source}")

            # # 아웃라이어 제거 (ICP 전 전처리)
            # statistical_nb_neighbors = 20
            # statistical_std_ratio = 3.0
            # pcd_source_clean, _ = pcd_source.remove_statistical_outlier(
            #     nb_neighbors=statistical_nb_neighbors, std_ratio=statistical_std_ratio)
            # pcd_target_clean, _ = pcd_target.remove_statistical_outlier(
            #     nb_neighbors=statistical_nb_neighbors, std_ratio=statistical_std_ratio)

            # # 아웃라이어 제거된 point cloud로 변환된 버전 생성
            # pcd_source_transformed_clean = copy.deepcopy(pcd_source_clean)
            # pcd_source_transformed_clean.transform(pose)

            # if not pcd_target_clean.has_normals():
            #     pcd_target_clean.estimate_normals(
            #         o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            # if not pcd_source_transformed_clean.has_normals():
            #     pcd_source_transformed_clean.estimate_normals(
            #         o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            # # Multi-stage ICP (점진적 정밀도 향상)
            # icp_stages = [
            #     {"max_correspondence_distance": 0.1, "max_iteration": 1000, "relative_fitness": 1e-4, "relative_rmse": 1e-4},  # Stage 1: 관대한 설정
            #     {"max_correspondence_distance": 0.05, "max_iteration": 1000, "relative_fitness": 1e-5, "relative_rmse": 1e-5},  # Stage 2: 중간 설정
            #     {"max_correspondence_distance": 0.02, "max_iteration": 2000, "relative_fitness": 1e-6, "relative_rmse": 1e-6},  # Stage 3: 정밀한 설정
            # ]

            # current_pose = np.eye(4)
            # result = None

            # for stage_idx, stage_params in enumerate(icp_stages):
            #     self.logger.info(f"ICP Stage {stage_idx + 1}: max_correspondence_distance={stage_params['max_correspondence_distance']}")

            #     stage_result = o3d.pipelines.registration.registration_icp(
            #         pcd_source_transformed_clean, pcd_target_clean,
            #         max_correspondence_distance=stage_params["max_correspondence_distance"],
            #         init=current_pose,
            #         estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            #         criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            #             relative_fitness=stage_params["relative_fitness"],
            #             relative_rmse=stage_params["relative_rmse"],
            #             max_iteration=stage_params["max_iteration"]
            #         )
            #     )

            #     self.logger.info(f"Stage {stage_idx + 1} Result - Fitness: {stage_result.fitness:.4f}, RMSE: {stage_result.inlier_rmse:.4f}")

            #     # 다음 stage를 위한 pose 업데이트
            #     current_pose = stage_result.transformation
            #     result = stage_result  # 최종 결과 저장

            #     # 조기 종료 조건 (충분히 좋은 결과)
            #     if stage_result.fitness > 0.8:
            #         self.logger.info(f"Early termination at stage {stage_idx + 1} due to high fitness")
            #         break
            # # result = o3d.pipelines.registration.registration_icp(
            # #         pcd_target, pcd_target, 0.01, pose,
            # #         o3d.pipelines.registration.TransformationEstimationPointToPlane())
            # self.logger.info(f"GICP Result - Fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.4f}")
            # self.logger.info(f"GICP converged: {result.fitness > 0.1}")  # fitness > 0.1이면 성공으로 간주

            # # Multi-stage ICP 결과를 원본 source point cloud에 적용
            # pcd_source_transformed2 = copy.deepcopy(pcd_source_clean)
            # # 초기 pose + ICP 결과를 결합
            # final_transformation = result.transformation @ pose
            # pcd_source_transformed2.transform(final_transformation)
            # pcd_source_transformed2.paint_uniform_color([0, 1, 1])  #

            # # 변환된 point cloud 저장
            # pcd_filename_transformed = f"{self.target_texture_name}_pcd_after_icp_transformed_source.ply"
            # pcd_path_transformed = output_path / pcd_filename_transformed
            # o3d.io.write_point_cloud(str(pcd_path_transformed), pcd_source_transformed2)
            # self.logger.info(f"Combined point cloud (after ICP) saved: {pcd_path_transformed}")

            # 8. 결과 반환
            return {
                # "fundamental_matrix": F,
                # "essential_matrix": E,
                # "points_3d_target": points_3d_target,
                # "points_3d_source": points_3d_source,
                # "gicp_result": result,
                # "fitness": result.fitness,
                # "rmse": result.inlier_rmse,
                "transformation": pose
            }

        except Exception as e:
            self.logger.error(f"3D matching failed: {e}")
            return None

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
