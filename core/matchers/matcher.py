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
from .errors import MatcherError
from .error_handler import ErrorCode
from .results import MatchResult
from .models.roma import Roma
from ..utils.image_utils import resize_image, process_depth_map, apply_roi_mask
from ..utils.viz_utils import visualize_matches, warp_images, visualize_3d_correspondences
from ..utils.processing_utils import (
    filter_matches,
    registration_ransac_based_on_correspondence,
    solve_rigid_transform_between_points,
)
from ..utils.io_utils import save_points_to_yaml, create_camera_from_yaml_config, load_photoneo_camera_config
from ..utils.pcd_utils import (
    create_point_cloud_from_depth_image,
    add_normal_line_to_pcd,
    add_3d_points_to_pcd,
    normal_to_angles,
    compute_plane_normal,
    is_ply_file,
    clip_pointcloud_by_depth,
)
from ..utils.camera_utils import Camera
from ..utils.depth_utils import (
    point_cloud_to_depth_map,
    find_depth_from_2d_robust,
)
from ..utils.geometry_utils import (
    project_open3d_pcd_to_image,
    project_3d_point_to_2d,
    create_transform_matrix_from_vectors,
    is_point_in_safe_zone,
    is_point_in_obb,
    transform_safe_zone,
    as_4x4_matrix,
    transform_point_3d,
)


class Matcher:
    """통합 이미지 매칭 클래스"""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        template_param: Optional[Dict[str, Any]] = None,
    ):
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
            "max_keypoints": 3000,
            "match_threshold": 0.2,
            "model_name": "minima_roma.pth",
            # RANSAC 설정
            "ransac_method": "CV2_USAC_MAGSAC",
            "ransac_reproj_threshold": 12.0,
            "ransac_confidence": 0.9999,
            "ransac_max_iter": 30000,
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
            "point_radius": 25,
            "depth_max": 2400.0,
            # 3D 매칭 설정
            "pose_estimation_method": "ransac",
            "stable_depth_range": 50.0,
            # 이미지 보정/시각화 기본값 (누락 시 KeyError 방지 — optional)
            "image_undistortion": False,
            "result_image_contrast": 1.0,
            "result_image_brighten": 0,
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
        self.template_param = template_param or None
        # 디바이스 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 시간 측정을 위한 변수들

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

        self._warmup_model()

        self.camera_target = None
        self.camera_source = None
        # hand-eye 캘리브레이션 (cam -> robot, 4x4). safe zone 로봇 프레임 검사용.
        #  - template(teaching): template_param 에서 init 시 1회 파싱해 캐싱.
        #  - runtime(매칭): run_pipeline 의 target_camera_extrinsic 인자로 매 호출 전달.
        #    (없으면 config 의 camera_calibration 으로 fallback)
        self.template_camera_calibration = None
        self.runtime_camera_calibration = None

        self.init_config(config=config, template_param=template_param)
        # Camera 객체 생성 및 이미지 undistortion
        # YAML 설정에서 카메라 파라미터 직접 읽기
        time.sleep(1)

    def _warmup_model(self) -> None:
        """
        모델 웜업 수행 (첫 실행 시 느린 문제 해결)

        더미 이미지를 사용하여 모델을 한 번 실행하여
        CUDA 커널 초기화, 메모리 할당 등을 미리 수행합니다.
        """
        warmup_start_time = time.time()
        self.logger.info("Warming up model...")
        try:
            # 작은 더미 이미지 생성 (웜업용)
            dummy_size = (3, 256, 256)  # (C, H, W)
            dummy_image0 = torch.randn(1, *dummy_size, device=self.device)
            dummy_image1 = torch.randn(1, *dummy_size, device=self.device)

            warmup_data = {
                "image0": dummy_image0,
                "image1": dummy_image1,
            }

            # 웜업 실행
            _ = self.model(warmup_data)

            warmup_time = time.time() - warmup_start_time
            self.logger.info(
                f"Model warmup completed (time: {warmup_time:.3f} seconds)"
            )
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")

    def init_config(
        self,
        config: Optional[Dict[str, Any]] = None,
        template_param: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize parameters

        Args:
            config: Configuration dictionary
            template_param: Template parameters dictionary
        """
        if config:
            self.default_config.update(config)
        self.config = self.default_config

        self.template_param = template_param or None
        if self.config:
            self.logger.info(f"Matcher Parameters: {self.config}")
        if self.template_param:
            self.logger.info(f"Template Parameters: {self.template_param}")


        if self.template_param:
            try:
                self.camera_source = create_camera_from_yaml_config(self.template_param)
                self.logger.info("Camera source template created from YAML configuration")
            except Exception as e:
                self.logger.error(f"YAML camera source template configuration load failed: {e}")
                raise e
        else:
            self.logger.error(
                "Camera source template YAML 설정에 camera_intrinsics 또는 camera_distortions가 없습니다."
            )
            raise ValueError("Camera source configuration file not found")


        try:
            self.camera_target = create_camera_from_yaml_config(self.config)
            self.logger.info("Camera target created from YAML configuration")
        except Exception as e:
            self.logger.error(f"YAML camera target configuration load failed: {e}")
            raise e

        # 템플릿(teaching) 카메라 hand-eye 캘리브레이션 (teaching cam -> robot, 4x4).
        # template_param 에 정의되며, intrinsic 처럼 init 시 한 번만 파싱해 캐싱한다.
        raw_teach = None
        if self.template_param:
            raw_teach = self.template_param.get("camera_calibration") or (
                self.template_param.get("matching_model", {}).get("camera_calibration")
            )
        self.template_camera_calibration = self._parse_camera_calibration(
            raw_teach, source="template"
        )

        # 현재(runtime/매칭) 카메라 hand-eye 캘리브레이션 fallback (current cam -> robot, 4x4).
        # 매칭용은 원칙적으로 run_pipeline 의 target_camera_extrinsic 인자로 받지만,
        # config 에 있으면 인자 미전달 시 fallback 으로 쓴다. (선택값)
        self.runtime_camera_calibration = self._parse_camera_calibration(
            self.config.get("camera_calibration"), source="runtime config"
        )


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
        transform_matrix: np.ndarray = None,
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
                if self.config["result_image_brighten"] > 0:
                    warp_contrast = cv2.convertScaleAbs(
                        warp_result[0], alpha=self.config["result_image_contrast"]
                    )
                else:
                    warp_contrast = warp_result[0]
                cv2.imwrite(output_file, cv2.cvtColor(warp_contrast, cv2.COLOR_RGB2BGR))
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
                camera.get_intrinsic_matrix(),
                texture_image=(
                    target_texture if target_texture is not None else None
                ),  # texture 이미지 (source 이미지 사용)
            )

            def get_scaled_point(point, scale):
                point_3d = np.array(point) / scale
                return point_3d

            scaledL_3d = get_scaled_point(result1_3d, 1000.0)
            scaledR_3d = get_scaled_point(result2_3d, 1000.0)
            scaledU_3d = get_scaled_point(result3_3d, 1000.0)

            # pcd_raw = copy.deepcopy(pcd)
            # pcd_raw.scale(1000.0, center=[0, 0, 0])
            # pcd_raw.transform(transform_matrix)

            # pcd_path = str(self.output_path / f"{result_image_name}_with_transform.ply")
            # o3d.io.write_point_cloud(
            #     pcd_path,
            #     pcd_raw,
            # )
            # self.logger.debug(f"PLY file saved: {pcd_path}")

            pcd = add_3d_points_to_pcd(pcd, [scaledL_3d, scaledR_3d, scaledU_3d])
            # scaled_center_point_3d = (scaledL_3d + scaledR_3d + scaledU_3d) / 3
            # pcd = add_normal_line_to_pcd(pcd, scaled_center_point_3d, plane_normal, line_length=0.1)

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
                # color_image = project_open3d_pcd_to_image(
                #     pcd=pcd,
                #     intrinsic_matrix=camera.get_intrinsic_matrix(),
                #     image_size=(
                #         target_image.shape[1],
                #         target_image.shape[0],
                #     ),  # (width, height)
                # )
                lr_diff = result2_3d - result1_3d
                lr_distance = np.linalg.norm(lr_diff)
                camera_distance = lr_distance * 3.0

                camera_pos = center_point_3d + plane_normal * camera_distance
                front_vector = -center_point_3d + camera_pos
                front_vector = front_vector / np.linalg.norm(front_vector)

                lr_vector = lr_diff / lr_distance
                up_vector = np.cross(front_vector, lr_vector)
                up_vector = up_vector / np.linalg.norm(up_vector)

                camera_transform = create_transform_matrix_from_vectors(
                    right_vector=lr_vector,
                    up_vector=up_vector,
                    front_vector=front_vector,
                    position=camera_pos,
                )

                pcd_transformed = pcd.transform(np.linalg.inv(camera_transform))

                pcd_clipped = clip_pointcloud_by_depth(
                    pcd_transformed,
                    near_z=0.0,
                    far_z=float(camera_distance + camera_distance * 0.1),
                )

                front_view_image = project_open3d_pcd_to_image(
                    pcd=pcd_clipped,
                    intrinsic_matrix=camera.get_intrinsic_matrix(),
                    image_size=(
                        target_image.shape[1],
                        target_image.shape[0],
                    ),  # (width, height)
                )

                if self.config["result_image_contrast"] > 0:
                    # color_image = cv2.convertScaleAbs(color_image, alpha=self.config["result_image_contrast"])
                    front_view_image = cv2.convertScaleAbs(
                        front_view_image, alpha=self.config["result_image_contrast"]
                    )

                # Save color projection
                color_path = str(
                    self.output_path / f"{result_image_name}_with_anchor.png"
                )
                # combined_image = np.vstack([color_image, front_view_image])
                cv2.imwrite(
                    color_path, cv2.cvtColor(front_view_image, cv2.COLOR_RGB2BGR)
                )
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
            if z1 is None:
                self.logger.error(
                    f"pointL depth calculation failed at pixel "
                    f"({int(point1_2d[0])}, {int(point1_2d[1])}), radius {radius}"
                )
                return None
            z2 = find_depth_from_2d_robust(
                target_depth, (int(point2_2d[0]), int(point2_2d[1])), radius
            )
            if z2 is None:
                self.logger.error(
                    f"pointR depth calculation failed at pixel "
                    f"({int(point2_2d[0])}, {int(point2_2d[1])}), radius {radius}"
                )
                return z1, None, None
            z3 = find_depth_from_2d_robust(
                target_depth, (int(point3_2d[0]), int(point3_2d[1])), radius
            )
            if z3 is None:
                self.logger.error(
                    f"pointU depth calculation failed at pixel "
                    f"({int(point3_2d[0])}, {int(point3_2d[1])}), radius {radius}"
                )
                return z1, z2, None
            else:
                self.logger.debug(
                    f"depth calculation completed: {z1:.1f}, {z2:.1f}, {z3:.1f}"
                )
                return z1, z2, z3

        try:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(target_depth_path)

            if not pcd.has_points():
                self.logger.warning("PLY file has no points")
                return None

            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None

            # 포인트 클라우드를 depth map으로 변환
            depth_image = point_cloud_to_depth_map(points, self.camera_target)

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

            missing = [n for n, z in (("L", z1), ("R", z2), ("U", z3)) if z is None]
            if missing:
                self.logger.error(
                    f"Depth calculation failed for point(s) {missing} "
                    f"(pixels L=({u1},{v1}) R=({u2},{v2}) U=({u3},{v3}), radius {radius})"
                )
                return None
            self.logger.debug(
                f"Depth calculation completed: {z1:.1f}, {z2:.1f}, {z3:.1f}"
            )
            return z1, z2, z3

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
            contrast=self.config["result_image_contrast"],
        )
        visualize_matches(
            target_clipped,
            source_image,
            ransac_result["filtered_kpts0"],
            ransac_result["filtered_kpts1"],
            ransac_result["filtered_conf"],
            str(output_path / f"{base_name}_failed_matches_ransac_filtered.png"),
            confidence_threshold=self.config["confidence_threshold"],
            contrast=self.config["result_image_contrast"],
        )

    def _backproject_to_3d(self, point: np.ndarray, camera: Camera) -> np.ndarray:
        """2D 포인트를 3D로 변환"""

        intrinsic = camera.get_intrinsic_matrix()

        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        z = point[2]
        x = (point[0] - cx) * z / fx
        y = (point[1] - cy) * z / fy
        return np.array([x, y, z])

    def _get_median_xy_from_region(
        self,
        pixel_2d: Tuple[int, int],
        depth_image: np.ndarray,
        radius: int,
        camera: Camera,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        주변 픽셀들의 3D 좌표에서 X, Y의 median 계산

        Args:
            pixel_2d: 2D 픽셀 좌표 (u, v)
            depth_image: depth 이미지
            radius: 탐색 반경

        Returns:
            (median_x, median_y) 또는 유효한 픽셀이 없으면 (None, None)
        """
        u, v = int(pixel_2d[0]), int(pixel_2d[1])
        h, w = depth_image.shape[:2]

        # depth 이미지가 3채널인 경우 첫 번째 채널 사용
        if len(depth_image.shape) == 3:
            depth_2d = depth_image[:, :, 0]
        else:
            depth_2d = depth_image

        xs_3d, ys_3d = [], []
        for dv in range(max(0, v - radius), min(h, v + radius + 1)):
            for du in range(max(0, u - radius), min(w, u + radius + 1)):
                # 원형 마스크 적용
                dist = np.sqrt((du - u) ** 2 + (dv - v) ** 2)
                if dist <= radius:
                    z = depth_2d[dv, du]
                    if z > 0:  # 유효한 depth만
                        point_3d = self._backproject_to_3d(np.array([du, dv, z]), camera=camera)
                        xs_3d.append(point_3d[0])
                        ys_3d.append(point_3d[1])

        if len(xs_3d) == 0:
            return None, None

        return float(np.median(xs_3d)), float(np.median(ys_3d))

    def _apply_depth_correction(
        self,
        result_3d: np.ndarray,
        result_2d: np.ndarray,
        calculated_depth: float,
        depth_diff: float,
        target_depth: np.ndarray,
        point_name: str,
    ) -> Optional[np.ndarray]:
        """
        Depth 차이가 threshold 이상인 경우 보정된 3D 좌표 반환

        Args:
            result_3d: 현재 3D 좌표 (X, Y 차이 검증용)
            result_2d: 2D 픽셀 좌표
            calculated_depth: depth map에서 읽은 Z값
            depth_diff: depth 차이 (절대값)
            target_depth: target depth 이미지
            point_name: 포인트 이름 (로깅용)

        Returns:
            보정된 3D 좌표 또는 None (보정 불필요/취소 시)

        Config (depth_correction 섹션):
            min_depth_diff: 보정을 적용할 최소 depth 차이 (기본값: 5.0mm)
            max_xy_diff: 보정을 취소할 최대 X,Y 차이 (기본값: 5.0mm)
        """
        correction_config = self.config.get("depth_correction", {})
        min_depth_diff = correction_config.get("min_depth_diff", 5.0)
        max_xy_diff = correction_config.get("max_xy_diff", 15.0)

        if depth_diff >= min_depth_diff:
            new_x, new_y = self._get_median_xy_from_region(
                result_2d, target_depth, self.config["point_radius"], camera=self.camera_target
            )
            if new_x is not None and new_y is not None:
                # X, Y 차이 검증
                x_diff = abs(new_x - result_3d[0])
                y_diff = abs(new_y - result_3d[1])

                if x_diff >= max_xy_diff or y_diff >= max_xy_diff:
                    self.logger.warning(
                        f"{point_name}: correction cancelled - XY diff too large "
                        f"(x_diff={x_diff:.2f}mm, y_diff={y_diff:.2f}mm, max={max_xy_diff}mm)"
                    )
                    return None

                self.logger.info(
                    f"{point_name}: corrected using depth map "
                    f"(depth_diff={depth_diff:.2f}mm, x_diff={x_diff:.2f}mm, y_diff={y_diff:.2f}mm)"
                )
                return np.array([new_x, new_y, calculated_depth])

        return None

    # stable depth 검사 기본 임계값(mm). config 미설정/축약 시 사용.
    DEFAULT_STABLE_DEPTH_RANGE = 50.0

    def _stable_range_for(self, name: str) -> Optional[float]:
        """anchor ``name``(L/R/U)에 대한 stable depth 임계값(mm)을 반환한다.
        None 이면 그 방향은 검사를 건너뛴다.

        ``config["stable_depth_range"]`` 는 두 형태를 모두 허용한다:
          - 숫자        : 전체 anchor 공통 임계값 (기존 동작)
          - 방향별 dict : ``{L: 50.0, R: 50.0, U: false}`` 처럼 방향마다 지정
              · 숫자        → 그 방향 임계값
              · false / null → 그 방향 검사 skip
              · true        → 기본 임계값(DEFAULT_STABLE_DEPTH_RANGE)으로 검사
              · dict 에 없는 방향 → 기본 임계값으로 검사
        scalar 가 false/null 이면 전체 검사 skip.
        """
        cfg = self.config.get("stable_depth_range", self.DEFAULT_STABLE_DEPTH_RANGE)
        if isinstance(cfg, dict):
            if name not in cfg:
                return self.DEFAULT_STABLE_DEPTH_RANGE
            v = cfg[name]
            if v is None or v is False:
                return None
            if v is True:
                return self.DEFAULT_STABLE_DEPTH_RANGE
            return float(v)
        if cfg is None or cfg is False:
            return None
        return float(cfg)

    def _parse_camera_calibration(
        self, raw, source: str
    ) -> Optional[np.ndarray]:
        """camera_calibration(hand-eye, 4x4) 값을 파싱한다. 없거나 형식이 잘못되면
        경고를 남기고 None 을 반환한다(검사는 직접 비교로 fallback).

        camera_calibration 은 **카메라 -> 로봇(base)** 변환이다 (= inv(T_base2cam)).
        즉 p_robot = camera_calibration @ p_cam 이므로, 역행렬 없이 그대로 적용한다.

        Args:
            raw: 4x4 중첩 리스트 / 길이 16 시퀀스 / (4,4) ndarray, 또는 None
            source: 로그용 출처 문자열 (예: "runtime config", "template")
        """
        if raw is None:
            return None
        try:
            T = as_4x4_matrix(raw)
            self.logger.info(f"camera_calibration loaded from {source} (4x4).")
            return T
        except (ValueError, TypeError) as e:
            self.logger.warning(f"camera_calibration ({source}) 파싱 실패: {e}")
            return None

    def _safe_zone_calibrations(self, runtime_calibration=None):
        """safe zone 을 로봇 프레임에서 검사하기 위한 (T_teach, T_runtime) 4x4 쌍을
        반환한다. 둘 다 있을 때만 쌍을 반환하고, 하나라도 없으면 None 을 반환해
        카메라 프레임에서 직접 비교한다(하위 호환).

        - T_teach   : 템플릿(teaching) 카메라 -> 로봇. **template_param** 에서 파싱.
                      safe_zones(템플릿 프레임)를 로봇 프레임으로 변환하는 데 사용.
        - T_runtime : 현재(매칭) 카메라 -> 로봇. **run_pipeline 의 target_camera_extrinsic
                      인자**(runtime_calibration)로 전달(우선). 없으면 모듈 config 의
                      camera_calibration(self.runtime_camera_calibration)로 fallback.
                      result_3d(현재 카메라 프레임)를 로봇 프레임으로 변환하는 데 사용.
        """
        # teaching(template) 캘리브레이션: init 시 1회 파싱해 캐싱한 값 사용
        T_teach = self.template_camera_calibration

        # 매칭(runtime) 캘리브레이션: run_pipeline 인자 우선, 없으면 config fallback
        T_runtime = self._parse_camera_calibration(
            runtime_calibration, source="run_pipeline argument"
        )
        if T_runtime is None:
            T_runtime = self.runtime_camera_calibration

        if T_teach is None and T_runtime is None:
            return None  # 캘리브레이션 미제공 -> 기존 동작(카메라 프레임 직접 비교)
        if T_teach is None or T_runtime is None:
            self.logger.warning(
                "camera_calibration 이 한쪽(teaching/runtime)만 설정됨 - 로봇 프레임 "
                "변환을 건너뛰고 카메라 프레임에서 직접 비교합니다."
            )
            return None

        return T_teach, T_runtime

    def check_safe_zones(
        self,
        result1_3d: np.ndarray,
        result2_3d: np.ndarray,
        runtime_calibration: Optional[np.ndarray] = None,
    ) -> None:
        """
        매칭으로 구한 anchor 포인트 L, R이 설정된 safe zone(회전 큐보이드, OBB)
        안에 있는지 검사하는 안전장치(safety guard).

        safe_zones 는 **템플릿(teaching) 카메라 프레임**에 고정으로 정의된 "유효한
        영역"이다. 매칭으로 추정한 포즈가 잘못되면 anchor 결과(result_3d)가 엉뚱한
        위치로 날아갈 수 있는데, 이를 모르고 로봇이 그 좌표로 이동하면 이상한 곳에
        충돌할 수 있다. 이를 막기 위해 포즈가 이미 적용된 결과(result_3d)를 safe zone
        과 비교한다.

        **로봇 프레임 검사**: teaching/runtime hand-eye 캘리브레이션(``camera_calibration``,
        4x4)이 모두 주어지면, result_3d 를 ``M = inv(T_teach) @ T_runtime`` 로 변환해
        로봇(고정·절대) 프레임 기준으로 비교한다. 카메라가 움직여도 재캘리브레이션으로
        ``camera_calibration`` 만 갱신하면 검사가 그대로 유효하다. 캘리브레이션이 없으면
        카메라 프레임에서 직접 비교한다(하위 호환).

        포인트가 zone 을 벗어나면 MatcherError 를 발생시켜, depth 계산 실패와 동일하게
        매칭 실패로 처리되도록 한다. safe zone 은 L, R 두 포인트에만 정의되어 있으며
        U 포인트는 검사하지 않는다. template_param 에 safe_zones 가 없으면 검사를
        건너뛴다(하위 호환).

        Args:
            result1_3d: Point L 의 3D 좌표 [x, y, z] (현재 카메라 프레임)
            result2_3d: Point R 의 3D 좌표 [x, y, z] (현재 카메라 프레임)
            runtime_calibration: 현재(매칭) 카메라 hand-eye (cam->robot, 4x4).
                run_pipeline 의 target_camera_extrinsic 인자로 전달된다. None 이면
                모듈 config 의 camera_calibration 로 fallback. (teaching 캘리브레이션은
                template_param 에서 읽는다.)
        """
        if not self.template_param:
            return

        # safe_zones 찾기: template_param 최상위 -> template_param.matching_model
        safe_zones = (
            self.template_param.get("safe_zones")
            or self.template_param.get("matching_model", {}).get("safe_zones")
        )
        if not safe_zones:
            self.logger.debug("No safe_zones configured - skipping safe zone check.")
            return

        # 로봇 프레임 검사용 캘리브레이션 쌍 (둘 다 있을 때만, 아니면 None=직접 비교)
        calibrations = self._safe_zone_calibrations(runtime_calibration)
        if calibrations is not None:
            self.logger.debug("Safe zone check in robot frame (hand-eye calibration applied).")

        # L, R 두 포인트만 각자의 safe zone에 대해 검사 (U는 검사 제외)
        for name, point_3d in (("L", result1_3d), ("R", result2_3d)):
            zone = safe_zones.get(name)
            if not zone:
                self.logger.debug(f"No safe zone for point {name} - skipping.")
                continue

            zone_min = np.array(zone["min"], dtype=float)
            zone_max = np.array(zone["max"], dtype=float)
            euler = np.array(zone["euler"], dtype=float)

            if calibrations is None:
                # 캘리브레이션 미제공: 카메라 프레임에서 직접 비교 (하위 호환)
                inside = is_point_in_safe_zone(point_3d, zone_min, zone_max, euler)
            else:
                # 다이어그램 그대로: 양쪽을 로봇 프레임으로 모은 뒤 OBB 내부 판정
                #  - result_3d(현재 카메라 프레임) -> 로봇:  T_runtime
                #  - safe_zone(템플릿 카메라 프레임) -> 로봇: T_teach
                T_teach, T_runtime = calibrations
                point_robot = transform_point_3d(point_3d, T_runtime)
                center_r, half_r, R_r = transform_safe_zone(
                    zone_min, zone_max, euler, T_teach
                )
                inside = is_point_in_obb(point_robot, center_r, half_r, R_r)

            if not inside:
                # 리포트 좌표는 원본(현재 카메라 프레임) result_3d 를 그대로 사용
                point = np.asarray(point_3d).tolist()
                raise MatcherError(
                    ErrorCode.SAFE_ZONE_VIOLATION,
                    f"point {name} {point} is outside its safe zone",
                    details={"point": name, "position": point},
                )
            self.logger.debug(f"Safe zone check passed for point {name}.")

    def _save_result_json(
        self,
        *,
        anchors: Dict[str, np.ndarray],
        runtime_calibration: Optional[np.ndarray],
        ok: bool,
        failed_at: Optional[str] = None,
        error_detail: Optional[str] = None,
        violation: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """rerun 디버그 시각화(core/utils/rerun_viz.py)용 ``{stem}_result.json`` 저장.

        파이프라인 동작에는 영향을 주지 않는 보조 산출물이다(실패해도 무시). anchor 와
        safe_zone 은 **카메라 프레임** 좌표 그대로 저장하고, hand-eye 캘리브레이션을 함께
        남겨 뷰어가 로봇 프레임으로 변환·비교할 수 있게 한다.

        Args:
            anchors: {"L","R","U": [x,y,z]} 현재(매칭) 카메라 프레임 anchor 3D 좌표
            runtime_calibration: run_pipeline 인자로 받은 현재 카메라 hand-eye(cam->robot)
            ok: 성공 여부
            failed_at: 실패 단계 이름(ErrorCode.name 등)
            error_detail: 실패 상세 메시지
            violation: safe zone 위반 정보 {"point","position"}
        """
        if self.config.get("save_essential", "none") == "none":
            return None
        try:
            import json

            safe_zones = (
                self.template_param.get("safe_zones")
                or self.template_param.get("matching_model", {}).get("safe_zones")
            ) if self.template_param else None

            # teaching: init 시 캐싱한 값 / runtime: 인자 우선, 없으면 config fallback
            T_teach = self.template_camera_calibration
            T_runtime = self._parse_camera_calibration(
                runtime_calibration, source="run_pipeline argument"
            )
            if T_runtime is None:
                T_runtime = self.runtime_camera_calibration

            def _mat(m):
                return np.asarray(m, dtype=float).tolist() if m is not None else None

            K = None
            if self.camera_target is not None:
                K = self.camera_target.get_intrinsic_matrix().tolist()

            result = {
                "frame_stem": self.target_texture_name,
                "ok": bool(ok),
                "failed_at": failed_at,
                "error_detail": error_detail,
                "anchors": {
                    name: (np.asarray(pos, dtype=float).tolist() if pos is not None else None)
                    for name, pos in anchors.items()
                },
                "safe_zones": safe_zones,
                "safe_zone_violation": violation,
                "camera_target_K": K,
                "T_teach_cam_to_base": _mat(T_teach),
                "T_runtime_cam_to_base": _mat(T_runtime),
                "inputs": {
                    "depth": getattr(self, "_last_target_depth_path", None),
                    "texture": getattr(self, "_last_target_texture_path", None),
                },
            }

            out_path = self.output_path / f"{self.target_texture_name}_result.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            self.logger.debug(f"Result json saved: {out_path}")
            return str(out_path)
        except Exception as e:
            self.logger.warning(f"Result json 저장 실패(무시): {e}")
            return None

    def run_pipeline(
        self,
        target_texture: Optional[np.ndarray] = None,
        target_depth: Optional[np.ndarray] = None,
        source_image: Optional[np.ndarray] = None,
        target_camera_path: Optional[str] = None,
        target_texture_path: Optional[str] = None,
        target_depth_path: Optional[str] = None,
        target_camera_extrinsic: Optional[np.ndarray] = None,
        output_dir: Optional[str] = None,
        
    ) -> MatchResult:
        """
        전체 파이프라인 실행

        Args:
            target_texture: Target texture 이미지 (매칭용) 
            target_depth: Target depth 이미지 (depth 계산용) 
            source_image: Source 이미지
            target_camera_path: Target camera parameter 경로
            target_texture_path: Target texture 이미지 경로 (debug mode 사용 시 사용)
            target_depth_path: Target depth 이미지 경로 (debug mode 사용 시 사용)
            output_dir: 출력 디렉토리

        Returns:
            MatchResult. 파이프라인 처리 중의 실패(매칭/depth/safe zone)는 예외 대신
            결과 객체로 반환한다. (초기 인자/config 검증 단계의 비정상 입력은 예외 가능.)
            - 성공: success=True, point_l/point_r/point_u/plane_normal 채워짐
            - 실패: success=False, error_code(ErrorCode)/error_message/details 채워짐
              (safe zone 위반 시 error_code=SAFE_ZONE_VIOLATION,
               details={"point", "position"}; depth 실패는 DEPTH_CALCULATION_FAILED /
               DEPTH_OUT_OF_RANGE; 그 외는 MATCH_FAILED)
        """
        try:
            # 경로 설정 (셋업도 try 안에서 처리 → 실패 시 예외 대신 MatchResult 반환)
            target_depth_path = target_depth_path or self.config.get("target_depth_path")
            if target_depth_path is None:
                raise MatcherError(
                    ErrorCode.INVALID_PARAM,
                    "target_depth_path is required (pass as argument or set in config)",
                )
            if target_texture_path is None or target_texture is None:
                target_texture_path = target_depth_path
            output_dir = output_dir or self.config["output_dir"]

            self.logger.debug(f"Target texture: {target_texture_path}")
            self.logger.debug(f"Target depth: {target_depth_path}")
            self.logger.debug(f"Target camera path: {target_camera_path}")
            self.logger.debug(f"Output directory: {output_dir}")

            # for output path
            self.target_texture_name = Path(target_texture_path).stem
            self.target_depth_name = Path(target_depth_path).stem
            # rerun 디버그 시각화(result.json)에서 점군 backproject 에 쓸 입력 경로 보관
            self._last_target_depth_path = target_depth_path
            self._last_target_texture_path = target_texture_path
            self.output_path = Path(output_dir)
            if self.config["save_essential"] != "none":
                self.output_path.mkdir(exist_ok=True)

            if target_camera_path is not None:
                target_camera_config = load_photoneo_camera_config(target_camera_path)
                self.camera_target = create_camera_from_yaml_config(target_camera_config)

            result1_3d = None
            result2_3d = None
            result3_3d = None
            plane_normal = None

            if "roi_2d_src" in self.config and self.config["roi_2d_src"] is not None:
                source_image = apply_roi_mask(source_image, self.config["roi_2d_src"])

            if self.config["image_undistortion"]:
                target_depth = self.camera_target.undistort_image(target_depth)

            if target_texture is not None:
                target_image = target_texture
                if self.config["image_undistortion"]:
                    target_image = self.camera_target.undistort_image(target_image)
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
            if self.config["image_undistortion"]:
                source_image = self.camera_source.undistort_image(source_image)

            def get_point_by_config(selected_points, point_name):
                return np.array(
                    [
                        selected_points[point_name]["x"],
                        selected_points[point_name]["y"],
                        selected_points[point_name]["z"],
                    ]
                )

            # selected_points 찾기: template_param 최상위 -> template_param.matching_model
            selected_points = (
                self.template_param.get("selected_points")
                or self.template_param.get("matching_model", {}).get("selected_points")
            )
            if selected_points is None or not selected_points:
                raise MatcherError(
                    ErrorCode.INVALID_PARAM,
                    "selected_points not set in template_param",
                )
            anchor_point1_3d = get_point_by_config(selected_points, "L")
            anchor_point2_3d = get_point_by_config(selected_points, "R")
            anchor_point3_3d = get_point_by_config(selected_points, "U")
            plane_normal = compute_plane_normal(
                anchor_point1_3d, anchor_point2_3d, anchor_point3_3d
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
                    contrast=self.config["result_image_contrast"],
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
                    contrast=self.config["result_image_contrast"],
                )

            if self.config["enable_3d_matching"]:
                time_start = time.time()
                source_depth = source_image  # 원본 depth 이미지 사용 (RGB 변환 전)
                result = self.run_matching_3d(
                    filtered_matches, depth_target=target_depth, depth_source=source_depth
                )
                time_end = time.time()
                self.logger.info(
                    f"3D matching time: {time_end - time_start:.3f} seconds"
                )
                if result is None:
                    raise Exception("3D matching failed")

                selected_points = self.template_param.get("selected_points", {})
                if selected_points is None:
                    raise MatcherError(
                        ErrorCode.INVALID_PARAM,
                        "selected_points not set in template_param",
                    )

                transform_matrix = result["transformation"]

                # 3D 포인트를 homogeneous coordinate로 변환 (4x1) 후 변환 적용
                def apply_transform_3d(point_3d, transform_4x4):

                    point_homo = np.append(point_3d, 1.0)
                    transformed_homo = transform_4x4 @ point_homo
                    return transformed_homo[:3]

                result1_3d = apply_transform_3d(anchor_point1_3d, transform_matrix)
                result2_3d = apply_transform_3d(anchor_point2_3d, transform_matrix)
                result3_3d = apply_transform_3d(anchor_point3_3d, transform_matrix)

                # Project 3D points to 2D and convert to integer coordinates
                result1_2d = project_3d_point_to_2d(
                    result1_3d, self.camera_target.get_intrinsic_matrix()
                ).astype(int)
                result2_2d = project_3d_point_to_2d(
                    result2_3d, self.camera_target.get_intrinsic_matrix()
                ).astype(int)
                result3_2d = project_3d_point_to_2d(
                    result3_3d, self.camera_target.get_intrinsic_matrix()
                ).astype(int)

                calculated_depths = self.calculate_anchor_depth(
                    target_depth_path=target_depth_path,
                    target_depth=target_depth,
                    point1_2d=result1_2d,
                    point2_2d=result2_2d,
                    point3_2d=result3_2d,
                    radius=self.config["point_radius"],
                )
                # Depth 계산 결과 검증
                if calculated_depths is None:
                    raise MatcherError(
                        ErrorCode.DEPTH_CALCULATION_FAILED,
                        "Anchor depth calculation failed: None",
                    )

                if any(d is None for d in calculated_depths):
                    raise MatcherError(
                        ErrorCode.DEPTH_CALCULATION_FAILED,
                        f"Anchor depth calculation failed: "
                        f"L={calculated_depths[0]}, R={calculated_depths[1]}, U={calculated_depths[2]}",
                        details={
                            "depths": {
                                "L": calculated_depths[0],
                                "R": calculated_depths[1],
                                "U": calculated_depths[2],
                            }
                        },
                    )

                # 앵커 포인트 데이터 구성
                anchor_points = [
                    {
                        "name": "L",
                        "pos_3d": result1_3d,
                        "pos_2d": result1_2d,
                        "depth": calculated_depths[0],
                    },
                    {
                        "name": "R",
                        "pos_3d": result2_3d,
                        "pos_2d": result2_2d,
                        "depth": calculated_depths[1],
                    },
                    {
                        "name": "U",
                        "pos_3d": result3_3d,
                        "pos_2d": result3_2d,
                        "depth": calculated_depths[2],
                    },
                ]

                # Depth 안정성 검사 및 보정 적용 (검사는 방향별 on/off 가능)
                for anchor in anchor_points:
                    depth_diff = abs(anchor["pos_3d"][2] - anchor["depth"])

                    stable_range = self._stable_range_for(anchor["name"])
                    if stable_range is None:
                        self.logger.debug(
                            f"Stable depth check disabled for {anchor['name']} "
                            f"(diff={depth_diff:.1f}mm)"
                        )
                    elif depth_diff > stable_range:
                        raise MatcherError(
                            ErrorCode.DEPTH_OUT_OF_RANGE,
                            f"Out of stable depth range {anchor['name']}: "
                            f"{depth_diff:.1f}mm > {stable_range}mm",
                            details={
                                "point": anchor["name"],
                                "depth_diff": float(depth_diff),
                                "stable_range": float(stable_range),
                            },
                        )
                    else:
                        self.logger.debug(
                            f"Stable depth {anchor['name']}: diff={depth_diff:.1f}mm <= {stable_range}mm"
                        )

                    # Depth 보정 적용 (stable 검사 skip 여부와 무관)
                    depth_correction = self.config.get("depth_correction", {})
                    if depth_correction and depth_correction.get("enabled", True):
                        corrected = self._apply_depth_correction(
                            anchor["pos_3d"],
                            anchor["pos_2d"],
                            anchor["depth"],
                            depth_diff,
                            target_depth,
                            f"{anchor['name']} point",
                        )
                        if corrected is not None:
                            anchor["pos_3d"][:] = corrected

                # 보정된 좌표 추출
                result1_3d, result2_3d, result3_3d = [
                    a["pos_3d"] for a in anchor_points
                ]

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
                    camera=self.camera_target,
                )
                time_end = time.time()
                self.logger.info(
                    f"3D points calculation time: {time_end - time_start:.3f} seconds"
                )

            self.logger.debug(
                f"3D points: pointL: {result1_3d}, pointR: {result2_3d}, pointU: {result3_3d}"
            )

            # Safe zone 안전장치: 포즈가 적용된 anchor 결과 L, R이 템플릿/월드 좌표에
            # 고정된 safe zone(회전 큐보이드) 안에 있는지 확인하고, 벗어나면(=매칭/포즈
            # 추정이 잘못되어 엉뚱한 위치) 매칭 실패로 처리한다. (depth 계산 실패와 동일)
            # 위반 시에도 rerun 디버그용 result.json 을 남긴 뒤 실패를 전파한다.
            anchors_cam = {"L": result1_3d, "R": result2_3d, "U": result3_3d}
            try:
                self.check_safe_zones(
                    result1_3d, result2_3d, runtime_calibration=target_camera_extrinsic
                )
            except MatcherError as e:
                self._save_result_json(
                    anchors=anchors_cam,
                    runtime_calibration=target_camera_extrinsic,
                    ok=False,
                    failed_at=e.error_code.name,
                    error_detail=e.message,
                    violation=e.details if e.error_code == ErrorCode.SAFE_ZONE_VIOLATION else None,
                )
                raise

            # 성공 경로: rerun 디버그용 result.json 저장(파이프라인 동작에는 영향 없음)
            self._save_result_json(
                anchors=anchors_cam,
                runtime_calibration=target_camera_extrinsic,
                ok=True,
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
                    transform_matrix=transform_matrix,
                    target_texture=target_clipped,
                    target_depth=target_depth,
                    source_image=source_image,
                    plane_normal=plane_normal,
                    result3d=result_3d_points,
                    ransac_result=filtered_matches,
                    camera=self.camera_target,
                    result_image_name=self.target_texture_name,
                )

            self.logger.info("\n=== Pipeline completed ===")

            return MatchResult.ok(result1_3d, result2_3d, result3_3d, plane_normal)

        except MatcherError as e:
            # 코드를 가진 도메인 실패(safe zone 위반, 깊이 계산 실패 등)를
            # 결과 객체로 변환해 반환한다. (예외를 외부로 던지지 않음)
            self.logger.error(str(e))
            return MatchResult.fail(e.error_code, e.message, e.details)
        except KeyError as e:
            # 설정/파라미터 키 누락은 매칭 실패와 구분해 INVALID_PARAM 으로 분류한다.
            # (어떤 키가 없는지 메시지에 남겨 진단을 돕는다.)
            self.logger.error(f"Missing required config/parameter key: {e}")
            return MatchResult.fail(
                ErrorCode.INVALID_PARAM, f"missing required config/parameter key: {e}"
            )
        except Exception as e:
            # 예상치 못한 실패(2D/3D 매칭·필터링 등)는 일반 코드(MATCH_FAILED)로
            # 묶어 결과 객체로 반환한다.
            self.logger.error(f"Matching failed: {e}")
            return MatchResult.fail(ErrorCode.MATCH_FAILED, str(e))

    def calculate_3d_points(
        self, result1_3d: np.ndarray, result2_3d: np.ndarray, result3_3d: np.ndarray, camera: Camera
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
        backprojected1_3d = self._backproject_to_3d(result1_3d, camera=camera)
        backprojected2_3d = self._backproject_to_3d(result2_3d, camera=camera)
        backprojected3_3d = self._backproject_to_3d(result3_3d, camera=camera)

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
            keypoints_target = filtered_matches["filtered_kpts0"].astype(np.int32)
            keypoints_source = filtered_matches["filtered_kpts1"].astype(np.int32)

            valid_indices = []
            points_3d_target = []
            points_3d_source = []

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

            for i, (pt0, pt1) in enumerate(zip(keypoints_target, keypoints_source)):
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

                    x3d_0 = (x0 - self.camera_target.K[0, 2]) * z0 / self.camera_target.K[0, 0]
                    y3d_0 = (y0 - self.camera_target.K[1, 2]) * z0 / self.camera_target.K[1, 1]

                    # intrinsic matrix of source camera is used only for 3d matching
                    x3d_1 = (x1 - self.camera_source.K[0, 2]) * z1 / self.camera_source.K[0, 0]
                    y3d_1 = (y1 - self.camera_source.K[1, 2]) * z1 / self.camera_source.K[1, 1]

                    points_3d_target.append([x3d_0, y3d_0, z0])
                    points_3d_source.append([x3d_1, y3d_1, z1])
                    valid_indices.append(i)

                else:
                    boundary_fail_count += 1

            # 통계 정보 로깅
            self.logger.debug(f"Point processing stats:")
            self.logger.debug(f"  - Total points: {len(keypoints_target)}")
            self.logger.debug(f"  - Valid 3D points: {valid_count}")

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

            pcd_source_transformed = copy.deepcopy(pcd_source)

            pcd_source_transformed.transform(pose)

            # 3D correspondence 디버그 플롯 (옵션). pose 를 적용한 source 대응점과
            # target 대응점을 같은 프레임에서 겹쳐 그려 정합 품질을 본다. matplotlib
            # 렌더 비용이 있어 config 플래그(save_correspondence_plot)로 opt-in 한다.
            if self.config.get("save_correspondence_plot", False) and self.config.get(
                "save_essential", "none"
            ) != "none":
                try:
                    ref_corr = points_3d_target                       # target 프레임
                    src_corr = np.asarray(pcd_source_transformed.points)  # pose 적용된 source
                    # inlier/fitness/rmse 를 max_correspondence_distance 기준으로 산출(제목용).
                    thr = float(self.config.get("ransac_3d", {}).get(
                        "max_correspondence_distance", 3.0))
                    d = np.linalg.norm(src_corr - ref_corr, axis=1)
                    inl = d < thr
                    n_inl = int(inl.sum())
                    rmse = float(d[inl].mean()) if n_inl else float("nan")
                    fitness = float(inl.mean()) if len(d) else 0.0
                    corr_path = str(
                        self.output_path / f"{self.target_texture_name}_correspondences.png"
                    )
                    visualize_3d_correspondences(
                        ref_corr, src_corr, corr_path,
                        max_points=len(ref_corr), line_step=0,   # 전부, 선 없음
                        title=(f"3D correspondences ({n_inl} inliers, "
                               f"rmse={rmse:.2f}mm fitness={fitness:.2f})"),
                    )
                except Exception as e:
                    self.logger.warning(f"correspondence plot 저장 실패(무시): {e}")

            # 8. 결과 반환
            return {
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
        if hasattr(self, "camera_target") and self.camera_target is not None:
            self.logger.debug("Camera object cleanup in progress...")
            del self.camera_target
            self.camera_target = None
        if hasattr(self, "camera_source") and self.camera_source is not None:
            self.logger.debug("Camera object cleanup in progress...")
            del self.camera_source
            self.camera_source = None

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
