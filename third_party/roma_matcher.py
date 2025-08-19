import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
import logging
import sys
from pathlib import Path
from PIL import Image
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RomaMatcher:
    """
    RoMA implementation for image matching.
    Uses the actual RoMA model from the existing codebase.
    Supports both 'roma' and 'minima_roma' configurations.
    """

    def __init__(self, device: str = "auto", model_name: str = "roma_outdoor.pth"):
        """
        Initialize RoMA matcher.

        Args:
            device: Device to use ('auto', 'cuda', 'cpu')
            model_name: Model file name ('roma_outdoor.pth', 'minima_roma.pth', etc.)
        """
        self.device = self._get_device(device)
        self.model_name = model_name
        self.logger = logger
        self.logger.info(
            f"RoMA matcher initialized on device: {self.device} with model: {self.model_name}"
        )

        # Configuration from roma/minima_roma config
        self.config = {
            "model_name": model_name,
            "model_utils_name": "dinov2_vitl14_pretrain.pth",
            "max_keypoints": 2000,
            "match_threshold": 0.2,
            "coarse_res": (560, 560),
            "upsample_res": (864, 1152),
            "preprocessing": {
                "grayscale": False,
                "force_resize": True,
                "resize_max": 1024,
                "width": 320,
                "height": 240,
                "dfactor": 8,
            },
        }

        # Initialize RoMA model
        self._load_roma_model()

    def _get_device(self, device: str) -> str:
        """Get the appropriate device for computation."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def _load_roma_model(self):
        """Load the actual RoMA model from files."""
        # Add RoMA path to sys.path (using the copied version in utils/)
        roma_path = Path(__file__).parent / "RoMa"
        if not roma_path.exists():
            raise FileNotFoundError(f"RoMA model path not found: {roma_path}")

        sys.path.append(str(roma_path))

        # Import RoMA model components
        try:
            from romatch.models.model_zoo import roma_model
        except ImportError as e:
            raise ImportError(f"Could not import RoMA model components: {e}")

        # Model configuration parameters (same as original Roma class)
        model_name = self.config["model_name"]
        model_utils_name = self.config["model_utils_name"]
        coarse_res = self.config["coarse_res"]
        upsample_res = self.config["upsample_res"]
        max_keypoints = self.config["max_keypoints"]

        # Try to find model files in common locations
        model_path = None
        dinov2_weights_path = None

        # Check weights directory first (user's preferred location)
        weights_dir = Path("weights")
        if weights_dir.exists():
            model_path = weights_dir / model_name
            dinov2_weights_path = weights_dir / model_utils_name

            if model_path.exists():
                self.logger.info(f"Found model file in weights directory: {model_path}")
            else:
                self.logger.warning(
                    f"Model file not found in weights directory: {model_path}"
                )
                model_path = None

        # Check workspace directory as fallback
        if model_path is None or not model_path.exists():
            workspace_dir = Path("workspace")
            if workspace_dir.exists():
                model_path = workspace_dir / model_name
                dinov2_weights_path = workspace_dir / model_utils_name

                if model_path.exists():
                    self.logger.info(
                        f"Found model file in workspace directory: {model_path}"
                    )
                else:
                    self.logger.warning(
                        f"Model file not found in workspace directory: {model_path}"
                    )
                    self.logger.info("Available files in workspace/ directory:")
                    for file in workspace_dir.iterdir():
                        if file.is_file():
                            self.logger.info(f"  - {file.name}")
                    model_path = None

        # If not found in workspace, try to download or use default paths
        if model_path is None or not model_path.exists():
            self.logger.warning(
                "Model files not found, trying to use default RoMA model"
            )
            # Use roma_model without weights (will use default initialization)
            if self.device == "cuda":
                amp_dtype = torch.float16
            else:
                amp_dtype = torch.float32

            self.roma_model = roma_model(
                resolution=coarse_res,
                upsample_preds=True,
                device=self.device,
                amp_dtype=amp_dtype,
            )
            self.roma_model.upsample_res = upsample_res
            self.logger.info("RoMA model loaded with default initialization")
            return

        # Load model weights
        try:
            weights = torch.load(model_path, map_location="cpu")

            # Handle different weight formats
            if model_name == "gim_roma_100h.ckpt":
                if "state_dict" in weights.keys():
                    weights = weights["state_dict"]
                for k in list(weights.keys()):
                    if k.startswith("model."):
                        weights[k.replace("model.", "", 1)] = weights.pop(k)

            # Load DINOv2 weights if available
            dinov2_weights = None
            if dinov2_weights_path.exists():
                dinov2_weights = torch.load(dinov2_weights_path, map_location="cpu")

            # Set amp_dtype based on device
            if self.device == "cuda":
                amp_dtype = torch.float16
            else:
                amp_dtype = torch.float32

            # Create model with weights
            self.roma_model = roma_model(
                resolution=coarse_res,
                upsample_preds=True,
                weights=weights,
                dinov2_weights=dinov2_weights,
                device=self.device,
                amp_dtype=amp_dtype,
            )
            self.roma_model.upsample_res = upsample_res

            self.logger.info(f"RoMA model loaded successfully from {model_path}")

        except Exception as e:
            self.logger.error(f"Error loading RoMA model weights: {e}")
            raise

    def match_images(
        self,
        image0: np.ndarray,
        image1: np.ndarray,
        match_threshold: float = 0.2,
        max_keypoints: int = 2000,
        ransac_method: str = "CV2_USAC_MAGSAC",
        ransac_reproj_threshold: float = 10.0,
        ransac_confidence: float = 0.9999,
        ransac_max_iter: int = 10000,
        geometry_type: str = "Homography",
    ) -> Dict[str, Any]:
        """
        Perform image matching using Minima(RoMa)-like approach.

        Args:
            image0: First image (RGB format)
            image1: Second image (RGB format)
            match_threshold: Matching threshold (default: 0.2 for Minima(RoMa))
            max_keypoints: Maximum number of keypoints (default: 2000 for Minima(RoMa))
            ransac_method: RANSAC method to use
            ransac_reproj_threshold: RANSAC reprojection threshold
            ransac_confidence: RANSAC confidence
            ransac_max_iter: RANSAC maximum iterations
            geometry_type: Type of geometry to estimate ('Homography' or 'Fundamental')

        Returns:
            Dictionary containing matching results
        """
        try:
            self.logger.info("Starting Minima(RoMa) image matching...")

            # Use actual RoMA model
            return self._match_with_roma(image0, image1, max_keypoints)

        except Exception as e:
            self.logger.error(f"Error in Minima(RoMa) image matching: {e}")
            raise

    def _match_with_roma(
        self, image0: np.ndarray, image1: np.ndarray, max_keypoints: int
    ) -> Dict[str, Any]:
        """Match images using the actual RoMA model."""
        try:
            # Convert numpy arrays to PIL Images (like in the original implementation)
            img0 = Image.fromarray(image0.astype("uint8"))
            img1 = Image.fromarray(image1.astype("uint8"))
            W_A, H_A = img0.size
            W_B, H_B = img1.size

            # Match using RoMA model
            warp, certainty = self.roma_model.match(img0, img1, device=self.device)

            # Sample matches for estimation
            matches, certainty = self.roma_model.sample(
                warp, certainty, num=max_keypoints
            )

            # Convert to pixel coordinates
            kpts1, kpts2 = self.roma_model.to_pixel_coordinates(
                matches, H_A, W_A, H_B, W_B
            )

            # Convert to numpy arrays for further processing
            kpts1 = kpts1.cpu().numpy()
            kpts2 = kpts2.cpu().numpy()
            certainty = certainty.cpu().numpy()

            # Create result structure
            result = {
                "boxes": self._create_bounding_boxes(kpts1, kpts2),
                "matches": [[i, i] for i in range(len(kpts1))],
                "geometry": self._estimate_geometry(kpts1, kpts2),
                "warped_image": None,  # Could be implemented later
                "num_matches": len(kpts1),
                "matched_keypoints_0": kpts1.tolist(),
                "matched_keypoints_1": kpts2.tolist(),
                "certainty": certainty.tolist(),
                "algorithm": "Minima(RoMa)",
                "config": self.config,
                "method": "roma_model",
            }

            self.logger.info(
                f"RoMA model matching completed. Found {len(kpts1)} matches."
            )
            return result

        except Exception as e:
            self.logger.error(f"Error in RoMA model matching: {e}")
            raise

    def _estimate_geometry(
        self, kpts1: np.ndarray, kpts2: np.ndarray
    ) -> Dict[str, Any]:
        """Estimate geometry from RoMA keypoints using default RANSAC settings."""
        try:
            # Use default RANSAC settings for RoMA
            ransac_method = self._get_ransac_method("CV2_USAC_MAGSAC")

            # Try homography first
            H, mask = cv2.findHomography(
                kpts1,
                kpts2,
                method=ransac_method,
                ransacReprojThreshold=10.0,
                confidence=0.9999,
                maxIters=10000,
            )

            if H is not None:
                return {
                    "type": "homography",
                    "matrix": H.tolist(),
                    "mask": mask.tolist() if mask is not None else None,
                    "inliers": int(np.sum(mask)) if mask is not None else 0,
                }

            # If homography fails, try fundamental matrix
            F, mask = cv2.findFundamentalMat(
                kpts1,
                kpts2,
                method=ransac_method,
                ransacReprojThreshold=10.0,
                confidence=0.9999,
                maxIters=10000,
            )

            if F is not None:
                return {
                    "type": "fundamental",
                    "matrix": F.tolist(),
                    "mask": mask.tolist() if mask is not None else None,
                    "inliers": int(np.sum(mask)) if mask is not None else 0,
                }

            return {"type": "none", "matrix": None, "mask": None, "inliers": 0}

        except Exception as e:
            self.logger.error(f"Error estimating geometry from RoMA: {e}")
            raise

    def _get_ransac_method(self, method_str: str) -> int:
        """Convert RANSAC method string to OpenCV constant."""
        method_map = {
            "CV2_RANSAC": cv2.RANSAC,
            "CV2_USAC_MAGSAC": cv2.USAC_MAGSAC,
            "CV2_USAC_DEFAULT": cv2.USAC_DEFAULT,
            "CV2_USAC_FM_8PTS": cv2.USAC_FM_8PTS,
            "CV2_USAC_PROSAC": cv2.USAC_PROSAC,
            "CV2_USAC_FAST": cv2.USAC_FAST,
        }
        return method_map.get(method_str, cv2.USAC_MAGSAC)

    def _create_bounding_boxes(
        self, kp0: np.ndarray, kp1: np.ndarray
    ) -> List[List[float]]:
        """Create bounding boxes around matched keypoints."""
        boxes = []

        # Create boxes for each matched keypoint pair
        for i in range(len(kp0)):
            x0, y0 = kp0[i]
            x1, y1 = kp1[i]

            # Create a small box around each keypoint
            box_size = 20
            box0 = [x0 - box_size, y0 - box_size, x0 + box_size, y0 + box_size]
            box1 = [x1 - box_size, y1 - box_size, x1 + box_size, y1 + box_size]

            boxes.extend([box0, box1])

        return boxes


# Example usage and testing
if __name__ == "__main__":
    # Test the Minima(RoMa) matcher
    matcher = RomaMatcher()

    path_source = "tmp/source.png"
    path_target = "tmp/target.png"

    image0 = cv2.imread(path_source)
    image1 = cv2.imread(path_target)

    # Check if images were loaded successfully
    if image0 is None:
        print(f"Error: Could not load {path_source}")
        exit(1)
    if image1 is None:
        print(f"Error: Could not load {path_target}")
        exit(1)

    # Perform matching
    result = matcher.match_images(image0, image1)

    print("Minima(RoMa) Matching result:")
    print(f"Number of matches: {result['num_matches']}")
    print(f"Geometry type: {result['geometry']['type']}")
    print(f"Geometry matrix: {result['geometry']['matrix']}")
    print(f"Number of inliers: {result['geometry']['inliers']}")
    print(f"Algorithm: {result['algorithm']}")
    print(f"Method: {result['method']}")
