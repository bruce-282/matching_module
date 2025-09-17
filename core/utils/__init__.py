"""
Utilities module for the matching functionality
"""

from .image_utils import load_image, resize_image, normalize_image, read_image
from .viz_utils import visualize_matches, visualize_keypoints, warp_images
from .processing_utils import (
    filter_matches,
    compute_geometry,
    proc_ransac_matches,
    set_null_pred,
    registration_ransac_based_on_correspondence,
    solve_rigid_transform_between_points,
)
from .pcd_utils import (
    load_ply_as_image,
    is_ply_file,
    # get_image_from_file,
    PointCloudToImageConverter,
)
from .depth_utils import (
    point_cloud_to_depth_map,
    find_3d_from_2d_depthmap_robust,
    find_3d_from_2d_depthmap,
    find_depth_from_2d_robust,
    get_pixels_in_radius,
    depth_estimation_mad,
    depth_estimation_histogram,
)

__all__ = [
    "load_image",
    "resize_image",
    "normalize_image",
    "read_image",
    "visualize_matches",
    "warp_images",
    "visualize_keypoints",
    "filter_matches",
    "compute_geometry",
    "proc_ransac_matches",
    "set_null_pred",
    "load_ply_as_image",
    "is_ply_file",
    "PointCloudToImageConverter",
    "point_cloud_to_depth_map",
    "find_3d_from_2d_depthmap_robust",
    "find_3d_from_2d_depthmap",
    "find_depth_from_2d_robust",
    "get_pixels_in_radius",
    "depth_estimation_mad",
    "depth_estimation_histogram",
    "registration_ransac_based_on_correspondence",
    "solve_rigid_transform_between_points",
]

MODEL_REPO_ID = "Realcat/imcui_checkpoints"
