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
    normal_to_angles,
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
from .geometry_utils import (
    se3_to_homography,
    homography_to_se3,
    decompose_homography,
    apply_homography,
    compute_homography_from_correspondences,
    validate_homography,
    normalize_points,
    robust_homography_estimation,
    project_pcd_to_image,
    project_pcd_to_depth_image,
    project_pcd_to_images,
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
    "normal_to_angles",
    "point_cloud_to_depth_map",
    "find_3d_from_2d_depthmap_robust",
    "find_3d_from_2d_depthmap",
    "find_depth_from_2d_robust",
    "get_pixels_in_radius",
    "depth_estimation_mad",
    "depth_estimation_histogram",
    "registration_ransac_based_on_correspondence",
    "solve_rigid_transform_between_points",
    "se3_to_homography",
    "homography_to_se3",
    "decompose_homography",
    "apply_homography",
    "compute_homography_from_correspondences",
    "validate_homography",
    "normalize_points",
    "robust_homography_estimation",
    "project_pcd_to_image",
    "project_pcd_to_depth_image",
    "project_pcd_to_images",
]

MODEL_REPO_ID = "Realcat/imcui_checkpoints"
