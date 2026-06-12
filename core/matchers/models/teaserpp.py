#!/usr/bin/env python3
"""
TEASER++ Point Cloud Registration Model
"""

import teaserpp_python
import open3d as o3d
import numpy as np
from typing import Dict, Any, Optional
from .base_model import BaseModel


def pcd2xyz(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """Convert Open3D point cloud to numpy array"""
    return np.asarray(pcd.points).T


def Rt2T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Convert rotation matrix and translation vector to 4x4 transformation matrix"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T


# def find_correspondences(source_fpfh: np.ndarray, target_fpfh: np.ndarray, mutual_filter: bool = True) -> tuple:
#     """Find correspondences between FPFH features"""
#     # Simple nearest neighbor search (can be improved with more sophisticated methods)
#     from sklearn.neighbors import NearestNeighbors
    
#     # Build k-d tree for target features
#     nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target_fpfh)
    
#     # Find nearest neighbors
#     distances, indices = nbrs.kneighbors(source_fpfh)
    
#     # Filter by distance threshold
#     distance_threshold = 0.1  # Adjust as needed
#     valid_mask = distances.flatten() < distance_threshold
    
#     corrs_source = np.where(valid_mask)[0]
#     corrs_target = indices[valid_mask].flatten()
    
#     if mutual_filter:
#         # Mutual consistency check
#         nbrs_source = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(source_fpfh)
#         distances_rev, indices_rev = nbrs_source.kneighbors(target_fpfh[corrs_target])
        
#         mutual_mask = indices_rev.flatten() == corrs_source
#         corrs_source = corrs_source[mutual_mask]
#         corrs_target = corrs_target[mutual_mask]
    
#     return corrs_source, corrs_target


def get_teaser_solver(noise_bound: float = 0.05) -> teaserpp_python.RobustRegistrationSolver:
    """Create TEASER++ solver with default parameters"""
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.inlier_selection_mode = \
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_tim_graph = \
        teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = \
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-12
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver


class Teaserpp(BaseModel):
    """
    TEASER++ point cloud registration class
    """
    
    default_conf = {
        "noise_bound": 0.05,
    }
    
    required_inputs = [
        "source_corr",
        "target_corr",
    ]
    
    def _init(self, conf):
        """
        Initialize TEASER++ with configuration
        
        Args:
            conf: Configuration dictionary
        """
        self.noise_bound = conf.get("noise_bound", 0.05)
        self.teaser_solver = None
    
    def log(self, message: str):
        """클래스 로깅 메서드"""
        print(f"[{self.__class__.__name__}] {message}")
    
    def register(self, 
                 source_points: np.ndarray, 
                 target_points: np.ndarray) -> Dict[str, Any]:
        """
        Register source point cloud to target point cloud
        
        Args:
            source_points: Source point cloud (Nx3 numpy array)
            target_points: Target point cloud (Mx3 numpy array)
            
        Returns:
            Dict containing transformation matrix and other results
        """
        self.log(f"TEASER++ registration - Source: {len(source_points)}, Target: {len(target_points)}")
        
        # Point cloud 크기 확인
        if len(source_points) < 3 or len(target_points) < 3:
            self.log("Not enough points for registration")
            return {
                'transformation': np.eye(4),
                'num_inliers': 0,
                'success': False
            }
        
        self.log("get_teaser_solver start")
        self.teaser_solver = get_teaser_solver(noise_bound=self.noise_bound)
        
        # TEASER++ registration - 전체 point cloud 사용
        self.teaser_solver.solve(source_points.T, target_points.T)
        solution = self.teaser_solver.getSolution()

        # Build transformation matrix
        T_teaser = Rt2T(solution.rotation, solution.translation)
        
        result = {
            'transformation': T_teaser,
            'num_inliers': len(source_points),
            'success': True,
            'correspondences': (source_points, target_points)
        }
        
        self.log(f"TEASER++ registration completed with {len(source_points)} points")
        return result
    
    # def visualize_correspondences(self,
    #                               source_pcd: o3d.geometry.PointCloud,
    #                               target_pcd: o3d.geometry.PointCloud,
    #                               corrs_source: np.ndarray,
    #                               corrs_target: np.ndarray):
    #     """Visualize correspondences between two point clouds"""
    #     source_xyz = pcd2xyz(source_pcd)
    #     target_xyz = pcd2xyz(target_pcd)
        
    #     source_corr = source_xyz[:, corrs_source].T
    #     target_corr = target_xyz[:, corrs_target].T
        
    #     points = np.concatenate((source_corr, target_corr), axis=0)
    #     lines = [[i, i + len(corrs_source)] for i in range(len(corrs_source))]
    #     colors = [[0, 1, 0] for _ in range(len(lines))]
        
    #     line_set = o3d.geometry.LineSet(
    #         points=o3d.utility.Vector3dVector(points),
    #         lines=o3d.utility.Vector2iVector(lines),
    #     )
    #     line_set.colors = o3d.utility.Vector3dVector(colors)
        
    #     o3d.visualization.draw_geometries([source_pcd, target_pcd, line_set])

    # def extract_fpfh_features(self, pcd: o3d.geometry.PointCloud, radius_normal: float = 0.1, radius_fpfh: float = 0.25) -> np.ndarray:
    #     """Extract FPFH features from point cloud"""
    #     # Estimate normals
    #     pcd.estimate_normals(
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    #     )
        
    #     # Compute FPFH features
    #     fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    #         pcd,
    #         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_fpfh, max_nn=100)
    #     )
        
    #     return np.array(fpfh.data).T

    def _forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass for TEASER++ registration
        
        Args:
            data: Dictionary containing:
                - feature_corres0: Source feature correspondences
                - feature_corres1: Target feature correspondences
                - source_pcd: Source point cloud (optional)
                - target_pcd: Target point cloud (optional)
                
        Returns:
            Registration result dictionary
        """
        self.log("TEASER++ _forward started")
        
        # Extract data
        feature_corres0 = data.get("feature_corres0")
        feature_corres1 = data.get("feature_corres1")

        
        if feature_corres0 is None or feature_corres1 is None:
            self.log("Error: feature_corres0 or feature_corres1 not found in data")
            return {
                'transformation': np.eye(4),
                'num_inliers': 0,
                'success': False,
                'error': 'Missing feature correspondences'
            }
        
        self.log(f"Source correspondences: {len(feature_corres0)}")
        self.log(f"Target correspondences: {len(feature_corres1)}")
        
        # Use provided feature correspondences as FPFH features

        # Run registration
        result = self.register(feature_corres0, feature_corres1)
        
        self.log("TEASER++ _forward completed")
        return result

    # def run_registration(self, 
    #                     source_pcd: o3d.geometry.PointCloud, 
    #                     target_pcd: o3d.geometry.PointCloud) -> Dict[str, Any]:
    #     """
    #     Complete registration pipeline (legacy method)
        
    #     Args:
    #         source_pcd: Source point cloud
    #         target_pcd: Target point cloud
            
    #     Returns:
    #         Registration result dictionary
    #     """
    #     self.log("Extracting FPFH features...")
        
    #     # Extract FPFH features
    #     source_fpfh = self.extract_fpfh_features(source_pcd)
    #     target_fpfh = self.extract_fpfh_features(target_pcd)
        
    #     # Run registration
    #     result = self.register(source_pcd, source_fpfh, target_pcd, target_fpfh)
        
    #     return result