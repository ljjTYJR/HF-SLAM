#!/usr/bin/env python3
"""
RGB-D SLAM with Gaussian Splatting
Clean implementation following proper software engineering practices.
"""
import argparse
import os
import sys
import time
from importlib.machinery import SourceFileLoader

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add project root to path
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

from datasets.gradslam_datasets import (
    ICLDataset, ReplicaDataset, ReplicaV2Dataset, AzureKinectDataset,
    ScannetDataset, Ai2thorDataset, Record3DDataset, RealsenseDataset,
    TUMDataset, ScannetPPDataset, NeRFCaptureDataset, load_dataset_config
)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transform_to_frame, l1_loss_v1,
    matrix_to_quaternion, quat_mult
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd



# ============================================================================
# Dataset Loading
# ============================================================================

DATASET_MAP = {
    'icl': ICLDataset,
    'replica': ReplicaDataset,
    'replicav2': ReplicaV2Dataset,
    'azure': AzureKinectDataset,
    'azurekinect': AzureKinectDataset,
    'scannet': ScannetDataset,
    'ai2thor': Ai2thorDataset,
    'record3d': Record3DDataset,
    'realsense': RealsenseDataset,
    'tum': TUMDataset,
    'scannetpp': ScannetPPDataset,
    'nerfcapture': NeRFCaptureDataset,
}


def get_dataset(config_dict, basedir, sequence, **kwargs):
    """Load dataset by name. No bullshit if-else chain."""
    dataset_name = config_dict["dataset_name"].lower()
    dataset_cls = DATASET_MAP.get(dataset_name)

    if dataset_cls is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_MAP.keys())}")

    # ScannetPP and NeRFCapture have different signatures
    if dataset_name in ['scannetpp', 'nerfcapture']:
        return dataset_cls(basedir, sequence, **kwargs)

    return dataset_cls(config_dict, basedir, sequence, **kwargs)


# ============================================================================
# Point Cloud Operations
# ============================================================================

def unproject_depth(depth, intrinsics):
    """Unproject depth map to 3D points in camera frame."""
    H, W = depth.shape[1], depth.shape[2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Create pixel grid
    x_grid, y_grid = torch.meshgrid(
        torch.arange(W, device='cuda', dtype=torch.float32),
        torch.arange(H, device='cuda', dtype=torch.float32),
        indexing='xy'
    )

    # Unproject
    z = depth[0].reshape(-1)
    x = (x_grid.reshape(-1) - cx) / fx * z
    y = (y_grid.reshape(-1) - cy) / fy * z

    return torch.stack([x, y, z], dim=-1)


def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, mask=None,
                   compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    """
    Convert RGB-D to point cloud.

    Args:
        color: [3, H, W] RGB image
        depth: [1, H, W] depth map
        intrinsics: [3, 3] camera intrinsics
        w2c: [4, 4] world-to-camera transformation
        transform_pts: transform to world frame
        mask: [H*W] valid point mask
        compute_mean_sq_dist: compute scale initialization metric

    Returns:
        point_cld: [N, 6] points with xyz + rgb
        mean_sq_dist: [N] scale metric (if requested)
    """
    H, W = color.shape[1], color.shape[2]

    # Unproject to camera frame
    pts_cam = unproject_depth(depth, intrinsics)

    # Transform to world frame
    if transform_pts:
        ones = torch.ones(H * W, 1, device='cuda', dtype=torch.float32)
        pts4 = torch.cat([pts_cam, ones], dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute scale metric for Gaussian initialization
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            fx, fy = intrinsics[0, 0], intrinsics[1, 1]
            scale = depth[0].reshape(-1) / ((fx + fy) / 2)
            mean_sq_dist = scale ** 2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method: {mean_sq_dist_method}")

    # Add color
    colors = color.permute(1, 2, 0).reshape(-1, 3)
    point_cld = torch.cat([pts, colors], dim=-1)

    # Apply mask
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean_sq_dist = mean_sq_dist[mask]

    return (point_cld, mean_sq_dist) if compute_mean_sq_dist else point_cld


# ============================================================================
# Gaussian Parameters
# ============================================================================

def initialize_params(init_pt_cld, num_frames, mean_sq_dist):
    """
    Initialize Gaussian parameters from point cloud.

    Args:
        init_pt_cld: [N, 6] initial points (xyz + rgb)
        num_frames: total number of frames
        mean_sq_dist: [N] scale initialization metric

    Returns:
        params: dict of torch.nn.Parameter
        variables: dict of auxiliary variables for optimization
    """
    num_pts = init_pt_cld.shape[0]

    # Gaussian parameters
    params = {
        'means3D': init_pt_cld[:, :3],
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': np.tile([1, 0, 0, 0], (num_pts, 1)),
        'logit_opacities': torch.zeros((num_pts, 1), dtype=torch.float32, device='cuda'),
        'log_scales': torch.log(torch.sqrt(mean_sq_dist)).unsqueeze(-1).repeat(1, 3),
    }

    # Camera trajectory (relative to first frame)
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    # Convert to parameters
    for k, v in params.items():
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, dtype=torch.float32, device='cuda')
        params[k] = torch.nn.Parameter(v.contiguous().requires_grad_(True))

    # Auxiliary variables for densification and regularization
    variables = {
        'max_2D_radius': torch.zeros(num_pts, device='cuda'),
        'means2D_gradient_accum': torch.zeros(num_pts, device='cuda'),
        'denom': torch.zeros(num_pts, device='cuda'),
        'timestep': torch.zeros(num_pts, device='cuda'),
        'seen_times': torch.zeros(num_pts, device='cuda', dtype=torch.int16),

        # For regularization
        'log_scale_last_frame': params['log_scales'].detach().clone(),
        'scale_importance_weights': torch.zeros_like(params['log_scales']),
        'scale_importance_weights_sum': torch.zeros_like(params['log_scales']),

        'last_rgb_colors': params['rgb_colors'].detach().clone(),
        'rgb_colors_importance_weights': torch.zeros_like(params['rgb_colors']),
        'rgb_colors_importance_weights_sum': torch.zeros_like(params['rgb_colors']),

        'last_depth': params['means3D'][:, 2].detach().clone(),
        'depth_importance_weights': torch.zeros_like(params['means3D'][:, 2]),
        'depth_importance_weights_sum': torch.zeros_like(params['means3D'][:, 2]),
    }

    return params, variables


def initialize_new_params(new_pt_cld, mean_sq_dist):
    """Initialize parameters for newly added Gaussians."""
    num_pts = new_pt_cld.shape[0]

    params = {
        'means3D': new_pt_cld[:, :3],
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': np.tile([1, 0, 0, 0], (num_pts, 1)),
        'logit_opacities': torch.zeros((num_pts, 1), dtype=torch.float32, device='cuda'),
        'log_scales': torch.log(torch.sqrt(mean_sq_dist)).unsqueeze(-1).repeat(1, 3),
    }

    for k, v in params.items():
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v, dtype=torch.float32, device='cuda')
        params[k] = torch.nn.Parameter(v.contiguous().requires_grad_(True))

    return params


# ============================================================================
# Optimizer
# ============================================================================

def initialize_optimizer(params, lrs_dict, tracking=False):
    """
    Initialize Adam optimizer with per-parameter learning rates.

    Args:
        params: dict of parameters
        lrs_dict: dict of learning rates per parameter
        tracking: if True, use normal LR; if False, use 0 (for mapping)
    """
    param_groups = [{'params': [v], 'name': k, 'lr': lrs_dict[k]} for k, v in params.items()]

    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


# ============================================================================
# Initialization
# ============================================================================

def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, mean_sq_dist_method):
    """Initialize scene from first frame."""
    # Load first frame
    color, depth, intrinsics, pose = dataset[0]

    # Process data
    color = color.permute(2, 0, 1) / 255.0
    depth = depth.permute(2, 0, 1)
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup camera
    cam = setup_camera(color.shape[2], color.shape[1],
                       intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    # Generate initial point cloud
    mask = (depth > 0).reshape(-1)
    init_pt_cld, initial_scales = get_pointcloud(
        color, depth, intrinsics, w2c,
        mask=mask, compute_mean_sq_dist=True, mean_sq_dist_method=mean_sq_dist_method
    )

    # Initialize parameters
    params, variables = initialize_params(init_pt_cld, num_frames, initial_scales)

    # Initialize scene radius for densification
    variables['scene_radius'] = torch.max(depth) / scene_radius_depth_ratio

    return params, variables, intrinsics, w2c, cam


# ============================================================================
# Loss Functions
# ============================================================================

def compute_loss_mask(depth, gt_depth, ignore_outliers=True):
    """Compute valid pixel mask for loss computation."""
    mask = (gt_depth > 0) & (~torch.isnan(depth))

    if ignore_outliers:
        depth_error = torch.abs(gt_depth - depth) * (gt_depth > 0)
        outlier_threshold = 10 * depth_error.median()
        mask = mask & (depth_error < outlier_threshold) & (depth > 0.01)

    return mask


def get_loss_tracking(params, curr_data, variables, time_idx, loss_weights,
                      use_l1=True, ignore_outliers=True, fea_mask=None):
    """
    Compute tracking loss (camera pose optimization).

    Only camera pose gets gradients, Gaussians are frozen.
    """
    # Transform Gaussians to current frame
    transformed_pts = transform_to_frame(params, time_idx,
                                         gaussians_grad=False, camera_grad=True)

    # Render
    rendervar, _ = transformed_params2rendervar(params, transformed_pts, drop_out_ratio=None)
    rendervar['means2D'].retain_grad()
    im, radius, depth, opacity = Renderer(raster_settings=curr_data['cam'])(**rendervar)

    variables['means2D'] = rendervar['means2D']

    # Compute mask
    if fea_mask is not None:
        mask = torch.from_numpy(fea_mask[None, :, :]).cuda().bool()
    else:
        mask = torch.ones_like(depth, dtype=torch.bool)

    mask = mask & compute_loss_mask(depth, curr_data['depth'], ignore_outliers)

    # Depth loss
    losses = {}
    if use_l1:
        losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()

    # RGB loss (LAB color space for robustness)
    mask = mask & (opacity > 0.99)
    color_mask = mask.repeat(3, 1, 1).detach()
    losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()

    # Weighted loss
    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    # Update variables
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def get_loss_mapping(params, curr_data, variables, time_idx, loss_weights,
                     use_l1=True, ignore_outliers=True, do_ba=False,
                     use_reg=False, drop_out=False, drop_out_ratio=0.5):
    """
    Compute mapping loss (Gaussian optimization).

    Gaussians get gradients, camera pose optionally gets gradients (BA).
    """
    # Transform Gaussians to current frame
    if do_ba:
        transformed_pts = transform_to_frame(params, time_idx,
                                             gaussians_grad=True, camera_grad=True)
    else:
        transformed_pts = transform_to_frame(params, time_idx,
                                             gaussians_grad=True, camera_grad=False)

    # Render
    drop_ratio = drop_out_ratio if drop_out else None
    rendervar, mask_drop_out = transformed_params2rendervar(params, transformed_pts, drop_ratio)
    rendervar['means2D'].retain_grad()
    im, radius, depth, opacity = Renderer(raster_settings=curr_data['cam'])(**rendervar)

    variables['means2D'] = rendervar['means2D']

    # Compute mask
    mask = compute_loss_mask(depth, curr_data['depth'], ignore_outliers)

    # Losses
    losses = {}

    # Depth loss
    if use_l1:
        losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask.detach()].mean()

    # RGB loss
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    # Regularization losses
    if use_reg:
        color_diff = params['rgb_colors'] - variables['last_rgb_colors']
        losses['color_reg'] = (variables['rgb_colors_importance_weights'] * torch.abs(color_diff)).mean()

        depth_diff = params['means3D'][:, 2] - variables['last_depth']
        losses['depth_reg'] = (variables['depth_importance_weights'] * torch.abs(depth_diff)).mean()

        scale_diff = params['log_scales'] - variables['log_scale_last_frame']
        losses['scale_reg'] = (variables['scale_importance_weights'] * torch.abs(scale_diff)).mean()

    # Weighted loss
    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    # Update variables
    seen = radius > 0
    if mask_drop_out is not None:
        variables['max_2D_radius'][mask_drop_out][seen] = torch.max(
            radius[seen], variables['max_2D_radius'][mask_drop_out][seen]
        )
    else:
        variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])

    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def update_importance_weights(params, variables, curr_data, time_idx, loss_weights):
    """
    Update importance weights for regularization.

    Computes gradients to determine which Gaussians are important for the current view.
    """
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=True, camera_grad=False)
    rendervar, _ = transformed_params2rendervar(params, transformed_pts, drop_out_ratio=None)
    im, radius, depth, opacity = Renderer(raster_settings=curr_data['cam'])(**rendervar)

    # Compute loss
    loss = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    loss = loss * loss_weights.get('im', 1.0)
    loss.backward()

    # Update importance weights
    seen = radius > 0
    variables['seen_times'][seen] += 1
    variables['rgb_colors_importance_weights_sum'][seen] += torch.abs(params['rgb_colors'].grad)[seen]
    variables['depth_importance_weights_sum'][seen] += torch.abs(params['means3D'].grad[:, 2])[seen]
    variables['scale_importance_weights_sum'][seen] += torch.abs(params['log_scales'].grad)[seen]

    # Compute average weights
    seen_count = variables['seen_times'].unsqueeze(1)
    variables['rgb_colors_importance_weights'] = variables['rgb_colors_importance_weights_sum'] / seen_count
    variables['depth_importance_weights'] = variables['depth_importance_weights_sum'] / variables['seen_times']
    variables['scale_importance_weights'] = variables['scale_importance_weights_sum'] / seen_count

    # Zero gradients
    for v in params.values():
        if v.requires_grad and v.grad is not None:
            v.grad.zero_()

    return variables


# ============================================================================
# Densification
# ============================================================================

def add_new_gaussians(params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method):
    """
    Add new Gaussians in under-represented regions.

    Identifies regions with:
    - Low opacity (holes)
    - Large RGB errors
    - Large depth errors
    """
    # Render current frame
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)
    rendervar, _ = transformed_params2rendervar(params, transformed_pts, drop_out_ratio=None)
    im, radius, render_depth, opacity = Renderer(raster_settings=curr_data['cam'])(**rendervar)

    # Compute errors
    depth_loss = torch.abs(curr_data['depth'] - render_depth)
    rgb_loss = torch.abs(curr_data['im'] - im).mean(dim=0)

    # Determine where to densify
    densify_mask = (opacity < sil_thres) | (rgb_loss > 0.6)

    depth_diff_ratio = depth_loss / (curr_data['depth'] + 1e-6)
    densify_mask = densify_mask | (depth_diff_ratio > 0.1)

    # Apply valid depth mask
    gt_depth = curr_data['depth'].squeeze()
    densify_mask = (gt_depth > 0) & densify_mask
    non_presence_mask = densify_mask.reshape(-1)

    # Add new Gaussians
    if torch.sum(non_presence_mask) > 0:
        # Get current camera pose
        curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        curr_cam_tran = params['cam_trans'][..., time_idx].detach()
        curr_w2c = torch.eye(4, device='cuda', dtype=torch.float32)
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran

        # Generate new point cloud
        valid_depth_mask = (curr_data['depth'][0] > 0)
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)

        new_pt_cld, mean_sq_dist = get_pointcloud(
            curr_data['im'], curr_data['depth'], curr_data['intrinsics'], curr_w2c,
            mask=non_presence_mask, compute_mean_sq_dist=True,
            mean_sq_dist_method=mean_sq_dist_method
        )

        # Initialize new parameters
        new_params = initialize_new_params(new_pt_cld, mean_sq_dist)

        # Append to existing parameters
        for k, v in new_params.items():
            params[k] = torch.nn.Parameter(
                torch.cat([params[k], v], dim=0).requires_grad_(True)
            )

        # Update variables
        num_new = new_pt_cld.shape[0]
        num_total = params['means3D'].shape[0]

        # Update regularization variables
        variables['last_rgb_colors'] = params['rgb_colors'].detach().clone()
        variables['rgb_colors_importance_weights'] = torch.cat([
            variables['rgb_colors_importance_weights'],
            torch.zeros(num_new, 3, device='cuda')
        ])
        variables['rgb_colors_importance_weights_sum'] = torch.cat([
            variables['rgb_colors_importance_weights_sum'],
            torch.zeros(num_new, 3, device='cuda')
        ])

        variables['log_scale_last_frame'] = params['log_scales'].detach().clone()
        variables['scale_importance_weights'] = torch.cat([
            variables['scale_importance_weights'],
            torch.zeros(num_new, 3, device='cuda')
        ])
        variables['scale_importance_weights_sum'] = torch.cat([
            variables['scale_importance_weights_sum'],
            torch.zeros(num_new, 3, device='cuda')
        ])

        variables['last_depth'] = params['means3D'][:, 2].detach().clone()
        variables['depth_importance_weights'] = torch.cat([
            variables['depth_importance_weights'],
            torch.zeros(num_new, device='cuda')
        ])
        variables['depth_importance_weights_sum'] = torch.cat([
            variables['depth_importance_weights_sum'],
            torch.zeros(num_new, device='cuda')
        ])

        # Update other variables
        variables['seen_times'] = torch.cat([
            variables['seen_times'],
            torch.zeros(num_new, device='cuda', dtype=torch.int16)
        ])
        variables['means2D_gradient_accum'] = torch.zeros(num_total, device='cuda')
        variables['denom'] = torch.zeros(num_total, device='cuda')
        variables['max_2D_radius'] = torch.zeros(num_total, device='cuda')
        variables['timestep'] = torch.cat([
            variables['timestep'],
            torch.full((num_new,), time_idx, device='cuda', dtype=torch.float32)
        ])

    return params, variables


# ============================================================================
# Camera Pose
# ============================================================================

def initialize_camera_pose(params, curr_time_idx, forward_prop=True):
    """
    Initialize camera pose for current frame.

    Uses constant velocity model if forward_prop=True and curr_time_idx > 1.
    """
    with torch.no_grad():
        if curr_time_idx > 1 and forward_prop:
            # Constant velocity model
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx - 1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx - 2].detach())

            # Quaternion velocity
            prev_rot2_inv = torch.cat([prev_rot2[:, 0:1], -prev_rot2[:, 1:4]], dim=1)
            rot_velocity = quat_mult(prev_rot1, prev_rot2_inv)
            new_rot = quat_mult(rot_velocity, prev_rot1)
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()

            # Translation velocity
            prev_tran1 = params['cam_trans'][..., curr_time_idx - 1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx - 2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else:
            # Copy previous frame
            params['cam_unnorm_rots'][..., curr_time_idx] = \
                params['cam_unnorm_rots'][..., curr_time_idx - 1].detach()
            params['cam_trans'][..., curr_time_idx] = \
                params['cam_trans'][..., curr_time_idx - 1].detach()

    return params


# ============================================================================
# Feature Matching
# ============================================================================

def compute_feature_mask(params, curr_data, time_idx, extractor, matcher):
    """
    Compute feature matching mask for robust tracking.

    Uses SuperPoint + LightGlue to find reliable correspondences.
    """
    with torch.no_grad():
        # Render current estimate
        transformed_pts = transform_to_frame(params, time_idx,
                                             gaussians_grad=False, camera_grad=False)
        rendervar, _ = transformed_params2rendervar(params, transformed_pts, drop_out_ratio=None)
        im, _, _, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)

        # Extract features
        feats0 = extractor.extract(im)
        feats1 = extractor.extract(curr_data['im'])
        matches01 = matcher({"image0": feats0, "image1": feats1})

        # Remove batch dimension
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

        # Get matched keypoints
        kpts0 = feats0["keypoints"]
        kpts1 = feats1["keypoints"]
        matches = matches01["matches"]
        m_kpts1 = kpts1[matches[..., 1]].long()

        # Create mask
        H, W = curr_data['im'].shape[1], curr_data['im'].shape[2]
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[m_kpts1[:, 1], m_kpts1[:, 0]] = 1

        # Dilate to cover surrounding area
        kernel = np.ones((25, 25), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask


# ============================================================================
# Main SLAM Function
# ============================================================================

def setup_config_defaults(config):
    """Set default config values if not present."""
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000

    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False

    return config


def setup_dataset_config(dataset_config):
    """Set default dataset config values."""
    defaults = {
        'ignore_bad': False,
        'use_train_split': True,
        'tracking_image_height': dataset_config.get('desired_image_height'),
        'tracking_image_width': dataset_config.get('desired_image_width'),
    }

    for k, v in defaults.items():
        if k not in dataset_config:
            dataset_config[k] = v

    # Check if separate tracking resolution is used
    seperate_tracking_res = (
        dataset_config['tracking_image_height'] != dataset_config['desired_image_height'] or
        dataset_config['tracking_image_width'] != dataset_config['desired_image_width']
    )

    return dataset_config, seperate_tracking_res


def rgbd_slam(config):
    """
    Main RGB-D SLAM pipeline.

    Args:
        config: configuration dictionary
    """
    # Setup config
    config = setup_config_defaults(config)
    print(f"Config: {config}")

    # Create output directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Setup device
    device = torch.device(config["primary_device"])

    # Load dataset
    print("Loading dataset...")
    dataset_config = config["data"]

    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {"dataset_name": dataset_config["dataset_name"]}
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])

    dataset_config, seperate_tracking_res = setup_dataset_config(dataset_config)

    # Load main dataset
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )

    num_frames = dataset_config.get("num_frames", -1)
    if num_frames == -1:
        num_frames = len(dataset)

    # Initialize from first frame
    params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(
        dataset, num_frames,
        config['scene_radius_depth_ratio'],
        config['mean_sq_dist_method']
    )

    # Load separate tracking dataset if needed
    tracking_dataset = None
    tracking_cam = None
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(2, 0, 1) / 255.0
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(
            tracking_color.shape[2], tracking_color.shape[1],
            tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy()
        )

    # Initialize keyframe list
    keyframe_list = []
    keyframe_time_indices = []

    # Initialize tracking variables
    gt_w2c_all_frames = []

    # Load checkpoint if requested
    checkpoint_time_idx = 0
    if config.get('load_checkpoint', False):
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading checkpoint for frame {checkpoint_time_idx}")

        ckpt_path = os.path.join(config['workdir'], config['run_name'],
                                 f"params{checkpoint_time_idx}.npz")
        params_data = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.nn.Parameter(torch.tensor(params_data[k]).cuda().requires_grad_(True))
                  for k in params_data.keys()}

        # Reset some variables
        num_gaussians = params['means3D'].shape[0]
        variables['max_2D_radius'] = torch.zeros(num_gaussians, device='cuda')
        variables['means2D_gradient_accum'] = torch.zeros(num_gaussians, device='cuda')
        variables['denom'] = torch.zeros(num_gaussians, device='cuda')
        variables['timestep'] = torch.zeros(num_gaussians, device='cuda')

        # Load keyframe indices
        kf_path = os.path.join(config['workdir'], config['run_name'],
                               f"keyframe_time_indices{checkpoint_time_idx}.npy")
        keyframe_time_indices = np.load(kf_path).tolist()

        # Rebuild keyframe list
        for time_idx in range(checkpoint_time_idx):
            color, depth, _, gt_pose = dataset[time_idx]
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)

            if time_idx in keyframe_time_indices:
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4, device='cuda', dtype=torch.float32)
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran

                color = color.permute(2, 0, 1) / 255.0
                depth = depth.permute(2, 0, 1)
                keyframe_list.append({
                    'id': time_idx,
                    'est_w2c': curr_w2c,
                    'color': color,
                    'depth': depth
                })

    # Initialize feature matcher
    extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
    matcher = LightGlue(features="superpoint").eval().cuda()

    # Main SLAM loop
    print(f"Starting SLAM from frame {checkpoint_time_idx} to {num_frames}")

    for time_idx in tqdm(range(checkpoint_time_idx, num_frames), desc="SLAM"):
        # Load current frame
        color, depth, _, gt_pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(gt_pose)

        # Process data
        color = color.permute(2, 0, 1) / 255.0
        depth = depth.permute(2, 0, 1)
        gt_w2c_all_frames.append(gt_w2c)

        # Prepare data dict
        curr_data = {
            'cam': cam,
            'im': color,
            'depth': depth,
            'id': time_idx,
            'intrinsics': intrinsics,
            'w2c': first_frame_w2c,
            'iter_gt_w2c_list': gt_w2c_all_frames
        }

        # Tracking data (potentially different resolution)
        if seperate_tracking_res:
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255.0
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data = {
                'cam': tracking_cam,
                'im': tracking_color,
                'depth': tracking_depth,
                'id': time_idx,
                'intrinsics': tracking_intrinsics,
                'w2c': first_frame_w2c,
                'iter_gt_w2c_list': gt_w2c_all_frames
            }
        else:
            tracking_curr_data = curr_data

        # Initialize camera pose for current frame
        if time_idx > 0:
            params = initialize_camera_pose(
                params, time_idx, forward_prop=config['tracking']['forward_prop']
            )

        # ====================================================================
        # TRACKING
        # ====================================================================
        if time_idx > 0 and not config['tracking'].get('use_gt_poses', False):
            # Initialize optimizer
            optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)

            # Compute feature mask for robust tracking
            # TODO
            # fea_mask = compute_feature_mask(params, tracking_curr_data, time_idx, extractor, matcher)
            fea_mask = None

            # Track best candidate
            best_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            best_tran = params['cam_trans'][..., time_idx].detach().clone()
            best_loss = float('inf')

            # Tracking iterations
            num_iters = config['tracking']['num_iters']

            for iter_idx in range(num_iters):
                # Compute loss
                loss, variables, losses = get_loss_tracking(
                    params, tracking_curr_data, variables, time_idx,
                    config['tracking']['loss_weights'],
                    use_l1=config['tracking']['use_l1'],
                    ignore_outliers=config['tracking']['ignore_outlier_depth_loss'],
                    fea_mask=fea_mask
                )

                # Optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Track best candidate
                with torch.no_grad():
                    if loss < best_loss:
                        best_loss = loss
                        best_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        best_tran = params['cam_trans'][..., time_idx].detach().clone()

            # Restore best candidate
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = best_rot
                params['cam_trans'][..., time_idx] = best_tran

        elif time_idx > 0:
            # Use ground truth poses
            with torch.no_grad():
                rel_w2c = gt_w2c_all_frames[-1]
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach()
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot)
                rel_w2c_tran = rel_w2c[:3, 3].detach()
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat
                params['cam_trans'][..., time_idx] = rel_w2c_tran

        # ====================================================================
        # MAPPING
        # ====================================================================
        if time_idx == 0 or (time_idx + 1) % config['map_every'] == 0:
            # Densification
            if config['mapping']['add_new_gaussians'] and time_idx > 0:
                params, variables = add_new_gaussians(
                    params, variables, curr_data,
                    config['mapping']['sil_thres'], time_idx,
                    config['mean_sq_dist_method']
                )

            # Keyframe selection
            with torch.no_grad():
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4, device='cuda', dtype=torch.float32)
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran

                # Select overlapping keyframes
                num_keyframes = config['mapping_window_size'] - 2
                if len(keyframe_list) > 0:
                    selected_keyframes = keyframe_selection_overlap(
                        depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes
                    )
                    selected_time_idx = [keyframe_list[idx]['id'] for idx in selected_keyframes]

                    # Add last keyframe
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list) - 1)
                else:
                    selected_keyframes = []
                    selected_time_idx = []

                # Add current frame
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)

                print(f"Selected keyframes at frame {time_idx}: {selected_time_idx}")

            # Initialize optimizer for mapping
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False)

            # Mapping iterations
            single_frame_iters = config['mapping']['single_frame_iterations']
            replay_iters = config['mapping']['num_iters']
            total_iters = single_frame_iters + replay_iters

            chosen_keyframes = selected_keyframes.copy()
            np.random.shuffle(chosen_keyframes)

            for iter_idx in range(total_iters):
                # Select frame to optimize
                if iter_idx < single_frame_iters:
                    # Current frame only
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                else:
                    # Random keyframe
                    if not chosen_keyframes:
                        chosen_keyframes = selected_keyframes.copy()
                        np.random.shuffle(chosen_keyframes)

                    kf_idx = chosen_keyframes.pop()

                    if kf_idx == -1:
                        iter_time_idx = time_idx
                        iter_color = color
                        iter_depth = depth
                    else:
                        iter_time_idx = keyframe_list[kf_idx]['id']
                        iter_color = keyframe_list[kf_idx]['color']
                        iter_depth = keyframe_list[kf_idx]['depth']

                # Prepare data
                iter_data = {
                    'cam': cam,
                    'im': iter_color,
                    'depth': iter_depth,
                    'id': iter_time_idx,
                    'intrinsics': intrinsics,
                    'w2c': first_frame_w2c,
                    'iter_gt_w2c_list': gt_w2c_all_frames[:iter_time_idx + 1]
                }

                # Compute loss
                use_reg = (
                    config['mapping']['reg_loss'].get('color_reg', False) or
                    config['mapping']['reg_loss'].get('depth_reg', False) or
                    config['mapping']['reg_loss'].get('scale_reg', False)
                )

                loss, variables, losses = get_loss_mapping(
                    params, iter_data, variables, iter_time_idx,
                    config['mapping']['loss_weights'],
                    use_l1=config['mapping']['use_l1'],
                    ignore_outliers=config['mapping']['ignore_outlier_depth_loss'],
                    do_ba=False,
                    use_reg=use_reg,
                    drop_out=config['mapping'].get('drop_out', False),
                    drop_out_ratio=config['mapping'].get('drop_out_ratio', 0.5)
                )

                # Optimize
                loss.backward()

                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping'].get('prune_gaussians', False):
                        params, variables, _ = prune_gaussians(
                            params, variables, optimizer, iter_idx,
                            config['mapping']['pruning_dict']
                        )

                    # Densify Gaussians
                    if config['mapping'].get('use_gaussian_splatting_densification', False):
                        params, variables = densify(
                            params, variables, optimizer, iter_idx,
                            config['mapping']['densify_dict']
                        )

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            # Update importance weights for regularization
            if use_reg:
                variables = update_importance_weights(
                    params, variables, iter_data, time_idx,
                    config['mapping']['loss_weights']
                )

        # Add frame to keyframe list
        if (time_idx == 0 or
            (time_idx + 1) % config['keyframe_every'] == 0 or
            time_idx == num_frames - 2):
            with torch.no_grad():
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4, device='cuda', dtype=torch.float32)
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran

                keyframe_list.append({
                    'id': time_idx,
                    'est_w2c': curr_w2c,
                    'color': color,
                    'depth': depth
                })
                keyframe_time_indices.append(time_idx)

        # Save checkpoint
        if time_idx % config.get("checkpoint_interval", 100) == 0 and config.get('save_checkpoints', False):
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(
                os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"),
                np.array(keyframe_time_indices)
            )

        torch.cuda.empty_cache()

    # Final evaluation
    print("Running final evaluation...")
    with torch.no_grad():
        eval(dataset, params, num_frames, eval_dir,
             sil_thres=config['mapping']['sil_thres'],
             mapping_iters=config['mapping']['num_iters'],
             add_new_gaussians=config['mapping']['add_new_gaussians'],
             eval_every=config.get('eval_every', 5))

    # Save final parameters
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = np.stack(
        [w2c.detach().cpu().numpy() for w2c in gt_w2c_all_frames], axis=0
    )
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)

    save_params(params, output_dir)

    print(f"SLAM complete. Results saved to {output_dir}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RGB-D SLAM with Gaussian Splatting")
    parser.add_argument("experiment", type=str, help="Path to experiment config file")
    args = parser.parse_args()

    # Load experiment config
    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set seed
    seed_everything(seed=experiment.config['seed'])

    # Create results directory
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )

    if not experiment.config.get('load_checkpoint', False):
        os.makedirs(results_dir, exist_ok=True)
        import shutil
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    # Run SLAM
    rgbd_slam(experiment.config)


if __name__ == "__main__":
    main()
