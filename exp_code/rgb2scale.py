# The files is for off-line bundle adjustment of gaussian splatting
import torch
import numpy as np
import cv2
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils import gui
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    l1_loss_v1, matrix_to_quaternion, quat_mult, transform_to_gui_view, parms_to_render_camera, render_warp
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
import matplotlib.pyplot as plt
import torchvision
from importlib.machinery import SourceFileLoader
import os
from tqdm import tqdm
from utils.slam_external import build_rotation,calc_psnr
import torch.nn.functional as F
from utils.eval_helpers import evaluate_ate

from datasets.gradslam_datasets import (
    load_dataset_config,
    ICLDataset,
    ReplicaDataset,
    ReplicaV2Dataset,
    AzureKinectDataset,
    ScannetDataset,
    Ai2thorDataset,
    Record3DDataset,
    RealsenseDataset,
    TUMDataset,
    ScannetPPDataset,
    NeRFCaptureDataset
)

from skimage.color import rgb2gray
from skimage import filters

def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")

def evaluation(dataset, frame_indices, params, eval_every=1):
    gt_w2c_list = []
    psnr_list = []
    print("Begin to evaluate the performance of the saved model.")
    for time_idx in tqdm(range(len(dataset))):
        color, depth, intrinsics, pose = dataset[time_idx]
        gt_w2c = torch.linalg.inv(pose)
        gt_w2c_list.append(gt_w2c)
        intrinsics = intrinsics[:3, :3]
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

        if time_idx == 0:
            first_frame_w2c = torch.linalg.inv(pose)
            cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())

        # Get current frame Gaussians
        transformed_pts = transform_to_frame_offline(params, time_idx, gaussians_grad=False, camera_grad=False)
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': time_idx, 'intrinsics': intrinsics, 'w2c': first_frame_w2c}
        rendervar, _ = transformed_params2rendervar(params, transformed_pts)
        im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
        valid_depth_mask = (curr_data['depth'] > 0)
        weighted_im = im * valid_depth_mask
        weighted_gt_im = curr_data['im'] * valid_depth_mask
        psnr = calc_psnr(weighted_im, weighted_gt_im).mean()

        psnr_list.append(psnr.cpu().numpy())

    # Compute the final ATE RMSE
    num_frames = params['cam_unnorm_rots'].shape[-1]
    latest_est_w2c = first_frame_w2c
    latest_est_w2c_list = []
    latest_est_w2c_list.append(latest_est_w2c)
    valid_gt_w2c_list = []
    valid_gt_w2c_list.append(gt_w2c_list[0])
    for idx in range(1, num_frames):
        # Check if gt pose is not nan for this time step
        if torch.isnan(gt_w2c_list[idx]).sum() > 0:
            continue
        interm_cam_rot = F.normalize(params['cam_unnorm_rots'][..., idx].detach())
        interm_cam_trans = params['cam_trans'][..., idx].detach()
        intermrel_w2c = torch.eye(4).cuda().float()
        intermrel_w2c[:3, :3] = build_rotation(interm_cam_rot)
        intermrel_w2c[:3, 3] = interm_cam_trans
        latest_est_w2c = intermrel_w2c
        latest_est_w2c_list.append(latest_est_w2c)
        valid_gt_w2c_list.append(gt_w2c_list[idx]) # estimate in a relative way
    gt_w2c_list = valid_gt_w2c_list
    ate_rmse = evaluate_ate(gt_w2c_list, latest_est_w2c_list)
    print("Final Average ATE RMSE of all frames: {:.2f} cm".format(ate_rmse*100))
    avg_psnr = np.array(psnr_list).mean()
    print("Average PSNR of all frames: {:.2f}".format(avg_psnr))

    keyframe_psnr_list = []
    keyframe_gt_w2c_list = []
    keyframe_est_w2c_list = []
    for key_idx in frame_indices:
        keyframe_psnr_list.append(psnr_list[key_idx])
        keyframe_gt_w2c_list.append(gt_w2c_list[key_idx])
        keyframe_est_w2c_list.append(latest_est_w2c_list[key_idx])
    keyframe_ate_rmse = evaluate_ate(keyframe_gt_w2c_list, keyframe_est_w2c_list)
    print("Average ATE RMSE of keyframes: {:.2f} cm".format(keyframe_ate_rmse*100))
    avg_keyframe_psnr = np.array(keyframe_psnr_list).mean()
    print("Average PSNR of keyframes: {:.2f}".format(avg_keyframe_psnr))

def transform_to_frame_offline(params, time_idx, gaussians_grad, camera_grad):
    """
    Function to transform Isotropic Gaussians from world frame to camera frame.

    Args:
        params: dict of parameters
        time_idx: time index to transform to
        gaussians_grad: enable gradients for Gaussians
        camera_grad: enable gradients for camera pose

    Returns:
        transformed_pts: Transformed Centers of Gaussians
    """
    # Get Frame Camera Pose
    if camera_grad and (time_idx != 0):
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx])
        cam_tran = params['cam_trans'][..., time_idx]
    else:
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        cam_tran = params['cam_trans'][..., time_idx].detach()
    rel_w2c = torch.eye(4).cuda().float()
    rel_w2c[:3, :3] = build_rotation(cam_rot)
    rel_w2c[:3, 3] = cam_tran

    # Get Centers and norm Rots of Gaussians in World Frame
    if gaussians_grad:
        pts = params['means3D']
    else:
        pts = params['means3D'].detach()

    # Transform Centers and Unnorm Rots of Gaussians to Camera Frame
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
    pts4 = torch.cat((pts, pts_ones), dim=1)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3]

    return transformed_pts

EVERY_FRAME = 5
MODEL_PATH = "/home/shuo/scannet/scene0106_00/params1499.npz"
saved_model = np.load(MODEL_PATH)
# check the gpu exists or not, if not, use cpu
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("The data saved in the model is:", saved_model.files)

means3D = saved_model["means3D"]
rgb_colors = saved_model["rgb_colors"]
unnorm_rotations = saved_model["unnorm_rotations"]
logit_opacities = saved_model["logit_opacities"]
log_scales = saved_model["log_scales"]
timesteps = saved_model["timestep"] if "timestep" in saved_model else None
keyframe_time_indices = saved_model["keyframe_time_indices"] if "keyframe_time_indices" in saved_model else None
est_w2c_all_frames = saved_model["est_w2c_all_frames"] if "est_w2c_all_frames" in saved_model else None
cam_unnorm_rots = saved_model['cam_unnorm_rots'] if "cam_unnorm_rots" in saved_model else None
cam_trans = saved_model['cam_trans'] if "cam_trans" in saved_model else None
width = saved_model['org_width'] if "org_width" in saved_model else 640
height = saved_model['org_height'] if "org_height" in saved_model else 480


means3D = torch.from_numpy(means3D).to(DEVICE).requires_grad_(False)
rgb_colors = torch.from_numpy(rgb_colors).to(DEVICE).requires_grad_(False)
unnorm_rotations = torch.from_numpy(unnorm_rotations).to(DEVICE).requires_grad_(False)
logit_opacities = torch.from_numpy(logit_opacities).to(DEVICE).requires_grad_(False)
log_scales = torch.from_numpy(log_scales).to(DEVICE).requires_grad_(False) # (N, 1)
cam_unnorm_rots = torch.from_numpy(cam_unnorm_rots).to(DEVICE).requires_grad_(False)
cam_trans = torch.from_numpy(cam_trans).to(DEVICE).requires_grad_(False)
default_w2c = torch.eye(4).to(DEVICE)

experiment = SourceFileLoader(os.path.basename("configs/scannet/scannet_eval.py"), "configs/scannet/scannet_eval.py").load_module()
config = experiment.config
dataset_config = config["data"]
if "ignore_bad" not in dataset_config:
    dataset_config["ignore_bad"] = False
if "use_train_split" not in dataset_config:
    dataset_config["use_train_split"] = True
gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=DEVICE,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    )

color, depth, intrinsics, pose = dataset[0]

color_np = color.cpu().numpy() / 255.0
gray = rgb2gray(color_np)

n_sample = 10_000
grad_y = filters.sobel_h(gray)
grad_x = filters.sobel_v(gray)
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
selected_index = np.argpartition(grad_mag, -10*n_sample, axis=None)[-10*n_sample:]
samples_index = np.random.choice(selected_index, size=n_sample, replace=False)
color_samples = color_np.reshape(-1, 3)[samples_index]
u, v = np.unravel_index(samples_index, gray.shape)
uv = np.stack((u, v), axis=-1)
# visualize uv in the gray image, make them red
gary_copy = gray.copy()
mask = np.zeros_like(gary_copy)
mask[u, v] = 1
gary_copy[mask == 1] = 1
gary_copy[mask == 0] = 0

fig, axes = plt.subplots(1, 3, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(color_np)
ax[0].set_title("Original")
ax[1].imshow(gray, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")
ax[2].imshow(gary_copy, cmap=plt.cm.gray)
# ax[2].scatter(v, u, c="r", s=1)
ax[2].set_title("Sampled Points")
for a in ax:
    a.axis('off')
fig.tight_layout()
plt.show()
exit(0)