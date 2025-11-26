import os

import numpy as np
import random
import torch
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage import filters

def seed_everything(seed=42):
    """
        Set the `seed` value for torch and numpy seeds. Also turns on
        deterministic execution for cudnn.

        Parameters:
        - seed:     A hashable seed value
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed} (type: {type(seed)})")


def params2cpu(params):
    res = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            res[k] = v.detach().cpu().contiguous().numpy()
        else:
            res[k] = v
    return res


def save_params(output_params, output_dir):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **to_save)


def save_params_ckpt(output_params, output_dir, time_idx):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **to_save)


def save_seq_params(all_params, output_dir):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **params_to_save)


def save_seq_params_ckpt(all_params, output_dir,time_idx):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **params_to_save)

def grad_sample(rgb_image, depth_image, intrinsics, w2c, mask, N):
    'sample top N pixels with highest gradients'
    # Compute indices of pixels
    width, height = rgb_image.shape[2], rgb_image.shape[1]
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(),
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth_image[0].reshape(-1)
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(w2c)
    pts = (c2w @ pts4.T).T[:, :3]

    rgb_image_cpu = rgb_image.permute(1, 2, 0).detach().cpu().numpy()
    depth_image_cpu = depth_image.permute(1, 2, 0).squeeze().detach().cpu().numpy()
    # TODO: what if depth has nan
    rgb_image_cpu_gray = rgb2gray(rgb_image_cpu)
    grad_rgb_y = filters.sobel_h(rgb_image_cpu_gray)
    grad_rgb_x = filters.sobel_v(rgb_image_cpu_gray)
    grad_rgb_mag = np.sqrt(grad_rgb_y**2 + grad_rgb_x**2)
    selected_rgb_index = np.argpartition(grad_rgb_mag, -N, axis=None)[-N:] # extract top N indices
    # achieve the corresponding point cloud

    grad_depth_y = filters.sobel_h(depth_image_cpu)
    grad_depth_x = filters.sobel_v(depth_image_cpu)
    grad_depth_mag = np.sqrt(grad_depth_y**2 + grad_depth_x**2)
    selected_index_depth = np.argpartition(grad_depth_mag, -N, axis=None)[-N:] # extract top 10*N indices
    # combine rgb and depth indices and remove duplicates
    selected_index = np.unique(np.concatenate((selected_rgb_index, selected_index_depth)))
    # add the random indices
    random_index = np.random.choice(height * width, 100_000, replace=False)
    selected_index = np.unique(np.concatenate((selected_index, random_index)))

    pts = pts[selected_index]
    grad_rgb_mag = torch.from_numpy(grad_rgb_mag.reshape(-1)[selected_index]).cuda().float()
    color = torch.from_numpy(rgb_image_cpu.reshape(-1, 3)[selected_index]).cuda().float()
    # cat pts and color
    pts_color = torch.cat((pts, color), dim=1)
    # set grad_rgb_mag min as 10e-2
    grad_rgb_mag[grad_rgb_mag < 0.01] = 0.01
    scale_initialized = 1 / (grad_rgb_mag * 1_000_000)


    return pts_color, scale_initialized

C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def SH2RGB(sh):
    return sh * C0 + 0.5
