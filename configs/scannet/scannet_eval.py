import os
from os.path import join as p_join

primary_device = "cuda:0"

scenes = ["scene0000_00", "scene0059_00", "scene0106_00",
          "scene0169_00", "scene0181_00", "scene0207_00",
          "scene0233_00"
          ]

# seed = int(os.environ["SEED"])
seed = int(0)
# scene_name = scenes[int(os.environ["SCENE_NUM"])]
scene_name = scenes[2]

map_every = 4 # Modify
keyframe_every = 5 # Modify
mapping_window_size = 40
tracking_iters = 100
mapping_iters = 100
scene_radius_depth_ratio = 3

group_name = "SLAM_ScanNet"
run_name = f"{scene_name}_seed{seed}"

config = dict(
    # workdir=f"/storage2/datasets/jkarhade/gaussian_slam/4DTrack/experiments/{group_name}",
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=500, # Report Global Progress every nth frame
    eval_every=5, # Evaluate every nth frame (at end of SLAM)     Modify
    scene_radius_depth_ratio=scene_radius_depth_ratio, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=100, # Checkpoint Interval
    use_wandb=False,
    sibr_viewer=dict(
        use_sibr_viewer=False,
        ip="127.0.0.1",
        port=6009,
        source_path="/media/shuo/T7/gaussian-splatting/data_for_experiments/Replica_gui_used",
    ),
    wandb=dict(
        entity="hitsshuo",
        project="GS-SLAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        # basedir="/storage2/datasets/nkeetha/4d/data/ScanNet/scans",
        basedir="/home/shuo/projects/shuo/implicit_reconstruction/SplaTAM/data/scannet",
        # basedir="/data/splaTAM/data/scannet",
        gradslam_data_cfg="/home/shuo/projects/shuo/implicit_reconstruction/SplaTAM/configs/data/scannet.yaml",
        # gradslam_data_cfg="/data/splaTAM/configs/data/scannet.yaml",
        sequence=scene_name,
        desired_image_height=480,
        desired_image_width=640,
        # desired_image_height=960,
        # desired_image_width=1280,
        start=0,
        end=-1,
        stride=1,
        num_frames=-1,
    ),
    tracking=dict(
        use_gt_poses=False, # Use GT Poses for Tracking
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=True,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.0005,
            cam_trans=0.0005,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        single_frame_iterations=20,
        add_new_gaussians=True,
        sil_thres=0.5, # For Addition of new Gaussians
        use_l1=True,
        use_sil_for_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        ignore_outlier_depth_loss=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            color_reg=1e7,
            depth_reg=1e5,
            scale_reg=1e8,
        ),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500, # Doesn't consider iter 0
        ),
        reg_loss=dict(
            color_reg = True,
            depth_reg = True,
            scale_reg = True,
        ),
        use_gaussian_splatting_densification=False, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000, # Doesn't consider iter 0
        ),
        drop_out=False, # Drop Out Gaussians during Mapping
        drop_out_ratio=0.7,
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth' or 'centers']
        custom_viz_cam=False, # [False or True], Uses the camera from params if False
        use_differentiable_depth_renderer=True, # [False or True]
        # additional_lines='trajectories', # [None, 'trajectories' or 'rotations']
        additional_lines=None,
        force_loop=False, # [False or True]
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=3.9,
        fps=20,
        traj_frac=25, # 4% of points
        traj_length=1, # Code setup only to show 1 frame trajectory
        frame_skip=1, # Number of frames to skip
        visualize_final_only=False, # Visualize only the final frame
    ),
)
