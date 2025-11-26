import os
from os.path import join as p_join

scenes = ["rgbd_dataset_freiburg1_desk",
          "rgbd_dataset_freiburg1_desk2",
          "rgbd_dataset_freiburg1_room",
          "rgbd_dataset_freiburg2_xyz",
          "rgbd_dataset_freiburg3_long_office_household"]
configs = [
"freiburg1_desk",
"freiburg1_desk2",
"freiburg1_room",
"freiburg2_xyz",
"freiburg3_long_office_household"
]
primary_device="cuda:0"
seed = 2
scene_name = scenes[seed]
config_name = configs[seed]

map_every = 1
keyframe_every = 5
mapping_window_size = 30
tracking_iters = 200
mapping_iters = 40
scene_radius_depth_ratio=2

group_name = "SLAM_Replica"
run_name = f"{scene_name}_{seed}_color_depth_scale_reg"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=seed,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=500, # Report Global Progress every nth frame
    eval_every=5, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=scene_radius_depth_ratio, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=100, # Checkpoint Interval
    use_wandb=True,
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
        # basedir="/home/shuo/projects/shuo/implicit_reconstruction/SplaTAM/data/Replica",
        basedir="/data/splaTAM/data/tum",
        # gradslam_data_cfg="/home/shuo/projects/shuo/implicit_reconstruction/SplaTAM/configs/data/replica.yaml",
        gradslam_data_cfg=f"./configs/data/TUM/{config_name}.yaml",
        sequence=scene_name,
        desired_image_height=480,
        desired_image_width=640,
        start=0,
        end=-1,
        stride=1,
        num_frames=-1,
    ),
    tracking=dict(
        use_gt_poses=False, # Use GT Poses for Tracking
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=False,
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
            cam_unnorm_rots=0.002,
            cam_trans=0.002,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        single_frame_iterations=20,
        add_new_gaussians=True,
        sil_thres=0.5, # For Addition of new Gaussians
        use_l1=True,
        use_sil_for_loss=False,
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
            # means3D=0.0000,
            rgb_colors=0.0025,
            # rgb_colors=0.00,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            # log_scales=0.015,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        reg_loss=dict(
            color_reg = True,
            depth_reg = True,
            scale_reg = True,
        ),
        prune_gaussians=False, # Prune Gaussians during Mapping
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
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=True, # Visualize Camera Frustums and Trajectory
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=False, # Enter Interactive Mode after Online Recon Viz
    ),
)
