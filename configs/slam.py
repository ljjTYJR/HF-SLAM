from os.path import join as p_join

primary_device = "cuda:0"

seed = 0
scene_name = "room0"

map_every = 1
keyframe_every = 5
mapping_window_size = 24
tracking_iters = 40
mapping_iters = 60

group_name = "SLAM"
run_name = f"GS_ME{map_every}_KE{keyframe_every}_MW{mapping_window_size}_T{tracking_iters}_M{mapping_iters}"

config = dict(
    workdir=f"/home/airlab/nkeetha/4d/experiments/{group_name}",
    run_name=run_name,
    seed=0,
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=5, # Report Global Progress every nth frame
    eval_every=5, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=3, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=100, # Checkpoint Interval
    use_wandb=True,
    wandb=dict(
        entity="theairlab",
        project="4DTrack",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        basedir="/home/airlab/nkeetha/4d/data/Replica",
        gradslam_data_cfg="/home/airlab/nkeetha/4d/data/Replica/gradslam_dataconfig.yaml",
        sequence=scene_name,
        desired_image_height=680,
        desired_image_width=1200,
        start=0,
        end=-1,
        stride=1,
        num_frames=2000,
    ),
    tracking=dict(
        use_gt_poses=False, # Use GT Poses for Tracking
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        use_uncertainty_for_loss_mask=True,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
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
            cam_unnorm_rots=0.0004,
            cam_trans=0.002,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5, # For Addition of new Gaussians
        use_l1=True,
        use_sil_for_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
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