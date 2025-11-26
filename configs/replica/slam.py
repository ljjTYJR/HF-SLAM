scenes = ["room0", "room1", "room2", "office0", "office1", "office2", "office_", "office4"]

# Experiment settings
seed = 0
scene = scenes[seed]
device = "cuda:0"

# SLAM scheduling
map_every = 1
keyframe_every = 5
window_size = 40
track_iters = 40
map_iters = 50

config = dict(
    workdir=f"./experiments/SLAM_Replica",
    run_name=f"{scene}_{seed}_color_depth_reg",
    seed=seed,
    primary_device=device,

    # Scheduling
    map_every=map_every,
    keyframe_every=keyframe_every,
    mapping_window_size=window_size,
    eval_every=5,

    # Scene initialization
    scene_radius_depth_ratio=3,
    mean_sq_dist_method="projective",

    # Checkpointing
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False,
    checkpoint_interval=100,

    # Dataset
    data=dict(
        basedir="data/Replica",
        gradslam_data_cfg="configs/data/replica.yaml",
        sequence=scene,
        desired_image_height=680,
        desired_image_width=1200,
        start=0,
        end=2000,
        stride=1,
        num_frames=-1,
    ),

    # Tracking (camera pose estimation)
    tracking=dict(
        use_gt_poses=False,
        forward_prop=True,
        num_iters=track_iters,
        use_sil_for_loss=False,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=True,
        loss_weights=dict(im=0.5, depth=1.0),
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

    # Mapping (3D reconstruction)
    mapping=dict(
        num_iters=map_iters,
        single_frame_iterations=20,
        add_new_gaussians=True,
        sil_thres=0.5,
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,

        # Loss weights
        loss_weights=dict(
            im=0.5,
            depth=1.0,
            color_reg=1e7,
            depth_reg=1e5,
            scale_reg=1e8,
        ),

        # Learning rates
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0,
            cam_trans=0.0,
        ),

        # Regularization
        reg_loss=dict(color_reg=True, depth_reg=True, scale_reg=True),

        # Pruning (disabled)
        prune_gaussians=False,
        pruning_dict=dict(
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500,
        ),

        # Densification (disabled)
        use_gaussian_splatting_densification=False,
        densify_dict=dict(
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000,
        ),

        # Dropout (disabled)
        drop_out=False,
        drop_out_ratio=0.7,
    ),

    # Visualization (for offline viz scripts)
    viz=dict(
        render_mode='color',
        custom_viz_cam=False,
        use_differentiable_depth_renderer=True,
        additional_lines=None,
        force_loop=False,
        viz_w=600,
        viz_h=340,
        viz_near=0.01,
        viz_far=100.0,
        view_scale=3.9,
        fps=30,
        traj_frac=25,
        traj_length=1,
        frame_skip=1,
        visualize_final_only=False,
    ),
)
