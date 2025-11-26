from os.path import join as p_join

primary_device = "cuda:0"

group_name = "SLAM_Replica_Tuning"
run_name = "Half_Res_R0_GS_Old_check"

config = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    seed=0,
    primary_device=primary_device,
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    report_iter_progress=False,
    use_wandb=False,
    wandb=dict(
        entity="theairlab",
        project="4DTrack",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        basedir="/home/jay/4DTrack/data/Replica",
        gradslam_data_cfg="/home/jay/4DTrack/configs/data/replica.yaml",
        sequence="room0",
        desired_image_height=340, #680,
        desired_image_width=600, #1200,
        start=0,
        end=-1,
        stride=20,
        num_frames=100,
        eval_stride=10,
        eval_num_frames=200,
        param_ckpt_path='/home/jay/4DTrack/experiments/SLAM_Replica_Tuning/Half_Res_R0/params.npz'
    ),
    train=dict(
        num_iters_mapping=0,
        sil_thres=0.5, # For Addition of new Gaussians & Visualization
        use_sil_for_loss=True, # Use Silhouette for Loss during Tracking
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs_mapping=dict(
            means3D=0.00032,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.005,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        lrs_mapping_means3D_final=0.0000032,
        lr_delay_mult=0.01,
        use_gaussian_splatting_densification=True, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=500,
            remove_big_after=3000,
            stop_after=15000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=True,
            reset_opacities_every=3000, # Doesn't consider iter 0
        ),
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth' or 'centers']
        custom_viz_cam=True, # [False or True], Uses the camera from params if False
        use_differentiable_depth_renderer=True, # [False or True]
        # additional_lines='trajectories', # [None, 'trajectories' or 'rotations']
        additional_lines=None,
        force_loop=False, # [False or True]
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=3.9,
        fps=2,
        traj_frac=25, # 4% of points
        traj_length=1, # Code setup only to show 1 frame trajectory
        frame_skip=1, # Number of frames to skip
        visualize_final_only=True, # Visualize only the final frame
    ),
)