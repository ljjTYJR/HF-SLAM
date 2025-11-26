from os.path import join as p_join

primary_device = "cuda:0"

group_name = "Tracking"
run_name = "Base_Forward_Prop"

config = dict(
    workdir=f"/home/airlab/nkeetha/4d/experiments/{group_name}",
    run_name=run_name,
    seed=42,
    primary_device=primary_device,
    report_progress=False,
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
        sequence="room0",
        desired_image_height=340,
        desired_image_width=600,
        start=0,
        end=-1,
        stride=1,
        num_frames=2000,
        scene_path="/home/airlab/nkeetha/4d/experiments/3DGS/Splatting_30k/params.npz",
    ),
    tracking=dict(
        use_gt_poses=False, # Use GT Poses for Tracking
        forward_prop=True, # Forward Propagate Poses
        num_iters=50,
        use_sil_for_loss=True,
        sil_thres=0.5,
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
    viz=dict(
        render_mode='centers', # ['color', 'depth' or 'centers']
        custom_viz_cam=False, # [False or True], Uses the camera from params if False
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
        visualize_final_only=False, # Visualize only the final frame
    ),
)