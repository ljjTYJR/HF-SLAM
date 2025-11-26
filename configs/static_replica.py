from os.path import join as p_join

primary_device = "cuda:0"

config = dict(
    workdir="/storage2/datasets/nkeetha/4d/experiments/static_recon",
    case_name="4frame_dense_rigid",
    primary_device=primary_device,
    data=dict(
        basedir="/storage2/datasets/nkeetha/4d/data/Replica",
        gradslam_data_cfg="/storage2/datasets/nkeetha/4d/data/Replica/gradslam_dataconfig.yaml",
        sequence="room0",
        desired_image_height=340,
        desired_image_width=600,
        start=340, # Default: 0
        end=-1, # Default: -1
        stride=5,
        num_frames=4,
    ),
    train=dict(
        optimize_prior_frames=False,
        densification=False,
        use_poses=False, # Default: False
        use_rigidity_losses=False,
        learn_poses=True,
        num_iters_first_ts=2000,
        num_iters_post_first_ts=500,
        loss_weights=dict(
            im=1.0,
            depth=0.2,
            rigid=4.0,
            rot=4.0,
            iso=2.0,
            soft_col_cons=0.001,
        ),
        lrs_first_ts=dict(
            means3D=0.00016, # This value gets multiplied by scene radius
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_rots=0.0,
            cam_trans=0.0,
        ),
        # lrs_post_first_ts=dict(
        #     means3D=0.00016, # This value gets multiplied by scene radius
        #     rgb_colors=0.0025,
        #     unnorm_rotations=0.001,
        #     logit_opacities=0.05,
        #     log_scales=0.001,
        #     cam_rots=0.001,
        #     cam_trans=0.00016, # This value gets multiplied by scene radius
        # ),
        lrs_post_first_ts=dict(
            means3D=0.0, # This value gets multiplied by scene radius
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_rots=0.001,
            cam_trans=0.00016, # This value gets multiplied by scene radius
        ),
    ),
    viz=dict(
        render_mode='centers', # ['color', 'depth' or 'centers']
        custom_viz_cam=False, # [False or True], Uses the camera from params if False
        use_differentiable_depth_renderer=False, # [False or True]
        additional_lines='trajectories', # [None, 'trajectories' or 'rotations']
        # additional_lines=None,
        force_loop=False, # [False or True]
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=3.9,
        fps=0.5,
        traj_frac=25, # 4% of points
        traj_length=1,
    ),
)