_base_ = ['./curl.py']
stack_frame = 1
num_heads = 4

env_cfg = dict(
    type='gym',
    unwrapped=False,
    obs_mode='pointcloud',
    reward_type='dense',
    stack_frame=stack_frame
)


agent = dict(
    type='CURL',
    batch_size=1024,
    gamma=0.95,
    update_coeff=0.01,
    policy_cfg=dict(
        type='ContinuousPolicy',
        if_contrast=True,
        policy_head_cfg=dict(
            type='GaussianHead',
            log_sig_min=-20,
            log_sig_max=2,
            epsilon=1e-6
        ),
        
        encoder_cfg=dict(
            type='PointNetV0',
            conv_cfg=dict(
                type='ConvMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + pcd_all_channel', 256, 512],
                bias='auto',
                inactivated_output=False,
                conv_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[512 * stack_frame, 256, 192],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            subtract_mean_coords=True,
            max_mean_mix_aggregation=True,
            stack_frame=stack_frame,
        ),
        nn_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=[192 , 256, 256, 256, 'action_shape * 2'],
            bias='auto',
            inactivated_output=True,
            linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
        ),
        optim_cfg=dict(type='Adam', lr=3e-4, weight_decay=5e-6),
    ),

    value_cfg=dict(
        type='ContinuousValue',
        num_heads=2,
        if_contrast=True,
        encoder_cfg=dict(
            type='PointNetV0',
            conv_cfg=dict(
                type='ConvMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + pcd_all_channel', 256, 512],
                bias='auto',
                inactivated_output=False,
                conv_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[512 * stack_frame, 256, 192],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            subtract_mean_coords=True,
            max_mean_mix_aggregation=True,
            stack_frame=stack_frame,
        ),
        nn_cfg=dict(
            type='LinearMLP',
            norm_cfg=None,
            mlp_spec=['192 + action_shape', 256, 256, 256, 1],
            bias='auto',
            inactivated_output=True,
            linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
        ),
        optim_cfg=dict(type='Adam', lr=5e-4, weight_decay=5e-6),
    ),
)

expert = dict(
    type='SAC',
    batch_size=1024,
    gamma=0.95,
    policy_cfg=dict(
        type='ContinuousPolicy',
        policy_head_cfg=dict(
            type='GaussianHead',
            log_sig_min=-20,
            log_sig_max=2,
            epsilon=1e-6
        ),
        nn_cfg=dict(
            type='PointNetV0',
            conv_cfg=dict(
                type='ConvMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + pcd_all_channel', 256, 512],
                bias='auto',
                inactivated_output=False,
                conv_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[512 * stack_frame, 256, 'action_shape * 2'],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            subtract_mean_coords=True,
            max_mean_mix_aggregation=True,
            stack_frame=stack_frame,
        ),
        optim_cfg=dict(type='Adam', lr=5e-4),
    ),
    value_cfg=dict(
        type='ContinuousValue',
        num_heads=2,
        nn_cfg=dict(
            type='PointNetV0',
            conv_cfg=dict(
                type='ConvMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + action_shape + pcd_all_channel', 256, 512],
                bias='auto',
                inactivated_output=False,
                conv_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[512 * stack_frame, 256, 1],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(
                    type='xavier_init',
                    gain=1,
                    bias=0,
                )
            ),
            subtract_mean_coords=True,
            max_mean_mix_aggregation=True,
            stack_frame=stack_frame,
        ),
        optim_cfg=dict(type='Adam', lr=5e-4),
    ),
)

replay_cfg = dict(
    type='ReplayMemory',
    capacity=800000,
)

train_mfrl_cfg = dict(
    total_steps=2000000,
    warm_steps=100000,
    n_eval=2000000,
    n_checkpoint=100000,
    n_steps=8,
    n_updates=4,
    m_steps=8,
)

rollout_cfg = dict(
    type='BatchRollout',
    with_info=False,
    use_cost=False,
    reward_only=False,
    num_procs=8,
)

eval_cfg = dict(
    type='BatchEvaluation',
    num=100,
    num_procs=2,
    use_hidden_state=False,
    start_state=None,
    save_traj=False,
    save_video=False,
    use_log=True,
)
