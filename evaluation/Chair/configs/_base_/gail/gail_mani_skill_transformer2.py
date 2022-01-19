_base_ = ['./gail.py']
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
    type='GAIL',
    batch_size=1024,
    discrim_batch = 1024,
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
            type='PointNetWithInstanceInfoV0',
            stack_frame=stack_frame,
            num_objs='num_objs',
            pcd_pn_cfg=dict(
                type='PointNetV0',
                conv_cfg=dict(
                    type='ConvMLP',
                    norm_cfg=None,
                    mlp_spec=['agent_shape + pcd_xyz_rgb_channel', 192, 192],
                    bias='auto',
                    inactivated_output=True,
                    conv_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                ),
                mlp_cfg=dict(
                    type='LinearMLP',
                    norm_cfg=None,
                    mlp_spec=[192 * stack_frame, 192, 192],
                    bias='auto',
                    inactivated_output=True,
                    linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                ),
                subtract_mean_coords=True,
                max_mean_mix_aggregation=True
            ),
            state_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape', 192, 192],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),                            
            transformer_cfg=dict(
                type='TransformerEncoder',
                block_cfg=dict(
                    attention_cfg=dict(
                        type='MultiHeadSelfAttention',
                        embed_dim=192,
                        num_heads=num_heads,
                        latent_dim=32,
                        dropout=0.1,
                    ),
                    mlp_cfg=dict(
                        type='LinearMLP',
                        norm_cfg=None,
                        mlp_spec=[192, 768, 192],
                        bias='auto',
                        inactivated_output=True,
                        linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                    ),
                    dropout=0.1,
                ),
                pooling_cfg=dict(
                    embed_dim=192,
                    num_heads=num_heads,
                    latent_dim=32,
                ),
                mlp_cfg=None,
                num_blocks=2,
            ),
            final_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[192, 128, 'action_shape * 2'],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
        ),
        optim_cfg=dict(type='Adam', lr=3e-4, weight_decay=5e-6),
    ),
    value_cfg=dict(
        type='ContinuousValue',
        num_heads=2,
        nn_cfg=dict(
            type='PointNetWithInstanceInfoV0',
            stack_frame=stack_frame,
            num_objs='num_objs',
            pcd_pn_cfg=dict(
                type='PointNetV0',
                conv_cfg=dict(
                    type='ConvMLP',
                    norm_cfg=None,
                    mlp_spec=['agent_shape + pcd_xyz_rgb_channel + action_shape', 192, 192],
                    bias='auto',
                    inactivated_output=True,
                    conv_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                ),
                mlp_cfg=dict(
                    type='LinearMLP',
                    norm_cfg=None,
                    mlp_spec=[192 * stack_frame, 192, 192],
                    bias='auto',
                    inactivated_output=True,
                    linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                ),
                subtract_mean_coords=True,
                max_mean_mix_aggregation=True
            ),
            state_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + action_shape', 192, 192],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),                            
            transformer_cfg=dict(
                type='TransformerEncoder',
                block_cfg=dict(
                    attention_cfg=dict(
                        type='MultiHeadSelfAttention',
                        embed_dim=192,
                        num_heads=num_heads,
                        latent_dim=32,
                        dropout=0.1,
                    ),
                    mlp_cfg=dict(
                        type='LinearMLP',
                        norm_cfg=None,
                        mlp_spec=[192, 768, 192],
                        bias='auto',
                        inactivated_output=True,
                        linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
                    ),
                    dropout=0.1,
                ),
                pooling_cfg=dict(
                    embed_dim=192,
                    num_heads=num_heads,
                    latent_dim=32,
                ),
                mlp_cfg=None,
                num_blocks=2,
            ),
            final_mlp_cfg=dict(
                type='LinearMLP',
                norm_cfg=None,
                mlp_spec=[192, 128, 1],
                bias='auto',
                inactivated_output=True,
                linear_init_cfg=dict(type='xavier_init', gain=1, bias=0),
            ),
        ),
        optim_cfg=dict(type='Adam', lr=5e-4, weight_decay=5e-6),
    ),
    discriminator_cfg = dict(
        type='ContinuousValue',
        num_heads=1,
        nn_cfg=dict(
            type='PointNetV0',
            conv_cfg=dict(
                type='ConvMLP',
                norm_cfg=None,
                mlp_spec=['agent_shape + action_shape + pcd_all_channel ', 256, 512],
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
            with_activation=True,
        ),
        optim_cfg=dict(type='Adam', lr=5e-4, weight_decay=5e-6),
    ),
)

expert_replay_cfg = dict(
    type='ReplayMemory',
    capacity=600000,
)

replay_cfg = dict(
    type='ReplayMemory',
    capacity=600000,
)
tmp_replay_cfg = dict(
    type='ReplayMemory',
    capacity=6000,
)

train_mfrl_cfg = dict(
    discrim_steps = 2,
    rl_steps = 1,
    total_steps=3000000,
    warm_steps=4000,
    n_eval=3000000,
    n_checkpoint=100000,
    n_steps=8,
    n_updates=4,
    m_steps=8,
    expert_replay_buffers=[
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3001-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3003-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3005-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3008-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3010-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3013-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3016-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3020-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3021-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3022-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3024-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3025-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3027-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3030-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3031-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3032-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3038-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3045-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3047-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3050-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3051-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3063-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3070-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3071-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3073-v0.h5',
        '/home/shenhao/ManiSkill-Learn/chair_data/PushChair_3076-v0.h5'
    ],
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
