from rlkit.torch.model_based.dreamer.experiments.arguments import get_args
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    preprocess_variant_llraps,
    setup_sweep_and_launch_exp,
)
from rlkit.torch.model_based.dreamer.experiments.ll_raps_experiment import experiment

if __name__ == "__main__":
    args = get_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=5,
            num_eval_steps_per_epoch=10,
            num_expl_steps_per_train_loop=50,
            min_num_steps_before_training=10,
            num_pretrain_steps=10,
            num_train_loops_per_epoch=1,
            num_trains_per_train_loop=1,
            batch_size=200,
        )
    else:
        algorithm_kwargs = dict(
            num_epochs=500,
            num_eval_steps_per_epoch=30,
            min_num_steps_before_training=2500,
            num_pretrain_steps=100,
            batch_size=200,
            num_expl_steps_per_train_loop=60,
            num_train_loops_per_epoch=20,
            num_trains_per_train_loop=20,
        )
    variant = dict(
        algorithm="LLRAPS",
        version="normal",
        algorithm_kwargs=algorithm_kwargs,
        env_suite="metaworld",
        env_name="disassemble-v2",
        env_kwargs=dict(
            use_image_obs=True,
            imwidth=64,
            imheight=64,
            reward_type="sparse",
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                unflatten_images=False,
            ),
            action_space_kwargs=dict(
                collect_primitives_info=True,
                render_intermediate_obs_to_info=True,
                control_mode="primitives",
                action_scale=1,
                camera_settings={
                    "distance": 0.38227044687537043,
                    "lookat": [0.21052547, 0.32329237, 0.587819],
                    "azimuth": 141.328125,
                    "elevation": -53.203125160653144,
                },
            ),
        ),
        actor_kwargs=dict(
            discrete_continuous_dist=True,
            init_std=0.0,
            num_layers=4,
            min_std=0.1,
            dist="tanh_normal_dreamer_v1",
        ),
        vf_kwargs=dict(
            num_layers=3,
        ),
        model_kwargs=dict(
            model_hidden_size=400,
            stochastic_state_size=50,
            deterministic_state_size=200,
            rssm_hidden_size=200,
            reward_num_layers=2,
            pred_discount_num_layers=3,
            gru_layer_norm=True,
            std_act="sigmoid2",
            depth=32,
            use_prior_instead_of_posterior=True,
        ),
        trainer_kwargs=dict(
            adam_eps=1e-5,
            discount=0.8,
            lam=0.95,
            forward_kl=False,
            free_nats=1.0,
            pred_discount_loss_scale=10.0,
            kl_loss_scale=0.0,
            transition_loss_scale=0.8,
            actor_lr=8e-5,
            vf_lr=8e-5,
            world_model_lr=3e-4,
            reward_loss_scale=2.0,
            use_pred_discount=True,
            policy_gradient_loss_scale=1.0,
            actor_entropy_loss_schedule="1e-4",
            target_update_period=100,
            detach_rewards=False,
            imagination_horizon=5,
        ),
        replay_buffer_kwargs=dict(
            prioritize_fraction=0.0,
            uniform_priorities=True,
            replace=False,
        ),
        primitive_model_kwargs=dict(
            hidden_sizes=[512, 512],
            apply_embedding=False,
        ),
        num_expl_envs=10,
        num_eval_envs=1,
        expl_amount=0.3,
        save_video=True,
        low_level_action_dim=9,
        num_low_level_actions_per_primitive=5,
        effective_batch_size=400,
        pass_render_kwargs=True,
        max_path_length=5,
    )

    setup_sweep_and_launch_exp(preprocess_variant_llraps, variant, experiment, args)
