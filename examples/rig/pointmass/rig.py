from multiworld.envs.pygame.point2d import Point2DWallEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.rig_experiments import grill_her_td3_full_experiment

if __name__ == "__main__":
    variant = dict(
        imsize=84,
        grill_variant=dict(
            save_video=True,
            save_video_period=50,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            algo_kwargs=dict(
                td3_kwargs=dict(
                    num_epochs=105,
                    num_steps_per_epoch=1000,
                    num_steps_per_eval=1000,
                    min_num_steps_before_training=4000,
                    batch_size=128,
                    max_path_length=100,
                    discount=0.99,
                    num_updates_per_env_step=4,
                    reward_scale=1,
                    tau=1e-2,
                ),
                her_kwargs=dict(),
            ),
            replay_buffer_kwargs=dict(
                max_size=int(1e6),
                fraction_goals_rollout_goals=0.1,
                fraction_goals_env_goals=0.5,
            ),
            normalize=False,
            render=False,
            exploration_noise=0.2,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            )
        ),
        train_vae_variant=dict(
            vae_path=None,
            representation_size=4,
            beta=10.0,
            num_epochs=101,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            generate_vae_dataset_kwargs=dict(
                test_p=.9,
                N=1000,
                oracle_dataset_using_set_to_goal=False,
                random_rollout_data=True,
                use_cached=False,
                vae_dataset_specific_kwargs=dict(
                ),
                show=False,
            ),
            vae_kwargs=dict(
                input_channels=3,
            ),
            algo_kwargs=dict(
                do_scatterplot=False,
                batch_size=128,
                lr=1e-3,
            ),
            save_period=5,
        ),

        env_class=Point2DWallEnv,
        env_kwargs=dict(
            render_onscreen=False,
            ball_radius=1,
            images_are_rgb=True,
            show_goal=False,
        ),

        algorithm='RIG',
    )

    run_experiment(
        grill_her_td3_full_experiment,
        exp_prefix='rlkit-pointmass-rig-example',
        mode='here_no_doodad',
        variant=variant,
        # use_gpu=True,  # Turn on if you have a GPU
    )
