from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_nips import SawyerPushAndReachXYEasyEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.state_based_goal_experiments import her_td3_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            td3_kwargs=dict(
                num_epochs=501,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=100,
                num_updates_per_env_step=4,
                batch_size=128,
                discount=0.99,
                min_num_steps_before_training=4000,
                reward_scale=1.0,
                render=False,
                tau=1e-2,
            ),
            her_kwargs=dict(
                observation_key='state_observation',
                desired_goal_key='state_desired_goal',
            ),
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=0.1,
            fraction_goals_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        version='normal',
        es_kwargs=dict(
            max_sigma=.2,
        ),
        exploration_type='ou',
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        init_camera=sawyer_pusher_camera_upright_v2,
        do_state_exp=True,

        save_video=True,
        imsize=84,

        snapshot_mode='gap_and_last',
        snapshot_gap=50,

        env_class=SawyerPushAndReachXYEasyEnv,
        env_kwargs=dict(
            hide_goal=True,
            reward_info=dict(
                type="state_distance",
            ),
        ),

        algorithm='Oracle',
    )

    n_seeds = 1
    mode = 'here_no_doodad'
    exp_prefix = 'rlkit-pusher-oracle-from-ari-fixed-logprob'

    for _ in range(n_seeds):
        run_experiment(
            her_td3_experiment,
            exp_prefix=exp_prefix,
            mode=mode,
            variant=variant,
            # use_gpu=True,  # Turn on if you have a GPU
        )
