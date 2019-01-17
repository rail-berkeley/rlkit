import time

from chickaboomboom.databaseinterface import DatabaseInterface
import gym
from rlkit.launchers import config

import rlkit.torch.pytorch_util as ptu

from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)

from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import (
    GaussianAndEpsilonStrategy
)

from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.her.her import HerTd3
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy

from rlkit.launchers.launcher_util import run_experiment

from rlkit.launchers.rig_experiments import get_video_save_func

import rlkit.samplers.rollout_functions as rf
from rlkit.launchers.launcher_util import run_experiment_here


def experiment(variant):
    print("trying to import mujoco")
    try:
        import mujoco_py
    except ImportError as e:
        print(e)
    print("trying to import gym fetch stack")
    try:
        import gym_fetch_stack
    except ImportError as e:
        print(e)

    env = gym.make(variant['env_id'])
    es = GaussianAndEpsilonStrategy(
        action_space=env.action_space,
        max_sigma=.2,
        min_sigma=.2,  # constant sigma
        epsilon=.3,
    )

    observation_key = 'observation'
    desired_goal_key = 'desired_goal'
    achieved_goal_key = desired_goal_key.replace("desired", "achieved")

    obs_dim = (
        env.observation_space.spaces[observation_key].low.size +
        env.observation_space.spaces[desired_goal_key].low.size
    )

    goal_dim = env.observation_space.spaces['desired_goal'].low.size
    action_dim = env.action_space.low.size

    # qf1 = FlattenMlp(
    #     input_size=obs_dim + goal_dim + action_dim,
    #     output_size=1,
    #     hidden_sizes=[400, 300],
    # )

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )

    # qf2 = FlattenMlp(
    #     input_size=obs_dim + goal_dim + action_dim,
    #     output_size=1,
    #     hidden_sizes=[400, 300],
    # )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[400, 300],
    )

    # policy = TanhMlpPolicy(
    #     input_size=obs_dim + goal_dim,
    #     output_size=action_dim,
    #     hidden_sizes=[400, 300],
    # )

    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )

    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    replay_buffer = ObsDictRelabelingBuffer(
        env=env,
        observation_key=observation_key,
        desired_goal_key=desired_goal_key,
        achieved_goal_key=achieved_goal_key,
        **variant['replay_buffer_kwargs']
    )

    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['replay_buffer'] = replay_buffer

    td3_kwargs = algo_kwargs['td3_kwargs']
    td3_kwargs['training_env'] = env
    td3_kwargs['render'] = variant["render"]
    her_kwargs = algo_kwargs['her_kwargs']
    her_kwargs['observation_key'] = observation_key
    her_kwargs['desired_goal_key'] = desired_goal_key


    algorithm = HerTd3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )


    # if variant.get("save_video", True):
    #     rollout_function = rf.create_rollout_function(
    #         rf.multitask_rollout,
    #         max_path_length=algorithm.max_path_length,
    #         observation_key=algorithm.observation_key,
    #         desired_goal_key=algorithm.desired_goal_key,
    #     )
    #     video_func = get_video_save_func(
    #         rollout_function,
    #         env,
    #         algorithm.eval_policy,
    #         variant,
    #     )
    #     algorithm.post_epoch_funcs.append(video_func)

    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_kwargs=dict(
            td3_kwargs=dict(
                num_epochs=105,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                min_num_steps_before_training=500,
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
            max_size=100000,
            fraction_goals_rollout_goals=1.0,  # equal to k = 4 in HER paper
            fraction_goals_env_goals=0.0,
        ),
        algorithm='HER',
        render='False',
        save_video='True',
        do_state_exp="True",
        env_id="FetchReach-v0",
    )
    # setup_logger('her-td3-fetch-experiment', variant=variant)
    import subprocess
    exp_prefix = variant['env_id']
    run_experiment(
        experiment,
        exp_prefix=exp_prefix,
        region="us-east-2",
        mode='local_docker',
        variant=variant,
        use_gpu=False,
        spot_price=.3,
    )
    # dbi = DatabaseInterface("db.db")
    # s3_base_dir = config.AWS_S3_PATH + "/" + time.strftime("%m-%d") + "-" + exp_prefix.replace("_", "-")
    # print("s3 base dir: " + str(s3_base_dir))
    # dbi.insert("dummy.conf", s3_base_dir=s3_base_dir,
    #
    #            table_name=exp_prefix.replace("-", "_"))
    # dbi.close()

    # run_experiment_here(
    #     experiment,
    #     variant=variant,
    #     use_gpu=False)