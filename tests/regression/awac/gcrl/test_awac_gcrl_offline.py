"""Test algorithm without requiring env setup. Incomplete due to reliance on env.sample_goals"""

from rlkit.core import logger
from rlkit.testing import csv_util


from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.launchers.experiments.awac.awac_gcrl import experiment, process_args

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.torch.sac.policies import GaussianPolicy

from rlkit.testing.stub_classes import StubEnv, StubMultiEnv
# from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_leap import SawyerPushAndReachXYEnv

def main():
    variant = dict(
        num_epochs=501,
        num_eval_steps_per_epoch=1000,
        num_trains_per_train_loop=4000,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=200,
        batch_size=1024,

        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.5,
        ),
        demo_replay_buffer_kwargs=dict(
            fraction_goals_rollout_goals=1.0,
            fraction_goals_env_goals=0.0,
        ),

        layer_size=256,
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256, 256],
            max_log_std=0,
            min_log_std=-4,
            std_architecture="shared",
            # num_gaussians=1,
        ),

        algorithm="SAC",
        version="normal",
        collection_mode='batch',
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            beta=1,
            use_automatic_entropy_tuning=False,
            alpha=0,

            bc_num_pretrain_steps=0,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=25000,
            policy_weight_decay=1e-4,
            q_weight_decay=0,

            rl_weight=1.0,
            use_awr_update=True,
            use_reparam_update=False,
            compute_bc=True,
            reparam_weight=0.0,
            awr_weight=1.0,
            bc_weight=0.0,

            reward_transform_kwargs=None, # r' = r + 1
            terminal_transform_kwargs=None, # t = 0
        ),
        num_exps_per_instance=1,
        region='us-west-2',

        path_loader_class=DictToMDPPathLoader,
        path_loader_kwargs=dict(
            demo_paths=[
                dict(
                    path="ashvin/icml2020/pusher/state2/random2/run12/id*/video_*_vae.p",
                    sync_dir="ashvin/icml2020/pusher/state2/random2/run12",
                    obs_dict=False, # misleading but this arg is really "unwrap_obs_dict"
                    is_demo=True,
                    data_split=0.1,
                ),
                # dict(
                #     path="demos/icml2020/hand/pen_bc5.npy",
                #     obs_dict=False,
                #     is_demo=False,
                #     train_split=0.9,
                # ),
            ],
        ),
        add_env_demos=False,
        add_env_offpolicy_data=False,

        # logger_variant=dict(
        #     tensorboard=True,
        # ),
        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,
        # save_pretrained_algorithm=True,
        # snapshot_mode="all",
        env_class=StubMultiEnv,
        env_kwargs=dict(
            obs_dims=dict(
                state_observation=4,
                state_desired_goal=4,
                state_achieved_goal=4,
            ),
            action_dim=2,
        ),

        observation_key="state_observation",
        desired_goal_key="state_desired_goal",
        achieved_goal_key="state_achieved_goal",
    )


    search_space = {
        'seedid': range(5),
        'trainer_kwargs.beta': [0.001, ],
        'num_trains_per_train_loop': [4000],
        'env_kwargs.reward_type': ['puck_distance', ],
        'policy_kwargs.min_log_std': [-6, ],
        # 'trainer_kwargs.bc_weight': [0, 1],

        # env-specific hacks
        'replay_buffer_kwargs.fraction_goals_env_goals': [0.0],
        'replay_buffer_kwargs.recompute_rewards': [False],
        'demo_replay_buffer_kwargs.recompute_rewards': [False],

        # make experiment short and only offline
        'batch_size': [5],
        'num_epochs': [0],
        'pretraining_logging_period': [1],
        'trainer_kwargs.q_num_pretrain2_steps': [10],
        'path_loader_kwargs.demo_paths': [
            [dict(
                    path=os.getcwd() + "/tests/regression/awac/gcrl/gcrl_data_mini/id0/video_0_vae.p",
                    obs_dict=False, # misleading but this arg is really "unwrap_obs_dict"
                    is_demo=True,
                    data_split=1,
            ),],
        ],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(experiment, variants, process_args)

import os
import sys
def test_awac_gcrl_online():
    cmd = "python experiments/references/awac/gcrl/pusher_offline1.py --1 --local --gpu --run 0 --seed 0"
    sys.argv = cmd.split(" ")[1:]
    main()
    reference_csv = "tests/regression/awac/gcrl/id0_offline/pretrain_q.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "pretrain_q.csv")
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["trainer/batch", "trainer/Advantage Score Max", "trainer/Q1 Predictions Mean", "trainer/replay_buffer_len"]
    csv_util.check_equal(reference, output, keys)

if __name__ == "__main__":
    test_awac_gcrl_online()
