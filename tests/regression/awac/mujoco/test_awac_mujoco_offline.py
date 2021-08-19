"""Test AWAC online on Mujoco benchmark tasks.

Data available for download:
https://drive.google.com/file/d/1edcuicVv2d-PqH1aZUVbO5CeRq3lqK89/view
"""

from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.launchers.experiments.awac.awac_rl import experiment, process_args

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.torch.sac.policies import GaussianPolicy

from rlkit.testing.stub_classes import StubEnv

def main():
    variant = dict(
        algo_kwargs=dict(
            num_epochs=501,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            batch_size=1024,
        ),
        max_path_length=1000,
        replay_buffer_size=int(1E6),
        layer_size=256,
        num_layers=2,
        algorithm="AWAC",
        version="normal",
        collection_mode='batch',

        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256] * 4,
            max_log_std=0,
            min_log_std=-6,
            std_architecture="values",
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256]
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            beta=1,
            alpha=0,
            use_automatic_entropy_tuning=False,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=50000,
            policy_weight_decay=1e-4,
            train_bc_on_rl_buffer=False,
            buffer_policy_sample_actions=False,

            reparam_weight=0.0,
            awr_weight=1.0,
            bc_weight=0.0,
            compute_bc=False,
            awr_use_mle_for_vf=False,
            awr_sample_actions=False,
            awr_min_q=True,
        ),
        path_loader_kwargs=dict(
            demo_paths=[  # these can be loaded in awac_rl.py per env
                # dict(
                #     path='demos/ant_action_noise_15.npy',
                #     obs_dict=False,
                #     is_demo=True,
                #     train_split=.9,
                # ),
            ],
        ),
        path_loader_class=DictToMDPPathLoader,

        pretrain_rl=True,
        use_validation_buffer=True,
        add_env_demos=True,
        add_env_offpolicy_data=True,
        load_demos=True,
    )

    search_space = {
        'trainer_kwargs.beta':[2, ],
        'train_rl':[True],
        'pretrain_rl':[True],
        'pretrain_policy':[False],
        # 'env_id': ['HalfCheetah-v2', 'Ant-v2', 'Walker2d-v2', ],
        'seedid': range(5),

        # make experiment short and only offline
        'batch_size': [5],
        'num_epochs': [0],
        'pretraining_logging_period': [1],
        'trainer_kwargs.q_num_pretrain2_steps': [10],
        'path_loader_kwargs.demo_paths': [
            [dict(
                    path=os.getcwd() + "/tests/regression/awac/mujoco/hc_action_noise_15.npy",
                    obs_dict=False, # misleading but this arg is really "unwrap_obs_dict"
                    is_demo=True,
                    data_split=1,
            ),],
        ],
        'env_class': [StubEnv], # replaces half-cheetah
        'env_kwargs': [dict(
            obs_dim=17,
            action_dim=6,
        ),],
        'add_env_demos': [False],
        'add_env_offpolicy_data': [False],
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

from rlkit.core import logger
from rlkit.testing import csv_util

def test_awac_mujoco_offline():
    cmd = "python experiments/references/awac/mujoco/awac_offline1.py --1 --local --gpu --run 0 --seed 0 --debug"
    sys.argv = cmd.split(" ")[1:]
    main()

    # check if offline training results matches
    reference_csv = "tests/regression/awac/mujoco/id0_offline/pretrain_q.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "pretrain_q.csv")
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["trainer/batch", "trainer/Advantage Score Max", "trainer/Q1 Predictions Mean", "trainer/replay_buffer_len"]
    csv_util.check_equal(reference, output, keys)

if __name__ == "__main__":
    test_awac_mujoco_offline()
