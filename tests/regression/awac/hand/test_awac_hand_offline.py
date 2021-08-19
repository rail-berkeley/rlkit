"""Test AWAC offline on Mujoco dextrous manipulation tasks.

Running the dexterous manipulation experiments requires setting up the
environments in this repository: https://github.com/aravindr93/hand_dapg.
You can also use the follwing docker image, which has the required
dependencies set up: anair17/railrl-hand-v3

For the mj_envs repository, please use: https://github.com/anair13/mj_envs

Data available for download:
https://drive.google.com/file/d/1SsVaQKZnY5UkuR78WrInp9XxTdKHbF0x/view
"""

from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.launchers.experiments.awac.awac_rl import experiment, process_args

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.arglauncher import run_variants

from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.torch.networks import Clamp

from rlkit.testing.stub_classes import StubEnv, StubMultiEnv

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
        algorithm="AWAC",
        replay_buffer_size=int(1E6),

        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, ],
            max_log_std=0,
            min_log_std=-6,
            std_architecture="values",
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256, ],
            output_activation=Clamp(max=0), # rewards are <= 0
        ),

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
            compute_bc=False,
            awr_min_q=True,

            bc_num_pretrain_steps=0,
            q_num_pretrain1_steps=0,
            q_num_pretrain2_steps=25000,
            policy_weight_decay=1e-4,
            q_weight_decay=0,

            rl_weight=1.0,
            use_awr_update=True,
            use_reparam_update=False,
            reparam_weight=0.0,
            awr_weight=1.0,
            bc_weight=0.0,

            reward_transform_kwargs=None,
            terminal_transform_kwargs=dict(m=0, b=0),
        ),
        launcher_config=dict(
            num_exps_per_instance=1,
            region='us-west-2',
        ),

        path_loader_class=DictToMDPPathLoader,
        path_loader_kwargs=dict(
            obs_key="state_observation",
            demo_paths=[  # these can be loaded in awac_rl.py per env
                # dict(
                #     path="demos/icml2020/hand/pen_bc5.npy",
                #     obs_dict=False,
                #     is_demo=False,
                #     train_split=0.9,
                # ),
            ],
        ),
        add_env_demos=True,
        add_env_offpolicy_data=True,
        normalize_env=False,

        load_demos=True,
        pretrain_policy=True,
        pretrain_rl=True,
    )

    search_space = {
        # 'env_id': ["pen-binary-v0", "door-binary-v0", "relocate-binary-v0", ],
        'seedid': range(5),
        'trainer_kwargs.beta': [0.5, ],
        'trainer_kwargs.clip_score': [0.5, ],
        'trainer_kwargs.awr_use_mle_for_vf': [True, ],

        # make experiment short and only offline
        'batch_size': [5],
        'num_epochs': [0],
        'pretraining_logging_period': [1],
        'trainer_kwargs.q_num_pretrain2_steps': [10],
        'path_loader_kwargs.demo_paths': [
            [dict(
                    path=os.getcwd() + "/tests/regression/awac/hand/pen2_sparse.npy",
                    obs_dict=True, # misleading but this arg is really "unwrap_obs_dict"
                    is_demo=True,
                    data_split=1,
            ),],
        ],
        'env_class': [StubEnv], # replaces pen
        'env_kwargs': [dict(
            obs_dim=45,
            action_dim=24,
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

def test_awac_hand_online():
    cmd = "python examples/awac/hand/awac_offline1.py --1 --local --gpu --run 0 --seed 0 --debug"
    sys.argv = cmd.split(" ")[1:]
    main()

    # check if online training results matches
    reference_csv = "tests/regression/awac/hand/id0_offline/pretrain_q.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "pretrain_q.csv")
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["trainer/batch", "trainer/Advantage Score Max", "trainer/Q1 Predictions Mean", "trainer/replay_buffer_len"]
    csv_util.check_equal(reference, output, keys)


if __name__ == "__main__":
    test_awac_hand_online()
