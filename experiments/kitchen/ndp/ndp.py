import argparse
import os
import random
import subprocess

import numpy as np
import torch

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment


def experiment(variant):
    from ndp.main_rl import dmp_experiment

    dmp_experiment(variant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    if args.debug:
        exp_prefix = "test" + args.exp_prefix
    else:
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm_kwargs=dict(
            entropy_coef=0.01,
            value_loss_coef=0.5,
            lr=3e-4,
            num_mini_batch=32,
            ppo_epoch=10,
            clip_param=0.1,
            eps=1e-5,
            max_grad_norm=0.5,
        ),
        rollout_kwargs=dict(
            use_gae=True,
            gamma=0.99,
            gae_lambda=0.95,
            use_proper_time_limits=False,
        ),
        env_kwargs=dict(
            dense=False,
            image_obs=False,
            action_scale=1,
            control_mode="end_effector",
            frame_skip=40,
            target_mode=True,
            reward_delay=1,
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=False,
                max_path_length=280,
                unflatten_images=False,
            ),
            image_kwargs=dict(),
        ),
        num_processes=12,
        num_env_steps=int(1e7),
        num_steps=280,
        log_interval=1,
        eval_interval=5,
        use_raw_actions=True,
        env_suite="kitchen",
        use_linear_lr_decay=True,
        dmp_kwargs={
            "recurrent": False,
            "hidden_size": 100,
            "T": 5,
            "N": 5,
            "goal_type": "multi_act",
            "hidden_sizes": (100, 100),
            "state_index": (0, 1, 2, 3, 4, 5),
            "vel_index": (),
            "rbf": "gaussian",
            "a_z": 15,
            "secondary_output": True,
        },
        num_int_steps=35,
        scale=1.0,
    )

    search_space = {
        "env_name": [
            "microwave",
            "kettle",
            "slide_cabinet",
            "top_left_burner",
            "hinge_cabinet",
            "light_switch",
            # "hinge_slide_bottom_left_burner_light",
            # "microwave_kettle_light_top_left_burner",
        ],
        "dmp_kwargs.T": [5],
        "env_kwargs.reward_delay": [1],
        "dmp_kwargs.N": [10],
        "dmp_kwargs.a_z": [25],
        "algorithm_kwargs.clip_param": [0.1],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(args.num_seeds):
            seed = random.randint(0, 100000)
            variant["seed"] = seed
            variant["exp_id"] = exp_id
            run_experiment(
                experiment,
                exp_prefix=args.exp_prefix,
                mode=args.mode,
                variant=variant,
                use_gpu=False,
                snapshot_mode="none",
                python_cmd=subprocess.check_output("which python", shell=True).decode(
                    "utf-8"
                )[:-1],
                seed=seed,
                exp_id=exp_id,
            )
