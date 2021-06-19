import argparse
import os
import random
import subprocess

import numpy as np
import torch

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment


def experiment(variant):
    from a2c_ppo_acktr.main import experiment

    experiment(variant)


if __name__ == "__main__":
    cam_settings_list = [
        {
            "distance": 0.3211473534266694,
            "lookat": [0.29015772, 0.63492059, 0.544268],
            "azimuth": 178.59375,
            "elevation": -60.46875041909516,
        },
        {
            "distance": 0.513599996134662,
            "lookat": [0.28850459, 0.56757972, 0.54530015],
            "azimuth": 179.296875,
            "elevation": -47.34375002793968,
        },
        {
            "distance": 0.513599996134662,
            "lookat": [0.28839241, 0.55843923, 0.70374719],
            "azimuth": 179.82421875,
            "elevation": -59.76562483236194,
        },
        {
            "distance": 0.37864894603997346,
            "lookat": [0.28839241, 0.55843923, 0.70374719],
            "azimuth": -180.0,
            "elevation": -64.68749995809048,
        },
        {
            "distance": 0.38227044687537043,
            "lookat": [0.21052547, 0.32329237, 0.587819],
            "azimuth": 141.328125,
            "elevation": -53.203125160653144,
        },
        {
            "distance": 0.513599996134662,
            "lookat": [0.28850459, 0.56757972, 0.54530015],
            "azimuth": 178.9453125,
            "elevation": -60.00000040512532,
        },
    ]
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
            num_mini_batch=64,
            ppo_epoch=10,
            clip_param=0.2,
            eps=1e-5,
            max_grad_norm=0.5,
        ),
        rollout_kwargs=dict(
            use_gae=True,
            gamma=0.8,
            gae_lambda=0.95,
            use_proper_time_limits=True,
        ),
        env_kwargs=dict(
            control_mode="primitives",
            action_scale=1,
            max_path_length=5,
            reward_type="sparse",
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=True,
                max_path_length=5,
                unflatten_images=True,
            ),
            image_kwargs=dict(imwidth=84, imheight=84),
        ),
        actor_kwargs=dict(recurrent=False, hidden_size=512, hidden_activation="relu"),
        num_processes=10,
        num_env_steps=int(2e6),
        num_steps=2048 // 10,
        log_interval=1,
        eval_interval=1,
        use_raw_actions=False,
        env_suite="metaworld",
        discrete_continuous_dist=False,
        use_linear_lr_decay=False,
    )

    search_space = {
        "env_name": [
            "basketball-v2",
            "assembly-v2",
            # "disassemble-v2",
            # "soccer-v2",
            # "sweep-into-v2",
            # "drawer-close-v2",
        ],
        "env_kwargs.camera_settings": [
            cam_settings_list[0],
            cam_settings_list[1],
            cam_settings_list[2],
            cam_settings_list[3],
            cam_settings_list[4],
            cam_settings_list[5],
        ],
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
                use_gpu=True,
                snapshot_mode="none",
                python_cmd=subprocess.check_output("which python", shell=True).decode(
                    "utf-8"
                )[:-1],
                seed=seed,
                exp_id=exp_id,
                skip_wait=False,
            )
