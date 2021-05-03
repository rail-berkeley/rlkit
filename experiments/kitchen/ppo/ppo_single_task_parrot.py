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
            gamma=0.99,
            gae_lambda=0.95,
            use_proper_time_limits=True,
        ),
        env_kwargs=dict(
            dense=False,
            image_obs=True,
            fixed_schema=False,
            action_scale=1,
            use_combined_action_space=True,
            proprioception=False,
            wrist_cam_concat_with_fixed_view=False,
            use_wrist_cam=False,
            normalize_proprioception_obs=True,
            use_workspace_limits=True,
            max_path_length=280,
            control_mode="joint_velocity",
            frame_skip=40,
            imwidth=84,
            imheight=84,
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=True,
                max_path_length=280,
                unflatten_images=True,
                use_real_nvp_wrappers=True,
                model_path="/home/mdalal/research/rlkit/data/railrl-realNVP-slide-cabinet-pixels/railrl-realNVP-slide_cabinet-pixels_2021_05_01_20_05_28_id719641--s10/itr_200.pkl",
            ),
            image_kwargs=dict(),
        ),
        actor_kwargs=dict(recurrent=False, hidden_size=512, hidden_activation="relu"),
        num_processes=5,
        num_env_steps=int(1e6),
        num_steps=2048 // 5,
        log_interval=1,
        eval_interval=1,
        use_raw_actions=True,
        env_suite="kitchen",
        use_linear_lr_decay=False,
    )

    search_space = {
        "env_name": [
            "microwave",
            "kettle",
            "slide_cabinet",
            "top_left_burner",
            "hinge_cabinet",
            "light_switch",
            "microwave_kettle_light_top_left_burner",
            "hinge_slide_bottom_left_burner_light",
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
            )
