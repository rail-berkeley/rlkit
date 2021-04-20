import argparse
import os
import random

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
            entropy_coef=0,
            value_loss_coef=0.5,
            lr=3e-4,
            num_mini_batch=32,
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
            control_mode="end_effector",
            use_combined_action_space=False,
            action_scale=1 / 100,
            reward_type="dense",
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=True,
                use_image_obs=False,
                max_path_length=500,
                unflatten_images=False,
            ),
            image_kwargs=dict(),
        ),
        actor_kwargs=dict(recurrent=False, hidden_size=64, hidden_activation="tanh"),
        num_processes=12,
        num_env_steps=int(5e4),
        num_steps=2048,
        log_interval=1,
        eval_interval=1,
        env_suite="metaworld",
        use_linear_lr_decay=False,
    )

    search_space = {
        "algorithm_kwargs.entropy_coef": [1e-2],
        "num_steps": [2048 // 12],
        "algorithm_kwargs.num_mini_batch": [64],
        "env_kwargs.usage_kwargs.max_path_length": [500],
        "env_name": [
            "reach-v2",
            # "door-close-v2",
            # "sweep-into-v2",
            # "button-press-wall-v2",
            # "button-press-topdown-v2",
            # "plate-slide-v2",
            # "coffee-button-v2",
            # "handle-press-v2",
            # "window-open-v2",
            # "drawer-close-v2",
            # "handle-press-side-v2",
            # "button-press-v2",
            # "plate-slide-back-side-v2",
            # "plate-slide-side-v2",
            # "coffee-push-v2",
            # "door-unlock-v2",
            # "plate-slide-back-v2",
            # "soccer-v2",
            # "button-press-topdown-wall-v2",
            # "door-lock-v2",
            # "door-open-v2",
            # "faucet-open-v2",
            # "faucet-close-v2",
            # "handle-pull-side-v2",
            # "handle-pull-v2",
            # # "push-v2", #errors
            # # "push-wall-v2", #errors
            # "sweep-v2",
            # "window-close-v2",
            # semi-solvable:
            # "peg-insert-side-v2",
            # "reach-wall-v2",
            # "dial-turn-v2",
            # "push-back-v2",
            # "basketball-v2",
            # "box-close-v2",
            # "coffee-pull-v2",
            # "drawer-open-v2",
            # "hand-insert-v2",
            # "lever-pull-v2",
            # "pick-place-v2",
            # "stick-pull-v2",
            # # completely unsolveable
            # "stick-push-v2",
            # # "shelf-place-v2", #errors
            # # "pick-place-wall-v2", #errors
            # "pick-out-of-hole-v2",
            # "peg-unplug-side-v2",
            # "hammer-v2",
            # "disassemble-v2",
            # "bin-picking-v2",
            # "assembly-v2",
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
                python_cmd="~/miniconda3/envs/hrl-exp-env/bin/python",
                seed=seed,
                exp_id=exp_id,
            )
