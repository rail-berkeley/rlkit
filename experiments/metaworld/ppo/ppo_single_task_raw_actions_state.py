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
            control_mode="end_effector",
            use_combined_action_space=False,
            action_scale=1 / 100,
            max_path_length=150,
            use_image_obs=False,
        ),
        num_processes=16,
        num_env_steps=int(2e6),
        num_steps=2048 // 16,
        log_interval=5,
        eval_interval=5,
        use_raw_actions=True,
        env_suite="metaworld",
    )

    search_space = {
        "env_class": [
            "assembly-v2",
            "basketball-v2",
            "bin-picking-v2",
            "box-close-v2",
            "button-press-topdown-v2",
            "button-press-topdown-wall-v2",
            "button-press-v2",
            "button-press-wall-v2",
            "coffee-button-v2",
            "coffee-pull-v2",
            "coffee-push-v2",
            "dial-turn-v2",
            "disassemble-v2",
            "door-close-v2",
            "door-lock-v2",
            "door-open-v2",
            "door-unlock-v2",
            "hand-insert-v2",  # no goal
            "drawer-close-v2",  # no goal
            "drawer-open-v2",  # no goal
            "faucet-open-v2",  # no goal
            "faucet-close-v2",  # no goal
            "hammer-v2",
            "handle-press-side-v2",
            "handle-press-v2",
            "handle-pull-side-v2",
            "handle-pull-v2",
            "lever-pull-v2",
            "peg-insert-side-v2",
            "pick-place-wall-v2",
            "pick-out-of-hole-v2",
            "reach-v2",
            "push-back-v2",
            "push-v2",
            "pick-place-v2",
            "plate-slide-v2",
            "plate-slide-side-v2",
            "plate-slide-back-v2",
            "plate-slide-back-side-v2",
            "peg-unplug-side-v2",
            "soccer-v2",
            "stick-push-v2",
            "stick-pull-v2",
            "push-wall-v2",
            "reach-wall-v2",
            "shelf-place-v2",
            "sweep-into-v2",
            "sweep-v2",
            "window-open-v2",  # no goal
            "window-close-v2",  # no goal
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
