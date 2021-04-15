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
            max_path_length=150,
            use_image_obs=False,
            reward_scale=1,  # let VecNormalize handle the reward scales
            use_dm_backend=True,
        ),
        actor_kwargs=dict(recurrent=False, hidden_size=64, hidden_activation="tanh"),
        num_processes=16,
        num_env_steps=int(5e6),
        num_steps=2048,
        log_interval=1,
        eval_interval=1,
        use_raw_actions=True,
        env_suite="metaworld",
        use_linear_lr_decay=True,
    )

    search_space = {
        # "algorithm_kwargs.entropy_coef": [1e-2, 5e-3, 0],
        # "algorithm_kwargs.lr": [3e-4, 5e-4],
        # "actor_kwargs.hidden_size": [64, 128],
        # "num_steps": [2048, 4096],
        # "use_dm_backend": [True, False],
        "env_class": [
            # "assembly-v2",
            # "basketball-v2",
            # "bin-picking-v2",
            # "box-close-v2",
            # "button-press-topdown-v2",
            # "button-press-topdown-wall-v2",
            # "button-press-v2",
            # "button-press-wall-v2",
            # "coffee-button-v2",
            # "coffee-pull-v2",
            # "coffee-push-v2",
            # "dial-turn-v2",
            # "disassemble-v2",
            # "door-close-v2",
            # "door-lock-v2",
            # "door-open-v2",
            # "door-unlock-v2",
            # "hand-insert-v2",
            # "drawer-close-v2",
            # "drawer-open-v2",
            # "faucet-open-v2",
            # "faucet-close-v2",
            # "hammer-v2",
            # "handle-press-side-v2",
            # "handle-press-v2",
            # "handle-pull-side-v2",
            # "handle-pull-v2",
            # "lever-pull-v2",
            # "peg-insert-side-v2",
            # "pick-place-wall-v2",
            # "pick-out-of-hole-v2",
            # "reach-v2",
            # "push-back-v2",
            # "push-v2",
            # "pick-place-v2",
            # "plate-slide-v2",
            # "plate-slide-side-v2",
            # "plate-slide-back-v2",
            # "plate-slide-back-side-v2",
            # "peg-unplug-side-v2",
            # "soccer-v2",
            # "stick-push-v2",
            # "stick-pull-v2",
            # "push-wall-v2",
            # "reach-wall-v2",
            # "shelf-place-v2",
            # "sweep-into-v2",
            # "sweep-v2",
            # "window-open-v2",
            # "window-close-v2",
            # v1:
            "reach-v1",
            "push-v1",
            "pick-place-v1",
            "door-open-v1",
            "drawer-open-v1",
            "drawer-close-v1",
            "button-press-topdown-v1",
            "peg-insert-side-v1",
            "window-open-v1",
            "window-close-v1",
            "door-close-v1",
            "reach-wall-v1",
            "pick-place-wall-v1",
            "push-wall-v1",
            "button-press-v1",
            "button-press-topdown-wall-v1",
            "button-press-wall-v1",
            "peg-unplug-side-v1",
            "disassemble-v1",
            "hammer-v1",
            "plate-slide-v1",
            "plate-slide-side-v1",
            "plate-slide-back-v1",
            "plate-slide-back-side-v1",
            "handle-press-v1",
            "handle-pull-v1",
            "handle-press-side-v1",
            "handle-pull-side-v1",
            "stick-push-v1",
            "stick-pull-v1",
            "basketball-v1",
            "soccer-v1",
            "faucet-open-v1",
            "faucet-close-v1",
            "coffee-push-v1",
            "coffee-pull-v1",
            "coffee-button-v1",
            "sweep-v1",
            "sweep-into-v1",
            "pick-out-of-hole-v1",
            "assembly-v1",
            "shelf-place-v1",
            "push-back-v1",
            "lever-pull-v1",
            "dial-turn-v1",
            "bin-picking-v1",
            "box-close-v1",
            "hand-insert-v1",
            "door-lock-v1",
            "door-unlock-v1",
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
