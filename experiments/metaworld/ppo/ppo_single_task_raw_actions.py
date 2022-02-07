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


from rlkit.torch.model_based.dreamer.experiments.arguments import get_args

if __name__ == "__main__":
    args = get_args()
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
            control_mode="end-effector",
            action_scale=1 / 100,
            max_path_length=500,
            reward_type="dense",
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=True,
                max_path_length=500,
                unflatten_images=True,
            ),
            image_kwargs=dict(),
        ),
        actor_kwargs=dict(recurrent=False, hidden_size=512, hidden_activation="relu"),
        num_processes=12,
        num_env_steps=int(1e7),
        num_steps=2048 // 12,
        log_interval=5,
        eval_interval=5,
        use_raw_actions=True,
        env_suite="metaworld",
        use_linear_lr_decay=False,
    )

    search_space = {
        "algorithm_kwargs.entropy_coef": [1e-2],
        "num_steps": [2048 // 12],
        "algorithm_kwargs.num_mini_batch": [64],
        "env_kwargs.max_path_length": [500],
        "env_class": [
            "reach-v2",
            "door-close-v2",
            "sweep-into-v2",
            "button-press-wall-v2",
            "button-press-topdown-v2",
            "plate-slide-v2",
            "coffee-button-v2",
            "handle-press-v2",
            "window-open-v2",
            "drawer-close-v2",
            "handle-press-side-v2",
            "button-press-v2",
            "plate-slide-back-side-v2",
            "plate-slide-side-v2",
            "coffee-push-v2",
            "door-unlock-v2",
            "plate-slide-back-v2",
            "soccer-v2",
            "button-press-topdown-wall-v2",
            "door-lock-v2",
            "door-open-v2",
            "faucet-open-v2",
            "faucet-close-v2",
            "handle-pull-side-v2",
            "handle-pull-v2",
            # "push-v2", #errors
            # "push-wall-v2", #errors
            "sweep-v2",
            "window-close-v2",
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
            # v1:
            # "reach-v1",
            # "push-v1",
            # "door-open-v1",
            # "drawer-open-v1",
            # "drawer-close-v1",
            # "button-press-topdown-v1",
            # "window-open-v1",
            # "window-close-v1",
            # "door-close-v1",
            # "reach-wall-v1",
            # "push-wall-v1",
            # "button-press-v1",
            # "button-press-topdown-wall-v1",
            # "button-press-wall-v1",
            # "peg-unplug-side-v1",
            # "plate-slide-v1",
            # "plate-slide-side-v1",
            # "plate-slide-back-v1",
            # "plate-slide-back-side-v1",
            # "handle-press-v1",
            # "handle-pull-v1",
            # "handle-press-side-v1",
            # "handle-pull-side-v1",
            # "soccer-v1",
            # "faucet-open-v1",
            # "faucet-close-v1",
            # "coffee-push-v1",
            # "coffee-pull-v1",
            # "coffee-button-v1",
            # "sweep-into-v1",
            # "push-back-v1",
            # "dial-turn-v1",
            # "hand-insert-v1",
            # "door-lock-v1",
            # unsolveable
            # "assembly-v1",
            # "basketball-v1",
            # "bin-picking-v1",
            # "box-close-v1",
            # "disassemble-v1",
            # "door-unlock-v1",
            # "hammer-v1",
            # "lever-pull-v1",
            # "peg-insert-side-v1",
            # "pick-out-of-hole-v1",
            # "pick-place-v1",
            # "pick-place-wall-v1",
            # "shelf-place-v1",
            # "stick-pull-v1",
            # "stick-push-v1",
            # "sweep-v1",
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
