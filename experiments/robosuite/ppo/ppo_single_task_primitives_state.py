import os
import random
import subprocess

import numpy as np
import torch
from robosuite import load_controller_config
from robosuite.utils.placement_samplers import UniformRandomSampler

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment


def experiment(variant):
    from a2c_ppo_acktr.main import experiment

    experiment(variant)


if __name__ == "__main__":
    config = load_controller_config(default_controller="OSC_POSE")
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
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            camera_heights=64,
            camera_widths=64,
            controller_configs=config,
            horizon=5,
            control_freq=20,
            reward_shaping=False,
            reset_action_space_kwargs=dict(
                control_mode="primitives",
                use_combined_action_space=True,
                action_scale=1,
                max_path_length=5,
            ),
            # placement_initializer = UniformRandomSampler(
            #     name="ObjectSampler",
            #     x_range=[0, 0],
            #     y_range=[0, 0.0],
            #     rotation=0,
            #     ensure_object_boundary_in_range=False,
            #     ensure_valid_placement=True,
            #     reference_pos=np.array((0, 0, 0.8)),
            #     z_offset=0.01,
            # ),
            usage_kwargs=dict(
                use_dm_backend=True,
                max_path_length=5,
            ),
            image_kwargs=dict(),
        ),
        actor_kwargs=dict(recurrent=False, hidden_size=64, hidden_activation="tanh"),
        num_processes=10,
        num_env_steps=int(1e7),
        num_steps=2048 // 10,
        log_interval=1,
        eval_interval=1,
        use_raw_actions=False,
        env_suite="robosuite",
        discrete_continuous_dist=False,
        use_linear_lr_decay=False,
    )

    search_space = {
        "env_name": [
            "Lift",
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
