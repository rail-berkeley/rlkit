import argparse
import random
import subprocess

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment


def experiment(variant):
    from rad.kitchen_train import experiment

    experiment(variant)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    exp_prefix = args.exp_prefix
    variant = dict(
        agent_kwargs=dict(
            discount=0.8,
            critic_lr=2e-4,
            actor_lr=2e-4,
            encoder_lr=2e-4,
            encoder_type="pixel",
            discrete_continuous_dist=False,
            data_augs="no_aug",
        ),
        num_train_steps=250000,
        frame_stack=4,
        replay_buffer_capacity=int(1e6),
        action_repeat=1,
        num_eval_episodes=5,
        init_steps=2500,
        pre_transform_image_size=64,
        image_size=64,
        env_name="slide_cabinet",
        batch_size=512,
        eval_freq=1000,
        log_interval=1000,
        env_kwargs=dict(
            dense=False,
            image_obs=True,
            action_scale=1.4,
            use_workspace_limits=True,
            control_mode="primitives",
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=True,
                max_path_length=5,
                unflatten_images=True,
            ),
            image_kwargs=dict(),
        ),
        seed=-1,
        use_raw_actions=False,
        env_suite="kitchen",
    )

    search_space = {
        "agent_kwargs.data_augs": [
            "no_aug",
        ],
        "agent_kwargs.discrete_continuous_dist": [False],
        "env_name": [
            "microwave",
            # "kettle",
            # "slide_cabinet",
            # "top_left_burner",
            # "hinge_cabinet",
            # "light_switch",
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
