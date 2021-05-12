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
        num_train_steps=int(1e6),
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
            control_mode="primitives",
            use_combined_action_space=True,
            action_scale=1,
            max_path_length=5,
            reward_type="sparse",
            camera_settings={
                "distance": 0.38227044687537043,
                "lookat": [0.21052547, 0.32329237, 0.587819],
                "azimuth": 141.328125,
                "elevation": -53.203125160653144,
            },
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=True,
                max_path_length=5,
                unflatten_images=True,
            ),
            image_kwargs=dict(imwidth=64, imheight=64),
        ),
        seed=-1,
        use_raw_actions=False,
        env_suite="metaworld",
    )

    search_space = {
        "agent_kwargs.data_augs": [
            "no_aug",
        ],
        "agent_kwargs.discrete_continuous_dist": [False],
        "env_name": [
            # solveable
            # "basketball-v2",
            "assembly-v2",
            "disassemble-v2"
            "hand-insert-v2",
            # verified and medium
            "soccer-v2",
            "sweep-into-v2",
            # easy
            # "plate-slide-v2",
            # "faucet-open-v2",
            # verified and unsolved:
            # "bin-picking-v2",
            # unverified and unsolved:
            # "stick-pull-v2",
            "drawer-close-v2",
            # "peg-insert-side-v2",
            # "pick-out-of-hole-v2",
            # "hammer-v2",
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for _ in range(args.num_seeds):
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
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
