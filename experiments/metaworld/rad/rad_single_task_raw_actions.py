import argparse
import random

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
            discount=0.99,
            critic_lr=2e-4,
            actor_lr=2e-4,
            encoder_lr=2e-4,
            encoder_type="pixel",
            discrete_continuous_dist=False,
            data_augs="no_aug",
        ),
        num_train_steps=int(2.5e6),
        frame_stack=1,
        replay_buffer_capacity=int(2.5e6),
        action_repeat=1,
        num_eval_episodes=5,
        init_steps=2500,
        pre_transform_image_size=64,
        image_size=64,
        batch_size=512,
        eval_freq=1000,
        log_interval=1000,
        env_kwargs=dict(
            control_mode="end_effector",
            use_combined_action_space=False,
            action_scale=1 / 100,
            max_path_length=500,
            use_image_obs=True,
            reward_scale=1 / 100,
            use_dm_backend=True,
        ),
        seed=-1,
        use_raw_actions=True,
        env_suite="metaworld",
    )

    search_space = {
        "agent_kwargs.data_augs": [
            "no_aug",
        ],
        "agent_kwargs.discrete_continuous_dist": [False],
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
