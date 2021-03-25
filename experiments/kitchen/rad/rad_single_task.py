import argparse
import os
import random

import rlkit.util.hyperparameter as hyp
from rlkit.launchers.launcher_util import run_experiment


def experiment(variant):
    data_augs = variant["algorithm_kwargs"]["data_augs"]
    if data_augs == "crop":
        pre_transform_image_size = 100
        image_size = 84
    elif data_augs == "translate":
        pre_transform_image_size = 100
        image_size = 108
    else:
        pre_transform_image_size = 84
        image_size = 84
    from rlkit.core import logger

    os.system(
        "~/miniconda3/envs/hrl-exp-env/bin/python ~/research/rad/kitchen_train.py \
            --encoder_type pixel --work_dir {work_dir} \
            --env_class {env_class} \
            --action_repeat 1 --num_eval_episodes 5 \
            --pre_transform_image_size {pre_transform_image_size} --image_size {image_size} \
            --data_augs {data_augs} --discount {discount} --init_steps 2500 --discrete_continuous_dist {discrete_continuous_dist}\
            --agent rad_sac --frame_stack {framestack} --save_tb --replay_buffer_capacity {replay_buffer_capacity}\
            --seed {seed} --critic_lr {lr} --actor_lr {lr} --encoder_lr {lr} --eval_freq 1000 --batch_size 512 --num_train_steps {num_train_steps}".format(
            **variant["algorithm_kwargs"],
            work_dir=logger.get_snapshot_dir(),
            image_size=image_size,
            pre_transform_image_size=pre_transform_image_size
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_prefix", type=str, default="test")
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--mode", type=str, default="local")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    if args.debug:
        algorithm_kwargs = dict()
        exp_prefix = "test" + args.exp_prefix
    else:
        algorithm_kwargs = dict(
            framestack=1,
            discount=0.99,
            lr=2e-4,
            data_augs="translate",
            num_train_steps=200000,
            replay_buffer_capacity=int(1e6),
        )
        exp_prefix = args.exp_prefix
    variant = dict(algorithm_kwargs=algorithm_kwargs)

    search_space = {
        "algorithm_kwargs.data_augs": [
            "no_aug",
        ],
        "algorithm_kwargs.discount": [0.8],
        "algorithm_kwargs.framestack": [
            1,
        ],
        "algorithm_kwargs.num_train_steps": [500000],
        "algorithm_kwargs.discrete_continuous_dist": [1],
        "algorithm_kwargs.env_class": [
            # "microwave",
            # "kettle",
            # "slide_cabinet",
            "top_left_burner",
            "hinge_cabinet",
            "light_switch",
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
            variant["algorithm_kwargs"]["seed"] = seed
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
