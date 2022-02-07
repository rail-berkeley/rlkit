def preprocess_variant_p2exp(variant):
    variant = variant.copy()
    if variant["reward_type"] == "intrinsic":
        variant["algorithm"] = variant["algorithm"] + "Intrinsic"
        variant["trainer_kwargs"]["exploration_intrinsic_reward_scale"] = 1.0
        variant["trainer_kwargs"]["exploration_extrinsic_reward_scale"] = 0.0
        variant["trainer_kwargs"]["evaluation_intrinsic_reward_scale"] = 0.0
        variant["trainer_kwargs"]["evaluation_extrinsic_reward_scale"] = 1.0
        variant["trainer_kwargs"]["detach_rewards"] = True
    elif variant["reward_type"] == "intrinsic+extrinsic":
        variant["algorithm"] = variant["algorithm"] + "IntrinsicExtrinsic"
        variant["trainer_kwargs"]["exploration_intrinsic_reward_scale"] = 1.0
        variant["trainer_kwargs"]["exploration_extrinsic_reward_scale"] = 1.0

        variant["trainer_kwargs"]["evaluation_intrinsic_reward_scale"] = 1.0
        variant["trainer_kwargs"]["evaluation_extrinsic_reward_scale"] = 1.0
        variant["trainer_kwargs"]["detach_rewards"] = False
    elif variant["reward_type"] == "extrinsic":
        variant["algorithm"] = variant["algorithm"] + "Extrinsic"
        variant["trainer_kwargs"]["exploration_intrinsic_reward_scale"] = 0.0
        variant["trainer_kwargs"]["exploration_extrinsic_reward_scale"] = 1.0
        variant["trainer_kwargs"]["evaluation_intrinsic_reward_scale"] = 0.0
        variant["trainer_kwargs"]["evaluation_extrinsic_reward_scale"] = 1.0
        variant["trainer_kwargs"]["detach_rewards"] = False
    return variant


def preprocess_variant_raps(variant):
    variant = variant.copy()
    variant["algorithm_kwargs"]["max_path_length"] = variant["max_path_length"]
    variant["replay_buffer_kwargs"]["max_path_length"] = variant["max_path_length"]

    variant["env_kwargs"]["usage_kwargs"]["max_path_length"] = variant[
        "max_path_length"
    ]
    return variant


def preprocess_variant_llraps(variant):
    variant = variant.copy()
    variant["trainer_kwargs"]["batch_length"] = int(
        variant["num_low_level_actions_per_primitive"] * variant["max_path_length"] + 1
    )
    variant["trainer_kwargs"]["effective_batch_size_iterations"] = (
        variant["effective_batch_size"] // variant["algorithm_kwargs"]["batch_size"]
    )
    variant["trainer_kwargs"]["num_low_level_actions_per_primitive"] = variant[
        "num_low_level_actions_per_primitive"
    ]

    variant["replay_buffer_kwargs"]["max_replay_buffer_size"] = int(
        3e6
        / (
            variant["num_low_level_actions_per_primitive"] * variant["max_path_length"]
            + 1
        )
    )
    variant["replay_buffer_kwargs"]["num_low_level_actions_per_primitive"] = variant[
        "num_low_level_actions_per_primitive"
    ]
    variant["replay_buffer_kwargs"]["low_level_action_dim"] = variant[
        "low_level_action_dim"
    ]

    variant["env_kwargs"]["action_space_kwargs"][
        "num_low_level_actions_per_primitive"
    ] = variant["num_low_level_actions_per_primitive"]

    variant = preprocess_variant_raps(variant)

    return variant


def setup_sweep_and_launch_exp(preprocess_variant_fn, variant, experiment_fn, args):
    import random
    import subprocess

    import rlkit.util.hyperparameter as hyp
    from rlkit.launchers.launcher_util import run_experiment

    search_space = {
        key: value for key, value in zip(args.search_keys, args.search_values)
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant = preprocess_variant_fn(variant)
        for _ in range(args.num_seeds):
            seed = random.randint(0, 100000)
            variant["seed"] = seed
            variant["exp_id"] = exp_id
            python_cmd = subprocess.check_output("which python", shell=True).decode(
                "utf-8"
            )[:-1]
            run_experiment(
                experiment_fn,
                exp_prefix=args.exp_prefix,
                mode=args.mode,
                variant=variant,
                use_gpu=args.use_gpu,
                snapshot_mode="none",
                python_cmd=python_cmd,
                seed=seed,
                exp_id=exp_id,
            )
