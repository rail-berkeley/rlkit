def preprocess_variant_p2exp(variant):
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


def preprocess_variant_llraps(variant):
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
    variant["replay_buffer_kwargs"]["max_path_length"] = variant["max_path_length"]

    variant["env_kwargs"]["usage_kwargs"]["max_path_length"] = variant[
        "max_path_length"
    ]
    variant["env_kwargs"]["action_space_kwargs"][
        "num_low_level_actions_per_primitive"
    ] = variant["num_low_level_actions_per_primitive"]

    variant["algorithm_kwargs"]["max_path_length"] = variant["max_path_length"]
    return variant
