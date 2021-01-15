def preprocess_variant(variant, debug):
    env_class = variant["env_class"]
    if env_class == "microwave":
        max_path_length = 3
    elif env_class == "kettle":
        max_path_length = 5
    elif env_class == "slide_cabinet":
        max_path_length = 3
    elif env_class == "hinge_cabinet":
        max_path_length = 6
    elif env_class == "top_left_burner":
        max_path_length = 3
    elif env_class == "bottom_left_burner":
        max_path_length = 3
    elif env_class == "light_switch":
        max_path_length = 5
    elif env_class == "multitask_all":
        max_path_length = 6
    else:
        raise EnvironmentError("invalid env provided")

    variant["algorithm_kwargs"]["max_path_length"] = max_path_length
    # variant["trainer_kwargs"]["imagination_horizon"] = max_path_length + 1
    variant["trainer_kwargs"][
        "imagination_horizon"
    ] = max_path_length  # todo: see if this works well or not
    num_steps_per_epoch = 1000
    num_expl_steps_per_train_loop = variant["num_expl_envs"] * (max_path_length + 1)
    num_train_loops_per_epoch = num_steps_per_epoch // num_expl_steps_per_train_loop
    num_trains_per_train_loop = num_expl_steps_per_train_loop
    if num_steps_per_epoch % num_expl_steps_per_train_loop != 0:
        num_train_loops_per_epoch += 1

    total_batch_size = 2500
    effective_batch_size = total_batch_size // (max_path_length + 1)

    if not debug:
        variant["algorithm_kwargs"][
            "num_expl_steps_per_train_loop"
        ] = num_expl_steps_per_train_loop
        variant["algorithm_kwargs"]["batch_size"] = effective_batch_size
        variant["algorithm_kwargs"][
            "num_train_loops_per_epoch"
        ] = num_train_loops_per_epoch
        variant["algorithm_kwargs"][
            "num_trains_per_train_loop"
        ] = num_trains_per_train_loop
    if variant.get("path_length_specific_discount", False):
        variant["trainer_kwargs"]["discount"] = 1 - 1 / max_path_length
    if variant.get("use_mcts_policy", False):
        variant["expl_policy_kwargs"]["discount"] = variant["trainer_kwargs"][
            "discount"
        ]
        variant["eval_policy_kwargs"]["discount"] = variant["trainer_kwargs"][
            "discount"
        ]
        variant["expl_policy_kwargs"]["dirichlet_alpha"] = variant["dirichlet_alpha"]
        variant["eval_policy_kwargs"]["dirichlet_alpha"] = variant["dirichlet_alpha"]

        variant["expl_policy_kwargs"]["batch_size"] = variant["batch_size"]
        variant["eval_policy_kwargs"]["batch_size"] = variant["batch_size"]

        variant["expl_policy_kwargs"]["progressive_widening_constant"] = variant[
            "progressive_widening_constant"
        ]
        variant["eval_policy_kwargs"]["progressive_widening_constant"] = variant[
            "progressive_widening_constant"
        ]

        variant["expl_policy_kwargs"]["mcts_iterations"] = variant["mcts_iterations"]
        variant["eval_policy_kwargs"]["mcts_iterations"] = variant["mcts_iterations"]

        variant["expl_policy_kwargs"]["randomly_sample_discrete_actions"] = variant[
            "randomly_sample_discrete_actions"
        ]
        if variant["mcts_algorithm"]:
            variant["trainer_kwargs"]["randomly_sample_discrete_actions"] = variant[
                "randomly_sample_discrete_actions"
            ]
            variant["trainer_kwargs"]["mcts_iterations"] = variant["mcts_iterations"]
            variant["trainer_kwargs"]["dirichlet_alpha"] = variant["dirichlet_alpha"]
            variant["trainer_kwargs"]["batch_size"] = variant["batch_size"]
            variant["trainer_kwargs"]["progressive_widening_constant"] = variant[
                "progressive_widening_constant"
            ]
        if variant["reward_type"] == "intrinsic":
            variant["algorithm"] = variant["algorithm"] + "Intrinsic"
            variant["trainer_kwargs"]["exploration_reward_scale"] = 10000

            variant["trainer_kwargs"][
                "train_exploration_actor_with_intrinsic_and_extrinsic_reward"
            ] = False
            variant["trainer_kwargs"][
                "train_actor_with_intrinsic_and_extrinsic_reward"
            ] = False

            variant["expl_policy_kwargs"]["intrinsic_reward_scale"] = 1.0
            variant["expl_policy_kwargs"]["extrinsic_reward_scale"] = 0.0
            variant["eval_policy_kwargs"]["intrinsic_reward_scale"] = 0.0
            variant["eval_policy_kwargs"]["extrinsic_reward_scale"] = 1.0

            if variant["mcts_algorithm"]:
                variant["trainer_kwargs"][
                    "exploration_actor_intrinsic_reward_scale"
                ] = 1.0
                variant["trainer_kwargs"][
                    "exploration_actor_extrinsic_reward_scale"
                ] = 0.0
                variant["trainer_kwargs"]["actor_intrinsic_reward_scale"] = 0.0
        elif variant["reward_type"] == "intrinsic+extrinsic":
            variant["algorithm"] = variant["algorithm"] + "IntrinsicExtrinsic"
            variant["trainer_kwargs"]["exploration_reward_scale"] = 1.0

            variant["trainer_kwargs"][
                "train_exploration_actor_with_intrinsic_and_extrinsic_reward"
            ] = True
            variant["trainer_kwargs"][
                "train_actor_with_intrinsic_and_extrinsic_reward"
            ] = True

            variant["expl_policy_kwargs"]["intrinsic_reward_scale"] = 1.0
            variant["expl_policy_kwargs"]["extrinsic_reward_scale"] = 1.0
            variant["eval_policy_kwargs"]["intrinsic_reward_scale"] = 1.0
            variant["eval_policy_kwargs"]["extrinsic_reward_scale"] = 1.0
            if variant["mcts_algorithm"]:
                variant["trainer_kwargs"][
                    "exploration_actor_intrinsic_reward_scale"
                ] = 1.0
                variant["trainer_kwargs"][
                    "exploration_actor_extrinsic_reward_scale"
                ] = 1.0
                variant["trainer_kwargs"]["actor_intrinsic_reward_scale"] = 1.0
        else:
            variant["algorithm"] = variant["algorithm"] + "Extrinsic"
            variant["trainer_kwargs"]["exploration_reward_scale"] = 0.0

            variant["trainer_kwargs"][
                "train_exploration_actor_with_intrinsic_and_extrinsic_reward"
            ] = True
            variant["trainer_kwargs"][
                "train_actor_with_intrinsic_and_extrinsic_reward"
            ] = True

            variant["expl_policy_kwargs"]["intrinsic_reward_scale"] = 0.0
            variant["expl_policy_kwargs"]["extrinsic_reward_scale"] = 1.0
            variant["eval_policy_kwargs"]["intrinsic_reward_scale"] = 0.0
            variant["eval_policy_kwargs"]["extrinsic_reward_scale"] = 1.0
            if variant["mcts_algorithm"]:
                variant["trainer_kwargs"][
                    "exploration_actor_intrinsic_reward_scale"
                ] = 0.0
                variant["trainer_kwargs"][
                    "exploration_actor_extrinsic_reward_scale"
                ] = 1.0
                variant["trainer_kwargs"]["actor_intrinsic_reward_scale"] = 0.0

    return variant
