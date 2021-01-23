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

    if (
        "proprioception" in variant["env_kwargs"]
        and variant["env_kwargs"]["proprioception"]
    ):
        variant["model_kwargs"]["state_output_size"] = 16

    if variant.get("use_image_goals", False):
        d = dict(
            slide_cabinet="/home/mdalal/research/hrl-exp/goals/slide_cabinet_goals.npy",
            hinge_cabinet="/home/mdalal/research/hrl-exp/goals/hinge_cabinet_goals.npy",
            microwave="/home/mdalal/research/hrl-exp/goals/microwave_goals.npy",
            top_left_burner="/home/mdalal/research/hrl-exp/goals/top_left_burner_goals.npy",
            kettle="/home/mdalal/research/hrl-exp/goals/kettle_goals.npy",
            light_switch="/home/mdalal/research/hrl-exp/goals/light_switch_goals.npy",
        )
        variant["trainer_kwargs"]["image_goals_path"] = d[variant["env_class"]]
    if variant.get("use_mcts_policy", False):
        discount = variant["trainer_kwargs"]["discount"]
        randomly_sample_discrete_actions = variant["randomly_sample_discrete_actions"]
        variant["mcts_kwargs"]["discount"] = discount

        variant["expl_policy_kwargs"][
            "randomly_sample_discrete_actions"
        ] = randomly_sample_discrete_actions

        variant["expl_policy_kwargs"]["mcts_kwargs"] = variant["mcts_kwargs"].copy()
        variant["eval_policy_kwargs"]["mcts_kwargs"] = variant["mcts_kwargs"].copy()

        variant["expl_policy_kwargs"]["mcts_kwargs"]["evaluation"] = False
        variant["eval_policy_kwargs"]["mcts_kwargs"]["evaluation"] = True
        if variant["mcts_algorithm"]:
            variant["trainer_kwargs"]["randomly_sample_discrete_actions"] = variant[
                "randomly_sample_discrete_actions"
            ]
            variant["trainer_kwargs"]["mcts_kwargs"] = variant["mcts_kwargs"]
            variant["trainer_kwargs"]["mcts_kwargs"]["evaluation"] = False

    if variant["reward_type"] == "intrinsic":
        variant["algorithm"] = variant["algorithm"] + "Intrinsic"
        variant["trainer_kwargs"]["exploration_reward_scale"] = 10000
        variant["trainer_kwargs"]["detach_rewards"] = True

        variant["trainer_kwargs"][
            "train_exploration_actor_with_intrinsic_and_extrinsic_reward"
        ] = False
        variant["trainer_kwargs"][
            "train_actor_with_intrinsic_and_extrinsic_reward"
        ] = False
        if variant.get("use_mcts_policy", False):
            variant["expl_policy_kwargs"]["mcts_kwargs"]["intrinsic_reward_scale"] = 1.0
            variant["expl_policy_kwargs"]["mcts_kwargs"]["extrinsic_reward_scale"] = 0.0

            variant["eval_policy_kwargs"]["mcts_kwargs"]["intrinsic_reward_scale"] = 0.0
            variant["eval_policy_kwargs"]["mcts_kwargs"]["extrinsic_reward_scale"] = 1.0

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
        variant["trainer_kwargs"]["detach_rewards"] = False

        variant["trainer_kwargs"][
            "train_exploration_actor_with_intrinsic_and_extrinsic_reward"
        ] = True
        variant["trainer_kwargs"][
            "train_actor_with_intrinsic_and_extrinsic_reward"
        ] = True

        if variant.get("use_mcts_policy", False):
            variant["expl_policy_kwargs"]["mcts_kwargs"]["intrinsic_reward_scale"] = 1.0
            variant["expl_policy_kwargs"]["mcts_kwargs"]["extrinsic_reward_scale"] = 1.0
            variant["eval_policy_kwargs"]["mcts_kwargs"]["intrinsic_reward_scale"] = 1.0
            variant["eval_policy_kwargs"]["mcts_kwargs"]["extrinsic_reward_scale"] = 1.0
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
        variant["trainer_kwargs"]["detach_rewards"] = False

        variant["trainer_kwargs"][
            "train_exploration_actor_with_intrinsic_and_extrinsic_reward"
        ] = True
        variant["trainer_kwargs"][
            "train_actor_with_intrinsic_and_extrinsic_reward"
        ] = True

        if variant.get("use_mcts_policy", False):
            variant["expl_policy_kwargs"]["mcts_kwargs"]["intrinsic_reward_scale"] = 0.0
            variant["expl_policy_kwargs"]["mcts_kwargs"]["extrinsic_reward_scale"] = 1.0
            variant["eval_policy_kwargs"]["mcts_kwargs"]["intrinsic_reward_scale"] = 0.0
            variant["eval_policy_kwargs"]["mcts_kwargs"]["extrinsic_reward_scale"] = 1.0
            if variant["mcts_algorithm"]:
                variant["trainer_kwargs"][
                    "exploration_actor_intrinsic_reward_scale"
                ] = 0.0
                variant["trainer_kwargs"][
                    "exploration_actor_extrinsic_reward_scale"
                ] = 1.0
                variant["trainer_kwargs"]["actor_intrinsic_reward_scale"] = 0.0
    return variant
