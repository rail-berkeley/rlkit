def preprocess_variant(variant, debug):
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
        randomly_sample_discrete_actions = variant["randomly_sample_discrete_actions"]

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

    if variant["algorithm"] == "Plan2Explore" and variant["reward_type"] == "intrinsic":
        variant["algorithm"] = variant["algorithm"] + "Intrinsic"
        variant["trainer_kwargs"]["exploration_intrinsic_reward_scale"] = 1.0
        variant["trainer_kwargs"]["exploration_extrinsic_reward_scale"] = 0.0

        variant["trainer_kwargs"]["evaluation_intrinsic_reward_scale"] = 0.0
        variant["trainer_kwargs"]["evaluation_extrinsic_reward_scale"] = 1.0

        variant["trainer_kwargs"]["detach_rewards"] = True

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
    elif (
        variant["algorithm"] == "Plan2Explore"
        and variant["reward_type"] == "intrinsic+extrinsic"
    ):
        variant["algorithm"] = variant["algorithm"] + "IntrinsicExtrinsic"
        variant["trainer_kwargs"]["exploration_intrinsic_reward_scale"] = 1.0
        variant["trainer_kwargs"]["exploration_extrinsic_reward_scale"] = 1.0

        variant["trainer_kwargs"]["evaluation_intrinsic_reward_scale"] = 1.0
        variant["trainer_kwargs"]["evaluation_extrinsic_reward_scale"] = 1.0
        variant["trainer_kwargs"]["detach_rewards"] = False

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
    elif (
        variant["algorithm"] == "Plan2Explore" and variant["reward_type"] == "extrinsic"
    ):
        variant["algorithm"] = variant["algorithm"] + "Extrinsic"
        variant["trainer_kwargs"]["exploration_intrinsic_reward_scale"] = 0.0
        variant["trainer_kwargs"]["exploration_extrinsic_reward_scale"] = 1.0

        variant["trainer_kwargs"]["evaluation_intrinsic_reward_scale"] = 0.0
        variant["trainer_kwargs"]["evaluation_extrinsic_reward_scale"] = 1.0
        variant["trainer_kwargs"]["detach_rewards"] = False

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
