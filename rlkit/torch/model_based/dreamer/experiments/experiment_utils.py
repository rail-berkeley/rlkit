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
    variant["trainer_kwargs"]["imagination_horizon"] = max_path_length + 1
    # variant["trainer_kwargs"][
    #     "imagination_horizon"
    # ] = max_path_length  # todo: see if this works well or not
    num_steps_per_epoch = 1000
    num_expl_steps_per_train_loop = 50 * (max_path_length + 1)
    num_train_loops_per_epoch = num_steps_per_epoch // num_expl_steps_per_train_loop
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
    if variant.get("path_length_specific_discount", False):
        variant["trainer_kwargs"]["discount"] = 1 - 1 / max_path_length
    return variant
