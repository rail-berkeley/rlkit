def make_base_robosuite_env(env_name, kwargs, use_dm_backend=True):
    import gym
    import numpy as np

    gym.logger.setLevel(40)
    import robosuite as suite

    from rlkit.envs.wrappers.primitives_wrappers import (
        NormalizeBoxEnvFixed,
        RobosuitePrimitives,
        RobosuiteWrapper,
    )

    if suite.environments.robot_env.RobotEnv.__bases__[0] != RobosuitePrimitives:
        suite.environments.robot_env.RobotEnv.__bases__ = (RobosuitePrimitives,)
    RobosuitePrimitives._use_dm_backend = use_dm_backend
    if kwargs["has_offscreen_renderer"]:
        keys = ["image-state"]
    else:
        keys = None
    reset_action_space_kwargs = kwargs.get("reset_action_space_kwargs", {})
    env_kwargs_new = kwargs.copy()
    if "reset_action_space_kwargs" in kwargs:
        del env_kwargs_new["reset_action_space_kwargs"]
    np.random.seed(42)
    env = suite.make(env_name, **env_kwargs_new)
    env = RobosuiteWrapper(
        env, keys=keys, reset_action_space_kwargs=reset_action_space_kwargs
    )
    if reset_action_space_kwargs["control_mode"] == "robosuite":
        env = NormalizeBoxEnvFixed(env)
    return env


def make_base_metaworld_env(env_name, env_kwargs=None, use_dm_backend=True):
    action_space_kwargs = env_kwargs["action_space_kwargs"]
    env_kwargs_new = env_kwargs.copy()
    if "action_space_kwargs" in env_kwargs_new:
        del env_kwargs_new["action_space_kwargs"]
    import gym

    gym.logger.setLevel(40)
    if env_kwargs is None:
        env_kwargs = {}
    from metaworld.envs.mujoco.env_dict import (
        ALL_V1_ENVIRONMENTS,
        ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
        MT50_V2,
    )
    from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv

    from rlkit.envs.wrappers.dm_backend_wrappers import (
        SawyerMocapBaseDMBackendMetaworld,
    )
    from rlkit.envs.wrappers.primitives_wrappers import (
        MetaworldWrapper,
        SawyerXYZEnvMetaworldPrimitives,
    )

    env_clses = list(MT50_V2.values())
    # hack from https://stackoverflow.com/questions/38397610/how-to-change-the-base-class-in-python
    for env_cls in env_clses:
        parent = env_cls
        while SawyerXYZEnv != parent.__bases__[0]:
            parent = parent.__bases__[0]
        if (
            parent != SawyerXYZEnvMetaworldPrimitives
        ):  # ensure if it is called multiple times you don't reset the base class
            parent.__bases__ = (SawyerXYZEnvMetaworldPrimitives,)
        if use_dm_backend:
            SawyerXYZEnv.__bases__ = (SawyerMocapBaseDMBackendMetaworld,)
    SawyerXYZEnvMetaworldPrimitives.control_mode = action_space_kwargs["control_mode"]
    SawyerMocapBaseDMBackendMetaworld.control_mode = action_space_kwargs["control_mode"]
    if env_name in ALL_V1_ENVIRONMENTS:
        env_cls = ALL_V1_ENVIRONMENTS[env_name]
    else:
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + "-goal-observable"]

    if env_name in ALL_V1_ENVIRONMENTS:
        env = env_cls()
        if env_name == "reach-v1" or env_name == "reach-wall-v1":
            env._set_task_inner(task_type="reach")
        elif env_name == "push-v1" or env_name == "push-wall-v1":
            env._set_task_inner(task_type="push")
        elif env_name == "pick-place-v1" or env_name == "pick-place-wall-v1":
            env._set_task_inner(task_type="pick_place")
        env._partially_observable = False
        env.random_init = False
        env._set_task_called = True
    else:
        env = env_cls(seed=42)
    env.reset_action_space(**action_space_kwargs)
    env.reset()
    env = MetaworldWrapper(env, **env_kwargs_new)
    return env


def make_base_kitchen_env(env_class, env_kwargs):
    from d4rl.kitchen.env_dict import ALL_KITCHEN_ENVIRONMENTS

    env = ALL_KITCHEN_ENVIRONMENTS[env_class](**env_kwargs)
    return env


def make_env(env_suite, env_name, env_kwargs):
    from rlkit.envs.wrappers.primitives_wrappers import (
        ActionRepeat,
        ImageUnFlattenWrapper,
        NormalizeActions,
        TimeLimit,
    )

    usage_kwargs = env_kwargs["usage_kwargs"]
    max_path_length = usage_kwargs["max_path_length"]
    use_dm_backend = usage_kwargs.get("use_dm_backend", True)
    use_raw_action_wrappers = usage_kwargs.get("use_raw_action_wrappers", False)
    unflatten_images = usage_kwargs.get("unflatten_images", False)

    env_kwargs_new = env_kwargs.copy()
    if "usage_kwargs" in env_kwargs_new:
        del env_kwargs_new["usage_kwargs"]
    if "image_kwargs" in env_kwargs_new:
        del env_kwargs_new["image_kwargs"]

    if env_suite == "kitchen":
        env = make_base_kitchen_env(env_name, env_kwargs_new)
    elif env_suite == "metaworld":
        env = make_base_metaworld_env(env_name, env_kwargs_new, use_dm_backend)
    elif env_suite == "robosuite":
        env = make_base_robosuite_env(env_name, env_kwargs_new, use_dm_backend)
    if unflatten_images:
        env = ImageUnFlattenWrapper(env)

    if use_raw_action_wrappers:
        env = ActionRepeat(env, 2)
        env = NormalizeActions(env)
        env = TimeLimit(env, max_path_length // 2)
    else:
        env = TimeLimit(env, max_path_length)
    env.reset
    return env
