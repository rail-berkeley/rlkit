def make_base_robosuite_env(env_name, kwargs):
    import gym

    from rlkit.envs.wrappers.normalized_box_env import NormalizedBoxEnv

    gym.logger.setLevel(40)
    from robosuite.environments.base import REGISTERED_ENVS, MujocoEnv
    from robosuite.wrappers.gym_wrapper import GymWrapper

    from rlkit.envs.primitives_wrappers import DMControlBackendMetaworldRobosuiteEnv

    env_cls = REGISTERED_ENVS[env_name]
    parent = env_cls
    while MujocoEnv != parent.__bases__[0]:
        parent = parent.__bases__[0]

    if parent != DMControlBackendMetaworldRobosuiteEnv:
        parent.__bases__ = (DMControlBackendMetaworldRobosuiteEnv,)
    return NormalizedBoxEnv(GymWrapper(REGISTERED_ENVS[env_name](**kwargs)))


def make_base_metaworld_env(env_name, env_kwargs=None, use_dm_backend=True):
    reward_type = env_kwargs["reward_type"]
    env_kwargs_new = env_kwargs.copy()
    if "reward_type" in env_kwargs_new:
        del env_kwargs_new["reward_type"]
    import gym

    gym.logger.setLevel(40)
    if env_kwargs is None:
        env_kwargs = {}
    from metaworld.envs.mujoco.env_dict import (
        ALL_V1_ENVIRONMENTS,
        ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
    )
    from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv

    from rlkit.envs.dm_backend_wrappers import SawyerMocapBaseDMBackendMetaworld
    from rlkit.envs.primitives_wrappers import (
        MetaworldWrapper,
        SawyerXYZEnvMetaworldPrimitives,
    )

    if env_name in ALL_V1_ENVIRONMENTS:
        env_cls = ALL_V1_ENVIRONMENTS[env_name]
    else:
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name + "-goal-observable"]

    # hack from https://stackoverflow.com/questions/38397610/how-to-change-the-base-class-in-python
    # assume linear hierarchy for now
    parent = env_cls
    while SawyerXYZEnv != parent.__bases__[0]:
        parent = parent.__bases__[0]
    if (
        parent != SawyerXYZEnvMetaworldPrimitives
    ):  # ensure if it is called multiple times you don't reset the base class
        parent.__bases__ = (SawyerXYZEnvMetaworldPrimitives,)
    if use_dm_backend:
        SawyerXYZEnv.__bases__ = (SawyerMocapBaseDMBackendMetaworld,)
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
    env.reset_action_space(**env_kwargs_new)
    env.reset()
    env = MetaworldWrapper(env, reward_type=reward_type)
    return env


def make_base_kitchen_env(env_class, env_kwargs):
    from d4rl.kitchen.env_dict import ALL_KITCHEN_ENVIRONMENTS

    env = ALL_KITCHEN_ENVIRONMENTS[env_class](**env_kwargs)
    return env


def make_env(env_suite, env_name, env_kwargs):
    from rlkit.envs.primitives_wrappers import (
        ActionRepeat,
        ImageEnvMetaworld,
        ImageUnFlattenWrapper,
        NormalizeActions,
        TimeLimit,
    )

    usage_kwargs = env_kwargs["usage_kwargs"]
    use_dm_backend = usage_kwargs["use_dm_backend"]
    use_raw_action_wrappers = usage_kwargs["use_raw_action_wrappers"]
    use_image_obs = usage_kwargs["use_image_obs"]
    max_path_length = usage_kwargs["max_path_length"]
    unflatten_images = usage_kwargs["unflatten_images"]
    image_kwargs = env_kwargs["image_kwargs"]

    env_kwargs_new = env_kwargs.copy()
    if "usage_kwargs" in env_kwargs_new:
        del env_kwargs_new["usage_kwargs"]
    if "image_kwargs" in env_kwargs_new:
        del env_kwargs_new["image_kwargs"]

    if env_suite == "kitchen":
        env = make_base_kitchen_env(env_name, env_kwargs_new)
    elif env_suite == "metaworld":
        env = make_base_metaworld_env(env_name, env_kwargs_new, use_dm_backend)
        if use_image_obs:
            env = ImageEnvMetaworld(
                env,
                **image_kwargs,
            )
    if unflatten_images:
        env = ImageUnFlattenWrapper(env)

    if use_raw_action_wrappers:
        env = ActionRepeat(env, 2)
        env = NormalizeActions(env)
        env = TimeLimit(env, max_path_length // 2)
    else:
        env = TimeLimit(env, max_path_length)
    return env
