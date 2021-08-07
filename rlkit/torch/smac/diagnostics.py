from rlkit.envs.pearl_envs import (
    AntDirEnv,
    HalfCheetahVelEnv,
)


def get_env_info_sizes(env):
    info_sizes = {}
    if isinstance(env.wrapped_env, AntDirEnv):
        info_sizes = dict(
            reward_forward=1,
            reward_ctrl=1,
            reward_contact=1,
            reward_survive=1,
            torso_velocity=3,
            torso_xy=2,
        )
    if isinstance(env.wrapped_env, HalfCheetahVelEnv):
        info_sizes = dict(
            reward_forward=1,
            reward_ctrl=1,
            goal_vel=1,
            forward_vel=1,
            xposbefore=1,
        )

    return info_sizes
