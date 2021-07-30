import os

import numpy as np
from arguments import get_args
from tqdm import tqdm

from rlkit.envs.mujoco_vec_wrappers import StableBaselinesVecEnv
from rlkit.envs.primitives_make_env import make_env


def collect_data(env, num_primitives, num_actions, num_envs):
    """
    Collect data from the environment.

    Args:
        env (Environment): the environment to collect data from
        num_primitives (int): the number of primitive actions to collect data for
        num_actions (int): the number of actions to collect
    Returns:
        dict: the data collected from the environment
    """
    data = {}
    data["actions"] = [[] for i in range(num_primitives)]
    data["inputs"] = [[] for i in range(num_primitives)]
    env.reset()

    for j in tqdm(range(num_actions // num_envs)):
        actions = [env.action_space.sample() for i in range(num_envs)]
        primitives = [np.argmax(a[:num_primitives]) for a in actions]
        o, r, d, i = env.step(actions)
        for j in range(num_envs):
            primitive = primitives[j]
            actions = i["actions"][j]
            robot_states = i["robot-states"][j]
            arguments = i["arguments"][j]
            data["actions"][primitive].append(actions)
            arguments = np.array(arguments)
            robot_states = np.array(robot_states)
            if len(arguments.shape) == 1:
                arguments = arguments.reshape(-1, 1)
            data["inputs"][primitive].append(
                np.concatenate((robot_states, arguments), axis=1)
            )
        if all(d):
            env.reset()

    return data


if __name__ == "__main__":
    env_kwargs = dict(
        control_mode="primitives",
        action_scale=1,
        max_path_length=5,
        reward_type="sparse",
        camera_settings={
            "distance": 0.38227044687537043,
            "lookat": [0.21052547, 0.32329237, 0.587819],
            "azimuth": 141.328125,
            "elevation": -53.203125160653144,
        },
        usage_kwargs=dict(
            use_dm_backend=True,
            use_raw_action_wrappers=False,
            use_image_obs=False,
            max_path_length=5,
            unflatten_images=False,
        ),
        image_kwargs=dict(imwidth=64, imheight=64),
        collect_primitives_info=True,
    )
    env_suite = "metaworld"
    env_name = "reach-v2"
    num_expl_envs = 25
    env_fns = [
        lambda: make_env(env_suite, env_name, env_kwargs) for _ in range(num_expl_envs)
    ]
    env = StableBaselinesVecEnv(env_fns=env_fns, start_method="fork")
    num_primitives = 10
    args = get_args()
    data = collect_data(env, num_primitives, args.num_actions, num_expl_envs)
    np.save("data/primitive_data/" + args.datafile + ".npy", data)
