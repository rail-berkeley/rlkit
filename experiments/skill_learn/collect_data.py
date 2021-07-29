import os

import numpy as np
from arguments import get_args
from tqdm import tqdm

from rlkit.envs.primitives_make_env import make_env


def collect_data(env, num_primitives, num_actions):
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

    for j in tqdm(range(num_actions)):
        a = env.action_space.sample()
        primitive = np.argmax(a[:num_primitives])
        arguments = a[num_primitives:]
        o, r, d, i = env.step(a)
        data["actions"][primitive].append(i["actions"])
        arguments = np.array(i["arguments"])
        robot_states = np.array(i["robot-states"])
        if len(arguments.shape) == 1:
            arguments = arguments.reshape(-1, 1)
        data["inputs"][primitive].append(
            np.concatenate((robot_states, arguments), axis=1)
        )
        if d:
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
    env = make_env(env_suite, env_name, env_kwargs)
    num_primitives = env.num_primitives
    args = get_args()
    data = collect_data(env, num_primitives, args.num_actions)
    np.save("data/primitive_data/" + args.datafile + ".npy", data)
