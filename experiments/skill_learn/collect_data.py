import os

import numpy as np
from arguments import get_args
from tqdm import tqdm

from rlkit.envs.mujoco_vec_wrappers import StableBaselinesVecEnv
from rlkit.envs.primitives_make_env import make_env


def collect_primitive_cloning_data(env, num_primitives, num_trajs, num_envs):
    """
    Collect data from the environment.

    Args:
        env (Environment): the environment to collect data from
        num_primitives (int): the number of primitive actions to collect data for
        num_trajs (int): the number of actions to collect
    Returns:
        dict: the data collected from the environment
    """
    data = {}
    data["actions"] = [[] for i in range(num_primitives)]
    data["inputs"] = [[] for i in range(num_primitives)]
    env.reset()

    for k in tqdm(range(num_trajs // num_envs)):
        actions = [env.action_space.sample() for i in range(num_envs)]
        primitives = [np.argmax(a[:num_primitives]) for a in actions]
        o, r, d, i = env.step(actions)
        for j in range(num_envs):
            primitive = primitives[j]
            actions = np.array(i["actions"][j])
            robot_states = np.array(i["robot-states"][j])
            arguments = np.array(i["arguments"][j])
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


def collect_world_model_data_low_level_primitives(
    env, num_trajs, num_envs, max_path_length
):
    """
    Collect world model data from the environment.

    Args:
        env (Environment): the environment to collect data from
        num_trajs (int): the number of actions to collect
        max_path_length (int): the length of each path
    Returns:
        dict: the data collected from the environment
            actions: list of H lists of H_t actions
            observations: list of H lists of H_t+1 observations

    """
    data = {}
    data["actions"] = [[] for i in range(num_trajs // max_path_length)]
    data["observations"] = [[] for i in range(num_trajs // max_path_length)]
    env.reset()
    reset_ctr = 0
    ctr = 0
    for k in tqdm(range(num_trajs // num_envs)):
        actions = [env.action_space.sample() for i in range(num_envs)]
        o, r, d, i = env.step(actions)
        low_level_actions = i["actions"]
        obs = i["observations"]
        for e in range(num_envs):
            obs[e].append(o[e : e + 1].reshape(3, 64, 64).transpose(1, 2, 0))
        for env_num, l in enumerate(data["actions"][reset_ctr : reset_ctr + num_envs]):
            l.append(low_level_actions[env_num])
        for env_num, l in enumerate(
            data["observations"][reset_ctr : reset_ctr + num_envs]
        ):
            l.append(obs[env_num])
        ctr += 1
        if ctr % max_path_length == 0:
            env.reset()
            reset_ctr += num_envs
    return data


def collect_world_model_data(env, num_trajs, num_envs, max_path_length):
    """
    Collect world model data from the environment.

    Args:
        env (Environment): the environment to collect data from
        num_trajs (int): the number of trajectories to collect
        max_path_length (int): the length of each path
    Returns:
        dict: the data collected from the environment
            actions: (num_trajs, max_path_length+1, )
            observations: list of H lists of H_t+1 observations

    """
    data = {}
    data["actions"] = np.zeros(
        (num_trajs, max_path_length + 1, env.action_space.low.shape[0]),
        dtype=np.float32,
    )
    data["observations"] = np.zeros(
        (num_trajs, max_path_length + 1, env.observation_space.low.shape[0]),
        dtype=np.uint8,
    )
    for k in tqdm(range(num_trajs // num_envs)):
        o = env.reset()
        data["actions"][k * num_envs : k * num_envs + num_envs, 0] = np.zeros(
            (num_envs, env.action_space.low.shape[0])
        )
        data["observations"][k * num_envs : k * num_envs + num_envs, 0] = o
        for p in range(1, max_path_length + 1):
            actions = [env.action_space.sample() for i in range(num_envs)]
            o, r, d, i = env.step(actions)
            data["actions"][k * num_envs : k * num_envs + num_envs, p] = np.array(
                actions
            )
            data["observations"][k * num_envs : k * num_envs + num_envs, p] = o
    return data


if __name__ == "__main__":
    args = get_args()
    env_kwargs = dict(
        control_mode=args.control_mode,
        action_scale=1,
        max_path_length=args.max_path_length,
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
            use_image_obs=True,
            max_path_length=args.max_path_length,
            unflatten_images=False,
        ),
        image_kwargs=dict(imwidth=64, imheight=64),
        collect_primitives_info=False,
        include_phase_variable=False,
        render_intermediate_obs_to_info=False,
    )
    env_suite = "metaworld"
    env_name = "reach-v2"
    env_fns = [
        lambda: make_env(env_suite, env_name, env_kwargs) for _ in range(args.num_envs)
    ]
    env = StableBaselinesVecEnv(env_fns=env_fns, start_method="fork")
    data = collect_world_model_data(
        env, args.num_trajs * args.num_envs, args.num_envs, args.max_path_length
    )
    os.makedirs("data/world_model_data", exist_ok=True)
    import h5py

    f = h5py.File("data/world_model_data/" + args.datafile + ".hdf5", "w")
    f.create_dataset("observations", data=data["observations"])
    f.create_dataset("actions", data=data["actions"])
