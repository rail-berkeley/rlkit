import os

import numpy as np
from arguments import get_args
from tqdm import tqdm

from rlkit.envs.mujoco_vec_wrappers import StableBaselinesVecEnv
from rlkit.envs.primitives_make_env import make_env


def save_wm_data(data, args):
    """
    Save world model data to a hdf5 file.
    :param: dictionary with two keys: observations and actions (numpy arrays)
    :param: args: arguments from command line
    """
    os.makedirs("data/world_model_data", exist_ok=True)
    import h5py

    f = h5py.File("data/world_model_data/" + args.datafile + ".hdf5", "w")
    f.create_dataset(
        "observations",
        data=data["observations"],
        compression="gzip",
        compression_opts=9,
    )
    f.create_dataset(
        "actions", data=data["actions"], compression="gzip", compression_opts=9
    )
    if "high_level_actions" in data:
        f.create_dataset(
            "high_level_actions",
            data=data["high_level_actions"],
            compression="gzip",
            compression_opts=9,
        )


def collect_primitive_cloning_data(
    env,
    num_primitives,
    num_trajs,
    num_envs,
    datafile,
):
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

    np.save("data/primitive_data/" + datafile + ".npy", data)


def collect_world_model_data_low_level_primitives(
    env, num_trajs, num_envs, max_path_length, num_low_level_actions_per_primitive
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
    data["actions"] = np.zeros(
        (
            num_trajs,
            (max_path_length * num_low_level_actions_per_primitive) + 1,
            9,
        ),
        dtype=np.float32,
    )
    data["high_level_actions"] = np.zeros(
        (
            num_trajs,
            (max_path_length * num_low_level_actions_per_primitive) + 1,
            env.action_space.low.shape[0] + 1,  # includes phase variable
        ),
        dtype=np.float32,
    )
    data["observations"] = np.zeros(
        (
            num_trajs,
            (max_path_length * num_low_level_actions_per_primitive) + 1,
            env.observation_space.low.shape[0],
        ),
        dtype=np.uint8,
    )
    for k in tqdm(range(num_trajs // num_envs)):
        o = env.reset()
        data["actions"][k * num_envs : k * num_envs + num_envs, 0] = np.zeros(
            (num_envs, 9)
        )
        data["high_level_actions"][
            k * num_envs : k * num_envs + num_envs, 0
        ] = np.zeros((num_envs, env.action_space.low.shape[0] + 1))
        data["observations"][k * num_envs : k * num_envs + num_envs, 0] = o
        for p in range(0, max_path_length):
            high_level_actions = [env.action_space.sample() for i in range(num_envs)]
            o, r, d, i = env.step(high_level_actions)
            low_level_actions = i["actions"]
            low_level_obs = i["observations"]
            actions = []
            obs = []
            for e in range(num_envs):
                # a0 + a1 + ...+a_space-1 -> o_space-1, o_space-1+space
                ll_a = np.array(low_level_actions[e])
                ll_o = np.array(low_level_obs[e])

                num_ll = ll_a.shape[0]
                idxs = np.linspace(0, num_ll, num_low_level_actions_per_primitive + 1)
                spacing = num_ll // (num_low_level_actions_per_primitive)
                a = ll_a.reshape(num_low_level_actions_per_primitive, spacing, -1)
                a = a.sum(axis=1)[:, :3]  # just keep sum of xyz deltas
                a = np.concatenate(
                    (a, ll_a[idxs.astype(np.int)[1:] - 1, 3:]), axis=1
                )  # try to get last index of each block
                o = ll_o[idxs.astype(np.int)[1:] - 1]  # o[space-1, 2*space-1, ...]
                actions.append(a)
                obs.append(o)

            data["actions"][
                k * num_envs : k * num_envs + num_envs,
                p * num_low_level_actions_per_primitive
                + 1 : p * num_low_level_actions_per_primitive
                + num_low_level_actions_per_primitive
                + 1,
            ] = np.array(actions)
            data["observations"][
                k * num_envs : k * num_envs + num_envs,
                p * num_low_level_actions_per_primitive
                + 1 : p * num_low_level_actions_per_primitive
                + num_low_level_actions_per_primitive
                + 1,
            ] = (
                np.array(obs)
                .transpose(0, 1, 4, 2, 3)
                .reshape(num_envs, num_low_level_actions_per_primitive, -1)
            )
            high_level_actions = np.repeat(
                np.array(high_level_actions).reshape(num_envs, 1, -1),
                num_low_level_actions_per_primitive,
                axis=1,
            )
            phases = (
                np.linspace(
                    0,
                    1,
                    num_low_level_actions_per_primitive,
                    endpoint=False,
                )
                + 1 / (num_low_level_actions_per_primitive)
            )
            phases = np.repeat(phases.reshape(1, -1), num_envs, axis=0)
            high_level_actions = np.concatenate(
                (high_level_actions, np.expand_dims(phases, -1)), axis=2
            )
            data["high_level_actions"][
                k * num_envs : k * num_envs + num_envs,
                p * num_low_level_actions_per_primitive
                + 1 : p * num_low_level_actions_per_primitive
                + num_low_level_actions_per_primitive
                + 1,
            ] = high_level_actions
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
        collect_primitives_info=True,
        include_phase_variable=True,
        render_intermediate_obs_to_info=not args.collect_data_fn
        == "collect_primitive_cloning_data",
    )
    env_suite = "metaworld"
    env_name = "reach-v2"
    env_fns = [
        lambda: make_env(env_suite, env_name, env_kwargs) for _ in range(args.num_envs)
    ]
    env = StableBaselinesVecEnv(env_fns=env_fns, start_method="fork")
    if args.collect_data_fn == "collect_world_model_data":
        data = collect_world_model_data(
            env, args.num_trajs * args.num_envs, args.num_envs, args.max_path_length
        )
        save_wm_data(data, args)
    elif args.collect_data_fn == "collect_world_model_data_low_level_primitives":
        data = collect_world_model_data_low_level_primitives(
            env,
            args.num_trajs * args.num_envs,
            args.num_envs,
            args.max_path_length,
            args.num_low_level_actions_per_primitive,
        )
        save_wm_data(data, args)
    elif args.collect_data_fn == "collect_primitive_cloning_data":
        data = collect_primitive_cloning_data(
            env,
            env_fns[0]().num_primitives,
            args.num_trajs * args.num_envs,
            args.num_envs,
            datafile=args.datafile,
        )
