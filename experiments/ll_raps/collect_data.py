import argparse
import os

import numpy as np
from tqdm import tqdm

from rlkit.envs.primitives_make_env import make_env
from rlkit.envs.wrappers.mujoco_vec_wrappers import StableBaselinesVecEnv


def get_args():
    parser = argparse.ArgumentParser()

    # General Arguments
    parser.add_argument("--logdir", type=str, default="clone_primitives")
    parser.add_argument("--datafile", type=str, default="data")
    parser.add_argument("--input_subselect", type=str, default="all")
    parser.add_argument("--num_trajs", type=int, default=int(500))
    parser.add_argument("--num_epochs", type=int, default=int(1e1))
    parser.add_argument("--batch_size", type=int, default=int(50))
    parser.add_argument("--batch_len", type=int, default=int(50))
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--max_path_length", type=int, default=500)
    parser.add_argument("--num_envs", type=int, default=5)
    parser.add_argument("--train_test_split", type=float, default=0.8)
    parser.add_argument("--control_mode", type=str, default="end_effector")
    parser.add_argument("--use_prior_instead_of_posterior", type=bool, default=False)
    parser.add_argument("--num_low_level_actions_per_primitive", type=int, default=100)
    parser.add_argument(
        "--collect_data_fn", type=str, default="collect_world_model_data"
    )
    parser.add_argument("--lr", type=float, default=float(1e-3))
    parser.add_argument("--env_name", type=str, default="reach-v2")

    # parse arguments
    args = parser.parse_args()
    return args


def save_data(data, datafile):
    """
    Save world model data to a hdf5 file.
    :param: dictionary with two keys: observations and actions (numpy arrays)
    :param: args: arguments from command line
    """
    os.makedirs("data/world_model_data", exist_ok=True)
    import h5py

    f = h5py.File("data/world_model_data/" + datafile + ".hdf5", "w")
    for k, v in data.items():
        f.create_dataset(k, data=v, compression="gzip", compression_opts=9)
    print(datafile)


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
    data["low_level_actions"] = np.zeros(
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
    data["rewards"] = np.zeros((num_trajs, max_path_length + 1, 1))
    data["terminals"] = np.zeros((num_trajs, max_path_length + 1, 1))
    for k in tqdm(range(num_trajs // num_envs)):
        o = env.reset()
        data["observations"][k * num_envs : (k + 1) * num_envs, 0] = o
        for p in range(0, max_path_length):
            high_level_actions = [env.action_space.sample() for _ in range(num_envs)]
            o, r, d, i = env.step(high_level_actions)
            data["rewards"][k * num_envs : (k + 1) * num_envs, p + 1] = r.reshape(-1, 1)
            data["terminals"][k * num_envs : (k + 1) * num_envs, p + 1] = d.reshape(
                -1, 1
            )
            low_level_actions = i["actions"]
            low_level_obs = i["observations"]
            data["low_level_actions"][
                k * num_envs : (k + 1) * num_envs,
                p * num_low_level_actions_per_primitive
                + 1 : (p + 1) * num_low_level_actions_per_primitive
                + 1,
            ] = np.array(low_level_actions)
            data["observations"][
                k * num_envs : (k + 1) * num_envs,
                p * num_low_level_actions_per_primitive
                + 1 : (p + 1) * num_low_level_actions_per_primitive
                + 1,
            ] = np.array(low_level_obs).reshape(
                num_envs, num_low_level_actions_per_primitive, -1
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
                k * num_envs : (k + 1) * num_envs,
                p * num_low_level_actions_per_primitive
                + 1 : (p + 1) * num_low_level_actions_per_primitive
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
        data["actions"][k * num_envs : (k + 1) * num_envs, 0] = np.zeros(
            (num_envs, env.action_space.low.shape[0])
        )
        data["observations"][k * num_envs : (k + 1) * num_envs, 0] = o
        for p in range(1, max_path_length + 1):
            actions = [env.action_space.sample() for i in range(num_envs)]
            o, r, d, i = env.step(actions)
            data["actions"][k * num_envs : (k + 1) * num_envs, p] = np.array(actions)
            data["observations"][k * num_envs : (k + 1) * num_envs, p] = o
    return data


if __name__ == "__main__":
    args = get_args()

    env_suite = "metaworld"
    env_names = [
        # "drawer-close-v2",
        "assembly-v2",
        "disassemble-v2",
        # "peg-unplug-side-v2",
        "sweep-into-v2",
        "soccer-v2",
    ]
    num_ts = [50]
    num_lls = [5, 10]

    # env_name = args.env_name
    # num_trajs = args.num_trajs
    # num_low_level_actions_per_primitive = args.num_low_level_actions_per_primitive
    for env_name in env_names:
        for num_trajs in num_ts:
            for num_low_level_actions_per_primitive in num_lls:
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
                    num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
                )
                datafile = "wm_H_{}_T_{}_E_{}_P_{}_raps_ll_hl_even_rt_{}".format(
                    args.max_path_length,
                    num_trajs,
                    args.num_envs,
                    num_low_level_actions_per_primitive,
                    env_name,
                )
                env_fns = [
                    lambda: make_env(env_suite, env_name, env_kwargs)
                    for _ in range(args.num_envs)
                ]
                env = StableBaselinesVecEnv(env_fns=env_fns, start_method="fork")
                if args.collect_data_fn == "collect_world_model_data":
                    data = collect_world_model_data(
                        env,
                        num_trajs * args.num_envs,
                        args.num_envs,
                        args.max_path_length,
                    )
                    save_data(data, datafile)
                elif (
                    args.collect_data_fn
                    == "collect_world_model_data_low_level_primitives"
                ):
                    data = collect_world_model_data_low_level_primitives(
                        env,
                        num_trajs * args.num_envs,
                        args.num_envs,
                        args.max_path_length,
                        num_low_level_actions_per_primitive,
                    )
                    save_data(data, datafile)
                elif args.collect_data_fn == "collect_primitive_cloning_data":
                    data = collect_primitive_cloning_data(
                        env,
                        env_fns[0]().num_primitives,
                        num_trajs * args.num_envs,
                        args.num_envs,
                        datafile=datafile,
                    )
