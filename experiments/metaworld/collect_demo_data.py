import argparse
import uuid
from glob import glob

import cv2
import h5py
import numpy as np
import torch

import rlkit.envs.primitives_make_env as primitives_make_env
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.pytorch_util import set_gpu_mode
from rlkit.torch.sac.policies.base import MakeDeterministic

filename = str(uuid.uuid4())


def append_dataset(
    env, policy, images, observations, rewards, terminals, actions, infos, num_demos=50
):
    num_demos_collected = 0
    while num_demos_collected < num_demos:
        d = False
        o = env.reset()
        policy.reset()
        (
            rollout_images,
            rollout_observations,
            rollout_rewards,
            rollout_terminals,
            rollout_actions,
            rollout_infos,
        ) = ([], [], [], [], [], [])
        did_succeed = False
        while not d:
            a, _ = policy.get_action(
                o,
            )
            o, r, d, i = env.step(a, render_every_step=False)
            t = int(d)
            im = env._get_image().astype(np.uint8).reshape(3, 64, 64).transpose(1, 2, 0)
            rollout_images.append(im)
            rollout_observations.append(o)
            rollout_rewards.append(r)
            rollout_terminals.append(t)
            rollout_actions.append(a)
            rollout_infos.append(i["success"])
            if i["success"] > 0:
                did_succeed = True
        if did_succeed:
            images.extend(rollout_images)
            observations.extend(rollout_observations)
            rewards.extend(rollout_rewards)
            terminals.extend(rollout_terminals)
            actions.extend(rollout_actions)
            infos.extend(rollout_infos)
            num_demos_collected += 1
    return images, observations, rewards, terminals, actions, infos


def simulate_policy(
    env_name, file, images, observations, rewards, terminals, actions, infos
):
    data = torch.load(file)
    policy = data["exploration/policy"]
    env_suite = "metaworld"
    env_kwargs = dict(
        control_mode="end_effector",
        action_scale=1 / 100,
        max_path_length=500,
        reward_type="dense",
        usage_kwargs=dict(
            use_dm_backend=True,
            use_raw_action_wrappers=False,
            use_image_obs=False,
            max_path_length=500,
            unflatten_images=False,
        ),
        image_kwargs=dict(),
    )

    env = primitives_make_env.make_env(env_suite, env_name, env_kwargs)
    env.env.imwidth = 64
    env.env.imheight = 64
    # camera_settings = {
    #         "distance": 0.38227044687537043,
    #         "lookat": [0.21052547, 0.32329237, 0.587819],
    #         "azimuth": 141.328125,
    #         "elevation": -53.203125160653144,
    #     }
    camera_settings = {
        "distance": 0.37864894603997346,
        "lookat": [0.28839241, 0.55843923, 0.70374719],
        "azimuth": -180.0,
        "elevation": -64.68749995809048,
    }
    env.env.reset_camera(camera_settings)
    set_gpu_mode(True)
    policy.to(ptu.device)
    print(policy)
    i = 0
    path_length = 0

    append_dataset(
        env,
        policy,
        images,
        observations,
        rewards,
        terminals,
        actions,
        infos,
        num_demos=50,
    )
    print(max(rewards))
    return images, observations, rewards, terminals, actions, infos


if __name__ == "__main__":
    env_names = [
        "basketball-v2",
        "assembly-v2",
        # "disassemble-v2",
        "soccer-v2",
        "sweep-into-v2",
        "drawer-close-v2",
    ]
    files = [
        "data/05-10-mw-sac-state-all-v1/05-10-mw_sac_state_all_v1_2021_05_10_00_10_24_0000--s-9702/params.pkl",
        "data/05-10-mw-sac-state-all-v1/05-10-mw_sac_state_all_v1_2021_05_10_00_10_24_0001--s-10364/params.pkl",
        # "data/05-10-mw-sac-state-all-v1/05-10-mw_sac_state_all_v1_2021_05_10_00_10_24_0000--s-9702/params.pkl",
        "data/05-10-mw-sac-state-all-v1/05-10-mw_sac_state_all_v1_2021_05_10_00_10_24_0003--s-93447/itr_3000.pkl",
        "data/05-10-mw-sac-state-all-v1/05-10-mw_sac_state_all_v1_2021_05_10_00_10_24_0004--s-90466/itr_3000.pkl",
        "data/05-10-mw-sac-state-all-v1/05-10-mw_sac_state_all_v1_2021_05_10_00_10_24_0005--s-35467/params.pkl",
    ]
    images, observations, rewards, terminals, actions, infos = [], [], [], [], [], []
    for env_name, file in zip(env_names, files):
        images, observations, rewards, terminals, actions, infos = simulate_policy(
            env_name, file, images, observations, rewards, terminals, actions, infos
        )

    dataset = dict(
        images=np.array(images),
        observations=np.array(observations),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
        actions=np.array(actions),
        infos=infos,
    )
    print(dataset.keys(), print(dataset["images"].shape))
    save_filename = (
        "/home/mdalal/research/spirl/data/metaworld-vision/%s.hdf5"
        % "metaworld-total-v0-vision-64"
    )
    print("Saving dataset to %s." % save_filename)
    h5_dataset = h5py.File(save_filename, "w")
    for key in dataset:
        h5_dataset.create_dataset(key, data=dataset[key], compression="gzip")
    print("Done.")
