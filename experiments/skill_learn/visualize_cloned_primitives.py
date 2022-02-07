import os.path as osp

import cv2
import matplotlib
import numpy as np
import torch
from arguments import get_args

from rlkit.envs.primitives_make_env import make_env
from rlkit.torch.model_based.dreamer.experiments.raps_experiment import load_primitives

matplotlib.use("Agg")
from matplotlib import pyplot as plt

if __name__ == "__main__":
    args = get_args()
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
            use_image_obs=True,
            max_path_length=5,
            unflatten_images=False,
        ),
        image_kwargs=dict(imwidth=64, imheight=64),
        collect_primitives_info=True,
        include_phase_variable=True,
    )
    env_suite = "metaworld"
    env_name = "reach-v2"

    env = make_env(env_suite, env_name, env_kwargs)

    file_path = osp.join("data/" + args.logdir + "/test.avi")

    a1 = env.action_space.sample()
    a1[0] = 100
    obs = env.reset()
    o, r, d, i = env.step(
        a1,
        render_every_step=True,
        render_mode="rgb_array",
        render_im_shape=(480, 480),
    )

    true_actions1 = np.array(i["actions"])
    true_states1 = np.array(i["robot-states"])
    img_array_true = env.img_array

    a2 = env.action_space.sample()
    a2[1] = 100
    o, r, d, i = env.step(
        a2,
        render_every_step=True,
        render_mode="rgb_array",
        render_im_shape=(480, 480),
    )

    true_actions2 = np.array(i["actions"])
    true_states2 = np.array(i["robot-states"])
    img_array_true.extend(env.img_array)

    fps = 100

    primitives = load_primitives(
        args.logdir,
        args.datafile,
        env.num_primitives,
        args.input_subselect,
        args.hidden_sizes,
    )
    env_kwargs["learned_primitives"] = primitives
    env_kwargs["use_learned_primitives"] = True
    env = make_env(env_suite, env_name, env_kwargs)

    obs = env.reset()
    o, r, d, i = env.step(
        a1,
        render_every_step=True,
        render_mode="rgb_array",
        render_im_shape=(480, 480),
    )
    predicted_actions1 = np.array(i["actions"])
    predicted_states1 = np.array(i["robot-states"])
    action_error1 = np.square(
        np.linalg.norm(true_actions1 - predicted_actions1, axis=1)
    )
    state_error1 = np.square(np.linalg.norm(true_states1 - predicted_states1, axis=1))
    img_array_predicted = env.img_array

    o, r, d, i = env.step(
        a2,
        render_every_step=True,
        render_mode="rgb_array",
        render_im_shape=(480, 480),
    )

    predicted_actions2 = np.array(i["actions"])
    predicted_states2 = np.array(i["robot-states"])
    action_error2 = np.square(
        np.linalg.norm(true_actions2 - predicted_actions2, axis=1)
    )
    state_error2 = np.square(np.linalg.norm(true_states2 - predicted_states2, axis=1))
    img_array_predicted.extend(env.img_array)

    img_array = [
        np.concatenate((img1, img2), axis=0)
        for img1, img2 in zip(img_array_true, img_array_predicted)
    ]

    print("Action error 1: ", action_error1)
    print("State error 1: ", state_error1)

    print("Action error 2: ", action_error2)
    print("State error 2: ", state_error2)
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(file_path, fourcc, 100.0, (480, 480 * 2))
    for im in img_array:
        out.write(im)
    out.release()
