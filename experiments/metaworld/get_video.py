import argparse
import uuid
from glob import glob

import cv2
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.mujoco_vec_wrappers import DummyVecEnv, make_env, make_metaworld_env
from rlkit.envs.primitives_wrappers import ImageEnvMetaworld
from rlkit.torch.pytorch_util import set_gpu_mode

filename = str(uuid.uuid4())


def simulate_policy(file):
    data = torch.load(file)
    policy = data["evaluation/policy"]
    env_params = data["evaluation/env_params"]
    env_class = data["evaluation/env_class"]

    eval_envs = [
        ImageEnvMetaworld(
            make_metaworld_env(env_class, env_params), imwidth=64, imheight=64
        )
    ]
    env = DummyVecEnv(eval_envs)
    print("Policy loaded")
    set_gpu_mode(True)
    policy.actor.to(ptu.device)
    policy.world_model.to(ptu.device)
    i = 0
    img_array = []
    o = env.reset()
    policy.reset()
    path_length = 0
    while path_length < env.envs[0].max_steps:
        a, agent_info = policy.get_action(
            o,
        )
        env.step(a, render_every_step=True, render_mode="rgb_array")
        path_length += 1
        img_array.extend(env.envs[0].img_array)
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    out = cv2.VideoWriter(file[:-10] + "final.avi", fourcc, 100.0, (1000, 1000))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print("video saved to :", file[:-10])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="path to the snapshot file")
    parser.add_argument("--H", type=int, default=5, help="Max length of rollout")
    args = parser.parse_args()
    for p in glob(args.path + "/*/"):
        file = p + "params.pkl"
        print(file)
        try:
            simulate_policy(file)
        except Exception as e:
            print(e)
