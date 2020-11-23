import argparse
import uuid

import torch
from hrl_exp.envs.wrappers import ImageEnvWrapper

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.torch.model_based.dreamer.rollout_functions import vec_rollout
from rlkit.torch.pytorch_util import set_gpu_mode

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = torch.load(args.file)
    policy = data["evaluation/policy"]
    env_params = data["evaluation/env_params"]
    env_class = data["evaluation/env_class"]
    env_params["scene"]["gui"] = 1
    eval_env = env_class(env_params, **env_params["env"])
    eval_env = ImageEnvWrapper(eval_env, env_params)
    print("Policy loaded")
    set_gpu_mode(True)
    policy.actor.to(ptu.device)
    policy.world_model.to(ptu.device)
    while True:
        path = vec_rollout(eval_env, policy, max_path_length=args.H, render=True,)
        if hasattr(eval_env, "log_diagnostics"):
            eval_env.log_diagnostics([path])
        logger.dump_tabular()
        print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="path to the snapshot file")
    parser.add_argument("--H", type=int, default=3, help="Max length of rollout")
    args = parser.parse_args()

    simulate_policy(args)
