"""
Run PyTorch Soft Actor Critic on HalfCheetahEnv with the "Twin" architecture
from TD3: https://arxiv.org/pdf/1802.09477.pdf
"""
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.twin_sac import TwinSAC


def experiment(variant):
    import gym
    env = NormalizedBoxEnv(gym.make('HalfCheetah-v2'))

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    net_size = variant['net_size']
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    algorithm = TwinSAC(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            max_path_length=1000,
            batch_size=128,
            discount=0.99,

            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        net_size=300,
    )
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    setup_logger('name-of-experiment', variant=variant)
    experiment(variant)
