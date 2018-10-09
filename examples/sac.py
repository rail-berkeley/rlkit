"""
Run Prototypical Soft Actor Critic on HalfCheetahEnv.

"""
import numpy as np
from rlkit.envs.half_cheetah_dir import HalfCheetahDirEnv
from rlkit.envs.point_mass import PointEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import ProtoTanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.sac import ProtoSoftActorCritic


def experiment(variant):
    #env = NormalizedBoxEnv(HalfCheetahDirEnv())
    env = NormalizedBoxEnv(PointEnv())
    tasks = env.get_all_task_idx()

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    latent_dim = 1
    reward_dim = 1

    net_size = variant['net_size']
    # start with linear task encoding
    task_enc = FlattenMlp(
            hidden_sizes=[],
            input_size=obs_dim + reward_dim,
            output_size=latent_dim,
    )
    qf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = ProtoTanhGaussianPolicy(
        hidden_sizes=[net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    algorithm = ProtoSoftActorCritic(
        env=env,
        train_tasks=tasks,
        eval_tasks=tasks,
        nets=[task_enc, policy, qf, vf],
        meta_batch=variant['mbatch_size'],
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        meta_epochs=10,
        mbatch_size=32,
        algo_params=dict(

            num_epochs=1000, # meta-train epochs
            num_steps_per_epoch=100, # num updates per epoch
            num_steps_per_eval=100, # num obs to eval on
            batch_size=128, # to compute training grads from
            max_path_length=10,
            discount=0.99,

            soft_target_tau=0.001,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
        ),
        net_size=300,
    )
    setup_logger('proto-sac-point-mass-fb', variant=variant)
    experiment(variant)
