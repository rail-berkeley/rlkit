"""
Run Prototypical Soft Actor Critic on HalfCheetahEnv.

"""
import numpy as np
import click
import datetime


from gym.envs.mujoco import HalfCheetahEnv
from rlkit.envs.half_cheetah_dir import HalfCheetahDirEnv
from rlkit.envs.half_cheetah_vel import HalfCheetahVelEnv
from rlkit.envs.point_mass import PointEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.sac.policies import ProtoTanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.sac import ProtoSoftActorCritic

def datetimestamp(divider=''):
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f').replace('-', divider)

def experiment(variant):
    task_params = variant['task_params']
    env = NormalizedBoxEnv(HalfCheetahVelEnv(n_tasks=task_params['n_tasks']))
    ptu.set_gpu_mode(True)

    tasks = env.get_all_task_idx()

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    latent_dim = 5
    reward_dim = 1

    net_size = variant['net_size']
    # start with linear task encoding
    task_enc = FlattenMlp(
        hidden_sizes=[200, 200, 200], # deeper net + higher dim space generalize better
        input_size=obs_dim + reward_dim,
        output_size=latent_dim,
    )
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = ProtoTanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )

    rf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1
    )

    algorithm = ProtoSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:-10]), # list(tasks[:30]),
        eval_tasks=list(tasks[-10:]), # list(tasks[30:]),
        nets=[task_enc, policy, qf1, qf2, vf, rf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )
    algorithm.cuda()
    algorithm.train()


@click.command()
@click.argument('docker', default=0)
def main(docker):
    log_dir = '/mounts/output' if docker == 1 else 'output'
    # noinspection PyTypeChecker
    variant = dict(
        task_params=dict(
            n_tasks=50, # 20 works pretty well
            randomize_tasks=True,
        ),
        algo_params=dict(
            meta_batch=2,
            num_epochs=1000, # meta-train epochs
            num_steps_per_epoch=2, # num updates per epoch
            num_steps_per_eval=5, # num trajectories to eval on
            num_train_steps_per_itr=1000,
            train_task_batch_size=10,
            batch_size=1024, # to compute training grads from
            max_path_length=200,
            discount=0.99,
            soft_target_tau=0.005,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            context_lr=3e-4,
            reward_scale=500.,
            reparameterize=True,
            embedding_source='initial_pool',
            pickle_output_dir='data/half-cheetah/', # proto_sac_point_mass', # change this to just log dir?
        ),
        net_size=300,
    )
    # setup_logger('proto-sac-point-mass-fb-16z', variant=variant, base_log_dir=log_dir)
    setup_logger('half-cheetah-fb-16z', variant=variant, base_log_dir=log_dir)
    experiment(variant)

if __name__ == "__main__":
    main()
