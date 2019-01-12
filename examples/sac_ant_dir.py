"""
Run Prototypical Soft Actor Critic on HalfCheetahEnv.

"""
import numpy as np
import click
import datetime
import pathlib

from gym.envs.mujoco import HalfCheetahEnv
from rlkit.envs.ant_dir import AntDirEnv
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
    env = NormalizedBoxEnv(AntDirEnv(n_tasks=task_params['n_tasks']))
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
        train_tasks=list(tasks[:-10]), # list(tasks[:-10]), # list(tasks[:30]),
        eval_tasks=list(tasks[-10:]),# list(tasks[-10:]), # list(tasks[30:]),
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
    max_path_length = 200
    # noinspection PyTypeChecker
    variant = dict(
        task_params=dict(
            n_tasks=50, # 20 works pretty well
            randomize_tasks=True,
        ),
        algo_params=dict(
            meta_batch=16,
            num_iterations=10000,
            num_tasks_sample=5,
            num_steps_per_task=2 * max_path_length,
            num_train_steps_per_itr=2000,
            num_steps_per_eval=10 * max_path_length,  # num transitions to eval on
            batch_size=256, # to compute training grads from
            max_path_length=max_path_length,
            discount=0.99,
            soft_target_tau=0.005,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            context_lr=3e-4,
            reward_scale=5.,
            reparameterize=True,
            use_information_bottleneck=False,  # only supports False for now
            eval_embedding_source='online_exploration_trajectories',
            train_embedding_source='online_exploration_trajectories',
            dump_eval_paths=False,
        ),
        net_size=300,
        use_gpu=True,
    )
    exp_name = 'proto-sac-ant-dir-16z-batch16'

    log_dir = '/mounts/output' if docker == 1 else 'output'
    experiment_log_dir = setup_logger(exp_name, variant=variant, base_log_dir=log_dir)

    # creates directories for pickle outputs of trajectories (point mass)
    pickle_dir = experiment_log_dir + '/eval_trajectories'
    pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)
    variant['algo_params']['output_dir'] = pickle_dir

    experiment(variant)

if __name__ == "__main__":
    main()
