import click
import os
import pathlib

from rlkit.launchers.launch_experiment import experiment
from examples.default import make_variant

@click.command()
@click.argument('gpu', default=0)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
def main(gpu, docker, debug):

    max_path_length = 20
    variant = make_variant(max_path_length)

    variant['env_name'] = 'point-robot'
    variant['n_train_tasks'] = 5
    variant['n_eval_tasks'] = 5

    env_params = variant['env_params']
    env_params['n_tasks'] = 10

    algo_params = variant['algo_params']
    algo_params['reward_scale'] = 100.
    algo_params['kl_lambda'] = .1
    algo_params['train_embedding_source'] ='online_exploration_trajectories'
    algo_params['eval_embedding_source'] ='online_exploration_trajectories'

    util_params = variant['util_params']
    util_params['gpu_id'] = gpu

    print(variant)
    experiment(variant)

if __name__ == "__main__":
    main()
