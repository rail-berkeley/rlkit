"""
Run Prototypical Soft Actor Critic on HalfCheetahEnv.

"""
import click
import pathlib
import os
from rlkit.launchers.launch_experiment import experiment
from rlkit.launchers.launcher_util import setup_logger


@click.command()
@click.argument('gpu', default=0)
@click.option('--docker', default=0)
def main(gpu, docker):
    # include only task
    max_path_length = 200
    variant = dict(
        task_params=dict(
            n_tasks=180, # 20 works pretty well
            forward_backward=False,
            randomize_tasks=True,
            low_gear=False,
        ),
        algo_params=dict(
            n_train_tasks=150,
            n_eval_tasks=30,
            meta_batch=10,
            num_iterations=10000,
            num_tasks_sample=5,
            num_steps_per_task=2 * max_path_length,
            num_train_steps_per_itr=4000,
            num_evals=2,
            num_steps_per_eval=2 * max_path_length,  # num transitions to eval on
            embedding_batch_size=256,
            embedding_mini_batch_size=256,
            batch_size=256, # to compute training grads from
            max_path_length=max_path_length,
            discount=0.99,
            soft_target_tau=0.005,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            context_lr=3e-4,
            reward_scale=5.,
            sparse_rewards=False,
            reparameterize=True,
            kl_lambda=1.,
            rf_loss_scale=1.,
            use_information_bottleneck=True,  # only supports False for now
            eval_embedding_source='online_exploration_trajectories',
            train_embedding_source='online_exploration_trajectories',
            recurrent=False, # recurrent or averaging encoder
            dump_eval_paths=False,
        ),
        net_size=300,
        use_gpu=True,
        gpu_id=gpu,
        max_path_length=max_path_length,
    )
    exp_name = 'no-rf-final/ant-dir/{}'.format(gpu)

    log_dir = '/mounts/output' if docker == 1 else 'output'
    experiment_log_dir = setup_logger(exp_name, variant=variant, exp_id='ant-dir', base_log_dir=log_dir)

    # creates directories for pickle outputs of trajectories (point mass)
    pickle_dir = experiment_log_dir + '/eval_trajectories'
    pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)
    variant['algo_params']['output_dir'] = pickle_dir

    # debugging triggers a lot of printing
    DEBUG = 0
    os.environ['DEBUG'] = str(DEBUG)

    experiment(variant)

if __name__ == "__main__":
    main()
