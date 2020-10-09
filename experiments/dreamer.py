from autolab_core import YamlConfig
from hrl_exp.envs.franka_lift import GymFrankaLiftVecEnv
from hrl_exp.envs.wrappers import ImageEnvWrapper
from rlkit.torch.model_based.dreamer.dreamer import DreamerTrainer
from rlkit.torch.model_based.dreamer.dreamer_policy import DreamerPolicy, ActionSpaceSamplePolicy
from rlkit.torch.model_based.dreamer.episode_replay_buffer import EpisodeReplayBuffer
from rlkit.torch.model_based.dreamer.mlp import Mlp
from rlkit.torch.model_based.dreamer.models import WorldModel, ActorModel
from rlkit.torch.model_based.dreamer.path_collector import VecMdpPathCollector
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import torch
import rlkit.util.hyperparameter as hyp
from os.path import join
import os
import rlkit
import argparse
import pickle

def experiment(variant):
    rlkit_project_dir = join(os.path.dirname(rlkit.__file__), os.pardir)
    cfg_path = join(rlkit_project_dir, 'experiments/run_franka_lift.yaml')

    train_cfg = YamlConfig(cfg_path)

    train_cfg['scene']['n_envs'] = variant['env_kwargs']['n_train_envs']
    train_cfg['rews']['block_distance_to_lift'] = variant['env_kwargs']['block_distance_to_lift']
    train_cfg['env']['fixed_schema'] = variant['env_kwargs']['fixed_schema']

    train_cfg['pytorch_format'] = True
    train_cfg['flatten'] = True
    expl_env = GymFrankaLiftVecEnv(train_cfg, **train_cfg['env'])
    expl_env = ImageEnvWrapper(expl_env, train_cfg)

    eval_cfg = pickle.loads(pickle.dumps(train_cfg))
    eval_cfg['scene']['n_envs'] = variant['env_kwargs']['n_eval_envs']

    eval_env = GymFrankaLiftVecEnv(eval_cfg, **eval_cfg['env'])
    eval_env = ImageEnvWrapper(eval_env, eval_cfg)

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    world_model = WorldModel(
        action_dim,
        **variant['model_kwargs'],
    )
    actor = ActorModel(
        [variant['model_kwargs']['model_hidden_size']]*4,
        variant['model_kwargs']['stochastic_state_size'] + variant['model_kwargs']['deterministic_state_size'],
        action_dim,
        hidden_activation=torch.nn.functional.elu,
        split_size=expl_env.wrapped_env.num_primitives,
        split_dist=variant['actor_kwargs']['split_dist'] and (not variant['env_kwargs']['fixed_schema'])
    )
    vf = Mlp(
        hidden_sizes=[variant['model_kwargs']['model_hidden_size']]*3,
        output_size=1,
        input_size=variant['model_kwargs']['stochastic_state_size'] + variant['model_kwargs']['deterministic_state_size'],
        hidden_activation=torch.nn.functional.elu,
    )

    expl_policy = DreamerPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
        split_dist=variant['actor_kwargs']['split_dist'] and (not variant['env_kwargs']['fixed_schema']),
        split_size=expl_env.wrapped_env.num_primitives,
        exploration=True
    )
    eval_policy = DreamerPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
        split_dist=variant['actor_kwargs'],
        split_size=expl_env.wrapped_env.num_primitives,
        exploration=False,
    )

    rand_policy = ActionSpaceSamplePolicy(expl_env)

    expl_path_collector = VecMdpPathCollector(
        expl_env,
        expl_policy,
    )

    eval_path_collector = VecMdpPathCollector(
        eval_env,
        eval_policy,
    )

    replay_buffer = EpisodeReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        4,
        obs_dim,
        action_dim,
        replace=False
    )
    trainer = DreamerTrainer(
        env=eval_env,
        world_model=world_model,
        actor=actor,
        vf=vf,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        pretrain_policy=rand_policy,
        **variant['algorithm_kwargs'],
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_prefix', type=str, default='')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    if args.debug:
        algorithm_kwargs = dict(
            num_epochs=3,
            num_eval_steps_per_epoch=30,
            num_trains_per_train_loop=10,
            num_expl_steps_per_train_loop=150,  # 200 samples since num_envs = 50 and max_path_length + 1 = 4
            min_num_steps_before_training=100,
            num_pretrain_steps=100,
            num_train_loops_per_epoch=1,
            max_path_length=3,
            batch_size=50,
        )
        exp_prefix = 'test'+args.exp_prefix
    else:
        algorithm_kwargs = dict(
            num_epochs=50,
            num_eval_steps_per_epoch=30,
            num_trains_per_train_loop=200,
            num_expl_steps_per_train_loop=150, #200 samples since num_envs = 50 and max_path_length + 1 = 4
            min_num_steps_before_training=5000,
            num_pretrain_steps=100,
            num_train_loops_per_epoch=5,
            max_path_length=3,
            batch_size=625,
        )
        exp_prefix=args.exp_prefix
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="Dreamer",
        version="normal",
        replay_buffer_size=int(1E5),
        algorithm_kwargs=algorithm_kwargs,
        env_kwargs = dict(
            block_distance_to_lift=0,
            n_train_envs=50,
            n_eval_envs=10,
            fixed_schema=True,
        ),
        actor_kwargs=dict(
          split_dist=True,
        ),
        model_kwargs=dict(
            model_hidden_size=400,
            stochastic_state_size=60,
            deterministic_state_size=400,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            reward_scale=1.0,

            actor_lr=8e-5,
            vf_lr=8e-5,
            world_model_lr=6e-4,
            use_amp=True,
            opt_level="O1",
            gradient_clip=100.0,
            lam=.95,
            imagination_horizon=4,
            free_nats=3.0,
            kl_scale=1.0,
            optimizer_class='apex_adam',
            pcont_scale=10.0,
            use_pcont=True,
        ),
    )

    search_space = {
        'trainer_kwargs.opt_level':[
            "O0",
            "O1",
        ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local' #never use here_no_doodad with IG (always install doodad!)

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=True,
                snapshot_mode='none',
            )
