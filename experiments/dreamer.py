from autolab_core import YamlConfig
from hrl_exp.envs.franka_lift import GymFrankaLiftVecEnv
from hrl_exp.envs.wrappers import ImageEnvWrapper
from rlkit.torch.model_based.dreamer.dreamer import DreamerTrainer
from rlkit.torch.model_based.dreamer.dreamer_policy import DreamerPolicy
from rlkit.torch.model_based.dreamer.episode_replay_buffer import EpisodeReplayBuffer
from rlkit.torch.model_based.dreamer.models import WorldModel, ActorModel
from rlkit.torch.model_based.dreamer.path_collector import VecMdpPathCollector
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import torch


def experiment(variant):
    cfg_path = 'run_franka_lift.yaml'

    train_cfg = YamlConfig(cfg_path)
    train_cfg['scene']['gui'] = 0
    train_cfg['scene']['n_envs'] = 50
    train_cfg['image_preprocessor'] = None
    train_cfg['rews']['block_distance_to_lift'] = 0
    train_cfg['camera']['imshape']['width'] = 64
    train_cfg['camera']['imshape']['height'] = 64
    train_cfg['pytorch_format'] = True
    train_cfg['flatten'] = True
    expl_env = GymFrankaLiftVecEnv(train_cfg, **train_cfg['env'])
    expl_env = ImageEnvWrapper(expl_env, train_cfg)

    eval_cfg = YamlConfig(cfg_path)
    eval_cfg['scene']['gui'] = 0
    eval_cfg['scene']['n_envs'] = 5
    eval_cfg['image_preprocessor'] = None
    eval_cfg['rews']['block_distance_to_lift'] = 0
    eval_cfg['camera']['imshape']['width'] = 64
    eval_cfg['camera']['imshape']['height'] = 64
    eval_cfg['pytorch_format'] = True
    eval_cfg['flatten'] = True
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
    )
    vf = Mlp(
        hidden_sizes=[variant['model_kwargs']['model_hidden_size']]*3,
        output_size=1,
        input_size=variant['model_kwargs']['stochastic_state_size'] + variant['model_kwargs']['deterministic_state_size'],
        hidden_activation=torch.nn.functional.elu,
    )
    policy = DreamerPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
        exploration=True
    )
    eval_policy = DreamerPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
        exploration=False,
    )
    eval_path_collector = VecMdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = VecMdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EpisodeReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        4,
        obs_dim,
        action_dim,
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
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="Dreamer",
        version="normal",
        replay_buffer_size=int(1E5),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=30,
            num_trains_per_train_loop=100,
            num_expl_steps_per_train_loop=150,
            min_num_steps_before_training=1200,
            max_path_length=3,
            batch_size=625,

            # num_epochs=3000,
            # num_eval_steps_per_epoch=1,
            # num_trains_per_train_loop=1,
            # num_expl_steps_per_train_loop=1,
            # min_num_steps_before_training=0,
            # max_path_length=3,
            # batch_size=1,
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

            gradient_clip=100.0,
            lam=.95,
            imagination_horizon=2,
            free_nats=3.0,
            kl_scale=1.0,
        ),
    )
    setup_logger('dreamer-franka-lift-test', variant=variant, snapshot_mode='none')
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True
    experiment(variant)
