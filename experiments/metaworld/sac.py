import argparse
import random

from gym.envs.mujoco import HalfCheetahEnv

import rlkit.envs.primitives_make_env as primitives_make_env
import rlkit.torch.pytorch_util as ptu
import rlkit.util.hyperparameter as hyp
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import run_experiment, setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithmOrig


def experiment(variant):
    env_suite = variant.get("env_suite", "kitchen")
    env_name = variant["env_name"]
    env_kwargs = variant["env_kwargs"]
    expl_env = primitives_make_env.make_env(env_suite, env_name, env_kwargs)
    eval_env = primitives_make_env.make_env(env_suite, env_name, env_kwargs)
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant["layer_size"]
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
        save_env_in_snapshot=False,
    )
    replay_buffer = EnvReplayBuffer(
        variant["replay_buffer_size"],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant["trainer_kwargs"]
    )
    algorithm = TorchBatchRLAlgorithmOrig(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant["algorithm_kwargs"]
    )
    algorithm.to(ptu.device)
    algorithm.train()


from rlkit.torch.model_based.dreamer.experiments.arguments import get_args

if __name__ == "__main__":
    args = get_args()
    if args.debug:
        exp_prefix = "test" + args.exp_prefix
    else:
        exp_prefix = args.exp_prefix
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1e7),
        algorithm_kwargs=dict(
            num_epochs=int(1e4),
            num_eval_steps_per_epoch=2500,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=10 * 1000,
            max_path_length=500,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3e-4,
            qf_lr=3e-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        env_kwargs=dict(
            control_mode="end_effector",
            action_scale=1 / 100,
            max_path_length=500,
            reward_type="dense",
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                use_image_obs=False,
                max_path_length=500,
                unflatten_images=False,
            ),
            image_kwargs=dict(),
        ),
        env_suite="metaworld",
    )
    search_space = {
        "env_name": [
            # "basketball-v2",
            # "assembly-v2",
            "disassemble-v2",
            "soccer-v2",
            "hand-insert-v2",
            # "sweep-into-v2",
            # "drawer-close-v2",
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(args.num_seeds):
            seed = random.randint(0, 100000)
            variant["seed"] = seed
            variant["exp_id"] = exp_id
            run_experiment(
                experiment,
                exp_prefix=args.exp_prefix,
                mode=args.mode,
                variant=variant,
                use_gpu=True,
                snapshot_mode="gap_and_last",
                snapshot_gap=1000,
                seed=seed,
                exp_id=exp_id,
                skip_wait=True,
            )
