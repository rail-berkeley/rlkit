import pickle

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.core.meta_rl_algorithm import MetaRLAlgorithm
from rlkit.envs.pearl_envs import ENVS, register_pearl_envs
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.util.io import load_local_or_remote_file
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.smac.agent import SmacAgent
from rlkit.torch.smac.diagnostics import (
    get_env_info_sizes,
)
from rlkit.torch.smac.networks import MlpEncoder, MlpDecoder
from rlkit.torch.smac.launcher_util import load_buffer_onto_algo
from rlkit.torch.smac.pearl import PEARLSoftActorCriticTrainer
from rlkit.torch.sac.policies import TanhGaussianPolicy


def pearl_experiment(
        qf_kwargs=None,
        vf_kwargs=None,
        trainer_kwargs=None,
        algo_kwargs=None,
        context_encoder_kwargs=None,
        context_decoder_kwargs=None,
        policy_kwargs=None,
        env_name=None,
        env_params=None,
        latent_dim=None,
        # video/debug
        debug=False,
        _debug_do_not_sqrt=False,
        # PEARL
        n_train_tasks=0,
        n_eval_tasks=0,
        use_next_obs_in_context=False,
        saved_tasks_path=None,
        tags=None,
):
    del tags
    register_pearl_envs()
    env_params = env_params or {}
    context_encoder_kwargs = context_encoder_kwargs or {}
    context_decoder_kwargs = context_decoder_kwargs or {}
    trainer_kwargs = trainer_kwargs or {}
    base_env = ENVS[env_name](**env_params)
    if saved_tasks_path:
        task_data = load_local_or_remote_file(
            saved_tasks_path, file_type='joblib')
        tasks = task_data['tasks']
        train_task_idxs = task_data['train_task_indices']
        eval_task_idxs = task_data['eval_task_indices']
        base_env.tasks = tasks
    else:
        tasks = base_env.tasks
        task_indices = base_env.get_all_task_idx()
        train_task_idxs = list(task_indices[:n_train_tasks])
        eval_task_idxs = list(task_indices[-n_eval_tasks:])
    if hasattr(base_env, 'task_to_vec'):
        train_tasks = [base_env.task_to_vec(tasks[i]) for i in train_task_idxs]
        eval_tasks = [base_env.task_to_vec(tasks[i]) for i in eval_task_idxs]
    else:
        train_tasks = [tasks[i] for i in train_task_idxs]
        eval_tasks = [tasks[i] for i in eval_task_idxs]
    expl_env = NormalizedBoxEnv(base_env)
    eval_env = NormalizedBoxEnv(ENVS[env_name](**env_params))
    eval_env.tasks = expl_env.tasks
    reward_dim = 1

    if debug:
        algo_kwargs['max_path_length'] = 50
        algo_kwargs['batch_size'] = 5
        algo_kwargs['num_epochs'] = 5
        algo_kwargs['num_eval_steps_per_epoch'] = 100
        algo_kwargs['num_expl_steps_per_train_loop'] = 100
        algo_kwargs['num_trains_per_train_loop'] = 10
        algo_kwargs['min_num_steps_before_training'] = 100

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    if use_next_obs_in_context:
        context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim
    else:
        context_encoder_input_dim = obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2

    def create_qf():
        return ConcatMlp(
            input_size=obs_dim + action_dim + latent_dim,
            output_size=1,
            **qf_kwargs
        )

    qf1 = create_qf()
    qf2 = create_qf()
    vf = ConcatMlp(
        input_size=obs_dim + latent_dim,
        output_size=1,
        **vf_kwargs
    )

    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + latent_dim,
        action_dim=action_dim,
        **policy_kwargs,
    )
    context_encoder = MlpEncoder(
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
        hidden_sizes=[200, 200, 200],
        **context_encoder_kwargs
    )
    context_decoder = MlpDecoder(
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
        **context_decoder_kwargs
    )
    reward_predictor = context_decoder
    agent = SmacAgent(
        latent_dim,
        context_encoder,
        policy,
        reward_predictor,
        use_next_obs_in_context=use_next_obs_in_context,
        _debug_do_not_sqrt=_debug_do_not_sqrt,
    )
    trainer = PEARLSoftActorCriticTrainer(
        latent_dim=latent_dim,
        agent=agent,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        reward_predictor=reward_predictor,
        context_encoder=context_encoder,
        context_decoder=context_decoder,
        **trainer_kwargs
    )
    algorithm = MetaRLAlgorithm(
        agent=agent,
        env=expl_env,
        trainer=trainer,
        train_task_indices=train_task_idxs,
        eval_task_indices=eval_task_idxs,
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
        use_next_obs_in_context=use_next_obs_in_context,
        env_info_sizes=get_env_info_sizes(expl_env),
        **algo_kwargs
    )
    saved_path = logger.save_extra_data(
        data=dict(
            tasks=expl_env.tasks,
            train_task_indices=train_task_idxs,
            eval_task_indices=eval_task_idxs,
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
        ),
        file_name='tasks_description',
    )
    print('saved tasks description to', saved_path)
    saved_path = logger.save_extra_data(
        expl_env.tasks,
        file_name='tasks',
        mode='pickle',
    )
    print('saved raw tasks to', saved_path)

    algorithm.to(ptu.device)

    algorithm.to(ptu.device)
    algorithm.train()
