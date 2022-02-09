import json
import os.path as osp

import numpy as np
import torch
import torch.nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.primitives_make_env import make_env
from rlkit.envs.wrappers.mujoco_vec_wrappers import DummyVecEnv
from rlkit.launchers.launcher_util import set_seed
from rlkit.torch.model_based.dreamer.actor_models import ActorModel
from rlkit.torch.model_based.dreamer.dreamer_policy import DreamerLowLevelRAPSPolicy
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    preprocess_variant_llraps,
)
from rlkit.torch.model_based.dreamer.mlp import Mlp
from rlkit.torch.model_based.dreamer.world_models import LowlevelRAPSWorldModel


def run_trained_policy(path):
    ptu.set_gpu_mode(True)
    variant = json.load(open(osp.join(path, "variant.json"), "r"))
    set_seed(variant["seed"])
    variant = preprocess_variant_llraps(variant)
    env_suite = variant.get("env_suite", "kitchen")
    env_kwargs = variant["env_kwargs"]
    num_low_level_actions_per_primitive = variant["num_low_level_actions_per_primitive"]
    low_level_action_dim = variant["low_level_action_dim"]

    env_name = variant["env_name"]
    make_env_lambda = lambda: make_env(env_suite, env_name, env_kwargs)

    eval_envs = [make_env_lambda() for _ in range(1)]
    eval_env = DummyVecEnv(
        eval_envs, pass_render_kwargs=variant.get("pass_render_kwargs", False)
    )

    discrete_continuous_dist = variant["actor_kwargs"]["discrete_continuous_dist"]
    num_primitives = eval_envs[0].num_primitives
    continuous_action_dim = eval_envs[0].max_arg_len
    discrete_action_dim = num_primitives
    if not discrete_continuous_dist:
        continuous_action_dim = continuous_action_dim + discrete_action_dim
        discrete_action_dim = 0
    action_dim = continuous_action_dim + discrete_action_dim
    obs_dim = eval_env.observation_space.low.size

    primitive_model = Mlp(
        output_size=variant["low_level_action_dim"],
        input_size=variant["model_kwargs"]["stochastic_state_size"]
        + variant["model_kwargs"]["deterministic_state_size"]
        + eval_env.envs[0].action_space.low.shape[0]
        + 1,
        hidden_activation=nn.ReLU,
        num_embeddings=eval_envs[0].num_primitives,
        embedding_dim=eval_envs[0].num_primitives,
        embedding_slice=eval_envs[0].num_primitives,
        **variant["primitive_model_kwargs"],
    )

    world_model = LowlevelRAPSWorldModel(
        low_level_action_dim,
        image_shape=eval_envs[0].image_shape,
        primitive_model=primitive_model,
        **variant["model_kwargs"],
    )
    actor = ActorModel(
        variant["model_kwargs"]["model_hidden_size"],
        world_model.feature_size,
        hidden_activation=nn.ELU,
        discrete_action_dim=discrete_action_dim,
        continuous_action_dim=continuous_action_dim,
        **variant["actor_kwargs"],
    )
    actor.load_state_dict(torch.load(osp.join(path, "actor.ptc")))
    world_model.load_state_dict(torch.load(osp.join(path, "world_model.ptc")))

    actor.to(ptu.device)
    world_model.to(ptu.device)

    eval_policy = DreamerLowLevelRAPSPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
        num_low_level_actions_per_primitive=num_low_level_actions_per_primitive,
        low_level_action_dim=low_level_action_dim,
        exploration=False,
        expl_amount=0.0,
        discrete_action_dim=discrete_action_dim,
        continuous_action_dim=continuous_action_dim,
        discrete_continuous_dist=discrete_continuous_dist,
    )
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for step in range(0, variant["algorithm_kwargs"]["max_path_length"] + 1):
                if step == 0:
                    observation = eval_env.envs[0].reset()
                    eval_policy.reset(observation.reshape(1, -1))
                    policy_o = (None, observation.reshape(1, -1))
                    reward = 0
                else:
                    high_level_action, _ = eval_policy.get_action(
                        policy_o,
                    )
                    observation, reward, done, info = eval_env.envs[0].step(
                        high_level_action[0],
                    )
                    low_level_obs = np.expand_dims(np.array(info["low_level_obs"]), 0)
                    low_level_action = np.expand_dims(
                        np.array(info["low_level_action"]), 0
                    )
                    policy_o = (low_level_action, low_level_obs)
    return reward


def test_disassemble_trained_policy_success():
    # path = "/home/mdalal/research/skill_learn/rlkit/data/02-05-ll-raps-mw-refactor-replicate/02-05-ll_raps_mw_refactor_replicate_2022_02_05_00_17_28_0000--s-93433"
    import os

    directory = os.getcwd()
    path = os.path.join(
        directory,
        "tests/ll_raps/02-07-ll_raps_mw_replicate_disassemble_2022_02_07_23_00_01_0000--s-23102",
    )
    final_reward = run_trained_policy(path)
    assert final_reward == 1.0
