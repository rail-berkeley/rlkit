import json
import os.path as osp

import numpy as np
import torch
import torch.nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.primitives_make_env import make_env
from rlkit.envs.wrappers.mujoco_vec_wrappers import DummyVecEnv
from rlkit.torch.model_based.dreamer.actor_models import ActorModel
from rlkit.torch.model_based.dreamer.dreamer_policy import DreamerLowLevelRAPSPolicy
from rlkit.torch.model_based.dreamer.experiments.experiment_utils import (
    preprocess_variant_llraps,
)
from rlkit.torch.model_based.dreamer.mlp import Mlp
from rlkit.torch.model_based.dreamer.visualization import visualize_rollout
from rlkit.torch.model_based.dreamer.world_models import LowlevelRAPSWorldModel


def run_trained_policy(path):
    ptu.set_gpu_mode(True)
    # # variant = json.load(open(osp.join(path, "variant.json"), "r")
    algorithm_kwargs = dict(
        num_epochs=250,
        num_eval_steps_per_epoch=30,
        min_num_steps_before_training=2500,
        num_pretrain_steps=100,
        batch_size=200,
        num_expl_steps_per_train_loop=60,
        num_train_loops_per_epoch=20,
        num_trains_per_train_loop=20,
    )
    variant = dict(
        algorithm="LLRAPS",
        version="normal",
        algorithm_kwargs=algorithm_kwargs,
        env_suite="metaworld",
        env_name="disassemble-v2",
        env_kwargs=dict(
            use_image_obs=True,
            imwidth=64,
            imheight=64,
            reward_type="sparse",
            usage_kwargs=dict(
                use_dm_backend=True,
                use_raw_action_wrappers=False,
                unflatten_images=False,
            ),
            action_space_kwargs=dict(
                collect_primitives_info=True,
                render_intermediate_obs_to_info=True,
                control_mode="primitives",
                action_scale=1,
                camera_settings={
                    "distance": 0.38227044687537043,
                    "lookat": [0.21052547, 0.32329237, 0.587819],
                    "azimuth": 141.328125,
                    "elevation": -53.203125160653144,
                },
            ),
        ),
        actor_kwargs=dict(
            discrete_continuous_dist=True,
            init_std=0.0,
            num_layers=4,
            min_std=0.1,
            dist="tanh_normal_dreamer_v1",
        ),
        vf_kwargs=dict(
            num_layers=3,
        ),
        model_kwargs=dict(
            model_hidden_size=400,
            stochastic_state_size=50,
            deterministic_state_size=200,
            rssm_hidden_size=200,
            reward_num_layers=2,
            pred_discount_num_layers=3,
            gru_layer_norm=True,
            std_act="sigmoid2",
            depth=32,
            use_prior_instead_of_posterior=True,
        ),
        trainer_kwargs=dict(
            adam_eps=1e-5,
            discount=0.8,
            lam=0.95,
            forward_kl=False,
            free_nats=1.0,
            pred_discount_loss_scale=10.0,
            kl_loss_scale=0.0,
            transition_loss_scale=0.8,
            actor_lr=8e-5,
            vf_lr=8e-5,
            world_model_lr=3e-4,
            reward_loss_scale=2.0,
            use_pred_discount=True,
            policy_gradient_loss_scale=1.0,
            actor_entropy_loss_schedule="1e-4",
            target_update_period=100,
            detach_rewards=False,
            imagination_horizon=5,
        ),
        replay_buffer_kwargs=dict(
            prioritize_fraction=0.0,
            uniform_priorities=True,
            replace=False,
        ),
        primitive_model_kwargs=dict(
            hidden_sizes=[512, 512],
            apply_embedding=False,
        ),
        num_expl_envs=5,
        num_eval_envs=1,
        expl_amount=0.3,
        save_video=True,
        low_level_action_dim=9,
        num_low_level_actions_per_primitive=5,
        effective_batch_size=400,
        pass_render_kwargs=True,
        max_path_length=5,
    )
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
    path = "/home/mdalal/research/skill_learn/rlkit/data/01-25-ll-raps-mw-raps-params-sweep-1/01-25-ll_raps_mw_raps_params_sweep_1_2022_01_25_15_28_37_0000--s-35924"
    final_reward = run_trained_policy(path)
    assert final_reward == 1.0
