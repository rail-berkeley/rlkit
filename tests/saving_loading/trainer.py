import tempfile

import torch.nn as nn

from rlkit.envs.primitives_make_env import make_env
from rlkit.envs.wrappers.mujoco_vec_wrappers import DummyVecEnv
from rlkit.torch.model_based.dreamer.actor_models import ActorModel
from rlkit.torch.model_based.dreamer.dreamer_policy import DreamerPolicy
from rlkit.torch.model_based.dreamer.dreamer_v2 import DreamerV2Trainer
from rlkit.torch.model_based.dreamer.mlp import Mlp
from rlkit.torch.model_based.dreamer.path_collector import VecMdpPathCollector
from rlkit.torch.model_based.dreamer.world_models import WorldModel


def test_trainer_save_load():
    env_kwargs = dict(
        use_image_obs=True,
        imwidth=64,
        imheight=64,
        reward_type="sparse",
        usage_kwargs=dict(
            max_path_length=5,
            use_dm_backend=True,
            use_raw_action_wrappers=False,
            unflatten_images=False,
        ),
        action_space_kwargs=dict(
            control_mode="primitives",
            action_scale=1,
            camera_settings={
                "distance": 0.38227044687537043,
                "lookat": [0.21052547, 0.32329237, 0.587819],
                "azimuth": 141.328125,
                "elevation": -53.203125160653144,
            },
        ),
    )
    actor_kwargs = dict(
        discrete_continuous_dist=True,
        init_std=0.0,
        num_layers=4,
        min_std=0.1,
        dist="tanh_normal_dreamer_v1",
    )
    vf_kwargs = dict(
        num_layers=3,
    )
    model_kwargs = dict(
        model_hidden_size=400,
        stochastic_state_size=50,
        deterministic_state_size=200,
        rssm_hidden_size=200,
        reward_num_layers=2,
        pred_discount_num_layers=3,
        gru_layer_norm=True,
        std_act="sigmoid2",
        use_prior_instead_of_posterior=False,
    )
    trainer_kwargs = dict(
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
    )
    env_suite = "metaworld"
    env_name = "disassemble-v2"
    eval_envs = [make_env(env_suite, env_name, env_kwargs) for _ in range(1)]
    eval_env = DummyVecEnv(
        eval_envs,
    )

    discrete_continuous_dist = True
    continuous_action_dim = eval_envs[0].max_arg_len
    discrete_action_dim = eval_envs[0].num_primitives
    if not discrete_continuous_dist:
        continuous_action_dim = continuous_action_dim + discrete_action_dim
        discrete_action_dim = 0
    action_dim = continuous_action_dim + discrete_action_dim

    world_model = WorldModel(
        action_dim,
        image_shape=eval_envs[0].image_shape,
        **model_kwargs,
    )
    actor = ActorModel(
        model_kwargs["model_hidden_size"],
        world_model.feature_size,
        hidden_activation=nn.ELU,
        discrete_action_dim=discrete_action_dim,
        continuous_action_dim=continuous_action_dim,
        **actor_kwargs,
    )

    vf = Mlp(
        hidden_sizes=[model_kwargs["model_hidden_size"]] * vf_kwargs["num_layers"],
        output_size=1,
        input_size=world_model.feature_size,
        hidden_activation=nn.ELU,
    )
    target_vf = Mlp(
        hidden_sizes=[model_kwargs["model_hidden_size"]] * vf_kwargs["num_layers"],
        output_size=1,
        input_size=world_model.feature_size,
        hidden_activation=nn.ELU,
    )

    trainer = DreamerV2Trainer(
        actor,
        vf,
        target_vf,
        world_model,
        eval_envs[0].image_shape,
        **trainer_kwargs,
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        trainer.save(tmpdirname, "trainer.pkl")
        trainer = DreamerV2Trainer(
            actor,
            vf,
            target_vf,
            world_model,
            eval_envs[0].image_shape,
            **trainer_kwargs,
        )
        new_trainer = trainer.load(tmpdirname, "trainer.pkl")
