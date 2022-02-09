import tempfile

import torch.nn as nn

from rlkit.envs.primitives_make_env import make_env
from rlkit.envs.wrappers.mujoco_vec_wrappers import DummyVecEnv
from rlkit.torch.model_based.dreamer.actor_models import ActorModel
from rlkit.torch.model_based.dreamer.dreamer_policy import DreamerPolicy
from rlkit.torch.model_based.dreamer.path_collector import VecMdpPathCollector
from rlkit.torch.model_based.dreamer.world_models import WorldModel


def test_path_collector_save_load():
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
    obs_dim = eval_env.observation_space.low.size

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

    eval_policy = DreamerPolicy(
        world_model,
        actor,
        obs_dim,
        action_dim,
        exploration=False,
        expl_amount=0.0,
        discrete_action_dim=discrete_action_dim,
        continuous_action_dim=continuous_action_dim,
        discrete_continuous_dist=discrete_continuous_dist,
    )

    eval_path_collector = VecMdpPathCollector(
        eval_env,
        eval_policy,
        save_env_in_snapshot=False,
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        eval_path_collector.save(tmpdirname, "path_collector.pkl")
        eval_path_collector = VecMdpPathCollector(
            eval_env,
            eval_policy,
            save_env_in_snapshot=False,
        )
        new_path_collector = eval_path_collector.load(tmpdirname, "path_collector.pkl")
