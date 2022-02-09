import tempfile

from rlkit.envs.primitives_make_env import make_env
from rlkit.envs.wrappers.mujoco_vec_wrappers import DummyVecEnv, StableBaselinesVecEnv


def test_dummy_vec_env_save_load():
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
    env_suite = "metaworld"
    env_name = "disassemble-v2"
    make_env_lambda = lambda: make_env(env_suite, env_name, env_kwargs)

    n_envs = 2
    envs = [make_env_lambda() for _ in range(n_envs)]
    env = DummyVecEnv(
        envs,
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        env.save(tmpdirname, "env.pkl")
        env = DummyVecEnv(
            envs[0:1],
        )
        new_env = env.load(tmpdirname, "env.pkl")
    assert new_env.n_envs == n_envs


def test_vec_env_save_load():
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
    env_suite = "metaworld"
    env_name = "disassemble-v2"
    n_envs = 2

    env_fns = [lambda: make_env(env_suite, env_name, env_kwargs) for _ in range(n_envs)]
    env = StableBaselinesVecEnv(
        env_fns=env_fns,
        start_method="fork",
        reload_state_args=(n_envs, make_env, (env_suite, env_name, env_kwargs)),
    )
    with tempfile.TemporaryDirectory() as tmpdirname:
        env.save(tmpdirname, "env.pkl")
        env = StableBaselinesVecEnv(env_fns=env_fns[0:1], start_method="fork")
        new_env = env.load(tmpdirname, "env.pkl")
    assert new_env.n_envs == n_envs
