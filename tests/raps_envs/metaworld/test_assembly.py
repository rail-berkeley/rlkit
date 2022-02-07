import numpy as np

from rlkit.envs.primitives_make_env import make_env


def test_run_assembly_success():
    env_suite = "metaworld"
    env_name = "assembly-v2"
    env_kwargs = dict(
        use_image_obs=True,
        imwidth=64,
        imheight=64,
        reward_type="sparse",
        usage_kwargs=dict(
            use_dm_backend=True,
            use_raw_action_wrappers=False,
            unflatten_images=False,
            max_path_length=5,
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
    render_mode = "rgb_array"
    render_im_shape = (64, 64)
    render_every_step = True
    env = make_env(
        env_suite,
        env_name,
        env_kwargs,
    )
    o = env.reset()
    for i in range(5):
        a = env.action_space.sample()
        a = np.zeros_like(a)
        if i % 5 == 0:
            primitive = "top_x_y_grasp"
            a[env.get_idx_from_primitive_name(primitive)] = 1
            a[
                env.num_primitives
                + np.array(env.primitive_name_to_action_idx[primitive])
            ] = [0.25, 0.0, -0.6, 1]
        elif i % 5 == 1:
            primitive = "lift"
            a[env.get_idx_from_primitive_name(primitive)] = 1
            a[
                env.num_primitives
                + np.array(env.primitive_name_to_action_idx[primitive])
            ] = 0.4
        elif i % 5 == 2:
            primitive = "move_forward"
            a[env.get_idx_from_primitive_name(primitive)] = 1
            a[
                env.num_primitives
                + np.array(env.primitive_name_to_action_idx[primitive])
            ] = 0.45
        elif i % 5 == 3:
            primitive = "move_right"
            a[env.get_idx_from_primitive_name(primitive)] = 1
            a[
                env.num_primitives
                + np.array(env.primitive_name_to_action_idx[primitive])
            ] = 0.05
        elif i % 5 == 3:
            primitive = "open_gripper"
            a[env.get_idx_from_primitive_name(primitive)] = 1
            a[
                env.num_primitives
                + np.array(env.primitive_name_to_action_idx[primitive])
            ] = 1
        o, r, d, info = env.step(
            a,
            render_every_step=render_every_step,
            render_mode=render_mode,
            render_im_shape=render_im_shape,
        )
    assert r == 1.0
