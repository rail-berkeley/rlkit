import numpy as np

from rlkit.envs.primitives_make_env import make_env


def test_top_burner_success():
    env_suite = "kitchen"
    env_name = "top_left_burner"
    env_kwargs = dict(
        reward_type="sparse",
        use_image_obs=True,
        action_scale=1.4,
        use_workspace_limits=True,
        control_mode="primitives",
        usage_kwargs=dict(
            use_dm_backend=True,
            use_raw_action_wrappers=False,
            unflatten_images=False,
            max_path_length=5,
        ),
        action_space_kwargs=dict(),
    )
    env = make_env(
        env_suite,
        env_name,
        env_kwargs,
    )
    env.reset()
    ctr = 0
    max_path_length = 3
    for i in range(max_path_length):
        a = np.zeros(env.action_space.low.size)
        if ctr % max_path_length == 0:
            env.reset()
            a[env.get_idx_from_primitive_name("lift")] = 1
            a[env.num_primitives + env.primitive_name_to_action_idx["lift"]] = 0.6
        if ctr % max_path_length == 1:
            a[env.get_idx_from_primitive_name("angled_x_y_grasp")] = 1
            a[
                env.num_primitives
                + np.array(env.primitive_name_to_action_idx["angled_x_y_grasp"])
            ] = np.array([0, 0.5, 1, 1])
        if ctr % max_path_length == 2:
            a[env.get_idx_from_primitive_name("rotate_about_y_axis")] = 1
            a[
                env.num_primitives
                + env.primitive_name_to_action_idx["rotate_about_y_axis"]
            ] = (-np.pi / 4)

        o, r, d, _ = env.step(
            a / 1.4,
        )

        ctr += 1
    assert r == 1.0
