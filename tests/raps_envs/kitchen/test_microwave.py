import numpy as np

from rlkit.envs.primitives_make_env import make_env


def test_run_microwave_success():
    env_suite = "kitchen"
    env_name = "microwave"
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
    for i in range(3):
        a = np.zeros(env.action_space.low.size)
        if ctr % max_path_length == 0:
            env.reset()
            a[env.get_idx_from_primitive_name("drop")] = 1
            a[env.num_primitives + env.primitive_name_to_action_idx["drop"]] = 0.55
        if ctr % max_path_length == 1:
            a[env.get_idx_from_primitive_name("angled_x_y_grasp")] = 1
            a[
                env.num_primitives
                + np.array(env.primitive_name_to_action_idx["angled_x_y_grasp"])
            ] = np.array([-np.pi / 6, -0.3, 0.95, 1])
        if ctr % max_path_length == 2:
            a[env.get_idx_from_primitive_name("move_backward")] = 1
            a[
                env.num_primitives + env.primitive_name_to_action_idx["move_backward"]
            ] = 0.6

        o, r, d, _ = env.step(
            a / 1.4,
        )
        ctr += 1
    assert r == 1.0
