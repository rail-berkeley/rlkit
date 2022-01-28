import numpy as np
from d4rl.kitchen.kitchen_envs import *


def test_run_hinge_success():
    env = KitchenHingeCabinetV0(
        control_mode="primitives",
        action_scale=1.4,
        use_image_obs=True,
        reward_type="sparse",
    )
    env.reset()
    done = False
    ctr = 0
    image_goals = []
    max_path_length = 5
    for i in range(max_path_length):
        a = np.zeros(env.action_space.low.size)
        if ctr % max_path_length == 0:
            env.reset()
            a[env.get_idx_from_primitive_name("lift")] = 1
            a[env.num_primitives + env.primitive_name_to_action_idx["lift"]] = 1
        if ctr % max_path_length == 1:
            a[env.get_idx_from_primitive_name("angled_x_y_grasp")] = 1
            a[
                env.num_primitives
                + np.array(env.primitive_name_to_action_idx["angled_x_y_grasp"])
            ] = np.array([-np.pi / 6, -0.3, 1.4, 0])
        if ctr % max_path_length == 2:
            a[env.get_idx_from_primitive_name("move_delta_ee_pose")] = 1
            a[
                env.num_primitives
                + np.array(env.primitive_name_to_action_idx["move_delta_ee_pose"])
            ] = np.array(np.array([0.5, -1, 0]))
        if ctr % max_path_length == 3:
            a[env.get_idx_from_primitive_name("rotate_about_x_axis")] = 1
            a[
                env.num_primitives
                + np.array(env.primitive_name_to_action_idx["rotate_about_x_axis"])
            ] = np.array(
                [
                    1,
                ]
            )
        if ctr % max_path_length == 4:
            a[env.get_idx_from_primitive_name("rotate_about_x_axis")] = 1
            a[
                env.num_primitives
                + np.array(env.primitive_name_to_action_idx["rotate_about_x_axis"])
            ] = np.array(
                [
                    0,
                ]
            )
        o, r, d, i = env.step(
            a / 1.4,
        )
        ctr += 1
        if d:
            assert r == 1.0
