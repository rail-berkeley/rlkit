import numpy as np

from rlkit.envs.primitives_make_env import make_env

if __name__ == "__main__":
    env_suite = "metaworld"
    V2_keys = [
        "assembly-v2",
    ]
    render_mode = "rgb_array"
    render_im_shape = (64, 64)
    render_every_step = True
    for env_name in V2_keys:
        num_envs = 1
        env = make_env(
            env_suite,
            env_name,
            env_kwargs=dict(
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
        )
        o = env.reset()
        rewards = []
        terminals = []
        low_level_actions = []
        high_level_actions = []
        observations = []
        d = True
        for i in range(5):
            if d:
                rewards_traj = [0]
                terminals_traj = [0]
                low_level_actions_traj = [np.zeros((1, 9))]
                high_level_actions_traj = [np.zeros((1, env.action_space.low.size + 1))]
                observations_traj = [o.reshape(1, *o.shape)]

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
            a[env.num_primitives :] = a[env.num_primitives :] + np.random.normal(
                0, 0.00, size=env.max_arg_len
            )
            o, r, d, info = env.step(
                a,
                render_every_step=render_every_step,
                render_mode=render_mode,
                render_im_shape=render_im_shape,
            )
            # cv2.imwrite(
            #     "assembly_{}.png".format(i),
            #     env.render(mode="rgb_array", imwidth=480, imheight=480),
            # )
            print(
                i % 5,
                r,
                "near_object",
                info["near_object"],  # reward quat
                "grasp_success",
                info["grasp_success"],
                "grasp_reward",
                info["grasp_reward"],
                "in_place_reward",
                info["in_place_reward"],  # reward success
                "unscaled_reward",
                info["unscaled_reward"],
            )
            print()
