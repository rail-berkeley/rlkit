import os
import os.path as osp

import cv2
import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger


def reconstruct_from_state(state, world_model):
    feat = world_model.get_features(state)
    feat = feat.reshape(-1, feat.shape[-1])
    new_img = (torch.clamp(world_model.decode(feat) + 0.5, 0, 1) * 255.0).type(
        torch.ByteTensor
    )
    new_img = ptu.get_numpy(new_img.permute(0, 2, 3, 1)[0]).astype(np.uint8)
    new_img = np.ascontiguousarray(np.copy(new_img), dtype=np.uint8)
    return new_img


def convert_img_to_save(img):
    img = np.copy(img.reshape(3, 64, 64).transpose(1, 2, 0))
    img = np.ascontiguousarray(img, dtype=np.uint8)
    return img


def add_text(vis, text, pos, scale, rgb):
    cv2.putText(
        vis,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        rgb,
        1,
        cv2.LINE_AA,
    )


@torch.no_grad()
@torch.cuda.amp.autocast()
def visualize_rollout(
    env,
    world_model,
    logdir,
    max_path_length,
    low_level_primitives,
    policy=None,
    img_size=64,
    num_rollouts=4,
    use_raps_obs=False,
    use_true_actions=True,
):
    file_path = logdir + "/"
    os.makedirs(file_path, exist_ok=True)
    print(
        f"Generating Imagination Reconstructions No Intermediate Obs: {use_raps_obs} Actual Actions: {use_true_actions}"
    )
    file_suffix = f"imagination_reconstructions_raps_obs_{use_raps_obs}_actual_actions_{use_true_actions}.png"
    file_path += file_suffix
    img_shape = (img_size, img_size, 3)
    reconstructions = np.zeros(
        (num_rollouts, max_path_length + 1, *img_shape),
        dtype=np.uint8,
    )
    obs = np.zeros(
        (num_rollouts, max_path_length + 1, *img_shape),
        dtype=np.uint8,
    )
    for rollout in range(num_rollouts):
        for step in range(0, max_path_length + 1):
            if step == 0:
                observation = env.reset()
                new_img = ptu.from_numpy(observation)
                policy.reset(observation.reshape(1, -1))
                reward = 0
                if low_level_primitives:
                    policy_o = (None, observation.reshape(1, -1))
                else:
                    policy_o = observation.reshape(1, -1)
                # hack to pass typing checks
                vis = convert_img_to_save(
                    world_model.get_image_from_obs(
                        torch.from_numpy(observation.reshape(1, -1))
                    ).numpy()
                )
                add_text(vis, "Ground Truth", (1, 60), 0.25, (0, 255, 0))
            else:
                high_level_action, state = policy.get_action(
                    policy_o, use_raps_obs, use_true_actions
                )
                state = state["state"]
                observation, reward, done, info = env.step(
                    high_level_action[0],
                )
                if low_level_primitives:
                    low_level_obs = np.expand_dims(np.array(info["low_level_obs"]), 0)
                    low_level_action = np.expand_dims(
                        np.array(info["low_level_action"]), 0
                    )
                    policy_o = (low_level_action, low_level_obs)
                else:
                    policy_o = observation.reshape(1, -1)
                (
                    primitive_name,
                    _,
                    _,
                ) = env.get_primitive_info_from_high_level_action(high_level_action[0])
                # hack to pass typing checks
                vis = convert_img_to_save(
                    world_model.get_image_from_obs(
                        torch.from_numpy(observation.reshape(1, -1))
                    ).numpy()
                )
                add_text(vis, primitive_name, (1, 60), 0.25, (0, 255, 0))
                add_text(vis, f"r: {reward}", (35, 7), 0.3, (0, 0, 0))

            obs[rollout, step] = vis
            if step != 0:
                new_img = reconstruct_from_state(state, world_model)
                if step == 1:
                    add_text(new_img, "Reconstruction", (1, 60), 0.25, (0, 255, 0))
                reconstructions[rollout, step - 1] = new_img
                reward_pred = (
                    world_model.reward(world_model.get_features(state))
                    .detach()
                    .cpu()
                    .numpy()
                    .item()
                )
                discount_pred = (
                    world_model.get_dist(
                        world_model.pred_discount(world_model.get_features(state)),
                        std=None,
                        normal=False,
                    )
                    .mean.detach()
                    .cpu()
                    .numpy()
                    .item(),
                )[0]

                print(
                    f"Rollout {rollout} Step {step - 1} Predicted Reward {reward_pred}"
                )
                print(f"Rollout {rollout} Step {step - 1} Reward {prev_r}")
                print(
                    f"Rollout {rollout} Step {step - 1} Predicted Discount {discount_pred}"
                )
                print()
            prev_r = reward
        _, state = policy.get_action(policy_o)
        state = state["state"]
        new_img = reconstruct_from_state(state, world_model)
        reconstructions[rollout, max_path_length] = new_img
        reward_pred = (
            world_model.reward(world_model.get_features(state))
            .detach()
            .cpu()
            .numpy()
            .item()
        )
        discount_pred = (
            world_model.get_dist(
                world_model.pred_discount(world_model.get_features(state)),
                std=None,
                normal=False,
            )
            .mean.detach()
            .cpu()
            .numpy()
            .item(),
        )[0]
        print(f"Rollout {rollout} Final Predicted Reward {reward_pred}")
        print(f"Rollout {rollout} Final Reward {reward}")
        print(f"Rollout {rollout} Final Predicted Discount {discount_pred}")
        print()

    im = np.zeros(
        (img_size * 2 * num_rollouts, (max_path_length + 1) * img_size, 3),
        dtype=np.uint8,
    )

    for rollout in range(num_rollouts):
        for step in range(max_path_length + 1):
            im[
                img_size * 2 * rollout : img_size * 2 * rollout + img_size,
                img_size * step : img_size * (step + 1),
            ] = obs[rollout, step]
            im[
                img_size * 2 * rollout + img_size : img_size * 2 * (rollout + 1),
                img_size * step : img_size * (step + 1),
            ] = reconstructions[rollout, step]
    cv2.imwrite(file_path, im)
    print(f"Saved Rollout Visualization to {file_path}")
    print()


def unsubsample_and_execute_ll(low_level_action, env, num_subsample_steps):
    for idx in range(low_level_action.shape[0]):
        la = low_level_action[idx]
        target = env.get_endeff_pos() + la[:3]
        for _ in range(num_subsample_steps):
            a = (target - env.get_endeff_pos()) / num_subsample_steps
            a = np.concatenate((a, la[3:]))
            observation, reward, done, info = env.low_level_step(a)
    return observation, reward, done, info


@torch.no_grad()
@torch.cuda.amp.autocast()
def visualize_primitive_unsubsampled_rollout(
    env1,
    env2,
    env3,
    logdir,
    max_path_length,
    num_low_level_actions_per_primitive,
    policy=None,
    img_size=64,
    num_rollouts=4,
):
    file_path = logdir + "/"
    os.makedirs(file_path, exist_ok=True)
    print("Generating Unsubsampled Rollout")
    file_suffix = "unsubsampled_rollout.png"
    file_path += file_suffix
    pl = max_path_length
    img_shape = (img_size, img_size, 3)
    obs1 = np.zeros(
        (num_rollouts, pl + 1, *img_shape),
        dtype=np.uint8,
    )
    obs2 = np.zeros(
        (num_rollouts, pl + 1, *img_shape),
        dtype=np.uint8,
    )
    obs3 = np.zeros(
        (num_rollouts, pl + 1, *img_shape),
        dtype=np.uint8,
    )
    for rollout in range(num_rollouts):
        for step in range(0, max_path_length + 1):
            if step == 0:
                o1 = env1.reset()
                o2 = env2.reset()
                o3 = env3.reset()
                policy_o = (None, o1.reshape(1, -1))
                policy.reset(policy_o[1])
                o1 = convert_img_to_save(o1)
                o2 = convert_img_to_save(o2)
                o3 = convert_img_to_save(o3)
                add_text(o1, "True", (1, 60), 0.25, (0, 255, 0))
                add_text(o2, "True Unsubsampled", (1, 60), 0.25, (0, 255, 0))
                add_text(o3, "Pred Unsubsampled", (1, 60), 0.25, (0, 255, 0))
                obs1[rollout, 0] = o1
                obs2[rollout, 0] = o2
                obs3[rollout, 0] = o3

            else:
                high_level_action, out = policy.get_action(policy_o)
                low_level_action_pred = out["low_level_action_pred"]
                o1, _, _, i1 = env1.step(
                    high_level_action[0],
                )
                low_level_obs = np.expand_dims(np.array(i1["low_level_obs"]), 0)
                low_level_action = np.expand_dims(np.array(i1["low_level_action"]), 0)
                policy_o = (low_level_action, low_level_obs)
                (
                    primitive_name,
                    _,
                    primitive_idx,
                ) = env1.get_primitive_info_from_high_level_action(high_level_action[0])
                num_subsample_steps = (
                    env1.primitive_idx_to_num_low_level_steps[primitive_idx]
                    // num_low_level_actions_per_primitive
                )
                o2, _, _, i2 = unsubsample_and_execute_ll(
                    low_level_action[0], env2, num_subsample_steps
                )
                if step > 1:
                    o3, _, _, i3 = unsubsample_and_execute_ll(
                        low_level_action_pred, env3, num_subsample_steps
                    )
                    obs3[rollout, step - 1] = convert_img_to_save(o3)
                    print(
                        f"Rollout: {rollout}",
                        f"Step: {step - 1}",
                        f"Primitive: {prev_primitive_name}",
                        f"LL Action Pred Error: {(np.linalg.norm(prev_low_level_action- low_level_action_pred) ** 2) / num_low_level_actions_per_primitive}",
                    )
                prev_low_level_action = low_level_action
                prev_primitive_name = primitive_name
                obs1[rollout, step] = convert_img_to_save(o1)
                obs2[rollout, step] = convert_img_to_save(o2)
        _, out = policy.get_action(policy_o)
        low_level_action_pred = out["low_level_action_pred"]
        o3, _, _, i3 = unsubsample_and_execute_ll(
            low_level_action_pred, env3, num_subsample_steps
        )
        obs3[rollout, max_path_length] = convert_img_to_save(o3)
        print(
            f"Rollout: {rollout}",
            f"Step: {max_path_length}",
            f"Primitive: {prev_primitive_name}",
            f"LL Action Pred Error: {(np.linalg.norm(prev_low_level_action- low_level_action_pred) ** 2) / num_low_level_actions_per_primitive}",
        )
        print(f"Rollout {rollout} Final Success True Actions: {i1['success']}")
        print(
            f"Rollout {rollout} Final Success True Actions Unsubsampled: {i2['success']}"
        )
        print(
            f"Rollout {rollout} Final Success Primitive Model Actions Unsubsampled: {i3['success']}"
        )

    im = np.zeros((img_size * 3 * num_rollouts, (pl + 1) * img_size, 3), dtype=np.uint8)

    for rollout in range(num_rollouts):
        for step in range(pl + 1):
            im[
                img_size * 3 * rollout : img_size * 3 * rollout + img_size,
                img_size * step : img_size * (step + 1),
            ] = obs1[rollout, step]
            im[
                img_size * 3 * rollout
                + img_size : img_size * 3 * rollout
                + 2 * img_size,
                img_size * step : img_size * (step + 1),
            ] = obs2[rollout, step]
            im[
                img_size * 3 * rollout
                + 2 * img_size : img_size * 3 * rollout
                + 3 * img_size,
                img_size * step : img_size * (step + 1),
            ] = obs3[rollout, step]
    cv2.imwrite(file_path, im)
    print(f"Saved Rollout Visualization to {file_path}")


def post_epoch_visualize_func(algorithm, epoch):
    if epoch % 10 == 0:
        visualize_rollout(
            algorithm.eval_env.envs[0],
            algorithm.trainer.world_model,
            logger.get_snapshot_dir(),
            algorithm.max_path_length,
            low_level_primitives=algorithm.low_level_primitives,
            policy=algorithm.eval_data_collector._policy,
            use_raps_obs=False,
            use_true_actions=True,
            num_rollouts=6,
        )
        if algorithm.low_level_primitives:
            visualize_rollout(
                algorithm.eval_env.envs[0],
                algorithm.trainer.world_model,
                logger.get_snapshot_dir(),
                algorithm.max_path_length,
                low_level_primitives=algorithm.low_level_primitives,
                policy=algorithm.eval_data_collector._policy,
                use_raps_obs=True,
                use_true_actions=True,
                num_rollouts=2,
            )
            visualize_rollout(
                algorithm.eval_env.envs[0],
                algorithm.trainer.world_model,
                logger.get_snapshot_dir(),
                algorithm.max_path_length,
                low_level_primitives=algorithm.low_level_primitives,
                policy=algorithm.eval_data_collector._policy,
                use_raps_obs=True,
                use_true_actions=False,
                num_rollouts=2,
            )
            visualize_rollout(
                algorithm.eval_env.envs[0],
                algorithm.trainer.world_model,
                logger.get_snapshot_dir(),
                algorithm.max_path_length,
                low_level_primitives=algorithm.low_level_primitives,
                policy=algorithm.eval_data_collector._policy,
                use_raps_obs=False,
                use_true_actions=False,
                num_rollouts=2,
            )


@torch.no_grad()
def post_epoch_video_func(
    algorithm,
    epoch,
    policy,
    img_size=256,
    mode="eval",
):
    if epoch == -1 or epoch % 100 == 0:
        print("Generating Video: ")
        env = algorithm.eval_env

        file_path = osp.join(
            logger.get_snapshot_dir(), mode + "_" + str(epoch) + "_video.avi"
        )

        img_array1 = []
        path_length = 0
        observation = env.reset()
        policy.reset(observation)
        obs = np.zeros(
            (4, algorithm.max_path_length, env.observation_space.shape[0]),
            dtype=np.uint8,
        )
        actions = np.zeros((4, algorithm.max_path_length, env.action_space.shape[0]))
        while path_length < algorithm.max_path_length:
            action, agent_info = policy.get_action(
                observation,
            )
            observation, reward, done, info = env.step(
                action,
                render_every_step=True,
                render_mode="rgb_array",
                render_im_shape=(img_size, img_size),
            )
            img_array1.extend(env.envs[0].img_array)
            obs[0, path_length] = observation
            actions[0, path_length] = action
            path_length += 1

        img_array2 = []
        path_length = 0
        observation = env.reset()
        policy.reset(observation)
        while path_length < algorithm.max_path_length:
            action, agent_info = policy.get_action(
                observation,
            )
            observation, reward, done, info = env.step(
                action,
                render_every_step=True,
                render_mode="rgb_array",
                render_im_shape=(img_size, img_size),
            )
            img_array2.extend(env.envs[0].img_array)
            obs[1, path_length] = o
            actions[1, path_length] = action
            path_length += 1

        img_array3 = []
        path_length = 0
        observation = env.reset()
        policy.reset(observation)
        while path_length < algorithm.max_path_length:
            action, agent_info = policy.get_action(
                observationo,
            )
            observation, r, d, i = env.step(
                action,
                render_every_step=True,
                render_mode="rgb_array",
                render_im_shape=(img_size, img_size),
            )
            img_array3.extend(env.envs[0].img_array)
            obs[2, path_length] = observation
            actions[2, path_length] = action
            path_length += 1

        img_array4 = []
        path_length = 0
        observation = env.reset()
        policy.reset(observation)
        while path_length < algorithm.max_path_length:
            action, agent_info = policy.get_action(
                observation,
            )
            observation, reward, done, info = env.step(
                action,
                render_every_step=True,
                render_mode="rgb_array",
                render_im_shape=(img_size, img_size),
            )
            img_array4.extend(env.envs[0].img_array)
            obs[3, path_length] = observation
            actions[3, path_length] = action
            path_length += 1

        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        out = cv2.VideoWriter(file_path, fourcc, 100.0, (img_size * 2, img_size * 2))
        max_len = max(
            len(img_array1), len(img_array2), len(img_array3), len(img_array4)
        )
        gif_clip = []
        for i in range(max_len):
            if i >= len(img_array1):
                im1 = img_array1[-1]
            else:
                im1 = img_array1[i]

            if i >= len(img_array2):
                im2 = img_array2[-1]
            else:
                im2 = img_array2[i]

            if i >= len(img_array3):
                im3 = img_array3[-1]
            else:
                im3 = img_array3[i]

            if i >= len(img_array4):
                im4 = img_array4[-1]
            else:
                im4 = img_array4[i]

            im12 = np.concatenate((im1, im2), 1)
            im34 = np.concatenate((im3, im4), 1)
            im = np.concatenate((im12, im34), 0)

            out.write(im)
            gif_clip.append(im)
        out.release()
        print("video saved to :", file_path)

        # gif_file_path = osp.join(
        #     logger.get_snapshot_dir(), mode + "_" + str(epoch) + ".gif"
        # )
        # clip = ImageSequenceClip(list(gif_clip), fps=20)
        # clip.write_gif(gif_file_path, fps=20)
        # takes way too much space
        obs, actions = ptu.from_numpy(obs), ptu.from_numpy(actions)
        (
            post,
            prior,
            post_dist,
            prior_dist,
            image_dist,
            reward_dist,
            pred_discount_dist,
            embed,
        ) = algorithm.trainer.world_model(obs.detach(), actions.detach())
        if isinstance(image_dist, tuple):
            image_dist, _ = image_dist
        image_dist_mean = image_dist.mean.detach()
        reconstructions = image_dist_mean[:, :3, :, :]
        reconstructions = (
            torch.clamp(
                reconstructions.permute(0, 2, 3, 1).reshape(
                    4, algorithm.max_path_length, 64, 64, 3
                )
                + 0.5,
                0,
                1,
            )
            * 255.0
        )
        reconstructions = ptu.get_numpy(reconstructions).astype(np.uint8)

        obs_np = ptu.get_numpy(
            obs[:, :, : 64 * 64 * 3]
            .reshape(4, algorithm.max_path_length, 3, 64, 64)
            .permute(0, 1, 3, 4, 2)
        ).astype(np.uint8)
        file_path = osp.join(
            logger.get_snapshot_dir(), mode + "_" + str(epoch) + "_reconstructions.png"
        )
        im = np.zeros((128 * 4, algorithm.max_path_length * 64, 3), dtype=np.uint8)
        for i in range(4):
            for j in range(algorithm.max_path_length):
                im[128 * i : 128 * i + 64, 64 * j : 64 * (j + 1)] = obs_np[i, j]
                im[
                    128 * i + 64 : 128 * (i + 1), 64 * j : 64 * (j + 1)
                ] = reconstructions[i, j]
        cv2.imwrite(file_path, im)
        if image_dist_mean.shape[1] == 6:
            reconstructions = image_dist_mean[:, 3:6, :, :]
            reconstructions = (
                torch.clamp(
                    reconstructions.permute(0, 2, 3, 1).reshape(
                        4, algorithm.max_path_length, 64, 64, 3
                    )
                    + 0.5,
                    0,
                    1,
                )
            ) * 255.0
            reconstructions = ptu.get_numpy(reconstructions).astype(np.uint8)

            file_path = osp.join(
                logger.get_snapshot_dir(),
                mode + "_" + str(epoch) + "_reconstructions_wrist_cam.png",
            )
            obs_np = ptu.get_numpy(
                obs[:, :, 64 * 64 * 3 : 64 * 64 * 6]
                .reshape(4, algorithm.max_path_length, 3, 64, 64)
                .permute(0, 1, 3, 4, 2)
            ).astype(np.uint8)
            im = np.zeros((128 * 4, algorithm.max_path_length * 64, 3), dtype=np.uint8)
            for i in range(4):
                for j in range(algorithm.max_path_length):
                    im[128 * i : 128 * i + 64, 64 * j : 64 * (j + 1)] = obs_np[i, j]
                    im[
                        128 * i + 64 : 128 * (i + 1), 64 * j : 64 * (j + 1)
                    ] = reconstructions[i, j]
            cv2.imwrite(file_path, im)
