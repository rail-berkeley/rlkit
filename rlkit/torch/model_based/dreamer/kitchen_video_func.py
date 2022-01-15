import os.path as osp

import cv2
import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.torch.model_based.dreamer.train_world_model import visualize_rollout


def video_post_epoch_func(algorithm, epoch):
    print(epoch)
    # video_post_epoch_func_(
    #     algorithm, epoch, algorithm.eval_data_collector._policy, mode="eval"
    # )
    # video_post_epoch_func_(
    #     algorithm, epoch, algorithm.expl_data_collector._policy, mode="expl"
    # )
    visualize_rollout(
        algorithm.eval_env.envs[0],
        None,
        None,
        algorithm.trainer.world_model,
        logger.get_snapshot_dir(),
        algorithm.max_path_length,
        use_env=True,
        forcing="none",
        tag="none",
        low_level_primitives=False,
        policy=algorithm.eval_data_collector._policy,
        num_low_level_actions_per_primitive=0,
    )


def visualize_policy_low_level_func(algorithm, epoch):
    if epoch % 10 == 0:
        visualize_rollout(
            algorithm.eval_env.envs[0],
            None,
            None,
            algorithm.trainer.world_model,
            logger.get_snapshot_dir(),
            algorithm.max_path_length,
            use_env=True,
            forcing="none",
            tag="none",
            low_level_primitives=True,
            num_low_level_actions_per_primitive=algorithm.trainer.num_low_level_actions_per_primitive,
            primitive_model=algorithm.trainer.world_model.primitive_model,
            policy=algorithm.eval_data_collector._policy,
        )


@torch.no_grad()
def video_post_epoch_func_(
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
        o = env.reset()
        policy.reset(o)
        obs = np.zeros(
            (4, algorithm.max_path_length, env.observation_space.shape[0]),
            dtype=np.uint8,
        )
        actions = np.zeros((4, algorithm.max_path_length, env.action_space.shape[0]))
        while path_length < algorithm.max_path_length:
            a, agent_info = policy.get_action(
                o,
            )
            o, r, d, i = env.step(
                a,
                render_every_step=True,
                render_mode="rgb_array",
                render_im_shape=(img_size, img_size),
            )
            img_array1.extend(env.envs[0].img_array)
            obs[0, path_length] = o
            actions[0, path_length] = a
            path_length += 1

        img_array2 = []
        path_length = 0
        o = env.reset()
        policy.reset(o)
        while path_length < algorithm.max_path_length:
            a, agent_info = policy.get_action(
                o,
            )
            o, r, d, i = env.step(
                a,
                render_every_step=True,
                render_mode="rgb_array",
                render_im_shape=(img_size, img_size),
            )
            img_array2.extend(env.envs[0].img_array)
            obs[1, path_length] = o
            actions[1, path_length] = a
            path_length += 1

        img_array3 = []
        path_length = 0
        o = env.reset()
        policy.reset(o)
        while path_length < algorithm.max_path_length:
            a, agent_info = policy.get_action(
                o,
            )
            o, r, d, i = env.step(
                a,
                render_every_step=True,
                render_mode="rgb_array",
                render_im_shape=(img_size, img_size),
            )
            img_array3.extend(env.envs[0].img_array)
            obs[2, path_length] = o
            actions[2, path_length] = a
            path_length += 1

        img_array4 = []
        path_length = 0
        o = env.reset()
        policy.reset(o)
        while path_length < algorithm.max_path_length:
            a, agent_info = policy.get_action(
                o,
            )
            o, r, d, i = env.step(
                a,
                render_every_step=True,
                render_mode="rgb_array",
                render_im_shape=(img_size, img_size),
            )
            img_array4.extend(env.envs[0].img_array)
            obs[3, path_length] = o
            actions[3, path_length] = a
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
        if type(image_dist) == tuple:
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
