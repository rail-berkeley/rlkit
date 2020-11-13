import os.path as osp

import cv2
import numpy as np

from rlkit.core import logger


def video_post_epoch_func(algorithm, epoch, img_size=256):
    print(epoch)
    if epoch == -1 or epoch % 10 == 0:
        print("Generating Eval Video: ")
        env = algorithm.eval_env
        policy = algorithm.eval_data_collector._policy

        file_path = osp.join(logger.get_snapshot_dir(), "video.avi")

        img_array1 = []
        path_length = 0
        o = env.reset()
        policy.reset()

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
            path_length += 1
            img_array1.extend(env.envs[0].img_array)

        img_array2 = []
        path_length = 0
        o = env.reset()
        policy.reset()
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
            path_length += 1
            img_array2.extend(env.envs[0].img_array)

        img_array3 = []
        path_length = 0
        o = env.reset()
        policy.reset()
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
            path_length += 1
            img_array3.extend(env.envs[0].img_array)

        img_array4 = []
        path_length = 0
        o = env.reset()
        policy.reset()
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
            path_length += 1
            img_array4.extend(env.envs[0].img_array)

        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        out = cv2.VideoWriter(file_path, fourcc, 100.0, (img_size * 2, img_size * 2))
        max_len = max(
            len(img_array1), len(img_array2), len(img_array3), len(img_array4)
        )
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
        out.release()
        print("video saved to :", file_path[:-9])
