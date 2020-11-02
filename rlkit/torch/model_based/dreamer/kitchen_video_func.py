import cv2
from rlkit.core import logger
import os.path as osp


def video_post_epoch_func(algorithm, epoch):
    print(epoch)
    if epoch == -1 or epoch % 10 == 0:
        print("Generating Eval Video: ")
        img_array = []
        env = algorithm.eval_env
        o = env.reset()
        policy = algorithm.eval_data_collector._policy
        policy.reset()
        path_length = 0
        file_path = osp.join(logger.get_snapshot_dir(), "video.avi")
        while path_length < algorithm.max_path_length:
            a, agent_info = policy.get_action(
                o,
            )
            o, r, d, i = env.step(a, render_every_step=True, render_mode="rgb_array")
            path_length += 1
            img_array.extend(env.envs[0].img_array)
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        out = cv2.VideoWriter(file_path, fourcc, 100.0, (1000, 1000))
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        print("video saved to :", file_path[:-9])
