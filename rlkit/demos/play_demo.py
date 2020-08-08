import numpy as np

import sys
# print(sys.path)
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

import cv2
import sys
import pickle

def play_demos(path):
    data = pickle.load(open(path, "rb"))
    # data = np.load(path, allow_pickle=True)

    for traj in data:
        obs = traj["observations"]

        for o in obs:
            img = o["image_observation"].reshape(3, 500, 300)[:, 60:, :240].transpose()
            img = img[:, :, ::-1]
            cv2.imshow('window', img)
            cv2.waitKey(100)

if __name__ == '__main__':
    demo_path = sys.argv[1]
    play_demos(demo_path)
