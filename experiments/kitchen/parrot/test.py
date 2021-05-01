import h5py
import numpy as np

with h5py.File(
    "/home/mdalal/research/spirl/data/kitchen-vision/kitchen-mixed-v0-vision-64.hdf5",
    "r",
) as f:
    dataset = dict(
        observations=np.array(f["images"]),
        terminals=np.array(f["terminals"]),
        rewards=np.array(f["rewards"]),
        actions=np.array(f["actions"]),
    )

print(dataset["observations"].shape)
import cv2

cv2.imwrite("test.png", dataset["observations"][0][:, :, ::-1])
