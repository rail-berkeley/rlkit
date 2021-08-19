"""Test AWAC online on Mujoco benchmark tasks.

Data available for download:
https://drive.google.com/file/d/1edcuicVv2d-PqH1aZUVbO5CeRq3lqK89/view
"""
import os
import sys

from examples.awac.mujoco.awac1 import main

from rlkit.core import logger
from rlkit.testing import csv_util

def test_awac_mujoco_online():
    cmd = "python examples/awac/mujoco/awac1.py --1 --local --gpu --run 0 --seed 0 --debug"
    sys.argv = cmd.split(" ")[1:]
    main()

    # check if online training results matches
    reference_csv = "tests/regression/awac/mujoco/id0/pretrain_q.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "pretrain_q.csv")
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["trainer/batch", "trainer/Advantage Score Max", "trainer/Q1 Predictions Mean", "trainer/replay_buffer_len"]
    csv_util.check_equal(reference, output, keys)

    # TODO: this test seems to have some extra stochasticity to control, perhaps from the env?
    # check if online training results match
    # reference_csv = "tests/regression/awac/mujoco/id0/progress.csv"
    # output_csv = os.path.join(logger.get_snapshot_dir(), "progress.csv")
    # output = csv_util.get_exp(output_csv)
    # reference = csv_util.get_exp(reference_csv)
    # keys = ["epoch", "expl/Average Returns", "trainer/Advantage Score Max", "trainer/Q1 Predictions Mean", "trainer/replay_buffer_len"]
    # csv_util.check_equal(reference, output, keys)

if __name__ == "__main__":
    test_awac_mujoco_online()
