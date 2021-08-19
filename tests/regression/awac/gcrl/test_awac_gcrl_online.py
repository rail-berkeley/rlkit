"""Test AWAC GCRL offline+online. Requires the Sawyer environment from this multiworld
branch: https://github.com/vitchyr/multiworld/tree/ashvin-awac

Data available for download:
https://drive.google.com/file/d/1rVoR3hrFjd6Ec5TLdPpC9Ii5eIM1j7Gc/view?usp=sharing
"""
import os
import sys

from examples.awac.gcrl.pusher1 import main

from rlkit.core import logger
from rlkit.testing import csv_util

def test_awac_gcrl_online():
    cmd = "python examples/awac/gcrl/pusher1.py --1 --local --gpu --run 0 --seed 0 --debug"
    sys.argv = cmd.split(" ")[1:]
    main()

    # check if offline training results matches
    reference_csv = "tests/regression/awac/gcrl/id0/pretrain_q.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "pretrain_q.csv")
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["trainer/batch", "trainer/Advantage Score Max", "trainer/Q1 Predictions Mean", "trainer/replay_buffer_len"]
    csv_util.check_equal(reference, output, keys)

    # check if online training results match
    reference_csv = "tests/regression/awac/gcrl/id0/progress.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "progress.csv")
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["epoch", "eval/Average Returns", "trainer/Advantage Score Max", "trainer/Q1 Predictions Mean", "trainer/replay_buffer_len"]
    csv_util.check_equal(reference, output, keys)

if __name__ == "__main__":
    test_awac_gcrl_online()
