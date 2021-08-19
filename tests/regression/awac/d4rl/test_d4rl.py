"""Test AWAC on Mujoco benchmark tasks."""
import os
import sys

from examples.awac.d4rl.d4rl_offline import main

from rlkit.core import logger
from rlkit.testing import csv_util

def test_awac_d4rl():
    cmd = "python examples/awac/d4rl/d4rl_offline.py --1 --local --gpu --run 0 --seed 0 --debug"
    sys.argv = cmd.split(" ")[1:]
    main()

    # check if training results match
    reference_csv = "tests/regression/awac/d4rl/id0/progress.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "progress.csv")
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["epoch", "trainer/Advantage Score Max", "trainer/Q1 Predictions Mean", "trainer/replay_buffer_len"]
    csv_util.check_equal(reference, output, keys)

    # TODO: this test seems to have some extra stochasticity to control, perhaps from the env?
    # this doesn't affect training because offline_rl is set to True
    # keys = ["expl/Average Returns", ]
    # csv_util.check_equal(reference, output, keys)

if __name__ == "__main__":
    test_awac_d4rl()
