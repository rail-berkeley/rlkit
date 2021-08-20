import os
import sys

from examples import ddpg

from rlkit.core import logger
from rlkit.testing import csv_util

def test_ddpg_online():
    # make tests small by mutating variant
    ddpg.variant["algorithm_kwargs"]["num_epochs"] = 2
    ddpg.variant["algorithm_kwargs"]["batch_size"] = 2
    ddpg.variant["qf_kwargs"] = dict(hidden_sizes=[2, 2])
    ddpg.variant["policy_kwargs"] = dict(hidden_sizes=[2, 2])
    ddpg.variant["seed"] = 25580

    ddpg.main()

    reference_csv = "tests/regression/basic/data_ddpg/progress.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "progress.csv")
    print("comparing reference %s against output %s" % (reference_csv, output_csv))
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["epoch", "expl/num steps total", "eval/Average Returns", "trainer/Q Predictions Mean", ]
    csv_util.check_equal(reference, output, keys)

if __name__ == "__main__":
    test_ddpg_online()
