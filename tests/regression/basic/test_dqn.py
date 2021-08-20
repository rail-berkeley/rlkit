import os
import sys

from examples import dqn_and_double_dqn as dqn

from rlkit.core import logger
from rlkit.testing import csv_util

def test_dqn_online():
    # make tests small by mutating variant
    dqn.variant["algorithm_kwargs"]["num_epochs"] = 2
    dqn.variant["algorithm_kwargs"]["batch_size"] = 2
    dqn.variant["seed"] = 25580

    dqn.main()

    # TODO: there is an extra source of randomness so it doesn't match numerically
    reference_csv = "tests/regression/basic/data_dqn/progress.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "progress.csv")
    print("comparing reference %s against output %s" % (reference_csv, output_csv))
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["epoch", "expl/num steps total", ] # "eval/Average Returns", "trainer/QF Loss"]
    csv_util.check_equal(reference, output, keys)

if __name__ == "__main__":
    test_dqn_online()
