import os

from examples.simplegym import sac

from rlkit.core import logger
from rlkit.testing import csv_util

def test_sac_online():
    logger.reset()

    # make tests small by mutating variant
    sac.variant["algorithm_kwargs"]["num_epochs"] = 2
    sac.variant["algorithm_kwargs"]["batch_size"] = 2
    sac.variant["algorithm_kwargs"]["num_trains_per_train_loop"] = 100
    sac.variant["qf_kwargs"] = dict(hidden_sizes=[2, 2])
    sac.variant["policy_kwargs"] = dict(hidden_sizes=[2, 2])
    sac.variant["seed"] = 25580

    sac.main()

    reference_csv = "tests/regression/simplegym/test_sac_progress.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "progress.csv")
    print("comparing reference %s against output %s" % (reference_csv, output_csv))
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["epoch", "expl/num steps total", "eval/Average Returns", "trainer/Q1 Predictions Mean", ]
    csv_util.check_equal(reference, output, keys)

if __name__ == "__main__":
    test_sac_online()
