import os
import sys

from examples.iql import mujoco_finetune as iql

from rlkit.core import logger
from rlkit.testing import csv_util

def test_iql():
    logger.reset()

    # make tests small by mutating variant
    iql.variant["algo_kwargs"]["start_epoch"] = -2
    iql.variant["algo_kwargs"]["num_epochs"] = 2
    iql.variant["algo_kwargs"]["batch_size"] = 2
    iql.variant["algo_kwargs"]["num_eval_steps_per_epoch"] = 2
    iql.variant["algo_kwargs"]["num_expl_steps_per_train_loop"] = 2
    iql.variant["algo_kwargs"]["num_trains_per_train_loop"] = 100
    iql.variant["algo_kwargs"]["min_num_steps_before_training"] = 2
    iql.variant["qf_kwargs"] = dict(hidden_sizes=[2, 2])

    iql.variant["seed"] = 25580

    iql.main()

    reference_csv = "tests/regression/iql/halfcheetah_online_progress.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "progress.csv")
    print("comparing reference %s against output %s" % (reference_csv, output_csv))
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["epoch", "expl/num steps total", "expl/Average Returns", "trainer/Q1 Predictions Mean", ]
    csv_util.check_equal(reference, output, keys)

if __name__ == "__main__":
    test_iql()
