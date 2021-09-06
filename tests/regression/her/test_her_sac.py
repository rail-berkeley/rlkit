import os
import sys

from examples.her import her_sac_gym_fetch_reach as her

from rlkit.core import logger
from rlkit.testing import csv_util

def test_her_online():
    logger.reset()

    # make tests small by mutating variant
    her.variant["algo_kwargs"] = dict(
        num_epochs=2,
        max_path_length=5,
        num_eval_steps_per_epoch=10,
        num_expl_steps_per_train_loop=10,
        num_trains_per_train_loop=10,
        min_num_steps_before_training=10,
        batch_size=2,
    )
    her.variant["qf_kwargs"] = dict(
        hidden_sizes=[2, 2],
    )
    her.variant["policy_kwargs"] = dict(
        hidden_sizes=[2, 2],
    )
    her.variant["seed"] = 25580

    her.main()

    reference_csv = "tests/regression/her/data_her_sac/progress.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "progress.csv")
    print("comparing reference %s against output %s" % (reference_csv, output_csv))
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["epoch", "expl/num steps total", "eval/Average Returns", "trainer/Q1 Predictions Mean", ]
    csv_util.check_equal(reference, output, keys)

if __name__ == "__main__":
    test_her_online()
