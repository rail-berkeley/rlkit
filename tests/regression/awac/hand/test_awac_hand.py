"""Test AWAC offline+online on Mujoco dextrous manipulation tasks.

Running the dexterous manipulation experiments requires setting up the
environments in this repository: https://github.com/aravindr93/hand_dapg.
You can also use the follwing docker image, which has the required
dependencies set up: anair17/railrl-hand-v3

For the mj_envs repository, please use: https://github.com/anair13/mj_envs

Data available for download:
https://drive.google.com/file/d/1SsVaQKZnY5UkuR78WrInp9XxTdKHbF0x/view
"""
import os
import sys

from examples.awac.hand.awac1 import main

from rlkit.core import logger
from rlkit.testing import csv_util

def test_awac_hand_online():
    # the following hack required because of a conflict between env naming in d4rl and mj_envs
    import gym
    custom_envs = ['door-v0', 'pen-v0', 'relocate-v0', 'hammer-v0']
    env_dict = gym.envs.registration.registry.env_specs.copy()
    for custom_env in custom_envs:
        if custom_env in env_dict:
            print("Remove {} from registry".format(custom_env))
            del gym.envs.registration.registry.env_specs[custom_env]

    cmd = "python examples/awac/hand/awac1.py --1 --local --gpu --run 0 --seed 0 --debug"
    sys.argv = cmd.split(" ")[1:]
    main()

    # check if online training results matches
    reference_csv = "tests/regression/awac/hand/id0/pretrain_q.csv"
    output_csv = os.path.join(logger.get_snapshot_dir(), "pretrain_q.csv")
    output = csv_util.get_exp(output_csv)
    reference = csv_util.get_exp(reference_csv)
    keys = ["trainer/batch", "trainer/Advantage Score Max", "trainer/Q1 Predictions Mean", "trainer/replay_buffer_len"]
    csv_util.check_equal(reference, output, keys)

    # TODO: this test seems to have some extra stochasticity to control, perhaps from the env?
    # check if online training results match
    # reference_csv = "tests/regression/awac/hand/id0/progress.csv"
    # output_csv = os.path.join(logger.get_snapshot_dir(), "progress.csv")
    # output = csv_util.get_exp(output_csv)
    # reference = csv_util.get_exp(reference_csv)
    # keys = ["epoch", "eval/Actions Mean", "expl/Actions Mean", "eval/Average Returns", "expl/Average Returns", "trainer/Advantage Score Max", "trainer/Q1 Predictions Mean", "trainer/replay_buffer_len"]
    # csv_util.check_equal(reference, output, keys)

if __name__ == "__main__":
    test_awac_hand_online()
