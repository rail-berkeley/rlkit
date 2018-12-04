"""
Example of running stuff on GCP
"""
import time

from rlkit.core import logger
from rlkit.launchers.launcher_util import run_experiment
from datetime import datetime
from pytz import timezone
import pytz


def example(variant):
    import torch
    import rlkit.torch.pytorch_util as ptu
    print("Starting")
    logger.log(torch.__version__)
    date_format = '%m/%d/%Y %H:%M:%S %Z'
    date = datetime.now(tz=pytz.utc)
    logger.log("start")
    logger.log('Current date & time is: {}'.format(date.strftime(date_format)))
    logger.log("Cuda available: {}".format(torch.cuda.is_available()))
    if torch.cuda.is_available():
        x = torch.randn(3)
        logger.log(str(x.to(ptu.device)))

    date = date.astimezone(timezone('US/Pacific'))
    logger.log('Local date & time is: {}'.format(date.strftime(date_format)))
    for i in range(variant['num_seconds']):
        logger.log("Tick, {}".format(i))
        time.sleep(1)
    logger.log("end")
    logger.log('Local date & time is: {}'.format(date.strftime(date_format)))

    logger.log("start mujoco")
    from gym.envs.mujoco import HalfCheetahEnv
    e = HalfCheetahEnv()
    img = e.sim.render(32, 32)
    logger.log(str(sum(img)))
    logger.log("end mujoco")

    logger.record_tabular('Epoch', 1)
    logger.dump_tabular()
    logger.record_tabular('Epoch', 2)
    logger.dump_tabular()
    logger.record_tabular('Epoch', 3)
    logger.dump_tabular()
    print("Done")


if __name__ == "__main__":
    # noinspection PyTypeChecker
    date_format = '%m/%d/%Y %H:%M:%S %Z'
    date = datetime.now(tz=pytz.utc)
    logger.log("start")
    variant = dict(
        num_seconds=10,
        launch_time=str(date.strftime(date_format)),
    )
    run_experiment(
        example,
        exp_prefix="gcp-test",
        mode='gcp',
        variant=variant,
        # use_gpu=True,  # GPUs are much more expensive!
    )
