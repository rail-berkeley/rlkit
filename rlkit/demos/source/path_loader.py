import glob
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.data_management.path_builder import PathBuilder
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.util.io import load_local_or_remote_file

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt


class PathLoader:
    """
    Loads demonstrations and/or off-policy data into a Trainer
    """

    def load_demos(
        self,
    ):
        pass
