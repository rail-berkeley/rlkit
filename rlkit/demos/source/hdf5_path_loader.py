from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

from rlkit.util.io import load_local_or_remote_file

import random
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.data_management.path_builder import PathBuilder

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from rlkit.core import logger

import glob

def load_hdf5(dataset, replay_buffer):
    _obs = dataset['observations']
    N = _obs.shape[0]
    assert replay_buffer._max_replay_buffer_size >= N, "dataset does not fit in replay buffer"

    _actions = dataset['actions']
    _next_obs = dataset['next_observations']
    _rew = dataset['rewards'][:N]
    _done = dataset['terminals'][:N]

    replay_buffer._observations[:N] = _obs[:N]
    replay_buffer._next_obs[:N] = _next_obs[:N]
    replay_buffer._actions[:N] = _actions[:N]
    replay_buffer._rewards[:N] = np.expand_dims(_rew, 1)[:N]
    replay_buffer._terminals[:N] = np.expand_dims(_done, 1)[:N]
    replay_buffer._size = N-1
    replay_buffer._top = replay_buffer._size

class HDF5PathLoader:
    """
    Path loader for that loads obs-dict demonstrations
    into a Trainer with EnvReplayBuffer
    """

    def __init__(
            self,
            trainer,
            replay_buffer,
            demo_train_buffer,
            demo_test_buffer,
            demo_paths=[], # list of dicts
            demo_train_split=0.9,
            demo_data_split=1,
            add_demos_to_replay_buffer=True,
            bc_num_pretrain_steps=0,
            bc_batch_size=64,
            bc_weight=1.0,
            rl_weight=1.0,
            q_num_pretrain_steps=0,
            weight_decay=0,
            eval_policy=None,
            recompute_reward=False,
            env_info_key=None,
            obs_key=None,
            load_terminals=True,

            **kwargs
    ):
        self.trainer = trainer

        self.add_demos_to_replay_buffer = add_demos_to_replay_buffer
        self.demo_train_split = demo_train_split
        self.demo_data_split = demo_data_split
        self.replay_buffer = replay_buffer
        self.demo_train_buffer = demo_train_buffer
        self.demo_test_buffer = demo_test_buffer

        self.demo_paths = demo_paths

        self.bc_num_pretrain_steps = bc_num_pretrain_steps
        self.q_num_pretrain_steps = q_num_pretrain_steps
        self.demo_trajectory_rewards = []

        self.env_info_key = env_info_key
        self.obs_key = obs_key
        self.recompute_reward = recompute_reward
        self.load_terminals = load_terminals

        self.trainer.replay_buffer = self.replay_buffer
        self.trainer.demo_train_buffer = self.demo_train_buffer
        self.trainer.demo_test_buffer = self.demo_test_buffer

    def load_demos(self, dataset):
        # Off policy
        load_hdf5(dataset, self.replay_buffer)

    def get_batch_from_buffer(self, replay_buffer):
        batch = replay_buffer.random_batch(self.bc_batch_size)
        batch = np_to_pytorch_batch(batch)
        return batch
