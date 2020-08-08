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

class DictToMDPStackedPathLoader:
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
            stack_obs=1,

            **kwargs
    ):
        self.trainer = trainer

        self.add_demos_to_replay_buffer = add_demos_to_replay_buffer
        self.demo_train_split = demo_train_split
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
        self.stack_obs = stack_obs

        self.trainer.replay_buffer = self.replay_buffer
        self.trainer.demo_train_buffer = self.demo_train_buffer
        self.trainer.demo_test_buffer = self.demo_test_buffer

    def load_path(self, path, replay_buffer, obs_dict=None):
        rewards = []
        path_builder = PathBuilder()

        print("loading path, length", len(path["observations"]), len(path["actions"]))
        H = min(len(path["observations"]), len(path["actions"]))
        print("actions", np.min(path["actions"]), np.max(path["actions"]))

        for i in range(H):
            if obs_dict:
                ob = path["observations"][i][self.obs_key]
                next_ob = path["next_observations"][i][self.obs_key]
            else:
                ob = path["observations"][i]
                next_ob = path["next_observations"][i]

            if i == 0:
                current_obs = np.zeros((self.stack_obs + 1, len(ob)))
                current_obs[-2, :] = ob
                current_obs[-1, :] = next_ob
            else:
                current_obs = np.vstack((
                    current_obs[1:, :],
                    next_ob
                ))
                assert (current_obs[-2, :] == ob).all(), "mismatch between obs and next_obs"
            obs1 = current_obs[:self.stack_obs, :].flatten()
            obs2 = current_obs[1:, :].flatten()

            action = path["actions"][i]
            reward = path["rewards"][i]
            terminal = path["terminals"][i]
            if not self.load_terminals:
                terminal = np.zeros(terminal.shape)
            agent_info = path["agent_infos"][i]
            env_info = path["env_infos"][i]

            if self.recompute_reward:
                reward = self.env.compute_reward(
                    action,
                    next_ob,
                )

            reward = np.array([reward])
            rewards.append(reward)
            terminal = np.array([terminal]).reshape((1, ))
            path_builder.add_all(
                observations=obs1,
                actions=action,
                rewards=reward,
                next_observations=obs2,
                terminals=terminal,
                agent_infos=agent_info,
                env_infos=env_info,
            )
        self.demo_trajectory_rewards.append(rewards)
        path = path_builder.get_all_stacked()
        replay_buffer.add_path(path)
        print("path sum rewards", sum(rewards), len(rewards))
        # self.env.initialize(zs)

    def load_demos(self, ):
        # Off policy
        for demo_path in self.demo_paths:
            self.load_demo_path(**demo_path)

    # Parameterize which demo is being tested (and all jitter variants)
    # If is_demo is False, we only add the demos to the
    # replay buffer, and not to the demo_test or demo_train buffers
    def load_demo_path(self, path, is_demo, obs_dict, train_split=None):
        print("loading off-policy path", path)
        data = list(load_local_or_remote_file(path))
        # if not is_demo:
            # data = [data]
        # random.shuffle(data)

        if train_split is None:
            train_split = self.demo_train_split

        N = int(len(data) * train_split)
        print("using", N, "paths for training")

        if self.add_demos_to_replay_buffer:
            for path in data[:N]:
                self.load_path(path, self.replay_buffer, obs_dict=obs_dict)

        if is_demo:
            for path in data[:N]:
                self.load_path(path, self.demo_train_buffer, obs_dict=obs_dict)
            for path in data[N:]:
                self.load_path(path, self.demo_test_buffer, obs_dict=obs_dict)

    def get_batch_from_buffer(self, replay_buffer):
        batch = replay_buffer.random_batch(self.bc_batch_size)
        batch = np_to_pytorch_batch(batch)
        # obs = batch['observations']
        # next_obs = batch['next_observations']
        # goals = batch['resampled_goals']
        # import ipdb; ipdb.set_trace()
        # batch['observations'] = torch.cat((
        #     obs,
        #     goals
        # ), dim=1)
        # batch['next_observations'] = torch.cat((
        #     next_obs,
        #     goals
        # ), dim=1)
        return batch

