from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.misc.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm
from rlkit.misc.asset_loader import load_local_or_remote_file

import abc
import copy
import pickle
import time
from collections import OrderedDict

import gtimer as gt
import numpy as np

from rlkit.core import logger
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.envs.remote import RemoteRolloutEnv
from rlkit.misc import eval_util
from rlkit.policies.base import ExplorationPolicy
from rlkit.samplers.in_place import InPlacePathSampler
import rlkit.envs.env_utils as env_utils

import random
from rlkit.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
from rlkit.torch.core import PyTorchModule, np_to_pytorch_batch

class BehaviorClone(TorchRLAlgorithm):
    """
    Behavior cloning implementation
    """

    def __init__(
            self,
            env,
            exploration_policy,
            policy,
            demo_path,
            test_replay_buffer,
            policy_learning_rate=1e-3,
            weight_decay=0,
            optimizer_class=optim.Adam,
            train_split=0.9,

            **kwargs
    ):
        super().__init__(
            env,
            exploration_policy=exploration_policy,
            eval_policy=policy,
            **kwargs
        )
        self.policy = policy
        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_learning_rate,
            weight_decay=weight_decay,
        )

        self.demo_path = demo_path
        self.train_split = train_split
        self.test_replay_buffer = test_replay_buffer
        self.load_demos(demo_path)

    def load_path(self, path, replay_buffer):
        path_builder = PathBuilder()
        for (
            ob,
            action,
            reward,
            next_ob,
            terminal,
            agent_info,
            env_info,
        ) in zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        ):
            # goal = path["goal"]["state_desired_goal"][0, :]
            # import pdb; pdb.set_trace()
            # print(goal.shape, ob["state_observation"])
            # state_observation = np.concatenate((ob["state_observation"], goal))
            action = action[:2]
            reward = np.array([reward])
            terminal = np.array([terminal])
            path_builder.add_all(
                observations=ob,
                actions=action,
                rewards=reward,
                next_observations=next_ob,
                terminals=terminal,
                agent_infos=agent_info,
                env_infos=env_info,
            )
        path = path_builder.get_all_stacked()
        replay_buffer.add_path(path)

    def load_demos(self, demo_path):
        data = load_local_or_remote_file(demo_path)
        random.shuffle(data)
        N = int(len(data) * self.train_split)
        print("using", N, "paths for training")
        for path in data[:N]:
            self.load_path(path, self.replay_buffer)

        for path in data[N:]:
            self.load_path(path, self.test_replay_buffer)

    def get_test_batch(self):
        batch = self.test_replay_buffer.random_batch(self.batch_size)
        batch = np_to_pytorch_batch(batch)
        obs = batch['observations']
        next_obs = batch['next_observations']
        goals = batch['resampled_goals']
        batch['observations'] = torch.cat((
            obs,
            goals
        ), dim=1)
        batch['next_observations'] = torch.cat((
            next_obs,
            goals
        ), dim=1)
        return batch

    def _do_training(self):
        batch = self.get_batch()
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        self._train_given_data(
            rewards,
            terminals,
            obs,
            actions,
            next_obs,
        )

    def _train_given_data(
        self,
        rewards,
        terminals,
        obs,
        actions,
        next_obs,
        logger_prefix="",
    ):
        """
        Critic operations.
        """

        predicted_actions = self.policy(obs)
        error = (predicted_actions - actions) ** 2
        bc_loss = error.mean()

        """
        Update Networks
        """
        self.policy_optimizer.zero_grad()
        bc_loss.backward()
        self.policy_optimizer.step()

        if self.need_to_update_eval_statistics:
            self.need_to_update_eval_statistics = False

            self.eval_statistics[logger_prefix + 'Policy Loss'] = np.mean(ptu.get_numpy(
                bc_loss
            ))

            test_batch = self.get_test_batch()
            test_o = test_batch["observations"]
            test_u = test_batch["actions"]
            test_pred_u = self.policy(test_o)
            test_error = (test_pred_u - test_u) ** 2
            test_bc_loss = test_error.mean()

            self.eval_statistics[logger_prefix + 'Test Policy Loss'] = np.mean(ptu.get_numpy(
                test_bc_loss
            ))

            self.eval_statistics[logger_prefix + 'Replay Buffer Size'] = self.replay_buffer._size

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        Modified here to always evaluate
        """
        return True

    def evaluate(self, epoch, eval_paths=None):
        statistics = OrderedDict()
        statistics.update(self.eval_statistics)

        logger.log("Collecting samples for evaluation")
        if eval_paths:
            test_paths = eval_paths
        else:
            test_paths = self.get_eval_paths()
        statistics.update(eval_util.get_generic_path_information(
            test_paths, stat_prefix="Test",
        ))
        # if len(self._exploration_paths) > 0:
        #     statistics.update(eval_util.get_generic_path_information(
        #         self._exploration_paths, stat_prefix="Exploration",
        #     ))
        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(test_paths, logger=logger)
        if hasattr(self.env, "get_diagnostics"):
            statistics.update(self.env.get_diagnostics(test_paths))

        average_returns = eval_util.get_average_returns(test_paths)
        statistics['AverageReturn'] = average_returns
        for key, value in statistics.items():
            logger.record_tabular(key, value)
        self.need_to_update_eval_statistics = True

    def get_epoch_snapshot(self, epoch):
        snapshot = super().get_epoch_snapshot(epoch)
        self.update_epoch_snapshot(snapshot)
        return snapshot

    def update_epoch_snapshot(self, snapshot):
        snapshot.update(
            policy=self.eval_policy,
            trained_policy=self.policy,
            exploration_policy=self.exploration_policy,
        )

    @property
    def networks(self):
        return [
            self.policy,
        ]
