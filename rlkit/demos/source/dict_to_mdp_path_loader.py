from collections import OrderedDict
import pickle
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import torch.nn.functional as F
import copy
import rlkit.torch.pytorch_util as ptu
from rlkit.misc.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer

from rlkit.misc.asset_loader import (
    load_local_or_remote_file, sync_down_folder, get_absolute_path, sync_down
)

import random
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.data_management.path_builder import PathBuilder

from rlkit.launchers.config import LOCAL_LOG_DIR, AWS_S3_PATH

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

from rlkit.core import logger

import glob

def load_encoder(encoder_file):
    encoder = load_local_or_remote_file(encoder_file)
    # TEMP #
    #encoder.representation_size = encoder.discrete_size * encoder.embedding_dim
    # TEMP #
    return encoder


class DictToMDPPathLoader:
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

            reward = np.array([reward]).flatten()
            rewards.append(reward)
            terminal = np.array([terminal]).reshape((1, ))
            path_builder.add_all(
                observations=ob,
                actions=action,
                rewards=reward,
                next_observations=next_ob,
                terminals=terminal,
                agent_infos=agent_info,
                env_infos=env_info,
            )
        self.demo_trajectory_rewards.append(rewards)
        path = path_builder.get_all_stacked()
        replay_buffer.add_path(path)
        print("path sum rewards", sum(rewards), len(rewards))

    def load_demos(self):
        # Off policy
        for demo_path in self.demo_paths:
            self.load_demo_path(**demo_path)

    # Parameterize which demo is being tested (and all jitter variants)
    # If is_demo is False, we only add the demos to the
    # replay buffer, and not to the demo_test or demo_train buffers
    def load_demo_path(self, path, is_demo, obs_dict, train_split=None, data_split=None, sync_dir=None):
        print("loading off-policy path", path)

        if sync_dir is not None:
            sync_down_folder(sync_dir)
            paths = glob.glob(get_absolute_path(path))
        else:
            paths = [path]

        data = []

        for filename in paths:
            data.extend(list(load_local_or_remote_file(filename)))

        # if not is_demo:
            # data = [data]
        # random.shuffle(data)

        if train_split is None:
            train_split = self.demo_train_split

        if data_split is None:
            data_split = self.demo_data_split

        M = int(len(data) * train_split * data_split)
        N = int(len(data) * data_split)
        print("using", N, "paths for training")

        if self.add_demos_to_replay_buffer:
            for path in data[:M]:
                self.load_path(path, self.replay_buffer, obs_dict=obs_dict)

        if is_demo:
            for path in data[:M]:
                self.load_path(path, self.demo_train_buffer, obs_dict=obs_dict)
            for path in data[M:N]:
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

class EncoderDictToMDPPathLoader(DictToMDPPathLoader):

    def __init__(
            self,
            trainer,
            replay_buffer,
            demo_train_buffer,
            demo_test_buffer,
            model_path=None,
            env=None,
            demo_paths=[], # list of dicts
            normalize=False,
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
            do_preprocess=True,
            **kwargs
    ):
        super().__init__(trainer,
            replay_buffer,
            demo_train_buffer,
            demo_test_buffer,
            demo_paths,
            demo_train_split,
            demo_data_split,
            add_demos_to_replay_buffer,
            bc_num_pretrain_steps,
            bc_batch_size,
            bc_weight,
            rl_weight,
            q_num_pretrain_steps,
            weight_decay,
            eval_policy,
            recompute_reward,
            env_info_key,
            obs_key,
            load_terminals,
            **kwargs)
        self.model = load_encoder(model_path)
        self.normalize = normalize
        self.env = env
        self.do_preprocess = do_preprocess

        print("ZEROING OUT GOALS")

    def resize_img(self, obs):
        from torchvision.transforms import Resize
        from PIL import Image
        resize = Resize((48, 48), interpolation=Image.NEAREST)

        obs = obs.reshape(84, 84, 3) * 255.0
        obs = Image.fromarray(obs, mode='RGB')
        obs = np.array(resize(obs))
        return obs.flatten() / 255.0

    def preprocess(self, observation):
        if not self.do_preprocess:
            for i in range(len(observation)):
                observation[i]["no_goal"] = np.zeros((0, ))
            return observation
        observation = copy.deepcopy(observation)
        images = np.stack([observation[i]['image_observation'] for i in range(len(observation))])
        goals = np.stack([np.zeros_like(observation[i]['image_observation']) for i in range(len(observation))])
        #images = np.stack([self.resize_img(observation[i]['image_observation']) for i in range(len(observation))])

        # latents = self.model.encode(ptu.from_numpy(images))
        # recon = ptu.get_numpy(self.model.decode(latents))

        # from torch.nn import functional as F

        # print(F.mse_loss(ptu.from_numpy(recon), ptu.from_numpy(images.reshape(50, 3, 48, 48))))
        # import ipdb; ipdb.set_trace()

        if self.normalize:
            images = images / 255.0

        latents = ptu.get_numpy(self.model.encode(ptu.from_numpy(images)))
        goals = ptu.get_numpy(self.model.encode(ptu.from_numpy(goals)))

        for i in range(len(observation)):
            observation[i]["latent_observation"] = latents[i]
            observation[i]["latent_achieved_goal"] = latents[i]
            observation[i]["latent_desired_goal"] = goals[-1]
            #observation[i]["latent_desired_goal"] = latents[-1]
            del observation[i]['image_observation']

        return observation

    def encode(self, obs):
        if self.normalize:
            return ptu.get_numpy(self.model.encode(ptu.from_numpy(obs) / 255.0))
        return ptu.get_numpy(self.model.encode(ptu.from_numpy(obs)))


    def load_path(self, path, replay_buffer, obs_dict=None):
        rewards = []
        path_builder = PathBuilder()
        H = min(len(path["observations"]), len(path["actions"]))

        if obs_dict:
            traj_obs = self.preprocess(path["observations"])
            next_traj_obs = self.preprocess(path["next_observations"])
        else:
            traj_obs = self.env.encode(path["observations"])
            next_traj_obs = self.env.encode(path["next_observations"])

        for i in range(H):
            ob = traj_obs[i]
            next_ob = next_traj_obs[i]
            action = path["actions"][i]

            # #temp fix#
            # ob['state_desired_goal'] = np.zeros_like(ob['state_desired_goal'])
            # ob['latent_desired_goal'] = np.zeros_like(ob['latent_desired_goal'])

            # next_ob['state_desired_goal'] = np.zeros_like(next_ob['state_desired_goal'])
            # next_ob['latent_desired_goal'] = np.zeros_like(next_ob['latent_desired_goal'])

            # action[3] /= 5
            # #temp fix#

            reward = path["rewards"][i]
            terminal = path["terminals"][i]
            if not self.load_terminals:
                terminal = np.zeros(terminal.shape)
            agent_info = path["agent_infos"][i]
            env_info = path["env_infos"][i]
            if self.recompute_reward:
                #reward = self.env.compute_rewards(action, path["next_observations"][i])
                reward = self.env._compute_reward(ob, action, next_ob, context=next_ob)

            reward = np.array([reward]).flatten()
            rewards.append(reward)
            terminal = np.array([terminal]).reshape((1, ))
            path_builder.add_all(
                observations=ob,
                actions=action,
                rewards=reward,
                next_observations=next_ob,
                terminals=terminal,
                agent_infos=agent_info,
                env_infos=env_info,
            )
        self.demo_trajectory_rewards.append(rewards)
        path = path_builder.get_all_stacked()
        replay_buffer.add_path(path)
        print("rewards", np.min(rewards), np.max(rewards))
        print("loading path, length", len(path["observations"]), len(path["actions"]))
        print("actions", np.min(path["actions"]), np.max(path["actions"]))
        print("path sum rewards", sum(rewards), len(rewards))



