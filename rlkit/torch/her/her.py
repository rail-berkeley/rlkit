import numpy as np
import torch
from rlkit.data_management.path_builder import PathBuilder
import rlkit.samplers.rollout_functions as rf
from rlkit.torch.ddpg.ddpg import DDPG
from rlkit.torch.her.her_replay_buffer import RelabelingReplayBuffer
from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.sac.twin_sac import TwinSAC
from rlkit.torch.td3.td3 import TD3
from rlkit.torch.torch_rl_algorithm import TorchRLAlgorithm


class HER(TorchRLAlgorithm):
    """
    Note: this assumes the env will sample the goal when reset() is called,
    i.e. use a "silent" env.

    Hindsight Experience Replay

    This is a template class that should be the first sub-class, i.e.[

    ```
    class HerDdpg(HER, DDPG):
    ```

    and not

    ```
    class HerDdpg(DDPG, HER):
    ```

    Or if you really want to make DDPG the first subclass, do alternatively:
    ```
    class HerDdpg(DDPG, HER):
        def get_batch(self):
            return HER.get_batch(self)
    ```
    for each function defined below.
    """

    def __init__(
            self,
            observation_key='observation',
            desired_goal_key='desired_goal',
    ):
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key
        self.eval_rollout_function = rf.create_rollout_function(
            rf.multitask_rollout,
            observation_key=self.observation_key,
            desired_goal_key=self.desired_goal_key
        )

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        # Note: we assume we're using a silent env.
        o = self.training_env.reset()
        self._rollout_goal = self.training_env.get_goal()
        return o

    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
            goals=self._rollout_goal,
        )

    def _handle_path(self, path):
        self._n_rollouts_total += 1
        self.replay_buffer.add_path(path)
        self._exploration_paths.append(path)

    def get_batch(self):
        batch = super().get_batch()
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

    def _handle_rollout_ending(self):
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            path = self._current_path_builder.get_all_stacked()
            self.replay_buffer.add_path(path)
            self._exploration_paths.append(path)
            self._current_path_builder = PathBuilder()

    def _get_action_and_info(self, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        self.exploration_policy.set_num_steps_total(self._n_env_steps_total)
        goal = self._rollout_goal
        if self.observation_key:
            observation = observation[self.observation_key]
        if self.desired_goal_key:
            goal = self._rollout_goal[self.desired_goal_key]
        new_obs = np.hstack((observation, goal))
        return self.exploration_policy.get_action(new_obs)

    def get_eval_paths(self):
        paths = []
        n_steps_total = 0
        while n_steps_total <= self.num_steps_per_eval:
            path = self.eval_multitask_rollout()
            paths.append(path)
            n_steps_total += len(path['observations'])
        return paths

    def eval_multitask_rollout(self):
        return self.eval_rollout_function(
            self.env,
            self.policy,
            self.max_path_length,
            animated=self.render_during_eval
        )


class HerTd3(HER, TD3):
    def __init__(
            self,
            *args,
            td3_kwargs,
            her_kwargs,
            base_kwargs,
            **kwargs
    ):
        HER.__init__(
            self,
            **her_kwargs,
        )
        TD3.__init__(self, *args, **kwargs, **td3_kwargs, **base_kwargs)
        assert isinstance(
            self.replay_buffer, RelabelingReplayBuffer
        )


class HerSac(HER, SoftActorCritic):
    def __init__(
            self,
            *args,
            observation_key=None,
            desired_goal_key=None,
            **kwargs
    ):
        HER.__init__(
            self,
            observation_key=observation_key,
            desired_goal_key=desired_goal_key,
        )
        SoftActorCritic.__init__(self, *args, **kwargs)
        assert isinstance(
            self.replay_buffer, RelabelingReplayBuffer
        )

    def get_eval_action(self, observation, goal):
        if self.observation_key:
            observation = observation[self.observation_key]
        if self.desired_goal_key:
            goal = goal[self.desired_goal_key]
        new_obs = np.hstack((observation, goal))
        return self.policy.get_action(new_obs, deterministic=True)


class HerDdpg(HER, DDPG):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert isinstance(
            self.replay_buffer, RelabelingReplayBuffer
        )


class HerTwinSAC(HER, TwinSAC):
    def __init__(
            self,
            *args,
            twin_sac_kwargs,
            her_kwargs,
            base_kwargs,
            **kwargs
    ):
        HER.__init__(
            self,
            **her_kwargs,
        )
        TwinSAC.__init__(self, *args, **kwargs, **twin_sac_kwargs, **base_kwargs)
        assert isinstance(
            self.replay_buffer, RelabelingReplayBuffer
        )

    def get_eval_action(self, observation, goal):
        if self.observation_key:
            observation = observation[self.observation_key]
        if self.desired_goal_key:
            goal = goal[self.desired_goal_key]
        new_obs = np.hstack((observation, goal))
        return self.policy.get_action(new_obs, deterministic=True)
