import abc
import pickle
import time

import gtimer as gt
import numpy as np

from rlkit.core import logger
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.policies.base import ExplorationPolicy
from rlkit.samplers.in_place import InPlacePathSampler


class MetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            policy,
            train_tasks,
            eval_tasks,
            meta_batch=64,
            num_epochs=100,
            num_steps_per_epoch=10000,
            num_eval_tasks=1,
            num_steps_per_eval=1000,
            batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False, # TODO was true
            eval_sampler=None,
            replay_buffer=None,
    ):
        """
        Base class for Meta RL Algorithms
        :param env: training env
        :param policy: policy
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval
        :param meta_batch: number of tasks used for meta-update
        :param num_epochs: number of meta-training epochs
        :param num_steps_per_epoch: number of updates per epoch
        :param num_eval_tasks: number of tasks to eval on
        :param num_steps_per_eval:
        :param batch_size:
        :param max_path_length:
        :param discount:
        :param replay_buffer_size:
        :param reward_scale:
        :param render:
        :param save_replay_buffer:
        :param save_algorithm:
        :param save_environment:
        :param eval_sampler:
        :param replay_buffer:
        """
        self.env = env
        self.policy = policy
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.exploration_policy = self.make_exploration_policy(policy)
        self.meta_batch = meta_batch
        self.num_epochs = num_epochs
        self.num_env_steps_per_epoch = num_steps_per_epoch
        self.num_eval_tasks = num_eval_tasks
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment
        if eval_sampler is None:
            eval_policy = self.make_eval_policy(policy)
            eval_exploration_policy = self.make_eval_policy(self.make_exploration_policy(policy))
            eval_sampler = InPlacePathSampler(
                env=env,
                policy=eval_policy,
                exploration_policy=eval_exploration_policy,
                max_samples=self.num_steps_per_eval + self.max_path_length,
                max_path_length=self.max_path_length,
            )
        self.eval_policy = eval_policy
        self.eval_sampler = eval_sampler

        if replay_buffer is None:
            self.replay_buffer = MultiTaskReplayBuffer(
                    self.replay_buffer_size,
                    env,
                    self.train_tasks,
                )

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []

    def make_exploration_policy(self, policy):
        return policy

    def make_eval_policy(self, policy):
        return policy

    def sample_task(self, eval=False):
        '''
        sample task randomly
        '''
        if eval:
            idx = np.random.randint(len(self.eval_tasks))
        else:
            idx = np.random.randint(len(self.train_tasks))
        return idx

    def train(self):
        '''
        meta-training loop
        '''
        self.pretrain()
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()
        self.train_obs = self._start_new_rollout()
        for epoch in gt.timed_for(
                range(self.num_epochs),
                save_itrs=True,
        ):
            self._n_env_steps_total = epoch * self.num_env_steps_per_epoch
            # train
            self._start_epoch(epoch)
            self.training_mode(True)
            for i in range(self.num_env_steps_per_epoch):
                self.collect_batch_updates()
                self.perform_meta_update()
            self.training_mode(False)

            # eval
            self._try_to_eval(epoch)
            gt.stamp('eval')

            self._end_epoch()

    def collect_batch_updates(self):
        '''
        for each task in the meta batch
         - collect data from exploration policy
         - sample data from replay buffer, compute grads
         - collect data with current policy
        '''
        for _ in range(self.meta_batch):
            # sample a task and set env accordingly
            self.task_idx = self.sample_task()
            self.train_obs[-1] = self.task_idx
            self.env.reset_task(self.train_tasks[self.task_idx])
            self.collect_data(self.exploration_policy, num_samples=1)
            if self._can_train():
                self._do_training()
                self._n_train_steps_total += 1
                gt.stamp('train')
                self.collect_data(self.policy, num_samples=1)

    def perform_meta_update(self):
        '''
        update networks with grads accumulated across meta batch
        '''
        pass

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass

    def collect_data(self, agent, num_samples=1):
        '''
        collect data from current env in batch mode
        with exploration policy
        '''
        for _ in range(num_samples):
            action, agent_info = self._get_action_and_info(agent, self.train_obs)
            if self.render:
                self.env.render()
            next_ob, raw_reward, terminal, env_info = (
                self.env.step(action)
            )
            self._n_env_steps_total += 1
            reward = raw_reward * self.reward_scale
            terminal = np.array([terminal])
            reward = np.array([reward])
            self._handle_step(
                self.task_idx,
                self.train_obs,
                action,
                reward,
                next_ob,
                terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
            if terminal or len(self._current_path_builder) >= self.max_path_length:
                self._handle_rollout_ending()
                self.train_obs = self._start_new_rollout()
            else:
                self.train_obs = next_ob

            gt.stamp('sample')

    def _try_to_eval(self, epoch):
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        return (
            len(self._exploration_paths) > 0
            and self.replay_buffer.num_steps_can_sample(self.task_idx) >= self.batch_size
        )

    def _can_train(self):
        return self.replay_buffer.num_steps_can_sample(self.task_idx) >= self.batch_size

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        # TODO: do all pi have this?
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation,)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        return self.env.reset()

    def _handle_path(self, path):
        """
        Naive implementation: just loop through each transition.
        :param path:
        :return:
        """
        for (
            ob,
            action,
            reward,
            next_ob,
            terminal,
            agent_info,
            env_info
        ) in zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        ):
            self._handle_step(
                ob,
                action,
                reward,
                next_ob,
                terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self._handle_rollout_ending()

    def _handle_step(
            self,
            task_idx,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        """
        Implement anything that needs to happen after every step
        :return:
        """
        self._current_path_builder.add_all(
            task=task_idx,
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )
        self.replay_buffer.add_sample(
            task=task_idx,
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            agent_info=agent_info,
            env_info=env_info,
        )

    def _handle_rollout_ending(self):
        """
        Implement anything that needs to happen after every rollout.
        """
        self.replay_buffer.terminate_episode(self.task_idx)
        self._n_rollouts_total += 1
        if len(self._current_path_builder) > 0:
            self._exploration_paths.append(
                self._current_path_builder.get_all_stacked()
            )
            self._current_path_builder = PathBuilder()

    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def cuda(self):
        """
        Turn cuda on.
        :return:
        """
        pass

    @abc.abstractmethod
    def evaluate(self, epoch):
        """
        Evaluate the policy, e.g. save/print progress.
        :param epoch:
        :return:
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass
