import abc
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
            num_iterations=100,
            num_train_steps_per_itr=1000,
            num_tasks_sample=100,
            num_steps_per_task=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            train_embedding_source='posterior_only',
            eval_embedding_source='initial_pool',
            eval_deterministic=True,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
    ):
        """
        Base class for Meta RL Algorithms
        :param env: training env
        :param policy: policy that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval
        :param meta_batch: number of tasks used for meta-update
        :param num_iterations: number of meta-updates taken
        :param num_train_steps_per_itr: number of meta-updates performed per iteration
        :param num_tasks_sample: number of train tasks to sample to collect data for
        :param num_steps_per_task: number of transitions to collect per task
        :param num_evals: number of independent evaluation runs, with separate task encodings
        :param num_steps_per_eval: number of transitions to sample for evaluation
        :param batch_size: size of batches used to compute RL update
        :param embedding_batch_size: size of batches used to compute embedding
        :param embedding_mini_batch_size: size of batch used for encoder update
        :param max_path_length: max episode length
        :param discount:
        :param replay_buffer_size: max replay buffer size
        :param reward_scale:
        :param render:
        :param save_replay_buffer:
        :param save_algorithm:
        :param save_environment:
        """
        self.env = env
        self.policy = policy
        self.exploration_policy = policy # Can potentially use a different policy purely for exploration rather than also solving tasks, currently not being used
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_per_task = num_steps_per_task
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.train_embedding_source = train_embedding_source
        self.eval_embedding_source = eval_embedding_source # TODO: add options for computing embeddings on train tasks too
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment

        self.eval_sampler = InPlacePathSampler(
            env=env,
            policy=policy,
            max_samples=self.num_steps_per_eval,
            max_path_length=self.max_path_length,
        )

        # separate replay buffers for
        # - training RL update
        # - training encoder update
        # - testing encoder
        self.replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
            )

        self.enc_replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
        )
        self.eval_enc_replay_buffer = MultiTaskReplayBuffer(
            self.replay_buffer_size,
            env,
            self.eval_tasks
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

    def sample_task(self, is_eval=False):
        '''
        sample task randomly
        '''
        if is_eval:
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

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            self._start_epoch(it_)
            self.training_mode(True)
            if it_ == 0:
                print('collecting initial pool of data for train and eval')
                # temp for evaluating
                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.collect_data_sampling_from_prior(num_samples=self.max_path_length * 10,
                                                          resample_z_every_n=self.max_path_length,
                                                          eval_task=False)
                """
                for idx in self.eval_tasks:
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    # TODO: make number of initial trajectories a parameter
                    self.collect_data_sampling_from_prior(num_samples=self.max_path_length * 20,
                                                          resample_z_every_n=self.max_path_length,
                                                          eval_task=True)
                """

            # Sample data from train tasks.
            for i in range(self.num_tasks_sample):
                idx = np.random.randint(len(self.train_tasks))
                self.task_idx = idx
                self.env.reset_task(idx)

                # TODO: there may be more permutations of sampling/adding to encoding buffer we may wish to try
                if self.train_embedding_source == 'initial_pool':
                    # embeddings are computed using only the initial pool of data
                    # sample data from posterior to train RL algorithm
                    self.collect_data_from_task_posterior(idx=idx,
                                                          num_samples=self.num_steps_per_task,
                                                          add_to_enc_buffer=False)
                elif self.train_embedding_source == 'posterior_only':
                    self.collect_data_from_task_posterior(idx=idx, num_samples=self.num_steps_per_task, eval_task=False,
                                                          add_to_enc_buffer=True)
                elif self.train_embedding_source == 'online_exploration_trajectories':
                    # embeddings are computed using only data collected using the prior
                    # sample data from posterior to train RL algorithm
                    self.enc_replay_buffer.task_buffers[idx].clear()
                    # resamples using current policy, conditioned on prior
                    self.collect_data_sampling_from_prior(num_samples=self.num_steps_per_task,
                                                          resample_z_every_n=self.max_path_length,
                                                          add_to_enc_buffer=True)

                    self.collect_data_from_task_posterior(idx=idx,
                                                          num_samples=self.num_steps_per_task,
                                                          add_to_enc_buffer=False)
                elif self.train_embedding_source == 'online_on_policy_trajectories':
                    # sample from prior, then sample more from the posterior
                    # embeddings computed from both prior and posterior data
                    self.enc_replay_buffer.task_buffers[idx].clear()
                    self.collect_data_online(idx=idx,
                                             num_samples=self.num_steps_per_task,
                                             add_to_enc_buffer=True)
                else:
                    raise Exception("Invalid option for computing train embedding {}".format(self.train_embedding_source))

            # Sample train tasks and compute gradient updates on parameters.
            for train_step in range(self.num_train_steps_per_itr):
                indices = np.random.choice(self.train_tasks, self.meta_batch)
                self._do_training(indices)
                self._n_train_steps_total += 1
            gt.stamp('train')

            #self.training_mode(False)

            # eval
            self._try_to_eval(it_)
            gt.stamp('eval')

            self._end_epoch()

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        pass

    def sample_z_from_prior(self):
        """
        Samples z from the prior distribution, which can be either a delta function at 0 or a standard Gaussian
        depending on whether we use the information bottleneck.
        :return: latent z as a Numpy array
        """
        pass

    def sample_z_from_posterior(self, idx, eval_task):
        """
        Samples z from the posterior distribution given data from task idx, where data comes from the encoding buffer
        :param idx: task idx from which to compute the posterior from
        :param eval_task: whether or not the task is an eval task
        :return: latent z as a Numpy array
        """
        pass

    # TODO: maybe find a better name for resample_z_every_n?
    def collect_data_sampling_from_prior(self, num_samples=1, resample_z_every_n=None, eval_task=False,
                                         add_to_enc_buffer=True):
        # do not resample z if resample_z_every_n is None
        if resample_z_every_n is None:
            self.policy.clear_z()
            self.collect_data(self.policy, num_samples=num_samples, eval_task=eval_task,
                              add_to_enc_buffer=add_to_enc_buffer)
        else:
            # collects more data in batches of resample_z_every_n until done
            while num_samples > 0:
                self.collect_data_sampling_from_prior(num_samples=min(resample_z_every_n, num_samples),
                                                      resample_z_every_n=None,
                                                      eval_task=eval_task,
                                                      add_to_enc_buffer=add_to_enc_buffer)
                num_samples -= resample_z_every_n

    def collect_data_from_task_posterior(self, idx, num_samples=1, resample_z_every_n=None, eval_task=False,
                                         add_to_enc_buffer=True):
        # do not resample z if resample_z_every_n is None
        if resample_z_every_n is None:
            self.sample_z_from_posterior(idx, eval_task=eval_task)
            self.collect_data(self.policy, num_samples=num_samples, eval_task=eval_task,
                              add_to_enc_buffer=add_to_enc_buffer)
        else:
            # collects more data in batches of resample_z_every_n until done
            while num_samples > 0:
                self.collect_data_from_task_posterior(idx=idx,
                                                      num_samples=min(resample_z_every_n, num_samples),
                                                      resample_z_every_n=None,
                                                      eval_task=eval_task,
                                                      add_to_enc_buffer=add_to_enc_buffer)
                num_samples -= resample_z_every_n

    # split number of prior and posterior samples
    def collect_data_online(self, idx, num_samples, eval_task=False, add_to_enc_buffer=True):
        self.collect_data_sampling_from_prior(num_samples=num_samples,
                                              resample_z_every_n=self.max_path_length,
                                              eval_task=eval_task,
                                              add_to_enc_buffer=True)
        self.collect_data_from_task_posterior(idx=idx,
                                              num_samples=num_samples,
                                              resample_z_every_n=self.max_path_length,
                                              eval_task=eval_task,
                                              add_to_enc_buffer=add_to_enc_buffer)


    # TODO: since switching tasks now resets the environment, we are not correctly handling episodes terminating
    # correctly. We also aren't using the episodes anywhere, but we should probably change this to make it gather paths
    # until we have more samples than num_samples, to make sure every episode cleanly terminates when intended.
    def collect_data(self, agent, num_samples=1, eval_task=False, add_to_enc_buffer=True):
        '''
        collect data from current env in batch mode
        with given policy
        '''
        for _ in range(num_samples):
            action, agent_info = self._get_action_and_info(agent, self.train_obs)
            if self.render:
                self.env.render()
            next_ob, raw_reward, terminal, env_info = (
                self.env.step(action)
            )
            reward = raw_reward
            terminal = np.array([terminal])
            reward = np.array([reward])
            self._handle_step(
                self.task_idx,
                self.train_obs,
                action,
                reward,
                next_ob,
                terminal,
                eval_task=eval_task,
                add_to_enc_buffer=add_to_enc_buffer,
                agent_info=agent_info,
                env_info=env_info,
            )
            if terminal or len(self._current_path_builder) >= self.max_path_length:
                self._handle_rollout_ending(eval_task=eval_task)
                self.train_obs = self._start_new_rollout()
            else:
                self.train_obs = next_ob

        if not eval_task:
            self._n_env_steps_total += num_samples
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
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
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
        return self.env.reset()

    # not used
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
            eval_task=False,
            add_to_enc_buffer=True,
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
        if eval_task:
            self.eval_enc_replay_buffer.add_sample(
                task=task_idx,
                observation=observation,
                action=action,
                reward=reward,
                terminal=terminal,
                next_observation=next_observation,
                agent_info=agent_info,
                env_info=env_info,
            )
        else:
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
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_sample(
                    task=task_idx,
                    observation=observation,
                    action=action,
                    reward=reward,
                    terminal=terminal,
                    next_observation=next_observation,
                    agent_info=agent_info,
                    env_info=env_info,
                )

    def _handle_rollout_ending(self, eval_task=False):
        """
        Implement anything that needs to happen after every rollout.
        """
        if eval_task:
            self.eval_enc_replay_buffer.terminate_episode(self.task_idx)
        else:
            self.replay_buffer.terminate_episode(self.task_idx)
            self.enc_replay_buffer.terminate_episode(self.task_idx)

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
