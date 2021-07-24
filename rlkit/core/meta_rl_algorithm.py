import abc
from collections import OrderedDict
import time

import gtimer as gt
import numpy as np
import torch
from torch.distributions import Normal, kl_divergence

from rlkit.core import logger
from rlkit.core.logging import append_log
from rlkit.data_management.meta_learning_replay_buffer import \
    MetaLearningReplayBuffer
from rlkit.data_management.multitask_replay_buffer import MultiTaskReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.core import eval_util
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.smac.sampler import SMACInPlacePathSampler
import copy


class MetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            trainer,
            train_task_indices,
            eval_task_indices,
            train_tasks,
            eval_tasks,
            meta_batch=64,
            num_iterations=100,
            num_train_steps_per_itr=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            num_exp_traj_eval=1,
            update_post_train=1,
            eval_deterministic=True,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            plotter=None,
            sparse_rewards=False,
            use_next_obs_in_context=False,
            num_iterations_with_reward_supervision=np.inf,
            save_extra_manual_epoch_list=(),
            save_extra_manual_beginning_epoch_list=(),
            save_extra_every_epoch=False,
            use_ground_truth_context=False,
            exploration_resample_latent_period=0,
            exploration_update_posterior_period=0,
            exploration_data_collector=None,
            evaluation_data_collector=None,
            use_meta_learning_buffer=False,
            env_info_sizes=None,
            sample_buffer_in_proportion_to_size=False,
            num_tasks_to_eval_on=10,
            add_exploration_data_to='train_and_self_generated_tasks',
            condition_on_posterior_guided_data_when_exploring=False,
            train_agent_in_unsupervised_phase=True,
            logging_period=1,
            # encoder parameters
            use_encoder_snapshot_for_reward_pred_in_unsupervised_phase=False,
            train_encoder_decoder_in_unsupervised_phase=False,
            use_rl_buffer_for_enc_buffer=False,
            debug_enc_buffer_matches_rl_buffer=False,
            freeze_encoder_buffer_in_unsupervised_phase=True,
            clear_encoder_buffer_before_every_update=True,
            num_tasks_to_generate=0,
            num_initial_steps_self_generated_tasks=200,
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_task_indices: list of tasks used for training
        :param eval_task_indices: list of tasks used for eval

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.debug_enc_buffer_matches_rl_buffer = (
            debug_enc_buffer_matches_rl_buffer
        )
        self.logging_period = logging_period
        self.train_agent_in_unsupervised_phase = (
            train_agent_in_unsupervised_phase
        )
        self._condition_on_posterior_guided_data_when_exploring = (
            condition_on_posterior_guided_data_when_exploring
        )
        self.add_exploration_data_to = add_exploration_data_to
        self._num_tasks_to_eval_on = num_tasks_to_eval_on
        self.num_initial_steps_self_generated_tasks = num_initial_steps_self_generated_tasks
        self._num_tasks_to_generate = num_tasks_to_generate
        if add_exploration_data_to not in {
            'self_generated_tasks',
            'train_tasks',
            'train_and_self_generated_tasks',
            'none',
        }:
            raise ValueError(add_exploration_data_to)
        self.train_encoder_decoder_in_unsupervised_phase = train_encoder_decoder_in_unsupervised_phase
        self.use_rl_buffer_for_enc_buffer = use_rl_buffer_for_enc_buffer
        self.use_meta_learning_buffer = use_meta_learning_buffer
        self._save_extra_every_epoch = save_extra_every_epoch
        self.save_extra_manual_epoch_set = set(save_extra_manual_epoch_list)
        if len(self.save_extra_manual_epoch_set) > 0:
            # always add the last epoch in case user had an OBO error
            self.save_extra_manual_epoch_set.add(num_iterations - 1)

        self.save_extra_manual_beginning_epoch_list = save_extra_manual_beginning_epoch_list
        self.use_encoder_snapshot_for_reward_pred_in_unsupervised_phase = (
            use_encoder_snapshot_for_reward_pred_in_unsupervised_phase
        )
        self.env = env
        self.agent = agent
        self.trainer = trainer
        self.exploration_agent = agent # Can potentially use a different policy purely for exploration rather than also solving tasks, currently not being used
        self.train_task_indices = train_task_indices
        self.exploration_task_indices = train_task_indices
        self.offline_train_task_indices = train_task_indices
        self.eval_task_indices = eval_task_indices
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.update_post_train = update_post_train
        self.post_train_funcs = []
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment
        if num_iterations_with_reward_supervision is None:
            num_iterations_with_reward_supervision = np.inf
        self.num_iterations_with_reward_supervision = num_iterations_with_reward_supervision
        self.freeze_encoder_buffer_in_unsupervised_phase = (
            freeze_encoder_buffer_in_unsupervised_phase
        )
        self.clear_encoder_buffer_before_every_update = (
            clear_encoder_buffer_before_every_update
        )
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector

        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter

        self.exploration_resample_latent_period = exploration_resample_latent_period
        self.exploration_update_posterior_period = exploration_update_posterior_period
        self.sampler = SMACInPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
        )

        self.meta_replay_buffer = None
        self.replay_buffer = None
        self.enc_replay_buffer = None
        self.meta_replay_buffer = MetaLearningReplayBuffer(
            self.replay_buffer_size,
            env,
            self.train_task_indices,
            use_next_obs_in_context=use_next_obs_in_context,
            sparse_rewards=sparse_rewards,
            mini_buffer_max_size=self.max_path_length + max(
                self.num_steps_prior,
                self.num_steps_posterior,
                self.num_extra_rl_steps_posterior,
            ),
            sample_buffer_in_proportion_to_size=sample_buffer_in_proportion_to_size,
        )
        self.replay_buffer = MultiTaskReplayBuffer(
            self.replay_buffer_size,
            env,
            self.train_task_indices,
            use_next_obs_in_context=use_next_obs_in_context,
            sparse_rewards=sparse_rewards,
            env_info_sizes=env_info_sizes,
        )
        if self.use_rl_buffer_for_enc_buffer:
            self.enc_replay_buffer = self.replay_buffer
        else:
            self.enc_replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_task_indices,
                use_next_obs_in_context=use_next_obs_in_context,
                sparse_rewards=sparse_rewards,
                use_ground_truth_context=use_ground_truth_context,
                ground_truth_tasks=train_tasks,
                env_info_sizes=env_info_sizes,
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
        self.in_unsupervised_phase = False
        self._debug_use_ground_truth_context = use_ground_truth_context

        self._reward_decoder_buffer = self.enc_replay_buffer
        self.fake_task_idx_to_z = {}

    def train(self):
        '''
        meta-training loop
        '''
        start_time = time.time()
        print("starting to pretrain")
        self.pretrain()
        print("done pretraining after time:", time.time() - start_time)
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            self._start_epoch(it_)
            self.training_mode(True)
            if it_ == 0 and self.num_initial_steps > 0:
                print('collecting initial pool of data for train and eval')
                # temp for evaluating
                for task_idx in self.train_task_indices:
                    if self.expl_data_collector:
                        init_expl_paths = self.expl_data_collector.collect_new_paths(
                            max_path_length=self.max_path_length,
                            num_steps=self.num_initial_steps,
                            discard_incomplete_paths=False,
                            task_idx=task_idx,
                        )
                        self.replay_buffer.add_paths(task_idx, init_expl_paths)
                        if not self.use_rl_buffer_for_enc_buffer:
                            self.enc_replay_buffer.add_paths(task_idx, init_expl_paths)
                        self.expl_data_collector.end_epoch(-1)
                    else:
                        self.collect_exploration_data(
                            self.num_initial_steps, 1, np.inf, task_idx)
            self.in_unsupervised_phase = (it_ >= self.num_iterations_with_reward_supervision)
            if it_ == self.num_iterations_with_reward_supervision:
                self._transition_to_unsupervised()
            update_encoder_buffer = not (
                    self.in_unsupervised_phase
                    and self.freeze_encoder_buffer_in_unsupervised_phase
            ) and not self.use_rl_buffer_for_enc_buffer
            clear_encoder_buffer = (
                                           update_encoder_buffer
                                           and self.clear_encoder_buffer_before_every_update
                                   ) and not self.use_rl_buffer_for_enc_buffer
            # TODO: propogate unsupervised mode elegantly
            # Sample data from train tasks.
            for i in range(self.num_tasks_sample):
                if len(self.exploration_task_indices) == 0:
                    # do no data collection
                    break
                task_idx = np.random.choice(self.exploration_task_indices)
                if clear_encoder_buffer:
                    self.enc_replay_buffer.task_buffers[task_idx].clear()
                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    if self.expl_data_collector:
                        # TODO: implement
                        new_expl_paths = self.expl_data_collector.collect_new_paths(
                            task_idx=task_idx,
                            max_path_length=self.max_path_length,
                            resample_latent_period=self.exploration_resample_latent_period,
                            update_posterior_period=np.inf,
                            num_steps=self.num_steps_prior,
                            use_predicted_reward=self.in_unsupervised_phase,
                            discard_incomplete_paths=False,
                        )
                        self.replay_buffer.add_paths(task_idx, new_expl_paths)
                        self._n_env_steps_total += sum(
                            len(p['actions']) for p in new_expl_paths
                        )
                        self._n_rollouts_total += len(new_expl_paths)
                        if update_encoder_buffer:
                            self.enc_replay_buffer.add_paths(task_idx, new_expl_paths)
                    else:
                        self.collect_exploration_data(
                            num_samples=self.num_steps_prior,
                            resample_latent_period=self.exploration_resample_latent_period,
                            update_posterior_period=np.inf,
                            add_to_enc_buffer=update_encoder_buffer,
                            use_predicted_reward=self.in_unsupervised_phase,
                            task_idx=task_idx,
                            # TODO: figure out if I want to replace this?
                            # it's only used when `clear_encoder_buffer_before_every_update` is True
                            # and when `freeze_encoder_buffer_in_unsupervised_phase` is False
                            # and when we're in unsupervised phase
                        )
                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    if self.expl_data_collector:
                        # TODO: implement
                        new_expl_paths = self.expl_data_collector.collect_new_paths(
                            task_idx=task_idx,
                            max_path_length=self.max_path_length,
                            resample_latent_period=self.exploration_resample_latent_period,
                            update_posterior_period=self.update_post_train,
                            num_steps=self.num_steps_posterior,
                            use_predicted_reward=self.in_unsupervised_phase,
                            discard_incomplete_paths=False,
                        )
                        self.replay_buffer.add_paths(task_idx, new_expl_paths)
                        self._n_env_steps_total += sum(
                            len(p['actions']) for p in new_expl_paths
                        )
                        self._n_rollouts_total += len(new_expl_paths)
                        if update_encoder_buffer and not self.use_rl_buffer_for_enc_buffer:
                            self.enc_replay_buffer.add_paths(task_idx, new_expl_paths)
                    else:
                        self.collect_exploration_data(
                            num_samples=self.num_steps_posterior,
                            resample_latent_period=self.exploration_resample_latent_period,
                            update_posterior_period=self.update_post_train,
                            add_to_enc_buffer=update_encoder_buffer,
                            use_predicted_reward=self.in_unsupervised_phase,
                            task_idx=task_idx,
                        )
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    # TODO: implement
                    if self.expl_data_collector:
                        new_expl_paths = self.expl_data_collector.collect_new_paths(
                            task_idx=task_idx,
                            max_path_length=self.max_path_length,
                            resample_latent_period=self.exploration_resample_latent_period,
                            update_posterior_period=self.update_post_train,
                            num_steps=self.num_extra_rl_steps_posterior,
                            use_predicted_reward=self.in_unsupervised_phase,
                            discard_incomplete_paths=False,
                        )
                        self.replay_buffer.add_paths(task_idx, new_expl_paths)
                        self._n_env_steps_total += sum(
                            len(p['actions']) for p in new_expl_paths
                        )
                        self._n_rollouts_total += len(new_expl_paths)
                        if not self.use_rl_buffer_for_enc_buffer:
                            self.enc_replay_buffer.add_paths(task_idx, new_expl_paths)
                    else:
                        add_to_enc_buffer = (
                                self.debug_enc_buffer_matches_rl_buffer
                                and not self.use_rl_buffer_for_enc_buffer
                        )
                        self.collect_exploration_data(
                            num_samples=self.num_extra_rl_steps_posterior,
                            resample_latent_period=self.exploration_resample_latent_period,
                            update_posterior_period=self.update_post_train,
                            add_to_enc_buffer=add_to_enc_buffer,
                            use_predicted_reward=self.in_unsupervised_phase,
                            task_idx=task_idx,
                        )
            gt.stamp('sample')

            # Sample train tasks and compute gradient updates on parameters.
            for train_step in range(self.num_train_steps_per_itr):
                if self.use_meta_learning_buffer:
                    batch = self.meta_replay_buffer.sample_meta_batch(
                        rl_batch_size=self.batch_size,
                        meta_batch_size=self.meta_batch,
                        embedding_batch_size=self.embedding_batch_size,
                    )
                    self.trainer.train(batch)
                else:
                    indices = np.random.choice(self.train_task_indices, self.meta_batch)

                    mb_size = self.embedding_mini_batch_size
                    num_updates = self.embedding_batch_size // mb_size

                    # sample context batch
                    # context_batch = self.sample_context(indices)
                    context_batch = self.enc_replay_buffer.sample_context(
                        indices,
                        self.embedding_batch_size
                    )

                    # zero out context and hidden encoder state
                    # self.agent.clear_z(num_tasks=len(indices))

                    # do this in a loop so we can truncate backprop in the recurrent encoder
                    for i in range(num_updates):
                        if self._debug_use_ground_truth_context:
                            context = context_batch
                        else:
                            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
                        # batch = self.sample_batch(indices)
                        batch = self.replay_buffer.sample_batch(indices, self.batch_size)
                        batch['context'] = context
                        batch['task_indices'] = indices
                        self.trainer.train(batch)
                        self._n_train_steps_total += 1

                    # stop backprop
                    # self.agent.detach_z()
                # train_data = self.replay_buffer.random_batch(self.batch_size)
            gt.stamp('train')

            self.training_mode(False)

            # eval
            self._try_to_eval(it_)

            self._end_epoch(it_)

    def _transition_to_unsupervised(self):
        self._reward_decoder_buffer = copy.deepcopy(self.enc_replay_buffer)
        self.trainer.train_encoder_decoder = self.train_encoder_decoder_in_unsupervised_phase
        self.trainer.train_agent = self.train_agent_in_unsupervised_phase
        self.agent.use_context_encoder_snapshot_for_reward_pred = (
            self.use_encoder_snapshot_for_reward_pred_in_unsupervised_phase
        )

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        # HACK: I'm assuming the train and eval task indices are consecutive.
        num_existing_tasks = len(self.offline_train_task_indices) + len(self.eval_task_indices)
        fake_task_idxs = list(range(
            num_existing_tasks,
            num_existing_tasks + self._num_tasks_to_generate,
            ))
        if self.add_exploration_data_to == 'self_generated_tasks':
            self.exploration_task_indices = fake_task_idxs
        elif self.add_exploration_data_to == 'train_tasks':
            self.exploration_task_indices = self.offline_train_task_indices
        elif self.add_exploration_data_to == 'train_and_self_generated_tasks':
            self.exploration_task_indices = (
                    self.offline_train_task_indices + fake_task_idxs
            )
        elif self.add_exploration_data_to == 'none':
            self.exploration_task_indices = []
            self.num_tasks_sample = 0
        else:
            raise ValueError(self.add_exploration_data_to)
        self.fake_task_idx_to_z = {
            task_idx: ptu.get_numpy(self.agent.latent_prior.sample())
            for task_idx in fake_task_idxs
        }
        for task_idx in self.fake_task_idx_to_z:
            if not self.use_rl_buffer_for_enc_buffer:
                self.enc_replay_buffer.create_new_task_buffer(task_idx)
            self.replay_buffer.create_new_task_buffer(task_idx)
            self.collect_exploration_data(
                self.num_initial_steps_self_generated_tasks, 1, np.inf, task_idx,
            )

    def collect_exploration_data(self, num_samples,
                                 resample_latent_period, update_posterior_period, task_idx, add_to_enc_buffer=True, use_predicted_reward=False,
                                 ):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_latent_period: how often to resample latent context z (in units of trajectories)
        :param update_posterior_period: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        :param use_predicted_reward: whether to replace the env reward with the predicted reward to simulate not having access to rewards.
        '''
        # start from the prior
        self.agent.clear_z()

        num_transitions = 0
        init_context = None
        while num_transitions < num_samples:
            initialized_z_reward = None
            initial_reward_context = None
            if task_idx in self.fake_task_idx_to_z:
                initialized_z_reward = self.fake_task_idx_to_z[task_idx]
                use_predicted_reward = True
            else:
                if use_predicted_reward:
                    if self.use_meta_learning_buffer:
                        initial_reward_context = self.meta_replay_buffer.sample_context(
                            self.embedding_batch_size
                        )
                    else:
                        initial_reward_context = self._reward_decoder_buffer.sample_context(
                            task_idx,
                            self.embedding_batch_size
                        )
            # TODO: replace with sampler
            paths, n_samples = self.sampler.obtain_samples(
                max_samples=num_samples - num_transitions,
                max_trajs=update_posterior_period,
                accum_context=self._condition_on_posterior_guided_data_when_exploring,
                resample_latent_period=resample_latent_period,
                update_posterior_period=self.exploration_update_posterior_period,
                use_predicted_reward=use_predicted_reward,
                task_idx=task_idx,
                initial_context=init_context,
                initial_reward_context=initial_reward_context,
                initialized_z_reward=initialized_z_reward,
            )
            num_transitions += n_samples
            self._n_rollouts_total += len(paths)
            if self.use_meta_learning_buffer:
                self.meta_replay_buffer.add_paths(paths)
            else:
                self.replay_buffer.add_paths(task_idx, paths)
                if add_to_enc_buffer and not self.use_rl_buffer_for_enc_buffer:
                    self.enc_replay_buffer.add_paths(task_idx, paths)
                if update_posterior_period != np.inf:
                    # init_context = self.sample_context(task_idx)
                    if self._condition_on_posterior_guided_data_when_exploring:
                        init_context = paths[-1]['context']  # TODO clean hack
                    else:
                        # TODO: check if it matters which version I use: above code or below?
                        init_context = self.enc_replay_buffer.sample_context(
                            task_idx,
                            self.embedding_batch_size
                        )
                        init_context = ptu.from_numpy(init_context)
        self._n_env_steps_total += num_transitions

    def _try_to_eval(self, epoch):
        if epoch % self.logging_period != 0:
            return
        if epoch in self.save_extra_manual_epoch_set:
            logger.save_extra_data(
                self.get_extra_data_to_save(epoch),
                file_name='extra_snapshot_itr{}'.format(epoch),
                mode='cloudpickle',
            )
        if self._save_extra_every_epoch:
            logger.save_extra_data(self.get_extra_data_to_save(epoch))
        gt.stamp('save-extra')
        if self._can_evaluate():
            self.evaluate(epoch)
            gt.stamp('eval')

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            gt.stamp('save-snapshot')
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_dict(
                self.trainer.get_diagnostics(),
                prefix='trainer/',
            )

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
            save_extra_time = times_itrs['save-extra'][-1]
            save_snapshot_time = times_itrs['save-snapshot'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + save_extra_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('in_unsupervised_model',
                                  float(self.in_unsupervised_phase))
            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Save Extra Time (s)', save_extra_time)
            logger.record_tabular('Save Snapshot Time (s)', save_snapshot_time)
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
        # eval collects its own context, so can eval any time
        return True

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_task_indices])

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
        if epoch in self.save_extra_manual_beginning_epoch_list:
            logger.save_extra_data(
                self.get_extra_data_to_save(epoch),
                file_name='extra_snapshot_beginning_itr{}'.format(epoch),
                mode='cloudpickle',
            )

    def _end_epoch(self, epoch):
        for post_train_func in self.post_train_funcs:
            post_train_func(self, epoch)

        self.trainer.end_epoch(epoch)
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        snapshot = {'epoch': epoch}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        snapshot['env'] = self.env
        snapshot['env_sampler'] = self.sampler
        snapshot['agent'] = self.agent
        snapshot['exploration_agent'] = self.exploration_agent
        return snapshot

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
            data_to_save['enc_replay_buffer'] = self.enc_replay_buffer
            data_to_save['meta_replay_buffer'] = self.meta_replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def collect_paths(self, idx, epoch, run):
        self.agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0
        init_context = None
        infer_posterior_at_start = False
        while num_transitions < self.num_steps_per_eval:
            # We follow the PEARL protocol and never update the posterior or resample z within an episode during evaluation.
            if idx in self.fake_task_idx_to_z:
                initialized_z_reward = self.fake_task_idx_to_z[idx]
            else:
                initialized_z_reward = None
            loop_paths, num = self.sampler.obtain_samples(
                deterministic=self.eval_deterministic,
                max_samples=self.num_steps_per_eval - num_transitions,
                max_trajs=1,
                accum_context=True,
                initial_context=init_context,
                task_idx=idx,
                resample_latent_period=self.exploration_resample_latent_period,  # PEARL had this=0.
                update_posterior_period=0,  # following PEARL protocol
                infer_posterior_at_start=infer_posterior_at_start,
                initialized_z_reward=initialized_z_reward,
                use_predicted_reward=initialized_z_reward is not None,
            )
            paths += loop_paths
            num_transitions += num
            num_trajs += 1
            # accumulated contexts across rollouts
            init_context = paths[-1]['context']  # TODO clean hack
            if num_trajs >= self.num_exp_traj_eval:
                infer_posterior_at_start = True

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal  # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths and epoch >= 0:
            logger.save_extra_data(paths, file_name='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def _do_eval(self, indices, epoch):
        final_returns = []
        online_returns = []
        task_idx_to_final_context = {}
        for idx in indices:
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
                task_idx_to_final_context[idx] = paths[-1]['context']
            final_returns.append(np.mean([a[-1] for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns, task_idx_to_final_context

    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### sample trajectories from prior for debugging / visualization
        if self.dump_eval_paths:
            # 100 arbitrarily chosen for visualizations of point_robot trajectories
            # just want stochasticity of z, not the policy
            self.agent.clear_z()
            prior_paths, _ = self.sampler.obtain_samples(
                deterministic=self.eval_deterministic,
                max_samples=self.max_path_length * 20,
                accum_context=False,
                resample_latent_period=self.exploration_resample_latent_period,
                update_posterior_period=self.exploration_update_posterior_period,  # following PEARL protocol
            )
            logger.save_extra_data(prior_paths, file_name='eval_trajectories/prior-epoch{}'.format(epoch))
        ### train tasks
        if self._num_tasks_to_eval_on >= len(self.train_task_indices):
            indices = self.train_task_indices
        else:
            # eval on a subset of train tasks in case num train tasks is huge
            indices = np.random.choice(self.offline_train_task_indices, self._num_tasks_to_eval_on)
        # logger.log('evaluating on {} train tasks'.format(len(indices)))
        ### eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        for idx in indices:
            self.env.reset_task(idx)
            paths = []
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                # init_context = self.sample_context(idx)
                if self.use_meta_learning_buffer:
                    init_context = self.meta_replay_buffer._sample_contexts(
                        [idx],
                        self.embedding_batch_size
                    )
                else:
                    init_context = self.enc_replay_buffer.sample_context(
                        idx,
                        self.embedding_batch_size
                    )
                if self.eval_data_collector:
                    p = self.eval_data_collector.collect_new_paths(
                        num_steps=self.max_path_length,  # TODO: also cap num trajs
                        max_path_length=self.max_path_length,
                        discard_incomplete_paths=False,
                        accum_context=False,
                        resample_latent_period=0,
                        update_posterior_period=0,
                        initial_context=init_context,
                        task_idx=idx,
                    )
                else:
                    init_context = ptu.from_numpy(init_context)
                    # TODO: replace with sampler
                    # self.agent.infer_posterior(context)
                    p, _ = self.sampler.obtain_samples(
                        deterministic=self.eval_deterministic,
                        max_samples=self.max_path_length,
                        accum_context=False,
                        max_trajs=1,
                        resample_latent_period=0,
                        update_posterior_period=0,
                        initial_context=init_context,
                        task_idx=idx,
                    )
                paths += p

            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

            train_returns.append(eval_util.get_average_returns(paths))

        train_returns_offline_buffer = self._get_returns_init_from_offline_buffer(indices)
        # train_returns = np.mean(train_returns)
        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns, train_task_to_final_context = (
            self._do_eval(indices, epoch)
        )
        # logger.log('train online returns')
        # logger.log(train_online_returns)

        ### test tasks
        # logger.log('evaluating on {} test tasks'.format(len(self.eval_task_indices)))
        test_final_returns, test_online_returns, test_task_to_final_context =(
            self._do_eval(self.eval_task_indices, epoch)
        )
        # logger.log('test online returns')
        # logger.log(test_online_returns)
        # save the final posterior
        self.agent.log_diagnostics(self.eval_statistics)

        z_dist_log = self._get_z_distribution_log(train_task_to_final_context)
        append_log(self.eval_statistics, z_dist_log, prefix='trainer/train_tasks/')

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(paths, prefix=None)

        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.eval_statistics.update(eval_util.create_stats_ordered_dict(
            'eval/init_from_offline_buffer/train_tasks/all_returns',
            train_returns_offline_buffer,
        ))
        self.eval_statistics.update(eval_util.create_stats_ordered_dict(
            'eval/init_from_buffer/train_tasks/all_returns',
            train_returns,
        ))
        self.eval_statistics.update(eval_util.create_stats_ordered_dict(
            'eval/adaptation/train_tasks/final_returns',
            train_final_returns,
        ))
        self.eval_statistics.update(eval_util.create_stats_ordered_dict(
            'eval/adaptation/test_tasks/final_returns',
            test_final_returns,
        ))
        self.eval_statistics.update(eval_util.create_stats_ordered_dict(
            'eval/adaptation/train_tasks/all_returns',
            avg_train_online_return,
        ))
        self.eval_statistics.update(eval_util.create_stats_ordered_dict(
            'eval/adaptation/test_tasks/all_returns',
            avg_test_online_return,
        ))

        if len(self.fake_task_idx_to_z) > 0:
            self_generated_indices = np.random.choice(
                np.array(list(self.fake_task_idx_to_z.keys())),
                self._num_tasks_to_eval_on,
            )
            self_generated_final_returns, self_generated_online_returns, _ = self._do_eval(self_generated_indices, epoch)
            avg_self_generated_return = np.mean(np.stack(self_generated_online_returns))
            self.eval_statistics.update(eval_util.create_stats_ordered_dict(
                'eval/adaptation/generated_tasks/final_returns',
                self_generated_final_returns,
            ))
            self.eval_statistics.update(eval_util.create_stats_ordered_dict(
                'eval/adaptation/generated_tasks/all_returns',
                avg_self_generated_return,
            ))

        try:
            import os
            import psutil
            process = psutil.Process(os.getpid())
            self.eval_statistics['RAM Usage (Mb)'] = int(process.memory_info().rss / 1000000)
        except ImportError:
            pass
        logger.save_extra_data(avg_train_online_return, file_name='online-train-epoch{}'.format(epoch))
        logger.save_extra_data(avg_test_online_return, file_name='online-test-epoch{}'.format(epoch))

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

    # @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    def to(self, device=None):
        self.trainer.to(device=device)

    def _get_z_distribution_log(self, idx_to_final_context):
        """Log diagnostics about the shift in z-distribution"""
        logs = OrderedDict()
        task_indices = list(sorted(idx_to_final_context.keys()))
        offline_context1 = self._reward_decoder_buffer.sample_context(
            task_indices,
            self.embedding_batch_size
        )
        context_distrib1 = self.agent.latent_posterior(offline_context1)
        offline_context2 = self._reward_decoder_buffer.sample_context(
            task_indices,
            self.embedding_batch_size
        )
        context_distrib2 = self.agent.latent_posterior(offline_context2)

        context_distribs1 = [
            Normal(m, s) for m, s in
            zip(context_distrib1.mean, context_distrib1.stddev)
        ]
        context_distribs2 = [
            Normal(m, s) for m, s in
            zip(context_distrib2.mean, context_distrib2.stddev)
        ]
        within_task_kl_2samples, between_task_kl_2samples = (
            self._compute_within_and_between_task_kl(
                context_distribs1,
                context_distribs2,
            )
        )

        logs['two_offline_z_posteriors/within_task_kl'] = (
            within_task_kl_2samples
        )
        logs['two_offline_z_posteriors/between_task_kl'] = (
            between_task_kl_2samples
        )
        offline_z_posterior_kl_prior = kl_divergence(
            context_distrib1, self.agent.latent_prior).sum(dim=1)
        logs.update(eval_util.create_stats_ordered_dict(
            'offline_z_posterior/kl_prior',
            ptu.get_numpy(offline_z_posterior_kl_prior),
        ))

        context_distribs_online = []
        for idx in task_indices:
            context = idx_to_final_context[idx]
            context_distrib = self.agent.latent_posterior(context)
            context_distribs_online.append(context_distrib)

        within_task_kl_off_on, between_task_kl_off_on = (
            self._compute_within_and_between_task_kl(
                context_distribs_online,
                context_distribs2,
            )
        )
        online_posterior_kls = np.array([
            ptu.get_numpy(kl_divergence(q, self.agent.latent_prior).sum())
            for q in context_distribs_online
        ])
        logs.update(eval_util.create_stats_ordered_dict(
            'online_z_posterior/kl_prior',
            online_posterior_kls,
        ))
        logs['offline_vs_online_z_posterior/within_task_kl'] = (
            within_task_kl_off_on
        )
        logs['offline_vs_online_z_posterior/between_task_kl'] = (
            between_task_kl_off_on
        )
        logs['offline_vs_online_z_posterior/within_task_kl/normalized'] = (
                (within_task_kl_off_on - within_task_kl_2samples) / (
            (between_task_kl_2samples - within_task_kl_2samples)
        )
        )
        return logs

    def _compute_within_and_between_task_kl(self, context_distribs1,
                                            context_distribs2):
        n_tasks = len(context_distribs1)
        divergences = np.zeros((n_tasks, n_tasks))
        for i1, d1 in enumerate(context_distribs1):
            for i2, d2 in enumerate(context_distribs2):
                kl = kl_divergence(d1, d2).sum().item()
                divergences[i1, i2] = kl
        within_task_avg_kl = (divergences * np.eye(n_tasks)).sum() / n_tasks
        between_task_avg_kl = (divergences * (1-np.eye(n_tasks))).sum() / (
                n_tasks * (n_tasks - 1)
        )
        return within_task_avg_kl, between_task_avg_kl

    def _get_returns_init_from_offline_buffer(self, indices):
        train_returns = []
        for idx in indices:
            self.env.reset_task(idx)
            paths = []
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                init_context = self._reward_decoder_buffer.sample_context(
                    idx,
                    self.embedding_batch_size
                )
                init_context = ptu.from_numpy(init_context)
                p, _ = self.sampler.obtain_samples(
                    deterministic=self.eval_deterministic,
                    max_samples=self.max_path_length,
                    accum_context=False,
                    max_trajs=1,
                    resample_latent_period=0,
                    update_posterior_period=0,
                    initial_context=init_context,
                    task_idx=idx,
                )
                paths += p

            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

            train_returns.append(eval_util.get_average_returns(paths))
        return train_returns
