import abc
from collections import OrderedDict
from typing import Iterable
import pickle

import numpy as np

import rlkit.core.eval_util
from rlkit.core.rl_algorithm import MetaRLAlgorithm
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule, np_ify, torch_ify
from rlkit.core import logger, eval_util


class MetaTorchRLAlgorithm(MetaRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(self, *args, render_eval_paths=False, plotter=None, output_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.plotter = plotter
        self.output_dir = output_dir

    def cuda(self):
        for net in self.networks:
            net.cuda()

    def get_batch(self, idx=None):
        if idx is None:
            idx = self.task_idx
        batch = self.replay_buffer.random_batch(idx, self.batch_size)
        return np_to_pytorch_batch(batch)

    # Get a batch from the separate encoding replay buffer.
    def get_encoding_batch(self, idx=None, eval_task=False):
        # n.b. if eval is online, training should match the distribution of context lengths
        is_online = (self.eval_embedding_source == 'online')
        if idx is None:
            idx = self.task_idx
        if eval_task:
            batch = self.eval_enc_replay_buffer.random_batch(idx, self.embedding_batch_size, padded=is_online)
        else:
            batch = self.enc_replay_buffer.random_batch(idx, self.embedding_batch_size, padded=is_online)
        return np_to_pytorch_batch(batch)

    # TODO: this whole function might be rewritten
    def obtain_eval_paths(self, idx, eval_task=False, deterministic=False):
        ''' collect paths with fixed task embedding '''
        # TODO: collect context tuples from replay buffer to match training stats
        is_online = (self.eval_embedding_source == 'online')
        self.policy.clear_z()
        if not is_online:
            self.sample_z_from_posterior(idx, eval_task=eval_task)

        print('task encoding ', self.policy.z)

        test_paths = self.eval_sampler.obtain_samples(deterministic=deterministic, is_online=is_online)
        return test_paths

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[PyTorchModule]:
        pass

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    # TODO: might be useful to use the logging info in this method for visualization and seeing how episodes progress as
    # stuff gets inferred, especially as we debug online evaluations
    def collect_data_for_embedding_online_with_logging(self, idx, statistics, epoch):
        self.task_idx = idx
        print('Task:', idx)
        self.eval_sampler.env.reset_task(idx)

        n_exploration_episodes = 10
        n_inference_episodes = 10
        all_init_paths = []
        all_inference_paths =[]

        self.enc_replay_buffer.clear_buffer(idx)

        for i in range(n_exploration_episodes):
            initial_z = self.sample_z_from_prior()

            init_paths = self.obtain_eval_paths(idx, z=initial_z, eval_task=True)
            all_init_paths += init_paths
            self.enc_replay_buffer.add_paths(idx, init_paths)
        print('enc_replay_buffer.task_buffers[idx]._size', self.enc_replay_buffer.task_buffers[idx]._size)

        for i in range(n_inference_episodes):
            paths = self.obtain_eval_paths(idx, eval_task=True)
            all_inference_paths += paths
            self.enc_replay_buffer.add_paths(idx, init_paths)

        # don't need test paths anymore
        """
        test_paths = self.obtain_eval_paths(idx, deterministic=True, eval_task=True)
        # TODO incorporate into proper logging
        for path in test_paths:
            path['goal'] = goal
        """

        # save evaluation rollouts for vis

        # all paths
        with open(self.output_dir +
                  "/proto-sac-point-mass-fb-16z-init-task{}-{}.pkl".format(idx, epoch), 'wb+') as f:
            pickle.dump(all_init_paths, f, pickle.HIGHEST_PROTOCOL)
        with open(self.output_dir +
                  "/proto-sac-point-mass-fb-16z-inference-task{}-{}.pkl".format(idx, epoch), 'wb+') as f:
            pickle.dump(all_inference_paths, f, pickle.HIGHEST_PROTOCOL)

        # statistics.update(eval_util.get_generic_path_information(
        #     test_paths, stat_prefix="Test_task{}".format(idx),
        # ))

        # average_returns = rlkit.core.eval_util.get_average_returns(test_paths)
        average_inference_returns = [rlkit.core.eval_util.get_average_returns(paths) for paths in all_inference_paths]
        # statistics['AverageReturn_test_task{}'.format(idx)] = average_returns
        statistics['AverageInferenceReturns_test_task{}'.format(idx)] = average_inference_returns
        # statistics['Goal_test_task{}'.format(idx)] = goal
        # return test_paths, statistics

    def evaluate(self, epoch):
        statistics = OrderedDict()
        # save stuff from training
        statistics.update(self.eval_statistics)
        self.eval_statistics = None
        print('evaluating on {} training tasks')
        train_avg_returns = []
        for idx in self.train_tasks:
            self.task_idx = idx
            print('Task:', idx)
            # TODO how to handle eval over multiple tasks?
            self.eval_sampler.env.reset_task(idx)

            goal = self.eval_sampler.env._goal
            print('task {} encoder RB size'.format(idx), self.enc_replay_buffer.task_buffers[idx]._size)

            # collects final evaluation trajectories
            test_paths = self.obtain_eval_paths(idx, eval_task=False, deterministic=True)
            # TODO incorporate into proper logging
            for path in test_paths:
                path['goal'] = goal # goal

            # save evaluation rollouts for vis
            with open(self.output_dir +
                      "/proto-sac-point-mass-fb-16z-train-task{}-{}.pkl".format(idx, epoch), 'wb+') as f:
                pickle.dump(test_paths, f, pickle.HIGHEST_PROTOCOL)

            statistics.update(eval_util.get_generic_path_information(
                test_paths, stat_prefix="Test_task{}".format(idx),
            ))
            statistics.update(eval_util.get_generic_path_information(
                self._exploration_paths, stat_prefix="Exploration_task{}".format(idx),
            )) # something is wrong with these exploration paths i'm pretty sure...
            # if hasattr(self.env, "log_diagnostics"):
            #     self.env.log_diagnostics(test_paths)

            average_returns = rlkit.core.eval_util.get_average_returns(test_paths)
            # TODO: add flag for disabling individual task logging info, since it's a lot of clutter
            statistics['AverageReturn_training_task{}'.format(idx)] = average_returns
            statistics['GoalPosition_training_task{}'.format(idx)] = goal
            train_avg_returns += [average_returns]
            print('GoalPosition_training_task')
            print(goal)



        print('evaluating on {} evaluation tasks'.format(len(self.eval_tasks)))

        test_avg_returns = []
        # This is calculating the embedding online, because every iteration
        # we clear the encoding buffer for the test tasks.
        for idx in self.eval_tasks:
            self.task_idx = idx
            print('Task:', idx)
            self.eval_sampler.env.reset_task(idx)

            # collect data fo computing embedding if needed
            if self.eval_embedding_source in ['online', 'initial_pool']:
                pass
            elif self.eval_embedding_source == 'online_exploration_trajectories':
                self.eval_enc_replay_buffer.task_buffers[idx].clear()
                # task embedding sampled from prior and held fixed
                self.collect_data_sampling_from_prior(num_samples=self.num_steps_per_task,
                                                      resample_z_every_n=self.max_path_length,
                                                      eval_task=True)
            elif self.eval_embedding_source == 'online_on_policy_trajectories':
                self.eval_enc_replay_buffer.task_buffers[idx].clear()
                # half the data from z sampled from prior, the other half from z sampled from posterior
                self.collect_data_online(idx=idx,
                                         num_samples=self.num_steps_per_task,
                                         eval_task=True)

                # TODO: decide whether we want this
                # run this instead to view extra logging
                # self.collect_data_for_embedding_online_with_logging(idx, statistics)
            else:
                raise Exception("Invalid option for computing eval embedding")

            # TODO: Add parameters for eval steps
            test_paths = []
            for _ in range(self.num_evals):
                test_paths += self.obtain_eval_paths(idx, eval_task=True, deterministic=self.eval_deterministic)
            # TODO incorporate into proper logging
            goal = self.eval_sampler.env._goal
            for path in test_paths:
                path['goal'] = goal

            # save evaluation rollouts for vis
            with open(self.output_dir +
                      "/proto-sac-point-mass-fb-16z-test-task{}-{}.pkl".format(idx, epoch), 'wb+') as f:
                pickle.dump(test_paths, f, pickle.HIGHEST_PROTOCOL)

            statistics.update(eval_util.get_generic_path_information(
                test_paths, stat_prefix="Test_task{}".format(idx),
            ))
            if hasattr(self.env, "log_diagnostics"):
                self.env.log_diagnostics(test_paths)

            average_returns = rlkit.core.eval_util.get_average_returns(test_paths)
            statistics['AverageReturn_test_task{}'.format(idx)] = average_returns
            statistics['GoalPosition_test_task{}'.format(idx)] = goal
            test_avg_returns += [average_returns]

            # TODO: flags for these, flesh out other embedding/evaluation schemes
            # UNCOMMENT THIS AND COMMENT OUT THE ABOVE CODE TO USE ONLINE EMBEDDING
            # test_paths, statistics = self.evaluate_with_online_embedding(idx, statistics, epoch)

        avg_train_return = np.mean(train_avg_returns)
        avg_test_return = np.mean(test_avg_returns)
        statistics['AverageReturn_all_train_tasks'] = avg_train_return
        statistics['AverageReturn_all_test_tasks'] = avg_test_return

        for key, value in statistics.items():
            logger.record_tabular(key, value)

        if self.render_eval_paths:
            self.env.render_paths(test_paths)

        if self.plotter:
            self.plotter.draw()

def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return ptu.from_numpy(elem_or_tuple).float()


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }
