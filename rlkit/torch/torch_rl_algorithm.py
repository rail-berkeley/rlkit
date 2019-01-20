import abc
from collections import OrderedDict
from typing import Iterable
import pickle

import numpy as np

from rlkit.core import logger
from rlkit.core.eval_util import dprint
from rlkit.core.rl_algorithm import MetaRLAlgorithm
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule, np_ify, torch_ify
from rlkit.core import logger, eval_util


class MetaTorchRLAlgorithm(MetaRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(self, *args, render_eval_paths=False, plotter=None, dump_eval_paths=False, output_dir=None, recurrent=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.plotter = plotter
        self.dump_eval_paths = dump_eval_paths
        self.output_dir = output_dir
        self.recurrent = recurrent

    ###### Torch stuff #####
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

    ##### Data handling #####
    def get_batch(self, idx=None):
        ''' get a batch from replay buffer for input into net '''
        if idx is None:
            idx = self.task_idx
        batch = self.replay_buffer.random_batch(idx, self.batch_size)
        return np_to_pytorch_batch(batch)

    def get_encoding_batch(self, idx=None, eval_task=False):
        ''' get a batch from the separate encoding replay buffer '''
        # n.b. if eval is online, training should sample trajectories rather than unordered batches to better match statistics
        is_online = (self.eval_embedding_source == 'online')
        if idx is None:
            idx = self.task_idx
        if eval_task:
            batch = self.eval_enc_replay_buffer.random_batch(idx, self.embedding_batch_size, trajs=is_online)
        else:
            batch = self.enc_replay_buffer.random_batch(idx, self.embedding_batch_size, trajs=is_online)
        return np_to_pytorch_batch(batch)

    ##### Eval stuff #####
    def obtain_eval_paths(self, idx, eval_task=False, deterministic=False):
        '''
        collect paths with current policy
        if online, task encoding will be updated after each transition
        otherwise, sample a task encoding once and keep it fixed
        '''
        is_online = (self.eval_embedding_source == 'online')
        self.policy.clear_z()

        if not is_online:
            self.sample_z_from_posterior(idx, eval_task=eval_task)

        dprint('task encoding ', self.policy.z)

        test_paths = self.eval_sampler.obtain_samples(deterministic=deterministic, is_online=is_online)
        if self.sparse_rewards:
            for p in test_paths:
                p['rewards'] = ptu.sparsify_rewards(p['rewards'])
        return test_paths


    # TODO: might be useful to use the logging info in this method for visualization and seeing how episodes progress as
    # stuff gets inferred, especially as we debug online evaluations
    def collect_data_for_embedding_online_with_logging(self, idx, epoch):
        self.task_idx = idx
        dprint('Task:', idx)
        self.env.reset_task(idx)

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
        dprint('enc_replay_buffer.task_buffers[idx]._size', self.enc_replay_buffer.task_buffers[idx]._size)

        for i in range(n_inference_episodes):
            paths = self.obtain_eval_paths(idx, eval_task=True)
            all_inference_paths += paths
            self.enc_replay_buffer.add_paths(idx, init_paths)

        # save evaluation rollouts for vis
        # all paths
        with open(self.output_dir +
                  "/proto-sac-point-mass-fb-16z-init-task{}-{}.pkl".format(idx, epoch), 'wb+') as f:
            pickle.dump(all_init_paths, f, pickle.HIGHEST_PROTOCOL)
        with open(self.output_dir +
                  "/proto-sac-point-mass-fb-16z-inference-task{}-{}.pkl".format(idx, epoch), 'wb+') as f:
            pickle.dump(all_inference_paths, f, pickle.HIGHEST_PROTOCOL)

        average_inference_returns = [eval_util.get_average_returns(paths) for paths in all_inference_paths]
        self.eval_statistics['AverageInferenceReturns_test_task{}'.format(idx)] = average_inference_returns

    def collect_paths(self, idx, epoch, eval_task=False):
        self.task_idx = idx
        dprint('Task:', idx)
        self.env.reset_task(idx)
        if eval_task:
            num_evals = self.num_evals
        else: 
            num_evals = 1

        paths = []
        for _ in range(num_evals):
            paths += self.obtain_eval_paths(idx, eval_task=eval_task, deterministic=True)
        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            split = 'test' if eval_task else 'train'
            logger.save_extra_data(paths, path='eval_trajectories/{}-task{}-epoch{}'.format(split, idx, epoch))
        return paths

    def log_statistics(self, paths, split=''):
        self.eval_statistics.update(eval_util.get_generic_path_information(
            paths, stat_prefix="{}_task{}".format(split, self.task_idx),
        ))
        # TODO(KR) what are these?
        self.eval_statistics.update(eval_util.get_generic_path_information(
            self._exploration_paths, stat_prefix="Exploration_task{}".format(self.task_idx),
        )) # something is wrong with these exploration paths i'm pretty sure...
        average_returns = eval_util.get_average_returns(paths)
        self.eval_statistics['AverageReturn_{}_task{}'.format(split, self.task_idx)] = average_returns
        goal = self.env._goal
        dprint('GoalPosition_{}_task'.format(split))
        dprint(goal)
        self.eval_statistics['GoalPosition_{}_task{}'.format(split, self.task_idx)] = goal

    def evaluate(self, epoch):
        statistics = OrderedDict()
        statistics.update(self.eval_statistics)
        self.eval_statistics = statistics

        ### train tasks
        dprint('evaluating on {} train tasks'.format(len(self.train_tasks)))
        train_avg_returns = []
        for idx in self.train_tasks:
            dprint('task {} encoder RB size'.format(idx), self.enc_replay_buffer.task_buffers[idx]._size)
            paths = self.collect_paths(idx, epoch, eval_task=False)
            train_avg_returns.append(eval_util.get_average_returns(paths))

        ### test tasks
        dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_avg_returns = []
        # This is calculating the embedding online, because every iteration
        # we clear the encoding buffer for the test tasks.
        for idx in self.eval_tasks:
            self.task_idx = idx
            self.env.reset_task(idx)

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
            else:
                raise Exception("Invalid option for computing eval embedding")

            dprint('task {} encoder RB size'.format(idx), self.eval_enc_replay_buffer.task_buffers[idx]._size)
            test_paths = self.collect_paths(idx, epoch, eval_task=True)

            test_avg_returns.append(eval_util.get_average_returns(test_paths))

            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.policy.z_dists[0].mean)))
                z_sig = np.mean(ptu.get_numpy(self.policy.z_dists[0].variance))
                self.eval_statistics['Z mean eval'] = z_mean
                self.eval_statistics['Z variance eval'] = z_sig

            # TODO(KR) what does this do
            if hasattr(self.env, "log_diagnostics"):
                self.env.log_diagnostics(test_paths)


        avg_train_return = np.mean(train_avg_returns)
        avg_test_return = np.mean(test_avg_returns)
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

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
