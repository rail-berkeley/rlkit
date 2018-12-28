import abc
from collections import OrderedDict
from typing import Iterable
import pickle

import numpy as np
from torch.autograd import Variable

import rlkit.core.eval_util
from rlkit.core.rl_algorithm import MetaRLAlgorithm
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.core import logger, eval_util


class MetaTorchRLAlgorithm(MetaRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(self, *args, render_eval_paths=False, plotter=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.plotter = plotter

    def cuda(self):
        for net in self.networks:
            net.cuda()

    def get_batch(self, idx=None):
        if idx is None:
            idx = self.task_idx
        batch = self.replay_buffer.random_batch(idx, self.batch_size)
        return np_to_pytorch_batch(batch)

    # Get a batch from the separate encoding replay buffer.
    def get_encoding_batch(self, eval_task=False, idx=None):
        if idx is None:
            idx = self.task_idx
        # if eval_task:
        #     batch = self.enc_eval_replay_buffer.random_batch(idx, self.batch_size)
        # else:
        batch = self.enc_replay_buffer.random_batch(idx, self.batch_size)
        return np_to_pytorch_batch(batch)

    @property
    @abc.abstractmethod
    def networks(self) -> Iterable[PyTorchModule]:
        pass

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)

    # def obtain_samples(self, env):
    #     '''
    #     get rollouts of current policy in env
    #     '''
    #     return self.eval_sampler.obtain_samples()

    def evaluate_with_online_embedding(self, idx, statistics, epoch):
        self.task_idx = idx
        print('Task:', idx)
        # self.enc_eval_replay_buffer.clear_buffer(idx)
        # TODO how to handle eval over multiple tasks?
        self.eval_sampler.env.reset_task(idx)

        goal = self.eval_sampler.env._goal

        n_exploration_episodes = 10
        all_init_paths = []
        all_inference_paths =[]
        
        self.enc_replay_buffer.clear_buffer(idx)
        
        for i in range(n_exploration_episodes):
            initial_z = np.random.normal(size=self.latent_dim)

            init_paths = self.obtain_eval_samples(idx, self.eval_sampler, z=initial_z, eval_task=True, explore=True)
            all_init_paths += init_paths
            self.enc_replay_buffer.add_paths(idx, init_paths)
            # if i % 10 == 0:
            #     for j in range(10):
            #         inference_paths = self.obtain_eval_samples(idx, self.eval_sampler, eval_task=True, explore=True)
            #         self.enc_eval_replay_buffer.add_paths(idx, inference_paths)
            #         all_inference_paths += [inference_paths]
            #         self.enc_eval_replay_buffer.clear_buffer(idx)


        # n_inference_episodes = 50
        # all_inference_paths =[]

        # for i in range(n_inference_episodes):
        #     inference_paths = self.obtain_eval_samples(idx, epoch, self.eval_sampler, explore=True)
        #     for path in inference_paths:
        #         path['goal'] = goal
        #     all_inference_paths += [inference_paths]
        #     self.replay_buffer.add_paths(idx, inference_paths)

        # sac.obtain_samples
        # print("self.enc_replay_buffer._size", self.enc_replay_buffer._size)
        print('enc_replay_buffer.task_buffers[idx]._size', self.enc_replay_buffer.task_buffers[idx]._size)

        test_paths = self.obtain_eval_samples(idx, self.eval_sampler, eval_task=True)
        # TODO incorporate into proper logging
        for path in test_paths:
            path['goal'] = goal

        # save evaluation rollouts for vis

        # all paths
        all_paths = []
        # for paths in all_inference_paths:
        #     all_paths += paths
        all_paths += test_paths
        # with open(self.pickle_output_dir +
        #           "/eval_trajectories/proto-sac-point-mass-fb-16z-inference-task{}-{}.pkl".format(idx, epoch), 'wb+') as f:
        #     pickle.dump(all_inference_paths, f, pickle.HIGHEST_PROTOCOL)
        with open(self.pickle_output_dir +
                  "/eval_trajectories/proto-sac-point-mass-fb-16z-init-task{}-{}.pkl".format(idx, epoch), 'wb+') as f:
            pickle.dump(all_init_paths, f, pickle.HIGHEST_PROTOCOL)
        with open(self.pickle_output_dir +
                  "/eval_trajectories/proto-sac-point-mass-fb-16z-test-task{}-{}.pkl".format(idx, epoch), 'wb+') as f:
            pickle.dump(all_paths, f, pickle.HIGHEST_PROTOCOL)

        statistics.update(eval_util.get_generic_path_information(
            test_paths, stat_prefix="Test_task{}".format(idx),
        ))
        # if hasattr(self.env, "log_diagnostics"):
        #     self.env.log_diagnostics(test_paths)

        average_returns = rlkit.core.eval_util.get_average_returns(test_paths)
        average_inference_returns = [rlkit.core.eval_util.get_average_returns(paths) for paths in all_inference_paths]
        statistics['AverageReturn_test_task{}'.format(idx)] = average_returns
        statistics['AverageInferenceReturns_test_task{}'.format(idx)] = average_inference_returns
        statistics['Goal_test_task{}'.format(idx)] = goal
        return test_paths, statistics

    def evaluate(self, epoch):
        statistics = OrderedDict()
        # save stuff from training
        statistics.update(self.eval_statistics)
        self.eval_statistics = None
        print('evaluating on {} training tasks')
        total_train_return = 0.
        for idx in self.train_tasks:
            self.task_idx = idx
            print('Task:', idx)
            # TODO how to handle eval over multiple tasks?
            self.eval_sampler.env.reset_task(idx)

            goal = self.eval_sampler.env._goal
            print('enc_replay_buffer.task_buffers[idx]._size', self.enc_replay_buffer.task_buffers[idx]._size)
            test_paths = self.obtain_eval_samples(idx, self.eval_sampler, eval_task=False)
            # TODO incorporate into proper logging
            for path in test_paths:
                path['goal'] = goal # goal

            # save evaluation rollouts for vis
            with open(self.pickle_output_dir +
                      "/eval_trajectories/proto-sac-point-mass-fb-16z-train-task{}-{}.pkl".format(idx, epoch), 'wb+') as f:
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
            statistics['AverageReturn_training_task{}'.format(idx)] = average_returns
            statistics['GoalPosition_training_task{}'.format(idx)] = goal
            total_train_return += average_returns
            print('GoalPosition_training_task')
            print(goal)

        

        print('evaluating on {} evaluation tasks'.format(len(self.eval_tasks)))
        total_test_return = 0.
        
        # This is calculating the embedding online, because every iteration
        # we clear the encoding buffer for the test tasks.
        for idx in self.eval_tasks:
            self.task_idx = idx
            print('Task:', idx)
            # TODO how to handle eval over multiple tasks?
            self.eval_sampler.env.reset_task(idx)

            # import ipdb; ipdb.set_trace()
            goal = self.eval_sampler.env._goal
            test_paths = self.obtain_eval_samples(idx, self.eval_sampler, eval_task=True)
            # TODO incorporate into proper logging
            for path in test_paths:
                path['goal'] = goal

            # save evaluation rollouts for vis
            with open(self.pickle_output_dir +
                      "/eval_trajectories/proto-sac-point-mass-fb-16z-test-task{}-{}.pkl".format(idx, epoch), 'wb+') as f:
                pickle.dump(test_paths, f, pickle.HIGHEST_PROTOCOL)

            statistics.update(eval_util.get_generic_path_information(
                test_paths, stat_prefix="Test_task{}".format(idx),
            ))
            if hasattr(self.env, "log_diagnostics"):
                self.env.log_diagnostics(test_paths)

            average_returns = rlkit.core.eval_util.get_average_returns(test_paths)
            statistics['AverageReturn_test_task{}'.format(idx)] = average_returns
            statistics['GoalPosition_test_task{}'.format(idx)] = goal
            total_test_return += average_returns

            # UNCOMMENT THIS AND COMMENT OUT THE ABOVE CODE TO USE ONLINE EMBEDDING
            # test_paths, statistics = self.evaluate_with_online_embedding(idx, statistics, epoch)
        
        avg_train_return = total_train_return / len(self.train_tasks)
        avg_test_return = total_test_return / len(self.eval_tasks)
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
    return Variable(ptu.from_numpy(elem_or_tuple).float(), requires_grad=False)


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
