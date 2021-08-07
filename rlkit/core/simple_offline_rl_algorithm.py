from collections import OrderedDict

import numpy as np


from rlkit.core.timer import timer
from rlkit.core import logger
from rlkit.core.logging import add_prefix
from rlkit.torch.core import np_to_pytorch_batch


def _get_epoch_timings():
    times_itrs = timer.get_times()
    times = OrderedDict()
    for key in sorted(times_itrs):
        time = times_itrs[key]
        times['time/{} (s)'.format(key)] = time
    return times


class SimpleOfflineRlAlgorithm(object):
    def __init__(
            self,
            trainer,
            replay_buffer,
            batch_size,
            logging_period,
            num_batches,
    ):
        self.trainer = trainer
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.logging_period = logging_period

    def train(self):
        # first train only the Q function
        iteration = 0
        for i in range(self.num_batches):
            train_data = self.replay_buffer.random_batch(self.batch_size)
            train_data = np_to_pytorch_batch(train_data)
            obs = train_data['observations']
            next_obs = train_data['next_observations']
            train_data['observations'] = obs
            train_data['next_observations'] = next_obs
            self.trainer.train_from_torch(train_data)
            if i % self.logging_period == 0:
                stats_with_prefix = add_prefix(
                    self.trainer.eval_statistics, prefix="trainer/")
                self.trainer.end_epoch(iteration)
                iteration += 1
                logger.record_dict(stats_with_prefix)
                logger.dump_tabular(with_prefix=True, with_timestamp=False)


class OfflineMetaRLAlgorithm(object):
    def __init__(
            self,
            # main objects needed
            meta_replay_buffer,
            replay_buffer,
            task_embedding_replay_buffer,
            trainer,
            train_tasks,
            # settings
            batch_size,
            logging_period,
            meta_batch_size,
            num_batches,
            task_embedding_batch_size,
            extra_eval_fns=(),
            use_meta_learning_buffer=False,
    ):
        self.trainer = trainer
        self.meta_replay_buffer = meta_replay_buffer
        self.replay_buffer = replay_buffer
        self.task_embedding_replay_buffer = task_embedding_replay_buffer
        self.batch_size = batch_size
        self.task_embedding_batch_size = task_embedding_batch_size
        self.num_batches = num_batches
        self.logging_period = logging_period
        self.train_tasks = train_tasks
        self.meta_batch_size = meta_batch_size
        self._extra_eval_fns = extra_eval_fns
        self.use_meta_learning_buffer = use_meta_learning_buffer

    def train(self):
        # first train only the Q function
        iteration = 0
        timer.return_global_times = True
        timer.reset()
        for i in range(self.num_batches):
            if self.use_meta_learning_buffer:
                train_data = self.meta_replay_buffer.sample_meta_batch(
                    rl_batch_size=self.batch_size,
                    meta_batch_size=self.meta_batch_size,
                    embedding_batch_size=self.task_embedding_batch_size,
                )
                train_data = np_to_pytorch_batch(train_data)
            else:
                task_indices = np.random.choice(
                    self.train_tasks, self.meta_batch_size,
                )
                train_data = self.replay_buffer.sample_batch(
                    task_indices,
                    self.batch_size,
                )
                train_data = np_to_pytorch_batch(train_data)
                obs = train_data['observations']
                next_obs = train_data['next_observations']
                train_data['observations'] = obs
                train_data['next_observations'] = next_obs
                train_data['context'] = (
                    self.task_embedding_replay_buffer.sample_context(
                        task_indices,
                        self.task_embedding_batch_size,
                    ))
            timer.start_timer('train', unique=False)
            self.trainer.train_from_torch(train_data)
            timer.stop_timer('train')
            if i % self.logging_period == 0 or i == self.num_batches - 1:
                stats_with_prefix = add_prefix(
                    self.trainer.eval_statistics, prefix="trainer/")
                self.trainer.end_epoch(iteration)
                logger.record_dict(stats_with_prefix)
                timer.start_timer('extra_fns', unique=False)
                for fn in self._extra_eval_fns:
                    extra_stats = fn()
                    logger.record_dict(extra_stats)
                timer.stop_timer('extra_fns')


                # TODO: evaluate during offline RL
                # eval_stats = self.get_eval_statistics()
                # eval_stats_with_prefix = add_prefix(eval_stats, prefix="eval/")
                # logger.record_dict(eval_stats_with_prefix)

                logger.record_tabular('iteration', iteration)
                logger.record_dict(_get_epoch_timings())
                try:
                    import os
                    import psutil
                    process = psutil.Process(os.getpid())
                    logger.record_tabular('RAM Usage (Mb)', int(process.memory_info().rss / 1000000))
                except ImportError:
                    pass
                logger.dump_tabular(with_prefix=True, with_timestamp=False)
                iteration += 1

    def to(self, device):
        self.trainer.to(device)
    # def get_eval_statistics(self):
    #     ### train tasks
    #     # eval on a subset of train tasks for speed
    #     stats = OrderedDict()
    #     indices = np.random.choice(self.train_task_indices, len(self.eval_task_indices))
    #     for key, path_collector in self.path_collectors.item():
    #         paths = path_collector.collect_paths()
    #         returns = eval_util.get_average_returns(paths)
    #         stats[key + '/AverageReturns'] = returns
    #     return stats
