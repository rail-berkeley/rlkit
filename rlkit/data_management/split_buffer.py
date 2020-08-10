import random

from rlkit.data_management.replay_buffer import ReplayBuffer


class SplitReplayBuffer(ReplayBuffer):
    """
    Split the data into a training and validation set.
    """
    def __init__(
            self,
            train_replay_buffer: ReplayBuffer,
            validation_replay_buffer: ReplayBuffer,
            fraction_paths_in_train,
    ):
        self.train_replay_buffer = train_replay_buffer
        self.validation_replay_buffer = validation_replay_buffer
        self.fraction_paths_in_train = fraction_paths_in_train
        self.replay_buffer = self.train_replay_buffer

    def add_sample(self, *args, **kwargs):
        self.replay_buffer.add_sample(*args, **kwargs)

    def add_path(self, path):
        self.replay_buffer.add_path(path)
        self._randomly_set_replay_buffer()

    def num_steps_can_sample(self):
        return min(
            self.train_replay_buffer.num_steps_can_sample(),
            self.validation_replay_buffer.num_steps_can_sample(),
        )

    def terminate_episode(self, *args, **kwargs):
        self.replay_buffer.terminate_episode(*args, **kwargs)
        self._randomly_set_replay_buffer()

    def _randomly_set_replay_buffer(self):
        if random.random() <= self.fraction_paths_in_train:
            self.replay_buffer = self.train_replay_buffer
        else:
            self.replay_buffer = self.validation_replay_buffer

    def get_replay_buffer(self, training=True):
        if training:
            return self.train_replay_buffer
        else:
            return self.validation_replay_buffer

    def random_batch(self, batch_size):
        return self.train_replay_buffer.random_batch(batch_size)

    def __getattr__(self, attrname):
        return getattr(self.replay_buffer, attrname)

    def __getstate__(self):
        # Do not save self.replay_buffer since it's a duplicate and seems to
        # cause joblib recursion issues.
        return dict(
            train_replay_buffer=self.train_replay_buffer,
            validation_replay_buffer=self.validation_replay_buffer,
            fraction_paths_in_train=self.fraction_paths_in_train,
        )

    def __setstate__(self, d):
        self.train_replay_buffer = d['train_replay_buffer']
        self.validation_replay_buffer = d['validation_replay_buffer']
        self.fraction_paths_in_train = d['fraction_paths_in_train']
        self.replay_buffer = self.train_replay_buffer
