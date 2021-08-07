import time

from collections import defaultdict


class Timer:
    def __init__(self, return_global_times=False):
        self.stamps = None
        self.epoch_start_time = None
        self.global_start_time = time.time()
        self._return_global_times = return_global_times

        self.reset()

    def reset(self):
        self.stamps = defaultdict(lambda: 0)
        self.start_times = {}
        self.epoch_start_time = time.time()

    def start_timer(self, name, unique=True):
        if unique:
            assert name not in self.start_times.keys()
        self.start_times[name] = time.time()

    def stop_timer(self, name):
        assert name in self.start_times.keys()
        start_time = self.start_times[name]
        end_time = time.time()
        self.stamps[name] += (end_time - start_time)

    def get_times(self):
        global_times = {}
        cur_time = time.time()
        global_times['epoch_time'] = (cur_time - self.epoch_start_time)
        if self._return_global_times:
            global_times['global_time'] = (cur_time - self.global_start_time)
        return {
            **self.stamps.copy(),
            **global_times,
        }

    @property
    def return_global_times(self):
        return self._return_global_times

    @return_global_times.setter
    def return_global_times(self, value):
        self._return_global_times = value


timer = Timer()
