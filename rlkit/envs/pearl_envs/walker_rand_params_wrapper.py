from rlkit.envs.pearl_envs.rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv


class WalkerRandParamsWrappedEnv(Walker2DRandParamsEnv):
    def __init__(self, n_tasks=2, randomize_tasks=True):
        super(WalkerRandParamsWrappedEnv, self).__init__()
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()
