from gym.envs.mujoco import AntEnv


class AntNormal(AntEnv):
    def __init__(
            self,
            *args,
            n_tasks=2, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
            randomize_tasks=True, # shuffle the tasks after creating them
            **kwargs
    ):
        self.tasks = [0 for _ in range(n_tasks)]
        self._goal = 0
        super().__init__(*args, **kwargs)

    def get_all_task_idx(self):
        return self.tasks

    def reset_task(self, idx):
        # not tasks. just give the same reward every time step.
        pass

    def sample_tasks(self, num_tasks):
        return [0 for _ in range(num_tasks)]
