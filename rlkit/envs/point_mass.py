import numpy as np
from gym import spaces
from gym import Env


class PointEnv(Env):
    """
    point mass on a 2-D plane
    two tasks: move to (-1, -1) or move to (1,1)
    """

    def __init__(self, task={'direction': 1}):
        directions = [-1, 1]
        self.tasks = [{'direction': direction} for direction in directions]
        self._task = task
        self._goal = self.reset_goal(task.get('direction', 1))
        self.reset_model()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self.reset_goal(self._task['direction'])

    def reset_goal(self, direction):
        if direction == 1:
            return np.array([1, 1])
        else:
            return np.array([-1, -1])

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_model(self):
        self._state = np.random.uniform(-1, 1, size=(2,))
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state)

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

