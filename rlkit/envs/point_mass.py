import numpy as np
from gym import spaces
from gym import Env


class PointEnv(Env):
    """
    point mass on a 2-D plane
    two tasks: move to (-1, -1) or move to (1,1)
    """

    def __init__(self, task={'direction': 1}, randomize_tasks=False, n_tasks=2):
        # directions = [-1, 0, 1, 2, 3, 4, 5, 6]
        directions = list(range(n_tasks))

        if randomize_tasks:
            goals = [[np.random.uniform(-1., 1.), np.random.uniform(-1., 1.)] for _ in directions]

            # goals = [1 * np.random.uniform(-1., 1., 2) for _ in directions]
        else:
            # add more goals in n_tasks > 7
            goals = [np.array([10, -10]),
                     np.array([10, 10]),
                     np.array([-10, 10]),
                     np.array([-10, -10]),
                     np.array([0, 0]),

                     np.array([7, 2]),
                     np.array([0, 4]),
                     np.array([-6, 9])
                     ]
        self.goals = goals

        self.tasks = [{'direction': direction} for direction in directions]
        self._task = task
        self._goal = self.reset_goal(task.get('direction', 1))
        self.reset_model()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self.reset_goal(self._task['direction'])
        self.reset()

    def reset_goal(self, direction):
        return self.goals[direction]
        """
        if direction == 6:
            return np.array([-6., 9.]) # 1,1 and -1,-1 originally
        if direction == 5:
            return np.array([0., 4.]) # 1,1 and -1,-1 originally
        if direction == 4:
            return np.array([7., 2.]) # 1,1 and -1,-1 originally
        if direction == 2:
            return np.array([-10., -10.]) # 1,1 and -1,-1 originally
        elif direction == 1:
            return np.array([5, 10]) # 1,1 and -1,-1 originally
        elif direction == 0:
            return np.array([10, -10])
        elif direction == 3:
            return np.array([0., 0.])
        else:
            return np.array([-10, 0])
        """

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_model(self):
        self._state = np.random.uniform(-1., 1., size=(2,))
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
        done = False # abs(x) < 0.01 and abs(y) < 0.01
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

