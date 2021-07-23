import numpy as np
from gym import spaces
from gym import Env


class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """
    GOAL_SIZE = 0.1  # fraction of image
    GOAL_COLOR = np.array([0, 255, 0], dtype=np.uint8)
    AGENT_SIZE = 0.05
    AGENT_COLOR = np.array([0, 0, 255], dtype=np.uint8)

    def __init__(self, randomize_tasks=False, n_tasks=2):
        if randomize_tasks:
            goals = [[np.random.uniform(-1., 1.), np.random.uniform(-1., 1.)] for _ in range(n_tasks)]
        else:
            # some hand-coded goals for debugging
            goals = [np.array([10, -10]),
                     np.array([10, 10]),
                     np.array([-10, 10]),
                     np.array([-10, -10]),
                     np.array([0, 0]),

                     np.array([7, 2]),
                     np.array([0, 4]),
                     np.array([-6, 9])
                     ]
            goals = [g / 10. for g in goals]
        self.goals = goals

        self.reset_task(0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    @property
    def tasks(self):
        return self.goals

    @tasks.setter
    def tasks(self, value):
        self.goals =value

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal = self.goals[idx]
        self.reset()

    def get_all_task_idx(self):
        return range(len(self.goals))

    def reset_model(self):
        # reset to a random location on the unit square
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
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)

    def get_image(self, width, height):
        white_img = np.zeros((height, width, 3), dtype=np.uint8)
        img_with_goal = draw(
            self._goal,
            width,
            height,
            white_img,
            self.GOAL_SIZE,
            self.GOAL_COLOR
        )
        final_img = draw(
            self._state,
            width,
            height,
            img_with_goal,
            self.AGENT_SIZE,
            self.AGENT_COLOR
        )
        return final_img


def draw(xy, width, height, img, size, color):
    x, y = xy
    x_pixel = map_to_int(x, [-1, 1], [0, width])
    y_pixel = map_to_int(y, [-1, 1], [0, height])

    x_min = int(max(x_pixel-size * width, 0))
    x_max = int(min(x_pixel+size * width, width))

    y_min = int(max(y_pixel - size * height, 0))
    y_max = int(min(y_pixel + size * height, height))

    img[y_min:y_max, x_min:x_max, :] = color
    return img


def map_to_int(x, in_range, out_range):
    min_x, max_x = in_range
    min_y, max_y = out_range
    normalized_x = (x - min_x) / (max_x - min_x)
    return (max_y - min_y) * normalized_x + min_y


class SparsePointEnv(PointEnv):
    '''
     - tasks sampled from unit half-circle
     - reward is L2 distance given only within goal radius

     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''
    def __init__(self, randomize_tasks=False, n_tasks=2, goal_radius=0.2):
        super().__init__(randomize_tasks, n_tasks)
        self.goal_radius = goal_radius

        if randomize_tasks:
            np.random.seed(1337)
            radius = 1.0
            angles = np.linspace(0, np.pi, num=n_tasks)
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            goals = np.stack([xs, ys], axis=1)
            np.random.shuffle(goals)
            goals = goals.tolist()

        self.goals = goals
        self.reset_task(0)

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r

    def reset_model(self):
        self._state = np.array([0, 0])
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        d.update({'sparse_reward': sparse_reward})
        return ob, reward, done, d
