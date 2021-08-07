import numpy as np
from gym import utils
from rlkit.envs.pearl_envs.rand_param_envs.base import RandomEnv
import os

class PR2Env(RandomEnv, utils.EzPickle):

    FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets/pr2.xml')

    def __init__(self, log_scale_limit=1.):
        self.viewer = None
        RandomEnv.__init__(self, log_scale_limit, 'pr2.xml', 4)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:7],
            self.model.data.qvel.flat[:7],  # Do not include the velocity of the target (should be 0).
            self.get_tip_position().flat,
            self.get_vec_tip_to_goal().flat,
        ])

    def get_tip_position(self):
        return self.model.data.site_xpos[0]

    def get_vec_tip_to_goal(self):
        tip_position = self.get_tip_position()
        goal_position = self.goal
        vec_tip_to_goal = goal_position - tip_position
        return vec_tip_to_goal

    @property
    def goal(self):
        return self.model.data.qpos.flat[-3:]

    def _step(self, action):

        self.do_simulation(action, self.frame_skip)

        vec_tip_to_goal = self.get_vec_tip_to_goal()
        distance_tip_to_goal = np.linalg.norm(vec_tip_to_goal)

        reward = - distance_tip_to_goal

        state = self.state_vector()
        notdone = np.isfinite(state).all()
        done = not notdone

        ob = self._get_obs()

        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        goal = np.random.uniform((0.2, -0.4, 0.5), (0.5, 0.4, 1.5))
        qpos[-3:] = goal
        qpos[:7] += self.np_random.uniform(low=-.005, high=.005,  size=7)
        qvel[:7] += self.np_random.uniform(low=-.005, high=.005,  size=7)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2
        # self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -50
        # self.viewer.cam.lookat[0] = self.model.stat.center[0]
        # self.viewer.cam.lookat[1] = self.model.stat.center[1]
        # self.viewer.cam.lookat[2] = self.model.stat.center[2]


if __name__ == "__main__":

    env = PR2Env()
    tasks = env.sample_tasks(40)
    while True:
        env.reset()
        env.set_task(np.random.choice(tasks))
        print(env.model.body_mass)
        for _ in range(100):
            env.render()
            env.step(env.action_space.sample())
