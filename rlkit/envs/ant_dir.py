import numpy as np
from rlkit.envs.ant_multitask_base import MultitaskAntEnv

class AntDirEnv(MultitaskAntEnv):



    """
    def step(self, action):
        torso_xyz_before = self.get_body_com("torso")
        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = self.get_body_com("torso")
        torso_velocity = torso_xyz_after - torso_xyz_before

        vel = torso_velocity[:2] / self.dt

        dir_reward = vel[0] # np.dot(vel, direct)

        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = dir_reward - ctrl_cost
        
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        # done = False

        infos = dict(reward_dir=dir_reward,
                     reward_ctrl=-ctrl_cost, task=self._task)
        return (observation, reward, done, infos)
    """

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        # obs_gym, r_gym, done_gym, infos_gym = super(AntDirEnv, self).step(action)
        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)
        # forward_reward = torso_velocity[0] / self.dt

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )

    def sample_tasks(self, num_tasks):
        #  choose this threshold

        # TODO: currently set to only one quadrant
        velocities = np.random.uniform(0., 1.0 * np.pi, size=(num_tasks,))
        tasks = [{'goal': velocity} for velocity in velocities]
        return tasks
