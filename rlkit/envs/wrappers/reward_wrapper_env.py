from rlkit.envs.proxy_env import ProxyEnv


class RewardWrapperEnv(ProxyEnv):
    """Substitute a different reward function"""

    def __init__(
            self,
            env,
            compute_reward_fn,
    ):
        ProxyEnv.__init__(self, env)
        self.spec = env.spec # hack for hand envs
        self.compute_reward_fn = compute_reward_fn

    def step(self, action):
        next_obs, reward, done, info = self._wrapped_env.step(action)
        info["env_reward"] = reward
        reward = self.compute_reward_fn(next_obs, reward, done, info)
        return next_obs, reward, done, info
