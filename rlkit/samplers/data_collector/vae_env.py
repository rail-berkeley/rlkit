from rlkit.envs.vae_wrapper import VAEWrappedEnv
from rlkit.samplers.data_collector import GoalConditionedPathCollector


class VAEWrappedEnvPathCollector(GoalConditionedPathCollector):
    def __init__(
            self,
            goal_sampling_mode,
            env: VAEWrappedEnv,
            policy,
            decode_goals=False,
            **kwargs
    ):
        super().__init__(env, policy, **kwargs)
        self._goal_sampling_mode = goal_sampling_mode
        self._decode_goals = decode_goals

    def collect_new_paths(self, *args, **kwargs):
        self._env.goal_sampling_mode = self._goal_sampling_mode
        self._env.decode_goals = self._decode_goals
        return super().collect_new_paths(*args, **kwargs)