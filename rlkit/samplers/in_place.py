from rlkit.samplers.util import rollout


class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_samples, max_path_length, render=False):
        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length
        self.max_samples = max_samples
        self.render = render
        assert max_samples >= max_path_length, "Need max_samples >= max_path_length"

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self):
        paths = []
        n_steps_total = 0
        while n_steps_total + self.max_path_length <= self.max_samples:
            path = rollout(
                self.env, self.policy, max_path_length=self.max_path_length,
                animated=self.render
            )
            paths.append(path)
            n_steps_total += len(path['observations'])
        return paths
