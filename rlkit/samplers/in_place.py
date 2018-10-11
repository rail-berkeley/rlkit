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
    def __init__(self, env, policy, exploration_policy, max_samples, max_path_length):
        self.env = env
        self.policy = policy
        self.exploration_policy = exploration_policy
        if exploration_policy is None:
            self.exploration_policy = policy
        self.max_path_length = max_path_length
        self.max_samples = max_samples
        assert (
            max_samples >= max_path_length,
            "Need max_samples >= max_path_length"
        )

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self, explore=False, num_samples=None):
        policy = self.policy if not explore else self.exploration_policy
        paths = []
        n_steps_total = 0
        max_samp = self.max_samples
        if num_samples is not None:
            max_samp = num_samples
        while n_steps_total + self.max_path_length <= max_samp:
            path = rollout(
                self.env, policy, max_path_length=self.max_path_length
            )
            paths.append(path)
            n_steps_total += len(path['observations'])
        return paths
