# Temporal Difference Models (TDMs)
The TDM implementation is a bit different from the other algorithms. One reason for this is that the goals and rewards are retroactively relabelled. Some notable implementation details:
 - The networks (policy and QF) take in the goal and tau (the number of time steps left).
 - The algorithm relabels the terminal and rewards, so the terminal/reward from the environment are ignored completely.
 - TdmNormalizer is used to normalize the observations/states. If you want, you can totally ignore it and set `num_pretrain_path=0`.
 - The environments need to be [MultitaskEnv](rlkit/torch/tdm/envs/multitask_env), meaning standard gym environments won't work out of the box. See below for details.

The example scripts have tuned hyperparameters. Specifically, the following hyperparameters are tuned, as they seem to be the most important ones to tune:
 - `num_updates_per_env_step`
 - `reward_scale`
 - `max_tau`


## Creating your own environment for TDM
A [MultitaskEnv](envs/multitask_env.py) instances needs to implement 3 functions:

```python
    def goal_dim(self) -> int:
        """
        :return: int, dimension of goal vector
        """
        pass

    @abc.abstractmethod
    def sample_goals(self, batch_size):
        pass

    @abc.abstractmethod
    def convert_obs_to_goals(self, obs):
        pass
```

If you want to see how to make an existing environment multitask, see [GoalXVelHalfCheetah](envs/half_cheetah_env.py), which builds off of Gym's HalfCheetah environments.

Another useful example might be [GoalXYPosAnt](envs/ant_env.py), which builds off a custom environment.

One important thing is that the environment should *not* include the goal as part of the state, since the goal will be separately given to the networks.
