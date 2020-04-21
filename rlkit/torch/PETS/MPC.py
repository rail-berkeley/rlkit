from torch import nn as nn

from rlkit.policies.base import Policy
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.PETS.optimizer import CEMOptimizer
from rlkit.torch.core import np_ify

class MPCPolicy(Policy):
    """
    Usage:
    ```
    policy = MPCPolicy(...)
    action, mean, log_std, _ = policy(obs)
    ```
    """
    def __init__(
            self,
            model,
            obs_dim,
            action_dim,
            num_bootstrap,
            num_particles,
            cem_horizon,
            cem_iters,
            cem_popsize,
            cem_num_elites,
    ):
        super().__init__()
        self.model = model
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # set up CEM optimizer
        sol_dim = action_dim * cem_horizon
        # assuming normalized environment
        self.ac_ub = 1
        self.ac_lb = -1
        self.optimizer = CEMOptimizer(sol_dim, cem_iters, cem_popsize, cem_num_elites, self._cost_function, upper_bound=self.ac_ub, lower_bound=self.ac_lb)
        self.cem_horizon = cem_horizon
        # 16 here comes from torch PETS implementation, unsure why
        self.cem_init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 16, [self.cem_horizon, self.action_dim])
        self.current_obs = None
        self.prev_sol = None
        self.num_particles = num_particles

    def reset(self):
        self.current_obs = None
        self.prev_sol = None

    def get_action(self, obs_np, deterministic=False):
        self.current_obs = obs_np
        self.optimizer.reset()
        init_mean = np.zeros(self.cem_horizon * self.action_dim)
        if self.prev_sol is not None:
            init_mean[:(self.cem_horizon - 1) * self.action_dim] = self.prev_sol[self.action_dim:]
        new_sol = self.optimizer.obtain_solution(init_mean, self.cem_init_var)
        return new_sol[:action_dim]


    def get_actions(self, obs_np, deterministic=False):
        # TODO: figure out how this is used
        # return eval_np(self, obs_np, deterministic=deterministic)[0]
        

    @torch.no_grad()
    def _cost_function(self, ac_seqs):
        '''
        a function from action sequence to cost, either from the model or the given 
        cost function. TODO: add the sampling strategies from the PETS paper

        ac_seqs: batch_size * (cem_horizon * action_dimension)
        requires self.current_obs to be accurately set
        '''
        batch_size = ac_seqs.shape[0]
        ac_seqs = ac_seqs.reshape((batch_size, self.cem_horizon, self.action_dim))
        obs = np.tile(reps=(batch_size, self.num_particles, 1), self.current_obs)
        obs = obs.reshape((batch_size * self.num_particles, self.obs_dim))
        ac_seqs = np.tile(reps=(1, self.num_particles, 1, 1), ac_seqs[:, np.newaxis, :, :])
        observations, rewards = self.model.unroll(obs, ac_seqs)
        rewards = np_ify(rewards).reshape((batch_size, self.num_particles, self.cem_horizon)).mean(axis=(1, 2))
        return rewards
        
