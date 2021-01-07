from multiprocessing.pool import ThreadPool as Pool

import numpy as np
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import Policy
from rlkit.torch.model_based.dreamer.basic_mcts_wm import UCT_search, run_parallel


class DiscreteMCTSPolicy(Policy):
    """"""

    def __init__(
        self,
        world_model,
        max_steps,
        num_primitives,
        action_dim,
        action_space,
        iterations,
        exploration_weight,
        open_loop_plan=False,
        parallelize=False,
    ):
        self.world_model = world_model
        cont_action = action_space.low[num_primitives:]
        actions = np.array(
            [
                np.concatenate((np.squeeze(np.eye(num_primitives)[i]), cont_action))
                for i in range(num_primitives)
            ]
        )
        actions = ptu.from_numpy(actions)
        self.world_model.actions = actions
        self.max_steps = max_steps
        self.num_primitives = num_primitives
        self.action_dim = action_dim
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.open_loop_plan = open_loop_plan
        self.parallelize = parallelize

    def get_action(self, observation):
        """

        :param observation:
        :return: action, debug_dictionary
        """
        observation = ptu.from_numpy(np.array(observation))
        if self.state:
            latent, action = self.state
        else:
            latent = self.world_model.initial(observation.shape[0])
            action = ptu.zeros((observation.shape[0], self.action_dim))
        if self.open_loop_plan:
            actions = [self.actions[i][self.ctr] for i in range(observation.shape[0])]
            action = [
                self.world_model.actions[action].reshape(1, -1) for action in actions
            ]
            action = torch.cat(action)
        else:
            embed = self.world_model.encode(observation)
            start_state, _ = self.world_model.obs_step(latent, action, embed)
            if self.parallelize:
                p = Pool(observation.shape[0])
                results = []
                for i in range(observation.shape[0]):
                    st = {}
                    for k, v in start_state.items():
                        st[k] = v[i : i + 1].detach()
                    results.append(
                        p.apply_async(
                            run_parallel,
                            args=(
                                [
                                    st,
                                    self.max_steps - self.ctr,
                                    self.world_model,
                                    self.num_primitives,
                                    self.exploration_weight,
                                    self.iterations,
                                    False,
                                ]
                            ),
                        )
                    )

                actions = [p.get()[0] for p in results]
                actions = [
                    self.world_model.actions[action].reshape(1, -1)
                    for action in actions
                ]
                action = torch.cat(actions)
            else:
                actions = []
                for i in range(observation.shape[0]):
                    st = {}
                    for k, v in start_state.items():
                        st[k] = v[i : i + 1].detach()

                    action = UCT_search(
                        self.world_model,
                        st,
                        self.iterations,
                        self.max_steps - self.ctr,
                        self.num_primitives,
                    )[0]
                    action = self.world_model.actions[action].reshape(1, -1)
                    actions.append(action)
                action = torch.cat(actions)
        self.ctr += 1
        self.state = (latent, action)
        return ptu.get_numpy(action), {}

    def reset(self, o):
        self.state = None
        self.ctr = 0
        o = np.concatenate([o_.reshape(1, -1) for o_ in o])
        latent = self.world_model.initial(o.shape[0])
        action = ptu.zeros((o.shape[0], self.action_dim))
        o = ptu.from_numpy(np.array(o))
        embed = self.world_model.encode(o)
        start_state, _ = self.world_model.obs_step(latent, action, embed)
        if self.open_loop_plan:
            if self.parallelize:
                p = Pool(o.shape[0])
                results = []
                for i in range(o.shape[0]):
                    st = {}
                    for k, v in start_state.items():
                        st[k] = v[i : i + 1].detach()
                    results.append(
                        p.apply_async(
                            run_parallel,
                            args=(
                                [
                                    st,
                                    self.max_steps,
                                    self.world_model,
                                    self.num_primitives,
                                    self.exploration_weight,
                                    self.iterations,
                                    True,
                                ]
                            ),
                        )
                    )
                self.actions = [p.get() for p in results]
            else:
                self.actions = [
                    UCT_search(
                        self.world_model,
                        start_state,
                        self.iterations,
                        self.max_steps,
                        self.num_primitives,
                        return_open_loop_plan=True,
                    )
                ]


class ActionSpaceSamplePolicy(Policy):
    def __init__(self, env):
        self.env = env

    def get_action(self, observation):
        return (
            np.array([self.env.action_space.sample() for _ in range(self.env.n_envs)]),
            {},
        )

    def reset(self, o):
        return super().reset()
