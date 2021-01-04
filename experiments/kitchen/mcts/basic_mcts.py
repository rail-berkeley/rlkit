# https://www.moderndescartes.com/essays/deep_dive_mcts/

import math

import numpy as np
from d4rl.kitchen.kitchen_envs import KitchenSlideCabinetV0


def random_rollout(env, state):
    env.set_env_state(state)
    env._get_obs()
    returns = env._get_reward_n_score(env.obs_dict)[0]["r_total"]
    for i in range(env.step_count, env.max_steps):
        idx = np.random.choice(env.num_primitives)
        # if i == 1:
        #     idx = 1
        # elif i == 2:
        #     idx = 1
        action = env.actions[idx]
        _, r, _, _ = env.step(action)
        returns += r
    return returns


def UCT_search(env, iterations):
    root = UCTNode(env, env.get_env_state())
    root.expand()
    ctr = {1: 0, 2: 0, 3: 0}
    for i in range(iterations):
        leaf = root.select_leaf()
        returns = random_rollout(env, leaf.state)
        if leaf.state[0] < env.max_steps:
            leaf.expand()
        else:
            leaf.is_expanded = True  # for terminal states
            leaf.is_terminal = True  # for terminal states
        ctr[leaf.state[0]] += 1
        leaf.backup(returns)
        # print(i, returns)
        # print(ctr)
        # print()
    return max(root.children.items(), key=lambda item: item[1].total_value)


def step_env(env, state, action):
    env.set_env_state(state)
    env.step(action)
    return env.get_env_state()


class UCTNode:
    def __init__(self, env, state, parent=None):
        self.env = env
        self.state = state
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.total_value = 0
        self.number_visits = 0
        self.is_terminal = False

    def Q(self) -> float:
        # return self.total_value / (1 + self.number_visits)
        if self.number_visits == 0:
            return self.total_value
        return self.total_value / (self.number_visits)

    def U(self) -> float:
        # return math.sqrt(self.parent.number_visits) / (1 + self.number_visits)
        if self.number_visits == 0:
            return np.inf
        return math.sqrt(self.parent.number_visits) / (self.number_visits)

    def best_child(self):
        # best child that is not known to be terminal
        return max(self.children.values(), key=lambda node: node.Q() + node.U())

    def get_all_values(self):
        return [node.Q() + node.U() for node in self.children.values()]

    def select_leaf(self):
        current = self
        while current.is_expanded and not current.is_terminal:
            current = current.best_child()
        return current

    def expand(self):
        self.is_expanded = True
        for action in env.actions:
            self.add_child(action)

    def add_child(self, action):
        child_state = step_env(self.env, self.state, action)
        self.children[np.argmax(action[: self.env.num_primitives])] = UCTNode(
            self.env, child_state, parent=self
        )

    def backup(self, returns):
        current = self
        while current is not None:
            current.number_visits += 1
            current.total_value += returns
            current = current.parent


if __name__ == "__main__":
    env = KitchenSlideCabinetV0(
        fixed_schema=False, delta=0.0, dense=False, image_obs=False
    )
    cont_action = env.action_space.low[env.num_primitives :]
    actions = [
        np.concatenate((np.squeeze(np.eye(env.num_primitives)[i]), cont_action))
        for i in range(env.num_primitives)
    ]
    env.actions = actions
    env.reset()
    state = env.get_env_state()
    action = UCT_search(env, 584)[0]
    print(action)

    action = env.actions[4]
    state = step_env(env, state, action)
    action = UCT_search(env, 584)[0]
    print(action)

    action = env.actions[1]
    state = step_env(env, state, action)
    action = UCT_search(env, 584)[0]
    print(action)
