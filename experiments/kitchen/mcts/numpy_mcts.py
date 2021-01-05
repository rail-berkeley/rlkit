import collections
import math
import time

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
        #     idx = 7
        action = env.actions[idx]
        _, r, _, _ = env.step(action)
        returns += r
    return returns


def step_env(env, state, action):
    env.set_env_state(state)
    env.step(action)
    return env.get_env_state()


class UCTNode:
    def __init__(self, env, state, action, parent=None):
        self.env = env
        self.state = state
        self.action = action
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.child_total_value = np.zeros([env.num_primitives], dtype=np.float32)
        self.child_number_visits = np.zeros([env.num_primitives], dtype=np.int64)
        self.is_terminal = False

    def convert_action_to_idx(self, action):
        return np.argmax(action[: self.env.num_primitives])

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.convert_action_to_idx(self.action)]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.convert_action_to_idx(self.action)] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.convert_action_to_idx(self.action)]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.convert_action_to_idx(self.action)] = value

    def child_Q(self):
        q = np.divide(
            self.child_total_value,
            self.child_number_visits.astype(np.float),
            where=self.child_number_visits != 0,
        )
        q = np.where(self.child_number_visits == 0, 0, q)
        return q

    def child_U(self):
        u = np.sqrt(
            np.divide(
                self.number_visits,
                self.child_number_visits.astype(np.float),
                where=self.child_number_visits != 0,
            )
        )
        u = np.where(self.child_number_visits == 0, np.inf, u)
        return u

    def best_child(self):
        return np.argmax(self.child_Q() + self.child_U())

    def select_leaf(self):
        current = self
        while current.is_expanded and not current.is_terminal:
            best_move = current.best_child()
            current = current.maybe_add_child(best_move)
        return current

    def expand(self):
        self.is_expanded = True

    def maybe_add_child(self, action_idx):
        action = env.actions[action_idx]
        if action_idx not in self.children:
            child_state = step_env(self.env, self.state, action)
            self.children[action_idx] = UCTNode(
                self.env, child_state, action, parent=self
            )
        return self.children[action_idx]

    def backup(self, returns: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += returns
            current = current.parent


class DummyNode:
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


def UCT_search(env, state, iterations):
    root = UCTNode(env, state, env.actions[0], parent=DummyNode())
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
    t = time.time()
    action = UCT_search(env, state, 584)[0]
    print(time.time() - t)
    print(action)

    # action = env.actions[4]
    # state = step_env(env, state, action)
    # action = UCT_search(env, state, 584)[0]
    # print(action)

    # action = env.actions[1]
    # state = step_env(env, state, action)
    # action = UCT_search(env, state, 584)[0]
    # print(action)
