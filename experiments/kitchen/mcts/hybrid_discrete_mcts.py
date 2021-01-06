# https://www.moderndescartes.com/essays/deep_dive_mcts/

import math

import numpy as np
from d4rl.kitchen.kitchen_envs import KitchenMicrowaveV0, KitchenSlideCabinetV0


def random_rollout(env, state, actions):
    env.set_env_state(state)
    env._get_obs()
    returns = env._get_reward_n_score(env.obs_dict)[0]["r_total"]
    if env.step_count == env.max_steps:
        return returns
    for i in range(env.step_count, env.max_steps):
        idx = np.random.choice(len(actions))
        # if i == 1:
        #     idx = 1
        # elif i == 2:
        #     idx = 9
        action = actions[idx]
        _, r, _, _ = env.step(action)
        returns += r
    return returns


def UCT_search(env, iterations, exploration_weight, return_open_loop_plan=False):
    root = UCTNode(
        env, env.get_env_state(), env.actions, exploration_weight=exploration_weight
    )
    root.expand()
    ctr = {1: 0, 2: 0, 3: 0}
    for i in range(iterations):
        leaf = root.select_leaf()
        returns = random_rollout(env, leaf.state, leaf.actions)
        if leaf.state[0] < env.max_steps:
            leaf.expand()
        else:
            leaf.is_expanded = True  # for terminal states
            leaf.is_terminal = True  # for terminal states
        ctr[leaf.state[0]] += 1
        leaf.backup(returns)
        # print(leaf.Q(), leaf.U(), leaf.Q() + leaf.exploration_weight * leaf.U())
        # print(i, returns)
        # print(ctr)
        # print()
        # in_order_traversal2(root)
    if return_open_loop_plan:
        output_actions = []
        cur = root
        while cur.children != {}:
            max_Q = 0
            max_a = None
            max_child = None
            for a, child in cur.children.items():
                if child.Q() >= max_Q:
                    max_Q = child.Q()
                    max_a = a
                    max_child = child
            output_actions.append(max_a)
            cur = max_child

        return output_actions
    else:
        return max(root.children.items(), key=lambda item: item[1].Q())


import queue

# def in_order_traversal(
#     root,
# ):
#     q = queue.Queue()
#     depth = 0
#     q.put((None, root, depth))
#     depth_printed = 0
#     while not q.empty():
#         action, node, d = q.get()
#         if d > depth_printed:
#             print()
#             depth_printed += 1
#         print(action, end=", ")
#         for i, (k, v) in enumerate(node.children.items()):
#             q.put((k, v, d + 1))
#     print()
#     print()


def in_order_traversal2(
    root,
):
    q = queue.Queue()
    q.put(("", root, 0, False))
    max_depth = 0
    while not q.empty():
        action, node, d, last_layer = q.get()
        for i, (k, v) in enumerate(node.children.items()):
            q.put((action + "+" + str(k), v, d + 1, False))
            max_depth = max(d + 1, max_depth)
    q = queue.Queue()
    q.put(("", root, 0, False))
    while not q.empty():
        action, node, d, last_layer = q.get()
        if d == max_depth:
            print(action, end=", ")
        for i, (k, v) in enumerate(node.children.items()):
            if v.children == {}:
                q.put((action + "+" + str(k), v, d + 1, True))
            else:
                q.put((action + "+" + str(k), v, d + 1, False))
    print()
    print()


def step_env(env, state, action):
    env.set_env_state(state)
    env.step(action)
    return env.get_env_state()


class UCTNode:
    def __init__(
        self,
        env,
        state,
        actions,
        progressively_widen=False,
        exploration_weight=1.0,
        parent=None,
    ):
        self.env = env
        self.state = state
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.total_value = 0
        self.number_visits = 0
        self.is_terminal = False
        self.actions = actions
        self.progressively_widen = progressively_widen
        self.exploration_weight = exploration_weight

    def Q(self) -> float:
        return self.total_value / (1 + self.number_visits)

    def U(self) -> float:
        return math.sqrt(self.parent.number_visits) / (1 + self.number_visits)

    def best_child(self):
        # best child that is not known to be terminal
        return max(
            self.children.values(),
            key=lambda node: node.Q() + self.exploration_weight * node.U(),
        )

    def get_all_values(self):
        return [node.Q() + node.U() for node in self.children.values()]

    def select_leaf(self):
        current = self
        while current.is_expanded and not current.is_terminal:
            if self.progressively_widen:
                current.add_continuous_actions()
            current = current.best_child()
        return current

    def add_continuous_actions(self):
        pass

    def expand(self):
        self.is_expanded = True
        for action in self.actions:
            self.add_child(action)

    def add_child(self, action):
        child_state = step_env(self.env, self.state, action)
        self.children[np.argmax(action[: self.env.num_primitives])] = UCTNode(
            self.env,
            child_state,
            self.env.actions,
            exploration_weight=self.exploration_weight,
            parent=self,
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
    action = UCT_search(env, 584, exploration_weight=0.1, return_open_loop_plan=True)
    print(action)

    # action = env.actions[action]
    # state = step_env(
    #     env,
    #     state,
    #     action,
    # )
    # action = UCT_search(env, 2379, exploration_weight=0.1)[0]
    # print(action)

    # action = env.actions[action]
    # state = step_env(env, state, action)
    # action = UCT_search(env, 2379, exploration_weight=0.1)[0]
    # print(action)
