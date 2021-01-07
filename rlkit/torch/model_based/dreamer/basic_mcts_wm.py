# https://www.moderndescartes.com/essays/deep_dive_mcts/

import math
import time

import numpy as np
import torch
from d4rl.kitchen.kitchen_envs import KitchenSlideCabinetV0

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.model_based.dreamer.world_models import WorldModel


def random_rollout(wm, state, max_steps, num_primitives, step_count):
    if step_count == max_steps:
        return wm.reward(wm.get_feat(state))[0].item()
    returns = 0
    for i in range(step_count, max_steps):
        idx = np.random.choice(num_primitives)
        action = wm.actions[idx : idx + 1, :]
        new_state = wm.action_step(state, action)
        r = wm.reward(wm.get_feat(new_state))[0]
        returns += r
    return returns.item()


@torch.no_grad()
def UCT_search(
    wm,
    state,
    iterations,
    max_steps,
    num_primitives,
    exploration_weight=1.0,
    return_open_loop_plan=False,
):
    root = UCTNode(wm, state, num_primitives, exploration_weight=exploration_weight)
    root.expand()
    for i in range(iterations):
        leaf = root.select_leaf()
        returns = random_rollout(
            wm, leaf.state, max_steps, num_primitives, leaf.step_count
        )
        if leaf.step_count < max_steps:
            leaf.expand()
        else:
            leaf.is_expanded = True  # for terminal states
            leaf.is_terminal = True  # for terminal states
        leaf.backup(returns)
    if return_open_loop_plan:
        output_actions = []
        cur = root
        while cur.children != {}:
            max_Q = -np.inf
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


def step_wm(wm, state, actions):
    state_n = {}
    for k, v in state.items():
        state_n[k] = v.repeat(actions.shape[0], 1)
    new_states = wm.action_step(state_n, actions)
    return new_states


class UCTNode:
    def __init__(
        self,
        wm,
        state,
        num_primitives,
        step_count=0,
        exploration_weight=1.0,
        parent=None,
    ):
        self.wm = wm
        self.state = state
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.total_value = 0
        self.number_visits = 0
        self.is_terminal = False
        self.num_primitives = num_primitives
        self.step_count = step_count
        self.exploration_weight = exploration_weight

    def Q(self) -> float:
        return self.total_value / (1 + self.number_visits)

    def U(self) -> float:
        return math.sqrt(self.parent.number_visits) / (1 + self.number_visits)

    def best_child(self):
        # best child that is not known to be terminal
        # todo: make it random choice between equal value children
        return max(
            self.children.values(),
            key=lambda node: node.Q() + self.exploration_weight * node.U(),
        )

    def get_all_values(self):
        return [node.Q() + node.U() for node in self.children.values()]

    def select_leaf(self):
        current = self
        while current.is_expanded and not current.is_terminal:
            current = current.best_child()
        return current

    def expand(self):
        self.is_expanded = True
        child_states = step_wm(self.wm, self.state, self.wm.actions)
        for i in range(self.wm.actions.shape[0]):
            child_state = {}
            for k, v in child_states.items():
                child_state[k] = v[i : i + 1, :]
            self.add_child(child_state, self.wm.actions[i, :])

    def add_child(self, state, action):
        self.children[torch.argmax(action[: self.num_primitives]).item()] = UCTNode(
            self.wm,
            state,
            self.num_primitives,
            self.step_count + 1,
            exploration_weight=self.exploration_weight,
            parent=self,
        )

    def backup(self, returns):
        current = self
        while current is not None:
            current.number_visits += 1
            current.total_value += returns
            current = current.parent


def run_parallel(
    state=None,
    max_steps=None,
    wm=None,
    num_primitives=None,
    exploration_weight=1,
    iterations=1000,
    return_open_loop_plan=False,
):
    st = {}
    for k, v in state.items():
        st[k] = v.clone().detach()
    return UCT_search(
        wm,
        st,
        iterations,
        max_steps,
        num_primitives,
        exploration_weight=exploration_weight,
        return_open_loop_plan=return_open_loop_plan,
    )


if __name__ == "__main__":
    env = KitchenSlideCabinetV0(
        fixed_schema=False, delta=0.0, dense=False, image_obs=True
    )
    f = "data/12-24-dreamer_v2_reinforce_reproduce_2020_12_24_14_33_19_0000--s-87504/params.pkl"
    data = torch.load(f)
    ptu.set_gpu_mode(True)

    wm = WorldModel(
        env.action_space.low.size,
        env.image_shape,
        env,
    ).to(ptu.device)

    cont_action = env.action_space.low[env.num_primitives :]
    actions = np.array(
        [
            np.concatenate((np.squeeze(np.eye(env.num_primitives)[i]), cont_action))
            for i in range(env.num_primitives)
        ]
    )
    actions = ptu.from_numpy(actions).detach()

    wm.actions = actions
    state = wm.initial(1)
    import time

    t = time.time()
    action = UCT_search(wm, state, 10000, env.max_steps, env.num_primitives)[0]
    print(time.time() - t)
    print(action)
    t = time.time()
    action = wm.actions[action]
    state = step_wm(wm, state, action)
    action = UCT_search(wm, state, 10000, env.max_steps, env.num_primitives)[0]
    print(action)
    action = wm.actions[1]
    state = step_wm(wm, state, action)
    action = UCT_search(wm, state, 10000, env.max_steps, env.num_primitives)[0]
    print(action)
