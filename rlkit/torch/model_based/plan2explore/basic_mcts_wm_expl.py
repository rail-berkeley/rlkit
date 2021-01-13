# https://www.moderndescartes.com/essays/deep_dive_mcts/

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from d4rl.kitchen.kitchen_envs import (
    KitchenHingeCabinetV0,
    KitchenKettleV0,
    KitchenSlideCabinetV0,
)

from rlkit.torch import pytorch_util as ptu
from rlkit.torch.model_based.dreamer.world_models import WorldModel
from rlkit.torch.model_based.plan2explore.actor_models import (
    ConditionalContinuousActorModel,
)
from rlkit.torch.model_based.plan2explore.latent_space_models import (
    OneStepEnsembleModel,
)


def compute_exploration_reward(
    one_step_ensemble, exploration_imag_deter_states, exploration_imag_actions
):
    pred_embeddings = []
    input_state = exploration_imag_deter_states
    for mdl in range(one_step_ensemble.num_models):
        inputs = torch.cat((input_state, exploration_imag_actions), 1)
        pred_embeddings.append(
            one_step_ensemble.forward_ith_model(inputs, mdl).mean.unsqueeze(0)
        )
    pred_embeddings = torch.cat(pred_embeddings)
    assert pred_embeddings.shape[0] == one_step_ensemble.num_models
    assert pred_embeddings.shape[1] == input_state.shape[0]
    assert len(pred_embeddings.shape) == 3

    reward = (pred_embeddings.std(dim=0) * pred_embeddings.std(dim=0)).mean(dim=1)
    assert (reward >= 0.0).all()
    return reward


def random_rollout(
    wm,
    one_step_ensemble,
    actor,
    state,
    max_steps,
    num_primitives,
    step_count,
    intrinsic_reward_scale=1.0,
    extrinsic_reward_scale=0.0,
    evaluation=False,
):
    state, r = state
    if step_count == max_steps:
        return r
    returns = 0
    for i in range(step_count, max_steps):
        idx = np.random.choice(num_primitives)
        feat = wm.get_feat(state)
        discrete_action = ptu.zeros(num_primitives).reshape(1, -1)
        discrete_action[0][idx] = 1.0
        action_input = (discrete_action, feat)
        action_dist = actor(action_input)
        if evaluation:
            continuous_action = action_dist.mode()
        else:
            continuous_action = actor.compute_exploration_action(
                action_dist.sample(), 0.3
            )
        action = torch.cat((discrete_action, continuous_action), 1)
        new_state = wm.action_step(state, action)
        deter_state = state["deter"]
        r = ptu.zeros(1)
        if intrinsic_reward_scale > 0.0:
            r += (
                compute_exploration_reward(one_step_ensemble, deter_state, action)
                * intrinsic_reward_scale
            )
        if extrinsic_reward_scale > 0.0:
            r += wm.reward(wm.get_feat(new_state)).flatten() * extrinsic_reward_scale
        state = new_state
        returns += r
    return returns[0].item()


def compute_all_paths(cur):
    if len(cur.children) == 0:
        return []
    else:
        new_l = []
        for a, child in cur.children.items():
            l = compute_all_paths(child)
            if len(l) == 0:
                new_l.append([[a, child.Q()]])
            else:
                for path in l:
                    total_value = path[0][1] + child.Q()
                    p = [(a, total_value)] + path
                    new_l.append(p)
        return new_l


def compute_top_k_paths(l, k):
    l.sort(key=lambda x: x[0][1])
    top_k_paths = l[-k:]
    actions = []
    for path in top_k_paths:
        path_as = []
        for val in path:
            a, v = val
            path_as.append(ptu.from_numpy(np.array(a).reshape(1, -1)))
        path_as = torch.cat(path_as)
        actions.append(path_as.unsqueeze(0))
    return torch.cat(actions)


@torch.no_grad()
def UCT_search(
    wm,
    one_step_ensemble,
    actor,
    state,
    iterations,
    max_steps,
    num_primitives,
    exploration_weight=1.0,
    return_open_loop_plan=False,
    return_top_k_paths=False,
    k=1,
    evaluation=False,
    intrinsic_reward_scale=1.0,
    extrinsic_reward_scale=0.0,
):
    root = UCTNode(
        wm,
        one_step_ensemble,
        actor,
        state,
        num_primitives,
        max_steps,
        intrinsic_reward_scale=intrinsic_reward_scale,
        extrinsic_reward_scale=extrinsic_reward_scale,
        exploration_weight=exploration_weight,
        evaluation=evaluation,
    )
    root.expand()
    ctr = 0
    has_not_reached_max_depth = True
    while ctr < iterations or has_not_reached_max_depth:
        leaf = root.select_leaf()
        returns = random_rollout(
            wm,
            one_step_ensemble,
            actor,
            leaf.state,
            max_steps,
            num_primitives,
            leaf.step_count,
            intrinsic_reward_scale=intrinsic_reward_scale,
            extrinsic_reward_scale=extrinsic_reward_scale,
            evaluation=evaluation,
        )
        if leaf.step_count < max_steps:
            leaf.expand()
        else:
            leaf.is_expanded = True  # for terminal states
            leaf.is_terminal = True  # for terminal states
            if ctr >= iterations:
                if return_top_k_paths:
                    l = compute_all_paths(root)
                    l.sort(key=lambda x: x[0][1])
                    top_k_paths = l[-k:]
                    for path in top_k_paths:
                        has_not_reached_max_depth = has_not_reached_max_depth and not (
                            len(path) == max_steps
                        )
                else:
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
                        output_actions.append(np.array(max_a).reshape(1, -1))
                        cur = max_child
                    has_not_reached_max_depth = not (len(output_actions) == max_steps)
        leaf.backup(returns)
        ctr += 1

    if return_open_loop_plan:
        if return_top_k_paths:
            l = compute_all_paths(root)
            actions = compute_top_k_paths(l, k)
            return actions
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
            output_actions.append(np.array(max_a).reshape(1, -1))
            cur = max_child
        actions = np.concatenate(output_actions, 0)
        actions = ptu.from_numpy(actions)
        return actions
    else:
        max_Q = -np.inf
        max_a = None
        for a, child in root.children.items():
            if child.Q() >= max_Q:
                max_Q = child.Q()
                max_a = a
        return ptu.from_numpy(np.array(max_a)).reshape(1, -1)


def generate_full_actions(
    wm,
    state,
    actor,
    num_primitives,
    evaluation=False,
):

    discrete_actions = ptu.eye(num_primitives)
    feat = wm.get_feat(state)
    action_input = (discrete_actions, feat)
    action_dist = actor(action_input)
    if evaluation:
        continuous_action = action_dist.mode()
    else:
        continuous_action = actor.compute_exploration_action(action_dist.sample(), 0.3)
    actions = torch.cat((discrete_actions, continuous_action), 1)
    return actions


def step_wm(
    wm,
    one_step_ensemble,
    state,
    actor,
    num_primitives,
    evaluation=False,
    intrinsic_reward_scale=1.0,
    extrinsic_reward_scale=0.0,
):
    state, _ = state
    state_n = {}
    for k, v in state.items():
        state_n[k] = v.repeat(num_primitives, 1)
    action = generate_full_actions(wm, state_n, actor, num_primitives, evaluation)
    deter_state = state_n["deter"]
    new_state = wm.action_step(state_n, action)
    r = ptu.zeros(deter_state.shape[0])
    if intrinsic_reward_scale > 0.0:
        r += (
            compute_exploration_reward(one_step_ensemble, deter_state, action).flatten()
            * intrinsic_reward_scale
        )
    if extrinsic_reward_scale > 0.0:
        r += wm.reward(wm.get_feat(new_state)).flatten() * extrinsic_reward_scale
    return new_state, r, action


class UCTNode:
    def __init__(
        self,
        wm,
        one_step_ensemble,
        actor,
        state,
        num_primitives,
        max_steps,
        step_count=0,
        exploration_weight=1.0,
        parent=None,
        evaluation=False,
        intrinsic_reward_scale=1.0,
        extrinsic_reward_scale=0.0,
    ):
        self.wm = wm
        self.one_step_ensemble = one_step_ensemble
        self.actor = actor
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
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.extrinsic_reward_scale = extrinsic_reward_scale
        self.evaluation = evaluation
        self.max_steps = max_steps

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

    def progressively_widen(self):
        child_states, r, self.actions = step_wm(
            self.wm,
            self.one_step_ensemble,
            self.state,
            self.actor,
            self.num_primitives,
            intrinsic_reward_scale=self.intrinsic_reward_scale,
            extrinsic_reward_scale=self.extrinsic_reward_scale,
            evaluation=self.evaluation,
        )
        for i in range(self.actions.shape[0]):
            child_state = {}
            for k, v in child_states.items():
                child_state[k] = v[i : i + 1, :]
            node = self.add_child((child_state, r[i].item()), self.actions[i, :])
            returns = random_rollout(
                wm,
                one_step_ensemble,
                actor,
                node.state,
                node.max_steps,
                node.num_primitives,
                node.step_count,
                intrinsic_reward_scale=self.intrinsic_reward_scale,
                extrinsic_reward_scale=self.extrinsic_reward_scale,
                evaluation=node.evaluation,
            )
            node.backup(returns)

    def expand(self):
        self.is_expanded = True
        child_states, r, self.actions = step_wm(
            self.wm,
            self.one_step_ensemble,
            self.state,
            self.actor,
            self.num_primitives,
            intrinsic_reward_scale=self.intrinsic_reward_scale,
            extrinsic_reward_scale=self.extrinsic_reward_scale,
            evaluation=self.evaluation,
        )
        for i in range(self.actions.shape[0]):
            child_state = {}
            for k, v in child_states.items():
                child_state[k] = v[i : i + 1, :]
            self.add_child((child_state, r[i].item()), self.actions[i, :])

    def add_child(self, state, action):
        key = tuple(ptu.get_numpy(action).tolist())
        node = UCTNode(
            self.wm,
            self.one_step_ensemble,
            self.actor,
            state,
            self.num_primitives,
            self.max_steps,
            self.step_count + 1,
            exploration_weight=self.exploration_weight,
            parent=self,
            intrinsic_reward_scale=self.intrinsic_reward_scale,
            extrinsic_reward_scale=self.extrinsic_reward_scale,
            evaluation=self.evaluation,
        )
        self.children[key] = node
        return node

    def backup(self, returns):
        current = self
        while current is not None:
            current.number_visits += 1
            current.total_value += returns
            current = current.parent


if __name__ == "__main__":
    env = KitchenHingeCabinetV0(
        fixed_schema=False, delta=0.0, dense=False, image_obs=True
    )
    ptu.set_gpu_mode(True)
    torch.backends.cudnn.benchmark = True
    wm = WorldModel(
        env.action_space.low.size,
        env.image_shape,
        env,
    ).to(ptu.device)
    one_step_ensemble = OneStepEnsembleModel(
        action_dim=env.action_space.low.size,
        deterministic_state_size=400,
        embedding_size=1024,
        num_models=5,
        hidden_size=400,
        num_layers=4,
        output_embeddings=False,
    ).to(ptu.device)
    actor = ConditionalContinuousActorModel(
        [400] * 4,
        wm.feature_size,
        env,
        discrete_action_dim=env.num_primitives,
        continuous_action_dim=env.max_arg_len,
    ).to(ptu.device)
    cont_action = env.action_space.low[env.num_primitives :]
    state = wm.initial(1)
    import time

    t = time.time()
    action = UCT_search(
        wm,
        one_step_ensemble,
        actor,
        (state, 0),
        10000,
        env.max_steps,
        env.num_primitives,
        return_open_loop_plan=True,
        exploration_weight=0.1,
        evaluation=True,
        intrinsic_reward_scale=0.0,
        extrinsic_reward_scale=1.0,
        return_top_k_paths=False,
        k=1,
    )
    print(action.shape)
    print(time.time() - t)
    t = time.time()
    # state, r, _ = step_wm(wm, one_step_ensemble, (state, 0), actor, env.num_primitives)
    # action = UCT_search(
    #     wm,
    #     one_step_ensemble,
    #     actor,
    #     (state, 0),
    #     10000,
    #     env.max_steps,
    #     env.num_primitives,
    # )[0]
    # print(action)
