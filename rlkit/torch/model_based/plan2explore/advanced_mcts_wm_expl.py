# https://www.moderndescartes.com/essays/deep_dive_mcts/
# https://github.com/werner-duvaud/muzero-general/blob/master/self_play.py
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
from rlkit.torch.model_based.dreamer.actor_models import ConditionalActorModel
from rlkit.torch.model_based.dreamer.mlp import Mlp
from rlkit.torch.model_based.dreamer.world_models import WorldModel
from rlkit.torch.model_based.plan2explore.actor_models import (
    ConditionalContinuousActorModel,
)
from rlkit.torch.model_based.plan2explore.latent_space_models import (
    OneStepEnsembleModel,
)


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    https://github.com/werner-duvaud/muzero-general/blob/master/self_play.py
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


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


@torch.no_grad()
def Advanced_UCT_search(
    wm,
    one_step_ensemble,
    actor,
    state,
    max_steps,
    num_primitives,
    vf,
    mcts_iterations=50,
    evaluation=False,
    intrinsic_reward_scale=1.0,
    extrinsic_reward_scale=0.0,
    discount=1.0,
    dirichlet_alpha=0.03,
    progressive_widening_constant=0.0,
    use_dirichlet_exploration_noise=False,
    use_puct=False,
    normalize_q=False,
    use_reward_discount_value=False,
    use_muzero_uct=False,
    use_max_visit_count=False,
    return_open_loop_plan=False,
    progressive_widening_type='all',
):
    root = UCTNode(
        wm,
        one_step_ensemble,
        actor,
        state,
        num_primitives,
        max_steps,
        reward=0,
        prior=0,
        step_count=0,
        parent=None,
    )
    exploration_fraction=0.25
    root.expand(
        intrinsic_reward_scale=intrinsic_reward_scale,
        extrinsic_reward_scale=extrinsic_reward_scale,
        evaluation=evaluation,
        use_dirichlet_exploration_noise=use_dirichlet_exploration_noise,
        dirichlet_alpha=dirichlet_alpha,
        exploration_fraction=exploration_fraction,
    )
    ctr = 0
    min_max_stats = MinMaxStats()
    c1 = 1.25
    c2 = 19625
    keep_searching = return_open_loop_plan
    while ctr < mcts_iterations or keep_searching:
        leaf = root.select_leaf(
            progressive_widening_constant,
            intrinsic_reward_scale,
            extrinsic_reward_scale,
            evaluation,
            min_max_stats,
            discount,
            normalize_q,
            use_reward_discount_value,
            c1,
            c2,
            use_puct,
            use_muzero_uct,
            use_dirichlet_exploration_noise,
            dirichlet_alpha,
            exploration_fraction,
            progressive_widening_type,
        )
        if not leaf.value:
            states = wm.get_feat(leaf.state)
            value = vf(states)[0].item()
            leaf.value = value
        if leaf.step_count < max_steps:
            leaf.expand(
                intrinsic_reward_scale=intrinsic_reward_scale,
                extrinsic_reward_scale=extrinsic_reward_scale,
                evaluation=evaluation,
                use_dirichlet_exploration_noise=use_dirichlet_exploration_noise,
                dirichlet_alpha=dirichlet_alpha,
                exploration_fraction=exploration_fraction,
            )
        else:
            leaf.is_expanded = True  # for terminal states
            leaf.is_terminal = True  # for terminal states
        leaf.backup(leaf.value, discount, min_max_stats, use_reward_discount_value)
        ctr += 1
        if return_open_loop_plan and ctr >= mcts_iterations:
            path = compute_best_path(root, use_max_visit_count)
            if path.shape[0] == max_steps:
                return path
    return compute_best_action(root, use_max_visit_count)[0]


def compute_best_path(root, use_max_visit_count):
    output_actions = []
    cur = root
    while cur.children != {}:
        action, cur = compute_best_action(cur, use_max_visit_count)
        output_actions.append(action)
    actions = np.concatenate(output_actions, 0)
    return actions


def compute_best_action(node, use_max_visit_count):
    max_val = -np.inf
    max_a = None
    max_child = None
    for a, child in node.children.items():
        if use_max_visit_count:
            val = child.number_visits
        else:
            val = child.average_value()
        if val >= max_val:
            max_val = val
            max_a = a
            max_child = child
    return np.array(max_a).reshape(1, -1), max_child


def generate_full_actions(
    wm,
    state,
    actor,
    discrete_actions,
    evaluation=False,
):
    feat = wm.get_feat(state)
    if type(actor) == ConditionalContinuousActorModel:
        action_input = (discrete_actions, feat)
        action_dist = actor(action_input)
        if evaluation:
            # continuous_action = action_dist.mode().float()
            continuous_action = action_dist.sample().float()
        else:
            continuous_action = actor.compute_exploration_action(
                action_dist.sample(),
                0.3,  # computing log prob of noisy action can lead to nans
            ).float()
        actions = torch.cat((discrete_actions, continuous_action), 1)
        priors = action_dist.log_prob(continuous_action)
    elif type(actor) == ConditionalActorModel:
        action_input = feat
        action_dist = actor(action_input)
        cont_dist = action_dist.compute_continuous_dist(discrete_actions)
        if evaluation:
            # continuous_action = cont_dist.mode().float()
            continuous_action = cont_dist.sample().float()
        else:
            continuous_action = actor.compute_continuous_exploration_action(
                cont_dist.sample(), 0.3
            ).float()
        actions = torch.cat((discrete_actions, continuous_action), 1)
        priors = action_dist.log_prob_given_continuous_dist(actions, cont_dist)
    return actions, priors

def step_wm(
    wm,
    state,
    actor,
    num_primitives,
    one_step_ensemble,
    evaluation=False,
    intrinsic_reward_scale=1.0,
    extrinsic_reward_scale=0.0,
    actions_type='all' #max_prior, max_value
):
    
    if actions_type=='all':
        discrete_actions = ptu.eye(num_primitives)
    elif actions_type=='max_prior':
        discrete_actions = step_wm_action(use_max_prior=True)
    elif actions_type == 'max_value':
        discrete_actions = step_wm_action(use_max_prior=False)
    state_n = {}
    for k, v in state.items():
        state_n[k] = v.repeat(discrete_actions.shape[0], 1)
    action, priors = generate_full_actions(
        wm, state_n, actor, discrete_actions, evaluation
    )
    new_state = wm.action_step(state_n, action)
    deter_state = state_n["deter"]
    r = ptu.zeros(deter_state.shape[0])
    if intrinsic_reward_scale > 0.0:
        r += (
            compute_exploration_reward(one_step_ensemble, deter_state, action).flatten()
            * intrinsic_reward_scale
        )
    if extrinsic_reward_scale > 0.0:
        r += wm.reward(wm.get_feat(new_state)).flatten() * extrinsic_reward_scale
    return new_state, action, priors, r

def step_wm_action(node, use_max_prior=True):
    max_val = -np.inf
    max_action = None
    for a, node in node.children.items():
        if use_max_prior:
            val = node.prior
        else:
            val = node.average_value()
        if max_val > val:
            max_action = a
            max_val = val
    discrete_action = max_action[:node.num_primitives]
    return ptu.from_numpy(discrete_action).reshape(1, -1).repeat(node.num_primitives, 1)

class UCTNode:
    def __init__(
        self,
        wm,
        one_step_ensemble,
        actor,
        state,
        num_primitives,
        max_steps,
        reward,
        prior,
        step_count=0,
        parent=None,
    ):
        self.wm = wm
        self.one_step_ensemble = one_step_ensemble
        self.actor = actor
        self.state = state
        self.reward = reward
        self.parent = parent
        self.num_primitives = num_primitives
        self.step_count = step_count
        self.max_steps = max_steps
        self.is_expanded = False
        self.children = {}
        self.total_value = 0
        self.number_visits = 0
        self.is_terminal = False
        self.value = None
        self.prior = prior

    def average_value(self):
        if self.number_visits > 0:
            value = self.total_value / (self.number_visits)
        else:
            value = 0
        return value

    def Q(self, min_max_stats, discount, normalize_q, use_reward_discount_value):
        q = self.average_value()
        if use_reward_discount_value:
            q = self.reward + discount * q
        if normalize_q:
            return min_max_stats.normalize(q)
        else:
            return q

    def U(self, c1, c2, use_muzero_uct, prior):
        prior_score = math.sqrt(self.parent.number_visits) / (self.number_visits + 1)
        if use_muzero_uct:
            prior_score *= math.log((self.parent.number_visits + c2 + 1) / c2) + c1
        prior_score *= prior
        return prior_score

    def score(
        self,
        min_max_stats,
        discount,
        normalize_q,
        use_reward_discount_value,
        c1,
        c2,
        use_muzero_uct,
        prior,
    ):
        q = self.Q(min_max_stats, discount, normalize_q, use_reward_discount_value)
        u = self.U(c1, c2, use_muzero_uct, prior)
        return q + u

    def best_child(
        self,
        min_max_stats,
        discount,
        normalize_q,
        use_reward_discount_value,
        c1,
        c2,
        use_puct,
        use_muzero_uct,
        use_dirichlet_exploration_noise,
    ):
        max_score = -np.inf
        best_node = None
        priors = np.array([node.prior for node in self.children.values()])
        max_prior = priors.max()
        priors = priors - max_prior
        normalization = np.exp(priors).sum()
        priors = np.exp(priors) / np.exp(priors).sum()
        assert np.allclose(priors.sum(), 1)
        for node in self.children.values():
            if use_reward_discount_value:
                min_max_stats.update(node.reward + discount * node.average_value())
            else:
                min_max_stats.update(node.average_value())
            if use_puct:
                prior = np.exp(node.prior - max_prior) / normalization
            else:
                prior = 1.0
            if self.parent is None and use_dirichlet_exploration_noise:  # must be root
                if not use_puct:
                    prior = prior / len(self.children)
                prior = (
                    prior * (1 - node.exploration_fraction)
                    + node.dirichlet_noise * node.exploration_fraction
                )
            
            score = node.score(
                min_max_stats,
                discount,
                normalize_q,
                use_reward_discount_value,
                c1,
                c2,
                use_muzero_uct,
                prior,
            )
            if score >= max_score:
                max_score = score
                best_node = node
        return best_node

    def select_leaf(
        self,
        progressive_widening_constant,
        intrinsic_reward_scale,
        extrinsic_reward_scale,
        evaluation,
        min_max_stats,
        discount,
        normalize_q,
        use_reward_discount_value,
        c1,
        c2,
        use_puct,
        use_muzero_uct,
        use_dirichlet_exploration_noise,
        dirichlet_alpha,
        exploration_fraction,
        progressive_widening_type,
    ):
        current = self
        while current.is_expanded and not current.is_terminal:
            current.progressively_widen(
                progressive_widening_constant,
                intrinsic_reward_scale,
                extrinsic_reward_scale,
                evaluation,
                use_dirichlet_exploration_noise,
                dirichlet_alpha,
                exploration_fraction,
                progressive_widening_type
            )

            current = current.best_child(
                min_max_stats,
                discount,
                normalize_q,
                use_reward_discount_value,
                c1,
                c2,
                use_puct,
                use_muzero_uct,
                use_dirichlet_exploration_noise,
            )
        return current

    def progressively_widen(
        self,
        progressive_widening_constant,
        intrinsic_reward_scale,
        extrinsic_reward_scale,
        evaluation,
        use_dirichlet_exploration_noise,
        dirichlet_alpha,
        exploration_fraction,
        progressive_widening_type, #all, max_prior, max_value
    ):
        # progressive widening
        alpha = 0.5
        thresh = math.ceil(progressive_widening_constant * self.number_visits ** alpha)
        if len(self.children) < thresh:
            """
            progressive widening ideas:
            1) widen all discrete actions by 1
            2) widen discrete action with highest prior probability by 1
            3) widen discrete action with highest average value by 1

            sample new actions from continuous policy or from action space bounds and scale to -1,1
            """
            child_states, actions, priors, rewards = step_wm(
                self.wm,
                self.state,
                self.actor,
                self.num_primitives,
                self.one_step_ensemble,
                intrinsic_reward_scale=intrinsic_reward_scale,
                extrinsic_reward_scale=extrinsic_reward_scale,
                evaluation=evaluation,
                actions_type=progressive_widening_type,
            )
            self.expand_given_states_actions(
                child_states,
                actions,
                priors,
                rewards,
                use_dirichlet_exploration_noise,
                dirichlet_alpha,
                exploration_fraction,
            )

    def expand(
        self,
        intrinsic_reward_scale,
        extrinsic_reward_scale,
        evaluation,
        use_dirichlet_exploration_noise,
        dirichlet_alpha,
        exploration_fraction,
    ):
        child_states, actions, priors, rewards = step_wm(
            self.wm,
            self.state,
            self.actor,
            self.num_primitives,
            self.one_step_ensemble,
            intrinsic_reward_scale=intrinsic_reward_scale,
            extrinsic_reward_scale=extrinsic_reward_scale,
            evaluation=evaluation,
        )
        self.expand_given_states_actions(
            child_states,
            actions,
            priors,
            rewards,
            use_dirichlet_exploration_noise,
            dirichlet_alpha,
            exploration_fraction,
        )

    def expand_given_states_actions(
        self,
        child_states,
        actions,
        priors,
        rewards,
        use_dirichlet_exploration_noise,
        dirichlet_alpha,
        exploration_fraction,
    ):
        if self.parent is None and use_dirichlet_exploration_noise:
            # therefore self must be the root
            noise = np.random.dirichlet([dirichlet_alpha] * self.num_primitives)
            frac = exploration_fraction
        for i in range(self.num_primitives):
            child_state = {}
            for k, v in child_states.items():
                child_state[k] = v[i : i + 1, :]
            node = self.add_child(
                child_state,
                actions[i, :],
                priors[i].item(),
                rewards[i].item(),
            )
            if self.parent is None and use_dirichlet_exploration_noise:
                node.exploration_fraction = exploration_fraction
                node.dirichlet_noise = noise[i]
        self.is_expanded = True

    def add_child(self, state, action, prior, reward):
        key = tuple(ptu.get_numpy(action).tolist())
        node = UCTNode(
            self.wm,
            self.one_step_ensemble,
            self.actor,
            state,
            self.num_primitives,
            self.max_steps,
            reward,
            prior,
            step_count=self.step_count + 1,
            parent=self,
        )
        self.children[key] = node
        return node

    def backup(self, value, discount, min_max_stats, use_reward_discount_value):
        current = self
        while current is not None:
            current.number_visits += 1
            current.total_value += value
            if use_reward_discount_value:
                min_max_stats.update(
                    current.reward + discount * current.average_value()
                )
                value = current.reward + discount * value
            else:
                min_max_stats.update(current.average_value())
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
    # actor = ConditionalContinuousActorModel(
    #     [400] * 4,
    #     wm.feature_size,
    #     env,
    #     discrete_action_dim=env.num_primitives,
    #     continuous_action_dim=env.max_arg_len,
    # ).to(ptu.device)
    actor = ConditionalActorModel(
        [400] * 4,
        wm.feature_size,
        env,
        discrete_continuous_dist=True,
        discrete_action_dim=env.num_primitives,
        continuous_action_dim=env.max_arg_len,
    ).to(ptu.device)
    state = wm.initial(1)

    vf = Mlp(
        hidden_sizes=[400] * 3,
        output_size=1,
        input_size=wm.feature_size,
        hidden_activation=torch.nn.functional.elu,
    ).to(ptu.device)
    import time

    num_tries = 1
    total_time = 0
    for i in range(num_tries):
        t = time.time()

        action = Advanced_UCT_search(
            wm,
            one_step_ensemble,
            actor,
            state,
            mcts_iterations=100,
            max_steps=env.max_steps,
            num_primitives=env.num_primitives,
            vf=vf,
            evaluation=True,
            intrinsic_reward_scale=0.0,
            extrinsic_reward_scale=1.0,
            progressive_widening_constant=0,
            use_dirichlet_exploration_noise=False,
            use_puct=True,
            normalize_q=False,
            use_reward_discount_value=False,
            use_muzero_uct=False,
            use_max_visit_count=True,
            return_open_loop_plan=True,
            dirichlet_alpha=.25,
        )
        total_time += time.time() - t
    print(total_time / num_tries)
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
