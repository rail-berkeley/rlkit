import torch
from ipdb import set_trace as db

GOAL_POSITION = 0.45
GOAL_VELOCITY = 0.

@torch.no_grad()
def mountain_car_continuous_reward(state, action, next_state):
    batch_size = state.shape[0]
    position = next_state[:, 0]
    velocity = next_state[:, 1]

    reward = (position >= GOAL_POSITION) * (velocity >= GOAL_POSITION) * 100.
    reward = reward[:, None].float()
    reward -= 0.1 * (action ** 2)
    return reward
