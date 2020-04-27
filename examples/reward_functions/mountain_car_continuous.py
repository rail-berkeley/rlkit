import numpy as np

GOAL_POSITION = 0.45
GOAL_VELOCITY = 0.

def mountain_car_continuous_reward(state, action, next_state):
    position = next_state[0]
    velocity = next_state[1]

    done = bool(position >= GOAL_POSITION and velocity >= GOAL_POSITION)
    reward = 100 if done else 0
    reward -= 0.1 * (action[0] ** 2)
    return reward
