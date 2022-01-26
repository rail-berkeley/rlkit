import copy
import gc

import cv2
import numpy as np


def vec_rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    rollout_function_kwargs=None,
):
    num_envs = env.n_envs

    policy_obs = env.reset()
    agent.reset(policy_obs)

    observations = [policy_obs]
    rewards = [np.zeros(num_envs)]
    actions = [np.zeros((num_envs, env.action_space.low.size))]
    terminals = [[False] * num_envs]
    agent_infos = [{}]
    env_infos = [{}]

    for step in range(0, max_path_length):
        action, agent_info = agent.get_action(policy_obs)

        obs, reward, done, info = env.step(copy.deepcopy(action))
        observations.append(obs)
        rewards.append(reward)
        terminals.append(done)
        actions.append(action)
        agent_infos.append(agent_info)
        env_infos.append(info)
        if done.all():
            break
        policy_obs = obs
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    env_info_final = {}
    for info in env_infos[1:]:
        for key, value in info.items():
            if key in env_info_final:
                env_info_final[key].append(value)
            else:
                env_info_final[key] = [value]
    for key, value in env_info_final.items():
        env_info_final[key] = np.concatenate(value, 1)
    env_infos = env_info_final
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=np.array(terminals),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def vec_rollout_low_level_raps(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    rollout_function_kwargs=None,
):
    num_low_level_actions_per_primitive = rollout_function_kwargs[
        "num_low_level_actions_per_primitive"
    ]
    num_primitives = rollout_function_kwargs["num_primitives"]
    low_level_action_dim = rollout_function_kwargs["low_level_action_dim"]
    num_envs = env.n_envs

    low_level_actions = np.zeros(
        (
            num_envs,
            (max_path_length * num_low_level_actions_per_primitive) + 1,
            low_level_action_dim,
        ),
        dtype=np.float32,
    )
    high_level_actions = np.zeros(
        (
            num_envs,
            (max_path_length * num_low_level_actions_per_primitive) + 1,
            env.action_space.low.shape[0] + 1,  # Plus 1 includes phase variable.
        ),
        dtype=np.float32,
    )
    observations = np.zeros(
        (
            num_envs,
            (max_path_length * num_low_level_actions_per_primitive) + 1,
            env.observation_space.low.shape[0],
        ),
        dtype=np.uint8,
    )

    obs = env.reset()
    agent.reset(obs)
    high_level_action = np.zeros((num_envs, env.action_space.low.size + 1))
    low_level_action = np.zeros((num_envs, low_level_action_dim))
    reward = np.zeros(num_envs)

    observations[:, 0] = obs
    high_level_actions[:, 0] = high_level_action
    low_level_actions[:, 0] = low_level_action
    rewards = [reward]
    actions = [np.zeros((num_envs, env.action_space.low.size))]
    terminals = [[False] * num_envs]
    agent_infos = [{}]
    env_infos = [{}]

    policy_obs = (None, np.array(obs))
    phases = (
        np.linspace(
            0,
            1,
            num_low_level_actions_per_primitive,
            endpoint=False,
        )
        + 1 / (num_low_level_actions_per_primitive)
    )
    phases = np.repeat(phases.reshape(1, -1), num_envs, axis=0)
    for step in range(0, max_path_length):
        high_level_action, agent_info = agent.get_action(policy_obs)
        argmax = np.argmax(high_level_action[:, :num_primitives], axis=-1)
        one_hots = np.eye(num_primitives)[argmax]
        high_level_action = np.concatenate(
            (one_hots, high_level_action[:, num_primitives:]), axis=-1
        )

        _, reward, done, info = env.step(high_level_action)
        rewards.append(reward)
        terminals.append(done)
        actions.append(high_level_action)
        agent_infos.append(agent_info)
        low_level_action = np.array(info["low_level_action"])
        low_level_obs = np.array(info["low_level_obs"])
        del info["low_level_action"]
        del info["low_level_obs"]
        gc.collect()
        env_infos.append(info)
        low_level_actions[
            :,
            step * num_low_level_actions_per_primitive
            + 1 : (step + 1) * num_low_level_actions_per_primitive
            + 1,
        ] = np.array(low_level_action)
        observations[
            :,
            step * num_low_level_actions_per_primitive
            + 1 : step * num_low_level_actions_per_primitive
            + num_low_level_actions_per_primitive
            + 1,
        ] = low_level_obs

        high_level_action = np.repeat(
            np.array(high_level_action).reshape(num_envs, 1, -1),
            num_low_level_actions_per_primitive,
            axis=1,
        )
        high_level_action = np.concatenate(
            (high_level_action, np.expand_dims(phases, -1)), axis=2
        )
        high_level_actions[
            :,
            step * num_low_level_actions_per_primitive
            + 1 : step * num_low_level_actions_per_primitive
            + num_low_level_actions_per_primitive
            + 1,
        ] = high_level_action

        if done.all():
            break
        policy_obs = (np.array(low_level_action), low_level_obs)
    rewards = np.array(rewards)
    actions = np.array(actions)
    env_info_final = {}
    for info in env_infos[1:]:
        for key, value in info.items():
            if key in env_info_final:
                env_info_final[key].append(value)
            else:
                env_info_final[key] = [value]
    for key, value in env_info_final.items():
        env_info_final[key] = np.concatenate(value, 1)
    env_infos = env_info_final
    return dict(
        observations=observations,
        actions=actions,
        high_level_actions=high_level_actions,
        low_level_actions=low_level_actions,
        rewards=rewards,
        terminals=np.array(terminals),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
