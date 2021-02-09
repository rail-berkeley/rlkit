import copy
from functools import partial

import cv2
import numpy as np

create_rollout_function = partial


def vec_rollout(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    preprocess_obs_for_policy_fn=None,
    get_action_kwargs=None,
    return_dict_obs=False,
    full_o_postprocess_func=None,
    reset_callback=None,
    save_video=True,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    raw_obs = []
    raw_next_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0

    o = env.reset()
    agent.reset(o)
    a = np.zeros((env.n_envs, env.action_space.low.size))
    r = np.zeros(env.n_envs)

    observations.append(o)
    rewards.append(r)
    terminals.append([False] * env.n_envs)
    actions.append(a)
    next_observations.append(o)
    raw_next_obs.append(o)
    agent_infos.append({})
    env_infos.append({})

    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        img = env.render(mode="rgb_array", imwidth=256, imheight=256)
        cv2.imshow("img", img)
        cv2.waitKey(1)
    while path_length < max_path_length:
        raw_obs.append(o)
        o_for_agent = preprocess_obs_for_policy_fn(o)
        a, agent_info = agent.get_action(o_for_agent, **get_action_kwargs)

        if full_o_postprocess_func:
            full_o_postprocess_func(env, agent, o)

        next_o, r, d, env_info = env.step(copy.deepcopy(a))
        if render:
            img = env.render(mode="rgb_array", imwidth=256, imheight=256)
            cv2.imshow("img", img)
            cv2.waitKey(1)
        observations.append(next_o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        raw_next_obs.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d.all():
            break
        o = next_o
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    if return_dict_obs:
        observations = raw_obs
        next_observations = raw_next_obs
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    env_info_final = {}
    for info in env_infos[1:]:
        for k, v in info.items():
            if k in env_info_final:
                env_info_final[k].append(v)
            else:
                env_info_final[k] = [v]
    for k, v in env_info_final.items():
        env_info_final[k] = np.concatenate(v, 1)
    env_infos = env_info_final
    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=np.array(terminals),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
