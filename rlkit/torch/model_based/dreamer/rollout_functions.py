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
    full_o_postprocess_func=None,
    reset_callback=None,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    path_length = 0

    o = env.reset()
    agent.reset(o)
    a = np.zeros((env.n_envs, env.action_space.low.size))
    r = np.zeros(env.n_envs)

    observations.append(o)
    rewards.append(r)
    terminals.append([False] * env.n_envs)
    actions.append(a)
    agent_infos.append({})
    env_infos.append({})

    if reset_callback:
        reset_callback(env, agent, o)
    if render:
        img = env.render(mode="rgb_array", imwidth=256, imheight=256)
        cv2.imshow("img", img)
        cv2.waitKey(1)
    while path_length < max_path_length:
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
        terminals=np.array(terminals),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def vec_rollout_low_level_raps(
    env,
    agent,
    max_path_length=np.inf,
    render=False,
    render_kwargs=None,
    preprocess_obs_for_policy_fn=None,
    get_action_kwargs=None,
    num_low_level_actions_per_primitive=100,
    low_level_action_dim=9,
):
    if render_kwargs is None:
        render_kwargs = {}
    if get_action_kwargs is None:
        get_action_kwargs = {}
    if preprocess_obs_for_policy_fn is None:
        preprocess_obs_for_policy_fn = lambda x: x
    low_level_actions = np.zeros(
        (
            env.n_envs,
            (max_path_length * num_low_level_actions_per_primitive) + 1,
            low_level_action_dim,
        ),
        dtype=np.float32,
    )
    high_level_actions = np.zeros(
        (
            env.n_envs,
            (max_path_length * num_low_level_actions_per_primitive) + 1,
            env.action_space.low.shape[0] + 1,  # includes phase variable
        ),
        dtype=np.float32,
    )
    observations = np.zeros(
        (
            env.n_envs,
            (max_path_length * num_low_level_actions_per_primitive) + 1,
            env.observation_space.low.shape[0],
        ),
        dtype=np.uint8,
    )
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    path_length = 0

    o = env.reset()

    agent.reset(o)
    ha = np.zeros((env.n_envs, env.action_space.low.size + 1))
    la = np.zeros((env.n_envs, low_level_action_dim))
    r = np.zeros(env.n_envs)

    observations[:, 0] = o
    high_level_actions[:, 0] = ha
    low_level_actions[:, 0] = la
    rewards.append(r)
    terminals.append([False] * env.n_envs)
    agent_infos.append({})
    env_infos.append({})

    if render:
        img = env.render(mode="rgb_array", imwidth=256, imheight=256)
        cv2.imshow("img", img)
        cv2.waitKey(1)
    for p in range(0, max_path_length):
        ha, agent_info = agent.get_action(o, **get_action_kwargs)
        if (ha[:, : agent.num_primitives].sum(axis=-1) != 1).any():
            argmax = np.argmax(ha[:, : agent.num_primitives], axis=-1)
            one_hots = np.eye(agent.num_primitives)[argmax]
            ha = np.concatenate((one_hots, ha[:, agent.num_primitives :]), axis=-1)

        next_o, r, d, i = env.step(ha)
        rewards.append(r)
        terminals.append(d)
        agent_infos.append(agent_info)
        env_infos.append(i)
        if render:
            img = env.render(mode="rgb_array", imwidth=256, imheight=256)
            cv2.imshow("img", img)
            cv2.waitKey(1)
        ll_as = i["actions"]
        ll_os = i["observations"]
        del i["actions"]
        del i["observations"]
        del i["robot-states"]
        del i["arguments"]
        la = []
        lo = []
        for e in range(env.n_envs):
            # a0 + a1 + ...+a_space-1 -> o_space-1, o_space-1+space
            ll_a = np.array(ll_as[e])
            ll_o = np.array(ll_os[e])

            num_ll = ll_a.shape[0]
            idxs = np.linspace(0, num_ll, num_low_level_actions_per_primitive + 1)
            spacing = num_ll // (num_low_level_actions_per_primitive)
            a = ll_a.reshape(num_low_level_actions_per_primitive, spacing, -1)
            a = a.sum(axis=1)[:, :3]  # just keep sum of xyz deltas
            a = np.concatenate(
                (a, ll_a[idxs.astype(np.int)[1:] - 1, 3:]), axis=1
            )  # try to get last index of each block
            o = ll_o[idxs.astype(np.int)[1:] - 1]  # o[space-1, 2*space-1, ...]
            la.append(a)
            lo.append(o)
        low_level_actions[
            :,
            p * num_low_level_actions_per_primitive
            + 1 : (p + 1) * num_low_level_actions_per_primitive
            + 1,
        ] = np.array(la)
        observations[
            :,
            p * num_low_level_actions_per_primitive
            + 1 : p * num_low_level_actions_per_primitive
            + num_low_level_actions_per_primitive
            + 1,
        ] = (
            np.array(lo)
            .transpose(0, 1, 4, 2, 3)
            .reshape(env.n_envs, num_low_level_actions_per_primitive, -1)
        )
        ha = np.repeat(
            np.array(ha).reshape(next_o.shape[0], 1, -1),
            num_low_level_actions_per_primitive,
            axis=1,
        )
        phases = (
            np.linspace(
                0,
                1,
                num_low_level_actions_per_primitive,
                endpoint=False,
            )
            + 1 / (num_low_level_actions_per_primitive)
        )
        phases = np.repeat(phases.reshape(1, -1), next_o.shape[0], axis=0)
        ha = np.concatenate((ha, np.expand_dims(phases, -1)), axis=2)
        high_level_actions[
            :,
            p * num_low_level_actions_per_primitive
            + 1 : p * num_low_level_actions_per_primitive
            + num_low_level_actions_per_primitive
            + 1,
        ] = ha

        path_length += 1
        if d.all():
            break
        o = next_o
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
        actions=high_level_actions,
        high_level_actions=high_level_actions,
        low_level_actions=low_level_actions,
        rewards=rewards,
        terminals=np.array(terminals),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )
