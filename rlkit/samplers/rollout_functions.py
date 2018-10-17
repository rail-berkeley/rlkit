import numpy as np


def create_rollout_function(rollout_function, **initial_kwargs):
    """
    initial_kwargs for
        rollout_function=tdm_rollout_function may contain:
            init_tau,
            decrement_tau,
            cycle_tau,
            get_action_kwargs,
            observation_key,
            desired_goal_key,
        rollout_function=multitask_rollout may contain:
            observation_key,
            desired_goal_key,
    """

    def wrapped_rollout_func(*args, **dynamic_kwargs):
        combined_args = {
            **initial_kwargs,
            **dynamic_kwargs
        }
        return rollout_function(*args, **combined_args)

    return wrapped_rollout_func


def multitask_rollout(
        env,
        agent,
        max_path_length=np.inf,
        animated=False,
        observation_key=None,
        desired_goal_key=None,
):
    full_observations = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    next_observations = []
    path_length = 0
    agent.reset()
    o = env.reset()
    if animated:
        env.render()
    goal = env.get_goal()
    if desired_goal_key:
        goal = goal[desired_goal_key]
    while path_length < max_path_length:
        full_observations.append(o)
        if observation_key:
            o = o[observation_key]
        new_obs = np.hstack((o, goal))
        a, agent_info = agent.get_action(new_obs)
        next_o, r, d, env_info = env.step(a)
        if animated:
            env.render()
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        next_observations.append(next_o)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
    full_observations.append(o)
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        goals=np.repeat(goal[None], path_length, 0),
        full_observations=full_observations,
    )


def rollout(env, agent, max_path_length=np.inf, animated=False):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param animated:
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    next_o = None
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def tdm_rollout(
        env,
        agent: UniversalPolicy,
        max_path_length=np.inf,
        animated=False,
        init_tau=0.0,
        decrement_tau=False,
        cycle_tau=False,
        get_action_kwargs=None,
        observation_key=None,
        desired_goal_key=None,
):
    full_observations = []
    from railrl.state_distance.rollout_util import _expand_goal
    if get_action_kwargs is None:
        get_action_kwargs = {}
    observations = []
    next_observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    taus = []
    agent.reset()
    path_length = 0
    if animated:
        env.render()

    tau = np.array([init_tau])
    o = env.reset()
    goal = env.get_goal()
    agent_goal = goal
    if desired_goal_key:
        agent_goal = agent_goal[desired_goal_key]
    while path_length < max_path_length:
        full_observations.append(o)
        agent_o = o
        if observation_key:
            agent_o = agent_o[observation_key]

        a, agent_info = agent.get_action(agent_o, agent_goal, tau,
                                         **get_action_kwargs)
        if animated:
            env.render()
        next_o, r, d, env_info = env.step(a)
        next_observations.append(next_o)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        taus.append(tau.copy())
        path_length += 1
        if decrement_tau:
            tau -= 1
        if tau < 0:
            if cycle_tau:
                tau = np.array([init_tau])
            else:
                tau = np.array([0])
        if d:
            break
        o = next_o
    full_observations.append(o)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)

    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=np.array(agent_infos),
        env_infos=np.array(env_infos),
        num_steps_left=np.array(taus),
        goals=_expand_goal(agent_goal, len(terminals)),
        full_observations=full_observations,
    )
