import multiprocessing as mp

import gym
import numpy as np
from gym import Env
from stable_baselines3.common.vec_env import CloudpickleWrapper, SubprocVecEnv, VecEnv

from rlkit.envs.wrappers import NormalizedBoxEnv


def make_robosuite_env(env_name, kwargs):
    gym.logger.setLevel(40)
    from robosuite.environments.base import REGISTERED_ENVS, MujocoEnv

    # import robosuite as suite
    from robosuite.wrappers.gym_wrapper import GymWrapper

    from rlkit.envs.primitives_wrappers import DMControlBackendMetaworldRobosuiteEnv

    env_cls = REGISTERED_ENVS[env_name]
    parent = env_cls
    while MujocoEnv != parent.__bases__[0]:
        parent = parent.__bases__[0]

    if parent != DMControlBackendMetaworldRobosuiteEnv:
        parent.__bases__ = (DMControlBackendMetaworldRobosuiteEnv,)
    return NormalizedBoxEnv(GymWrapper(REGISTERED_ENVS[env_name](**kwargs)))


def make_metaworld_env(env_name, env_kwargs=None, use_dm_backend=True):
    env_kwargs_new = env_kwargs.copy()
    del env_kwargs_new["use_image_obs"]
    del env_kwargs_new["reward_scale"]
    del env_kwargs_new["use_dm_backend"]
    env_kwargs = env_kwargs_new
    gym.logger.setLevel(40)
    if env_kwargs is None:
        env_kwargs = {}
    from metaworld.envs.mujoco.env_dict import (
        ALL_V1_ENVIRONMENTS,
        ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
    )
    from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv

    from rlkit.envs.primitives_wrappers import (
        MetaworldWrapper,
        SawyerMocapBaseDMBackendMetaworld,
        SawyerXYZEnvMetaworldPrimitives,
    )

    if env_name in ALL_V1_ENVIRONMENTS:
        env_cls = ALL_V1_ENVIRONMENTS[env_name]
    else:
        env_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]

    # hack from https://stackoverflow.com/questions/38397610/how-to-change-the-base-class-in-python
    # assume linear hierarchy for now
    parent = env_cls
    while SawyerXYZEnv != parent.__bases__[0]:
        parent = parent.__bases__[0]
    if (
        parent != SawyerXYZEnvMetaworldPrimitives
    ):  # ensure if it is called multiple times you don't reset the base class
        parent.__bases__ = (SawyerXYZEnvMetaworldPrimitives,)
    if use_dm_backend:
        SawyerXYZEnv.__bases__ = (SawyerMocapBaseDMBackendMetaworld,)
    if env_name in ALL_V1_ENVIRONMENTS:
        env = env_cls()
        if env_name == "reach-v1" or env_name == "reach-wall-v1":
            env._set_task_inner(task_type="reach")
        elif env_name == "push-v1" or env_name == "push-wall-v1":
            env._set_task_inner(task_type="push")
        elif env_name == "pick-place-v1" or env_name == "pick-place-wall-v1":
            env._set_task_inner(task_type="pick_place")
        env._partially_observable = False
        env.random_init = False
        env._set_task_called = True
    else:
        env = env_cls(seed=42)
    env.reset_action_space(**env_kwargs)
    env.reset()
    env = MetaworldWrapper(env)
    return env


def make_kitchen_env(env_class, env_kwargs):
    from d4rl.kitchen.env_dict import ALL_KITCHEN_ENVIRONMENTS

    env = ALL_KITCHEN_ENVIRONMENTS[env_class](**env_kwargs)
    return env


class DummyVecEnv(Env):
    def __init__(self, envs, pass_render_kwargs=True):
        self.envs = envs
        self.n_envs = len(envs)
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self.pass_render_kwargs = pass_render_kwargs

    def step(
        self,
        actions,
        render_every_step=False,
        render_mode="rgb_array",
        render_im_shape=(1000, 1000),
    ):
        if self.pass_render_kwargs:
            promises = [
                e.step(
                    a,
                    render_every_step=render_every_step,
                    render_mode=render_mode,
                    render_im_shape=render_im_shape,
                )
                for e, a in zip(self.envs, actions)
            ]
        else:
            promises = [
                e.step(
                    a,
                )
                for e, a in zip(self.envs, actions)
            ]
        obs, rewards, done, infos = zip(*[p for p in promises])
        obs = np.stack(obs)
        done = np.stack(done)
        rewards = np.stack(rewards)
        info_ = {}
        for i in infos:
            for k, v in i.items():
                if k in info_.keys():
                    info_[k].append(np.array(v).reshape(1, 1))
                else:
                    info_[k] = [np.array(v).reshape(1, 1)]
        for k, v in info_.items():
            info_[k] = np.concatenate(v)
        return obs, rewards, done, info_

    def reset(self):
        obs = [None] * self.n_envs
        promises = [self.envs[i].reset() for i in range(self.n_envs)]
        for index, promise in zip(range(self.n_envs), promises):
            obs[index] = promise
        return obs

    def render(self, **kwargs):
        return self.envs[0].render(**kwargs)


def _worker(
    remote: mp.connection.Connection,
    parent_remote: mp.connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
) -> None:
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, done, info = env.step(data)
                remote.send((observation, reward, done, info))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "reset":
                observation = env.reset()
                remote.send(observation)
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break


class StableBaselinesVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, start_method=None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        self.n_envs = n_envs

        if start_method is None:
            # Fork is not a thread safe method (see issue #217)
            # but is more user friendly (does not require to wrap the code in
            # a `if __name__ == "__main__":`)
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context("spawn")

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(
            self.work_remotes, self.remotes, env_fns
        ):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(
                target=_worker, args=args, daemon=True
            )  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            work_remote.close()
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step(self, actions):
        obs, rewards, dones, infos = super(StableBaselinesVecEnv, self).step(actions)
        info_ = {}
        for i in infos:
            for k, v in i.items():
                if k in info_.keys():
                    info_[k].append(np.array(v).reshape(1, 1))
                else:
                    info_[k] = [np.array(v).reshape(1, 1)]
        for k, v in info_.items():
            info_[k] = np.concatenate(v)
        return obs, rewards, dones, info_
