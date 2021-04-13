import multiprocessing as mp

import numpy as np
from d4rl.kitchen.env_dict import ALL_KITCHEN_ENVIRONMENTS
from gym import Env
from metaworld import _encode_task
from metaworld.envs.mujoco.env_dict import ALL_V1_ENVIRONMENTS, ALL_V2_ENVIRONMENTS
from stable_baselines3.common.vec_env import CloudpickleWrapper, SubprocVecEnv, VecEnv

# def make_metaworld_env(env_class, env_kwargs):
#     if env_class.endswith("-v1"):
#         env_dict = ALL_V1_ENVIRONMENTS
#     elif env_class.endswith("-v2"):
#         env_dict = ALL_V2_ENVIRONMENTS
#     env = env_dict[env_class]()
#     env.reset_action_space(**env_kwargs)
#     env.random_init = False
#     task = _encode_task(
#         env_class,
#         {
#             "env_cls": env_dict[env_class],
#             "rand_vec": None,
#             "partially_observable": False,  # if true: goal is part of the observation
#         },
#     )
#     env.set_task(task)
#     return env


def make_metaworld_env(env_name, env_kwargs):
    import metaworld

    if env_name in ALL_V1_ENVIRONMENTS:
        env_cls = ALL_V1_ENVIRONMENTS[env_name]
    else:
        env_cls = ALL_V2_ENVIRONMENTS[env_name]

    env = env_cls()
    env.reset_action_space(**env_kwargs)

    kwargs = {
        "rand_vec": env._last_rand_vec,
        "env_cls": env_cls,
        "partially_observable": False,
    }
    if env_name == "reach-v1" or env_name == "reach-wall-v1":
        kwargs["task_type"] = "reach"
        env._set_task_inner(task_type=kwargs["task_type"])
    elif env_name == "push-v1" or env_name == "push-wall-v1":
        kwargs["task_type"] = "push"
        env._set_task_inner(task_type=kwargs["task_type"])
    elif env_name == "pick-place-v1" or env_name == "pick-place-wall-v1":
        kwargs["task_type"] = "pick_place"
        env._set_task_inner(task_type=kwargs["task_type"])

    rand_vec = env._last_rand_vec
    if rand_vec is None and hasattr(env, "goal"):
        rand_vec = env.goal
    kwargs["rand_vec"] = rand_vec
    env._freeze_rand_vec = True
    env.random_init = False
    env.set_task(metaworld._encode_task(env_name, kwargs))
    return env


def make_kitchen_env(env_class, env_kwargs):
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
