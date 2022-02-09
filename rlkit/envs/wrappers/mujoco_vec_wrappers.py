import multiprocessing as mp
import os
import pickle

import numpy as np
from gym import Env
from stable_baselines3.common.vec_env import CloudpickleWrapper, SubprocVecEnv, VecEnv


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
        render_im_shape=(64, 64),
    ):
        if self.pass_render_kwargs:
            promises = [
                env.step(
                    action,
                    render_every_step=render_every_step,
                    render_mode=render_mode,
                    render_im_shape=render_im_shape,
                )
                for env, action in zip(self.envs, actions)
            ]
        else:
            promises = [
                env.step(
                    action,
                )
                for env, action in zip(self.envs, actions)
            ]
        obs, rewards, done, infos = zip(*[promise for promise in promises])
        obs = np.stack(obs)
        done = np.stack(done)
        rewards = np.stack(rewards)
        new_info = {}
        for info in infos:
            for key, value in info.items():
                value = np.array(value)
                if key in new_info.keys():
                    if value.shape != ():
                        new_info[key].append(value.reshape(1, *np.array(value).shape))
                    else:
                        new_info[key].append(value.reshape(1, 1))
                else:
                    if value.shape != ():
                        new_info[key] = [value.reshape(1, *np.array(value).shape)]
                    else:
                        new_info[key] = [value.reshape(1, 1)]
        for key, value in new_info.items():
            new_info[key] = np.concatenate(value)
        return obs, rewards, done, new_info

    def reset(self):
        obs = [None] * self.n_envs
        promises = [self.envs[env_idx].reset() for env_idx in range(self.n_envs)]
        for index, promise in zip(range(self.n_envs), promises):
            obs[index] = promise
        return obs

    def render(self, **kwargs):
        return self.envs[0].render(**kwargs)

    def save(self, path, suffix):
        pickle.dump(self, open(os.path.join(path, suffix), "wb"))

    def load(self, path, suffix):
        return pickle.load(open(os.path.join(path, suffix), "rb"))


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
    def __init__(self, env_fns, start_method=None, reload_state_args=None):
        self.waiting = False
        self.closed = False
        self.reload_state_args = reload_state_args
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
        new_info = {}
        for info in infos:
            for key, value in info.items():
                if key in new_info.keys():
                    if np.isscalar(value):
                        new_info[key].append(np.array(value).reshape(1, 1))
                    else:
                        new_info[key].append(value)
                else:
                    if np.isscalar(value):
                        new_info[key] = [np.array(value).reshape(1, 1)]
                    else:
                        new_info[key] = [value]
        for key, value in new_info.items():
            if np.array(value).shape == (1, 1):
                new_info[key] = np.concatenate(value)
            else:
                new_info[key] = value
        return obs, rewards, dones, new_info

    def save(self, path, suffix):
        n_envs, make_env, make_env_args = self.reload_state_args
        reload_state_dict = dict()
        reload_state_dict["n_envs"] = n_envs
        reload_state_dict["make_env"] = make_env
        reload_state_dict["make_env_args"] = make_env_args
        pickle.dump(reload_state_dict, open(os.path.join(path, suffix), "wb"))

    def load(self, path, suffix):
        """
        Since we cannot pickle a vec env directly, we have to rebuild it from the env_kwargs.
        NOTE: This will lose saved elements of the vec env / reset them.
        """
        reload_state_dict = pickle.load(open(os.path.join(path, suffix), "rb"))
        make_env = reload_state_dict["make_env"]
        n_envs = reload_state_dict["n_envs"]
        make_env_args = reload_state_dict["make_env_args"]
        del reload_state_dict["make_env"]
        del reload_state_dict["n_envs"]
        del reload_state_dict["make_env_args"]

        env_fns = [lambda: make_env(*make_env_args) for _ in range(n_envs)]
        return StableBaselinesVecEnv(env_fns=env_fns, start_method="fork")
