from rlkit.demos.source.demo_source import DemoSource
import pickle

from rlkit.data_management.path_builder import PathBuilder

from rlkit.util.io import load_local_or_remote_file

class HandDemoSource(DemoSource):
    def __init__(self, filename):
        self.data = load_local_or_remote_file(filename)

    def load_paths(self):
        paths = []
        for i in range(len(self.data)):
            p = self.data[i]
            H = len(p["observations"]) - 1

            path_builder = PathBuilder()

            for t in range(H):
                p["observations"][t]

                ob = path["observations"][t, :]
                action = path["actions"][t, :]
                reward = path["rewards"][t]
                next_ob = path["observations"][t+1, :]
                terminal = 0
                agent_info = {} # todo (need to unwrap each key)
                env_info = {} # todo (need to unwrap each key)

                path_builder.add_all(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    next_observations=next_ob,
                    terminals=terminal,
                    agent_infos=agent_info,
                    env_infos=env_info,
                )

            path = path_builder.get_all_stacked()
            paths.append(path)
        return paths
