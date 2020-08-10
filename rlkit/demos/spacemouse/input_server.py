# first to start the nameserver start: python -m Pyro4.naming

import Pyro4
from threading import Thread
import time
import numpy as np
from rlkit.launchers import conf as config

Pyro4.config.SERIALIZERS_ACCEPTED = set(['pickle','json', 'marshal', 'serpent'])
Pyro4.config.SERIALIZER='pickle'

device_state = None

@Pyro4.expose
class DeviceState(object):
    state = None

    def get_state(self):
        return device_state

    def set_state(self, state):
        global device_state
        device_state = state

class SpaceMouseExpert:
    def __init__(
            self,
            xyz_dims=3,
            xyz_remap=[0, 1, 2],
            xyz_scale=[1, 1, 1],
            xyz_abs_threshold=0.0,
            rot_dims=3,
            rot_remap=[0, 1, 2],
            rot_scale=[1, 1, 1],
            rot_abs_threshold=0.0,
            rot_discrete=False,
            min_clip=-np.inf,
            max_clip=np.inf
    ):
        """TODO: fill in other params"""
        self.xyz_dims = xyz_dims
        self.xyz_remap = np.array(xyz_remap)
        self.xyz_scale = np.array(xyz_scale)
        self.xyz_abs_threshold = xyz_abs_threshold
        self.rot_dims = rot_dims
        self.rot_remap = rot_remap
        self.rot_scale = rot_scale
        self.rot_abs_threshold = rot_abs_threshold
        self.rot_discrete = rot_discrete
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.thread = Thread(target = start_server)
        self.thread.daemon = True
        self.thread.start()
        self.device_state = DeviceState()

    def get_action(self, obs):
        """Must return (action, valid, reset, accept)"""
        state = self.device_state.get_state()
        # time.sleep(0.1)
        if state is None:
            return None, False, False, False

        dpos, rotation, roll, pitch, yaw, accept, reset = (
            state["dpos"],
            state["rotation"],
            state["roll"],
            state["pitch"],
            state["yaw"],
            state["grasp"], #["left_click"],
            state["reset"], #["right_click"],
        )

        xyz = dpos[self.xyz_remap]
        xyz[np.abs(xyz) < self.xyz_abs_threshold] = 0.0
        xyz = xyz * self.xyz_scale
        xyz = np.clip(xyz, self.min_clip, self.max_clip)

        rot = np.array([roll, pitch, yaw])
        rot[np.abs(rot) < self.rot_abs_threshold] = 0.0
        if self.rot_discrete:
            max_i = np.argmax(np.abs(rot))
            for i in range(len(rot)):
                if i != max_i:
                    rot[i] = 0.0
        rot = rot * self.rot_scale
        rot = np.clip(rot, self.min_clip, self.max_clip)

        a = np.concatenate([xyz[:self.xyz_dims], rot[:self.rot_dims]])

        valid = not np.all(np.isclose(a, 0))

        # print(a, roll, pitch, yaw, valid)

        return (a, valid, reset, accept)


def start_server():
    daemon = Pyro4.Daemon(config.SPACEMOUSE_HOSTNAME)
    ns = Pyro4.locateNS()                  # find the name server
    uri = daemon.register(DeviceState)   # register the greeting maker as a Pyro object
    ns.register("example.greeting", uri)   # register the object with a name in the name server
    print("uri:", uri)
    print("Server ready.")
    daemon.requestLoop()                   # start the event loop of the server to wait for calls

if __name__ == "__main__":
    expert = SpaceMouseExpert()

    for i in range(100):
        time.sleep(1)
        print(expert.get_action(None))
