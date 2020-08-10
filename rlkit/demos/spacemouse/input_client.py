"""
Should be run on a machine connected to a spacemouse
"""

from robosuite.devices import SpaceMouse
import time
import Pyro4
from rlkit.launchers import config

Pyro4.config.SERIALIZERS_ACCEPTED = set(['pickle','json', 'marshal', 'serpent'])
Pyro4.config.SERIALIZER='pickle'

nameserver = Pyro4.locateNS(host=config.SPACEMOUSE_HOSTNAME)
uri = nameserver.lookup("example.greeting")
device_state = Pyro4.Proxy(uri)
device = SpaceMouse()
while True:
    state = device.get_controller_state()
    print(state)
    time.sleep(0.1)
    device_state.set_state(state)
