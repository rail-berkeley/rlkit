# Spacemouse Demonstrations

This code allows an agent (including physical robots) to be controlled by a 7 degree-of-freedom 3Dconnexion Spacemouse device.

## Setup Instructions

### Spacemouse (on a Mac)

1. Clone robosuite ([https://github.com/anair13/robosuite](https://github.com/anair13/robosuite)) and add it to the python path
2. Run `pip install hidapi` and install Spacemouse drivers
2. Ensure you can run the following file (and
see input values from the spacemouse): `robosuite/devices/spacemouse.py`
4. Follow the example in `railrl/demos/collect_demo.py` to collect demonstrations

### Server
We haven't been able to install Spacemouse drivers for Linux but instead we use a Spacemouse on a Mac ("client") and send messages over a network to a Linux machine ("server").

#### Setup
On the client, run the setup above. On the server, run:
1. Run `pip install Pyro4`
2. Make sure the hostname in `railrl/demos/spacemouse/config.py` is correct (in the example I use gauss1.banatao.berkeley.edu). This hostname needs to be visible (eg. you can ping it) from both the client and server

#### Run

1. On the server, start the nameserver:
```export PYRO_SERIALIZERS_ACCEPTED=serpent,json,marshal,pickle
python -m Pyro4.naming -n euler1.dyn.berkeley.edu
```
2. On the server, run a script that uses the `SpaceMouseExpert` imported from `railrl/demos/spacemouse/input_server.py` such as ```python experiments/ashvin/iros2019/collect_demos_spacemouse.py```
2. On the client, run ```python railrl/demos/spacemouse/input_client.py```
