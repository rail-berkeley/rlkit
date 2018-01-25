README last updated on: 01/24/2018

# rlkit
Reinforcement learning framework implemented in PyTorch.

Some implemented algorithms:
 - [Deep Deterministic Policy Gradient (DDPG)](examples/ddpg.py)
 - [(Double) Deep Q-Network (DQN)](examples/dqn_and_double_dqn.py)

To get started, checkout the example scripts, linked above.

## Installation
Install and use the included ananconda environment
```
$ conda env create -f docker/rlkit/rlkit-env.yml
$ source activate rlkit-env
(rlkit-env) $ # Ready to run examples/ddpg_cheetah_no_doodad.py
```
Or if you want you can use the docker image included.

## Visualizing a policy and seeing results
During training, the results will be saved to a file called under
```
LOCAL_LOG_DIR/<exp_prefix>/<foldername>
```
 - `LOCAL_LOG_DIR` is the directory set by `rlkit.launchers.config.LOCAL_LOG_DIR`
 - `<exp_prefix>` is given either to `setup_logger`.
 - `<foldername>` is auto-generated and based off of `exp_prefix`.
 - inside this folder, you should see a file called `params.pkl`. To visualize a policy, run

```
(rlkit-env) $ python scripts/sim_policy LOCAL_LOG_DIR/<exp_prefix>/<foldername>/params.pkl
```

If you have rllab installed, you can also visualize the results
using `rllab`'s viskit, described at
the bottom of [this page](http://rllab.readthedocs.io/en/latest/user/cluster.html)

tl;dr run

```bash
python rllab/viskit/frontend.py LOCAL_LOG_DIR/<exp_prefix>/
```

## Credit
A lot of the coding infrastructure is based on [rllab](https://github.com/rll/rllab).
Also, the serialization and logger code are basically a carbon copy.
