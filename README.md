README last updated on: 02/19/2018

# rlkit
Reinforcement learning framework and algorithms implemented in PyTorch.

Some implemented algorithms:
 - Temporal Difference Models (TDMs)
    - [example script](examples/tdm/cheetah.py)
    - [TDM paper](https://openreview.net/forum?id=HkLRnW-CW)
    - [Details on implementation](rlkit/torch/tdm/TDMs.md)
 - Deep Deterministic Policy Gradient (DDPG)
    - [example script](examples/ddpg.py)
    - [DDPG paper](https://arxiv.org/pdf/1509.02971.pdf)
 - (Double) Deep Q-Network (DQN)
    - [example script](examples/dqn_and_double_dqn.py)
    - [DQN paper](https://arxiv.org/pdf/1509.06461.pdf)
    - [Double Q-learning paper](https://arxiv.org/pdf/1509.06461.pdf)
 - Soft Actor Critic (SAC)
    - [example script](examples/sac.py)
    - [SAC paper](https://drive.google.com/file/d/0Bxz3x8U2LH_2QllDZVlUQ1BJVEJHeER2YU5mODNaeFZmc3dz/view)
    - [TensorFlow implementation from author](https://github.com/haarnoja/sac)

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

## Algorithm-Specific Comments
### SAC
The SAC implementation provided here only uses Gaussian policy, rather than a Gaussian mixture model, as described in the original SAC paper.

## Credits
A lot of the coding infrastructure is based on [rllab](https://github.com/rll/rllab).
The serialization and logger code are basically a carbon copy of the rllab versions.
