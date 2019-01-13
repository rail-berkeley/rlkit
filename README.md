# rlkit
Reinforcement learning framework and algorithms implemented in PyTorch.

Some implemented algorithms:
 - Reinforcement Learning with Imagined Goals (RIG)
    - [example script](examples/rig/pusher/rig.py)
    - [RIG paper](https://arxiv.org/abs/1807.04742)
    - [Documentation](docs/RIG.md)
 - Temporal Difference Models (TDMs)
    - [example script](examples/tdm/cheetah.py)
    - [TDM paper](https://arxiv.org/abs/1802.09081)
    - [Documentation](docs/TDMs.md)
 - Hindsight Experience Replay (HER)
    - [example script](examples/her/her_td3_gym_fetch_reach.py)
    - [HER paper](https://arxiv.org/abs/1707.01495)
    - [Documentation](docs/HER.md)
 - Deep Deterministic Policy Gradient (DDPG)
    - [example script](examples/ddpg.py)
    - [DDPG paper](https://arxiv.org/pdf/1509.02971.pdf)
 - (Double) Deep Q-Network (DQN)
    - [example script](examples/dqn_and_double_dqn.py)
    - [DQN paper](https://arxiv.org/pdf/1509.06461.pdf)
    - [Double Q-learning paper](https://arxiv.org/pdf/1509.06461.pdf)
 - (Twin) Soft Actor Critic (SAC)
    - [example script](examples/tsac.py)
    - [SAC paper](https://arxiv.org/abs/1801.01290)
    - [TensorFlow implementation from author](https://github.com/rail-berkeley/softlearning)
    - Includes the "min of Q" method and the entropy-constrained implementation
 - Twin Delayed Deep Determinstic Policy Gradient (TD3)
    - [example script](examples/td3.py)
    - [TD3 paper](https://arxiv.org/abs/1802.09477)
 - (Non-Twin/Old) Soft Actor Critic
    - [example script](examples/sac.py)
    - SAC without the "min of Q" method.
    - The canonical SAC implementation is the twin version, listed earlier.

To get started, checkout the example scripts, linked above.

## What's New
12/04/2018
 - Add RIG implementation

12/03/2018
 - Add HER implementation
 - Add doodad support

10/16/2018
 - Upgraded to PyTorch v0.4
 - Added Twin Soft Actor Critic Implementation
 - Various small refactor (e.g. logger, evaluate code)

## Installation
1. Copy `config_template.py` to `config.py`:
```
cp rlkit/launchers/config_template.py rlkit/launchers/config.py
```
2. Install and use the included Ananconda environment
```
$ conda env create -f environment/[linux-cpu|linux-gpu|mac]-env.yml
$ source activate rlkit
(rlkit) $ python examples/ddpg.py
```
Choose the appropriate `.yml` file for your system.
These Anaconda environments use MuJoCo 1.5 and gym 0.10.5.
You'll need to [get your own MuJoCo key](https://www.roboti.us/license.html) if you want to use MuJoCo.

DISCLAIMER: the mac environment has only been tested without a GPU.

For an even more portable solution, try using the docker image provided in `environment/docker`.
The Anaconda env should be enough, but this docker image addresses some of the rendering issues that may arise when using MuJoCo 1.5 and GPUs.
The docker image supports GPU, but it should work without a GPU.
To use a GPU with the image, you need to have [nvidia-docker installed](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)).

## Using a GPU
You can use a GPU by calling
```
import rlkit.torch.pytorch_util as ptu
ptu.set_gpu_mode(True)
```
before launching the scripts.

If you are using `doodad` (see below), simply use the `use_gpu` flag:
```
run_experiment(..., use_gpu=True)
```

## Visualizing a policy and seeing results
During training, the results will be saved to a file called under
```
LOCAL_LOG_DIR/<exp_prefix>/<foldername>
```
 - `LOCAL_LOG_DIR` is the directory set by `rlkit.launchers.config.LOCAL_LOG_DIR`. Default name is 'output'.
 - `<exp_prefix>` is given either to `setup_logger`.
 - `<foldername>` is auto-generated and based off of `exp_prefix`.
 - inside this folder, you should see a file called `params.pkl`. To visualize a policy, run

```
(rlkit) $ python scripts/sim_policy.py LOCAL_LOG_DIR/<exp_prefix>/<foldername>/params.pkl
```

If you have rllab installed, you can also visualize the results
using `rllab`'s viskit, described at
the bottom of [this page](http://rllab.readthedocs.io/en/latest/user/cluster.html)

tl;dr run

```bash
python rllab/viskit/frontend.py LOCAL_LOG_DIR/<exp_prefix>/
```
to visualize all experiments with a prefix of `exp_prefix`. To only visualize a single run, you can do
```bash
python rllab/viskit/frontend.py LOCAL_LOG_DIR/<exp_prefix>/<folder name>
```

Alternatively, if you don't want to clone all of `rllab`, a repository containing only viskit can be found [here](https://github.com/vitchyr/viskit). You can similarly visualize results with.
```bash
python viskit/viskit/frontend.py LOCAL_LOG_DIR/<exp_prefix>/
```
This `viskit` repo also has a few extra nice features, like plotting multiple Y-axis values at once, figure-splitting on multiple keys, and being able to filter hyperparametrs out.

## Visualizing a TDM/HER policy
To visualize a TDM policy, run
```
(rlkit) $ python scripts/sim_tdm_policy.py LOCAL_LOG_DIR/<exp_prefix>/<foldername>/params.pkl
```
To visualize a HER policy, run
```
(rlkit) $ python scripts/sim_goal_conditioned_policy.py
LOCAL_LOG_DIR/<exp_prefix>/<foldername>/params.pkl
```

## Launching jobs with `doodad`
The `run_experiment` function makes it easy to run Python code on Amazon Web
Services (AWS) or Google Cloud Platform (GCP) by using
[doodad](https://github.com/justinjfu/doodad/).

It's as easy as:
```
from rlkit.launchers.launcher_util import run_experiment

def function_to_run(variant):
    learning_rate = variant['learning_rate']
    ...

run_experiment(
    function_to_run,
    exp_prefix="my-experiment-name",
    mode='ec2',  # or 'gcp'
    variant={'learning_rate': 1e-3},
)
```
You will need to set up parameters in config.py (see step one of Installation).
This requires some knowledge of AWS and/or GCP, which is beyond the scope of
this README.
To learn more, more about `doodad`, [go to the repository](https://github.com/justinjfu/doodad/).

## Credits
A lot of the coding infrastructure is based on [rllab](https://github.com/rll/rllab).
The serialization and logger code are basically a carbon copy of the rllab versions.

The Dockerfile is based on the [OpenAI mujoco-py Dockerfile](https://github.com/openai/mujoco-py/blob/master/Dockerfile).

## TODOs
 - Include policy-gradient algorithms.
 - Include model-based algorithms.
