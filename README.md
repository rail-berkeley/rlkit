# rlkit
Reinforcement learning framework and algorithms implemented in PyTorch.

Implemented algorithms:
 - Skew-Fit
    - [example script](examples/skewfit/sawyer_door.py)
    - [paper](https://arxiv.org/abs/1903.03698)
    - [Documentation](examples/skewfit/sawyer_door.py)
 - Reinforcement Learning with Imagined Goals (RIG)
    - Special case of Skew-Fit: set power = 0
    - [paper](https://arxiv.org/abs/1807.04742)
 - Temporal Difference Models (TDMs)
    - Only implemented in v0.1.2-. See Legacy Documentation section below.
    - [paper](https://arxiv.org/abs/1802.09081)
    - [Documentation](docs/TDMs.md)
 - Hindsight Experience Replay (HER)
    - [example script](examples/her/her_sac_gym_fetch_reach.py)
    - [paper](https://arxiv.org/abs/1707.01495)
    - [Documentation](docs/HER.md)
 - (Double) Deep Q-Network (DQN)
    - [example script](examples/dqn_and_double_dqn.py)
    - [paper](https://arxiv.org/pdf/1509.06461.pdf)
    - [Double Q-learning paper](https://arxiv.org/pdf/1509.06461.pdf)
 - Soft Actor Critic (SAC)
    - [example script](examples/sac.py)
    - [original paper](https://arxiv.org/abs/1801.01290) and [updated
    version](https://arxiv.org/abs/1812.05905)
    - [TensorFlow implementation from author](https://github.com/rail-berkeley/softlearning)
    - Includes the "min of Q" method, the entropy-constrained implementation,
     reparameterization trick, and numerical tanh-Normal Jacbian calcuation.
 - Twin Delayed Deep Determinstic Policy Gradient (TD3)
    - [example script](examples/td3.py)
    - [paper](https://arxiv.org/abs/1802.09477)

To get started, checkout the example scripts, linked above.

## What's New
### Version 0.2

#### 04/05/2019

The initial release for 0.2 has the following major changes:
 - Remove `Serializable` class and use default pickle scheme.
 - Remove `PyTorchModule` class and use native `torch.nn.Module` directly.
 - Switch to batch-style training rather than online training.
   - Makes code more amenable to parallelization.
   - Implementing the online-version is straightforward.
 - Refactor training code to be its own object, rather than being integrated 
 inside of `RLAlgorithm`.
 - Refactor sampling code to be its own object, rather than being integrated
 inside of `RLAlgorithm`.
 - Implement [Skew-Fit: 
State-Covering Self-Supervised Reinforcement Learning](https://arxiv.org/abs/1903.03698),
a method for performing goal-directed exploration to maximize the entropy of 
visited states.
 - Update soft actor-critic to more closely match TensorFlow implementation:
   - Rename `TwinSAC` to just `SAC`.
   - Only have Q networks.
   - Remove unnecessary policy regualization terms.
   - Use numerically stable Jacobian computation.

Overall, the refactors are intended to make the code more modular and 
readable than the previous versions.

### Version 0.1
#### 12/04/2018
 - Add RIG implementation

#### 12/03/2018
 - Add HER implementation
 - Add doodad support

#### 10/16/2018
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
(rlkit) $ python scripts/run_policy.py LOCAL_LOG_DIR/<exp_prefix>/<foldername>/params.pkl
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

## Visualizing a goal-conditioned policy
To visualize a goal-conditioned policy, run
```
(rlkit) $ python scripts/run_goal_conditioned_policy.py
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

## TODOs/Pull-Request requests
 - Implement policy-gradient algorithms.
 - Implement model-based algorithms.

# Legacy Code (v0.1.2)
For Temporal Difference Models (TDMs) and the original implementation of 
Reinforcement Learning with Imagined Goals (RIG), do
`git checkout tags/v0.1.2`.
