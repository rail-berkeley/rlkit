# RLkit
Reinforcement learning framework and algorithms implemented in PyTorch.

## Installation

Setup Dependencies:
```
sudo apt-get update
sudo apt-get install curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev
sudo apt-get install libglfw3-dev libgles2-mesa-dev patchelf
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz
tar -xvf mujoco-2.1.1-linux-x86_64.tar.gz
```

Add the following to your bashrc:
```
export MUJOCO_GL='egl'
export MKL_THREADING_LAYER=GNU
export D4RL_SUPPRESS_IMPORT_ERROR='1'
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mdalal/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

Setup Directories:
```
mkdir ~/research/<project_name>/
cd ~/research/<project_name>/
git clone git@github.com:mihdalal/d4rl.git
git clone git@github.com:mihdalal/doodad.git
git clone git@github.com:mihdalal/metaworld.git
git clone git@github.com:mihdalal/rlkit.git
git clone git@github.com:mihdalal/robosuite.git
git clone git@github.com:mihdalal/viskit.git
```

Install Ananconda environment
```
conda create -n skill_learn python=3.8
source activate skill_learn
```

Install packages
```
cd d4rl
git checkout skill_learn
pip install -e .
cd ../doodad
pip install -r requirements.txt
pip install -e .
cd ../metaworld
git checkout torque_mods
pip install -e .
cd ../robosuite
pip install -r requirements-extra.txt
pip install -e requirements.txt
pip install -e .
cd ../viskit
pip install -e .
cd ../rlkit
pip install -r requirements.txt
pip install -e .
```

3. (Optional) Copy `conf.py` to `conf_private.py` and edit to override defaults:
```
cp rlkit/launchers/conf.py rlkit/launchers/conf_private.py
```

## Example Commands:
Debug:

`python /path/to/experiment.py --debug`

Replicate RAPS Kitchen results:

`python experiments/ll_raps/dreamer_v2_single_task_primitives_kitchen.py -sk env_name -sv hinge_cabinet microwave kettle light_switch top_left_burner -st str --mode ssm --exp_prefix raps_kitchen_replicate --num_seeds 5`

Replicate RAPS Metaworld results:

`python experiments/ll_raps/dreamer_v2_single_task_primitives.py -sk env_name -sv assembly-v2 disassemble-v2 soccer-v2 sweep-into-v2 -st str --mode ssm --exp_prefix raps_mw_replicate --num_seeds 5`

LLRAPS Kitchen:

`python experiments/ll_raps/dreamer_v2_single_task_low_level_primitives_kitchen_raps_params.py -sk env_name -sv microwave hinge_cabinet top_left_burner kettle light_switch -st str --num_seeds 5 --mode ssm --exp_prefix ll_raps_kitchen_replicate`

LLRAPS Metaworld:

`python experiments/ll_raps/dreamer_v2_single_task_low_level_primitives_raps_params.py -sk env_name -sv assembly-v2 disassemble-v2 soccer-v2 sweep-into-v2 -st str --num_seeds 5 --mode ssm --exp_prefix ll_raps_mw_refactor_replicate`

## How to run sweeps from command line
`python /path/to/experiment.py -sk key1 key2 key3 -sv v11 v12 ... -sv v21 v22 ... -sv v31 v32 ... -st k1type k2type k3type`

Example:

`python experiments/ll_raps/dreamer_v2_single_task_low_level_primitives_raps_params.py -sk env_name num_low_level_actions_per_primitive -sv assembly-v2 disassemble-v2 soccer-v2 sweep-into-v2 -sv 5 10 -st str int`

Basically, for each key, add `-sv <list of values>`

## Visualizing results
During training, the results will be saved to a file called under
```
LOCAL_LOG_DIR/<exp_prefix>/<foldername>
```
 - `LOCAL_LOG_DIR` is the directory set by `rlkit.launchers.config.LOCAL_LOG_DIR`. Default name is 'results/'.
 - `<exp_prefix>` is given either to `setup_logger`.
 - `<foldername>` is auto-generated and based off of `exp_prefix`.

To visualize graphs of the results:
```
python viskit/viskit/frontend.py LOCAL_LOG_DIR/<exp_prefix>/<foldername>
```

you can add an alias of the form to your `~/.aliases` file:
```
alias vis='python viskit/viskit/frontend.py'
```

## Launching jobs with `doodad`
The `run_experiment` function makes it easy to run Python code on Amazon Web Services (AWS) or Google Cloud Platform (GCP) or Slurm by using
[this fork of doodad](git@github.com:mihdalal/doodad.git).

It's as easy as:
```
from rlkit.launchers.launcher_util import run_experiment

def function_to_run(variant):
    learning_rate = variant['learning_rate']
    ...

run_experiment(
    function_to_run,
    exp_prefix="my-experiment-name",
    mode='ssm',  # or 'ec2'
    variant={'learning_rate': 1e-3},
)
