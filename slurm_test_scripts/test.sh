#!/bin/bash
# source ~/.bashrc
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mdalal/.mujoco/mujoco200/bin
# /home/mdalal/miniconda3/envs/hrl-exp-env/bin/python ~/research/hrl-exp/examples/kitchen/run_kitchen_multitask.py
# /home/mdalal/miniconda3/envs/hrl-exp-env/bin/python ~/research/rlkit/test2.py
export MUJOCO_GL='egl'
/home/mdalal/miniconda3/envs/hrl-exp-env/bin/python ~/research/rlkit/experiments/kitchen/dreamer_v2_learn_schema.py --num_expl_envs 6 --debug