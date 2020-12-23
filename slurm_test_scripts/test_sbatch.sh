#!/bin/bash
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mdalal/.mujoco/mujoco200/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-000
# source ~/.bashrc
# echo $LD_LIBRARY_PATH
# /opt/singularity/bin/singularity exec --nv ~/dreamer_v1_latest.sif /home/mdalal/miniconda3/envs/hrl-exp-env/bin/python ~/research/hrl-exp/examples/kitchen/run_kitchen_multitask.py
/opt/singularity/bin/singularity exec --nv ~/dreamer_v1_latest.sif /home/mdalal/research/rlkit/slurm_test_scripts/test.sh
# /opt/singularity/bin/singularity exec --nv ~/dreamer_v1_latest.sif /home/mdalal/miniconda3/envs/hrl-exp-env/bin/python ~/research/hrl-exp/examples/kitchen/run_kitchen_multitask.py
# singularity exec --nv ~/dreamer_v1_latest.sif /home/mdalal/miniconda3/envs/hrl-exp-env/bin/python ~/research/rlkit/test.py
