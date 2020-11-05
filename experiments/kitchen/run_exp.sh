#!/bin/bash
for value in {0..1} # number of GPUs
do
  echo $value
  CUDA_VISIBLE_DEVICES=$value python /home/mdalal/research/rlkit/experiments/kitchen/dreamer.py --exp_prefix test_sbatch --gpu_id $value --num_seeds 3 --debug --num_expl_envs 1 --tmux_session_name 0 --num_gpus 4
done
