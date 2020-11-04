#!/bin/bash
for value in {0..0} # number of GPUs
do
  CUDA_VISIBLE_DEVICES=$value python experiments/kitchen/dreamer.py --exp_prefix kitchen_fixed_schema_sweep \
  --num_seeds 3 --debug --num_expl_envs 1 --tmux --tmux_session_name research-0
done