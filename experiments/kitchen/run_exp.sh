#!/bin/bash
for value in {0..1}
do
  CUDA_VISIBLE_DEVICES=$value python experiments/kitchen/dreamer.py --exp_prefix kitchen_fixed_schema_sweep --num_seeds 1 --debug --num_expl_envs 1
done