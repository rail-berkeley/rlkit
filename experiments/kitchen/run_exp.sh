#!/bin/bash
#for value in {0..3} # number of GPUs
#do
#  CUDA_VISIBLE_DEVICES=$value python /home/mdalal/research/rlkit/experiments/kitchen/multitask_dreamer.py --exp_prefix kitchen_multitask_fixed_schema_sweep --gpu_id $value --num_seeds 3 --num_expl_envs 4 --num_gpus 4 &
#done

#for value in {0..7} # number of GPUs
#do
#  CUDA_VISIBLE_DEVICES=$value python /home/mdalal/research/rlkit/experiments/kitchen/dreamer.py --exp_prefix kitchen_fixed_schema_sweep --gpu_id $value --num_seeds 3 --num_expl_envs 4 --num_gpus 8 &
#done

for value in {0..3} # number of GPUs
do
  CUDA_VISIBLE_DEVICES=$value python /home/mdalal/research/rlkit/experiments/kitchen/dreamer_learn_schema.py --exp_prefix kitchen_learn_schema_sweep --gpu_id $value --num_seeds 3 --num_expl_envs 4 --num_gpus 4 &
done
