#!/bin/bash
module load singularity
export PYTHONPATH=$PYTHONPATH:/home/mdalal/research/rlkit/
export PYTHONPATH=$PYTHONPATH:/home/mdalal/research/hrl-exp/
export PYTHONPATH=$PYTHONPATH:/home/mdalal/research/d4rl/
export PYTHONPATH=$PYTHONPATH:/home/mdalal/research/doodad/
singularity exec --nv /home/mdalal/dreamer_v1_latest.sif /home/mdalal/miniconda3/envs/test/bin/python experiments/kitchen/dreamer_learn_schema.py --debug --mode here_no_doodad
#for value in {0..3} # number of GPUs
#do
#  CUDA_VISIBLE_DEVICES=$value python /home/mdalal/research/rlkit/experiments/kitchen/multitask_dreamer.py --exp_prefix kitchen_multitask_fixed_schema_sweep --gpu_id $value --num_seeds 3 --num_expl_envs 4 --num_gpus 4 &
#done

#for value in {0..7} # number of GPUs
#do
#  CUDA_VISIBLE_DEVICES=$value python /home/mdalal/research/rlkit/experiments/kitchen/dreamer.py --exp_prefix kitchen_fixed_schema_sweep --gpu_id $value --num_seeds 3 --num_expl_envs 4 --num_gpus 8 &
#done

#for value in {0..3} # number of GPUs
#do
#  singularity exec --nv /home/mdalal/dreamer_v1_latest.sif python /home/mdalal/research/rlkit/experiments/kitchen/dreamer_learn_schema.py --exp_prefix test --gpu_id 0 --num_seeds 3 --num_expl_envs 4 --debug --num_gpus 1
#done
