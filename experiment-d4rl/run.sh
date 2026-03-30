#!/bin/bash
# Usage: bash run.sh <env> <dataset> <sample_ratio> <description> <seed> <gpu_id>
#
# Examples:
#   bash run.sh hopper medium 1.0 baseline 42 0
#   bash run.sh halfcheetah medium-replay 0.1 low_data 0 1
#   bash run.sh walker2d medium-expert 1.0 full_data 123 0
#
# Arguments:
#   env:          hopper | halfcheetah | walker2d
#   dataset:      medium | medium-replay | medium-expert
#   sample_ratio: fraction of data to use (1.0 = full dataset)
#   description:  experiment description tag
#   seed:         random seed
#   gpu_id:       CUDA device ID

export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=0

lr=1e-4
weight_decay=1e-5
dropout=0.1
warmup_steps=2500
num_steps_per_iter=2500
max_iters=40
num_eval_episodes=20

env=${1}
if [ "$env" == "reacher2d" ]; then
    K=5
else
    K=20
fi
dataset=${2}
sample_ratio=${3}
pretrained_lm="gpt2"
description=${4}
seed=${5}
description="${pretrained_lm}_pretrained-ratio=${sample_ratio}_${description}"
gpu=${6}
outdir="checkpoints/${env}_${dataset}_${description}_${seed}"

CUDA_VISIBLE_DEVICES=${gpu} python experiment.py --env ${env} \
        --dataset ${dataset} \
        --seed ${seed} \
        --K ${K} \
        -lr ${lr} \
        --num_steps_per_iter ${num_steps_per_iter} \
        --weight_decay ${weight_decay} \
        --max_iters ${max_iters} \
        --num_eval_episodes ${num_eval_episodes} \
        --sample_ratio ${sample_ratio} \
        --warmup_steps ${warmup_steps} \
        --pretrained_lm ${pretrained_lm} \
        --mlp_embedding \
        --outdir ${outdir} \
        --dropout ${dropout} \
        --description ${description} \
        --log_to_wandb
