export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=0

lr=1e-4 # default is 1e-4
weight_decay=1e-5 # default is 1e-4
dropout=0.1
warmup_steps=2500 # default is 10000
num_steps_per_iter=2500 # default is 2500
max_iters=60 # default is 40
num_eval_episodes=20 # default is 100

env=${1}
K=20
dataset=${2}
sample_ratio=${3}
pretrained_lm="gpt2"
description=${4}
seed=${5}
description="${pretrained_lm}_pretrained-ratio=${sample_ratio}_${description}"
gpu=${6}
outdir="checkpoints/${env}_${dataset}_${description}_${seed}"
h_id=3

CUDA_VISIBLE_DEVICES=${gpu} python experiment.py --env ${env} \
        --dataset ${dataset} \
        --seed ${seed} \
        --K ${K} \
        --lr ${lr} \
        --num_steps_per_iter ${num_steps_per_iter} \
        --weight_decay ${weight_decay} \
        --max_iters ${max_iters} \
        --num_eval_episodes ${num_eval_episodes} \
        --sample_ratio ${sample_ratio} \
        --warmup_steps ${warmup_steps} \
        --pretrained_lm ${pretrained_lm} \
	    --hidden_index ${h_id} \
        --outdir ${outdir} \
        --dropout ${dropout} \
        --description ${description} \
        --position_embed \
        # --log_to_wandb \
        # --reinit_markov_head \
