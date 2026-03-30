#!/bin/bash
# Usage: bash run.sh
#
# Configure the maze type by changing data_dir below:
#   ./rl_umaze        - U-maze environment
#   ./rl_medium_maze  - Medium maze environment
#   ./rl_large_maze   - Large maze environment

data_dir="./rl_medium_maze"
task=pointmaze
horizon=300

for seed in 0
do
python examples/pointmaze/run_dt_maze.py \
    --seed ${seed} \
    --data_dir ${data_dir} \
    --task ${task} \
    --horizon ${horizon}
done
