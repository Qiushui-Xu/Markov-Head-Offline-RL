# ! Change it to your own directory

# data_dir="./rl_umaze"
data_dir="./rl_medium_maze"
# data_dir="./rl_large_maze"

task=pointmaze
horizon=300

# dt
for seed in 0
do
python examples/pointmaze/run_dt_maze.py \
    --seed ${seed} \
    --data_dir ${data_dir} \
    --task ${task} \
    --horizon ${horizon} 
done
