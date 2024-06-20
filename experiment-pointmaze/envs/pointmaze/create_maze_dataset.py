import gymnasium

import argparse
import numpy as np
import json
import os
import pickle

from .envs.point_maze import PointMaze
from .utils.maze_utils import set_map_cell

def create_env_dataset(args):
    '''
    Create env and dataset (if not created)
    '''
    maze_config = json.load(open(args.maze_config_file, 'r'))
    maze = maze_config["maze"]
    map = maze['map']  

    start = maze['start']
    goal = maze['goal']

    sample_args = maze_config["sample_args"]

    print(f"Create point maze")
    point_maze = PointMaze(data_path = os.path.join(args.data_dir, args.data_file), 
                        horizon = args.horizon,
                        maze_map = map,
                        start = np.array(start),
                        goal = np.array(goal),
                        sample_args = sample_args,
                        debug=False,
                        render=False)   
    env = point_maze.env_cls()
    trajs = point_maze.dataset[0]
    return env, trajs
