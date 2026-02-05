import h5py
import numpy as np
import d3rlpy
from huggingface_sb3 import EnvironmentName
from rl_zoo3.enjoy import create_test_env
from rl_zoo3.exp_manager import ExperimentManager
import gymnasium as gym
import ale_py
import os
import sys
from pathlib import Path
Path_root = Path(os.path.dirname(os.path.abspath(__file__))).parent
sys.path.insert(0, str(Path_root))
from afr.collect_atari_data import make_env
import torch

data_path = '/u/mrudolph/documents/rl-baselines3-zoo/atari_data/a2c_QbertNoFrameskip-v4_0_100000.pth'

data = torch.load(data_path, weights_only=False)

observations = data['obs']
actions = data['action']
rewards = data['reward']
terminals = data['done']

log_dir = 'd3rlpy_qbert'
env = make_env("QbertNoFrameskip-v4")

dataset = d3rlpy.dataset.MDPDataset(
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminals=terminals,
    action_space=d3rlpy.constants.ActionSpace.DISCRETE,
    action_size=env.action_space.n,
)

# prepare algorithm
cql = d3rlpy.algos.DiscreteCQLConfig(
    observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
    reward_scaler=d3rlpy.preprocessing.ClipRewardScaler(-1.0, 1.0),
    compile_graph=False,
).create(device='cuda:0')

cql.build_with_dataset(dataset)
# start training
from d3rlpy.logging import FileAdapterFactory

import pdb; pdb.set_trace()

logger_factory = FileAdapterFactory(log_dir)

cql.fit(
    dataset,
    n_steps=1000000,
    save_interval=1000,
    logger_adapter=logger_factory,
    evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env, epsilon=0.001)},
)

final_evaluator = d3rlpy.metrics.EnvironmentEvaluator(env, epsilon=0.001, n_trials=1, return_frames=True)

mean_return, frames = final_evaluator(cql, dataset)
