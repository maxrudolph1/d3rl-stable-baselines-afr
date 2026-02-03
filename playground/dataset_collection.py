import d3rlpy
import gym
import torch 
import numpy as np
import h5py
import d4rl_atari
np.bool8 = np.bool
from gym.wrappers import AtariPreprocessing, ResizeObservation
env = gym.make("BreakoutNoFrameskip-v4", stack=True)
env = AtariPreprocessing(env, screen_size=84, frame_skip=4, terminal_on_life_loss=False, grayscale_obs=True, noop_max=30)
env = ResizeObservation(env, (84, 84))
from gym.wrappers import FrameStack
env = FrameStack(env, 4)

env = gym.make("breakout-mixed-v0", stack=True)

class LazyFrameToNumpy(gym.ObservationWrapper):
    def observation(self, observation):
        if hasattr(observation, 'shape'):
            return np.array(observation)
        if isinstance(observation, (tuple, list)):
            return np.array([np.array(o) for o in observation])
        return np.array(observation)
    
    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        # Gym 0.26+ returns (obs, info), Gym <0.26 returns obs only.
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return self.observation(obs), info
        obs = result
        return self.observation(obs)

env = LazyFrameToNumpy(env)
data_path = '/u/mrudolph/documents/rl-baselines3-zoo/atari_data/BreakoutNoFrameskip-v4_old/train_a2c_10000.hdf5'
data = h5py.File(data_path, 'r')

observations = np.moveaxis(data['frames'][:100], -1, 1)
actions = data['action'][:100]
rewards = data['reward'][:100]
terminals = np.concatenate([np.diff(data['episode_index'][:100]), [1]])


dataset = d3rlpy.dataset.MDPDataset(
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminals=terminals,
    action_space=d3rlpy.constants.ActionSpace.DISCRETE,
    action_size=env.action_space.n,
)

obs, info = env.reset()

import matplotlib.pyplot as plt

# Create a single figure with 2 rows and 4 columns for subplots
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Plot obs variable frames in the first row
for i in range(4):
    axes[0, i].imshow(obs[i], cmap='gray')
    axes[0, i].set_title(f'obs Frame {i}')
    axes[0, i].axis('off')

# Plot first element of observations in the second row
first_obs = observations[0]
for i in range(4):
    axes[1, i].imshow(first_obs[i], cmap='gray')
    axes[1, i].set_title(f'observations[0] Frame {i}')
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig("obs_and_first_observations_frames.png")
plt.show()


# # prepare algorithm
# cql = d3rlpy.algos.DiscreteCQLConfig(compile_graph=True).create(device='cuda:0')

# # train
# cql.fit(
#     dataset,
#     n_steps=1000000,
#     evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
    
# )
# env = gym.make("CartPole-v1")
# setup algorithm
# random_policy = d3rlpy.algos.DiscreteRandomPolicyConfig().create()

# # prepare experience replay buffer
# buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=100000, env=env)

# # start data collection
# random_policy.collect(env, buffer, n_steps=100)
# import pdb; pdb.set_trace()
# # save ReplayBuffer
# with open("random_policy_dataset.h5", "w+b") as f:
#     buffer.dump(f)