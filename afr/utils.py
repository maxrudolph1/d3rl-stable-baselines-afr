"""
Utility functions for AFR scripts.
"""

import gymnasium as gym
import ale_py  # noqa: F401 (registers Atari envs)
from gymnasium.wrappers import AtariPreprocessing, ResizeObservation, FrameStackObservation


def make_atari_env(env_id: str):
    """
    Create an Atari environment with standard preprocessing.

    Args:
        env_id: Atari environment ID (e.g. 'QbertNoFrameskip-v4').

    Returns:
        Preprocessed Gymnasium environment with:
        - 84x84 grayscale frames
        - Frame skip of 4
        - Frame stacking of 4
    """
    env = gym.make(env_id)
    env = AtariPreprocessing(
        env,
        screen_size=84,
        frame_skip=4,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        noop_max=30,
    )
    env = ResizeObservation(env, (84, 84))
    env = FrameStackObservation(env, 4)
    return env
