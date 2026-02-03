import numpy as np
from typing import List, Union, Optional, NamedTuple
from d3rlpy.dataset import (
    MDPDataset,
    ReplayBuffer,
    InfiniteBuffer,
    Episode,
    PartialTrajectory,
    TrajectoryMiniBatch,
    TransitionMiniBatch,
    Transition,
    BasicTrajectorySlicer,
    BasicTransitionPicker,
    TrajectorySlicerProtocol,
    TransitionPickerProtocol,
)


class TransitionWithSource(NamedTuple):
    """Transition with source dataset identifier."""
    transition: Transition
    source_id: int
    source_name: Optional[str] = None


class TrajectoryWithSource(NamedTuple):
    """Trajectory with source dataset identifier."""
    trajectory: PartialTrajectory
    source_id: int
    source_name: Optional[str] = None


class TransitionBatchWithSource(NamedTuple):
    """Transition mini-batch with source dataset identifiers."""
    batch: TransitionMiniBatch
    source_ids: np.ndarray  # (batch_size,)
    source_names: Optional[List[str]] = None


class TrajectoryBatchWithSource(NamedTuple):
    """Trajectory mini-batch with source dataset identifiers."""
    batch: TrajectoryMiniBatch
    source_ids: np.ndarray  # (batch_size,)
    source_names: Optional[List[str]] = None


class CombinedMDPDataset:
    """
    A wrapper class that combines multiple MDPDatasets and tracks
    which dataset each sample comes from.
    
    Args:
        datasets: List of MDPDataset or ReplayBuffer objects to combine.
        names: Optional list of names for each dataset. If not provided,
               datasets are identified by their index.
        transition_picker: Optional transition picker for sampling.
        trajectory_slicer: Optional trajectory slicer for sampling.
    
    Example:
        >>> dataset1 = MDPDataset(obs1, actions1, rewards1, terminals1)
        >>> dataset2 = MDPDataset(obs2, actions2, rewards2, terminals2)
        >>> combined = CombinedMDPDataset(
        ...     datasets=[dataset1, dataset2],
        ...     names=["expert", "random"]
        ... )
        >>> batch, source_ids = combined.sample_transition_batch(32)
        >>> # source_ids[i] tells you which dataset transition i came from
    """
    
    def __init__(
        self,
        datasets: List[Union[MDPDataset, ReplayBuffer]],
        names: Optional[List[str]] = None,
        transition_picker: Optional[TransitionPickerProtocol] = None,
        trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
    ):
        if len(datasets) == 0:
            raise ValueError("At least one dataset must be provided")
        
        self._datasets = datasets
        self._names = names or [f"dataset_{i}" for i in range(len(datasets))]
        self._transition_picker = transition_picker or BasicTransitionPicker()
        self._trajectory_slicer = trajectory_slicer or BasicTrajectorySlicer()
        
        if len(self._names) != len(self._datasets):
            raise ValueError("Number of names must match number of datasets")
        
        # Build cumulative transition counts for efficient indexing
        self._cumulative_counts = [0]
        for dataset in self._datasets:
            self._cumulative_counts.append(
                self._cumulative_counts[-1] + dataset.transition_count
            )
        
        # Build episode-to-dataset mapping
        self._episode_to_dataset: List[int] = []
        for dataset_idx, dataset in enumerate(self._datasets):
            self._episode_to_dataset.extend(
                [dataset_idx] * len(dataset.episodes)
            )
    
    @property
    def datasets(self) -> List[Union[MDPDataset, ReplayBuffer]]:
        """Returns list of underlying datasets."""
        return self._datasets
    
    @property
    def names(self) -> List[str]:
        """Returns list of dataset names."""
        return self._names
    
    @property
    def transition_count(self) -> int:
        """Returns total number of transitions across all datasets."""
        return self._cumulative_counts[-1]
    
    @property
    def episodes(self) -> List[Episode]:
        """Returns all episodes from all datasets."""
        all_episodes = []
        for dataset in self._datasets:
            all_episodes.extend(dataset.episodes)
        return all_episodes
    
    def size(self) -> int:
        """Returns total number of episodes."""
        return sum(dataset.size() for dataset in self._datasets)
    
    def _global_to_local_index(self, global_index: int) -> tuple:
        """
        Convert a global transition index to (dataset_idx, local_index).
        """
        for dataset_idx, dataset in enumerate(self._datasets):
            start = self._cumulative_counts[dataset_idx]
            end = self._cumulative_counts[dataset_idx + 1]
            if start <= global_index < end:
                local_index = global_index - start
                return dataset_idx, local_index
        raise IndexError(f"Global index {global_index} out of range")
    
    def sample_transition(self) -> TransitionWithSource:
        """
        Sample a single transition with source dataset identifier.
        
        Returns:
            TransitionWithSource containing the transition and source info.
        """
        global_index = np.random.randint(self.transition_count)
        dataset_idx, local_index = self._global_to_local_index(global_index)
        
        dataset = self._datasets[dataset_idx]
        episode, transition_index = dataset.buffer[local_index]
        transition = self._transition_picker(episode, transition_index)
        
        return TransitionWithSource(
            transition=transition,
            source_id=dataset_idx,
            source_name=self._names[dataset_idx],
        )
    
    def sample_transition_batch(
        self, batch_size: int
    ) -> TransitionBatchWithSource:
        """
        Sample a mini-batch of transitions with source dataset identifiers.
        
        Args:
            batch_size: Number of transitions to sample.
        
        Returns:
            TransitionBatchWithSource containing the batch and source info.
        """
        samples = [self.sample_transition() for _ in range(batch_size)]
        
        transitions = [s.transition for s in samples]
        source_ids = np.array([s.source_id for s in samples], dtype=np.int64)
        source_names = [s.source_name for s in samples]
        
        batch = TransitionMiniBatch.from_transitions(transitions)
        
        return TransitionBatchWithSource(
            batch=batch,
            source_ids=source_ids,
            source_names=source_names,
        )
    
    def sample_trajectory(self, length: int) -> TrajectoryWithSource:
        """
        Sample a single trajectory with source dataset identifier.
        
        Args:
            length: Length of the trajectory to sample.
        
        Returns:
            TrajectoryWithSource containing the trajectory and source info.
        """
        global_index = np.random.randint(self.transition_count)
        dataset_idx, local_index = self._global_to_local_index(global_index)
        
        dataset = self._datasets[dataset_idx]
        episode, transition_index = dataset.buffer[local_index]
        trajectory = self._trajectory_slicer(episode, transition_index, length)
        
        return TrajectoryWithSource(
            trajectory=trajectory,
            source_id=dataset_idx,
            source_name=self._names[dataset_idx],
        )
    
    def sample_trajectory_batch(
        self, batch_size: int, length: int
    ) -> TrajectoryBatchWithSource:
        """
        Sample a mini-batch of trajectories with source dataset identifiers.
        
        Args:
            batch_size: Number of trajectories to sample.
            length: Length of each trajectory.
        
        Returns:
            TrajectoryBatchWithSource containing the batch and source info.
        """
        samples = [self.sample_trajectory(length) for _ in range(batch_size)]
        
        trajectories = [s.trajectory for s in samples]
        source_ids = np.array([s.source_id for s in samples], dtype=np.int64)
        source_names = [s.source_name for s in samples]
        
        batch = TrajectoryMiniBatch.from_partial_trajectories(trajectories)
        
        return TrajectoryBatchWithSource(
            batch=batch,
            source_ids=source_ids,
            source_names=source_names,
        )
    
    def sample_from_dataset(
        self, dataset_idx: int, batch_size: int
    ) -> TransitionMiniBatch:
        """
        Sample transitions from a specific dataset only.
        
        Args:
            dataset_idx: Index of the dataset to sample from.
            batch_size: Number of transitions to sample.
        
        Returns:
            TransitionMiniBatch from the specified dataset.
        """
        if dataset_idx < 0 or dataset_idx >= len(self._datasets):
            raise IndexError(f"Dataset index {dataset_idx} out of range")
        
        return self._datasets[dataset_idx].sample_transition_batch(batch_size)
    
    def sample_balanced_batch(
        self, batch_size_per_dataset: int
    ) -> TransitionBatchWithSource:
        """
        Sample an equal number of transitions from each dataset.
        
        Args:
            batch_size_per_dataset: Number of transitions to sample from each dataset.
        
        Returns:
            TransitionBatchWithSource with balanced sampling.
        """
        all_transitions = []
        all_source_ids = []
        all_source_names = []
        
        for dataset_idx, dataset in enumerate(self._datasets):
            batch = dataset.sample_transition_batch(batch_size_per_dataset)
            # Extract individual transitions
            for i in range(batch_size_per_dataset):
                transition = self._datasets[dataset_idx].sample_transition()
                all_transitions.append(transition)
                all_source_ids.append(dataset_idx)
                all_source_names.append(self._names[dataset_idx])
        
        batch = TransitionMiniBatch.from_transitions(all_transitions)
        source_ids = np.array(all_source_ids, dtype=np.int64)
        
        return TransitionBatchWithSource(
            batch=batch,
            source_ids=source_ids,
            source_names=all_source_names,
        )
    
    def get_dataset_info(self) -> dict:
        """Returns information about each dataset."""
        return {
            name: {
                "index": idx,
                "transition_count": dataset.transition_count,
                "episode_count": dataset.size(),
            }
            for idx, (name, dataset) in enumerate(zip(self._names, self._datasets))
        }
    
    def __repr__(self) -> str:
        info = self.get_dataset_info()
        lines = [f"CombinedMDPDataset with {len(self._datasets)} datasets:"]
        for name, data in info.items():
            lines.append(
                f"  [{data['index']}] {name}: "
                f"{data['transition_count']} transitions, "
                f"{data['episode_count']} episodes"
            )
        lines.append(f"  Total: {self.transition_count} transitions, {self.size()} episodes")
        return "\n".join(lines)
    


if __name__ == "__main__":
    import torch
    import d3rlpy
    # from gym.wrappers import make_atari_deepmind
    import gymnasium as gym
    import ale_py
    from gymnasium.wrappers import AtariPreprocessing, ResizeObservation, FrameStackObservation
    

    def make_env(env_id: str):
        env = gym.make(env_id)
        env = AtariPreprocessing(env, screen_size=84, frame_skip=4, terminal_on_life_loss=False, grayscale_obs=True, noop_max=30)
        env = ResizeObservation(env, (84, 84))
        env = FrameStackObservation(env, 4)
        return env

    data_paths = [
        '/u/mrudolph/documents/rl-baselines3-zoo/atari_data/a2c_QbertNoFrameskip-v4_0_100000.pth',
        '/u/mrudolph/documents/rl-baselines3-zoo/atari_data/dqn_QbertNoFrameskip-v4_0_100000.pth',
        # Add more paths here as needed
    ]
    env_id = "QbertNoFrameskip-v4"
    log_dir = 'd3rlpy_qbert'
    env = make_env(env_id)

    datasets = []
    for data_path in data_paths:
        data = torch.load(data_path, weights_only=False)
        observations = data['obs']
        actions = data['action']
        rewards = data['reward']
        terminals = data['done']

        dataset = d3rlpy.dataset.MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            action_space=d3rlpy.constants.ActionSpace.DISCRETE,
            action_size=env.action_space.n,
        )
        datasets.append(dataset)
        
    combined = CombinedMDPDataset(datasets=datasets, names=["a2c", "dqn"])
    import pdb; pdb.set_trace()