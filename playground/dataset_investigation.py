import h5py
import numpy as np
import imageio

data_path = '/u/mrudolph/documents/rl-baselines3-zoo/atari_data/BreakoutNoFrameskip-v4_old/train_a2c_10000.hdf5'
data = h5py.File(data_path, 'r')

observations = np.moveaxis(data['frames'][:1500], -1, 1)
actions = data['action'][:1500]
rewards = data['reward'][:1500]
terminals = np.concatenate([np.diff(data['episode_index'][:1500]), [1]])

# Extract the first channel from each observation
first_channel_frames = observations[:, 0, :, :]  # Shape: (1500, H, W)

# Convert grayscale to RGB
rgb_frames = np.stack([first_channel_frames] * 3, axis=-1)  # Shape: (1500, H, W, 3)

# Find terminal indices and flash red for 4 frames after each
terminal_indices = np.where(terminals == 1)[0]
for idx in terminal_indices:
    for flash_offset in range(4):
        flash_idx = idx + flash_offset
        if flash_idx < len(rgb_frames):
            # Create red square (set R channel to max, G and B to 0)
            rgb_frames[flash_idx, :, :, 0] = 255  # Red
            rgb_frames[flash_idx, :, :, 1] = 0    # Green
            rgb_frames[flash_idx, :, :, 2] = 0    # Blue

# Create MP4 movie using imageio
output_path = 'first_channel_movie.mp4'
imageio.mimwrite(output_path, rgb_frames.astype(np.uint8), fps=30)
print(f"Saved movie to {output_path}")
