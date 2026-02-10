import torch
import numpy as np
path = '/u/mrudolph/documents/d3rlpy/atari_data/atari_data_with_state/BreakoutNoFrameskip-v4/dqn'
import os

def list_pth_files(directory):
    """List all .pth files in the given directory (non-recursive)."""
    return [f for f in os.listdir(directory) if f.endswith('.pth')]

# Example usage:
# pth_dir = os.path.dirname(path)
pth_files = list_pth_files(path)
print("Found .pth files in directory:", path)
for fname in pth_files:
    print("processing file: ", fname)
    data = torch.load(os.path.join(path, fname), weights_only=False)
    state = data['state']
    normalized_state = state / 128.0 - 1
    valid_indices = np.where(normalized_state.std(0) > 1e-6)[0]
    normalized_state = normalized_state[:, valid_indices]
    data['normalized_state'] = normalized_state
    save_path = os.path.join(path, fname.replace('.pth', '_with_state.pth'))
    torch.save(data, save_path)
    print("saved file: ", fname)
