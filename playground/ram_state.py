import torch
import numpy as np
path = '/u/mrudolph/documents/d3rlpy/atari_data/SeaquestNoFrameskip-v4/dqn_test'
import os

def list_pth_files(directory):
    """List all .pth files in the given directory (non-recursive)."""
    return [f for f in os.listdir(directory) if f.endswith('.pth')]

# Example usage:
# pth_dir = os.path.dirname(path)
pth_files = list_pth_files(path)
import pdb; pdb.set_trace()
print("Found .pth files in directory:", path)
for fname in pth_files:
    print(fname)

    data = torch.load(os.path.join(path, fname), weights_only=False)
    

    state = data['state']
    normalized_state = state - state.mean(0)

    normalized_state = normalized_state / np.maximum(normalized_state.std(0), 1e-6)

    data['normalized_state'] = normalized_state
    torch.save(data, os.path.join(path, fname))

import pdb; pdb.set_trace()