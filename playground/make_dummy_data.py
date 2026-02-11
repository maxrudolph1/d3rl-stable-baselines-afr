import torch
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt

path = Path('/u/mrudolph/documents/d3rlpy/atari_data/atari_data_with_state/BreakoutNoFrameskip-v4/dqn/ckpt_7_10000_with_state.pth')


data = torch.load(path, weights_only=False)
N = data['obs'].shape[0]

ns = np.random.randint(N, size=20)
import pdb; pdb.set_trace()
fig, axes = plt.subplots(4, 5, figsize=(10, 8))
for i, n in enumerate(ns):
    img = data['obs'][n, :3]
    # Transpose if needed: (C, H, W) to (H, W, C)
    if img.shape[0] == 3:
        img_to_show = np.transpose(img, (1, 2, 0))
    else:
        img_to_show = img
    ax = axes[i // 5, i % 5]
    ax.imshow(img_to_show.astype(np.uint8))
    ax.axis('off')
    ax.set_title(f"{n}", fontsize=8)
plt.tight_layout()
plt.savefig('figures/dummy_obs_grid.png')
plt.close(fig)
