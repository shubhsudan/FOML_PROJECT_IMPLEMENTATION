"""
replay_buffer.py — Circular experience replay buffer for SAC.

Stores (obs, action, reward, next_obs, done) tuples as float32 numpy arrays
and returns batches as torch tensors on the target device.

Dimensions (from config):
    obs_dim = 1 + NUM_MARKETS + EMBED_DIM = 1 + 7 + 64 = 72
    act_dim = 6
"""

import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from config import EMBED_DIM, NUM_MARKETS


class ReplayBuffer:
    """
    Simple circular (ring) replay buffer.

    All data is stored as float32 on CPU numpy arrays for memory efficiency.
    Batches are returned as torch tensors moved to the specified device.

    Args:
        capacity  : maximum number of transitions to store
        obs_dim   : dimension of the observation vector (default 72)
        act_dim   : dimension of the action vector (default 6)
        device    : torch.device — where sampled tensors will be placed
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int = 1 + NUM_MARKETS + EMBED_DIM,
        act_dim: int = 6,
        device: torch.device = torch.device("cpu"),
    ):
        self.capacity = capacity
        self.obs_dim  = obs_dim
        self.act_dim  = act_dim
        self.device   = device

        # Pre-allocate fixed-size arrays — avoids repeated allocation overhead
        self._obs      = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self._actions  = np.zeros((capacity, act_dim),  dtype=np.float32)
        self._rewards  = np.zeros((capacity, 1),        dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self._dones    = np.zeros((capacity, 1),        dtype=np.float32)

        self._ptr  = 0       # write pointer (circular)
        self._size = 0       # current number of valid entries

    # ─── Push ─────────────────────────────────────────────────────────────────

    def push(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        """
        Store one transition.

        Args:
            obs      : (obs_dim,) numpy float32
            action   : (act_dim,) numpy float32
            reward   : scalar float
            next_obs : (obs_dim,) numpy float32
            done     : bool or 0/1
        """
        idx = self._ptr

        self._obs[idx]      = obs.astype(np.float32)
        self._actions[idx]  = action.astype(np.float32)
        self._rewards[idx]  = float(reward)
        self._next_obs[idx] = next_obs.astype(np.float32)
        self._dones[idx]    = float(done)

        # Advance circular pointer
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ─── Sample ───────────────────────────────────────────────────────────────

    def sample(self, batch_size: int) -> dict:
        """
        Sample a random mini-batch of transitions.

        Returns:
            dict with keys:
                'obs'      : (batch, obs_dim)  float32 tensor
                'actions'  : (batch, act_dim)  float32 tensor
                'rewards'  : (batch, 1)        float32 tensor
                'next_obs' : (batch, obs_dim)  float32 tensor
                'dones'    : (batch, 1)        float32 tensor
        """
        assert self._size >= batch_size, (
            f"Buffer has only {self._size} entries; "
            f"need at least batch_size={batch_size}"
        )

        idxs = np.random.randint(0, self._size, size=batch_size)

        def _to_tensor(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr[idxs]).to(self.device)

        return {
            "obs"      : _to_tensor(self._obs),
            "actions"  : _to_tensor(self._actions),
            "rewards"  : _to_tensor(self._rewards),
            "next_obs" : _to_tensor(self._next_obs),
            "dones"    : _to_tensor(self._dones),
        }

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(capacity={self.capacity}, "
            f"size={self._size}, "
            f"obs_dim={self.obs_dim}, "
            f"act_dim={self.act_dim})"
        )
