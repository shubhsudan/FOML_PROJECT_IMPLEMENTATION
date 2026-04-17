"""
ttfe.py — Transformer-Based Temporal Feature Extractor (TTFE).

Implements Section IV-A of Li et al. (2024) exactly:

  Input:  S_t ∈ R^{L × F}   (temporal segment of L price vectors, F=7 markets)
  Output: f   ∈ R^{1 × F'}  (extracted feature vector, F'=EMBED_DIM)

Architecture (Fig. 3):
  1. Feature Embedding  : Linear(F → F')               [eq. 14]
  2. Stacked MHA        : N_MHA × MultiHeadAttention    [eq. 15–20]
     Each MHA block:
       - h parallel self-attention heads                [eq. 15–18]
       - Concat + Linear                               [eq. 19]
       - Residual + LayerNorm
       - Forward Net: Linear(F') → ReLU → Linear(F')   [eq. 20]
       - Residual + LayerNorm
  3. Feature Aggregation: Global Average Pooling (dim=L) [eq. 21]

Output shape: (batch, F')  where F' = EMBED_DIM = 64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    NUM_MARKETS, TEMPORAL_SEG_LEN, EMBED_DIM,
    NUM_MHA_HEADS, NUM_MHA_LAYERS, FF_INNER_DIM, DROPOUT
)


class SelfAttentionHead(nn.Module):
    """
    Single scaled dot-product attention head (right side of Fig. 4).
    Computes: SA_j(Q, K, V) = softmax(Q_j K_j^T / sqrt(d_k)) V_j
    """

    def __init__(self, embed_dim: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.W_Q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, head_dim, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, L, F')
        Returns:
            out:        (batch, L, head_dim)
            attn_weights: (batch, L, L)  — for interpretability
        """
        Q = self.W_Q(x)   # (B, L, d_k)
        K = self.W_K(x)   # (B, L, d_k)
        V = self.W_V(x)   # (B, L, d_k)

        scale = self.head_dim ** 0.5
        scores = torch.bmm(Q, K.transpose(1, 2)) / scale   # (B, L, L)
        attn_weights = F.softmax(scores, dim=-1)            # W^att_j

        out = torch.bmm(attn_weights, V)                    # (B, L, d_k)
        return out, attn_weights


class MultiHeadAttentionBlock(nn.Module):
    """
    One full MHA block from Fig. 4:
      - h parallel SA heads
      - Concat + Linear (eq. 19)
      - Residual + LayerNorm
      - Forward Net: LT → ReLU → LT (eq. 20)
      - Residual + LayerNorm
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_MHA_HEADS,
        ff_inner_dim: int = FF_INNER_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        # h parallel SA heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_dim, self.head_dim)
            for _ in range(num_heads)
        ])

        # Concat projection (eq. 19): maps h*head_dim → embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Residual + LayerNorm after MHA
        self.norm1 = nn.LayerNorm(embed_dim)

        # Forward Net (eq. 20): two LT layers with ReLU
        # Paper: first LT dim = ff_inner_dim (2048), second = embed_dim (64)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_inner_dim),
            nn.ReLU(),
            nn.Linear(ff_inner_dim, embed_dim),
        )

        # Residual + LayerNorm after Forward Net
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: (batch, L, embed_dim)
        Returns:
            out:        (batch, L, embed_dim)
            attn_list:  list of h attention matrices (batch, L, L) — for interpretability
        """
        # Run all heads in parallel, collect attention weights
        head_outputs = []
        attn_list    = []
        for head in self.heads:
            h_out, attn = head(x)
            head_outputs.append(h_out)
            attn_list.append(attn)

        # Concatenate heads: (B, L, h * head_dim) = (B, L, embed_dim)
        concat = torch.cat(head_outputs, dim=-1)            # eq. 19 Concat(...)

        # Linear projection + dropout
        mha_out = self.dropout(self.out_proj(concat))       # eq. 19 LT(...)

        # Residual connection + LayerNorm (Fig. 4)
        x = self.norm1(x + mha_out)

        # Forward Net + residual + LayerNorm (eq. 20)
        ff_out = self.dropout(self.ff(x))
        x = self.norm2(x + ff_out)

        return x, attn_list


class TTFE(nn.Module):
    """
    Full Transformer-Based Temporal Feature Extractor (Fig. 3).

    Forward pass:
      1. feature_embedding: Linear(F=7 → F'=EMBED_DIM)
      2. stacked_mha:       N_MHA × MultiHeadAttentionBlock
      3. feature_aggregation: GlobalAveragePooling over L dimension

    Input:  (batch, L, F)  — L=12, F=7
    Output: (batch, F')    — F'=64, the temporal feature vector f
    """

    def __init__(
        self,
        input_dim:    int = NUM_MARKETS,           # F  = 7
        seg_len:      int = TEMPORAL_SEG_LEN,      # L  = 12
        embed_dim:    int = EMBED_DIM,             # F' = 64
        num_heads:    int = NUM_MHA_HEADS,         # h  = 4
        num_layers:   int = NUM_MHA_LAYERS,        # N_MHA = 2
        ff_inner_dim: int = FF_INNER_DIM,          # 2048
        dropout:      float = DROPOUT,
    ):
        super().__init__()
        self.seg_len   = seg_len
        self.embed_dim = embed_dim

        # ── Component 1: Feature Embedding (eq. 14) ──────────────────────────
        # S' = LT(S) = S W^embed + b^embed  ∈ R^{L × F'}
        self.feature_embedding = nn.Linear(input_dim, embed_dim)

        # ── Component 2: Stacked MHA (eq. 15–20) ─────────────────────────────
        self.mha_layers = nn.ModuleList([
            MultiHeadAttentionBlock(embed_dim, num_heads, ff_inner_dim, dropout)
            for _ in range(num_layers)
        ])

        # ── Component 3: Feature Aggregation (eq. 21) ────────────────────────
        # Global Average Pooling: mean over temporal dimension L
        # f_n = (1/L) Σ_{m=1}^{L} s_{m,n}
        # No learnable params — just torch.mean(..., dim=1)

    def forward(
        self,
        S: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            S:               (batch, L, F)   — temporal segment (eq. 13)
            return_attention: if True, also return attention weight matrices

        Returns:
            f:         (batch, F')           — extracted temporal feature vector
            attn_all:  list[layer][head] of (batch, L, L)  — only if return_attention=True
        """
        # ── Step 1: Feature Embedding ─────────────────────────────────────────
        x = self.feature_embedding(S)               # (B, L, F')

        # ── Step 2: Stacked MHA ───────────────────────────────────────────────
        attn_all = []
        for mha in self.mha_layers:
            x, attn_list = mha(x)
            attn_all.append(attn_list)

        # ── Step 3: Global Average Pooling  (eq. 21) ─────────────────────────
        f = x.mean(dim=1)                           # (B, F')

        if return_attention:
            return f, attn_all
        return f

    def extract_numpy(self, segment_np: np.ndarray) -> np.ndarray:
        """
        Convenience method: takes raw numpy temporal segment,
        returns feature vector as numpy.

        Args:
            segment_np: (L, F)  or  (B, L, F)
        Returns:
            np.ndarray: (F',)  or  (B, F')
        """
        squeezed = segment_np.ndim == 2
        if squeezed:
            segment_np = segment_np[np.newaxis]          # (1, L, F)

        tensor = torch.from_numpy(segment_np.astype(np.float32))
        self.eval()
        with torch.no_grad():
            f = self.forward(tensor)

        out = f.numpy()
        if squeezed:
            out = out[0]                                  # (F',)
        return out


# ─── Temporal Segment Builder ─────────────────────────────────────────────────

def build_temporal_segment(
    price_array: np.ndarray,
    t:           int,
    L:           int = TEMPORAL_SEG_LEN,
) -> np.ndarray:
    """
    Constructs S_t = [ρ_{t-L+1}, ..., ρ_t] ∈ R^{L × F}  (eq. 13).

    Args:
        price_array: (T, 7)   — full price array for the episode
        t:           int       — current timestep index (0-based)
        L:           int       — segment length

    Returns:
        segment: (L, 7)  — zero-padded at the start if t < L-1
    """
    F = price_array.shape[1]
    segment = np.zeros((L, F), dtype=np.float32)

    start = t - L + 1
    if start < 0:
        # Pad with zeros for timesteps before episode start
        available = price_array[max(0, start):t+1]
        segment[L - len(available):] = available
    else:
        segment = price_array[start:t+1].copy()

    return segment  # (L, 7)
