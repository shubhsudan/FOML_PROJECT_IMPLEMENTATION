"""
ttfe_stage2.py — TTFE Stage 2: 12-dim input, 32-step window, progressive unfreezing.

Architecture identical to Stage 1 TTFE except:
  - feature_embedding: Linear(12 → 64)   [was Linear(5 → 64)]
  - seg_len: 32                           [was 12]
  - MHA layers: 2 × MultiHeadAttentionBlock (same dims: 64→64)

Weight transfer from Stage 1 (upgrade_ttfe_weights):
  - mha_layers[*]: fully transferred (64→64, no shape change)
  - feature_embedding: re-initialized (5→12 input dim changes)

Progressive unfreezing methods:
  - freeze_all()        : Phase A — TTFE completely frozen
  - unfreeze_top_layer(): Phase B — only last MHA layer trainable (lr=1e-5)
  - unfreeze_all()      : Phase C — all TTFE params trainable (lr=1e-5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    EMBED_DIM, NUM_MHA_HEADS, NUM_MHA_LAYERS, FF_INNER_DIM, DROPOUT
)
from config_stage2 import (
    TTFE_INPUT_DIM_S2, TTFE_SEG_LEN_S2,
    STAGE1_CKPT_PATH,
)
# Import MHA block from Stage 1 (same architecture, reusable)
from ttfe import MultiHeadAttentionBlock


class TTFE_S2(nn.Module):
    """
    Stage 2 TTFE: processes 32-step × 12-dim price segments.

    Input:  (batch, 32, 12)
    Output: (batch, 64)

    Identical MHA architecture to Stage 1 — weights are directly transferable.
    Only feature_embedding differs (12-dim input vs 5-dim).
    """

    def __init__(
        self,
        input_dim:    int = TTFE_INPUT_DIM_S2,   # 12
        seg_len:      int = TTFE_SEG_LEN_S2,      # 32
        embed_dim:    int = EMBED_DIM,             # 64
        num_heads:    int = NUM_MHA_HEADS,         # 4
        num_layers:   int = NUM_MHA_LAYERS,        # 2
        ff_inner_dim: int = FF_INNER_DIM,          # 2048
        dropout:      float = DROPOUT,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seg_len   = seg_len
        self.embed_dim = embed_dim

        # Feature Embedding: Linear(12 → 64) — must be re-initialized (shape differs from S1)
        self.feature_embedding = nn.Linear(input_dim, embed_dim)

        # Stacked MHA: same dims as Stage 1 → weights directly transferable
        self.mha_layers = nn.ModuleList([
            MultiHeadAttentionBlock(embed_dim, num_heads, ff_inner_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S: (batch, L, F) where L=32, F=12

        Returns:
            f: (batch, 64)
        """
        x = self.feature_embedding(S)          # (B, L, 64)
        for mha in self.mha_layers:
            x, _ = mha(x)                      # (B, L, 64)
        return x.mean(dim=1)                   # (B, 64) — global avg pool

    def extract_numpy(self, segment_np: np.ndarray) -> np.ndarray:
        """
        Convenience: numpy → numpy.
        segment_np: (L, 12) or (B, L, 12)
        """
        squeezed = segment_np.ndim == 2
        if squeezed:
            segment_np = segment_np[np.newaxis]
        device = next(self.parameters()).device
        t = torch.from_numpy(segment_np.astype(np.float32)).to(device)
        self.eval()
        with torch.no_grad():
            f = self.forward(t)
        out = f.cpu().numpy()
        return out[0] if squeezed else out

    # ── Freeze / unfreeze helpers ─────────────────────────────────────────────

    def freeze_all(self):
        """Phase A: freeze all TTFE parameters."""
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze_top_layer(self):
        """Phase B: unfreeze only the last MHA layer (+ its norms)."""
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.mha_layers[-1].parameters():
            p.requires_grad_(True)

    def unfreeze_all(self):
        """Phase C: unfreeze all parameters (embedding + all MHA)."""
        for p in self.parameters():
            p.requires_grad_(True)

    def trainable_params(self):
        """Returns list of currently trainable parameters."""
        return [p for p in self.parameters() if p.requires_grad]


# ── Weight transfer from Stage 1 ──────────────────────────────────────────────

def upgrade_ttfe_weights(
    ttfe_s2:        TTFE_S2,
    checkpoint_path: str = STAGE1_CKPT_PATH,
    device:          torch.device = torch.device("cpu"),
) -> TTFE_S2:
    """
    Transfers MHA layer weights from Stage 1 checkpoint into TTFE_S2.
    Re-initializes feature_embedding (shape changed 5→12).

    Stage 1 checkpoint key: 'ttfe_state'  (NOT 'ttfe')
    MHA layers have identical shapes → direct state_dict copy.

    Args:
        ttfe_s2:         freshly initialized TTFE_S2 instance
        checkpoint_path: path to Stage 1 best_model.pt
        device:          target device

    Returns:
        ttfe_s2 with MHA weights loaded from Stage 1
    """
    if not os.path.exists(checkpoint_path):
        print(f"[TTFE_S2] WARNING: Stage 1 checkpoint not found at {checkpoint_path}")
        print(f"[TTFE_S2] Starting with random weights.")
        return ttfe_s2.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"[TTFE_S2] Stage 1 checkpoint keys: {list(ckpt.keys())}")

    # Correct key: 'ttfe_state' (the handout incorrectly lists 'ttfe')
    if "ttfe_state" in ckpt:
        s1_state = ckpt["ttfe_state"]
    elif "ttfe" in ckpt:
        s1_state = ckpt["ttfe"]
    else:
        print(f"[TTFE_S2] WARNING: No TTFE weights in checkpoint. "
              f"Available keys: {list(ckpt.keys())}")
        return ttfe_s2.to(device)

    # Transfer MHA layer weights (same dims: 64→64, no shape change)
    n_mha = len(ttfe_s2.mha_layers)
    transferred = 0
    for i in range(n_mha):
        prefix = f"mha_layers.{i}."
        mha_state = {
            k[len(prefix):]: v
            for k, v in s1_state.items()
            if k.startswith(prefix)
        }
        if mha_state:
            ttfe_s2.mha_layers[i].load_state_dict(mha_state, strict=True)
            transferred += 1
            print(f"[TTFE_S2] MHA layer {i} weights transferred from Stage 1")
        else:
            print(f"[TTFE_S2] WARNING: No weights found for mha_layers.{i}")

    # feature_embedding: re-initialize (Linear(5→64) → Linear(12→64), shape differs)
    nn.init.xavier_uniform_(ttfe_s2.feature_embedding.weight)
    nn.init.zeros_(ttfe_s2.feature_embedding.bias)
    print(f"[TTFE_S2] feature_embedding re-initialized "
          f"(5→12 input dim, shape incompatible with Stage 1)")
    print(f"[TTFE_S2] Transferred {transferred}/{n_mha} MHA layers. "
          f"Stage 2 TTFE ready.")

    return ttfe_s2.to(device)


# ── Build Stage 2 TTFE (convenience) ──────────────────────────────────────────

def build_ttfe_s2(
    checkpoint_path: str = STAGE1_CKPT_PATH,
    device: torch.device = torch.device("cpu"),
) -> TTFE_S2:
    """
    Creates TTFE_S2 and loads Stage 1 MHA weights.
    Starts in Phase A (fully frozen).
    """
    ttfe = TTFE_S2().to(device)
    ttfe = upgrade_ttfe_weights(ttfe, checkpoint_path, device)
    ttfe.freeze_all()
    ttfe.eval()
    print("[TTFE_S2] Initialized in Phase A (fully frozen)")
    return ttfe
