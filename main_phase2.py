"""
main_phase2.py — Entry point for TempDRL SAC training (multi-year ERCOT).

Multi-year retrain: 2020-2023 pre-ECRS (1,243 complete days, ~4.5x more than 2022-only).
  - 5 markets: spot + RegUp + RegDn + RRS + NSRS
  - Energy price: dam_spp (DAM Settlement Point Price — fully populated all years)
  - 8-dim action space
  - 72-dim state space (SoC + prices + TTFE + hour_sin_cos)
  - Chronological 70/10/20 train/val/test split
  - TTFE saved in every checkpoint

Hardware target:
    NVIDIA A16, CUDA 12.4, GPU index 26
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch
from src.trainer import train
from src.config import (
    BATCH_SIZE, STATE_DIM, ACTION_DIM, NUM_MARKETS,
    CKPT_DIR, LOG_DIR,
)


def _print_banner(gpu_id, num_episodes, eval_every, warmup_steps,
                  batch_size, save_dir, log_dir):
    if torch.cuda.is_available():
        device_str  = f"cuda:{gpu_id}"
        device_name = torch.cuda.get_device_name(gpu_id) if torch.cuda.is_available() else "?"
        cuda_ver    = torch.version.cuda or "unknown"
        device_info = f"{device_str}  ({device_name}, CUDA {cuda_ver})"
    else:
        device_info = "cpu  (CUDA not available)"

    print("=" * 70)
    print("  TempDRL — ERCOT-Correct Joint-Market Bidding (Fresh Retrain)")
    print("=" * 70)
    print(f"  Device        : {device_info}")
    print(f"  Markets       : {NUM_MARKETS}  [spot, RegUp, RegDn, RRS, NSRS]")
    print(f"  State dim     : {STATE_DIM}  (SoC + prices + TTFE + hour_sin_cos)")
    print(f"  Action dim    : {ACTION_DIM}  (v_dch/v_ch + spot_dch/ch + regup/regdn + rrs + nsrs)")
    print(f"  Episodes      : {num_episodes}")
    print(f"  Eval every    : {eval_every} episodes")
    print(f"  Warmup steps  : {warmup_steps}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Save dir      : {save_dir}")
    print(f"  Log dir       : {log_dir}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    NUM_EPISODES      = 60_000
    EVAL_EVERY        = 100      # less frequent: val set is 3x larger than 2022-only
    WARMUP_STEPS      = 10_000   # ~35 episodes of random actions to seed 300k buffer
    GRAD_STEPS_PER_EP = 72
    GPU_ID            = 26
    SAVE_DIR          = "outputs/checkpoints"
    LOG_DIR_RUN       = "outputs/logs"
    RESUME_CKPT       = None     # fresh training — no resume
    START_EPISODE     = 1

    _print_banner(
        gpu_id=GPU_ID,
        num_episodes=NUM_EPISODES,
        eval_every=EVAL_EVERY,
        warmup_steps=WARMUP_STEPS,
        batch_size=BATCH_SIZE,
        save_dir=SAVE_DIR,
        log_dir=LOG_DIR_RUN,
    )

    train(
        num_episodes=NUM_EPISODES,
        eval_every=EVAL_EVERY,
        warmup_steps=WARMUP_STEPS,
        grad_steps_per_ep=GRAD_STEPS_PER_EP,
        gpu_id=GPU_ID,
        save_dir=SAVE_DIR,
        log_dir=LOG_DIR_RUN,
        resume_ckpt=RESUME_CKPT,
        start_episode=START_EPISODE,
    )
