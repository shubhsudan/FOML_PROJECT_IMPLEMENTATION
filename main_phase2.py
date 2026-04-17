"""
main_phase2.py — Entry point for Phase 2: SAC training of TempDRL BESS agent.

Usage:
    python main_phase2.py

Hardware target:
    NVIDIA A16, CUDA 12.4, GPU index 26
"""

import sys
import os

# Ensure src/ is on the import path regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import torch
from src.trainer import train


def _print_banner(
    gpu_id:       int,
    num_episodes: int,
    eval_every:   int,
    warmup_steps: int,
    batch_size:   int,
    save_dir:     str,
    log_dir:      str,
) -> None:
    """Print a concise startup banner with key configuration."""
    if torch.cuda.is_available():
        device_str = f"cuda:{gpu_id}"
        try:
            device_name = torch.cuda.get_device_name(gpu_id)
        except Exception:
            device_name = "unknown GPU"
        cuda_ver = torch.version.cuda or "unknown"
        device_info = f"{device_str}  ({device_name}, CUDA {cuda_ver})"
    else:
        device_info = "cpu  (CUDA not available)"

    print("=" * 70)
    print("  TempDRL — Temporal-Aware DRL for Energy Storage Bidding")
    print("  Phase 2: SAC Training")
    print("=" * 70)
    print(f"  Device        : {device_info}")
    print(f"  Episodes      : {num_episodes}")
    print(f"  Eval every    : {eval_every} episodes")
    print(f"  Warmup steps  : {warmup_steps}")
    print(f"  Batch size    : {batch_size}")
    print(f"  Save dir      : {save_dir}")
    print(f"  Log dir       : {log_dir}")
    print("=" * 70)
    print()


if __name__ == "__main__":
    NUM_EPISODES      = 500                 # TTFE top-up: pair actor with fixed-seed TTFE
    EVAL_EVERY        = 50
    WARMUP_STEPS      = 0                   # no warmup — actor already trained
    GRAD_STEPS_PER_EP = 72
    GPU_ID            = 26
    SAVE_DIR          = "outputs/checkpoints"
    LOG_DIR           = "outputs/logs"
    RESUME_CKPT       = "outputs/checkpoints/best_model.pt"  # ep 17100 actor weights
    START_EPISODE     = 30_001                                # top-up episodes 30001–30500

    from src.config import BATCH_SIZE

    _print_banner(
        gpu_id=GPU_ID,
        num_episodes=NUM_EPISODES,
        eval_every=EVAL_EVERY,
        warmup_steps=WARMUP_STEPS,
        batch_size=BATCH_SIZE,
        save_dir=SAVE_DIR,
        log_dir=LOG_DIR,
    )

    train(
        num_episodes=NUM_EPISODES,
        eval_every=EVAL_EVERY,
        warmup_steps=WARMUP_STEPS,
        grad_steps_per_ep=GRAD_STEPS_PER_EP,
        gpu_id=GPU_ID,
        save_dir=SAVE_DIR,
        log_dir=LOG_DIR,
        resume_ckpt=RESUME_CKPT,
        start_episode=START_EPISODE,
    )
