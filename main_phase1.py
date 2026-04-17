"""
main_phase1.py — Phase 1 entry point.

Runs all components and verifies shapes/values.
Run this from ~/tempdrl/ (or the IMPLEMENTATION directory) with:
    source venv/bin/activate
    python main_phase1.py

Expected output (all PASS):
  [1] Data loading       ...  PASS
  [2] Episode iterator   ...  PASS
  [3] TTFE forward pass  ...  PASS
  [4] Temporal segment   ...  PASS
  [5] Environment step   ...  PASS
  [6] Full episode loop  ...  PASS
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import torch
from tqdm import tqdm

from config import (
    TRAIN_YEAR, TEMPORAL_SEG_LEN, EMBED_DIM, NUM_MARKETS,
    TIMESTEPS_PER_DAY, OUTPUT_DIR
)
from data_loader import load_all, iter_daily_episodes
from ttfe import TTFE, build_temporal_segment
from environment import BESSEnvironment


def test_data_loading():
    print("\n[1] Testing data loading...")
    data = load_all(TRAIN_YEAR)

    assert data["train_raw"].ndim == 2
    assert data["train_raw"].shape[1] == NUM_MARKETS, \
        f"Expected 7 markets, got {data['train_raw'].shape[1]}"
    assert data["eval_raw"].shape[1] == NUM_MARKETS
    assert data["train_scaled"].shape == data["train_raw"].shape

    print(f"    Train shape: {data['train_raw'].shape}")
    print(f"    Eval  shape: {data['eval_raw'].shape}")
    print(f"    Columns: {data['columns']}")
    print(f"    Train price stats (raw): mean={data['train_raw'].mean():.2f}, "
          f"std={data['train_raw'].std():.2f}")
    print("    [1] PASS ✓")
    return data


def test_episode_iterator(data):
    print("\n[2] Testing episode iterator...")
    train_episodes = iter_daily_episodes(data["train_scaled"])
    eval_episodes  = iter_daily_episodes(data["eval_scaled"])

    assert len(train_episodes) > 0
    assert train_episodes[0].shape == (TIMESTEPS_PER_DAY, NUM_MARKETS), \
        f"Episode shape: {train_episodes[0].shape}"
    print(f"    Train episodes: {len(train_episodes)}")
    print(f"    Eval  episodes: {len(eval_episodes)}")
    print(f"    Episode shape:  {train_episodes[0].shape}")
    print("    [2] PASS ✓")
    return train_episodes, eval_episodes


def test_ttfe():
    print("\n[3] Testing TTFE forward pass...")
    model = TTFE()
    print(f"    TTFE parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Single sample
    S = torch.randn(1, TEMPORAL_SEG_LEN, NUM_MARKETS)
    f, attn = model(S, return_attention=True)
    assert f.shape == (1, EMBED_DIM), f"Feature shape: {f.shape}"
    print(f"    Input shape:  {S.shape}")
    print(f"    Output shape: {f.shape}")

    # Batch
    S_batch = torch.randn(32, TEMPORAL_SEG_LEN, NUM_MARKETS)
    f_batch = model(S_batch)
    assert f_batch.shape == (32, EMBED_DIM)

    # Check attention weight sums to 1 (softmax)
    for layer_attn in attn:
        for head_attn in layer_attn:
            sums = head_attn.sum(dim=-1)   # should be all-ones
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
                "Attention weights do not sum to 1"
    print("    Attention weights sum check: OK")

    # Numpy convenience method
    seg_np = np.random.randn(TEMPORAL_SEG_LEN, NUM_MARKETS).astype(np.float32)
    f_np = model.extract_numpy(seg_np)
    assert f_np.shape == (EMBED_DIM,), f"numpy output shape: {f_np.shape}"
    print("    numpy extract_numpy: OK")
    print("    [3] PASS ✓")
    return model


def test_temporal_segment(train_episodes):
    print("\n[4] Testing temporal segment builder...")
    ep = train_episodes[0]

    # At beginning of episode (pad test)
    seg_t0 = build_temporal_segment(ep, t=0)
    assert seg_t0.shape == (TEMPORAL_SEG_LEN, NUM_MARKETS)
    # First rows should be zero-padded
    assert np.all(seg_t0[:-1] == 0), "Expected zero padding at t=0"

    # Mid-episode
    seg_mid = build_temporal_segment(ep, t=50)
    assert seg_mid.shape == (TEMPORAL_SEG_LEN, NUM_MARKETS)
    assert np.all(seg_mid == ep[50 - TEMPORAL_SEG_LEN + 1: 51])

    print(f"    Segment shape:      {seg_mid.shape}")
    print(f"    Zero-pad at t=0:    {TEMPORAL_SEG_LEN - 1} rows padded")
    print("    [4] PASS ✓")


def test_environment(train_episodes, ttfe_model):
    print("\n[5] Testing environment step...")
    ep = train_episodes[0]

    # Use raw (unscaled) prices for the environment
    # (scaled prices are for TTFE input only)
    env = BESSEnvironment(price_episode=ep, feature_dim=EMBED_DIM, mode="joint")

    # Build initial temporal segment and feature
    seg = build_temporal_segment(ep, t=0)
    f = ttfe_model.extract_numpy(seg)
    obs = env.reset(init_feature=f)

    expected_obs_dim = 1 + NUM_MARKETS + EMBED_DIM
    assert obs.shape == (expected_obs_dim,), \
        f"obs shape: {obs.shape}, expected ({expected_obs_dim},)"
    print(f"    Obs dim: {obs.shape[0]}  (1 SoC + 7 prices + {EMBED_DIM} features)")

    # One random step
    raw_action = np.random.uniform(-1, 1, size=6).astype(np.float32)
    seg_next = build_temporal_segment(ep, t=1)
    f_next = ttfe_model.extract_numpy(seg_next)
    next_obs, reward, done, info = env.step(raw_action, next_feature=f_next)

    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert not done
    print(f"    Reward:  {reward:.4f}")
    print(f"    SoC:     {info['soc']:.3f}")
    print(f"    v_dch={info['v_dch']}, v_ch={info['v_ch']}")
    print("    [5] PASS ✓")


def test_full_episode(train_episodes, ttfe_model):
    print("\n[6] Testing full episode loop...")
    ep = train_episodes[0]

    env = BESSEnvironment(price_episode=ep, feature_dim=EMBED_DIM, mode="joint")

    total_reward = 0.0
    soc_trajectory = []
    violations = 0

    seg = build_temporal_segment(ep, t=0)
    f = ttfe_model.extract_numpy(seg)
    obs = env.reset(init_feature=f)

    for t in tqdm(range(len(ep)), desc="    Episode", leave=False, ncols=60):
        raw_action = np.random.uniform(-1, 1, size=6).astype(np.float32)

        t_next = min(t + 1, len(ep) - 1)
        seg_next = build_temporal_segment(ep, t=t_next)
        f_next = ttfe_model.extract_numpy(seg_next)

        next_obs, reward, done, info = env.step(raw_action, next_feature=f_next)
        total_reward += reward
        soc_trajectory.append(info["soc"])
        if info["violated"]:
            violations += 1

        obs = next_obs
        if done:
            break

    print(f"    Total episode reward:  {total_reward:.2f}")
    print(f"    SoC min/max:           {min(soc_trajectory):.3f} / {max(soc_trajectory):.3f}")
    print(f"    Violations (random):   {violations} / {len(ep)}")
    print("    [6] PASS ✓")
    print()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  TempDRL Phase 1 — Component Verification")
    print("=" * 60)

    data           = test_data_loading()
    train_ep, eval_ep = test_episode_iterator(data)
    ttfe_model     = test_ttfe()
    test_temporal_segment(train_ep)
    test_environment(train_ep, ttfe_model)
    test_full_episode(train_ep, ttfe_model)

    print("=" * 60)
    print("  ALL PHASE 1 TESTS PASSED ✓")
    print("=" * 60)
    print()
    print("Next: Implement Phase 2 (SAC agent + training loop).")
