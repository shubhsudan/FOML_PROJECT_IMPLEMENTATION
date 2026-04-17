"""
trainer.py — SAC training loop for the TempDRL BESS bidding agent.

Per-episode pipeline:
  1. Sample a random training daily episode (288 timesteps of price data)
  2. Reset BESSEnvironment with that day's raw prices
  3. For each timestep t:
       a. Build temporal segment from SCALED prices (L=12 window)
       b. Run TTFE to extract feature vector f_t  (64-dim)
       c. Construct obs = [SoC | raw_price_t | f_t]  (72-dim)
       d. Select action (random during warmup, else SAC actor)
       e. Step environment with raw action + next feature
       f. Push transition to replay buffer
  4. After episode: perform grad_steps_per_ep SAC gradient updates
  5. Log episode metrics; save checkpoint every eval_every episodes
  6. Every eval_every episodes: evaluate on all eval daily episodes

Logging:
  - outputs/logs/train_log.csv : episode-level training metrics
  - outputs/logs/eval_log.csv  : evaluation metrics every eval_every episodes
"""

import os
import sys
import csv
import time
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    EMBED_DIM, NUM_MARKETS, TEMPORAL_SEG_LEN,
    REPLAY_BUFFER_SIZE, BATCH_SIZE,
    TIMESTEPS_PER_DAY, ALPHA_ENTROPY, LR_POLICY,
    REWARD_CLIP, CAPACITY_MWH,
)
from data_loader import load_all, iter_daily_episodes
from environment import BESSEnvironment
from ttfe import TTFE, build_temporal_segment
from replay_buffer import ReplayBuffer
from sac_agent import SACAgent


# ─── Checkpoint helpers (include TTFE state) ──────────────────────────────────

def _save_checkpoint(agent: SACAgent, ttfe: TTFE, path: str) -> None:
    """Save agent weights + TTFE state dict to a single .pt file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "actor_state"          : agent.actor.state_dict(),
            "critic1_state"        : agent.critic1.state_dict(),
            "critic2_state"        : agent.critic2.state_dict(),
            "critic1_target_state" : agent.critic1_target.state_dict(),
            "critic2_target_state" : agent.critic2_target.state_dict(),
            "log_alpha"            : agent.log_alpha.detach().cpu(),
            "actor_opt_state"      : agent.actor_optimizer.state_dict(),
            "critic1_opt_state"    : agent.critic1_optimizer.state_dict(),
            "critic2_opt_state"    : agent.critic2_optimizer.state_dict(),
            "alpha_opt_state"      : agent.alpha_optimizer.state_dict(),
            "ttfe_state"           : ttfe.state_dict(),
        },
        path,
    )
    print(f"[Trainer] Saved checkpoint → {path}")


# ─── CSV Logger ───────────────────────────────────────────────────────────────

class CSVLogger:
    """Lightweight CSV logger that appends one row per call."""

    def __init__(self, path: str, fieldnames: list):
        self.path       = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write header if file does not already exist
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def write(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


# ─── TTFE feature extraction helper ──────────────────────────────────────────

def extract_feature(
    ttfe:          TTFE,
    scaled_prices: np.ndarray,
    t:             int,
    device:        torch.device,
) -> np.ndarray:
    """
    Build temporal segment from scaled prices and run TTFE in no-grad / eval mode.

    Args:
        ttfe          : TTFE model instance
        scaled_prices : (T, 7) StandardScaled price array for the episode
        t             : current timestep index
        device        : torch device

    Returns:
        feature: (EMBED_DIM,) numpy float32
    """
    segment = build_temporal_segment(scaled_prices, t, L=TEMPORAL_SEG_LEN)  # (L, 7)
    tensor  = torch.from_numpy(segment).unsqueeze(0).to(device)              # (1, L, 7)

    ttfe.eval()
    with torch.no_grad():
        feat = ttfe(tensor)                                                   # (1, F')
    return feat.squeeze(0).cpu().numpy()                                      # (F',)


# ─── Single episode rollout ───────────────────────────────────────────────────

def run_episode(
    env:           BESSEnvironment,
    raw_prices:    np.ndarray,
    scaled_prices: np.ndarray,
    agent:         SACAgent,
    buffer:        ReplayBuffer,
    ttfe:          TTFE,
    device:        torch.device,
    total_steps:   int,
    warmup_steps:  int,
    deterministic: bool = False,
) -> dict:
    """
    Execute one full episode (288 steps).

    Args:
        env            : BESSEnvironment instance (will be reset inside)
        raw_prices     : (T, 7) unscaled prices for this episode
        scaled_prices  : (T, 7) scaled prices (for TTFE input only)
        agent          : SACAgent
        buffer         : ReplayBuffer (push transitions here; None during eval)
        ttfe           : TTFE encoder
        device         : torch.device
        total_steps    : global step counter (determines warmup)
        warmup_steps   : number of random-action steps before SAC kicks in
        deterministic  : if True, use greedy actor (eval mode)

    Returns:
        dict with episode statistics:
            total_reward, n_violations, soc_min, soc_max, soc_mean, n_steps
    """
    T = raw_prices.shape[0]

    # Pre-compute ALL TTFE features for the episode in one batched forward pass
    all_segs = np.stack(
        [build_temporal_segment(scaled_prices, t, L=TEMPORAL_SEG_LEN) for t in range(T)],
        axis=0,
    )  # (T, L, NUM_MARKETS)
    seg_tensor = torch.from_numpy(all_segs).to(device)
    ttfe.eval()
    with torch.no_grad():
        all_feats = ttfe(seg_tensor).cpu().numpy()  # (T, EMBED_DIM)

    obs = env.reset(price_episode_raw=raw_prices, features=all_feats)

    total_reward  = 0.0
    n_violations  = 0
    soc_values    = [env.energy / CAPACITY_MWH]
    step_count    = 0

    for t in range(T):
        # Select action
        if (not deterministic) and (total_steps + t < warmup_steps):
            action = np.random.uniform(-1.0, 1.0, size=(agent.act_dim,)).astype(np.float32)
        else:
            action = agent.select_action(obs, deterministic=deterministic)

        # Step environment (new API: takes only raw_action)
        next_obs, reward, done, info = env.step(action)

        # Store transition in replay buffer (skip during eval)
        if buffer is not None:
            reward_clipped = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))
            buffer.push(obs, action, reward_clipped, next_obs, float(done))

        total_reward += reward
        n_violations += int(info["violated"])
        soc_values.append(info["soc"])
        step_count   += 1

        obs = next_obs

        if done:
            break

    soc_arr = np.array(soc_values, dtype=np.float32)
    return {
        "total_reward" : total_reward,
        "n_violations" : n_violations,
        "soc_min"      : float(soc_arr.min()),
        "soc_max"      : float(soc_arr.max()),
        "soc_mean"     : float(soc_arr.mean()),
        "n_steps"      : step_count,
    }


# ─── Evaluation pass ─────────────────────────────────────────────────────────

def evaluate(
    env:            BESSEnvironment,
    eval_raw:       list,
    eval_scaled:    list,
    agent:          SACAgent,
    ttfe:           TTFE,
    device:         torch.device,
) -> dict:
    """
    Run greedy rollouts on all evaluation episodes.

    Returns:
        dict with mean metrics over all eval episodes.
    """
    rewards    = []
    violations = []

    for raw_ep, sc_ep in zip(eval_raw, eval_scaled):
        stats = run_episode(
            env=env,
            raw_prices=raw_ep,
            scaled_prices=sc_ep,
            agent=agent,
            buffer=None,         # no replay storage during eval
            ttfe=ttfe,
            device=device,
            total_steps=int(1e9),  # always use policy (no warmup)
            warmup_steps=0,
            deterministic=True,
        )
        rewards.append(stats["total_reward"])
        violations.append(stats["n_violations"])

    return {
        "mean_reward"     : float(np.mean(rewards)),
        "std_reward"      : float(np.std(rewards)),
        "mean_violations" : float(np.mean(violations)),
        "n_episodes"      : len(rewards),
    }


# ─── Main training function ───────────────────────────────────────────────────

def train(
    num_episodes:      int   = 500_000,
    eval_every:        int   = 1_000,
    warmup_steps:      int   = 1000,
    grad_steps_per_ep: int   = 288,
    gpu_id:            int   = 26,
    save_dir:          str   = "outputs/checkpoints",
    log_dir:           str   = "outputs/logs",
    resume_ckpt:       str   = None,
    start_episode:     int   = 1,
) -> None:
    """
    Full SAC training loop.

    Args:
        num_episodes      : total number of training episodes (each = 1 trading day)
        eval_every        : frequency (in episodes) of evaluation runs and checkpointing
        warmup_steps      : number of environment steps with random actions before SAC starts
        grad_steps_per_ep : number of gradient updates performed after each episode
        gpu_id            : CUDA device index (e.g. 26 for A16)
        save_dir          : directory for model checkpoints
        log_dir           : directory for CSV logs
        resume_ckpt       : path to checkpoint to resume from (None = start fresh)
        start_episode     : episode number to start counting from (use with resume_ckpt)
    """
    # ── Device setup ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    print(f"[Trainer] Using device: {device}")

    # ── Resolve absolute output paths ─────────────────────────────────────────
    # Allow relative paths to be anchored at the project root (one level above src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(project_root, save_dir)
    if not os.path.isabs(log_dir):
        log_dir = os.path.join(project_root, log_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[Trainer] Loading data ...")
    data = load_all()

    train_raw_episodes    = iter_daily_episodes(data["train_raw"],    TIMESTEPS_PER_DAY)
    train_scaled_episodes = iter_daily_episodes(data["train_scaled"], TIMESTEPS_PER_DAY)
    eval_raw_episodes     = iter_daily_episodes(data["eval_raw"],     TIMESTEPS_PER_DAY)
    eval_scaled_episodes  = iter_daily_episodes(data["eval_scaled"],  TIMESTEPS_PER_DAY)

    n_train = len(train_raw_episodes)
    n_eval  = len(eval_raw_episodes)
    print(f"[Trainer] Train episodes: {n_train}  |  Eval episodes: {n_eval}")

    # ── Instantiate models ────────────────────────────────────────────────────
    ttfe  = TTFE().to(device)
    ttfe.eval()   # TTFE is used in inference mode during rollout

    agent = SACAgent(device=device)

    # ── Resume from checkpoint if provided ───────────────────────────────────
    if resume_ckpt is not None:
        if not os.path.isabs(resume_ckpt):
            resume_ckpt = os.path.join(project_root, resume_ckpt)
        agent.load(resume_ckpt)
        print(f"[Trainer] Resumed from checkpoint: {resume_ckpt}")
        # Fix B: override log_alpha from checkpoint — reset to match ALPHA_ENTROPY=0.05
        import math
        new_log_alpha = math.log(ALPHA_ENTROPY)
        agent.log_alpha = torch.tensor(
            new_log_alpha, dtype=torch.float32, device=device, requires_grad=True
        )
        agent.alpha_optimizer = torch.optim.Adam([agent.log_alpha], lr=LR_POLICY)
        print(f"[FIX B] log_alpha reset to {new_log_alpha:.4f}  (alpha={ALPHA_ENTROPY})")
        # Halve all optimizer LRs on resume (manual guard recovery)
        for opt in [agent.actor_optimizer, agent.critic1_optimizer,
                    agent.critic2_optimizer, agent.alpha_optimizer]:
            for g in opt.param_groups:
                g['lr'] = g['lr'] * 0.5
        print(f"[RESUME] All optimizer LRs halved.")
        # Load TTFE weights if saved in checkpoint; else use reproducible seed=0 init
        _ckpt_data = torch.load(resume_ckpt, map_location=device)
        if "ttfe_state" in _ckpt_data:
            ttfe.load_state_dict(_ckpt_data["ttfe_state"])
            print(f"[RESUME] TTFE weights loaded from checkpoint.")
        else:
            torch.manual_seed(0)
            ttfe = TTFE().to(device)
            ttfe.eval()
            print(f"[RESUME] No TTFE in checkpoint — reinitialised with seed=0.")

    buffer = ReplayBuffer(
        capacity=REPLAY_BUFFER_SIZE,
        device=device,
    )

    # ── Environment (reused across episodes via reset()) ──────────────────────
    # Initialise with the first training episode as a placeholder
    env = BESSEnvironment(mode="joint")

    # ── CSV loggers ───────────────────────────────────────────────────────────
    train_logger = CSVLogger(
        path=os.path.join(log_dir, "train_log.csv"),
        fieldnames=[
            "episode", "total_reward", "n_violations",
            "soc_min", "soc_max", "soc_mean",
            "mean_q1_loss", "mean_q2_loss",
            "mean_actor_loss", "mean_alpha_loss", "mean_alpha",
        ],
    )
    eval_logger = CSVLogger(
        path=os.path.join(log_dir, "eval_log.csv"),
        fieldnames=[
            "episode", "mean_reward", "std_reward", "mean_violations",
        ],
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    total_steps       = 0
    t_start           = time.time()
    best_eval_reward  = -float('inf')
    best_eval_episode = 0

    for ep in range(start_episode, start_episode + num_episodes):

        # Sample a random training day
        idx          = random.randint(0, n_train - 1)
        raw_ep       = train_raw_episodes[idx]
        sc_ep        = train_scaled_episodes[idx]

        # Collect one episode of experience
        ep_stats = run_episode(
            env=env,
            raw_prices=raw_ep,
            scaled_prices=sc_ep,
            agent=agent,
            buffer=buffer,
            ttfe=ttfe,
            device=device,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            deterministic=False,
        )
        total_steps += ep_stats["n_steps"]

        # ── Gradient updates ──────────────────────────────────────────────────
        q1_losses, q2_losses, actor_losses, alpha_losses, alphas = [], [], [], [], []

        if len(buffer) >= BATCH_SIZE:
            for _ in range(grad_steps_per_ep):
                batch = buffer.sample(BATCH_SIZE)
                loss_dict = agent.update(batch)
                q1_losses.append(loss_dict["q1_loss"])
                q2_losses.append(loss_dict["q2_loss"])
                actor_losses.append(loss_dict["actor_loss"])
                alpha_losses.append(loss_dict["alpha_loss"])
                alphas.append(loss_dict["alpha"])

        # ── Logging ───────────────────────────────────────────────────────────
        mean_q1    = float(np.mean(q1_losses))    if q1_losses    else 0.0
        mean_q2    = float(np.mean(q2_losses))    if q2_losses    else 0.0
        mean_actor = float(np.mean(actor_losses)) if actor_losses else 0.0
        mean_alpha_loss = float(np.mean(alpha_losses)) if alpha_losses else 0.0
        mean_alpha = float(np.mean(alphas))        if alphas        else float(agent.log_alpha.exp().item())

        train_logger.write({
            "episode"         : ep,
            "total_reward"    : f"{ep_stats['total_reward']:.4f}",
            "n_violations"    : ep_stats["n_violations"],
            "soc_min"         : f"{ep_stats['soc_min']:.4f}",
            "soc_max"         : f"{ep_stats['soc_max']:.4f}",
            "soc_mean"        : f"{ep_stats['soc_mean']:.4f}",
            "mean_q1_loss"    : f"{mean_q1:.6f}",
            "mean_q2_loss"    : f"{mean_q2:.6f}",
            "mean_actor_loss" : f"{mean_actor:.6f}",
            "mean_alpha_loss" : f"{mean_alpha_loss:.6f}",
            "mean_alpha"      : f"{mean_alpha:.6f}",
        })

        # Console progress
        elapsed = time.time() - t_start
        print(
            f"[Ep {ep:>4d}/{num_episodes}] "
            f"reward={ep_stats['total_reward']:>9.2f}  "
            f"viol={ep_stats['n_violations']:>3d}  "
            f"Q1={mean_q1:.4f}  actor={mean_actor:.4f}  "
            f"alpha={mean_alpha:.4f}  "
            f"buf={len(buffer):>7d}  "
            f"t={elapsed:.0f}s"
        )

        # ── Save checkpoint every episode (overwrite latest) ─────────────────
        _save_checkpoint(agent, ttfe, os.path.join(save_dir, "sac_latest.pt"))

        # ── Periodic evaluation and milestone checkpoint ───────────────────────
        if ep % eval_every == 0:
            print(f"[Trainer] --- Evaluation at episode {ep} ---")
            eval_stats = evaluate(
                env=env,
                eval_raw=eval_raw_episodes,
                eval_scaled=eval_scaled_episodes,
                agent=agent,
                ttfe=ttfe,
                device=device,
            )
            mean_reward    = eval_stats['mean_reward']
            mean_violations = eval_stats['mean_violations']

            eval_logger.write({
                "episode"          : ep,
                "mean_reward"      : f"{mean_reward:.4f}",
                "std_reward"       : f"{eval_stats['std_reward']:.4f}",
                "mean_violations"  : f"{mean_violations:.2f}",
            })

            # ── Best model tracking ───────────────────────────────────────────
            if mean_reward > best_eval_reward:
                best_eval_reward  = mean_reward
                best_eval_episode = ep
                best_path = os.path.join(save_dir, "best_model.pt")
                _save_checkpoint(agent, ttfe, best_path)
                print(
                    f"[BEST] ep {ep}: reward={mean_reward:.1f}, "
                    f"violations={mean_violations:.2f}  → saved best_model.pt"
                )

            print(
                f"[EVAL] ep={ep} | reward={mean_reward:.1f} | "
                f"std={eval_stats['std_reward']:.1f} | "
                f"violations={mean_violations:.2f} | "
                f"best={best_eval_reward:.1f} (ep{best_eval_episode})"
            )

            # ── Divergence guard ─────────────────────────────────────────────
            if mean_violations > 15.0 and ep > 200 and best_eval_reward > 0:
                print(
                    f"[GUARD] Violations={mean_violations:.1f} exceeded threshold "
                    f"at ep {ep}. Reloading best checkpoint from ep {best_eval_episode}."
                )
                agent.load(os.path.join(save_dir, "best_model.pt"))
                for opt in [agent.actor_optimizer,
                            agent.critic1_optimizer, agent.critic2_optimizer,
                            agent.alpha_optimizer]:
                    for g in opt.param_groups:
                        g['lr'] = g['lr'] * 0.5
                print(f"[GUARD] Learning rates halved after reload.")

            # Save milestone checkpoint
            ckpt_path = os.path.join(save_dir, f"sac_ep{ep}.pt")
            _save_checkpoint(agent, ttfe, ckpt_path)

    print(f"[Trainer] Training complete. Total steps: {total_steps:,}")
    # Final checkpoint
    _save_checkpoint(agent, ttfe, os.path.join(save_dir, "sac_final.pt"))
