"""
main_stage2.py — TempDRL Stage 2: RTC+B Regime Shift Adaptation.

Transfers Stage 1 best checkpoint to the 6-market post-RTC+B setting:
  - TTFE_S2: MHA weights from Stage 1, feature_embedding re-initialized (5→12)
  - Actor:   fresh 78-dim → 9-dim (ECRS output dim near-zero init)
  - Critics: fresh initialization (pre-RTC+B Q-values invalid)

Progressive TTFE Unfreezing:
  Phase A (ep   1–499): TTFE frozen  — only actor/critics train
  Phase B (ep 500–1499): top MHA unfrozen at lr=1e-5
  Phase C (ep 1500–3000): full TTFE unfreeze at lr=1e-5

Replay buffer stores raw states + temporal segments → TTFE runs online in updates.
Total: 3,000 episodes on ~70 post-RTC+B days.
Hardware: NVIDIA A16, CUDA 12.4, GPU index 26
"""

import os
import sys
import time
import random
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import (
    EMBED_DIM, HIDDEN_DIM, NUM_HIDDEN_LAYERS,
    GAMMA, TAU_TARGET, ALPHA_ENTROPY,
    E_MIN_MWH, E_MAX_MWH, E_INIT_MWH, CAPACITY_MWH,
    DT_H, EFF_CH, EFF_DCH, TIMESTEPS_PER_DAY,
    REWARD_CLIP,
)
from config_stage2 import (
    STATE_DIM_S2, ACTION_DIM_S2,
    STAGE2_EPISODES, STAGE2_BATCH_SIZE, STAGE2_BUFFER_SIZE,
    GRAD_STEPS_PER_EP_S2, EVAL_EVERY_S2,
    LR_ACTOR_S2, LR_CRITIC_S2, LR_TTFE_TOP, LR_TTFE_FULL,
    TARGET_ENTROPY_S2, REWARD_CLIP_S2,
    UNFREEZE_TOP_AT_EP, UNFREEZE_FULL_AT_EP,
    TTFE_SEG_LEN_S2, NUM_PRICE_DIMS, NUM_SYSCOND_DIMS,
    STAGE2_CKPT_DIR, STAGE2_LOG_DIR, STAGE1_CKPT_PATH,
    MAX_VIOLATIONS_PER_EP,
)
from ttfe_stage2 import TTFE_S2, build_ttfe_s2
from sac_agent import Actor, Critic
from data_loader_stage2 import (
    load_stage2_data, iter_daily_episodes_s2,
    build_temporal_segment_12, build_state_78, build_time_6,
)
from environment_stage2 import (
    BESSEnvironment_S2, decode_action_s2, compute_step_revenue_s2,
    hourly_soc_bounds_s2,
)

# ── SAC constants ──────────────────────────────────────────────────────────────
LOG_STD_MIN = -20
LOG_STD_MAX = 2


# ── Stage 2 Replay Buffer ─────────────────────────────────────────────────────

class ReplayBufferS2:
    """
    Stores (raw_obs_14, segment_32x12, action_9, reward, next_raw_obs_14,
            next_segment_32x12, done).

    raw_obs = [SoC(1), syscond_norm(7), time_6(6)] = 14 dims.
    Temporal segments stored raw (32×12) so TTFE can be fine-tuned end-to-end.
    TTFE features are NOT stored — they are computed online during updates.
    """

    RAW_OBS_DIM  = 1 + NUM_SYSCOND_DIMS + 6    # 14
    SEG_FLAT     = TTFE_SEG_LEN_S2 * NUM_PRICE_DIMS  # 32×12 = 384

    def __init__(self, capacity: int, device: torch.device):
        self.capacity = capacity
        self.device   = device
        self._obs     = np.zeros((capacity, self.RAW_OBS_DIM), dtype=np.float32)
        self._seg     = np.zeros((capacity, self.SEG_FLAT),    dtype=np.float32)
        self._acts    = np.zeros((capacity, ACTION_DIM_S2),    dtype=np.float32)
        self._rew     = np.zeros((capacity, 1),                dtype=np.float32)
        self._nobs    = np.zeros((capacity, self.RAW_OBS_DIM), dtype=np.float32)
        self._nseg    = np.zeros((capacity, self.SEG_FLAT),    dtype=np.float32)
        self._done    = np.zeros((capacity, 1),                dtype=np.float32)
        self._ptr  = 0
        self._size = 0

    def push(self, obs, seg, action, reward, next_obs, next_seg, done):
        i = self._ptr
        self._obs[i]  = obs.astype(np.float32)
        self._seg[i]  = seg.reshape(-1).astype(np.float32)
        self._acts[i] = action.astype(np.float32)
        self._rew[i]  = float(reward)
        self._nobs[i] = next_obs.astype(np.float32)
        self._nseg[i] = next_seg.reshape(-1).astype(np.float32)
        self._done[i] = float(done)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        assert self._size >= batch_size
        idxs = np.random.randint(0, self._size, size=batch_size)
        def _t(arr):
            return torch.from_numpy(arr[idxs]).to(self.device)
        return {
            "raw_obs":   _t(self._obs),
            "segments":  _t(self._seg).view(batch_size, TTFE_SEG_LEN_S2, NUM_PRICE_DIMS),
            "actions":   _t(self._acts),
            "rewards":   _t(self._rew),
            "raw_nobs":  _t(self._nobs),
            "nsegments": _t(self._nseg).view(batch_size, TTFE_SEG_LEN_S2, NUM_PRICE_DIMS),
            "dones":     _t(self._done),
        }

    def __len__(self): return self._size


# ── Model initialization ──────────────────────────────────────────────────────

def load_stage1_and_upgrade(device: torch.device):
    """
    1. Build TTFE_S2 with Stage 1 MHA weights (embedding re-initialized).
    2. Create fresh Actor(78→9) and Critics(78+9→1).
    3. Initialize ECRS output dim of actor near zero.
    Returns: (ttfe_s2, actor, critic1, critic2, critic1_tgt, critic2_tgt)
    """
    # TTFE: transfer MHA weights from Stage 1, freeze all (Phase A)
    ttfe = build_ttfe_s2(checkpoint_path=STAGE1_CKPT_PATH, device=device)

    # Fresh actor and critics with Stage 2 dimensions
    actor    = Actor(obs_dim=STATE_DIM_S2, act_dim=ACTION_DIM_S2,
                     hidden_dim=HIDDEN_DIM, num_hidden=NUM_HIDDEN_LAYERS).to(device)
    critic1  = Critic(obs_dim=STATE_DIM_S2, act_dim=ACTION_DIM_S2,
                      hidden_dim=HIDDEN_DIM, num_hidden=NUM_HIDDEN_LAYERS).to(device)
    critic2  = Critic(obs_dim=STATE_DIM_S2, act_dim=ACTION_DIM_S2,
                      hidden_dim=HIDDEN_DIM, num_hidden=NUM_HIDDEN_LAYERS).to(device)

    # Target critics: hard copies, no grad
    critic1_tgt = Critic(obs_dim=STATE_DIM_S2, act_dim=ACTION_DIM_S2,
                         hidden_dim=HIDDEN_DIM, num_hidden=NUM_HIDDEN_LAYERS).to(device)
    critic2_tgt = Critic(obs_dim=STATE_DIM_S2, act_dim=ACTION_DIM_S2,
                         hidden_dim=HIDDEN_DIM, num_hidden=NUM_HIDDEN_LAYERS).to(device)
    critic1_tgt.load_state_dict(critic1.state_dict())
    critic2_tgt.load_state_dict(critic2.state_dict())
    for p in critic1_tgt.parameters(): p.requires_grad_(False)
    for p in critic2_tgt.parameters(): p.requires_grad_(False)

    return ttfe, actor, critic1, critic2, critic1_tgt, critic2_tgt


def init_near_zero_ecrs(actor: Actor, ecrs_idx: int = 7, scale: float = 1e-3):
    """
    Zeroes (near-zero) the ECRS action dimension in actor's output heads.
    Actor starts as a functional 5-market policy; ECRS gradually activates.

    ecrs_idx: index of ECRS in the 9-dim action vector (index 7).
    """
    with torch.no_grad():
        actor.mean_head.weight[ecrs_idx].mul_(scale)
        actor.mean_head.bias[ecrs_idx].zero_()
        actor.log_std_head.weight[ecrs_idx].mul_(scale)
        actor.log_std_head.bias[ecrs_idx].zero_()
    print(f"[Stage2] Actor ECRS output dim (idx {ecrs_idx}) initialized near-zero")


# ── Build full state from raw obs + TTFE features ─────────────────────────────

def make_full_state(raw_obs: np.ndarray, ttfe_feat: np.ndarray) -> np.ndarray:
    """
    Combines TTFE features and raw_obs into 78-dim state.
    raw_obs: [SoC(1), syscond(7), time(6)] = 14 dims
    full: [ttfe(64), syscond(7), time(6), SoC(1)] = 78 dims
    """
    soc        = raw_obs[0:1]
    syscond_7  = raw_obs[1:8]
    time_6     = raw_obs[8:14]
    return np.concatenate([ttfe_feat, syscond_7, time_6, soc]).astype(np.float32)


def make_full_state_batch(
    raw_obs_batch: torch.Tensor,
    ttfe_feat_batch: torch.Tensor,
) -> torch.Tensor:
    """
    Batch version of make_full_state.
    raw_obs_batch: (B, 14) — [SoC, syscond(7), time(6)]
    ttfe_feat_batch: (B, 64)
    Returns: (B, 78)
    """
    soc       = raw_obs_batch[:, 0:1]         # (B, 1)
    syscond_7 = raw_obs_batch[:, 1:8]         # (B, 7)
    time_6    = raw_obs_batch[:, 8:14]        # (B, 6)
    return torch.cat([ttfe_feat_batch, syscond_7, time_6, soc], dim=-1)  # (B, 78)


# ── SAC update (Stage 2 — TTFE runs online) ───────────────────────────────────

def sac_update_s2(
    batch:       dict,
    ttfe:        TTFE_S2,
    actor:       Actor,
    critic1:     Critic,
    critic2:     Critic,
    critic1_tgt: Critic,
    critic2_tgt: Critic,
    log_alpha:   torch.Tensor,
    actor_opt:   torch.optim.Optimizer,
    critic1_opt: torch.optim.Optimizer,
    critic2_opt: torch.optim.Optimizer,
    alpha_opt:   torch.optim.Optimizer,
    ttfe_opt:    torch.optim.Optimizer,
    gamma:       float = GAMMA,
    tau:         float = TAU_TARGET,
    target_ent:  float = TARGET_ENTROPY_S2,
    train_ttfe:  bool  = False,
) -> dict:
    """
    One SAC gradient step for Stage 2. TTFE runs online so gradients can
    flow back to TTFE in Phase B/C.

    batch keys: raw_obs(B,14), segments(B,32,12), actions(B,9),
                rewards(B,1), raw_nobs(B,14), nsegments(B,32,12), dones(B,1)
    """
    raw_obs   = batch["raw_obs"]
    segs      = batch["segments"]
    actions   = batch["actions"]
    rewards   = batch["rewards"]
    raw_nobs  = batch["raw_nobs"]
    nsegs     = batch["nsegments"]
    dones     = batch["dones"]

    alpha = log_alpha.exp().detach()

    # ── Build full states via TTFE (online) ────────────────────────────────────
    if train_ttfe:
        ttfe_feat  = ttfe(segs)      # (B, 64), gradients flow
        ttfe_nfeat = ttfe(nsegs)
    else:
        with torch.no_grad():
            ttfe_feat  = ttfe(segs)
            ttfe_nfeat = ttfe(nsegs)

    obs      = make_full_state_batch(raw_obs,  ttfe_feat)   # (B, 78)
    next_obs = make_full_state_batch(raw_nobs, ttfe_nfeat)  # (B, 78)

    # ── Critic update ──────────────────────────────────────────────────────────
    with torch.no_grad():
        next_acts, next_log_pi = actor.sample(next_obs.detach())
        q1_nxt = critic1_tgt(next_obs.detach(), next_acts)
        q2_nxt = critic2_tgt(next_obs.detach(), next_acts)
        q_tgt  = rewards + gamma * (1.0 - dones) * (
            torch.min(q1_nxt, q2_nxt) - alpha * next_log_pi
        )

    q1_pred  = critic1(obs.detach(), actions)
    q2_pred  = critic2(obs.detach(), actions)
    q1_loss  = F.mse_loss(q1_pred, q_tgt)
    q2_loss  = F.mse_loss(q2_pred, q_tgt)

    critic1_opt.zero_grad(); q1_loss.backward()
    nn.utils.clip_grad_norm_(critic1.parameters(), 1.0)
    critic1_opt.step()

    critic2_opt.zero_grad(); q2_loss.backward()
    nn.utils.clip_grad_norm_(critic2.parameters(), 1.0)
    critic2_opt.step()

    # ── Actor update ──────────────────────────────────────────────────────────
    if train_ttfe:
        # Re-compute with grad for actor loss + TTFE fine-tune
        ttfe_feat_actor = ttfe(segs)
        obs_actor = make_full_state_batch(raw_obs, ttfe_feat_actor)
    else:
        obs_actor = obs.detach()

    pi_acts, log_pi = actor.sample(obs_actor)
    q1_pi = critic1(obs_actor, pi_acts)
    q2_pi = critic2(obs_actor, pi_acts)
    actor_loss = (alpha * log_pi - torch.min(q1_pi, q2_pi)).mean()

    actor_opt.zero_grad()
    if train_ttfe and ttfe_opt is not None:
        ttfe_opt.zero_grad()
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    if train_ttfe and ttfe_opt is not None:
        nn.utils.clip_grad_norm_(ttfe.trainable_params(), 1.0)
        ttfe_opt.step()
    actor_opt.step()

    # ── Alpha update ──────────────────────────────────────────────────────────
    alpha_loss = -(log_alpha * (log_pi.detach() + target_ent)).mean()
    alpha_opt.zero_grad(); alpha_loss.backward()
    nn.utils.clip_grad_norm_([log_alpha], 1.0)
    alpha_opt.step()

    # ── Soft target update ────────────────────────────────────────────────────
    for tp, p in zip(critic1_tgt.parameters(), critic1.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
    for tp, p in zip(critic2_tgt.parameters(), critic2.parameters()):
        tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    return {
        "q1_loss":    q1_loss.item(),
        "q2_loss":    q2_loss.item(),
        "actor_loss": actor_loss.item(),
        "alpha_loss": alpha_loss.item(),
        "alpha":      log_alpha.exp().item(),
    }


# ── Evaluation (greedy rollout) ───────────────────────────────────────────────

@torch.no_grad()
def evaluate_s2(
    ttfe:       TTFE_S2,
    actor:      Actor,
    episodes:   list,
    device:     torch.device,
    mode:       str = "joint",
) -> float:
    """Greedy rollout on val episodes. Returns mean USD/day."""
    ttfe.eval(); actor.eval()
    total_rev = []

    for ep in episodes:
        prices_raw = ep["prices"]     # (288, 12) raw
        prices_sc  = ep.get("prices_sc", prices_raw)  # scaled for TTFE
        syscond_sc = ep["syscond"]    # (288, 7) normalized
        dow        = ep["day_of_week"]
        month_     = ep["month"]
        T          = len(prices_raw)

        # Pre-compute TTFE features
        segs = np.stack([
            build_temporal_segment_12(prices_sc, t) for t in range(T)
        ])
        feats = ttfe(torch.FloatTensor(segs).to(device)).cpu().numpy()

        energy    = E_INIT_MWH
        ep_rev    = 0.0
        ema_spot  = float(prices_raw[0, 0])

        for t in range(T):
            soc    = energy / CAPACITY_MWH
            time_6 = build_time_6(t, dow, month_)
            raw_obs = np.concatenate([[soc], syscond_sc[t], time_6]).astype(np.float32)
            state   = make_full_state(raw_obs, feats[t])

            raw_act = actor.get_mean_action(
                torch.FloatTensor(state).unsqueeze(0).to(device)
            ).squeeze(0).cpu().numpy()

            (v_dch, v_ch, p_spot_dch, p_spot_ch,
             p_regup, p_regdn, p_rrs, p_ecrs, p_nsrs) = decode_action_s2(raw_act)

            if mode == "spot":
                p_regup = p_regdn = p_rrs = p_ecrs = p_nsrs = 0.0
            elif mode == "as":
                p_spot_dch = p_spot_ch = 0.0

            rev = compute_step_revenue_s2(
                v_dch, v_ch, p_spot_dch, p_spot_ch,
                p_regup, p_regdn, p_rrs, p_ecrs, p_nsrs,
                prices_raw[t],
            )
            ep_rev += rev["total"]
            new_e   = energy + DT_H * (EFF_CH * p_spot_ch - (1.0/EFF_DCH) * p_spot_dch)
            energy  = float(np.clip(new_e, E_MIN_MWH, E_MAX_MWH))

        total_rev.append(ep_rev)

    ttfe.train(); actor.train()
    return float(np.mean(total_rev)) if total_rev else 0.0


# ── Checkpoint save / load ────────────────────────────────────────────────────

def save_checkpoint_s2(path, ttfe, actor, critic1, critic2,
                        critic1_tgt, critic2_tgt, log_alpha, episode):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "episode":             episode,
        "ttfe_s2_state":       ttfe.state_dict(),
        "actor_state":         actor.state_dict(),
        "critic1_state":       critic1.state_dict(),
        "critic2_state":       critic2.state_dict(),
        "critic1_target_state": critic1_tgt.state_dict(),
        "critic2_target_state": critic2_tgt.state_dict(),
        "log_alpha":           log_alpha.detach().cpu(),
    }, path)


def load_checkpoint_s2(path, ttfe, actor, critic1, critic2,
                        critic1_tgt, critic2_tgt, log_alpha, device):
    ckpt = torch.load(path, map_location=device)
    ttfe.load_state_dict(ckpt["ttfe_s2_state"])
    actor.load_state_dict(ckpt["actor_state"])
    critic1.load_state_dict(ckpt["critic1_state"])
    critic2.load_state_dict(ckpt["critic2_state"])
    critic1_tgt.load_state_dict(ckpt["critic1_target_state"])
    critic2_tgt.load_state_dict(ckpt["critic2_target_state"])
    log_alpha.data = ckpt["log_alpha"].to(device)
    return ckpt.get("episode", 0)


# ── Main training loop ────────────────────────────────────────────────────────

def run_stage2(
    num_episodes:    int   = STAGE2_EPISODES,
    batch_size:      int   = STAGE2_BATCH_SIZE,
    grad_steps:      int   = GRAD_STEPS_PER_EP_S2,
    eval_every:      int   = EVAL_EVERY_S2,
    gpu_id:          int   = 26,
    save_dir:        str   = STAGE2_CKPT_DIR,
    log_dir:         str   = STAGE2_LOG_DIR,
):
    t_start = time.time()
    device  = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*68}")
    print("  TempDRL Stage 2 — RTC+B Regime Shift Adaptation")
    print(f"  Phase A: TTFE frozen   (ep 1–{UNFREEZE_TOP_AT_EP-1})")
    print(f"  Phase B: top MHA lr={LR_TTFE_TOP} (ep {UNFREEZE_TOP_AT_EP}–{UNFREEZE_FULL_AT_EP-1})")
    print(f"  Phase C: full unfreeze (ep {UNFREEZE_FULL_AT_EP}–{num_episodes})")
    print(f"  Device: {device}")
    print(f"{'='*68}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[Stage2] Loading post-RTC+B data...")
    data = load_stage2_data()
    train_eps = iter_daily_episodes_s2(
        data["train_prices"], data["train_syscond_sc"],
        index=data["train_index"],
    )
    val_eps = iter_daily_episodes_s2(
        data["val_prices"], data["val_syscond_sc"],
        index=data["val_index"],
    )
    # Add scaled prices to episodes for TTFE input
    n_train_days = len(data["train_prices"]) // TIMESTEPS_PER_DAY
    n_val_days   = len(data["val_prices"])   // TIMESTEPS_PER_DAY
    for i, ep in enumerate(train_eps):
        s = i * TIMESTEPS_PER_DAY
        ep["prices_sc"] = data["train_prices_sc"][s:s + TIMESTEPS_PER_DAY]
    for i, ep in enumerate(val_eps):
        s = i * TIMESTEPS_PER_DAY
        ep["prices_sc"] = data["val_prices_sc"][s:s + TIMESTEPS_PER_DAY]

    print(f"[Stage2] Train: {len(train_eps)} days | Val: {len(val_eps)} days")

    if len(train_eps) == 0:
        raise RuntimeError("[Stage2] No training episodes. "
                           "Check post-RTC+B data in data/processed/")

    # ── Build models ──────────────────────────────────────────────────────────
    ttfe, actor, critic1, critic2, critic1_tgt, critic2_tgt = \
        load_stage1_and_upgrade(device)
    init_near_zero_ecrs(actor, ecrs_idx=7)
    ttfe.train(); actor.train(); critic1.train(); critic2.train()

    log_alpha = torch.tensor(
        np.log(ALPHA_ENTROPY), dtype=torch.float32, device=device, requires_grad=True
    )

    # ── Optimizers ────────────────────────────────────────────────────────────
    actor_opt   = torch.optim.Adam(actor.parameters(),   lr=LR_ACTOR_S2)
    critic1_opt = torch.optim.Adam(critic1.parameters(), lr=LR_CRITIC_S2)
    critic2_opt = torch.optim.Adam(critic2.parameters(), lr=LR_CRITIC_S2)
    alpha_opt   = torch.optim.Adam([log_alpha],           lr=LR_ACTOR_S2)
    # TTFE optimizer: None for Phase A (frozen); instantiated at Phase B/C
    ttfe_opt    = None

    # ── Replay buffer ─────────────────────────────────────────────────────────
    buffer      = ReplayBufferS2(capacity=STAGE2_BUFFER_SIZE, device=device)
    best_val    = -float("inf")
    best_path   = os.path.join(save_dir, "best_model_s2.pt")
    current_phase = "A"

    # ── Logging ───────────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)
    train_log_path = os.path.join(log_dir, "train_stage2.log")
    eval_log_path  = os.path.join(log_dir, "eval_stage2.csv")
    eval_header    = ["episode", "val_usd_day", "phase", "alpha", "wall_time_s"]
    with open(eval_log_path, "w", newline="") as f:
        csv.writer(f).writerow(eval_header)

    def _log(msg):
        print(msg, flush=True)
        with open(train_log_path, "a") as f:
            f.write(msg + "\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    for ep in range(1, num_episodes + 1):

        # ── Phase transitions ─────────────────────────────────────────────────
        if ep == UNFREEZE_TOP_AT_EP and current_phase == "A":
            current_phase = "B"
            ttfe.unfreeze_top_layer()
            ttfe.train()
            ttfe_opt = torch.optim.Adam(ttfe.trainable_params(), lr=LR_TTFE_TOP)
            _log(f"[Phase B] ep {ep}: top MHA layer unfrozen (lr={LR_TTFE_TOP})")

        elif ep == UNFREEZE_FULL_AT_EP and current_phase == "B":
            current_phase = "C"
            ttfe.unfreeze_all()
            ttfe.train()
            ttfe_opt = torch.optim.Adam(ttfe.trainable_params(), lr=LR_TTFE_FULL)
            _log(f"[Phase C] ep {ep}: all TTFE params unfrozen (lr={LR_TTFE_FULL})")

        train_ttfe = (current_phase in ("B", "C"))

        # ── Sample episode ────────────────────────────────────────────────────
        ep_data    = random.choice(train_eps)
        prices_raw = ep_data["prices"]      # (288, 12)
        prices_sc  = ep_data["prices_sc"]   # (288, 12) scaled
        syscond_sc = ep_data["syscond"]     # (288, 7)
        dow        = ep_data["day_of_week"]
        month_     = ep_data["month"]
        T          = len(prices_raw)

        # Pre-compute TTFE features (frozen Phase A; used for rollout only)
        with torch.no_grad():
            segs_ep = np.stack([
                build_temporal_segment_12(prices_sc, t) for t in range(T)
            ])
            feats_ep = ttfe(
                torch.FloatTensor(segs_ep).to(device)
            ).cpu().numpy()   # (288, 64)

        # ── Rollout ───────────────────────────────────────────────────────────
        energy   = E_INIT_MWH
        ema_spot = float(prices_raw[0, 0])
        ep_rev   = 0.0
        n_vio    = 0

        for t in range(T):
            soc    = energy / CAPACITY_MWH
            time_6 = build_time_6(t, dow, month_)
            raw_obs = np.concatenate([[soc], syscond_sc[t], time_6]).astype(np.float32)
            state   = make_full_state(raw_obs, feats_ep[t])
            seg_t   = segs_ep[t]   # (32, 12)

            # Action selection
            with torch.no_grad():
                s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                raw_act, _ = actor.sample(s_t)
                raw_act    = raw_act.squeeze(0).cpu().numpy()

            (v_dch, v_ch, p_spot_dch, p_spot_ch,
             p_regup, p_regdn, p_rrs, p_ecrs, p_nsrs) = decode_action_s2(raw_act)

            e_min_eff, e_max_eff = hourly_soc_bounds_s2(
                t, p_regup, p_regdn, p_rrs, p_ecrs, p_nsrs
            )
            delta_e    = DT_H * (EFF_CH * p_spot_ch - (1.0/EFF_DCH) * p_spot_dch)
            new_energy = energy + delta_e
            violated   = (new_energy < e_min_eff) or (new_energy > e_max_eff)

            if violated:
                new_energy = float(np.clip(new_energy, e_min_eff, e_max_eff))
                p_spot_dch = p_spot_ch = 0.0
                p_regup = p_regdn = p_rrs = p_ecrs = p_nsrs = 0.0
                v_dch = v_ch = 0
                n_vio += 1

            energy = float(new_energy)

            from environment_stage2 import compute_shaped_reward_s2
            reward = compute_shaped_reward_s2(
                v_dch, v_ch, p_spot_dch, p_spot_ch,
                p_regup, p_regdn, p_rrs, p_ecrs, p_nsrs,
                prices_raw[t], ema_spot, violated,
            )
            reward = float(np.clip(reward, -REWARD_CLIP_S2, REWARD_CLIP_S2))

            rev     = compute_step_revenue_s2(
                v_dch, v_ch, p_spot_dch, p_spot_ch,
                p_regup, p_regdn, p_rrs, p_ecrs, p_nsrs, prices_raw[t],
            )
            ep_rev += rev["total"]

            # Next state
            t_next    = t + 1
            done      = (t_next >= T)
            if not done:
                soc_next   = energy / CAPACITY_MWH
                time_6_nx  = build_time_6(t_next, dow, month_)
                nraw_obs   = np.concatenate([[soc_next], syscond_sc[t_next], time_6_nx]).astype(np.float32)
                nseg_t     = segs_ep[t_next]
            else:
                nraw_obs = np.zeros(ReplayBufferS2.RAW_OBS_DIM, dtype=np.float32)
                nseg_t   = np.zeros((TTFE_SEG_LEN_S2, NUM_PRICE_DIMS), dtype=np.float32)

            buffer.push(raw_obs, seg_t, raw_act, reward, nraw_obs, nseg_t, done)
            ema_spot = 0.99 * ema_spot + 0.01 * float(prices_raw[t, 0])

        # ── Divergence guard ─────────────────────────────────────────────────
        if n_vio > MAX_VIOLATIONS_PER_EP and os.path.exists(best_path):
            _log(f"[Stage2] ep {ep}: violations={n_vio} > {MAX_VIOLATIONS_PER_EP} — "
                 f"reloading best checkpoint + halving LRs")
            load_checkpoint_s2(best_path, ttfe, actor, critic1, critic2,
                               critic1_tgt, critic2_tgt, log_alpha, device)
            for g in actor_opt.param_groups:   g["lr"] /= 2.0
            for g in critic1_opt.param_groups: g["lr"] /= 2.0
            for g in critic2_opt.param_groups: g["lr"] /= 2.0

        # ── Gradient updates ──────────────────────────────────────────────────
        if len(buffer) >= batch_size:
            for _ in range(grad_steps):
                batch = buffer.sample(batch_size)
                sac_update_s2(
                    batch, ttfe, actor, critic1, critic2,
                    critic1_tgt, critic2_tgt, log_alpha,
                    actor_opt, critic1_opt, critic2_opt, alpha_opt, ttfe_opt,
                    train_ttfe=train_ttfe,
                )

        # ── Logging ───────────────────────────────────────────────────────────
        if ep % 10 == 0:
            wall = time.time() - t_start
            _log(f"ep {ep:5d} | phase={current_phase} | rev=${ep_rev:.2f} | "
                 f"vio={n_vio:3d} | buf={len(buffer)} | "
                 f"alpha={log_alpha.exp().item():.4f} | t={wall:.0f}s")

        # ── Evaluation ────────────────────────────────────────────────────────
        if ep % eval_every == 0 and val_eps:
            val_rev = evaluate_s2(ttfe, actor, val_eps, device, mode="joint")
            wall    = time.time() - t_start
            _log(f"  [EVAL] ep {ep:5d} | val=${val_rev:.2f}/day | "
                 f"phase={current_phase} | best=${best_val:.2f} | t={wall:.0f}s")

            with open(eval_log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    ep, f"{val_rev:.4f}", current_phase,
                    f"{log_alpha.exp().item():.5f}", f"{wall:.1f}",
                ])

            if val_rev > best_val:
                best_val = val_rev
                save_checkpoint_s2(
                    best_path, ttfe, actor, critic1, critic2,
                    critic1_tgt, critic2_tgt, log_alpha, ep,
                )
                _log(f"  [BEST] Saved new best: ${best_val:.2f}/day → {best_path}")

        # Periodic checkpoint
        if ep % 500 == 0:
            ckpt_path = os.path.join(save_dir, f"stage2_ep{ep}.pt")
            save_checkpoint_s2(
                ckpt_path, ttfe, actor, critic1, critic2,
                critic1_tgt, critic2_tgt, log_alpha, ep,
            )

    wall = time.time() - t_start
    print(f"\n[Stage2] Training complete. Best val: ${best_val:.2f}/day | "
          f"wall time: {wall:.0f}s")
    print(f"[Stage2] Best checkpoint: {best_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("  TempDRL Stage 2 — RTC+B Regime Shift Adaptation")
    print("=" * 68)

    run_stage2(
        num_episodes = STAGE2_EPISODES,
        batch_size   = STAGE2_BATCH_SIZE,
        grad_steps   = GRAD_STEPS_PER_EP_S2,
        eval_every   = EVAL_EVERY_S2,
        gpu_id       = 26,
        save_dir     = STAGE2_CKPT_DIR,
        log_dir      = STAGE2_LOG_DIR,
    )
