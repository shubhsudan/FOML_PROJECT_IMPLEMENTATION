# main_stage3.py
"""
Stage 3 training entry point.
Loads Stage 2 best checkpoint, applies Stage 3 corrections, trains.

Run with:
  tmux new-session -d -s stage3 \
    "PYTHONUNBUFFERED=1 \
     /home/stu9/s7/ss2401/miniconda3/envs/tempdrl/bin/python \
     main_stage3.py 2>&1 | tee outputs/logs/train_stage3.log"
"""

import os, math, torch
import numpy as np

from src.config_stage3 import (
    DEVICE, STATE_DIM, ACTION_DIM, BATCH_SIZE, REPLAY_SIZE,
    GRAD_STEPS, WARMUP_STEPS, N_EPISODES, EVAL_EVERY,
    LR_ACTOR, LR_CRITIC, LR_TTFE, TARGET_ENTROPY, REWARD_CLIP
)
from src.data_loader_stage2 import load_stage2_data
from src.data_bridge_stage3 import make_stage3_splits
from src.ttfe_stage2 import TTFE_S2
from src.sac_agent import SACAgent
from src.environment_stage3 import BESSEnvStage3
from src.dah_baseline import DAHBaseline

STAGE2_CKPT = "outputs/checkpoints/stage2/best_model_s2.pt"
STAGE3_DIR  = "outputs/checkpoints/stage3"
SEGMENT_LEN = 32

os.makedirs(STAGE3_DIR, exist_ok=True)
os.makedirs("outputs/logs", exist_ok=True)


def main():
    device = torch.device(DEVICE)

    # --- Load and bridge data ---
    raw  = load_stage2_data()
    data = make_stage3_splits(raw)

    train_prices  = data["train"]["prices"]
    train_syscond = data["train"]["syscond"]
    train_days    = data["train"]["days"]

    val_prices    = data["val"]["prices"]
    val_syscond   = data["val"]["syscond"]
    val_days      = data["val"]["days"]

    # --- DAH thresholds from training set ---
    rt_arr    = train_prices["rt_lmp"]
    disch_thr = float(np.nanmean(rt_arr) + 0.5 * np.nanstd(rt_arr))
    charg_thr = float(np.nanmean(rt_arr) - 0.5 * np.nanstd(rt_arr))
    print(f"DAH thresholds: discharge=${disch_thr:.2f}  charge=${charg_thr:.2f}")

    dah_train = DAHBaseline(train_prices, train_syscond, disch_thr, charg_thr)
    dah_val   = DAHBaseline(val_prices,   val_syscond,   disch_thr, charg_thr)

    # --- Model setup ---
    ttfe  = TTFE_S2().to(device)
    agent = SACAgent(obs_dim=STATE_DIM, act_dim=ACTION_DIM,
                     lr_policy=LR_ACTOR, lr_q=LR_CRITIC, device=device)

    # Load Stage 2 weights
    if os.path.exists(STAGE2_CKPT):
        ckpt = torch.load(STAGE2_CKPT, map_location=device, weights_only=False)
        # Stage 2 checkpoint uses 'ttfe_s2_state', 'actor_state', 'critic1_state', ...
        ttfe_key   = "ttfe_s2_state" if "ttfe_s2_state" in ckpt else "ttfe_state"
        actor_key  = "actor_state"   if "actor_state"   in ckpt else "actor"
        critic1_key= "critic1_state" if "critic1_state" in ckpt else "critic1"
        critic2_key= "critic2_state" if "critic2_state" in ckpt else "critic2"
        ttfe.load_state_dict(ckpt[ttfe_key])
        try:
            agent.actor.load_state_dict(ckpt[actor_key],   strict=False)
            agent.critic1.load_state_dict(ckpt[critic1_key], strict=False)
            agent.critic2.load_state_dict(ckpt[critic2_key], strict=False)
            print("Loaded Stage 2 actor/critic weights (partial, strict=False)")
        except Exception as e:
            print(f"Actor/critic weight load failed ({e}), starting fresh")
        log_alpha_init = math.log(0.05)
        agent.log_alpha.data.fill_(log_alpha_init)
        print(f"Reset log_alpha to {log_alpha_init:.3f}")
    else:
        print(f"No Stage 2 checkpoint at {STAGE2_CKPT}, starting fresh")

    # --- Training phases ---
    # Phase A (ep 1–499):    TTFE frozen
    # Phase B (ep 500–1499): top MHA layer only, LR=1e-5
    # Phase C (ep 1500–5000): all TTFE, LR=1e-5

    replay_buffer = []
    best_val      = -1e9
    ttfe_opt      = None

    env_train = BESSEnvStage3(train_prices, train_syscond, ttfe, device)
    env_val   = BESSEnvStage3(val_prices,   val_syscond,   ttfe, device)

    # Freeze TTFE for Phase A
    for p in ttfe.parameters():
        p.requires_grad_(False)

    global_step         = 0
    divergence_cooldown = 0

    for ep in range(1, N_EPISODES + 1):

        # LR decay + entropy reduction at ep 1500 to break plateau
        if ep == 1500:
            print("Ep 1500: halving LR and reducing entropy for exploitation phase")
            for pg in agent.actor_optimizer.param_groups:
                pg["lr"] *= 0.5
            for pg in agent.critic1_optimizer.param_groups:
                pg["lr"] *= 0.5
            for pg in agent.critic2_optimizer.param_groups:
                pg["lr"] *= 0.5
            # Reduce entropy temperature: clamp log_alpha downward
            with torch.no_grad():
                agent.log_alpha.clamp_(max=math.log(0.02))

        # TTFE stays frozen for all 5000 eps in Stage 3.
        # Stage 2 already converged TTFE; fine-tuning it here caused
        # severe oscillations because the separate TTFE actor-loss pass
        # fought the SAC critic update. The actor/critic alone are
        # sufficient to learn the Stage 3 reward structure (DART, $15
        # degradation, single ESR dispatch).

        # Sample random training day
        ep_start = train_days[np.random.randint(len(train_days))]
        obs      = env_train.reset(ep_start)

        ep_reward = 0.0
        for step in range(288):
            t = ep_start + step

            # Build TTFE segment (local index into train_prices_sc)
            segment = _build_segment(raw["train_prices_sc"], t, SEGMENT_LEN)

            with torch.no_grad():
                seg_t     = torch.FloatTensor(segment).unsqueeze(0).to(device)
                ttfe_feat = ttfe(seg_t).cpu().numpy().flatten()
            obs[0:64] = ttfe_feat

            # Action
            if global_step < WARMUP_STEPS:
                action = np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)
            else:
                action = agent.select_action(obs)

            next_obs, reward, done, info = env_train.step(action)
            reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))

            replay_buffer.append((segment.copy(), obs.copy(), action,
                                   reward, next_obs.copy(), float(done)))
            if len(replay_buffer) > REPLAY_SIZE:
                replay_buffer.pop(0)

            obs        = next_obs
            ep_reward += reward
            global_step += 1

            if done:
                break

        # Gradient updates
        if len(replay_buffer) >= BATCH_SIZE and global_step >= WARMUP_STEPS:
            for _ in range(GRAD_STEPS):
                batch_idx = np.random.randint(0, len(replay_buffer), BATCH_SIZE)
                batch     = [replay_buffer[i] for i in batch_idx]

                segs   = torch.FloatTensor(np.stack([b[0] for b in batch])).to(device)
                obs_b  = torch.FloatTensor(np.stack([b[1] for b in batch])).to(device)
                act_b  = torch.FloatTensor(np.stack([b[2] for b in batch])).to(device)
                rew_b  = torch.FloatTensor(np.array([b[3] for b in batch])).to(device)
                nobs_b = torch.FloatTensor(np.stack([b[4] for b in batch])).to(device)
                done_b = torch.FloatTensor(np.array([b[5] for b in batch])).to(device)

                # Detach for SAC update — avoids double-backward through the
                # shared TTFE graph (sac_agent calls .backward() twice: critic
                # then actor, both through obs_b which shares the TTFE node)
                with torch.no_grad():
                    ttfe_feats_d = ttfe(segs)
                obs_b[:,  0:64] = ttfe_feats_d
                nobs_b[:, 0:64] = ttfe_feats_d

                batch_dict = {
                    "obs":      obs_b,
                    "actions":  act_b,
                    "rewards":  rew_b.unsqueeze(1),
                    "next_obs": nobs_b,
                    "dones":    done_b.unsqueeze(1),
                }
                agent.update(batch_dict)


        # Eval
        if ep % EVAL_EVERY == 0:
            val_revs_rl  = []
            val_revs_dah = []

            ttfe.eval()
            with torch.no_grad():
                for val_start in val_days[:10]:
                    obs_v  = env_val.reset(val_start)
                    ep_val = 0.0
                    for step in range(288):
                        t = val_start + step
                        segment   = _build_segment(raw["val_prices_sc"], t, SEGMENT_LEN)
                        seg_t     = torch.FloatTensor(segment).unsqueeze(0).to(device)
                        ttfe_feat = ttfe(seg_t).cpu().numpy().flatten()
                        obs_v[0:64] = ttfe_feat
                        a           = agent.select_action(obs_v, deterministic=True)
                        obs_v, r, done, info = env_val.step(a)
                        ep_val += info["total_rev"]
                        if done:
                            break
                    val_revs_rl.append(ep_val)
                    val_revs_dah.append(dah_val.run_episode(val_start)["total_rev"])

            ttfe.train()
            mean_rl  = np.mean(val_revs_rl)
            mean_dah = np.mean(val_revs_dah)
            ratio    = mean_rl / max(1.0, mean_dah) * 100

            print(f"[Ep {ep:5d}] val_rl=${mean_rl:.2f}  "
                  f"val_dah=${mean_dah:.2f}  ratio={ratio:.1f}%  "
                  f"ep_reward={ep_reward:.1f}")

            if mean_rl > best_val:
                best_val = mean_rl
                torch.save({
                    "ttfe_state": ttfe.state_dict(),
                    "actor":      agent.actor.state_dict(),
                    "critic1":    agent.critic1.state_dict(),
                    "critic2":    agent.critic2.state_dict(),
                    "log_alpha":  agent.log_alpha.item(),
                    "episode":    ep,
                    "val_rl":     mean_rl,
                    "val_dah":    mean_dah,
                }, os.path.join(STAGE3_DIR, "best_model_s3.pt"))
                print(f"  >> New best: ${best_val:.2f}/day saved")

            # Divergence guard
            if divergence_cooldown > 0:
                divergence_cooldown -= 1
            if mean_rl < -50 and divergence_cooldown == 0:
                best_path = os.path.join(STAGE3_DIR, "best_model_s3.pt")
                if os.path.exists(best_path):
                    print("  !! Divergence guard: reloading best checkpoint")
                    ckpt = torch.load(best_path, map_location=device, weights_only=False)
                    ttfe.load_state_dict(ckpt["ttfe_state"])
                    agent.actor.load_state_dict(ckpt["actor"])
                    agent.critic1.load_state_dict(ckpt["critic1"])
                    agent.critic2.load_state_dict(ckpt["critic2"])
                    for pg in agent.actor_optimizer.param_groups:
                        pg["lr"] *= 0.5
                    for pg in agent.critic1_optimizer.param_groups:
                        pg["lr"] *= 0.5
                    for pg in agent.critic2_optimizer.param_groups:
                        pg["lr"] *= 0.5
                    divergence_cooldown = 50

    print(f"\nTraining complete. Best val: ${best_val:.2f}/day")
    print("Run evaluate_stage3.py for test set results.")


def _build_segment(price_sc: np.ndarray, t: int, seg_len: int) -> np.ndarray:
    """Build (seg_len, 12) TTFE input from scaled price array at local index t."""
    n_feat  = price_sc.shape[1]
    segment = np.zeros((seg_len, n_feat), dtype=np.float32)
    start   = t - seg_len + 1
    if start < 0:
        avail = price_sc[max(0, start):t + 1]
        segment[seg_len - len(avail):] = avail
    else:
        segment = price_sc[start:t + 1].copy()
    return segment


if __name__ == "__main__":
    main()
