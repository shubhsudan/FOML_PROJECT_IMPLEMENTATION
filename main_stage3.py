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

    # --- Load data ---
    data = load_stage2_data()
    price_data   = data["prices"]
    syscond_data = data["syscond"]
    train_days   = data["train_days"]
    val_days     = data["val_days"]

    # --- DAH thresholds (for eval comparison during training) ---
    all_train = []
    for d in train_days:
        all_train.extend(range(d, d + 288))
    rt_arr = np.array([price_data["rt_lmp"][i] for i in all_train])
    disch_thr = float(np.nanmean(rt_arr) + 0.5 * np.nanstd(rt_arr))
    charg_thr = float(np.nanmean(rt_arr) - 0.5 * np.nanstd(rt_arr))
    dah = DAHBaseline(price_data, syscond_data, disch_thr, charg_thr)

    # --- Model setup ---
    ttfe  = TTFE_S2().to(device)
    agent = SACAgent(obs_dim=STATE_DIM, act_dim=ACTION_DIM,
                     lr_policy=LR_ACTOR, lr_q=LR_CRITIC, device=device)

    # Load Stage 2 weights
    if os.path.exists(STAGE2_CKPT):
        ckpt = torch.load(STAGE2_CKPT, map_location=device)
        ttfe.load_state_dict(ckpt["ttfe_state"])
        # Actor/critic weights incompatible: STATE_DIM changed 78→88
        # Load what we can, reinitialise the rest
        try:
            agent.actor.load_state_dict(ckpt["actor"], strict=False)
            agent.critic1.load_state_dict(ckpt["critic1"], strict=False)
            agent.critic2.load_state_dict(ckpt["critic2"], strict=False)
            print("Loaded Stage 2 actor/critic weights (partial, strict=False)")
        except Exception as e:
            print(f"Actor/critic weight load failed ({e}), starting fresh")
        # Reset log_alpha to config value
        log_alpha_init = math.log(0.05)
        agent.log_alpha.data.fill_(log_alpha_init)
        print(f"Reset log_alpha to {log_alpha_init:.3f}")
    else:
        print(f"No Stage 2 checkpoint at {STAGE2_CKPT}, starting fresh")

    # --- Training phases (same progressive TTFE unfreezing as Stage 2) ---
    # Phase A (ep 1–499):   TTFE frozen
    # Phase B (ep 500–1499): top MHA layer only, LR=1e-5
    # Phase C (ep 1500–5000): all TTFE, LR=1e-5

    replay_buffer = []   # simple list; replace with replay_buffer.py if needed
    best_val = -1e9
    ttfe_opt = None

    env = BESSEnvStage3(price_data, syscond_data, ttfe, device)

    # Freeze TTFE for Phase A
    for p in ttfe.parameters():
        p.requires_grad_(False)

    global_step = 0
    divergence_cooldown = 0

    for ep in range(1, N_EPISODES + 1):

        # Phase transitions
        if ep == 500:
            print("Phase B: unfreezing top MHA layer")
            for p in ttfe.parameters():
                p.requires_grad_(False)
            for p in ttfe.mha_layers[-1].parameters():
                p.requires_grad_(True)
            ttfe_opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, ttfe.parameters()),
                lr=LR_TTFE
            )
        elif ep == 1500:
            print("Phase C: unfreezing all TTFE parameters")
            for p in ttfe.parameters():
                p.requires_grad_(True)
            ttfe_opt = torch.optim.Adam(ttfe.parameters(), lr=LR_TTFE)

        # Sample random training day
        ep_start = train_days[np.random.randint(len(train_days))]
        obs = env.reset(ep_start)

        ep_reward = 0.0
        for step in range(288):
            t = ep_start + step

            # Build TTFE features
            seg_start = max(0, t - SEGMENT_LEN + 1)
            segment = np.zeros((SEGMENT_LEN, 12), dtype=np.float32)
            actual_len = t - seg_start + 1
            for fi, feat in enumerate([
                "rt_lmp","rt_mcpc_regup","rt_mcpc_regdn","rt_mcpc_rrs",
                "rt_mcpc_ecrs","rt_mcpc_nsrs",
                "dam_spp","dam_as_regup","dam_as_regdn","dam_as_rrs",
                "dam_as_ecrs","dam_as_nsrs"
            ]):
                segment[SEGMENT_LEN - actual_len:, fi] = \
                    price_data[feat][seg_start:t+1]

            with torch.no_grad():
                seg_t = torch.FloatTensor(segment).unsqueeze(0).to(device)
                ttfe_feat = ttfe(seg_t).cpu().numpy().flatten()
            obs[0:64] = ttfe_feat

            # Action
            if global_step < WARMUP_STEPS:
                action = np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)
            else:
                action = agent.select_action(obs)

            next_obs, reward, done, info = env.step(action)
            reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))

            # Store (raw segment, obs, action, reward, next_obs, done)
            replay_buffer.append((segment.copy(), obs.copy(), action,
                                   reward, next_obs.copy(), float(done)))
            if len(replay_buffer) > REPLAY_SIZE:
                replay_buffer.pop(0)

            obs = next_obs
            ep_reward += reward
            global_step += 1

            if done:
                break

        # Gradient updates
        if len(replay_buffer) >= BATCH_SIZE and global_step >= WARMUP_STEPS:
            for _ in range(GRAD_STEPS):
                batch_idx = np.random.randint(0, len(replay_buffer), BATCH_SIZE)
                batch = [replay_buffer[i] for i in batch_idx]

                segs     = torch.FloatTensor(np.stack([b[0] for b in batch])).to(device)
                obs_b    = torch.FloatTensor(np.stack([b[1] for b in batch])).to(device)
                act_b    = torch.FloatTensor(np.stack([b[2] for b in batch])).to(device)
                rew_b    = torch.FloatTensor(np.array([b[3] for b in batch])).to(device)
                nobs_b   = torch.FloatTensor(np.stack([b[4] for b in batch])).to(device)
                done_b   = torch.FloatTensor(np.array([b[5] for b in batch])).to(device)

                # Inject TTFE features into obs
                ttfe_feats = ttfe(segs)
                obs_b[:, 0:64]  = ttfe_feats
                nobs_b[:, 0:64] = ttfe_feats   # approx: same segment for next obs

                batch_dict = {
                    "obs":      obs_b,
                    "actions":  act_b,
                    "rewards":  rew_b.unsqueeze(1),
                    "next_obs": nobs_b,
                    "dones":    done_b.unsqueeze(1),
                }
                agent.update(batch_dict)

                if ttfe_opt is not None:
                    ttfe_opt.zero_grad()
                    # TTFE gradient flows through actor loss
                    ttfe_opt.step()

        # Eval
        if ep % EVAL_EVERY == 0:
            val_revs_rl  = []
            val_revs_dah = []

            ttfe.eval()
            with torch.no_grad():
                for val_start in val_days[:10]:
                    obs_v = env.reset(val_start)
                    ep_val = 0.0
                    for step in range(288):
                        t = val_start + step
                        seg_start = max(0, t - SEGMENT_LEN + 1)
                        segment = np.zeros((SEGMENT_LEN, 12), dtype=np.float32)
                        actual_len = t - seg_start + 1
                        for fi, feat in enumerate([
                            "rt_lmp","rt_mcpc_regup","rt_mcpc_regdn","rt_mcpc_rrs",
                            "rt_mcpc_ecrs","rt_mcpc_nsrs",
                            "dam_spp","dam_as_regup","dam_as_regdn","dam_as_rrs",
                            "dam_as_ecrs","dam_as_nsrs"
                        ]):
                            segment[SEGMENT_LEN - actual_len:, fi] = \
                                price_data[feat][seg_start:t+1]
                        seg_t = torch.FloatTensor(segment).unsqueeze(0).to(device)
                        ttfe_feat = ttfe(seg_t).cpu().numpy().flatten()
                        obs_v[0:64] = ttfe_feat
                        a = agent.select_action(obs_v, deterministic=True)
                        obs_v, r, done, info = env.step(a)
                        ep_val += info["total_rev"]
                        if done:
                            break
                    val_revs_rl.append(ep_val)

                    d = dah.run_episode(val_start)
                    val_revs_dah.append(d["total_rev"])

            ttfe.train()
            mean_rl  = np.mean(val_revs_rl)
            mean_dah = np.mean(val_revs_dah)
            ratio    = mean_rl / max(1, mean_dah) * 100

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
            if mean_rl < 0 and divergence_cooldown == 0:
                print("  !! Divergence guard: reloading best checkpoint")
                ckpt = torch.load(
                    os.path.join(STAGE3_DIR, "best_model_s3.pt"),
                    map_location=device
                )
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


if __name__ == "__main__":
    main()
