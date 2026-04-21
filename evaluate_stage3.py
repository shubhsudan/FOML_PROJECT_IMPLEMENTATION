# evaluate_stage3.py
"""
Stage 3 evaluation: TempDRL vs DAH baseline.
Runs both policies on the test set and produces a results CSV.
No PIO. No parallel workers needed for DAH (it's fast).
"""

import os, csv
import numpy as np
import torch

from src.config_stage3 import DEVICE, STATE_DIM, ACTION_DIM
from src.data_loader_stage2 import load_stage2_data
from src.ttfe_stage2 import TTFE_S2
from src.sac_agent import SACAgent
from src.environment_stage3 import BESSEnvStage3
from src.dah_baseline import DAHBaseline

CHECKPOINT  = "outputs/checkpoints/stage3/best_model_s3.pt"
RESULTS_CSV = "outputs/logs/stage3_revenue_results.csv"
SEGMENT_LEN = 32   # TTFE window

# Compute DAH thresholds from training set
def compute_dah_thresholds(price_data, train_indices):
    rt_lmp_train = np.array([price_data["rt_lmp"][i] for i in train_indices])
    mu  = float(np.nanmean(rt_lmp_train))
    sig = float(np.nanstd(rt_lmp_train))
    return mu + 0.5 * sig, mu - 0.5 * sig


def run_evaluation():
    device = torch.device(DEVICE)

    # Load data
    data = load_stage2_data()
    price_data   = data["prices"]
    syscond_data = data["syscond"]
    test_days    = data["test_days"]      # list of episode start indices
    train_days   = data["train_days"]

    # Compute DAH thresholds on training set
    all_train_steps = []
    for d in train_days:
        all_train_steps.extend(range(d, d + 288))
    disch_thr, charg_thr = compute_dah_thresholds(price_data, all_train_steps)
    print(f"DAH thresholds: discharge={disch_thr:.2f}, charge={charg_thr:.2f} $/MWh")

    # Load TempDRL model
    ttfe = TTFE_S2().to(device)
    agent = SACAgent(obs_dim=STATE_DIM, act_dim=ACTION_DIM, device=device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    ttfe.load_state_dict(ckpt["ttfe_state"])
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic1.load_state_dict(ckpt["critic1"])
    agent.critic2.load_state_dict(ckpt["critic2"])
    ttfe.eval()
    agent.actor.eval()

    # Environments
    env_rl  = BESSEnvStage3(price_data, syscond_data, ttfe, device)
    dah     = DAHBaseline(price_data, syscond_data, disch_thr, charg_thr)

    results = []

    for ep_idx, ep_start in enumerate(test_days):
        # --- Run TempDRL ---
        obs = env_rl.reset(ep_start)
        ep_rev_rl = 0.0
        rev_spot_rl = rev_as_rl = rev_degrad_rl = 0.0
        soc_clips_rl = 0

        for step in range(288):
            t = ep_start + step
            # Build TTFE segment
            seg_start = max(0, t - SEGMENT_LEN + 1)
            segment = np.zeros((SEGMENT_LEN, 12), dtype=np.float32)
            actual_len = t - seg_start + 1
            for feat_i, feat in enumerate([
                "rt_lmp","rt_mcpc_regup","rt_mcpc_regdn","rt_mcpc_rrs",
                "rt_mcpc_ecrs","rt_mcpc_nsrs",
                "dam_spp","dam_as_regup","dam_as_regdn","dam_as_rrs",
                "dam_as_ecrs","dam_as_nsrs"
            ]):
                segment[SEGMENT_LEN - actual_len:, feat_i] = \
                    price_data[feat][seg_start:t+1]

            with torch.no_grad():
                seg_t = torch.FloatTensor(segment).unsqueeze(0).to(device)
                ttfe_feat = ttfe(seg_t).cpu().numpy().flatten()

            obs[0:64] = ttfe_feat
            action = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env_rl.step(action)

            ep_rev_rl    += info["total_rev"]
            rev_spot_rl  += info["rev_spot"]
            rev_as_rl    += info["rev_as"]
            rev_degrad_rl+= info["rev_degrad"]
            if info["soc_violation"]:
                soc_clips_rl += 1

        # --- Run DAH ---
        dah_result = dah.run_episode(ep_start)

        results.append({
            "episode":        ep_idx,
            "ep_start":       ep_start,
            "tempdrl_total":  round(ep_rev_rl, 2),
            "tempdrl_spot":   round(rev_spot_rl, 2),
            "tempdrl_as":     round(rev_as_rl, 2),
            "tempdrl_degrad": round(rev_degrad_rl, 2),
            "tempdrl_clips":  soc_clips_rl,
            "dah_total":      round(dah_result["total_rev"], 2),
            "dah_spot":       round(dah_result["rev_spot"], 2),
            "dah_as":         round(dah_result["rev_as"], 2),
            "dah_degrad":     round(dah_result["rev_degrad"], 2),
            "dah_cycles":     round(dah_result["cycles"], 2),
        })
        print(f"  Day {ep_idx+1:3d}: TempDRL=${ep_rev_rl:7.2f}  "
              f"DAH=${dah_result['total_rev']:7.2f}  "
              f"ratio={ep_rev_rl/max(1, dah_result['total_rev']):.2f}x")

    # Write CSV
    os.makedirs("outputs/logs", exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Summary
    tempdrl_vals = [r["tempdrl_total"] for r in results]
    dah_vals     = [r["dah_total"] for r in results]
    print("\n=== Stage 3 Evaluation Summary ===")
    print(f"  Test days:           {len(results)}")
    print(f"  TempDRL mean/day:    ${np.mean(tempdrl_vals):.2f}  "
          f"(std ${np.std(tempdrl_vals):.2f})")
    print(f"  DAH mean/day:        ${np.mean(dah_vals):.2f}  "
          f"(std ${np.std(dah_vals):.2f})")
    print(f"  TempDRL vs DAH:      "
          f"{np.mean(tempdrl_vals)/max(1,np.mean(dah_vals))*100:.1f}%")
    print(f"  Results saved to:    {RESULTS_CSV}")


if __name__ == "__main__":
    run_evaluation()
