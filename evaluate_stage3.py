# evaluate_stage3.py
"""
Stage 3 evaluation: TempDRL vs DAH baseline on test set.
Produces outputs/logs/stage3_revenue_results.csv
"""

import os, csv
import numpy as np
import torch

from src.config_stage3 import DEVICE, STATE_DIM, ACTION_DIM
from src.data_loader_stage2 import load_stage2_data
from src.data_bridge_stage3 import make_stage3_splits
from src.ttfe_stage2 import TTFE_S2
from src.sac_agent import SACAgent
from src.environment_stage3 import BESSEnvStage3
from src.dah_baseline import DAHBaseline
from main_stage3 import _build_segment

CHECKPOINT  = "outputs/checkpoints/stage3/best_model_s3.pt"
RESULTS_CSV = "outputs/logs/stage3_revenue_results.csv"
SEGMENT_LEN = 32


def run_evaluation():
    device = torch.device(DEVICE)

    raw  = load_stage2_data()
    data = make_stage3_splits(raw)

    train_prices  = data["train"]["prices"]
    train_syscond = data["train"]["syscond"]
    train_days    = data["train"]["days"]

    test_prices   = data["test"]["prices"]
    test_syscond  = data["test"]["syscond"]
    test_days     = data["test"]["days"]

    # DAH thresholds from training set
    rt_arr    = train_prices["rt_lmp"]
    disch_thr = float(np.nanmean(rt_arr) + 0.5 * np.nanstd(rt_arr))
    charg_thr = float(np.nanmean(rt_arr) - 0.5 * np.nanstd(rt_arr))
    print(f"DAH thresholds: discharge=${disch_thr:.2f}  charge=${charg_thr:.2f}")

    # Load TempDRL model
    ttfe  = TTFE_S2().to(device)
    agent = SACAgent(obs_dim=STATE_DIM, act_dim=ACTION_DIM, device=device)
    ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    ttfe.load_state_dict(ckpt["ttfe_state"])
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic1.load_state_dict(ckpt["critic1"])
    agent.critic2.load_state_dict(ckpt["critic2"])
    ttfe.eval()
    agent.actor.eval()

    env_rl = BESSEnvStage3(test_prices, test_syscond, ttfe, device)
    dah    = DAHBaseline(test_prices, test_syscond, disch_thr, charg_thr)

    results = []

    for ep_idx, ep_start in enumerate(test_days):
        # --- TempDRL ---
        obs          = env_rl.reset(ep_start)
        ep_rev_rl    = 0.0
        rev_spot_rl  = rev_as_rl = rev_degrad_rl = 0.0
        soc_clips_rl = 0

        for step in range(288):
            t         = ep_start + step
            segment   = _build_segment(raw["test_prices_sc"], t, SEGMENT_LEN)
            with torch.no_grad():
                seg_t     = torch.FloatTensor(segment).unsqueeze(0).to(device)
                ttfe_feat = ttfe(seg_t).cpu().numpy().flatten()
            obs[0:64] = ttfe_feat
            action    = agent.select_action(obs, deterministic=True)
            obs, reward, done, info = env_rl.step(action)

            ep_rev_rl    += info["total_rev"]
            rev_spot_rl  += info["rev_spot"]
            rev_as_rl    += info["rev_as"]
            rev_degrad_rl+= info["rev_degrad"]
            if info["soc_violation"]:
                soc_clips_rl += 1

        # --- DAH ---
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
              f"ratio={ep_rev_rl/max(1.0, dah_result['total_rev']):.2f}x")

    # Write CSV
    os.makedirs("outputs/logs", exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    tempdrl_vals = [r["tempdrl_total"] for r in results]
    dah_vals     = [r["dah_total"]     for r in results]
    print("\n=== Stage 3 Evaluation Summary ===")
    print(f"  Test days:        {len(results)}")
    print(f"  TempDRL mean/day: ${np.mean(tempdrl_vals):.2f}  (std ${np.std(tempdrl_vals):.2f})")
    print(f"  DAH mean/day:     ${np.mean(dah_vals):.2f}  (std ${np.std(dah_vals):.2f})")
    print(f"  TempDRL vs DAH:   {np.mean(tempdrl_vals)/max(1.0,np.mean(dah_vals))*100:.1f}%")
    print(f"  Results saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    run_evaluation()
