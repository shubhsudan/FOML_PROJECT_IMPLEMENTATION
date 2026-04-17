"""
main_evaluate.py — Phase 3: Day-to-day market performance evaluation.

Loads best_model.pt and runs deterministic greedy rollouts on all 43 eval
days (Nov–Dec 2022). Compares against 3 baselines:
  1. Idle        — BESS does nothing (zero revenue reference)
  2. Random      — uniform random actions
  3. Spot-Greedy — simple EMA heuristic: discharge when spot > EMA, charge when < EMA

Outputs (all saved to outputs/eval/):
  day_results.csv     — per-day revenue breakdown for each policy
  step_results.csv    — per-step detail for TempDRL (SoC, bids, revenue)
  summary.txt         — printed summary table

Plots saved to outputs/plots/:
  01_daily_revenue.png        — daily revenue bar chart (all 43 days)
  02_revenue_breakdown.png    — stacked: spot vs FCAS vs degradation
  03_cumulative_revenue.png   — cumulative revenue over 43 days
  04_policy_comparison.png    — TempDRL vs baselines
  05_soc_trajectory.png       — SoC on best / worst / median day
  06_bid_distribution.png     — histogram of bid sizes per market
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import (
    EMBED_DIM, NUM_MARKETS, TEMPORAL_SEG_LEN, TIMESTEPS_PER_DAY, BESS_CAPACITY_MWH
)
from data_loader import load_all, iter_daily_episodes
from environment import BESSEnvironment, BESSParams
from ttfe import TTFE, build_temporal_segment
from sac_agent import SACAgent


# ─── Output dirs ──────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR  = os.path.join(BASE_DIR, "outputs", "eval")
PLOT_DIR  = os.path.join(BASE_DIR, "outputs", "plots")
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

GPU_ID   = 26
CKPT     = os.path.join(BASE_DIR, "outputs", "checkpoints", "best_model.pt")


# ─── Batched TTFE pre-computation ─────────────────────────────────────────────

def precompute_features(ttfe, scaled_prices, device):
    T = scaled_prices.shape[0]
    segs = np.stack(
        [build_temporal_segment(scaled_prices, t, L=TEMPORAL_SEG_LEN) for t in range(T)],
        axis=0,
    )
    with torch.no_grad():
        feats = ttfe(torch.from_numpy(segs).to(device)).cpu().numpy()
    return feats  # (T, EMBED_DIM)


# ─── Baseline policies ────────────────────────────────────────────────────────

def idle_action():
    """Do nothing — all actions map to 0 bids after tanh clipping."""
    return np.full(6, -1.0, dtype=np.float32)   # tanh(-1)→-0.76 → maps to ~0 bid

def random_action():
    return np.random.uniform(-1.0, 1.0, size=(6,)).astype(np.float32)

def spot_greedy_action(spot_price, ema_spot):
    """
    Simple EMA-based heuristic:
      - discharge at full power when spot > EMA
      - charge at full power when spot < EMA
      - no FCAS bids
    """
    action = np.zeros(6, dtype=np.float32)
    if spot_price > ema_spot:      # discharge
        action[0] =  1.0           # v_dch signal > 0
        action[1] = -1.0           # v_ch signal < 0
        action[2] =  1.0           # full spot bid
    else:                          # charge
        action[0] = -1.0
        action[1] =  1.0
        action[2] =  1.0
    # no FCAS (action[3:6] = 0 already)
    return action


# ─── Single episode rollout with full per-step logging ────────────────────────

def rollout(
    env, raw_prices, scaled_prices,
    policy,            # callable(obs, t, spot, ema) → action
    ttfe, device,
    collect_steps=False,
):
    """
    Run one episode. Returns day-level dict and optional step-level list.
    policy signature: policy(obs, t, spot_price, ema_spot) → np.ndarray (6,)
    """
    T = raw_prices.shape[0]
    feats = precompute_features(ttfe, scaled_prices, device)

    obs = env.reset(price_episode=raw_prices, init_feature=feats[0])

    total_reward  = 0.0
    r_spot_total  = 0.0
    r_fast_total  = 0.0
    r_slow_total  = 0.0
    r_delay_total = 0.0
    deg_total     = 0.0
    n_violations  = 0
    soc_list      = [env.energy / BESS_CAPACITY_MWH]
    steps         = []

    for t in range(T):
        spot = float(raw_prices[t, 0])
        action = policy(obs, t, spot, env.ema_spot)
        next_feat = feats[min(t + 1, T - 1)]
        next_obs, reward, done, info = env.step(action, next_feat)

        total_reward  += reward
        r_spot_total  += info["r_spot"]
        r_fast_total  += info["r_fast"]
        r_slow_total  += info["r_slow"]
        r_delay_total += info["r_delay"]
        deg_total     += info["deg_cost"]
        n_violations  += int(info["violated"])
        soc_list.append(info["soc"])

        if collect_steps:
            p = BESSParams()
            from environment import BESSEnvironment as _E
            v_dch, v_ch, a_S, a_fast, a_slow, a_delay = _E.map_action(action, p)
            steps.append({
                "t": t,
                "soc": info["soc"],
                "spot_price": spot,
                "v_dch": info["v_dch"],
                "v_ch": info["v_ch"],
                "a_spot_mw": a_S,
                "a_fast_mw": a_fast,
                "a_slow_mw": a_slow,
                "a_delay_mw": a_delay,
                "r_spot": info["r_spot"],
                "r_fast": info["r_fast"],
                "r_slow": info["r_slow"],
                "r_delay": info["r_delay"],
                "deg_cost": info["deg_cost"],
                "step_reward": reward,
                "violated": int(info["violated"]),
            })

        obs = next_obs
        if done:
            break

    soc_arr = np.array(soc_list)
    day = {
        "total_revenue": total_reward,
        "r_spot":        r_spot_total,
        "r_fast":        r_fast_total,
        "r_slow":        r_slow_total,
        "r_delay":       r_delay_total,
        "r_fcas":        r_fast_total + r_slow_total + r_delay_total,
        "deg_cost":      deg_total,
        "n_violations":  n_violations,
        "soc_min":       float(soc_arr.min()),
        "soc_max":       float(soc_arr.max()),
        "soc_mean":      float(soc_arr.mean()),
        "soc_trajectory": soc_arr.tolist(),
    }
    return day, steps


# ─── Main evaluation ──────────────────────────────────────────────────────────

def main():
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    # Load data
    print("[Eval] Loading data ...")
    data = load_all()
    eval_raw     = iter_daily_episodes(data["eval_raw"],     TIMESTEPS_PER_DAY)
    eval_scaled  = iter_daily_episodes(data["eval_scaled"],  TIMESTEPS_PER_DAY)
    n_eval = len(eval_raw)
    print(f"[Eval] {n_eval} eval days (Nov–Dec 2022)")

    # Load TTFE + Agent
    ttfe = TTFE().to(device)
    ttfe.eval()

    agent = SACAgent(device=device)
    agent.load(CKPT)
    print(f"[Eval] Loaded checkpoint: {CKPT}")

    # Environment
    env = BESSEnvironment(price_episode=eval_raw[0], mode="joint")

    # Define policies
    def tempdrl_policy(obs, t, spot, ema):
        return agent.select_action(obs, deterministic=True)

    def idle_policy(obs, t, spot, ema):
        return idle_action()

    def rand_policy(obs, t, spot, ema):
        return random_action()

    def greedy_policy(obs, t, spot, ema):
        return spot_greedy_action(spot, ema)

    policies = {
        "TempDRL":      tempdrl_policy,
        "Idle":         idle_policy,
        "Random":       rand_policy,
        "SpotGreedy":   greedy_policy,
    }

    # ── Run all policies on all eval days ─────────────────────────────────────
    results = {name: [] for name in policies}
    tempdrl_steps_all = []

    print("\n[Eval] Running rollouts ...")
    for day_idx in range(n_eval):
        raw_ep = eval_raw[day_idx]
        sc_ep  = eval_scaled[day_idx]

        for name, pol in policies.items():
            collect = (name == "TempDRL")
            day_res, steps = rollout(
                env, raw_ep, sc_ep, pol, ttfe, device, collect_steps=collect
            )
            day_res["day"] = day_idx
            day_res["policy"] = name
            results[name].append(day_res)
            if collect:
                for s in steps:
                    s["day"] = day_idx
                tempdrl_steps_all.extend(steps)

        if (day_idx + 1) % 10 == 0:
            print(f"  ... {day_idx + 1}/{n_eval} days done")

    print("[Eval] Rollouts complete.\n")

    # ── Build DataFrames ───────────────────────────────────────────────────────
    day_rows = []
    for name, days in results.items():
        for d in days:
            row = {k: v for k, v in d.items() if k != "soc_trajectory"}
            day_rows.append(row)

    df_days  = pd.DataFrame(day_rows)
    df_steps = pd.DataFrame(tempdrl_steps_all)

    df_days.to_csv(os.path.join(EVAL_DIR, "day_results.csv"), index=False)
    df_steps.to_csv(os.path.join(EVAL_DIR, "step_results.csv"), index=False)
    print(f"[Eval] Saved CSVs to {EVAL_DIR}")

    # ── Summary statistics ─────────────────────────────────────────────────────
    summary_lines = []
    summary_lines.append("=" * 72)
    summary_lines.append("  TempDRL — Phase 3 Evaluation  |  43 eval days (Nov–Dec 2022)")
    summary_lines.append("=" * 72)
    summary_lines.append(f"{'Policy':<14} {'Mean Rev':>10} {'Std Rev':>9} {'Total Rev':>11} "
                         f"{'FCAS%':>7} {'Viol/day':>9} {'SoC ok%':>8}")
    summary_lines.append("-" * 72)

    for name in policies:
        df_p = df_days[df_days["policy"] == name]
        mean_r   = df_p["total_revenue"].mean()
        std_r    = df_p["total_revenue"].std()
        total_r  = df_p["total_revenue"].sum()
        mean_v   = df_p["n_violations"].mean()
        soc_ok   = 100.0 * (1.0 - df_p["n_violations"].sum() / (n_eval * TIMESTEPS_PER_DAY))
        fcas_sum = df_p["r_fcas"].sum()
        tot_sum  = df_p["total_revenue"].sum()
        fcas_pct = 100.0 * fcas_sum / tot_sum if tot_sum > 0 else 0.0
        summary_lines.append(
            f"{name:<14} {mean_r:>10.2f} {std_r:>9.2f} {total_r:>11.2f} "
            f"{fcas_pct:>6.1f}% {mean_v:>9.2f} {soc_ok:>7.2f}%"
        )

    summary_lines.append("=" * 72)

    # TempDRL market breakdown
    df_td = df_days[df_days["policy"] == "TempDRL"]
    summary_lines.append("\n  TempDRL — Revenue Breakdown (mean per day, AU$)")
    summary_lines.append(f"    Spot market   : {df_td['r_spot'].mean():>8.2f}")
    summary_lines.append(f"    Fast FCAS     : {df_td['r_fast'].mean():>8.2f}")
    summary_lines.append(f"    Slow FCAS     : {df_td['r_slow'].mean():>8.2f}")
    summary_lines.append(f"    Delayed FCAS  : {df_td['r_delay'].mean():>8.2f}")
    summary_lines.append(f"    Degradation   : {-df_td['deg_cost'].mean():>8.2f}  (cost)")
    summary_lines.append(f"    NET daily rev : {df_td['total_revenue'].mean():>8.2f}")
    summary_lines.append(f"    Best day      : {df_td['total_revenue'].max():>8.2f}  (day {df_td['total_revenue'].idxmax() % n_eval})")
    summary_lines.append(f"    Worst day     : {df_td['total_revenue'].min():>8.2f}  (day {df_td['total_revenue'].idxmin() % n_eval})")
    summary_lines.append("=" * 72)

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open(os.path.join(EVAL_DIR, "summary.txt"), "w") as f:
        f.write(summary_text)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[Eval] Generating plots ...")
    COLORS = {
        "TempDRL":    "#2196F3",
        "Idle":       "#9E9E9E",
        "Random":     "#FF9800",
        "SpotGreedy": "#4CAF50",
    }
    td_days  = results["TempDRL"]
    td_revs  = [d["total_revenue"] for d in td_days]
    td_spot  = [d["r_spot"]  for d in td_days]
    td_fast  = [d["r_fast"]  for d in td_days]
    td_slow  = [d["r_slow"]  for d in td_days]
    td_delay = [d["r_delay"] for d in td_days]
    td_deg   = [-d["deg_cost"] for d in td_days]   # shown as negative

    day_ids = list(range(n_eval))

    # ── Plot 1: Daily revenue bar chart ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#2196F3" if r >= 0 else "#F44336" for r in td_revs]
    ax.bar(day_ids, td_revs, color=colors, edgecolor="none", width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(np.mean(td_revs), color="#FF9800", linewidth=1.5,
               linestyle="--", label=f"Mean = {np.mean(td_revs):.1f} AU$")
    ax.set_xlabel("Eval Day (Nov–Dec 2022)")
    ax.set_ylabel("Daily Revenue (AU$)")
    ax.set_title("TempDRL — Daily Revenue on 43 Unseen Eval Days")
    ax.legend()
    ax.set_xlim(-0.5, n_eval - 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "01_daily_revenue.png"), dpi=150)
    plt.close()

    # ── Plot 2: Stacked revenue breakdown ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    bottom = np.zeros(n_eval)
    for vals, label, color in [
        (td_spot,  "Spot",         "#1565C0"),
        (td_fast,  "Fast FCAS",    "#42A5F5"),
        (td_slow,  "Slow FCAS",    "#80CBC4"),
        (td_delay, "Delayed FCAS", "#A5D6A7"),
        (td_deg,   "Degradation",  "#EF9A9A"),
    ]:
        arr = np.array(vals)
        ax.bar(day_ids, arr, bottom=bottom, label=label, color=color,
               edgecolor="none", width=0.8)
        bottom += arr
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Eval Day")
    ax.set_ylabel("Revenue (AU$)")
    ax.set_title("TempDRL — Revenue Breakdown by Market")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(-0.5, n_eval - 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "02_revenue_breakdown.png"), dpi=150)
    plt.close()

    # ── Plot 3: Cumulative revenue ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, pol_days in results.items():
        revs = np.array([d["total_revenue"] for d in pol_days])
        ax.plot(np.cumsum(revs), label=name, color=COLORS[name],
                linewidth=2 if name == "TempDRL" else 1.2,
                linestyle="-" if name == "TempDRL" else "--")
    ax.set_xlabel("Eval Day")
    ax.set_ylabel("Cumulative Revenue (AU$)")
    ax.set_title("Cumulative Revenue — TempDRL vs Baselines")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "03_cumulative_revenue.png"), dpi=150)
    plt.close()

    # ── Plot 4: Policy comparison ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    policy_names = list(policies.keys())

    # Mean revenue
    means = [df_days[df_days["policy"] == n]["total_revenue"].mean() for n in policy_names]
    stds  = [df_days[df_days["policy"] == n]["total_revenue"].std()  for n in policy_names]
    bar_colors = [COLORS[n] for n in policy_names]
    axes[0].bar(policy_names, means, yerr=stds, color=bar_colors,
                capsize=5, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Mean Daily Revenue (AU$)")
    axes[0].set_title("Mean Daily Revenue ± Std")
    axes[0].axhline(0, color="black", linewidth=0.8)
    for i, (m, s) in enumerate(zip(means, stds)):
        axes[0].text(i, m + s + 20, f"{m:.0f}", ha="center", fontsize=9)

    # Mean violations
    viols = [df_days[df_days["policy"] == n]["n_violations"].mean() for n in policy_names]
    axes[1].bar(policy_names, viols, color=bar_colors,
                edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Mean Violations / Day")
    axes[1].set_title("Mean SoC Constraint Violations / Day")
    for i, v in enumerate(viols):
        axes[1].text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)

    plt.suptitle("TempDRL vs Baselines", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "04_policy_comparison.png"), dpi=150)
    plt.close()

    # ── Plot 5: SoC trajectories (best / median / worst day) ──────────────────
    sorted_days = np.argsort(td_revs)
    worst_idx   = sorted_days[0]
    median_idx  = sorted_days[len(sorted_days) // 2]
    best_idx    = sorted_days[-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    ts = np.arange(TIMESTEPS_PER_DAY + 1) * 5 / 60  # hours

    for ax, idx, label in [
        (axes[0], best_idx,   "Best Day"),
        (axes[1], median_idx, "Median Day"),
        (axes[2], worst_idx,  "Worst Day"),
    ]:
        soc = td_days[idx]["soc_trajectory"]
        rev = td_days[idx]["total_revenue"]
        ax.plot(ts[:len(soc)], soc, color="#2196F3", linewidth=1.5)
        ax.axhline(0.05, color="red",   linewidth=1, linestyle="--", label="SoC min (5%)")
        ax.axhline(0.95, color="green", linewidth=1, linestyle="--", label="SoC max (95%)")
        ax.fill_between(ts[:len(soc)], 0.05, 0.95, alpha=0.05, color="green")
        ax.set_title(f"{label}\nRevenue = {rev:.1f} AU$")
        ax.set_xlabel("Hour of Day")
        if ax == axes[0]:
            ax.set_ylabel("State of Charge")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 24)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("TempDRL — SoC Trajectory: Best / Median / Worst Day", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "05_soc_trajectory.png"), dpi=150)
    plt.close()

    # ── Plot 6: Bid size distributions ────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    cols   = ["a_spot_mw", "a_fast_mw", "a_slow_mw", "a_delay_mw"]
    labels = ["Spot Bid (MW)", "Fast FCAS (MW)", "Slow FCAS (MW)", "Delayed FCAS (MW)"]
    colors_bid = ["#1565C0", "#42A5F5", "#80CBC4", "#A5D6A7"]

    for i, (col, lbl, c) in enumerate(zip(cols, labels, colors_bid)):
        ax = axes[i // 3][i % 3] if i < 3 else axes[1][i - 3]
        axes_flat = [axes[0][0], axes[0][1], axes[0][2], axes[1][0]]
        ax = axes_flat[i]
        vals = df_steps[col].values
        ax.hist(vals, bins=40, color=c, edgecolor="none", alpha=0.85)
        ax.set_xlabel(lbl)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {lbl}")
        ax.axvline(vals.mean(), color="black", linewidth=1.2,
                   linestyle="--", label=f"Mean={vals.mean():.3f}")
        ax.legend(fontsize=8)

    # discharge/charge frequency
    axes[1][1].bar(
        ["Discharge", "Charge", "Idle"],
        [df_steps["v_dch"].sum(), df_steps["v_ch"].sum(),
         len(df_steps) - df_steps["v_dch"].sum() - df_steps["v_ch"].sum()],
        color=["#EF5350", "#42A5F5", "#9E9E9E"]
    )
    axes[1][1].set_title("Dispatch Mode Frequency")
    axes[1][1].set_ylabel("Timesteps")

    # violations per day
    viol_per_day = [d["n_violations"] for d in td_days]
    axes[1][2].hist(viol_per_day, bins=20, color="#FF7043", edgecolor="none")
    axes[1][2].set_title("SoC Violations per Day")
    axes[1][2].set_xlabel("Violations")
    axes[1][2].set_ylabel("Days")

    plt.suptitle("TempDRL — Bid & Dispatch Behaviour", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "06_bid_distribution.png"), dpi=150)
    plt.close()

    print(f"[Eval] Plots saved to {PLOT_DIR}")
    print("\n[Eval] Done.")


if __name__ == "__main__":
    main()
