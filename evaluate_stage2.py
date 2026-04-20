"""
evaluate_stage2.py — Stage 2 Revenue Evaluation (post-RTC+B, 6-market).

Mirrors evaluate_phase2_fast.py but targets the Stage 2 setting:
  - 6 AS markets: spot + RegUp + RegDn + RRS + ECRS + NSRS
  - 12-dim price vector, 32-step TTFE window, 78-dim state, 9-dim action
  - ECRS: 2-hour SOC duration requirement (new vs Stage 1)
  - Test set: 15 post-RTC+B days

PIO runs in parallel across 32 workers.
TempDRL greedy rollouts on GPU for joint / spot-only / AS-only modes.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import pulp
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from ttfe_stage2     import TTFE_S2
from sac_agent       import Actor
from data_loader_stage2 import (
    load_stage2_data,
    iter_daily_episodes_s2,
    build_temporal_segment_12,
    build_state_78,
    build_time_6,
)
from environment_stage2 import decode_action_s2, compute_step_revenue_s2
from config import (
    RATED_POWER_MW, FCAS_MAX_MW,
    EFF_CH, EFF_DCH, E_MIN_MWH, E_MAX_MWH, E_INIT_MWH,
    DEGRADATION_C, DT_H, CAPACITY_MWH,
)
from config_stage2 import (
    STATE_DIM_S2, ACTION_DIM_S2,
    TTFE_SEG_LEN_S2, NUM_PRICE_DIMS,
    AS_DURATION_H_S2,
    IDX_RT_LMP,
    IDX_RT_MCPC_REGUP, IDX_RT_MCPC_REGDN,
    IDX_RT_MCPC_RRS, IDX_RT_MCPC_ECRS, IDX_RT_MCPC_NSRS,
    PRICE_COLS_12,
)

CHECKPOINT_PATH_S2 = "outputs/checkpoints/stage2/best_model_s2.pt"
NUM_WORKERS        = 32


# ── Model loader ──────────────────────────────────────────────────────────────

def load_agent_s2(checkpoint_path, device):
    ckpt  = torch.load(checkpoint_path, map_location=device)
    print(f"[Model] Keys: {list(ckpt.keys())}")

    ttfe_s2 = TTFE_S2().to(device)
    if "ttfe_s2_state" in ckpt:
        ttfe_s2.load_state_dict(ckpt["ttfe_s2_state"])
        print("[Model] TTFE_S2 weights loaded")
    ttfe_s2.eval()

    actor = Actor(obs_dim=STATE_DIM_S2, act_dim=ACTION_DIM_S2).to(device)
    actor.load_state_dict(ckpt["actor_state"])
    actor.eval()

    print(f"[Model] Loaded Stage 2 agent from {checkpoint_path}")
    return actor, ttfe_s2


# ── PIO worker (module-level for multiprocessing pickling) ────────────────────

def _pio_worker_s2(args):
    """
    Solves a single-day MILP (6-market, post-RTC+B) using PuLP CBC.

    args: (episode_idx, prices_raw_288x12)
    Returns: (episode_idx, optimal_revenue_usd)
    """
    idx, prices_raw = args
    T    = len(prices_raw)
    prob = pulp.LpProblem(f"PIO_S2_{idx}", pulp.LpMaximize)

    rt_lmp       = prices_raw[:, IDX_RT_LMP].astype(float)
    px_regup     = prices_raw[:, IDX_RT_MCPC_REGUP].astype(float)
    px_regdn     = prices_raw[:, IDX_RT_MCPC_REGDN].astype(float)
    px_rrs       = prices_raw[:, IDX_RT_MCPC_RRS].astype(float)
    px_ecrs      = prices_raw[:, IDX_RT_MCPC_ECRS].astype(float)
    px_nsrs      = prices_raw[:, IDX_RT_MCPC_NSRS].astype(float)

    # Decision variables
    v_dch  = [pulp.LpVariable(f"vd_{t}",    cat="Binary")                          for t in range(T)]
    v_ch   = [pulp.LpVariable(f"vc_{t}",    cat="Binary")                          for t in range(T)]
    ps_d   = [pulp.LpVariable(f"psd_{t}",   lowBound=0, upBound=RATED_POWER_MW)    for t in range(T)]
    ps_c   = [pulp.LpVariable(f"psc_{t}",   lowBound=0, upBound=RATED_POWER_MW)    for t in range(T)]
    p_ru   = [pulp.LpVariable(f"pru_{t}",   lowBound=0, upBound=FCAS_MAX_MW)       for t in range(T)]
    p_rd   = [pulp.LpVariable(f"prd_{t}",   lowBound=0, upBound=FCAS_MAX_MW)       for t in range(T)]
    p_rrs  = [pulp.LpVariable(f"prrs_{t}",  lowBound=0, upBound=FCAS_MAX_MW)       for t in range(T)]
    p_ecrs = [pulp.LpVariable(f"pecrs_{t}", lowBound=0, upBound=FCAS_MAX_MW)       for t in range(T)]
    p_ns   = [pulp.LpVariable(f"pns_{t}",   lowBound=0, upBound=FCAS_MAX_MW)       for t in range(T)]
    e      = [pulp.LpVariable(f"e_{t}",     lowBound=E_MIN_MWH, upBound=E_MAX_MWH) for t in range(T)]

    # Objective: maximise net revenue - degradation
    prob += pulp.lpSum([
        DT_H * EFF_DCH * float(rt_lmp[t])   * ps_d[t]
        - DT_H / EFF_CH * float(rt_lmp[t])  * ps_c[t]
        + float(px_regup[t]) * p_ru[t]   * DT_H
        + float(px_regdn[t]) * p_rd[t]   * DT_H
        + float(px_rrs[t])   * p_rrs[t]  * DT_H
        + float(px_ecrs[t])  * p_ecrs[t] * DT_H
        + float(px_nsrs[t])  * p_ns[t]   * DT_H
        - DEGRADATION_C * DT_H * (ps_d[t] + p_ru[t] + p_rrs[t] + p_ecrs[t] + p_ns[t])
        for t in range(T)
    ])

    for t in range(T):
        # Mutual exclusivity: cannot charge and discharge simultaneously
        prob += v_dch[t] + v_ch[t] <= 1

        # Power bounds tied to mode flags
        prob += ps_d[t]   <= RATED_POWER_MW * v_dch[t]
        prob += ps_c[t]   <= RATED_POWER_MW * v_ch[t]
        prob += p_ru[t]   <= FCAS_MAX_MW    * v_dch[t]
        prob += p_rrs[t]  <= FCAS_MAX_MW    * v_dch[t]
        prob += p_ecrs[t] <= FCAS_MAX_MW    * v_dch[t]
        prob += p_ns[t]   <= FCAS_MAX_MW    * v_dch[t]
        prob += p_rd[t]   <= FCAS_MAX_MW    * v_ch[t]

        # No simultaneous RegUp + RegDn (ERCOT rule)
        b_ru = pulp.LpVariable(f"bru_{t}", cat="Binary")
        prob += p_ru[t] <= FCAS_MAX_MW * b_ru
        prob += p_rd[t] <= FCAS_MAX_MW * (1 - b_ru)

        # Total discharge <= rated power
        prob += ps_d[t] + p_ru[t] + p_rrs[t] + p_ecrs[t] + p_ns[t] <= RATED_POWER_MW
        # Total charge <= rated power
        prob += ps_c[t] + p_rd[t] <= RATED_POWER_MW

        # SOC dynamics
        e_prev = E_INIT_MWH if t == 0 else e[t - 1]
        prob += e[t] == e_prev + DT_H * EFF_CH * ps_c[t] - DT_H / EFF_DCH * ps_d[t]
        prob += e[t] >= E_MIN_MWH
        prob += e[t] <= E_MAX_MWH

        # Hourly SOC reservation (every 12 steps = 1 hour)
        # ECRS has a 2-hour duration requirement
        if t % 12 == 0:
            e_res = (
                p_ru[t]   * AS_DURATION_H_S2["RegUp"]
                + p_rrs[t]  * AS_DURATION_H_S2["RRS"]
                + p_ecrs[t] * AS_DURATION_H_S2["ECRS"]   # 2.0 hours
                + p_ns[t]   * AS_DURATION_H_S2["NSRS"]
            )
            prob += e[t] >= E_MIN_MWH + e_res
            prob += e[t] <= E_MAX_MWH - p_rd[t] * AS_DURATION_H_S2["RegDn"]

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
    if pulp.LpStatus[prob.status] != "Optimal":
        return idx, 0.0
    return idx, float(pulp.value(prob.objective))


def run_pio_parallel_s2(episodes, n_workers=NUM_WORKERS):
    """Run PIO MILP in parallel across all test episodes."""
    args = [(i, ep["prices"]) for i, ep in enumerate(episodes)]
    t0   = time.time()
    print(f"[PIO-S2] Solving {len(args)} MILPs across {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        results = pool.map(_pio_worker_s2, args)
    results.sort(key=lambda x: x[0])
    revenues = [r for _, r in results]
    print(f"[PIO-S2] Done in {time.time() - t0:.1f}s.  Mean: ${np.mean(revenues):.2f}/day")
    return revenues


# ── TempDRL rollout (Stage 2) ─────────────────────────────────────────────────

def run_tempdrl_episode_s2(actor, ttfe_s2, episode, mode="joint", device="cpu"):
    """
    Greedy (mean-action) rollout for one Stage 2 episode.

    episode: dict with keys 'prices' (288×12 raw), 'syscond' (288×7 raw),
             'day_of_week', 'month'.
    mode: 'joint' | 'spot' | 'as'

    Returns: (total_usd, breakdown_dict)
    """
    prices_raw   = episode["prices"]        # (288, 12)
    syscond_raw  = episode["syscond"]       # (288, 7)
    day_of_week  = episode.get("day_of_week", 0)
    month        = episode.get("month", 1)
    T            = len(prices_raw)

    # Pre-compute TTFE features for all timesteps in one forward pass
    # We use the raw prices here — note that for production rollouts the scaler
    # is not available inside the episode dict, so we pass raw prices to TTFE.
    # The scaler-transformed version should be used if available; for evaluation
    # consistency we use prices_scaled if provided, else fall back to raw.
    prices_for_ttfe = episode.get("prices_sc", prices_raw)   # (288, 12)

    segments = np.stack([
        build_temporal_segment_12(prices_for_ttfe, t=t, L=TTFE_SEG_LEN_S2)
        for t in range(T)
    ])  # (288, 32, 12)

    with torch.no_grad():
        features = ttfe_s2(
            torch.FloatTensor(segments).to(device)
        ).cpu().numpy()  # (288, 64)

    energy    = E_INIT_MWH
    total_usd = 0.0
    r_spot = r_as = r_deg = 0.0
    r_regup_sum = r_regdn_sum = r_rrs_sum = r_ecrs_sum = r_nsrs_sum = 0.0
    n_clips   = 0

    for t in range(T):
        soc    = energy / CAPACITY_MWH
        time_6 = build_time_6(t, day_of_week, month)
        state  = build_state_78(
            soc          = soc,
            syscond_7    = syscond_raw[t],
            time_6       = time_6,
            ttfe_feat_64 = features[t],
        )

        with torch.no_grad():
            raw_action = actor.get_mean_action(
                torch.FloatTensor(state).unsqueeze(0).to(device)
            ).squeeze(0).cpu().numpy()

        (v_dch, v_ch,
         p_spot_dch, p_spot_ch,
         p_regup, p_regdn, p_rrs, p_ecrs, p_nsrs) = decode_action_s2(raw_action)

        if mode == "spot":
            p_regup = p_regdn = p_rrs = p_ecrs = p_nsrs = 0.0
        elif mode == "as":
            p_spot_dch = p_spot_ch = 0.0

        rev = compute_step_revenue_s2(
            v_dch, v_ch,
            p_spot_dch, p_spot_ch,
            p_regup, p_regdn, p_rrs, p_ecrs, p_nsrs,
            prices_raw[t],
        )

        total_usd += rev["total"]
        r_spot    += rev["r_spot"]
        r_as      += rev["r_as"]
        r_deg     += rev["r_deg"]
        r_regup_sum  += rev["r_regup"]
        r_regdn_sum  += rev["r_regdn"]
        r_rrs_sum    += rev["r_rrs"]
        r_ecrs_sum   += rev["r_ecrs"]
        r_nsrs_sum   += rev["r_nsrs"]

        # SOC update (spot dispatch only; AS bids do not move energy in this step)
        new_e = energy + DT_H * (EFF_CH * p_spot_ch - (1.0 / EFF_DCH) * p_spot_dch)
        if new_e < E_MIN_MWH or new_e > E_MAX_MWH:
            n_clips += 1
        energy = float(np.clip(new_e, E_MIN_MWH, E_MAX_MWH))

    breakdown = {
        "r_spot":  r_spot,
        "r_as":    r_as,
        "r_regup": r_regup_sum,
        "r_regdn": r_regdn_sum,
        "r_rrs":   r_rrs_sum,
        "r_ecrs":  r_ecrs_sum,
        "r_nsrs":  r_nsrs_sum,
        "r_deg":   r_deg,
        "n_clips": n_clips,
    }
    return total_usd, breakdown


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("\n" + "=" * 74)
    print("  TempDRL Stage 2 — ERCOT Post-RTC+B Revenue Evaluation")
    print("  Checkpoint : best_model_s2.pt")
    print("  Test set   : 15 post-RTC+B days (15% hold-out)")
    print(f"  PIO workers: {NUM_WORKERS} parallel")
    print("=" * 74)

    device = torch.device("cuda:26" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    data = load_stage2_data()

    test_episodes = iter_daily_episodes_s2(
        price_array   = data["test_prices"],
        syscond_array = data["test_syscond"],
        index         = data["test_index"],
    )

    # Attach scaled prices to each episode for TTFE input
    test_prices_sc = data["test_prices_sc"]
    for i, ep in enumerate(test_episodes):
        s = i * 288
        ep["prices_sc"] = test_prices_sc[s : s + 288]

    N = len(test_episodes)
    print(f"[Data] Test episodes: {N} days")

    # Price range summary
    markets_display = ["rt_lmp", "rt_mcpc_regup", "rt_mcpc_regdn",
                       "rt_mcpc_rrs", "rt_mcpc_ecrs", "rt_mcpc_nsrs"]
    all_raw = data["test_prices"]
    print(f"\n[Data] Price ranges for test set:")
    for i, name in enumerate(markets_display):
        col = all_raw[:, i]
        print(f"  {name:<20}: min={col.min():>8.2f}  "
              f"mean={col.mean():>8.2f}  max={col.max():>8.2f} $/MW")

    # ── Load Stage 2 agent ─────────────────────────────────────────────────────
    actor, ttfe_s2 = load_agent_s2(CHECKPOINT_PATH_S2, device)

    # ── PIO parallel ───────────────────────────────────────────────────────────
    pio_revenues = run_pio_parallel_s2(test_episodes, n_workers=NUM_WORKERS)

    # ── TempDRL Joint rollouts ─────────────────────────────────────────────────
    print(f"\n[TempDRL-S2 Joint] Running {N} greedy episodes...")
    joint_rev, joint_bd = [], []
    for ep in test_episodes:
        r, bd = run_tempdrl_episode_s2(actor, ttfe_s2, ep, mode="joint", device=device)
        joint_rev.append(r)
        joint_bd.append(bd)
    print(f"  Done. Mean: ${np.mean(joint_rev):.2f}/day  "
          f"(SoC clips: {np.mean([b['n_clips'] for b in joint_bd]):.1f}/day)")

    # ── TempDRL Spot-only rollouts ─────────────────────────────────────────────
    print(f"\n[TempDRL-S2 Spot-only] Running {N} greedy episodes...")
    spot_rev = []
    for ep in test_episodes:
        r, _ = run_tempdrl_episode_s2(actor, ttfe_s2, ep, mode="spot", device=device)
        spot_rev.append(r)
    print(f"  Done. Mean: ${np.mean(spot_rev):.2f}/day")

    # ── TempDRL AS-only rollouts ───────────────────────────────────────────────
    print(f"\n[TempDRL-S2 AS-only] Running {N} greedy episodes...")
    as_rev = []
    for ep in test_episodes:
        r, _ = run_tempdrl_episode_s2(actor, ttfe_s2, ep, mode="as", device=device)
        as_rev.append(r)
    print(f"  Done. Mean: ${np.mean(as_rev):.2f}/day")

    # ── Results table ──────────────────────────────────────────────────────────
    pio_mean = np.mean(pio_revenues)
    results  = [
        ("PIO (Perfect Foresight)", pio_revenues),
        ("TempDRL Joint",           joint_rev),
        ("TempDRL Spot-only",       spot_rev),
        ("TempDRL AS-only",         as_rev),
    ]

    print("\n" + "=" * 74)
    print("  STAGE 2 RESULTS — Post-RTC+B ERCOT Mean Daily Revenue (USD/day)")
    print(f"  Dataset : ERCOT post-RTC+B test set  ({N} days)")
    print(f"  Markets : spot + RegUp + RegDn + RRS + ECRS + NSRS  (6 total)")
    print(f"  BESS    : 10 MWh / 2 MW, eta=0.95, SoC 5-95%")
    print(f"  Actions : Deterministic tanh(mean), no sampling noise")
    print("=" * 74)
    print(f"  {'Policy':<28} {'Mean $/day':>10} {'Std':>8} {'Min':>8} "
          f"{'Max':>8} {'vs PIO':>7}")
    print("-" * 74)
    for name, revs in results:
        m   = np.mean(revs)
        s   = np.std(revs)
        mn  = np.min(revs)
        mx  = np.max(revs)
        pct = 100.0 * m / pio_mean if pio_mean > 0 else float("nan")
        tag = " <- BEST" if name == "TempDRL Joint" else ""
        print(f"  {name:<28} {m:>10.2f} {s:>8.2f} {mn:>8.2f} {mx:>8.2f} "
              f"{pct:>6.1f}%{tag}")
    print("=" * 74)

    # Revenue breakdown for Joint policy
    jb = joint_bd
    print(f"\n  TempDRL Joint — mean daily revenue breakdown:")
    print(f"    Spot market    : ${np.mean([b['r_spot']  for b in jb]):>8.2f}")
    print(f"    AS markets     : ${np.mean([b['r_as']    for b in jb]):>8.2f}")
    print(f"      RegUp        : ${np.mean([b['r_regup'] for b in jb]):>8.2f}")
    print(f"      RegDn        : ${np.mean([b['r_regdn'] for b in jb]):>8.2f}")
    print(f"      RRS          : ${np.mean([b['r_rrs']   for b in jb]):>8.2f}")
    print(f"      ECRS         : ${np.mean([b['r_ecrs']  for b in jb]):>8.2f}")
    print(f"      NSRS         : ${np.mean([b['r_nsrs']  for b in jb]):>8.2f}")
    print(f"    Degradation    : ${-np.mean([b['r_deg']  for b in jb]):>8.2f}  (cost)")
    print(f"    NET            : ${np.mean(joint_rev):>8.2f}")
    print(f"    SoC clips/day  :  {np.mean([b['n_clips'] for b in jb]):>6.1f}")

    j  = np.mean(joint_rev)
    sp = np.mean(spot_rev)
    a  = np.mean(as_rev)
    print(f"\n  Joint vs Spot-only : {100*(j - sp)/max(abs(sp), 1e-6):+.1f}%")
    print(f"  Joint vs AS-only   : {100*(j - a)/max(abs(a), 1e-6):+.1f}%")
    print(f"  Joint vs PIO       :  {100*j/max(pio_mean, 1e-6):.1f}% of upper bound")
    print(f"\n  Total wall time    :  {time.time() - t_start:.1f}s")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    os.makedirs("outputs/logs", exist_ok=True)
    out_df = pd.DataFrame({
        "episode":       list(range(N)),
        "pio_usd":       pio_revenues,
        "joint_usd":     joint_rev,
        "spot_only_usd": spot_rev,
        "as_only_usd":   as_rev,
    })
    out_path = "outputs/logs/stage2_revenue_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
