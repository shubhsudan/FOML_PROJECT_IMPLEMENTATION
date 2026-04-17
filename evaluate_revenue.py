"""
evaluate_revenue.py — ERCOT-correct Stage 1 Revenue Evaluation.

Computes real USD/day for 4 policies on ERCOT 2022 Nov-Dec eval set.

Policies:
  1. PIO           — Perfect foresight MILP (oracle upper bound)
  2. TempDRL Joint — Trained agent, all 5 markets
  3. TempDRL Spot  — Trained agent, AS zeroed out
  4. TempDRL AS    — Trained agent, spot zeroed out

Revenue formulas (ERCOT-correct):
  - Spot: DT_H × (η_dch × rt_lmp × p_spot_dch − rt_lmp/η_ch × p_spot_ch)
  - AS:   price ($/MW) × MW_bid × DT_H  (capacity payment, not energy)
  - Deg:  DEGRADATION_C × DT_H × v_dch × total_dch_mw

Markets: spot + RegUp + RegDn + RRS + NSRS (5 total, ECRS excluded)
Actions: Deterministic tanh(mean), no sampling noise.

Run:
    cd ~/tempdrl
    conda activate tempdrl
    python evaluate_revenue.py 2>&1 | tee outputs/logs/stage1_eval_ercot.log
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch
import pulp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from ttfe        import TTFE, build_temporal_segment
from sac_agent   import Actor
from data_loader import build_state, load_all, iter_daily_episodes
from environment import compute_step_revenue, decode_action
from config import (
    CAPACITY_MWH, RATED_POWER_MW, FCAS_MAX_MW,
    EFF_CH, EFF_DCH, E_MIN_MWH, E_MAX_MWH, E_INIT_MWH,
    DEGRADATION_C, DT_H, TIMESTEPS_PER_DAY, NUM_MARKETS,
    AS_DURATION_H, TEMPORAL_SEG_LEN,
)

# ─────────────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "outputs/checkpoints/best_model.pt"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_agent(checkpoint_path, device):
    """
    Loads Actor and TTFE from checkpoint.
    If 'ttfe_state' is present (ERCOT retrain), loads paired TTFE weights.
    Falls back to seed=0 init if not found.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"[Model] Checkpoint keys: {list(ckpt.keys())}")

    actor = Actor().to(device)
    actor.load_state_dict(ckpt["actor_state"])
    actor.eval()
    print(f"[Model] Actor loaded from {checkpoint_path}")

    ttfe = TTFE().to(device)
    if "ttfe_state" in ckpt:
        ttfe.load_state_dict(ckpt["ttfe_state"])
        print(f"[Model] TTFE loaded from checkpoint  ← PAIRED WEIGHTS")
    else:
        torch.manual_seed(0)
        ttfe = TTFE().to(device)
        print(f"[Model] TTFE not in checkpoint — seed=0 init (random fallback)")
    ttfe.eval()

    return actor, ttfe


# ─────────────────────────────────────────────────────────────────────────────
# PIO — ERCOT-correct MILP
# ─────────────────────────────────────────────────────────────────────────────

def run_pio_episode(prices_raw):
    """
    Perfect Information Optimisation for ERCOT 2022.
    prices_raw: (288, 5) — [rt_lmp, regup, regdn, rrs, nsrs]

    Enforces:
    - Duration-based AS qualification
    - Hourly SOC floor (discharge AS energy reservation)
    - Hourly SOC ceiling (RegDn headroom)
    - No simultaneous RegUp + RegDn
    - Total power constraints per direction
    """
    T    = len(prices_raw)
    prob = pulp.LpProblem("BESS_PIO_ERCOT", pulp.LpMaximize)

    # Unpack prices (all float)
    rt_lmp   = prices_raw[:, 0].astype(float)
    px_regup = prices_raw[:, 1].astype(float)
    px_regdn = prices_raw[:, 2].astype(float)
    px_rrs   = prices_raw[:, 3].astype(float)
    px_nsrs  = prices_raw[:, 4].astype(float)

    # Binary mode flags
    v_dch = [pulp.LpVariable(f"vd_{t}", cat="Binary") for t in range(T)]
    v_ch  = [pulp.LpVariable(f"vc_{t}", cat="Binary") for t in range(T)]

    # Bid sub-variables — linearised (separate dch/ch to avoid bilinear products)
    ps_d  = [pulp.LpVariable(f"psd_{t}",  lowBound=0, upBound=RATED_POWER_MW) for t in range(T)]
    ps_c  = [pulp.LpVariable(f"psc_{t}",  lowBound=0, upBound=RATED_POWER_MW) for t in range(T)]
    p_ru  = [pulp.LpVariable(f"pru_{t}",  lowBound=0, upBound=FCAS_MAX_MW)    for t in range(T)]
    p_rd  = [pulp.LpVariable(f"prd_{t}",  lowBound=0, upBound=FCAS_MAX_MW)    for t in range(T)]
    p_rrs = [pulp.LpVariable(f"prrs_{t}", lowBound=0, upBound=FCAS_MAX_MW)    for t in range(T)]
    p_ns  = [pulp.LpVariable(f"pns_{t}",  lowBound=0, upBound=FCAS_MAX_MW)    for t in range(T)]

    # SoC state variable
    e = [pulp.LpVariable(f"e_{t}", lowBound=E_MIN_MWH, upBound=E_MAX_MWH) for t in range(T)]

    # Objective
    prob += pulp.lpSum([
        DT_H * EFF_DCH * float(rt_lmp[t]) * ps_d[t]
        - DT_H / EFF_CH * float(rt_lmp[t]) * ps_c[t]
        + float(px_regup[t]) * p_ru[t]  * DT_H
        + float(px_regdn[t]) * p_rd[t]  * DT_H
        + float(px_rrs[t])   * p_rrs[t] * DT_H
        + float(px_nsrs[t])  * p_ns[t]  * DT_H
        - DEGRADATION_C * DT_H * (ps_d[t] + p_ru[t] + p_rrs[t] + p_ns[t])
        for t in range(T)
    ])

    for t in range(T):
        # Mutual exclusivity
        prob += v_dch[t] + v_ch[t] <= 1

        # Big-M: bids zero unless corresponding mode active
        prob += ps_d[t]  <= RATED_POWER_MW * v_dch[t]
        prob += ps_c[t]  <= RATED_POWER_MW * v_ch[t]
        prob += p_ru[t]  <= FCAS_MAX_MW * v_dch[t]
        prob += p_rrs[t] <= FCAS_MAX_MW * v_dch[t]
        prob += p_ns[t]  <= FCAS_MAX_MW * v_dch[t]
        prob += p_rd[t]  <= FCAS_MAX_MW * v_ch[t]

        # No simultaneous RegUp + RegDn
        b_ru = pulp.LpVariable(f"bru_{t}", cat="Binary")
        prob += p_ru[t] <= FCAS_MAX_MW * b_ru
        prob += p_rd[t] <= FCAS_MAX_MW * (1 - b_ru)

        # Total power constraints
        prob += ps_d[t] + p_ru[t] + p_rrs[t] + p_ns[t] <= RATED_POWER_MW
        prob += ps_c[t] + p_rd[t] <= RATED_POWER_MW

        # Energy dynamics (spot market only — AS is capacity reservation)
        e_prev = E_INIT_MWH if t == 0 else e[t - 1]
        prob += e[t] == e_prev + DT_H * EFF_CH * ps_c[t] - DT_H / EFF_DCH * ps_d[t]

        # SoC bounds
        prob += e[t] >= E_MIN_MWH
        prob += e[t] <= E_MAX_MWH

        # ERCOT hourly SOC floor/ceiling rule (every hour start)
        if t % 12 == 0:
            e_reserved = (p_ru[t]  * AS_DURATION_H["RegUp"]
                         + p_rrs[t] * AS_DURATION_H["RRS"]
                         + p_ns[t]  * AS_DURATION_H["NSRS"])
            prob += e[t] >= E_MIN_MWH + e_reserved
            prob += e[t] <= E_MAX_MWH - p_rd[t] * AS_DURATION_H["RegDn"]

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        print(f"  [PIO] WARNING: {pulp.LpStatus[prob.status]}")
        return 0.0

    return float(pulp.value(prob.objective))


# ─────────────────────────────────────────────────────────────────────────────
# TEMPDRL GREEDY ROLLOUT
# ─────────────────────────────────────────────────────────────────────────────

def run_tempdrl_episode(actor, ttfe, ep_raw, ep_scaled, mode="joint", device="cpu"):
    """
    Deterministic greedy rollout. Returns real USD/day via ERCOT revenue formulas.

    ep_raw:    (288, 5) raw prices — for revenue calculation
    ep_scaled: (288, 5) scaled prices — for TTFE input
    mode:      "joint" | "spot" | "as"
    """
    T = len(ep_raw)

    # Batched TTFE forward pass
    segments = np.stack([
        build_temporal_segment(ep_scaled, t=t, L=TEMPORAL_SEG_LEN) for t in range(T)
    ])  # (288, L, 5)
    with torch.no_grad():
        features = ttfe(torch.FloatTensor(segments).to(device)).cpu().numpy()  # (288, 64)

    energy        = E_INIT_MWH
    total_usd     = 0.0
    r_spot_total  = 0.0
    r_as_total    = 0.0
    r_deg_total   = 0.0
    n_soc_clips   = 0

    for t in range(T):
        soc    = energy / CAPACITY_MWH
        prices = ep_raw[t]
        f_t    = features[t]

        state = build_state(soc, prices, f_t, t)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            raw_action = actor.get_mean_action(state_t).squeeze(0).cpu().numpy()

        (v_dch, v_ch, p_spot_dch, p_spot_ch,
         p_regup, p_regdn, p_rrs, p_nsrs) = decode_action(raw_action)

        if mode == "spot":
            p_regup = p_regdn = p_rrs = p_nsrs = 0.0
        elif mode == "as":
            p_spot_dch = p_spot_ch = 0.0

        rev = compute_step_revenue(
            v_dch, v_ch, p_spot_dch, p_spot_ch,
            p_regup, p_regdn, p_rrs, p_nsrs, prices
        )

        total_usd    += rev["total"]
        r_spot_total += rev["r_spot"]
        r_as_total   += rev["r_as"]
        r_deg_total  += rev["r_deg"]

        # Energy dynamics
        delta_e    = DT_H * (EFF_CH * p_spot_ch - (1.0 / EFF_DCH) * p_spot_dch)
        new_energy = energy + delta_e
        if new_energy < E_MIN_MWH or new_energy > E_MAX_MWH:
            n_soc_clips += 1
        energy = np.clip(new_energy, E_MIN_MWH, E_MAX_MWH)

    return total_usd, {
        "r_spot": r_spot_total,
        "r_as":   r_as_total,
        "r_deg":  r_deg_total,
        "n_soc_clips": n_soc_clips,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  TempDRL — ERCOT-Correct Stage 1 Revenue Evaluation")
    print("=" * 65)

    device = torch.device(f"cuda:26" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    data = load_all()
    eval_raw    = iter_daily_episodes(data["eval_raw"])
    eval_scaled = iter_daily_episodes(data["eval_scaled"])
    N = len(eval_raw)

    print(f"\n[Data] Price ranges for eval set:")
    markets = ["spot", "RegUp", "RegDn", "RRS", "NSRS"]
    all_raw = np.concatenate(eval_raw, axis=0)
    for i, name in enumerate(markets):
        col = all_raw[:, i]
        print(f"  {name:<8}: min={col.min():>8.2f}  mean={col.mean():>8.2f}  max={col.max():>8.2f} $/MW")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    actor, ttfe = load_agent(CHECKPOINT_PATH, device)

    # ── 3. PIO ────────────────────────────────────────────────────────────────
    print(f"\n[PIO] Running perfect foresight MILP on {N} episodes...")
    pio_revenues = []
    for i, ep_raw in enumerate(eval_raw):
        rev = run_pio_episode(ep_raw)
        pio_revenues.append(rev)
        if (i + 1) % 10 == 0 or (i + 1) == N:
            print(f"  ep {i+1:3d}/{N}  mean so far: ${np.mean(pio_revenues):.2f}/day")

    # ── 4. TempDRL Joint ──────────────────────────────────────────────────────
    print(f"\n[TempDRL Joint] Running {N} greedy episodes...")
    joint_revenues, joint_breaks = [], []
    for ep_raw, ep_sc in zip(eval_raw, eval_scaled):
        rev, bd = run_tempdrl_episode(actor, ttfe, ep_raw, ep_sc, mode="joint", device=device)
        joint_revenues.append(rev); joint_breaks.append(bd)
    print(f"  Done. Mean: ${np.mean(joint_revenues):.2f}/day  "
          f"(SoC clips: {np.mean([b['n_soc_clips'] for b in joint_breaks]):.1f}/day)")

    # ── 5. TempDRL Spot-only ──────────────────────────────────────────────────
    print(f"\n[TempDRL Spot-only] Running {N} greedy episodes...")
    spot_revenues = []
    for ep_raw, ep_sc in zip(eval_raw, eval_scaled):
        rev, _ = run_tempdrl_episode(actor, ttfe, ep_raw, ep_sc, mode="spot", device=device)
        spot_revenues.append(rev)
    print(f"  Done. Mean: ${np.mean(spot_revenues):.2f}/day")

    # ── 6. TempDRL AS-only ────────────────────────────────────────────────────
    print(f"\n[TempDRL AS-only] Running {N} greedy episodes...")
    as_revenues = []
    for ep_raw, ep_sc in zip(eval_raw, eval_scaled):
        rev, _ = run_tempdrl_episode(actor, ttfe, ep_raw, ep_sc, mode="as", device=device)
        as_revenues.append(rev)
    print(f"  Done. Mean: ${np.mean(as_revenues):.2f}/day")

    # ── 7. Results table ──────────────────────────────────────────────────────
    pio_mean = np.mean(pio_revenues)
    results = [
        ("PIO (Perfect Foresight)", pio_revenues),
        ("TempDRL Joint",           joint_revenues),
        ("TempDRL Spot-only",       spot_revenues),
        ("TempDRL AS-only",         as_revenues),
    ]

    print("\n" + "=" * 72)
    print("  STAGE 1 RESULTS — ERCOT-Correct Mean Daily Revenue (USD/day)")
    print(f"  Dataset : ERCOT 2022, Nov–Dec eval set  ({N} days)")
    print(f"  Markets : spot + RegUp + RegDn + RRS + NSRS  (5 total, ECRS excluded)")
    print(f"  BESS    : 10 MWh / 2 MW, η=0.95, SoC 5–95%")
    print(f"  Actions : Deterministic tanh(mean), no sampling noise")
    print(f"  Revenue : ERCOT capacity payments — price × MW × dt_h")
    print("=" * 72)
    print(f"  {'Policy':<28} {'Mean $/day':>10} {'Std':>8} {'Min':>8} {'Max':>8} {'vs PIO':>7}")
    print("-" * 72)
    for name, revs in results:
        m   = np.mean(revs); s = np.std(revs)
        mn  = np.min(revs);  mx = np.max(revs)
        pct = 100.0 * m / pio_mean if pio_mean > 0 else float("nan")
        tag = " ◀ BEST" if name == "TempDRL Joint" else ""
        print(f"  {name:<28} {m:>10.2f} {s:>8.2f} {mn:>8.2f} {mx:>8.2f} {pct:>6.1f}%{tag}")
    print("=" * 72)

    jb = joint_breaks
    print(f"\n  TempDRL Joint — mean daily revenue breakdown:")
    print(f"    Spot market    : ${np.mean([b['r_spot'] for b in jb]):>8.2f}")
    print(f"    AS markets     : ${np.mean([b['r_as']   for b in jb]):>8.2f}")
    print(f"    Degradation    : ${-np.mean([b['r_deg']  for b in jb]):>8.2f}  (cost)")
    print(f"    NET            : ${np.mean(joint_revenues):>8.2f}")
    print(f"    SoC clips/day  :  {np.mean([b['n_soc_clips'] for b in jb]):>6.1f}")

    j   = np.mean(joint_revenues)
    sp  = np.mean(spot_revenues)
    a   = np.mean(as_revenues)
    print(f"\n  Joint vs Spot-only : {100*(j-sp)/max(abs(sp),1e-6):+.1f}%")
    print(f"  Joint vs AS-only   : {100*(j-a)/max(abs(a),1e-6):+.1f}%")
    print(f"  Joint vs PIO       :  {100*j/max(pio_mean,1e-6):.1f}% of upper bound")

    # ── 8. Save results ───────────────────────────────────────────────────────
    os.makedirs("outputs/logs", exist_ok=True)
    out_df = pd.DataFrame({
        "episode":      list(range(N)),
        "pio_usd":      pio_revenues,
        "joint_usd":    joint_revenues,
        "spot_only_usd": spot_revenues,
        "as_only_usd":  as_revenues,
    })
    out_path = "outputs/logs/stage1_revenue_results_ercot.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
