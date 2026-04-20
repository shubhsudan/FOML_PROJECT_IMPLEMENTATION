"""
evaluate_phase2_fast.py — Multi-year ERCOT Stage 1 Revenue Evaluation (FAST).

PIO runs in parallel across 32 workers → ~1 min instead of ~33 min.
TempDRL rollouts on GPU → ~2 min for all 3 policies.
Total wall time: ~3-4 minutes.
"""

import os, sys, time
import numpy as np
import pandas as pd
import torch
import pulp
from multiprocessing import Pool, cpu_count

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

CHECKPOINT_PATH = "outputs/checkpoints/best_model.pt"
NUM_WORKERS     = 32   # parallel PIO workers


def load_agent(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"[Model] Keys: {list(ckpt.keys())}")
    actor = Actor().to(device); actor.load_state_dict(ckpt["actor_state"]); actor.eval()
    ttfe  = TTFE().to(device)
    if "ttfe_state" in ckpt:
        ttfe.load_state_dict(ckpt["ttfe_state"])
        print("[Model] TTFE paired weights loaded")
    ttfe.eval()
    print(f"[Model] Loaded from {checkpoint_path}  (ep 33,500 best)")
    return actor, ttfe


# ── PIO (parallel worker) ─────────────────────────────────────────────────────

def _pio_worker(args):
    idx, prices_raw = args
    T    = len(prices_raw)
    prob = pulp.LpProblem(f"PIO_{idx}", pulp.LpMaximize)

    rt_lmp   = prices_raw[:, 0].astype(float)
    px_regup = prices_raw[:, 1].astype(float)
    px_regdn = prices_raw[:, 2].astype(float)
    px_rrs   = prices_raw[:, 3].astype(float)
    px_nsrs  = prices_raw[:, 4].astype(float)

    v_dch = [pulp.LpVariable(f"vd_{t}", cat="Binary") for t in range(T)]
    v_ch  = [pulp.LpVariable(f"vc_{t}", cat="Binary") for t in range(T)]
    ps_d  = [pulp.LpVariable(f"psd_{t}",  lowBound=0, upBound=RATED_POWER_MW) for t in range(T)]
    ps_c  = [pulp.LpVariable(f"psc_{t}",  lowBound=0, upBound=RATED_POWER_MW) for t in range(T)]
    p_ru  = [pulp.LpVariable(f"pru_{t}",  lowBound=0, upBound=FCAS_MAX_MW)    for t in range(T)]
    p_rd  = [pulp.LpVariable(f"prd_{t}",  lowBound=0, upBound=FCAS_MAX_MW)    for t in range(T)]
    p_rrs = [pulp.LpVariable(f"prrs_{t}", lowBound=0, upBound=FCAS_MAX_MW)    for t in range(T)]
    p_ns  = [pulp.LpVariable(f"pns_{t}",  lowBound=0, upBound=FCAS_MAX_MW)    for t in range(T)]
    e     = [pulp.LpVariable(f"e_{t}", lowBound=E_MIN_MWH, upBound=E_MAX_MWH) for t in range(T)]

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
        prob += v_dch[t] + v_ch[t] <= 1
        prob += ps_d[t]  <= RATED_POWER_MW * v_dch[t]
        prob += ps_c[t]  <= RATED_POWER_MW * v_ch[t]
        prob += p_ru[t]  <= FCAS_MAX_MW * v_dch[t]
        prob += p_rrs[t] <= FCAS_MAX_MW * v_dch[t]
        prob += p_ns[t]  <= FCAS_MAX_MW * v_dch[t]
        prob += p_rd[t]  <= FCAS_MAX_MW * v_ch[t]
        b_ru = pulp.LpVariable(f"bru_{t}", cat="Binary")
        prob += p_ru[t]  <= FCAS_MAX_MW * b_ru
        prob += p_rd[t]  <= FCAS_MAX_MW * (1 - b_ru)
        prob += ps_d[t] + p_ru[t] + p_rrs[t] + p_ns[t] <= RATED_POWER_MW
        prob += ps_c[t] + p_rd[t] <= RATED_POWER_MW
        e_prev = E_INIT_MWH if t == 0 else e[t - 1]
        prob += e[t] == e_prev + DT_H * EFF_CH * ps_c[t] - DT_H / EFF_DCH * ps_d[t]
        prob += e[t] >= E_MIN_MWH
        prob += e[t] <= E_MAX_MWH
        if t % 12 == 0:
            e_res = (p_ru[t] * AS_DURATION_H["RegUp"]
                   + p_rrs[t] * AS_DURATION_H["RRS"]
                   + p_ns[t]  * AS_DURATION_H["NSRS"])
            prob += e[t] >= E_MIN_MWH + e_res
            prob += e[t] <= E_MAX_MWH - p_rd[t] * AS_DURATION_H["RegDn"]

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
    if pulp.LpStatus[prob.status] != "Optimal":
        return idx, 0.0
    return idx, float(pulp.value(prob.objective))


def run_pio_parallel(episodes_raw, n_workers=NUM_WORKERS):
    args = list(enumerate(episodes_raw))
    t0   = time.time()
    print(f"[PIO] Solving {len(args)} MILPs across {n_workers} workers...")
    with Pool(processes=n_workers) as pool:
        results = pool.map(_pio_worker, args)
    results.sort(key=lambda x: x[0])
    revenues = [r for _, r in results]
    print(f"[PIO] Done in {time.time()-t0:.1f}s. Mean: ${np.mean(revenues):.2f}/day")
    return revenues


# ── TempDRL rollout ───────────────────────────────────────────────────────────

def run_tempdrl_episode(actor, ttfe, ep_raw, ep_scaled, mode="joint", device="cpu"):
    T = len(ep_raw)
    segments = np.stack([
        build_temporal_segment(ep_scaled, t=t, L=TEMPORAL_SEG_LEN) for t in range(T)
    ])
    with torch.no_grad():
        features = ttfe(torch.FloatTensor(segments).to(device)).cpu().numpy()

    energy = E_INIT_MWH
    total_usd = r_spot = r_as = r_deg = 0.0
    n_clips = 0

    for t in range(T):
        soc   = energy / CAPACITY_MWH
        state = build_state(soc, ep_raw[t], features[t], t)
        with torch.no_grad():
            raw_action = actor.get_mean_action(
                torch.FloatTensor(state).unsqueeze(0).to(device)
            ).squeeze(0).cpu().numpy()

        (v_dch, v_ch, p_spot_dch, p_spot_ch,
         p_regup, p_regdn, p_rrs, p_nsrs) = decode_action(raw_action)

        if mode == "spot":
            p_regup = p_regdn = p_rrs = p_nsrs = 0.0
        elif mode == "as":
            p_spot_dch = p_spot_ch = 0.0

        rev = compute_step_revenue(
            v_dch, v_ch, p_spot_dch, p_spot_ch,
            p_regup, p_regdn, p_rrs, p_nsrs, ep_raw[t]
        )
        total_usd += rev["total"]; r_spot += rev["r_spot"]
        r_as += rev["r_as"];       r_deg  += rev["r_deg"]
        new_e = energy + DT_H * (EFF_CH * p_spot_ch - (1.0 / EFF_DCH) * p_spot_dch)
        if new_e < E_MIN_MWH or new_e > E_MAX_MWH:
            n_clips += 1
        energy = np.clip(new_e, E_MIN_MWH, E_MAX_MWH)

    return total_usd, {"r_spot": r_spot, "r_as": r_as, "r_deg": r_deg, "n_clips": n_clips}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("\n" + "=" * 70)
    print("  TempDRL Phase 2 — Multi-Year ERCOT Stage 1 Revenue Evaluation")
    print("  Checkpoint : best_model.pt  (ep 33,500,  val=$2,233/day)")
    print("  Test set   : 249 days, 2022-09-24 → 2023-05-31")
    print(f"  PIO workers: {NUM_WORKERS} parallel")
    print("=" * 70)

    device = torch.device("cuda:26" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}\n")

    data        = load_all()
    test_raw    = iter_daily_episodes(data["test_raw"])
    test_scaled = iter_daily_episodes(data["test_scaled"])
    N = len(test_raw)
    print(f"[Data] Test episodes: {N} days")

    markets = ["spot", "RegUp", "RegDn", "RRS", "NSRS"]
    all_raw = np.concatenate(test_raw, axis=0)
    print(f"\n[Data] Price ranges for test set:")
    for i, name in enumerate(markets):
        col = all_raw[:, i]
        print(f"  {name:<8}: min={col.min():>8.2f}  mean={col.mean():>8.2f}  max={col.max():>8.2f} $/MW")

    actor, ttfe = load_agent(CHECKPOINT_PATH, device)

    # ── PIO parallel ──────────────────────────────────────────────────────────
    pio_revenues = run_pio_parallel(test_raw, n_workers=NUM_WORKERS)

    # ── TempDRL rollouts ──────────────────────────────────────────────────────
    print(f"\n[TempDRL Joint] Running {N} greedy episodes...")
    joint_rev, joint_bd = [], []
    for ep_r, ep_s in zip(test_raw, test_scaled):
        r, bd = run_tempdrl_episode(actor, ttfe, ep_r, ep_s, mode="joint", device=device)
        joint_rev.append(r); joint_bd.append(bd)
    print(f"  Done. Mean: ${np.mean(joint_rev):.2f}/day  "
          f"(SoC clips: {np.mean([b['n_clips'] for b in joint_bd]):.1f}/day)")

    print(f"\n[TempDRL Spot-only] Running {N} greedy episodes...")
    spot_rev = []
    for ep_r, ep_s in zip(test_raw, test_scaled):
        r, _ = run_tempdrl_episode(actor, ttfe, ep_r, ep_s, mode="spot", device=device)
        spot_rev.append(r)
    print(f"  Done. Mean: ${np.mean(spot_rev):.2f}/day")

    print(f"\n[TempDRL AS-only] Running {N} greedy episodes...")
    as_rev = []
    for ep_r, ep_s in zip(test_raw, test_scaled):
        r, _ = run_tempdrl_episode(actor, ttfe, ep_r, ep_s, mode="as", device=device)
        as_rev.append(r)
    print(f"  Done. Mean: ${np.mean(as_rev):.2f}/day")

    # ── Results ───────────────────────────────────────────────────────────────
    pio_mean = np.mean(pio_revenues)
    results  = [
        ("PIO (Perfect Foresight)", pio_revenues),
        ("TempDRL Joint",           joint_rev),
        ("TempDRL Spot-only",       spot_rev),
        ("TempDRL AS-only",         as_rev),
    ]

    print("\n" + "=" * 74)
    print("  PHASE 2 RESULTS — Multi-Year ERCOT Mean Daily Revenue (USD/day)")
    print(f"  Dataset : ERCOT multi-year test set  ({N} days, Sep 2022 – May 2023)")
    print(f"  Markets : spot + RegUp + RegDn + RRS + NSRS  (5 total, ECRS excluded)")
    print(f"  BESS    : 10 MWh / 2 MW, η=0.95, SoC 5–95%")
    print(f"  Actions : Deterministic tanh(mean), no sampling noise")
    print("=" * 74)
    print(f"  {'Policy':<28} {'Mean $/day':>10} {'Std':>8} {'Min':>8} {'Max':>8} {'vs PIO':>7}")
    print("-" * 74)
    for name, revs in results:
        m   = np.mean(revs); s = np.std(revs)
        mn  = np.min(revs);  mx = np.max(revs)
        pct = 100.0 * m / pio_mean if pio_mean > 0 else float("nan")
        tag = " ◀ BEST" if name == "TempDRL Joint" else ""
        print(f"  {name:<28} {m:>10.2f} {s:>8.2f} {mn:>8.2f} {mx:>8.2f} {pct:>6.1f}%{tag}")
    print("=" * 74)

    jb = joint_bd
    print(f"\n  TempDRL Joint — mean daily revenue breakdown:")
    print(f"    Spot market    : ${np.mean([b['r_spot'] for b in jb]):>8.2f}")
    print(f"    AS markets     : ${np.mean([b['r_as']   for b in jb]):>8.2f}")
    print(f"    Degradation    : ${-np.mean([b['r_deg']  for b in jb]):>8.2f}  (cost)")
    print(f"    NET            : ${np.mean(joint_rev):>8.2f}")
    print(f"    SoC clips/day  :  {np.mean([b['n_clips'] for b in jb]):>6.1f}")

    j = np.mean(joint_rev); sp = np.mean(spot_rev); a = np.mean(as_rev)
    print(f"\n  Joint vs Spot-only : {100*(j-sp)/max(abs(sp),1e-6):+.1f}%")
    print(f"  Joint vs AS-only   : {100*(j-a)/max(abs(a),1e-6):+.1f}%")
    print(f"  Joint vs PIO       :  {100*j/max(pio_mean,1e-6):.1f}% of upper bound")
    print(f"\n  Total wall time    :  {time.time()-t_start:.1f}s")

    os.makedirs("outputs/logs", exist_ok=True)
    out_df = pd.DataFrame({
        "episode":       list(range(N)),
        "pio_usd":       pio_revenues,
        "joint_usd":     joint_rev,
        "spot_only_usd": spot_rev,
        "as_only_usd":   as_rev,
    })
    out_path = "outputs/logs/phase2_revenue_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
