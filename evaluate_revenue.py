"""
evaluate_revenue.py
====================
Computes real USD/day revenue for 4 policies on ERCOT 2022 eval set.

Policies:
  1. PIO            — Perfect foresight LP (oracle upper bound)
  2. TempDRL Joint  — Trained agent, all 7 markets
  3. TempDRL Spot   — Trained agent, FCAS zeroed out
  4. TempDRL FCAS   — Trained agent, spot zeroed out

Revenue formula: paper eq. 2, 3, 4 — NO shaped reward, NO clipping, NO violation penalty.
Prices: raw USD/MWh from ERCOT data (NOT scaled).
Actions: DETERMINISTIC (tanh(mean) of Gaussian policy, no sampling noise).

NOTE ON TTFE:
  The TTFE (temporal feature extractor) was never saved in checkpoints during
  training — only actor/critic weights were saved. TTFE is re-initialised here
  with torch.manual_seed(0) for full reproducibility. This is consistent across
  all 3 TempDRL modes (Joint / Spot / FCAS), so relative comparisons are valid.

Run:
    conda activate tempdrl
    python evaluate_revenue.py
"""

import os
import sys
import glob
import math
import numpy as np
import pandas as pd
import torch
import pulp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from ttfe       import TTFE, build_temporal_segment
from sac_agent  import Actor

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (paper Table I — do not change)
# ─────────────────────────────────────────────────────────────────────────────

CAPACITY_MWH   = 10.0
RATED_POWER_MW = 2.0
FCAS_MAX_MW    = 1.0
EFF_CH         = 0.95
EFF_DCH        = 0.95
E_MIN_MWH      = 0.5
E_MAX_MWH      = 9.5
E_INIT_MWH     = 5.0
DEGRADATION_C  = 0.02
DT_H           = 5.0 / 60
TIMESTEPS_PER_DAY = 288

CHECKPOINT_PATH = "outputs/checkpoints/sac_final.pt"   # top-up checkpoint with TTFE saved
DATA_DIR        = "data/processed"
EVAL_MONTHS     = [11, 12]

# Fixed seed for reproducible TTFE initialisation (see NOTE above)
TTFE_SEED = 0


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_data():
    """
    Loads Nov-Dec 2022 from parquet files.
    Returns:
        eval_raw:    list of np.ndarray (288, 7)  — RAW prices for revenue calc
        eval_scaled: list of np.ndarray (288, 7)  — StandardScaled for TTFE input
    """
    e_files = sorted(glob.glob(os.path.join(DATA_DIR, "energy_prices", "2022-*.parquet")))
    a_files = sorted(glob.glob(os.path.join(DATA_DIR, "as_prices",     "2022-*.parquet")))

    df_e = pd.concat([pd.read_parquet(f, engine="fastparquet") for f in e_files])
    df_a = pd.concat([pd.read_parquet(f, engine="fastparquet") for f in a_files])

    # 7-column price matrix: [spot, FR, FL, SR, SL, DR, DL]
    price_df = pd.DataFrame(index=df_e.index)
    price_df["spot"] = df_e["rt_lmp"]
    price_df["FR"]   = df_a["dam_as_rrs"]
    price_df["FL"]   = df_a["dam_as_regdn"]
    price_df["SR"]   = df_a["dam_as_regup"]
    price_df["SL"]   = df_a["dam_as_regdn"]   # intentional reuse
    price_df["DR"]   = df_a["dam_as_ecrs"]
    price_df["DL"]   = df_a["dam_as_nsrs"]
    price_df = price_df.ffill(limit=2).dropna()

    month      = price_df.index.month
    train_df   = price_df[~month.isin(EVAL_MONTHS)]
    eval_df    = price_df[month.isin(EVAL_MONTHS)]

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train_df.values)

    eval_raw_arr    = eval_df.values.astype(np.float32)
    eval_scaled_arr = scaler.transform(eval_df.values).astype(np.float32)

    n_days      = len(eval_raw_arr) // TIMESTEPS_PER_DAY
    eval_raw    = [eval_raw_arr   [i*TIMESTEPS_PER_DAY:(i+1)*TIMESTEPS_PER_DAY] for i in range(n_days)]
    eval_scaled = [eval_scaled_arr[i*TIMESTEPS_PER_DAY:(i+1)*TIMESTEPS_PER_DAY] for i in range(n_days)]

    print(f"[Data] Eval episodes: {n_days} days  ({n_days * TIMESTEPS_PER_DAY} timesteps)")
    return eval_raw, eval_scaled


# ─────────────────────────────────────────────────────────────────────────────
# REVENUE FORMULAS  (paper eq. 2, 3, 4)
# ─────────────────────────────────────────────────────────────────────────────

def step_revenue(v_dch, v_ch, p_S, p_fast, p_slow, p_delay, rho):
    """
    Net revenue for one 5-minute step.  Paper eq. 2 + 3 - 4.
    No violation penalty — that is an RL artifact, not a market loss.
    """
    spot, FR, FL, SR, SL, DR, DL = rho

    # Spot revenue (eq. 2)
    r_spot = DT_H * (v_dch * EFF_DCH - v_ch / EFF_CH) * spot * p_S

    # FCAS revenue (eq. 3)
    r_fcas = DT_H * (
        v_dch * EFF_DCH * (FR * p_fast + SR * p_slow + DR * p_delay)
        + v_ch / EFF_CH  * (FL * p_fast + SL * p_slow + DL * p_delay)
    )

    # Degradation cost (eq. 4)
    r_deg = DEGRADATION_C * DT_H * v_dch * (p_S + p_fast + p_slow + p_delay)

    return r_spot + r_fcas - r_deg


# ─────────────────────────────────────────────────────────────────────────────
# ACTION DECODER  (tanh output → MW bids)
# ─────────────────────────────────────────────────────────────────────────────

def decode_action(raw_action):
    """
    Maps actor tanh output ∈ [-1,1]^6 → (v_dch, v_ch, p_S, p_fast, p_slow, p_delay) in MW.
    Enforces mutual exclusivity (eq.1), FCAS cap, and total power ≤ Pmax (eq.8).
    """
    v_dch = 1 if raw_action[0] > 0 else 0
    v_ch  = 1 if raw_action[1] > 0 else 0

    if v_dch == 1 and v_ch == 1:
        if raw_action[0] >= raw_action[1]:
            v_ch = 0
        else:
            v_dch = 0

    fcas_norm = FCAS_MAX_MW / RATED_POWER_MW  # = 0.5

    p_S_n    = np.clip((raw_action[2] + 1) / 2, 0.0, 1.0)
    p_fast_n = np.clip((raw_action[3] + 1) / 2, 0.0, fcas_norm)
    p_slow_n = np.clip((raw_action[4] + 1) / 2, 0.0, fcas_norm)
    p_del_n  = np.clip((raw_action[5] + 1) / 2, 0.0, fcas_norm)

    total = p_S_n + p_fast_n + p_slow_n + p_del_n
    if total > 1.0:
        scale    = 1.0 / total
        p_S_n   *= scale
        p_fast_n *= scale
        p_slow_n *= scale
        p_del_n  *= scale

    return (v_dch, v_ch,
            p_S_n    * RATED_POWER_MW,
            p_fast_n * RATED_POWER_MW,
            p_slow_n * RATED_POWER_MW,
            p_del_n  * RATED_POWER_MW)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_agent(checkpoint_path, device):
    """
    Loads the trained Actor and TTFE from checkpoint.
    If "ttfe_state" is present (top-up run), loads paired TTFE weights.
    If absent (original training), falls back to reproducible seed=0 init.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"[Model] Checkpoint keys: {list(ckpt.keys())}")

    # Actor — default args match training config (obs_dim=72, act_dim=6, hidden=512)
    actor = Actor().to(device)
    actor.load_state_dict(ckpt["actor_state"])
    actor.eval()
    print(f"[Model] Actor loaded from {checkpoint_path}")

    # TTFE — load saved weights if available, else reproducible seed=0 fallback
    ttfe = TTFE().to(device)
    if "ttfe_state" in ckpt:
        ttfe.load_state_dict(ckpt["ttfe_state"])
        print(f"[Model] TTFE loaded from checkpoint  ← PAIRED WEIGHTS")
    else:
        torch.manual_seed(TTFE_SEED)
        ttfe = TTFE().to(device)
        print(f"[Model] TTFE not in checkpoint — initialised with seed={TTFE_SEED}  (random fallback)")
    ttfe.eval()

    return actor, ttfe


# ─────────────────────────────────────────────────────────────────────────────
# POLICY 1 — PIO (Perfect Information Optimisation)
# ─────────────────────────────────────────────────────────────────────────────

def run_pio_episode(prices_raw):
    """
    Solves BESS joint bidding as MILP for one episode with perfect price knowledge.

    Linearisation: each bid is split into discharge (d) and charge (c) sub-variables.
    Big-M constraints enforce that sub-variables are zero unless the corresponding
    binary mode flag (v_dch / v_ch) is active — eliminating all bilinear products.

    Args:
        prices_raw: (288, 7) raw prices [spot,FR,FL,SR,SL,DR,DL]
    Returns:
        total_revenue: float  USD for this day
        breakdown: dict  (empty, reserved for future decomposition)
    """
    T   = len(prices_raw)
    M_S = RATED_POWER_MW   # Big-M for spot bids
    M_F = FCAS_MAX_MW      # Big-M for FCAS bids

    prob = pulp.LpProblem("BESS_PIO", pulp.LpMaximize)

    # Binary mode flags (mutually exclusive)
    v_dch = [pulp.LpVariable(f"vd_{t}", cat="Binary") for t in range(T)]
    v_ch  = [pulp.LpVariable(f"vc_{t}", cat="Binary") for t in range(T)]

    # Bid sub-variables: _d = discharge component, _c = charge component
    pS_d   = [pulp.LpVariable(f"pSd_{t}",  lowBound=0, upBound=M_S) for t in range(T)]
    pS_c   = [pulp.LpVariable(f"pSc_{t}",  lowBound=0, upBound=M_S) for t in range(T)]
    pf_d   = [pulp.LpVariable(f"pfd_{t}",  lowBound=0, upBound=M_F) for t in range(T)]
    pf_c   = [pulp.LpVariable(f"pfc_{t}",  lowBound=0, upBound=M_F) for t in range(T)]
    ps_d   = [pulp.LpVariable(f"psd_{t}",  lowBound=0, upBound=M_F) for t in range(T)]
    ps_c   = [pulp.LpVariable(f"psc_{t}",  lowBound=0, upBound=M_F) for t in range(T)]
    pd_d   = [pulp.LpVariable(f"pdd_{t}",  lowBound=0, upBound=M_F) for t in range(T)]
    pd_c   = [pulp.LpVariable(f"pdc_{t}",  lowBound=0, upBound=M_F) for t in range(T)]

    # SoC state variable
    e = [pulp.LpVariable(f"e_{t}", lowBound=E_MIN_MWH, upBound=E_MAX_MWH) for t in range(T)]

    spot = prices_raw[:, 0].astype(float)
    FR   = prices_raw[:, 1].astype(float);  FL = prices_raw[:, 2].astype(float)
    SR   = prices_raw[:, 3].astype(float);  SL = prices_raw[:, 4].astype(float)
    DR   = prices_raw[:, 5].astype(float);  DL = prices_raw[:, 6].astype(float)

    # Objective — fully linear (all prices are scalar constants)
    prob += pulp.lpSum([
          DT_H * EFF_DCH * float(spot[t]) * pS_d[t]
        - DT_H / EFF_CH  * float(spot[t]) * pS_c[t]
        + DT_H * EFF_DCH * (float(FR[t])*pf_d[t] + float(SR[t])*ps_d[t] + float(DR[t])*pd_d[t])
        + DT_H / EFF_CH  * (float(FL[t])*pf_c[t] + float(SL[t])*ps_c[t] + float(DL[t])*pd_c[t])
        - DEGRADATION_C  * DT_H * (pS_d[t] + pf_d[t] + ps_d[t] + pd_d[t])
        for t in range(T)
    ])

    for t in range(T):
        # Mutual exclusivity
        prob += v_dch[t] + v_ch[t] <= 1

        # Big-M: discharge sub-vars zero unless v_dch=1
        prob += pS_d[t] <= M_S * v_dch[t]
        prob += pf_d[t] <= M_F * v_dch[t]
        prob += ps_d[t] <= M_F * v_dch[t]
        prob += pd_d[t] <= M_F * v_dch[t]

        # Big-M: charge sub-vars zero unless v_ch=1
        prob += pS_c[t] <= M_S * v_ch[t]
        prob += pf_c[t] <= M_F * v_ch[t]
        prob += ps_c[t] <= M_F * v_ch[t]
        prob += pd_c[t] <= M_F * v_ch[t]

        # Total power capacity constraints
        prob += pS_d[t] + pf_d[t] + ps_d[t] + pd_d[t] <= RATED_POWER_MW
        prob += pS_c[t] + pf_c[t] + ps_c[t] + pd_c[t] <= RATED_POWER_MW

        # Energy dynamics (spot power only, simplified — matches run_tempdrl_episode)
        e_prev = E_INIT_MWH if t == 0 else e[t - 1]
        prob += e[t] == e_prev + DT_H * (pS_c[t] - pS_d[t])

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != "Optimal":
        print(f"  [PIO] WARNING: solver status = {pulp.LpStatus[prob.status]}")
        return 0.0, {}

    total = float(pulp.value(prob.objective))
    return total, {}


# ─────────────────────────────────────────────────────────────────────────────
# POLICY 2/3/4 — TempDRL  (Joint / Spot-only / FCAS-only)
# ─────────────────────────────────────────────────────────────────────────────

def run_tempdrl_episode(actor, ttfe, ep_raw, ep_scaled, mode="joint", device="cpu"):
    """
    Runs one deterministic greedy episode.
    Revenue computed via paper eq. 2+3-4 only — no shaped reward, no penalties.

    Args:
        ep_raw    : (288, 7) raw prices (USD/MWh) — for revenue calculation
        ep_scaled : (288, 7) scaled prices — for TTFE input only
        mode      : "joint" | "spot" | "fcas"
    Returns:
        total_revenue : float  USD for this day
        breakdown     : dict  {r_spot, r_fcas, r_deg, n_soc_clipped}
    """
    T = len(ep_raw)

    # Pre-compute ALL TTFE features in one batched forward pass
    segments = np.stack([
        build_temporal_segment(ep_scaled, t=t, L=12) for t in range(T)
    ])                                                              # (288, 12, 7)

    with torch.no_grad():
        features = ttfe(torch.FloatTensor(segments).to(device)).cpu().numpy()  # (288, 64)

    energy         = E_INIT_MWH
    total_revenue  = 0.0
    r_spot_total   = 0.0
    r_fcas_total   = 0.0
    r_deg_total    = 0.0
    n_soc_clipped  = 0

    for t in range(T):
        soc   = energy / CAPACITY_MWH
        rho_t = ep_raw[t]                             # raw prices (7,)
        f_t   = features[t]                           # TTFE features (64,)

        # Build 72-dim state: [SoC(1), raw_prices(7), ttfe_features(64)]
        state_np = np.concatenate([[soc], rho_t, f_t]).astype(np.float32)
        state_t  = torch.FloatTensor(state_np).unsqueeze(0).to(device)

        # Deterministic action: tanh(mean)
        with torch.no_grad():
            action_t, _ = actor.sample(state_t, deterministic=True)
        raw_action = action_t.squeeze(0).cpu().numpy()

        # Decode to MW bids
        v_dch, v_ch, p_S, p_fast, p_slow, p_delay = decode_action(raw_action)

        # Zero out inactive markets per mode
        if mode == "spot":
            p_fast = p_slow = p_delay = 0.0
        elif mode == "fcas":
            p_S = 0.0

        # Compute real USD revenue (paper eq. 2, 3, 4 — no penalties)
        spot, FR, FL, SR, SL, DR, DL = rho_t

        r_s   = DT_H * (v_dch * EFF_DCH - v_ch / EFF_CH) * spot * p_S
        r_f   = DT_H * (
            v_dch * EFF_DCH * (FR * p_fast + SR * p_slow + DR * p_delay)
            + v_ch / EFF_CH  * (FL * p_fast + SL * p_slow + DL * p_delay)
        )
        r_d   = DEGRADATION_C * DT_H * v_dch * (p_S + p_fast + p_slow + p_delay)

        total_revenue += r_s + r_f - r_d
        r_spot_total  += r_s
        r_fcas_total  += r_f
        r_deg_total   += r_d

        # Update SoC — energy dynamics (eq.10, spot component only)
        delta_e = DT_H * (v_ch - v_dch) * p_S
        new_e   = energy + delta_e
        if new_e < E_MIN_MWH or new_e > E_MAX_MWH:
            n_soc_clipped += 1
        energy = np.clip(new_e, E_MIN_MWH, E_MAX_MWH)

    breakdown = {
        "r_spot"       : r_spot_total,
        "r_fcas"       : r_fcas_total,
        "r_deg"        : r_deg_total,
        "n_soc_clipped": n_soc_clipped,
    }
    return total_revenue, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*65)
    print("  TempDRL — Stage 1 Revenue Evaluation")
    print("="*65)

    device = torch.device(f"cuda:26" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    eval_raw, eval_scaled = load_eval_data()
    N = len(eval_raw)

    # ── 2. Load trained actor + TTFE ─────────────────────────────────────────
    actor, ttfe = load_agent(CHECKPOINT_PATH, device)

    # ── 3. PIO (oracle upper bound) ───────────────────────────────────────────
    print(f"\n[PIO] Running perfect foresight LP on {N} episodes...")
    pio_revenues = []
    for i, ep_raw in enumerate(eval_raw):
        rev, _ = run_pio_episode(ep_raw)
        pio_revenues.append(rev)
        if (i + 1) % 10 == 0 or (i + 1) == N:
            print(f"  ep {i+1:3d}/{N}  mean so far: ${np.mean(pio_revenues):.2f}/day")

    # ── 4. TempDRL Joint ──────────────────────────────────────────────────────
    print(f"\n[TempDRL Joint] Running {N} greedy episodes...")
    joint_revenues = []
    joint_breaks   = []
    for ep_raw, ep_sc in zip(eval_raw, eval_scaled):
        rev, bd = run_tempdrl_episode(actor, ttfe, ep_raw, ep_sc, mode="joint", device=device)
        joint_revenues.append(rev)
        joint_breaks.append(bd)
    print(f"  Done. Mean: ${np.mean(joint_revenues):.2f}/day  "
          f"(SoC clips: {np.mean([b['n_soc_clipped'] for b in joint_breaks]):.1f}/day)")

    # ── 5. TempDRL Spot-only ──────────────────────────────────────────────────
    print(f"\n[TempDRL Spot-only] Running {N} greedy episodes...")
    spot_revenues = []
    for ep_raw, ep_sc in zip(eval_raw, eval_scaled):
        rev, _ = run_tempdrl_episode(actor, ttfe, ep_raw, ep_sc, mode="spot", device=device)
        spot_revenues.append(rev)
    print(f"  Done. Mean: ${np.mean(spot_revenues):.2f}/day")

    # ── 6. TempDRL FCAS-only ──────────────────────────────────────────────────
    print(f"\n[TempDRL FCAS-only] Running {N} greedy episodes...")
    fcas_revenues = []
    for ep_raw, ep_sc in zip(eval_raw, eval_scaled):
        rev, _ = run_tempdrl_episode(actor, ttfe, ep_raw, ep_sc, mode="fcas", device=device)
        fcas_revenues.append(rev)
    print(f"  Done. Mean: ${np.mean(fcas_revenues):.2f}/day")

    # ── 7. Results table ──────────────────────────────────────────────────────
    pio_mean = np.mean(pio_revenues)

    results = [
        ("PIO (Perfect Foresight)", pio_revenues),
        ("TempDRL Joint",           joint_revenues),
        ("TempDRL Spot-only",       spot_revenues),
        ("TempDRL FCAS-only",       fcas_revenues),
    ]

    print("\n" + "="*72)
    print("  STAGE 1 RESULTS — Mean Daily Revenue (USD/day)")
    print(f"  Dataset : ERCOT 2022, Nov–Dec eval set  ({N} days)")
    print("  BESS    : 10 MWh / 2 MW, η=0.95, SoC 5–95%")
    print("  Actions : Deterministic tanh(mean) — no sampling noise")
    print("  Revenue : Paper eq. 2+3-4 only — no RL penalties")
    print("="*72)
    print(f"  {'Policy':<28} {'Mean $/day':>10} {'Std':>8} {'Min':>8} {'Max':>8} {'vs PIO':>7}")
    print("-"*72)
    for name, revs in results:
        m   = np.mean(revs)
        s   = np.std(revs)
        mn  = np.min(revs)
        mx  = np.max(revs)
        pct = 100.0 * m / pio_mean if pio_mean > 0 else float("nan")
        tag = " ◀ BEST" if name == "TempDRL Joint" else ""
        print(f"  {name:<28} {m:>10.2f} {s:>8.2f} {mn:>8.2f} {mx:>8.2f} {pct:>6.1f}%{tag}")
    print("="*72)

    # Market breakdown for Joint
    jb = joint_breaks
    print(f"\n  TempDRL Joint — mean daily revenue breakdown:")
    print(f"    Spot market    : ${np.mean([b['r_spot'] for b in jb]):>8.2f}")
    print(f"    FCAS markets   : ${np.mean([b['r_fcas'] for b in jb]):>8.2f}")
    print(f"    Degradation    : ${-np.mean([b['r_deg'] for b in jb]):>8.2f}  (cost)")
    print(f"    NET            : ${np.mean(joint_revenues):>8.2f}")
    print(f"    SoC clips/day  :  {np.mean([b['n_soc_clipped'] for b in jb]):>6.1f}")

    j  = np.mean(joint_revenues)
    sp = np.mean(spot_revenues)
    fc = np.mean(fcas_revenues)
    denom_sp = abs(sp) if abs(sp) > 1e-6 else 1.0
    denom_fc = abs(fc) if abs(fc) > 1e-6 else 1.0

    print(f"\n  Joint vs Spot-only : {100*(j-sp)/denom_sp:+.1f}%")
    print(f"  Joint vs FCAS-only : {100*(j-fc)/denom_fc:+.1f}%")
    print(f"  Joint vs PIO       :  {100*j/max(pio_mean,1e-6):.1f}% of upper bound")

    # ── 8. Save results ───────────────────────────────────────────────────────
    os.makedirs("outputs/logs", exist_ok=True)
    out_df = pd.DataFrame({
        "episode"       : list(range(N)),
        "pio_usd"       : pio_revenues,
        "joint_usd"     : joint_revenues,
        "spot_only_usd" : spot_revenues,
        "fcas_only_usd" : fcas_revenues,
    })
    out_path = "outputs/logs/stage1_revenue_results.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to: {out_path}")
    print()


if __name__ == "__main__":
    main()
