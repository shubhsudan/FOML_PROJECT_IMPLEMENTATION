"""
environment_stage2.py — ERCOT RTC+B BESS joint-market bidding environment (Stage 2).

Market structure: ERCOT post-RTC+B (Dec 5, 2025+)
  - 6 AS markets: RegUp, RegDn, RRS, ECRS (NEW), NSRS
  - RT clearing prices (rt_mcpc_*) used for AS revenue
  - ECRS: 2-hour duration, discharge direction
  - No simultaneous RegUp + RegDn
  - Hourly SOC floor/ceiling rule (extended for ECRS 2-hr)

Action space (9-dim, tanh output from actor):
  [v_dch, v_ch, a_spot_dch, a_spot_ch, a_regup, a_regdn, a_rrs, a_ecrs, a_nsrs]
   idx: 0     1       2          3         4        5       6      7       8

State space (78-dim):
  [TTFE(64), SysCond(7), CyclicalTime(6), SoC(1)]

Price vector (12-dim, matches PRICE_COLS_12):
  [rt_lmp(0), rt_mcpc_regup(1), rt_mcpc_regdn(2), rt_mcpc_rrs(3),
   rt_mcpc_ecrs(4), rt_mcpc_nsrs(5),
   dam_spp(6), dam_as_regup(7), dam_as_regdn(8), dam_as_rrs(9),
   dam_as_ecrs(10), dam_as_nsrs(11)]
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    CAPACITY_MWH, RATED_POWER_MW, FCAS_MAX_MW,
    EFF_CH, EFF_DCH, E_MIN_MWH, E_MAX_MWH, E_INIT_MWH,
    DEGRADATION_C, DT_H, TIMESTEPS_PER_DAY,
)
from config_stage2 import (
    NUM_PRICE_DIMS, NUM_SYSCOND_DIMS, STATE_DIM_S2, ACTION_DIM_S2,
    AS_DURATION_H_S2, TAU_EMA_S2, BETA_S_S2, PENALTY_VIO_S2,
    IDX_RT_LMP, IDX_RT_MCPC_REGUP, IDX_RT_MCPC_REGDN,
    IDX_RT_MCPC_RRS, IDX_RT_MCPC_ECRS, IDX_RT_MCPC_NSRS, IDX_DAM_SPP,
)


# ── Action Decoder (9-dim) ────────────────────────────────────────────────────

def decode_action_s2(raw_action: np.ndarray) -> tuple:
    """
    Maps actor tanh output [-1,1]^9 → physical ERCOT RTC+B-compliant MW bids.

    Input layout (9-dim):
        0: v_dch        discharge mode flag
        1: v_ch         charge mode flag
        2: a_spot_dch   spot discharge power
        3: a_spot_ch    spot charge power
        4: a_regup      RegUp bid   (discharge-direction)
        5: a_regdn      RegDn bid   (charge-direction)
        6: a_rrs        RRS bid     (discharge-direction)
        7: a_ecrs       ECRS bid    (discharge-direction, NEW)
        8: a_nsrs       NSRS bid    (discharge-direction)

    Returns:
        (v_dch, v_ch, p_spot_dch, p_spot_ch,
         p_regup, p_regdn, p_rrs, p_ecrs, p_nsrs)  all in MW
    """
    v_dch = 1 if raw_action[0] > 0 else 0
    v_ch  = 1 if raw_action[1] > 0 else 0

    # Mutual exclusivity
    if v_dch == 1 and v_ch == 1:
        if raw_action[0] >= raw_action[1]:
            v_ch = 0
        else:
            v_dch = 0

    def scale(x, limit):
        return float(np.clip((x + 1.0) / 2.0, 0.0, 1.0) * limit)

    p_spot_dch = scale(raw_action[2], RATED_POWER_MW) if v_dch else 0.0
    p_spot_ch  = scale(raw_action[3], RATED_POWER_MW) if v_ch  else 0.0
    p_regup    = scale(raw_action[4], FCAS_MAX_MW)    if v_dch else 0.0
    p_rrs      = scale(raw_action[6], FCAS_MAX_MW)    if v_dch else 0.0
    p_ecrs     = scale(raw_action[7], FCAS_MAX_MW)    if v_dch else 0.0
    p_nsrs     = scale(raw_action[8], FCAS_MAX_MW)    if v_dch else 0.0
    p_regdn    = scale(raw_action[5], FCAS_MAX_MW)    if v_ch  else 0.0

    # ERCOT Rule: No simultaneous RegUp + RegDn
    if p_regup > 0 and p_regdn > 0:
        if p_regup >= p_regdn:
            p_regdn = 0.0
        else:
            p_regup = 0.0

    # Total discharge power <= Pmax
    dch_total = p_spot_dch + p_regup + p_rrs + p_ecrs + p_nsrs
    if dch_total > RATED_POWER_MW:
        s = RATED_POWER_MW / dch_total
        p_spot_dch *= s; p_regup *= s; p_rrs *= s; p_ecrs *= s; p_nsrs *= s

    # Total charge power <= Pmax
    ch_total = p_spot_ch + p_regdn
    if ch_total > RATED_POWER_MW:
        s = RATED_POWER_MW / ch_total
        p_spot_ch *= s; p_regdn *= s

    return (v_dch, v_ch, p_spot_dch, p_spot_ch,
            p_regup, p_regdn, p_rrs, p_ecrs, p_nsrs)


# ── Revenue Formulas (RTC+B — uses RT clearing prices) ───────────────────────

def compute_step_revenue_s2(
    v_dch, v_ch,
    p_spot_dch, p_spot_ch,
    p_regup, p_regdn, p_rrs, p_ecrs, p_nsrs,
    prices_12: np.ndarray,
) -> dict:
    """
    Real USD revenue for one 5-minute ERCOT RTC+B dispatch interval.

    prices_12: 12-dim vector [rt_lmp, rt_mcpc_regup, rt_mcpc_regdn, rt_mcpc_rrs,
                               rt_mcpc_ecrs, rt_mcpc_nsrs, dam_spp, dam_as_*]
    AS revenue: rt_mcpc_* prices × MW_bid × dt_h (RT clearing, not DAM)
    Spot revenue: rt_lmp × power × efficiency × dt_h
    """
    rt_lmp       = float(prices_12[IDX_RT_LMP])
    px_regup     = float(prices_12[IDX_RT_MCPC_REGUP])
    px_regdn     = float(prices_12[IDX_RT_MCPC_REGDN])
    px_rrs       = float(prices_12[IDX_RT_MCPC_RRS])
    px_ecrs      = float(prices_12[IDX_RT_MCPC_ECRS])
    px_nsrs      = float(prices_12[IDX_RT_MCPC_NSRS])

    r_spot  = DT_H * (EFF_DCH * rt_lmp * p_spot_dch
                      - (1.0 / EFF_CH) * rt_lmp * p_spot_ch)
    r_regup = px_regup * p_regup * DT_H
    r_regdn = px_regdn * p_regdn * DT_H
    r_rrs   = px_rrs   * p_rrs   * DT_H
    r_ecrs  = px_ecrs  * p_ecrs  * DT_H
    r_nsrs  = px_nsrs  * p_nsrs  * DT_H
    r_as    = r_regup + r_regdn + r_rrs + r_ecrs + r_nsrs

    total_dch = p_spot_dch + p_regup + p_rrs + p_ecrs + p_nsrs
    r_deg     = DEGRADATION_C * DT_H * v_dch * total_dch

    total = r_spot + r_as - r_deg
    return {
        "total":  total,
        "r_spot": r_spot,
        "r_as":   r_as,
        "r_regup": r_regup, "r_regdn": r_regdn,
        "r_rrs":   r_rrs,   "r_ecrs":  r_ecrs,
        "r_nsrs":  r_nsrs,  "r_deg":   r_deg,
    }


# ── ERCOT Hourly SOC Rule (RTC+B — includes ECRS 2-hr) ───────────────────────

def hourly_soc_bounds_s2(
    timestep: int,
    p_regup:  float,
    p_regdn:  float,
    p_rrs:    float,
    p_ecrs:   float,
    p_nsrs:   float,
) -> tuple:
    """
    Returns (e_min_eff, e_max_eff) enforcing ERCOT hourly SOC rule.
    Applied at hour start (every 12 timesteps).
    ECRS: 2-hour duration (must hold 2×ECRS MWh headroom above floor).
    """
    if timestep % 12 != 0:
        return E_MIN_MWH, E_MAX_MWH

    e_reserved = (
        p_regup * AS_DURATION_H_S2["RegUp"]
        + p_rrs  * AS_DURATION_H_S2["RRS"]
        + p_ecrs * AS_DURATION_H_S2["ECRS"]
        + p_nsrs * AS_DURATION_H_S2["NSRS"]
    )
    e_min_eff = min(E_MAX_MWH, E_MIN_MWH + e_reserved)
    e_headroom = p_regdn * AS_DURATION_H_S2["RegDn"]
    e_max_eff  = max(E_MIN_MWH, E_MAX_MWH - e_headroom)

    return e_min_eff, e_max_eff


# ── Shaped RL Reward (training only) ─────────────────────────────────────────

def compute_shaped_reward_s2(
    v_dch, v_ch,
    p_spot_dch, p_spot_ch,
    p_regup, p_regdn, p_rrs, p_ecrs, p_nsrs,
    prices_12:  np.ndarray,
    ema_spot:   float,
    violated:   bool,
) -> float:
    """Shaped RL reward for Stage 2 training."""
    rev    = compute_step_revenue_s2(
        v_dch, v_ch, p_spot_dch, p_spot_ch,
        p_regup, p_regdn, p_rrs, p_ecrs, p_nsrs, prices_12,
    )
    rt_lmp = float(prices_12[IDX_RT_LMP])
    I_ch   = 1 if rt_lmp < ema_spot else 0
    I_dch  = 1 if rt_lmp > ema_spot else 0
    r_shape = (BETA_S_S2 * (p_spot_dch + p_spot_ch) * abs(rt_lmp - ema_spot)
               * (I_dch * v_dch * EFF_DCH + I_ch * v_ch / EFF_CH))
    reward = rev["total"] + r_shape
    if violated:
        reward -= PENALTY_VIO_S2
    return float(reward)


# ── Main Environment Class ────────────────────────────────────────────────────

class BESSEnvironment_S2:
    """
    ERCOT RTC+B BESS joint-bidding environment.
    One episode = one trading day = 288 timesteps × 5 minutes.

    Observation (78-dim): [TTFE(64), SysCond(7), Time(6), SoC(1)]
    Action (9-dim): [v_dch, v_ch, spot_dch, spot_ch, regup, regdn, rrs, ecrs, nsrs]

    mode: "joint" | "spot" | "as"
    """

    def __init__(self, mode: str = "joint"):
        assert mode in ("joint", "spot", "as")
        self.mode     = mode
        self.t        = 0
        self.energy   = E_INIT_MWH
        self.ema_spot = 0.0
        self.done     = False
        self._ep_prices  = None   # (288, 12) raw prices
        self._ep_syscond = None   # (288, 7)  normalized syscond
        self._ep_feat    = None   # (288, 64) TTFE features
        self._ep_dow     = 0      # day of week
        self._ep_month   = 1      # month

    def reset(
        self,
        prices_raw:   np.ndarray,
        syscond_norm: np.ndarray,
        ttfe_features: np.ndarray,
        day_of_week:  int = 0,
        month:        int = 1,
    ) -> np.ndarray:
        """
        Reset for a new episode.

        Args:
            prices_raw:    (288, 12) raw prices
            syscond_norm:  (288, 7)  normalized system conditions
            ttfe_features: (288, 64) pre-computed TTFE features
            day_of_week:   0=Mon…6=Sun
            month:         1–12
        """
        assert prices_raw.shape   == (TIMESTEPS_PER_DAY, NUM_PRICE_DIMS)
        assert syscond_norm.shape == (TIMESTEPS_PER_DAY, NUM_SYSCOND_DIMS)
        assert ttfe_features.shape == (TIMESTEPS_PER_DAY, 64)

        self._ep_prices  = prices_raw
        self._ep_syscond = syscond_norm
        self._ep_feat    = ttfe_features
        self._ep_dow     = day_of_week
        self._ep_month   = month
        self.t           = 0
        self.energy      = E_INIT_MWH
        self.ema_spot    = float(prices_raw[0, IDX_RT_LMP])
        self.done        = False
        return self._make_obs()

    def _make_obs(self) -> np.ndarray:
        from data_loader_stage2 import build_state_78, build_time_6
        time_6 = build_time_6(self.t, self._ep_dow, self._ep_month)
        return build_state_78(
            soc          = self.energy / CAPACITY_MWH,
            syscond_7    = self._ep_syscond[self.t],
            time_6       = time_6,
            ttfe_feat_64 = self._ep_feat[self.t],
        )

    def step(self, raw_action: np.ndarray) -> tuple:
        """
        Execute one 5-minute dispatch step.

        Args:
            raw_action: (9,) tanh outputs from actor

        Returns:
            (next_obs, shaped_reward, done, info)
        """
        assert not self.done

        prices = self._ep_prices[self.t]
        rt_lmp = float(prices[IDX_RT_LMP])
        self.ema_spot = TAU_EMA_S2 * self.ema_spot + (1.0 - TAU_EMA_S2) * rt_lmp

        (v_dch, v_ch,
         p_spot_dch, p_spot_ch,
         p_regup, p_regdn, p_rrs, p_ecrs, p_nsrs) = decode_action_s2(raw_action)

        if self.mode == "spot":
            p_regup = p_regdn = p_rrs = p_ecrs = p_nsrs = 0.0
        elif self.mode == "as":
            p_spot_dch = p_spot_ch = 0.0

        e_min_eff, e_max_eff = hourly_soc_bounds_s2(
            self.t, p_regup, p_regdn, p_rrs, p_ecrs, p_nsrs
        )

        delta_e    = DT_H * (EFF_CH * p_spot_ch - (1.0 / EFF_DCH) * p_spot_dch)
        new_energy = self.energy + delta_e
        violated   = (new_energy < e_min_eff) or (new_energy > e_max_eff)

        if violated:
            new_energy = np.clip(new_energy, e_min_eff, e_max_eff)
            p_spot_dch = p_spot_ch = 0.0
            p_regup = p_regdn = p_rrs = p_ecrs = p_nsrs = 0.0
            v_dch = v_ch = 0

        self.energy = float(new_energy)

        reward = compute_shaped_reward_s2(
            v_dch, v_ch, p_spot_dch, p_spot_ch,
            p_regup, p_regdn, p_rrs, p_ecrs, p_nsrs,
            prices, self.ema_spot, violated,
        )

        rev = compute_step_revenue_s2(
            v_dch, v_ch, p_spot_dch, p_spot_ch,
            p_regup, p_regdn, p_rrs, p_ecrs, p_nsrs, prices,
        )

        info = {
            "usd_revenue": rev["total"],
            "r_spot":  rev["r_spot"],  "r_as":    rev["r_as"],
            "r_regup": rev["r_regup"], "r_regdn": rev["r_regdn"],
            "r_rrs":   rev["r_rrs"],   "r_ecrs":  rev["r_ecrs"],
            "r_nsrs":  rev["r_nsrs"],  "r_deg":   rev["r_deg"],
            "soc":     self.energy / CAPACITY_MWH,
            "violated": violated,
            "v_dch": v_dch, "v_ch": v_ch,
        }

        self.t   += 1
        self.done = (self.t >= TIMESTEPS_PER_DAY)
        next_obs  = (self._make_obs() if not self.done
                     else np.zeros(STATE_DIM_S2, dtype=np.float32))
        return next_obs, reward, self.done, info
