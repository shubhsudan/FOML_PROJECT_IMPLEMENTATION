"""
environment.py — ERCOT-correct BESS joint-market bidding environment.

Market structure: ERCOT 2022
  - 5 markets: spot energy + RegUp + RegDn + RRS + NSRS
  - ECRS excluded (not active until June 2023; all-zero in 2022 data)
  - AS revenue = capacity payment: price ($/MW) × MW_bid × dt_h
  - Hourly SOC floor/ceiling rule (ERCOT BPM Dec 2022 / NPRR 1186)
  - No simultaneous RegUp + RegDn
  - Duration-based AS qualification (5-hr BESS qualifies for full rated power)

Action space (8-dim, tanh output from actor):
  [v_dch, v_ch, a_spot_dch, a_spot_ch, a_regup, a_regdn, a_rrs, a_nsrs]

State space (72-dim):
  [SoC(1), spot(1), RegUp(1), RegDn(1), RRS(1), NSRS(1), TTFE(64), hour_sin(1), hour_cos(1)]
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    CAPACITY_MWH, RATED_POWER_MW, FCAS_MAX_MW,
    EFF_CH, EFF_DCH, E_MIN_MWH, E_MAX_MWH, E_INIT_MWH,
    DEGRADATION_C, DT_H, TIMESTEPS_PER_DAY,
    TAU_EMA, BETA_S, PENALTY_VIOLATE,
    NUM_MARKETS, STATE_DIM, ACTION_DIM,
    AS_DURATION_H,
)


# ── Action Decoder ────────────────────────────────────────────────────────────

def decode_action(raw_action: np.ndarray) -> tuple:
    """
    Maps actor tanh output [-1,1]^8 → physical ERCOT-compliant MW bids.

    Input layout:
        0: v_dch      discharge mode flag
        1: v_ch       charge mode flag
        2: a_spot_dch spot discharge power
        3: a_spot_ch  spot charge power
        4: a_regup    RegUp bid  (discharge-direction AS)
        5: a_regdn    RegDn bid  (charge-direction AS)
        6: a_rrs      RRS bid   (discharge-direction AS)
        7: a_nsrs     NSRS bid  (discharge-direction AS)

    Returns:
        (v_dch, v_ch, p_spot_dch, p_spot_ch, p_regup, p_regdn, p_rrs, p_nsrs)
        all in MW
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
    p_nsrs     = scale(raw_action[7], FCAS_MAX_MW)    if v_dch else 0.0
    p_regdn    = scale(raw_action[5], FCAS_MAX_MW)    if v_ch  else 0.0

    # ERCOT Rule: No simultaneous RegUp + RegDn
    if p_regup > 0 and p_regdn > 0:
        if p_regup >= p_regdn:
            p_regdn = 0.0
        else:
            p_regup = 0.0

    # Total discharge power <= Pmax
    dch_total = p_spot_dch + p_regup + p_rrs + p_nsrs
    if dch_total > RATED_POWER_MW:
        s = RATED_POWER_MW / dch_total
        p_spot_dch *= s; p_regup *= s; p_rrs *= s; p_nsrs *= s

    # Total charge power <= Pmax
    ch_total = p_spot_ch + p_regdn
    if ch_total > RATED_POWER_MW:
        s = RATED_POWER_MW / ch_total
        p_spot_ch *= s; p_regdn *= s

    return (v_dch, v_ch, p_spot_dch, p_spot_ch, p_regup, p_regdn, p_rrs, p_nsrs)


# ── Revenue Formulas (ERCOT-correct) ─────────────────────────────────────────

def compute_step_revenue(v_dch, v_ch,
                         p_spot_dch, p_spot_ch,
                         p_regup, p_regdn, p_rrs, p_nsrs,
                         prices_5: np.ndarray) -> dict:
    """
    Real USD revenue for one 5-minute ERCOT dispatch interval.

    prices_5: [rt_lmp, regup_price, regdn_price, rrs_price, nsrs_price]
    AS prices are $/MW capacity prices (paid for reservation, not delivery).
    Revenue = price × MW_bid × dt_h
    """
    rt_lmp, px_regup, px_regdn, px_rrs, px_nsrs = prices_5

    r_spot = DT_H * (EFF_DCH * rt_lmp * p_spot_dch
                     - (1.0 / EFF_CH) * rt_lmp * p_spot_ch)
    r_regup = px_regup * p_regup * DT_H
    r_regdn = px_regdn * p_regdn * DT_H
    r_rrs   = px_rrs   * p_rrs   * DT_H
    r_nsrs  = px_nsrs  * p_nsrs  * DT_H
    r_as    = r_regup + r_regdn + r_rrs + r_nsrs

    total_dch = p_spot_dch + p_regup + p_rrs + p_nsrs
    r_deg = DEGRADATION_C * DT_H * v_dch * total_dch

    total = r_spot + r_as - r_deg
    return {
        "total": total, "r_spot": r_spot, "r_as": r_as,
        "r_regup": r_regup, "r_regdn": r_regdn,
        "r_rrs": r_rrs, "r_nsrs": r_nsrs, "r_deg": r_deg,
    }


# ── ERCOT Hourly SOC Rule ─────────────────────────────────────────────────────

def get_effective_soc_bounds(timestep: int,
                              p_regup: float, p_regdn: float,
                              p_rrs: float, p_nsrs: float) -> tuple:
    """
    Returns (e_min_eff, e_max_eff) enforcing ERCOT hourly SOC rule.
    Applied at hour start (every 12 timesteps). Standard bounds otherwise.
    """
    if timestep % 12 != 0:
        return E_MIN_MWH, E_MAX_MWH

    e_reserved = (p_regup * AS_DURATION_H["RegUp"]
                 + p_rrs   * AS_DURATION_H["RRS"]
                 + p_nsrs  * AS_DURATION_H["NSRS"])
    e_min_eff  = min(E_MAX_MWH, E_MIN_MWH + e_reserved)
    e_headroom = p_regdn * AS_DURATION_H["RegDn"]
    e_max_eff  = max(E_MIN_MWH, E_MAX_MWH - e_headroom)
    return e_min_eff, e_max_eff


# ── Shaped RL Reward (training only) ─────────────────────────────────────────

def compute_shaped_reward(v_dch, v_ch,
                          p_spot_dch, p_spot_ch,
                          p_regup, p_regdn, p_rrs, p_nsrs,
                          prices_5: np.ndarray,
                          ema_spot: float,
                          violated: bool) -> float:
    """Shaped RL reward for training. NOT used for evaluation."""
    rev    = compute_step_revenue(v_dch, v_ch, p_spot_dch, p_spot_ch,
                                  p_regup, p_regdn, p_rrs, p_nsrs, prices_5)
    rt_lmp = float(prices_5[0])
    I_ch   = 1 if rt_lmp < ema_spot else 0
    I_dch  = 1 if rt_lmp > ema_spot else 0
    r_shape = (BETA_S * (p_spot_dch + p_spot_ch) * abs(rt_lmp - ema_spot)
               * (I_dch * v_dch * EFF_DCH + I_ch * v_ch / EFF_CH))
    reward = rev["total"] + r_shape
    if violated:
        reward -= PENALTY_VIOLATE
    return float(reward)


# ── Main Environment Class ────────────────────────────────────────────────────

class BESSEnvironment:
    """
    ERCOT-correct BESS joint-bidding environment.
    One episode = one trading day = 288 timesteps × 5 minutes.
    mode: "joint" | "spot" | "as"
    """

    def __init__(self, mode: str = "joint"):
        assert mode in ("joint", "spot", "as")
        self.mode     = mode
        self.t        = 0
        self.energy   = E_INIT_MWH
        self.ema_spot = 0.0
        self.done     = False
        self._ep_raw  = None
        self._ep_feat = None

    def reset(self, price_episode_raw: np.ndarray,
              features: np.ndarray) -> np.ndarray:
        """
        Reset for a new episode.
        price_episode_raw: (288, 5) raw prices [spot, RegUp, RegDn, RRS, NSRS]
        features:          (288, 64) pre-computed TTFE features
        """
        assert price_episode_raw.shape == (TIMESTEPS_PER_DAY, NUM_MARKETS)
        assert features.shape          == (TIMESTEPS_PER_DAY, 64)
        self._ep_raw  = price_episode_raw
        self._ep_feat = features
        self.t        = 0
        self.energy   = E_INIT_MWH
        self.ema_spot = float(price_episode_raw[0, 0])
        self.done     = False
        return self._make_obs()

    def _make_obs(self) -> np.ndarray:
        from data_loader import build_state
        return build_state(
            self.energy / CAPACITY_MWH,
            self._ep_raw[self.t],
            self._ep_feat[self.t],
            self.t,
        )

    def step(self, raw_action: np.ndarray) -> tuple:
        """
        Execute one 5-minute dispatch step.
        raw_action: (8,) tanh outputs from actor
        Returns: (next_obs, shaped_reward, done, info)
        """
        assert not self.done

        prices = self._ep_raw[self.t]
        rt_lmp = float(prices[0])
        self.ema_spot = TAU_EMA * self.ema_spot + (1.0 - TAU_EMA) * rt_lmp

        (v_dch, v_ch,
         p_spot_dch, p_spot_ch,
         p_regup, p_regdn,
         p_rrs, p_nsrs) = decode_action(raw_action)

        if self.mode == "spot":
            p_regup = p_regdn = p_rrs = p_nsrs = 0.0
        elif self.mode == "as":
            p_spot_dch = p_spot_ch = 0.0

        e_min_eff, e_max_eff = get_effective_soc_bounds(
            self.t, p_regup, p_regdn, p_rrs, p_nsrs
        )

        delta_e    = DT_H * (EFF_CH * p_spot_ch - (1.0 / EFF_DCH) * p_spot_dch)
        new_energy = self.energy + delta_e
        violated   = (new_energy < e_min_eff) or (new_energy > e_max_eff)

        if violated:
            new_energy = np.clip(new_energy, e_min_eff, e_max_eff)
            p_spot_dch = p_spot_ch = 0.0
            p_regup = p_regdn = p_rrs = p_nsrs = 0.0
            v_dch = v_ch = 0

        self.energy = float(new_energy)

        reward = compute_shaped_reward(
            v_dch, v_ch, p_spot_dch, p_spot_ch,
            p_regup, p_regdn, p_rrs, p_nsrs,
            prices, self.ema_spot, violated,
        )

        rev = compute_step_revenue(
            v_dch, v_ch, p_spot_dch, p_spot_ch,
            p_regup, p_regdn, p_rrs, p_nsrs, prices,
        )

        info = {
            "usd_revenue": rev["total"],
            "r_spot":  rev["r_spot"],  "r_as":    rev["r_as"],
            "r_regup": rev["r_regup"], "r_regdn": rev["r_regdn"],
            "r_rrs":   rev["r_rrs"],   "r_nsrs":  rev["r_nsrs"],
            "r_deg":   rev["r_deg"],
            "soc":     self.energy / CAPACITY_MWH,
            "violated": violated,
            "v_dch": v_dch, "v_ch": v_ch,
        }

        self.t   += 1
        self.done = (self.t >= TIMESTEPS_PER_DAY)
        next_obs  = self._make_obs() if not self.done else np.zeros(STATE_DIM, dtype=np.float32)
        return next_obs, reward, self.done, info
