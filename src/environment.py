"""
environment.py — BESS joint-bidding MDP environment.

Implements the MDP described in Section IV-B of the paper:
  - State space S  (eq. 22):  [SoC_{t-1}, ρ_{t-1}, f_{t-1}]
  - Action space A (eq. 23):  [v^dch, v^ch, a^S, a^fast, a^slow, a^delay]
  - Reward R        (eq. 30):  r_t = r^S_t + r^fast_t + r^slow_t + r^delay_t
  - Constraints     (eq. 1-11): power limits, SoC limits

NOTE: The TTFE feature vector f_{t-1} is passed IN from outside;
      this environment does NOT instantiate TTFE — that is the caller's job.
      This keeps environment and model cleanly decoupled.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    BESS_CAPACITY_MWH, BESS_RATED_POWER_MW, BESS_FCAS_MAX_MW,
    BESS_EFF_CH, BESS_EFF_DCH, BESS_E_MIN_MWH, BESS_E_MAX_MWH,
    BESS_DEGRADATION_C, DISPATCH_INTERVAL_MIN, TAU_EMA, BETA_S,
    PENALTY_VIOLATE, NUM_MARKETS, EMBED_DIM
)


@dataclass
class BESSParams:
    """Physical parameters of the BESS."""
    capacity_mwh    : float = BESS_CAPACITY_MWH
    rated_power_mw  : float = BESS_RATED_POWER_MW
    fcas_max_mw     : float = BESS_FCAS_MAX_MW
    eff_ch          : float = BESS_EFF_CH
    eff_dch         : float = BESS_EFF_DCH
    e_min_mwh       : float = BESS_E_MIN_MWH
    e_max_mwh       : float = BESS_E_MAX_MWH
    degradation_c   : float = BESS_DEGRADATION_C
    dt_h            : float = DISPATCH_INTERVAL_MIN / 60.0  # hours per step


class BESSEnvironment:
    """
    Joint-market BESS bidding environment.

    Observation: np.ndarray  shape (1 + 7 + EMBED_DIM,)
        [SoC, spot, FR, FL, SR, SL, DR, DL, f_1, ..., f_{F'}]

    Action: np.ndarray  shape (6,) — raw network output in [-1, 1]
        Network output is mapped to physical actions:
        [v_dch ∈ {0,1}, v_ch ∈ {0,1}, a^S ∈ [0,1],
         a^fast ∈ [0, P^FCAS_max/P_max],
         a^slow ∈ [0, P^FCAS_max/P_max],
         a^delay ∈ [0, P^FCAS_max/P_max]]

    Returns: (next_obs, reward, done, info)
    """

    def __init__(
        self,
        price_episode: np.ndarray,          # (T, 7)  — one day of prices (raw, not scaled)
        feature_dim: int = EMBED_DIM,
        params: BESSParams = None,
        mode: str = "joint",                # "spot", "fcas", or "joint"
    ):
        assert price_episode.ndim == 2 and price_episode.shape[1] == NUM_MARKETS
        self.prices    = price_episode       # (T, 7)
        self.T         = price_episode.shape[0]
        self.feat_dim  = feature_dim
        self.p         = params or BESSParams()
        self.mode      = mode

        self.obs_dim   = 1 + NUM_MARKETS + feature_dim   # SoC + ρ + f
        self.act_dim   = 6                               # eq. 23

        self._reset_state()

    # ─── Reset ────────────────────────────────────────────────────────────────

    def _reset_state(self):
        """Reset to start of episode."""
        self.t      = 0
        self.energy = self.p.capacity_mwh * 0.5         # start at 50% SoC
        self.ema_spot = self.prices[0, 0]                # EMA of spot price
        self.done   = False

    def reset(
        self,
        price_episode: np.ndarray = None,
        init_feature: np.ndarray = None,
    ) -> np.ndarray:
        """
        Reset environment, optionally with a new price episode.
        Returns initial observation.
        """
        if price_episode is not None:
            self.prices = price_episode
            self.T      = price_episode.shape[0]
        self._reset_state()
        feat = init_feature if init_feature is not None else np.zeros(self.feat_dim, dtype=np.float32)
        return self._make_obs(feat)

    # ─── Observation ──────────────────────────────────────────────────────────

    def _make_obs(self, feature_vec: np.ndarray) -> np.ndarray:
        """
        Constructs state s_t = [SoC_{t-1}, ρ_{t-1}, f_{t-1}]  (eq. 22)
        using CURRENT internal state (before the step).
        """
        soc = np.array([self.energy / self.p.capacity_mwh], dtype=np.float32)
        t_idx = min(self.t, self.T - 1)
        rho = self.prices[t_idx].astype(np.float32)             # (7,)
        f   = feature_vec.astype(np.float32)                    # (EMBED_DIM,)
        return np.concatenate([soc, rho, f])                    # (8 + EMBED_DIM,)

    # ─── Action mapping ───────────────────────────────────────────────────────

    @staticmethod
    def map_action(raw_action: np.ndarray, p: BESSParams) -> Tuple:
        """
        Maps raw network output ∈ [-1,1]^6 to physical BESS decisions.

        raw_action indices:
          0 → v_dch  (discharge flag)
          1 → v_ch   (charge flag)
          2 → a^S    (spot bid, normalised by Pmax)
          3 → a^fast (fast FCAS bid, normalised by Pmax)
          4 → a^slow (slow FCAS bid, normalised by Pmax)
          5 → a^delay(delayed FCAS bid, normalised by Pmax)

        Returns: (v_dch, v_ch, a_S_mw, a_fast_mw, a_slow_mw, a_delay_mw)
        """
        # Continuous → binary flags via threshold at 0
        v_dch = 1 if raw_action[0] > 0 else 0
        v_ch  = 1 if raw_action[1] > 0 else 0

        # Enforce mutual exclusivity (eq. 1): cannot charge AND discharge
        if v_dch == 1 and v_ch == 1:
            # Keep whichever has stronger signal
            if raw_action[0] >= raw_action[1]:
                v_ch = 0
            else:
                v_dch = 0

        # Map continuous bids from [-1,1] to [0, Pmax] — tanh output
        fcas_max_norm = p.fcas_max_mw / p.rated_power_mw   # ≤ 1.0

        a_S_norm     = np.clip((raw_action[2] + 1) / 2, 0, 1)          # [0,1]
        a_fast_norm  = np.clip((raw_action[3] + 1) / 2, 0, fcas_max_norm)
        a_slow_norm  = np.clip((raw_action[4] + 1) / 2, 0, fcas_max_norm)
        a_delay_norm = np.clip((raw_action[5] + 1) / 2, 0, fcas_max_norm)

        # Enforce total bid ≤ Pmax (eq. 8) by clipping sum
        total_norm = a_S_norm + a_fast_norm + a_slow_norm + a_delay_norm
        if total_norm > 1.0:
            scale = 1.0 / total_norm
            a_S_norm     *= scale
            a_fast_norm  *= scale
            a_slow_norm  *= scale
            a_delay_norm *= scale

        # Convert to MW
        a_S_mw     = a_S_norm     * p.rated_power_mw
        a_fast_mw  = a_fast_norm  * p.rated_power_mw
        a_slow_mw  = a_slow_norm  * p.rated_power_mw
        a_delay_mw = a_delay_norm * p.rated_power_mw

        return v_dch, v_ch, a_S_mw, a_fast_mw, a_slow_mw, a_delay_mw

    # ─── Step ─────────────────────────────────────────────────────────────────

    def step(
        self,
        raw_action: np.ndarray,
        next_feature: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute one 5-minute dispatch step.

        Args:
            raw_action:    (6,) from actor network, values in [-1, 1]
            next_feature:  (EMBED_DIM,) TTFE output for NEXT observation

        Returns:
            next_obs  : (obs_dim,)
            reward    : float
            done      : bool
            info      : dict with breakdown of revenue components
        """
        assert not self.done, "Episode finished — call reset() first."

        p = self.p
        rho = self.prices[self.t]           # [spot, FR, FL, SR, SL, DR, DL]
        spot, FR, FL, SR, SL, DR, DL = rho

        # ── Map action ────────────────────────────────────────────────────────
        v_dch, v_ch, a_S, a_fast, a_slow, a_delay = self.map_action(raw_action, p)

        # ── Update EMA spot price baseline  (eq. 25) ─────────────────────────
        tau = TAU_EMA
        self.ema_spot = tau * self.ema_spot + (1 - tau) * spot

        # ── Compute energy change (eq. 10, 11) ───────────────────────────────
        # Spot market energy change
        delta_e_spot = p.dt_h * (v_ch - v_dch) * a_S          # MWh

        # FCAS: only dispatched on contingency event
        # Simulate contingency with low probability (paper: stochastic)
        contingency_raise = 1 if (np.random.rand() < 0.01) else 0
        contingency_lower = 1 if (np.random.rand() < 0.01) else 0

        dt_fast  = 6   / 3600   # 6 seconds in hours
        dt_slow  = 60  / 3600   # 60 seconds
        dt_delay = 300 / 3600   # 5 minutes

        delta_e_fcas = (v_ch - v_dch) * (contingency_raise + contingency_lower) * (
            dt_fast * a_fast + dt_slow * a_slow + dt_delay * a_delay
        )

        delta_e = delta_e_spot + delta_e_fcas

        # ── Check SoC limits (eq. 9) ──────────────────────────────────────────
        new_energy = self.energy + delta_e
        violated   = (new_energy < p.e_min_mwh) or (new_energy > p.e_max_mwh)

        if violated:
            # Clip to valid range (paper also clips as safety net)
            new_energy = np.clip(new_energy, p.e_min_mwh, p.e_max_mwh)
            # Zero out the offending bids
            a_S = a_fast = a_slow = a_delay = 0.0
            v_dch = v_ch = 0

        self.energy = new_energy

        # ── Compute reward (eq. 26–30) ────────────────────────────────────────

        # Spot market reward (eq. 26)
        I_ch  = 1 if spot < self.ema_spot else 0
        I_dch = 1 if spot > self.ema_spot else 0

        r_spot = (
            a_S * spot * (v_dch * p.eff_dch - v_ch / p.eff_ch)          # revenue term
            + BETA_S * a_S * abs(spot - self.ema_spot)                   # shaping term
            * (I_dch * v_dch * p.eff_dch + I_ch * v_ch / p.eff_ch)
        )

        # FCAS rewards (eq. 27–29)
        r_fast  = a_fast  * (v_dch * p.eff_dch * FR + v_ch / p.eff_ch * FL)
        r_slow  = a_slow  * (v_dch * p.eff_dch * SR + v_ch / p.eff_ch * SL)
        r_delay = a_delay * (v_dch * p.eff_dch * DR + v_ch / p.eff_ch * DL)

        # Scale rewards by dispatch interval (Δt in hours)
        r_spot  *= p.dt_h
        r_fast  *= p.dt_h
        r_slow  *= p.dt_h
        r_delay *= p.dt_h

        # Degradation cost (objective eq. 4)
        deg_cost = p.degradation_c * p.dt_h * v_dch * (a_S + a_fast + a_slow + a_delay)

        # Mode selector: zero out unwanted markets
        if self.mode == "spot":
            r_fast = r_slow = r_delay = 0.0
        elif self.mode == "fcas":
            r_spot = 0.0

        # Total reward (eq. 30) minus degradation
        reward = r_spot + r_fast + r_slow + r_delay - deg_cost

        # SoC violation penalty
        if violated:
            reward -= PENALTY_VIOLATE

        # ── Advance timestep ──────────────────────────────────────────────────
        self.t += 1
        self.done = (self.t >= self.T)

        next_obs = self._make_obs(next_feature)

        info = {
            "r_spot"   : r_spot,
            "r_fast"   : r_fast,
            "r_slow"   : r_slow,
            "r_delay"  : r_delay,
            "deg_cost" : deg_cost,
            "violated" : violated,
            "soc"      : self.energy / p.capacity_mwh,
            "v_dch"    : v_dch,
            "v_ch"     : v_ch,
        }

        return next_obs, reward, self.done, info
