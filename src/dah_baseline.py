# src/dah_baseline.py
"""
DA-Aware Heuristic (DAH) baseline for RTC+B-era ERCOT.

Strategy:
  1. At the start of each hour, bid maximum qualifying MW into every
     DAM AS product (constrained by SOC feasibility under NPRR1282 durations).
  2. Each 5-min RT interval, execute greedy spot arbitrage:
     - If rt_lmp > DISCHARGE_THRESHOLD: discharge at P_MAX
     - If rt_lmp < CHARGE_THRESHOLD: charge at P_MAX
     - Else: hold
  3. Revenue computed using the correct two-settlement DART formula.
  4. Degradation cost: $15/MWh discharged.

This is what a well-informed, rule-following operator would do without ML.
TempDRL beating DAH over 60+ days is a publishable and meaningful claim.

Thresholds are set to represent an informed operator:
  DISCHARGE_THRESHOLD = mean(rt_lmp) + 0.5 × std(rt_lmp)  [computed on train set]
  CHARGE_THRESHOLD    = mean(rt_lmp) - 0.5 × std(rt_lmp)
"""

import numpy as np
from src.config_stage3 import (
    E_MAX, E_MIN, P_MAX, P_MIN, ETA_CH, ETA_DCH,
    FCAS_MAX, AS_DURATION, DEGRADATION_COST, DAM_MCPC_PRIOR
)


class DAHBaseline:

    PRODUCTS = ["regup", "regdn", "rrs", "ecrs", "nsrs"]

    def __init__(self, price_data, syscond_data,
                 discharge_threshold: float,
                 charge_threshold: float,
                 dam_mcpc_data=None):
        """
        Args:
            price_data:           dict of price arrays (same format as env)
            syscond_data:         dict of syscond arrays (unused by DAH but
                                  kept for interface consistency)
            discharge_threshold:  RT LMP above which DAH discharges
            charge_threshold:     RT LMP below which DAH charges
            dam_mcpc_data:        optional hourly DAM MCPC data; if None,
                                  uses DAM_MCPC_PRIOR constants
        """
        self.prices    = price_data
        self.dam_mcpc  = dam_mcpc_data
        self.disch_thr = discharge_threshold
        self.charg_thr = charge_threshold

    def run_episode(self, episode_start: int, n_steps: int = 288):
        """
        Run one full episode (day) of the DAH policy.

        Returns:
            dict with per-step revenues and summary metrics
        """
        soc       = (E_MAX + E_MIN) / 2.0
        dam_awards = np.zeros(5)
        dt = 5.0 / 60.0

        rev_spot_total   = 0.0
        rev_as_total     = 0.0
        rev_degrad_total = 0.0
        soc_clips = 0
        cycles    = 0.0
        prev_sign = 0   # track charge/discharge sign changes for cycle count

        for step in range(n_steps):
            t = episode_start + step

            # --- 1. Update DAM awards at the start of each hour ---
            if step % 12 == 0:
                dam_awards = self._compute_dam_awards(soc)

            # --- 2. Get prices ---
            rt_lmp = float(self.prices["rt_lmp"][t])
            rt_mcpc = np.array([
                float(self.prices[f"rt_mcpc_{p}"][t])
                for p in self.PRODUCTS
            ])
            dam_mcpc = self._get_dam_mcpc(t)

            # --- 3. Greedy energy dispatch ---
            # Compute available headroom
            discharge_res = (
                dam_awards[0] * AS_DURATION["regup"] +
                dam_awards[2] * AS_DURATION["rrs"]   +
                dam_awards[3] * AS_DURATION["ecrs"]  +
                dam_awards[4] * AS_DURATION["nsrs"]
            )
            charge_res = dam_awards[1] * AS_DURATION["regdn"]
            soc_floor  = E_MIN + discharge_res
            soc_ceil   = E_MAX - charge_res

            p_dispatch = 0.0
            if rt_lmp > self.disch_thr and soc > soc_floor + P_MAX * dt / ETA_DCH:
                p_dispatch = P_MAX
            elif rt_lmp < self.charg_thr and soc < soc_ceil - P_MAX * dt * ETA_CH:
                p_dispatch = P_MIN

            # Clamp to SOC bounds
            if p_dispatch > 0:
                max_out = (soc - soc_floor) * ETA_DCH / dt
                p_dispatch = min(p_dispatch, max(0.0, max_out))
            elif p_dispatch < 0:
                max_in = (soc_ceil - soc) / (ETA_CH * dt)
                p_dispatch = max(p_dispatch, min(0.0, -max_in))

            # --- 4. Update SoC ---
            if p_dispatch >= 0:
                energy_delta = -p_dispatch * dt / ETA_DCH
            else:
                energy_delta = -p_dispatch * dt * ETA_CH

            soc_new = soc + energy_delta
            if soc_new < E_MIN or soc_new > E_MAX:
                soc_clips += 1
            soc = np.clip(soc_new, E_MIN, E_MAX)

            # Cycle counting (sign changes)
            cur_sign = int(np.sign(p_dispatch))
            if cur_sign != 0 and prev_sign != 0 and cur_sign != prev_sign:
                cycles += 0.5
            if cur_sign != 0:
                prev_sign = cur_sign

            # --- 5. Revenue ---
            energy_mwh = p_dispatch * dt
            rev_spot   = energy_mwh * rt_lmp

            rev_as = 0.0
            for i, prod in enumerate(self.PRODUCTS):
                dart = dam_awards[i] * (dam_mcpc[i] - rt_mcpc[i]) * dt
                rt_r = dam_awards[i] * rt_mcpc[i] * dt   # DAH holds DAM position in RT
                rev_as += dart + rt_r

            rev_degrad = -DEGRADATION_COST * max(0.0, energy_mwh)

            rev_spot_total   += rev_spot
            rev_as_total     += rev_as
            rev_degrad_total += rev_degrad

        total = rev_spot_total + rev_as_total + rev_degrad_total
        return {
            "total_rev":   total,
            "rev_spot":    rev_spot_total,
            "rev_as":      rev_as_total,
            "rev_degrad":  rev_degrad_total,
            "soc_clips":   soc_clips,
            "cycles":      cycles,
        }

    def _compute_dam_awards(self, soc: float):
        """
        Compute max feasible DAM AS awards given current SoC.
        Bid FCAS_MAX for each product, then scale down if SOC infeasible.
        """
        awards = np.array([FCAS_MAX] * 5, dtype=float)

        # Check discharge reservation feasibility
        discharge_res = (
            awards[0] * AS_DURATION["regup"] +
            awards[2] * AS_DURATION["rrs"]   +
            awards[3] * AS_DURATION["ecrs"]  +
            awards[4] * AS_DURATION["nsrs"]
        )
        charge_res = awards[1] * AS_DURATION["regdn"]

        avail_discharge = max(0.0, soc - E_MIN)
        avail_charge    = max(0.0, E_MAX - soc)

        if discharge_res > avail_discharge:
            scale = avail_discharge / discharge_res
            awards[0] *= scale
            awards[2] *= scale
            awards[3] *= scale
            awards[4] *= scale

        if charge_res > avail_charge:
            awards[1] *= (avail_charge / charge_res)

        return awards

    def _get_dam_mcpc(self, t: int):
        if self.dam_mcpc is not None:
            return np.array([
                float(self.dam_mcpc[f"dam_mcpc_{p}"][t])
                for p in self.PRODUCTS
            ])
        return np.array([DAM_MCPC_PRIOR[p] for p in self.PRODUCTS])
