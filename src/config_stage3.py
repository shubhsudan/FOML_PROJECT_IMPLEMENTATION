# src/config_stage3.py
# Stage 3: RTC+B-compliant config

# Hardware
DEVICE = "cuda:26"

# Battery specs (unchanged)
E_MAX      = 10.0   # MWh
E_MIN      = 0.5    # MWh  (5% of 10)
P_MAX      = 2.0    # MW max discharge
P_MIN      = -2.0   # MW max charge (negative = charging)
ETA_CH     = 0.92   # charge efficiency  (was 0.95 — corrected for LFP)
ETA_DCH    = 0.92   # discharge efficiency
SOC_INIT   = 0.5    # fractional starting SoC

# Markets (6 products, post-RTC+B)
MARKETS = ["regup", "regdn", "rrs", "ecrs", "nsrs"]  # 5 AS products
FCAS_MAX   = 1.0    # MW cap per AS product

# NPRR1282 SOC duration multipliers (hours) — CRITICAL FIX
AS_DURATION = {
    "regup": 0.5,   # was 1.0 — FIXED
    "regdn": 0.5,   # was 1.0 — FIXED
    "rrs":   0.5,   # was 1.0 — FIXED
    "ecrs":  1.0,   # was 2.0 — FIXED
    "nsrs":  4.0,   # unchanged
}

# Degradation — CRITICAL FIX
DEGRADATION_COST = 15.0   # $/MWh discharged (was 0.02)
CYCLE_HURDLE     = 20.0   # $/MWh bid-offer spread encoded in reward

# RTC+B settlement parameters
# DAM MCPC priors ($/MW-h) — used to simulate DAM layer when not in data
# Based on Dec 2025 ERCOT observed averages
DAM_MCPC_PRIOR = {
    "regup": 2.23,
    "regdn": 1.86,
    "rrs":   1.72,
    "ecrs":  1.72,
    "nsrs":  3.75,
}

# State dimension
# [TTFE(64) | SysCond(7) | CyclicalTime(6) | SoC(1) | DAM_awards(5) | prev_rt_mcpc(5)]
# DAM_awards: MW awarded in DAM for each AS product (known at start of hour)
# prev_rt_mcpc: RT MCPC from previous 5-min interval (market signal)
STATE_DIM = 64 + 7 + 6 + 1 + 5 + 5   # = 88

# Action space: single-scalar ESR dispatch + 5 AS bids
# action[0]: P in [-1, 1] scaled to [P_MIN, P_MAX] (positive=discharge, negative=charge)
# action[1..5]: AS MW bids [regup, regdn, rrs, ecrs, nsrs] in [0, FCAS_MAX]
ACTION_DIM = 6   # was 9 (v_dch, v_ch, spot_dch, spot_ch, regup, regdn, rrs, ecrs, nsrs)

# SAC hyperparams
LR_ACTOR      = 1e-4
LR_CRITIC     = 3e-4
LR_TTFE       = 1e-5
GAMMA         = 0.99
TAU           = 0.005
ALPHA_ENTROPY = 0.05
TARGET_ENTROPY = -ACTION_DIM * 0.5   # = -3.0
BATCH_SIZE    = 128
REPLAY_SIZE   = 30_000
GRAD_STEPS    = 32
WARMUP_STEPS  = 2_000

# Training
N_EPISODES    = 5_000
EVAL_EVERY    = 50
REWARD_CLIP   = 500.0
SOC_PENALTY   = 50.0

# SoC penalty coefficients
SOC_VIOLATION_PENALTY = 50.0
