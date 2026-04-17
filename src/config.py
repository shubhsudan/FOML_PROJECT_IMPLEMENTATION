"""
config.py — TempDRL hyperparameters, ERCOT 2022 market structure.
All market rules sourced from official ERCOT documentation.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CKPT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR    = os.path.join(OUTPUT_DIR, "logs")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")

# ── ERCOT 2022 Market Structure ───────────────────────────────────────────────
# 5 markets: 1 energy + 4 AS products
# ECRS excluded — launched June 2023, all-zero in 2022 data (verified)
NUM_MARKETS  = 5
MARKET_NAMES = ["spot", "RegUp", "RegDn", "RRS", "NSRS"]

# Data column mapping (verified against actual parquet files)
ENERGY_COL = "rt_lmp"           # Real-time LMP $/MWh — energy_prices table
AS_COLS = {
    "RegUp": "dam_as_regup",    # Regulation Up   — discharge direction, 1-hr
    "RegDn": "dam_as_regdn",    # Regulation Down — charge direction,    1-hr
    "RRS":   "dam_as_rrs",      # Responsive Reserve — discharge,        1-hr
    "NSRS":  "dam_as_nsrs",     # Non-Spinning Reserve — discharge,      4-hr
    # dam_as_ecrs excluded — not active until June 2023; all-zero in 2022
}
# Price vector order: [spot, RegUp, RegDn, RRS, NSRS]
PRICE_ORDER  = ["spot", "RegUp", "RegDn", "RRS", "NSRS"]
# Backward-compat alias
SPOT_COL = ENERGY_COL

# ── ERCOT AS Duration Requirements (2022) ────────────────────────────────────
# Used for hourly SOC floor/ceiling enforcement (ERCOT BPM Dec 2022 / NPRR 1186)
AS_DURATION_H = {
    "RegUp": 1.0,
    "RegDn": 1.0,
    "RRS":   1.0,
    "NSRS":  4.0,
}

# ── Data Split ────────────────────────────────────────────────────────────────
TRAIN_YEAR        = 2022
TRAIN_MONTHS      = list(range(1, 11))   # Jan–Oct (10 months)
EVAL_MONTHS       = [11, 12]             # Nov–Dec (2 months)
TIMESTEPS_PER_DAY = 288                  # 5-min intervals per day
TIMESTEPS_PER_HOUR = 12                  # 5-min intervals per hour

# ── BESS Physical Parameters (paper Table I) ──────────────────────────────────
CAPACITY_MWH    = 10.0
RATED_POWER_MW  = 2.0
FCAS_MAX_MW     = 1.0        # Max MW per individual AS product bid (50% droop)
EFF_CH          = 0.95
EFF_DCH         = 0.95
E_MIN_MWH       = 0.5        # 5% SoC floor
E_MAX_MWH       = 9.5        # 95% SoC ceiling
E_INIT_MWH      = 5.0        # Start every episode at 50% SoC
DEGRADATION_C   = 0.02       # $/MWh degradation cost coefficient
DT_H            = 5.0 / 60  # 5-minute dispatch interval in hours

# Backward-compat aliases (used by older modules)
BESS_CAPACITY_MWH   = CAPACITY_MWH
BESS_RATED_POWER_MW = RATED_POWER_MW
BESS_FCAS_MAX_MW    = FCAS_MAX_MW
BESS_EFF_CH         = EFF_CH
BESS_EFF_DCH        = EFF_DCH
BESS_E_MIN_MWH      = E_MIN_MWH
BESS_E_MAX_MWH      = E_MAX_MWH
BESS_DEGRADATION_C  = DEGRADATION_C

# ── Network Dimensions ────────────────────────────────────────────────────────
# State:  [SoC(1), prices(5), TTFE_features(64), hour_sin_cos(2)] = 72
# Action: [v_dch, v_ch, a_spot_dch, a_spot_ch, a_regup, a_regdn, a_rrs, a_nsrs] = 8
STATE_DIM  = 1 + NUM_MARKETS + 64 + 2   # = 72
ACTION_DIM = 8

# ── TTFE Architecture ─────────────────────────────────────────────────────────
TEMPORAL_SEG_LEN = 12       # L: historical window (timesteps)
EMBED_DIM        = 64       # F': output feature dimension
NUM_MHA_HEADS    = 4        # attention heads per MHA
NUM_MHA_LAYERS   = 2        # stacked MHA layers
FF_INNER_DIM     = 2048     # ForwardNet inner dim
DROPOUT          = 0.0

# ── SAC Hyperparameters (all stability fixes baked in) ────────────────────────
HIDDEN_DIM          = 512
NUM_HIDDEN_LAYERS   = 2
LR_POLICY           = 1e-4
LR_VALUE            = 1e-4
LR_Q                = 1e-4
GAMMA               = 0.99
TAU_TARGET          = 0.005
ALPHA_ENTROPY       = 0.05                        # initial entropy temperature
TARGET_ENTROPY      = -float(ACTION_DIM) * 0.5   # = -4.0
REPLAY_BUFFER_SIZE  = 100_000
BATCH_SIZE          = 1024
GRAD_STEPS_PER_EP   = 72
MAX_GRAD_NORM       = 1.0

# ── Reward Shaping ────────────────────────────────────────────────────────────
TAU_EMA         = 0.99       # EMA smoothing for spot price baseline
BETA_S          = 0.1        # spot reward shaping coefficient
REWARD_CLIP     = 500.0      # clip rewards before buffer push
PENALTY_VIOLATE = 50.0       # SoC violation penalty
