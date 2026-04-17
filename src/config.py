"""
config.py — All hyperparameters and constants.
Sourced from Table I of Li et al. (2024) and verified against actual data.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")

# ─── Data / Market ───────────────────────────────────────────────────────────
# Training year — 2022 selected for best coverage (79,700 non-null timesteps)
TRAIN_YEAR  = 2022
TRAIN_MONTHS = list(range(1, 11))   # Jan–Oct  (10 months)
EVAL_MONTHS  = [11, 12]             # Nov–Dec  (2 months)

DISPATCH_INTERVAL_MIN = 5           # NEM: 5-minute dispatch
TIMESTEPS_PER_DAY     = 288         # 24 * 60 / 5
NUM_MARKETS           = 7           # spot + 6 FCAS sub-markets

# Column names in parquet files
SPOT_COL = "rt_lmp"                 # energy_prices table
AS_COLS  = {                        # as_prices table — mapped to paper's 7 markets
    "FR": "dam_as_rrs",             # Fast Raise  ← Responsive Reserve
    "FL": "dam_as_regdn",           # Fast Lower  ← Regulation Down
    "SR": "dam_as_regup",           # Slow Raise  ← Regulation Up
    "SL": "dam_as_regdn",           # Slow Lower  ← Regulation Down (reused)
    "DR": "dam_as_ecrs",            # Delayed Raise ← ECRS
    "DL": "dam_as_nsrs",            # Delayed Lower ← Non-Spinning Reserve
}
# Ordered price vector: [spot, FR, FL, SR, SL, DR, DL]  (eq.12 in paper)
PRICE_ORDER = ["spot", "FR", "FL", "SR", "SL", "DR", "DL"]

# ─── BESS Physical Parameters ─────────────────────────────────────────────────
BESS_CAPACITY_MWH    = 10.0         # E  (MWh)
BESS_RATED_POWER_MW  = 2.0          # Pmax (MW)
BESS_FCAS_MAX_MW     = 1.0          # P^FCAS_max (MW) — 50% droop setting
BESS_EFF_CH          = 0.95         # η^ch
BESS_EFF_DCH         = 0.95         # η^dch
BESS_E_MIN_MWH       = 0.5          # Emin = 5% SoC
BESS_E_MAX_MWH       = 9.5          # Emax = 95% SoC
BESS_DEGRADATION_C   = 0.02         # c (AU$/MWh) — degradation cost coefficient

# ─── TTFE Architecture (Table I) ─────────────────────────────────────────────
TEMPORAL_SEG_LEN = 12               # L: historical window length (timesteps)
EMBED_DIM        = 64               # F': embedding dimension after linear projection
NUM_MHA_HEADS    = 4                # h: attention heads per MHA layer
NUM_MHA_LAYERS   = 2                # N_MHA: number of stacked MHA layers
FF_INNER_DIM     = 2048             # inner dim of Forward Net's first LT layer
DROPOUT          = 0.0              # paper does not specify dropout; start at 0

# ─── Reward Shaping ───────────────────────────────────────────────────────────
TAU_EMA         = 0.99              # τ^S: EMA smoothing for spot price baseline (eq.25)
BETA_S          = 0.1               # β^S: reward shaping weight (eq.26)
PENALTY_VIOLATE = 50.0              # constant penalty for SoC limit violation

# ─── SAC Hyperparameters (Phase 2 reference — do NOT implement now) ───────────
HIDDEN_DIM           = 512
NUM_HIDDEN_LAYERS    = 2
LR_POLICY            = 1e-4
LR_VALUE             = 1e-4
LR_Q                 = 1e-4
GAMMA                = 0.99
TAU_TARGET           = 0.005
ALPHA_ENTROPY        = 0.05
REPLAY_BUFFER_SIZE   = 100_000
BATCH_SIZE           = 1024
BETA_L               = 1.0          # ancillary loss coefficient (eq.41)
