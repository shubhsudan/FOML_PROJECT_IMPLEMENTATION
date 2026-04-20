"""
config_stage2.py — TempDRL Stage 2 hyperparameters for ERCOT RTC+B regime.

RTC+B (Real-Time Co-optimization Plus Batteries) launched December 5, 2025.
Stage 2 adds ECRS to the market set and switches from DAM to RT clearing prices.

Key changes vs Stage 1:
  - Price vector: 5-dim  → 12-dim  (adds rt_mcpc_* RT prices + dam_as_ecrs)
  - Temporal window: 12  → 32 steps (60 min → 160 min)
  - State:  72-dim → 78-dim  (adds syscond(7) + richer time(6), drops 2 hour sin/cos)
  - Action:  8-dim →  9-dim  (adds ECRS bid)
  - TTFE: input 5-dim → 12-dim  (re-initialized embedding, MHA weights transferred)
"""

import os

# ── Paths (inherit from Stage 1 base) ─────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data", "processed")
STAGE2_CKPT_DIR = os.path.join(BASE_DIR, "outputs", "checkpoints", "stage2")
STAGE2_LOG_DIR  = os.path.join(BASE_DIR, "outputs", "logs")

# Stage 1 best checkpoint (transfer weights from here)
STAGE1_CKPT_PATH = os.path.join(BASE_DIR, "outputs", "checkpoints", "best_model.pt")

# ── Post-RTC+B Data Coverage ──────────────────────────────────────────────────
STAGE2_DATA_YEARS  = [2025, 2026]
STAGE2_START_DATE  = "2025-12-05"          # RTC+B launch date

# ── 12-Dim Price Vector (ERCOT RTC+B) ─────────────────────────────────────────
# Order: [rt_lmp, rt_mcpc_regup, rt_mcpc_regdn, rt_mcpc_rrs,
#          rt_mcpc_ecrs, rt_mcpc_nsrs,
#          dam_spp, dam_as_regup, dam_as_regdn, dam_as_rrs,
#          dam_as_ecrs, dam_as_nsrs]
PRICE_COLS_12 = [
    "rt_lmp",
    "rt_mcpc_regup",
    "rt_mcpc_regdn",
    "rt_mcpc_rrs",
    "rt_mcpc_ecrs",
    "rt_mcpc_nsrs",
    "dam_spp",
    "dam_as_regup",
    "dam_as_regdn",
    "dam_as_rrs",
    "dam_as_ecrs",
    "dam_as_nsrs",
]
NUM_PRICE_DIMS = 12

# Price index map (for environment_stage2)
IDX_RT_LMP        = 0
IDX_RT_MCPC_REGUP = 1
IDX_RT_MCPC_REGDN = 2
IDX_RT_MCPC_RRS   = 3
IDX_RT_MCPC_ECRS  = 4
IDX_RT_MCPC_NSRS  = 5
IDX_DAM_SPP       = 6
IDX_DAM_REGUP     = 7
IDX_DAM_REGDN     = 8
IDX_DAM_RRS       = 9
IDX_DAM_ECRS      = 10
IDX_DAM_NSRS      = 11

# ── System Condition Features (7-dim) ─────────────────────────────────────────
SYSCOND_COLS = [
    "total_load_mw",
    "load_forecast_mw",
    "wind_actual_mw",
    "wind_forecast_mw",
    "solar_actual_mw",
    "solar_forecast_mw",
    "net_load_mw",
]
NUM_SYSCOND_DIMS = 7

# ── State / Action Dimensions ─────────────────────────────────────────────────
# State layout: [TTFE_features(64), SysCond(7), CyclicalTime(6), SoC(1)] = 78
STATE_DIM_S2   = 78
ACTION_DIM_S2  = 9    # [v_dch, v_ch, spot_dch, spot_ch, regup, regdn, rrs, ecrs, nsrs]

# ── TTFE Stage 2 Architecture ─────────────────────────────────────────────────
TTFE_INPUT_DIM_S2 = 12     # F = 12 price features
TTFE_SEG_LEN_S2   = 32     # L = 32 steps (160 minutes)
# MHA architecture same as Stage 1:
#   EMBED_DIM=64, NUM_MHA_HEADS=4, NUM_MHA_LAYERS=2, FF_INNER_DIM=2048

# ── Progressive Unfreezing Schedule ───────────────────────────────────────────
UNFREEZE_TOP_AT_EP  = 500    # Phase B: unfreeze top MHA layer  (lr=1e-5)
UNFREEZE_FULL_AT_EP = 1500   # Phase C: unfreeze all TTFE params (lr=1e-5)

# ── Stage 2 Training Hyperparameters ──────────────────────────────────────────
STAGE2_EPISODES    = 5000
STAGE2_BATCH_SIZE  = 128
STAGE2_BUFFER_SIZE = 30_000
GRAD_STEPS_PER_EP_S2 = 32    # fewer steps per ep (small dataset)
EVAL_EVERY_S2        = 50

# Learning rates
LR_ACTOR_S2  = 1e-4
LR_CRITIC_S2 = 3e-4
LR_TTFE_TOP  = 1e-5     # top MHA layer (Phase B)
LR_TTFE_FULL = 1e-5     # all TTFE layers (Phase C)

# SAC entropy target: -ACTION_DIM_S2 * 0.5 = -4.5
TARGET_ENTROPY_S2 = -float(ACTION_DIM_S2) * 0.5

# ── ERCOT AS Duration Requirements (RTC+B era) ────────────────────────────────
AS_DURATION_H_S2 = {
    "RegUp":  1.0,
    "RegDn":  1.0,
    "RRS":    1.0,
    "ECRS":   2.0,   # Emergency Credit Reserve Service: 2-hour duration
    "NSRS":   4.0,
}

# ── Divergence Guard ──────────────────────────────────────────────────────────
MAX_VIOLATIONS_PER_EP = 30    # Phase A threshold; Phase B/C uses 50

# ── Train/Val/Test Split ──────────────────────────────────────────────────────
STAGE2_TRAIN_FRAC = 0.70
STAGE2_VAL_FRAC   = 0.15
STAGE2_TEST_FRAC  = 0.15

# ── Reward Shaping (same as Stage 1) ─────────────────────────────────────────
TAU_EMA_S2       = 0.99
BETA_S_S2        = 0.1
REWARD_CLIP_S2   = 500.0
PENALTY_VIO_S2   = 50.0
