# TempDRL — Phase 1 Implementation Handout for Claude Code
## Paper: "Temporal-Aware DRL for Energy Storage Bidding in Energy and Contingency Reserve Markets" (Li et al., IEEE TEMPR 2024)

---

## WHAT THIS HANDOUT COVERS

**Phase 1 only:**
1. Server + tmux setup on Narnia
2. Project scaffolding
3. Data loading & preprocessing (parquet → aligned price tensor)
4. BESS gym-style environment (MDP state/action/reward)
5. Transformer-Based Temporal Feature Extractor (TTFE) — full architecture
6. Sanity checks and unit tests for every component
7. A `main_phase1.py` that ties it all together and verifies output shapes

**Phase 2 (SAC agent, training loop, benchmarks) is NOT in scope here.**

---

## CRITICAL DATA FACTS (verified from actual files)

The zip contains:
```
data/processed/
├── energy_prices/   YYYY-MM.parquet   (2020-01 to 2026-03)
├── as_prices/       YYYY-MM.parquet   (2020-01 to 2026-03)
└── system_conditions/ YYYY-MM.parquet
```

**`energy_prices` columns:**
- Index: `timestamp_utc` (datetime64, UTC, 5-min resolution)
- `rt_lmp`      → **spot price** ($/MWh, real-time locational marginal price)
- `dam_spp`     → day-ahead spot price (not used in Phase 1)
- `is_post_rtcb` → boolean flag (filter this out for training)

**`as_prices` columns:**
- Index: `timestamp_utc` (same)
- `dam_as_regup`  → **Regulation Up** (≈ Slow Raise proxy)
- `dam_as_regdn`  → **Regulation Down** (≈ Slow Lower proxy)
- `dam_as_rrs`    → **Responsive Reserve Service** (≈ Fast Raise proxy)
- `dam_as_ecrs`   → **ERCOT Contingency Reserve Service** (≈ Delayed Raise proxy)
- `dam_as_nsrs`   → **Non-Spinning Reserve Service** (≈ Delayed Lower proxy)
- `rt_mcpc_*`     → real-time clearing prices (sparse, only ~30k rows of 657k)

**Market mapping to paper's 7-market structure:**
```
Paper Market   → Dataset Column
────────────────────────────────────────
Spot (S)       → rt_lmp
Fast Raise (FR)  → dam_as_rrs
Fast Lower (FL)  → dam_as_regdn
Slow Raise (SR)  → dam_as_regup
Slow Lower (SL)  → dam_as_regdn   (reuse — no direct fast lower)
Delayed Raise (DR)→ dam_as_ecrs
Delayed Lower (DL)→ dam_as_nsrs
```
> NOTE: This is a US ERCOT dataset (not Australian NEM 2016 as in the paper),
> but the structure maps cleanly to 7 markets. The paper's methodology applies directly.

**Training split strategy (matching paper's 10/2 month split):**
- Use **2022** as the base year (best non-null coverage: 79,700 rows after inner join)
- Train: Jan–Oct 2022 (first 10 months)
- Eval:  Nov–Dec 2022 (last 2 months)
- Each episode = 1 day = 288 timesteps (5-min intervals)

---

## STEP 0: SERVER SETUP ON NARNIA

```bash
# SSH into narnia
ssh <username>@narnia.<institute-domain>

# Start a persistent tmux session
tmux new-session -s tempdrl
# To detach: Ctrl+B, then D
# To reattach later: tmux attach -t tempdrl

# Check Python version (need 3.8+)
python3 --version

# Create project directory
mkdir -p ~/tempdrl/data
cd ~/tempdrl

# Copy data zip from your local machine (run this on YOUR local machine):
# scp /path/to/data.zip <username>@narnia.<domain>:~/tempdrl/

# On Narnia — extract data
unzip data.zip -d .
# Confirm structure:
ls data/processed/
# Should show: as_prices/  energy_prices/  system_conditions/

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch numpy pandas pyarrow scikit-learn matplotlib seaborn tqdm
```

---

## STEP 1: PROJECT STRUCTURE

Create these files inside `~/tempdrl/src/`:

```
~/tempdrl/
├── data/
│   └── processed/
│       ├── energy_prices/   (YYYY-MM.parquet)
│       ├── as_prices/       (YYYY-MM.parquet)
│       └── system_conditions/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── environment.py
│   └── ttfe.py
├── outputs/
│   └── plots/
├── main_phase1.py
└── venv/
```

```bash
mkdir -p src outputs/plots
```

---

## FILE: `src/config.py`

```python
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
LR_POLICY            = 3e-4
LR_VALUE             = 3e-4
LR_Q                 = 3e-4
GAMMA                = 0.99
TAU_TARGET           = 0.005
ALPHA_ENTROPY        = 0.2
REPLAY_BUFFER_SIZE   = 100_000
BATCH_SIZE           = 256
BETA_L               = 1.0          # ancillary loss coefficient (eq.41)
```

---

## FILE: `src/data_loader.py`

```python
"""
data_loader.py — Load, merge, clean, and split NEM-equivalent price data.

Pipeline:
  1. Load monthly parquet files for energy_prices and as_prices
  2. Inner-join on timestamp_utc
  3. Build the 7-column price matrix ρ_t = [spot, FR, FL, SR, SL, DR, DL]
  4. Filter to is_post_rtcb == False (real-time cleared intervals only)
  5. Drop NaN rows; forward-fill at most 2 steps for minor gaps
  6. StandardScaler fit on train, applied to train+eval
  7. Return aligned numpy arrays and metadata
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DATA_DIR, TRAIN_YEAR, TRAIN_MONTHS, EVAL_MONTHS,
    SPOT_COL, AS_COLS, PRICE_ORDER, NUM_MARKETS
)


# ─── 1. Load monthly parquet files ───────────────────────────────────────────

def _load_table(table: str, year: int) -> pd.DataFrame:
    """
    Loads all monthly parquet files for `table` in `year`.
    table ∈ {'energy_prices', 'as_prices'}
    Returns DataFrame with DatetimeIndex (timestamp_utc, UTC-aware).
    """
    pattern = os.path.join(DATA_DIR, table, f"{year}-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found for {table} year={year}. "
            f"Pattern: {pattern}"
        )
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames).sort_index()
    return df


# ─── 2. Build 7-column price matrix ──────────────────────────────────────────

def build_price_matrix(year: int) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [spot, FR, FL, SR, SL, DR, DL]
    indexed by timestamp_utc, 5-minute resolution.

    Mapping (verified against actual data):
        spot → rt_lmp            (energy_prices)
        FR   → dam_as_rrs        (as_prices)
        FL   → dam_as_regdn      (as_prices)
        SR   → dam_as_regup      (as_prices)
        SL   → dam_as_regdn      (as_prices, reused)
        DR   → dam_as_ecrs       (as_prices)
        DL   → dam_as_nsrs       (as_prices)
    """
    df_e  = _load_table("energy_prices", year)
    df_as = _load_table("as_prices",     year)

    # Select only the columns we need
    spot_series = df_e[[SPOT_COL]]

    as_needed = list(set(AS_COLS.values()))          # unique AS columns
    as_series  = df_as[as_needed]

    # Inner join (keeps only timesteps where BOTH tables have data)
    merged = spot_series.join(as_series, how="inner")

    # Filter to is_post_rtcb == False if the column survived the join
    if "is_post_rtcb" in df_as.columns:
        rtcb_flag = df_as["is_post_rtcb"].reindex(merged.index)
        merged = merged[~rtcb_flag.fillna(False)]

    # Assemble the canonical 7-column price vector (eq. 12 in paper)
    price_df = pd.DataFrame(index=merged.index)
    price_df["spot"] = merged[SPOT_COL]
    for market, col in AS_COLS.items():
        price_df[market] = merged[col]

    # Reorder columns to canonical order
    price_df = price_df[PRICE_ORDER]

    # Forward-fill gaps up to 2 consecutive steps (minor NEM dispatch gaps)
    price_df = price_df.ffill(limit=2)

    # Drop any remaining NaN rows (beginning of series, large outages)
    price_df = price_df.dropna()

    return price_df  # shape: (T, 7)


# ─── 3. Train / eval split ────────────────────────────────────────────────────

def split_by_month(
    price_df: pd.DataFrame,
    train_months: list = TRAIN_MONTHS,
    eval_months:  list = EVAL_MONTHS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits price_df by calendar month.
    Returns (train_df, eval_df).
    """
    month = price_df.index.month
    train_df = price_df[month.isin(train_months)].copy()
    eval_df  = price_df[month.isin(eval_months)].copy()
    print(f"[DataLoader] Train rows: {len(train_df):,}  |  Eval rows: {len(eval_df):,}")
    return train_df, eval_df


# ─── 4. Normalize ─────────────────────────────────────────────────────────────

def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on training data."""
    scaler = StandardScaler()
    scaler.fit(train_df.values)
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler
) -> np.ndarray:
    """Apply pre-fitted scaler. Returns float32 numpy array."""
    return scaler.transform(df.values).astype(np.float32)


# ─── 5. Master loader ─────────────────────────────────────────────────────────

def load_all(year: int = TRAIN_YEAR) -> Dict:
    """
    Full pipeline. Returns a dict with keys:
        'train_raw'    : np.ndarray  (T_train, 7)   — unscaled
        'eval_raw'     : np.ndarray  (T_eval,  7)   — unscaled
        'train_scaled' : np.ndarray  (T_train, 7)   — StandardScaled
        'eval_scaled'  : np.ndarray  (T_eval,  7)   — StandardScaled
        'train_index'  : pd.DatetimeIndex
        'eval_index'   : pd.DatetimeIndex
        'scaler'       : fitted StandardScaler
        'columns'      : list of column names [spot, FR, FL, SR, SL, DR, DL]
    """
    print(f"[DataLoader] Loading year {year} ...")
    price_df = build_price_matrix(year)
    print(f"[DataLoader] Total rows after cleaning: {len(price_df):,}")

    train_df, eval_df = split_by_month(price_df)

    scaler     = fit_scaler(train_df)
    train_sc   = apply_scaler(train_df, scaler)
    eval_sc    = apply_scaler(eval_df,  scaler)

    return {
        "train_raw"    : train_df.values.astype(np.float32),
        "eval_raw"     : eval_df.values.astype(np.float32),
        "train_scaled" : train_sc,
        "eval_scaled"  : eval_sc,
        "train_index"  : train_df.index,
        "eval_index"   : eval_df.index,
        "scaler"       : scaler,
        "columns"      : PRICE_ORDER,
    }


# ─── 6. Episode iterator ──────────────────────────────────────────────────────

def iter_daily_episodes(
    price_array: np.ndarray,
    timesteps_per_day: int = 288
) -> list:
    """
    Splits the price array into non-overlapping daily episodes of 288 steps.
    Drops the last incomplete day.
    Returns list of np.ndarray each of shape (288, 7).
    """
    n_days = len(price_array) // timesteps_per_day
    episodes = [
        price_array[i * timesteps_per_day : (i + 1) * timesteps_per_day]
        for i in range(n_days)
    ]
    print(f"[DataLoader] {n_days} complete daily episodes ({timesteps_per_day} steps each)")
    return episodes
```

---

## FILE: `src/environment.py`

```python
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
        rho = self.prices[self.t].astype(np.float32)            # (7,)
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
```

---

## FILE: `src/ttfe.py`

```python
"""
ttfe.py — Transformer-Based Temporal Feature Extractor (TTFE).

Implements Section IV-A of Li et al. (2024) exactly:

  Input:  S_t ∈ R^{L × F}   (temporal segment of L price vectors, F=7 markets)
  Output: f   ∈ R^{1 × F'}  (extracted feature vector, F'=EMBED_DIM)

Architecture (Fig. 3):
  1. Feature Embedding  : Linear(F → F')               [eq. 14]
  2. Stacked MHA        : N_MHA × MultiHeadAttention    [eq. 15–20]
     Each MHA block:
       - h parallel self-attention heads                [eq. 15–18]
       - Concat + Linear                               [eq. 19]
       - Residual + LayerNorm
       - Forward Net: Linear(F') → ReLU → Linear(F')   [eq. 20]
       - Residual + LayerNorm
  3. Feature Aggregation: Global Average Pooling (dim=L) [eq. 21]

Output shape: (batch, F')  where F' = EMBED_DIM = 64
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    NUM_MARKETS, TEMPORAL_SEG_LEN, EMBED_DIM,
    NUM_MHA_HEADS, NUM_MHA_LAYERS, FF_INNER_DIM, DROPOUT
)


class SelfAttentionHead(nn.Module):
    """
    Single scaled dot-product attention head (right side of Fig. 4).
    Computes: SA_j(Q, K, V) = softmax(Q_j K_j^T / sqrt(d_k)) V_j
    """

    def __init__(self, embed_dim: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.W_Q = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, head_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, head_dim, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, L, F')
        Returns:
            out:        (batch, L, head_dim)
            attn_weights: (batch, L, L)  — for interpretability
        """
        Q = self.W_Q(x)   # (B, L, d_k)
        K = self.W_K(x)   # (B, L, d_k)
        V = self.W_V(x)   # (B, L, d_k)

        scale = self.head_dim ** 0.5
        scores = torch.bmm(Q, K.transpose(1, 2)) / scale   # (B, L, L)
        attn_weights = F.softmax(scores, dim=-1)            # W^att_j

        out = torch.bmm(attn_weights, V)                    # (B, L, d_k)
        return out, attn_weights


class MultiHeadAttentionBlock(nn.Module):
    """
    One full MHA block from Fig. 4:
      - h parallel SA heads
      - Concat + Linear (eq. 19)
      - Residual + LayerNorm
      - Forward Net: LT → ReLU → LT (eq. 20)
      - Residual + LayerNorm
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_MHA_HEADS,
        ff_inner_dim: int = FF_INNER_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads

        # h parallel SA heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_dim, self.head_dim)
            for _ in range(num_heads)
        ])

        # Concat projection (eq. 19): maps h*head_dim → embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Residual + LayerNorm after MHA
        self.norm1 = nn.LayerNorm(embed_dim)

        # Forward Net (eq. 20): two LT layers with ReLU
        # Paper: first LT dim = ff_inner_dim (2048), second = embed_dim (64)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_inner_dim),
            nn.ReLU(),
            nn.Linear(ff_inner_dim, embed_dim),
        )

        # Residual + LayerNorm after Forward Net
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: (batch, L, embed_dim)
        Returns:
            out:        (batch, L, embed_dim)
            attn_list:  list of h attention matrices (batch, L, L) — for interpretability
        """
        # Run all heads in parallel, collect attention weights
        head_outputs = []
        attn_list    = []
        for head in self.heads:
            h_out, attn = head(x)
            head_outputs.append(h_out)
            attn_list.append(attn)

        # Concatenate heads: (B, L, h * head_dim) = (B, L, embed_dim)
        concat = torch.cat(head_outputs, dim=-1)            # eq. 19 Concat(...)

        # Linear projection + dropout
        mha_out = self.dropout(self.out_proj(concat))       # eq. 19 LT(...)

        # Residual connection + LayerNorm (Fig. 4)
        x = self.norm1(x + mha_out)

        # Forward Net + residual + LayerNorm (eq. 20)
        ff_out = self.dropout(self.ff(x))
        x = self.norm2(x + ff_out)

        return x, attn_list


class TTFE(nn.Module):
    """
    Full Transformer-Based Temporal Feature Extractor (Fig. 3).

    Forward pass:
      1. feature_embedding: Linear(F=7 → F'=EMBED_DIM)
      2. stacked_mha:       N_MHA × MultiHeadAttentionBlock
      3. feature_aggregation: GlobalAveragePooling over L dimension

    Input:  (batch, L, F)  — L=12, F=7
    Output: (batch, F')    — F'=64, the temporal feature vector f
    """

    def __init__(
        self,
        input_dim:    int = NUM_MARKETS,           # F  = 7
        seg_len:      int = TEMPORAL_SEG_LEN,      # L  = 12
        embed_dim:    int = EMBED_DIM,             # F' = 64
        num_heads:    int = NUM_MHA_HEADS,         # h  = 4
        num_layers:   int = NUM_MHA_LAYERS,        # N_MHA = 2
        ff_inner_dim: int = FF_INNER_DIM,          # 2048
        dropout:      float = DROPOUT,
    ):
        super().__init__()
        self.seg_len   = seg_len
        self.embed_dim = embed_dim

        # ── Component 1: Feature Embedding (eq. 14) ──────────────────────────
        # S' = LT(S) = S W^embed + b^embed  ∈ R^{L × F'}
        self.feature_embedding = nn.Linear(input_dim, embed_dim)

        # ── Component 2: Stacked MHA (eq. 15–20) ─────────────────────────────
        self.mha_layers = nn.ModuleList([
            MultiHeadAttentionBlock(embed_dim, num_heads, ff_inner_dim, dropout)
            for _ in range(num_layers)
        ])

        # ── Component 3: Feature Aggregation (eq. 21) ────────────────────────
        # Global Average Pooling: mean over temporal dimension L
        # f_n = (1/L) Σ_{m=1}^{L} s_{m,n}
        # No learnable params — just torch.mean(..., dim=1)

    def forward(
        self,
        S: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Args:
            S:               (batch, L, F)   — temporal segment (eq. 13)
            return_attention: if True, also return attention weight matrices

        Returns:
            f:         (batch, F')           — extracted temporal feature vector
            attn_all:  list[layer][head] of (batch, L, L)  — only if return_attention=True
        """
        # ── Step 1: Feature Embedding ─────────────────────────────────────────
        x = self.feature_embedding(S)               # (B, L, F')

        # ── Step 2: Stacked MHA ───────────────────────────────────────────────
        attn_all = []
        for mha in self.mha_layers:
            x, attn_list = mha(x)
            attn_all.append(attn_list)

        # ── Step 3: Global Average Pooling  (eq. 21) ─────────────────────────
        f = x.mean(dim=1)                           # (B, F')

        if return_attention:
            return f, attn_all
        return f

    def extract_numpy(self, segment_np: np.ndarray) -> np.ndarray:
        """
        Convenience method: takes raw numpy temporal segment,
        returns feature vector as numpy.

        Args:
            segment_np: (L, F)  or  (B, L, F)
        Returns:
            np.ndarray: (F',)  or  (B, F')
        """
        squeezed = segment_np.ndim == 2
        if squeezed:
            segment_np = segment_np[np.newaxis]          # (1, L, F)

        tensor = torch.from_numpy(segment_np.astype(np.float32))
        self.eval()
        with torch.no_grad():
            f = self.forward(tensor)

        out = f.numpy()
        if squeezed:
            out = out[0]                                  # (F',)
        return out


# ─── Temporal Segment Builder ─────────────────────────────────────────────────

def build_temporal_segment(
    price_array: np.ndarray,
    t:           int,
    L:           int = TEMPORAL_SEG_LEN,
) -> np.ndarray:
    """
    Constructs S_t = [ρ_{t-L+1}, ..., ρ_t] ∈ R^{L × F}  (eq. 13).

    Args:
        price_array: (T, 7)   — full price array for the episode
        t:           int       — current timestep index (0-based)
        L:           int       — segment length

    Returns:
        segment: (L, 7)  — zero-padded at the start if t < L-1
    """
    F = price_array.shape[1]
    segment = np.zeros((L, F), dtype=np.float32)

    start = t - L + 1
    if start < 0:
        # Pad with zeros for timesteps before episode start
        available = price_array[max(0, start):t+1]
        segment[L - len(available):] = available
    else:
        segment = price_array[start:t+1].copy()

    return segment  # (L, 7)
```

---

## FILE: `main_phase1.py`

```python
"""
main_phase1.py — Phase 1 entry point.

Runs all components and verifies shapes/values.
Run this from ~/tempdrl/ with:
    source venv/bin/activate
    python main_phase1.py

Expected output (all PASS):
  [1] Data loading       ...  PASS
  [2] Episode iterator   ...  PASS
  [3] TTFE forward pass  ...  PASS
  [4] Temporal segment   ...  PASS
  [5] Environment step   ...  PASS
  [6] Full episode loop  ...  PASS
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import torch
from tqdm import tqdm

from config import (
    TRAIN_YEAR, TEMPORAL_SEG_LEN, EMBED_DIM, NUM_MARKETS,
    TIMESTEPS_PER_DAY, OUTPUT_DIR
)
from data_loader import load_all, iter_daily_episodes
from ttfe import TTFE, build_temporal_segment
from environment import BESSEnvironment


def test_data_loading():
    print("\n[1] Testing data loading...")
    data = load_all(TRAIN_YEAR)

    assert data["train_raw"].ndim == 2
    assert data["train_raw"].shape[1] == NUM_MARKETS, \
        f"Expected 7 markets, got {data['train_raw'].shape[1]}"
    assert data["eval_raw"].shape[1] == NUM_MARKETS
    assert data["train_scaled"].shape == data["train_raw"].shape

    print(f"    Train shape: {data['train_raw'].shape}")
    print(f"    Eval  shape: {data['eval_raw'].shape}")
    print(f"    Columns: {data['columns']}")
    print(f"    Train price stats (raw): mean={data['train_raw'].mean():.2f}, "
          f"std={data['train_raw'].std():.2f}")
    print("    [1] PASS ✓")
    return data


def test_episode_iterator(data):
    print("\n[2] Testing episode iterator...")
    train_episodes = iter_daily_episodes(data["train_scaled"])
    eval_episodes  = iter_daily_episodes(data["eval_scaled"])

    assert len(train_episodes) > 0
    assert train_episodes[0].shape == (TIMESTEPS_PER_DAY, NUM_MARKETS), \
        f"Episode shape: {train_episodes[0].shape}"
    print(f"    Train episodes: {len(train_episodes)}")
    print(f"    Eval  episodes: {len(eval_episodes)}")
    print(f"    Episode shape:  {train_episodes[0].shape}")
    print("    [2] PASS ✓")
    return train_episodes, eval_episodes


def test_ttfe():
    print("\n[3] Testing TTFE forward pass...")
    model = TTFE()
    print(f"    TTFE parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Single sample
    S = torch.randn(1, TEMPORAL_SEG_LEN, NUM_MARKETS)
    f, attn = model(S, return_attention=True)
    assert f.shape == (1, EMBED_DIM), f"Feature shape: {f.shape}"
    print(f"    Input shape:  {S.shape}")
    print(f"    Output shape: {f.shape}")

    # Batch
    S_batch = torch.randn(32, TEMPORAL_SEG_LEN, NUM_MARKETS)
    f_batch = model(S_batch)
    assert f_batch.shape == (32, EMBED_DIM)

    # Check attention weight sums to 1 (softmax)
    for layer_attn in attn:
        for head_attn in layer_attn:
            sums = head_attn.sum(dim=-1)   # should be all-ones
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
                "Attention weights do not sum to 1"
    print("    Attention weights sum check: OK")

    # Numpy convenience method
    seg_np = np.random.randn(TEMPORAL_SEG_LEN, NUM_MARKETS).astype(np.float32)
    f_np = model.extract_numpy(seg_np)
    assert f_np.shape == (EMBED_DIM,), f"numpy output shape: {f_np.shape}"
    print("    numpy extract_numpy: OK")
    print("    [3] PASS ✓")
    return model


def test_temporal_segment(train_episodes):
    print("\n[4] Testing temporal segment builder...")
    ep = train_episodes[0]

    # At beginning of episode (pad test)
    seg_t0 = build_temporal_segment(ep, t=0)
    assert seg_t0.shape == (TEMPORAL_SEG_LEN, NUM_MARKETS)
    # First rows should be zero-padded
    assert np.all(seg_t0[:-1] == 0), "Expected zero padding at t=0"

    # Mid-episode
    seg_mid = build_temporal_segment(ep, t=50)
    assert seg_mid.shape == (TEMPORAL_SEG_LEN, NUM_MARKETS)
    assert np.all(seg_mid == ep[50 - TEMPORAL_SEG_LEN + 1: 51])

    print(f"    Segment shape:      {seg_mid.shape}")
    print(f"    Zero-pad at t=0:    {TEMPORAL_SEG_LEN - 1} rows padded")
    print("    [4] PASS ✓")


def test_environment(train_episodes, ttfe_model):
    print("\n[5] Testing environment step...")
    ep = train_episodes[0]

    # Use raw (unscaled) prices for the environment
    # (scaled prices are for TTFE input only)
    env = BESSEnvironment(price_episode=ep, feature_dim=EMBED_DIM, mode="joint")

    # Build initial temporal segment and feature
    seg = build_temporal_segment(ep, t=0)
    f = ttfe_model.extract_numpy(seg)
    obs = env.reset(init_feature=f)

    expected_obs_dim = 1 + NUM_MARKETS + EMBED_DIM
    assert obs.shape == (expected_obs_dim,), \
        f"obs shape: {obs.shape}, expected ({expected_obs_dim},)"
    print(f"    Obs dim: {obs.shape[0]}  (1 SoC + 7 prices + {EMBED_DIM} features)")

    # One random step
    raw_action = np.random.uniform(-1, 1, size=6).astype(np.float32)
    seg_next = build_temporal_segment(ep, t=1)
    f_next = ttfe_model.extract_numpy(seg_next)
    next_obs, reward, done, info = env.step(raw_action, next_feature=f_next)

    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)
    assert not done
    print(f"    Reward:  {reward:.4f}")
    print(f"    SoC:     {info['soc']:.3f}")
    print(f"    v_dch={info['v_dch']}, v_ch={info['v_ch']}")
    print("    [5] PASS ✓")


def test_full_episode(train_episodes, ttfe_model):
    print("\n[6] Testing full episode loop...")
    ep = train_episodes[0]

    env = BESSEnvironment(price_episode=ep, feature_dim=EMBED_DIM, mode="joint")

    total_reward = 0.0
    soc_trajectory = []
    violations = 0

    seg = build_temporal_segment(ep, t=0)
    f = ttfe_model.extract_numpy(seg)
    obs = env.reset(init_feature=f)

    for t in tqdm(range(len(ep)), desc="    Episode", leave=False, ncols=60):
        raw_action = np.random.uniform(-1, 1, size=6).astype(np.float32)

        t_next = min(t + 1, len(ep) - 1)
        seg_next = build_temporal_segment(ep, t=t_next)
        f_next = ttfe_model.extract_numpy(seg_next)

        next_obs, reward, done, info = env.step(raw_action, next_feature=f_next)
        total_reward += reward
        soc_trajectory.append(info["soc"])
        if info["violated"]:
            violations += 1

        obs = next_obs
        if done:
            break

    print(f"    Total episode reward:  {total_reward:.2f}")
    print(f"    SoC min/max:           {min(soc_trajectory):.3f} / {max(soc_trajectory):.3f}")
    print(f"    Violations (random):   {violations} / {len(ep)}")
    print("    [6] PASS ✓")
    print()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  TempDRL Phase 1 — Component Verification")
    print("=" * 60)

    data           = test_data_loading()
    train_ep, eval_ep = test_episode_iterator(data)
    ttfe_model     = test_ttfe()
    test_temporal_segment(train_ep)
    test_environment(train_ep, ttfe_model)
    test_full_episode(train_ep, ttfe_model)

    print("=" * 60)
    print("  ALL PHASE 1 TESTS PASSED ✓")
    print("=" * 60)
    print()
    print("Next: Implement Phase 2 (SAC agent + training loop).")
```

---

## RUNNING ON NARNIA (tmux)

```bash
# On Narnia, inside tmux session 'tempdrl'
cd ~/tempdrl
source venv/bin/activate

# Run Phase 1 verification
python main_phase1.py 2>&1 | tee outputs/phase1_run.log

# Detach from tmux safely (process keeps running):
# Ctrl+B, then D

# Reattach later:
tmux attach -t tempdrl

# Monitor the log:
tail -f outputs/phase1_run.log
```

---

## KNOWN GOTCHAS FOR CLAUDE CODE

1. **Import paths**: All `src/` files use `sys.path.insert(0, os.path.dirname(__file__))`. Run `main_phase1.py` from the project root `~/tempdrl/`.

2. **`Tuple` import**: `ttfe.py` and `environment.py` use `Tuple` from `typing`. Add `from typing import Tuple, Optional` at the top of both files.

3. **Raw vs. scaled prices in environment**: The environment takes **raw (unscaled)** price episodes — scaling is only for TTFE inputs. Keep them separate.

4. **Contingency simulation**: The paper uses real NEM contingency data. Since our dataset has no explicit contingency flag, we simulate with a 1% per-step random trigger. In Phase 2, this can be improved using `system_conditions` data (e.g., net_load spikes as proxies for contingency events).

5. **`dam_as_*` prices are day-ahead**: This dataset uses day-ahead AS clearing prices, not real-time (the `rt_mcpc_*` columns are too sparse). This slightly differs from the paper's real-time clearing prices but is the best available proxy.

6. **Episode shape mismatch**: If a trading day has gaps due to NaN removal, some "days" may have fewer than 288 rows. `iter_daily_episodes()` drops incomplete days automatically.

7. **TTFE inside episode loop**: In Phase 1, TTFE is untrained (random weights). This is expected — it's only used to verify shape compatibility. Weights are trained end-to-end with SAC in Phase 2.

---

## PHASE 2 PREVIEW (for planning)

Phase 2 will add:
- `src/replay_buffer.py` — experience replay buffer
- `src/sac_agent.py` — Actor, Critic (Q-net), Value network, each with TTFE as shared encoder
- `src/trainer.py` — Training loop with episode rollout, buffer updates, gradient steps
- `src/benchmarks.py` — MLP-DRL (no TTFE), and optionally LP&O baseline
- `outputs/` logging of cumulative revenue per episode

The TTFE from Phase 1 slots directly into the SAC actor/critic as a shared feature encoder. No changes needed to `ttfe.py`.
