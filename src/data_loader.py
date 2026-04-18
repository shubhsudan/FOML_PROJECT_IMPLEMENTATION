"""
data_loader.py — Multi-year ERCOT price data loader with ECRS auto-detection.

Pipeline:
  1. Load monthly parquet files for energy_prices and as_prices (all DATA_YEARS)
  2. Auto-skip months where ECRS is active (>50% of intervals non-zero)
  3. Build the 5-column price matrix ρ_t = [spot, RegUp, RegDn, RRS, NSRS]
  4. Forward-fill at most 2 steps for minor gaps; drop remaining NaNs
  5. Chronological 70/10/20 train/val/test split by complete days (288 steps)
  6. StandardScaler fit on train only, applied to train+val+test
  7. Return aligned numpy arrays and metadata

Energy price column: dam_spp (DAM Settlement Point Price)
  - Fully populated for 2020-2025
  - rt_lmp is null for 2024-2025 in this dataset

ECRS auto-filter:
  - 2020-2023 May: ECRS 0% active → included
  - 2023 Jun:      ECRS 69% active → skipped
  - 2023 Jul+:     ECRS 100% active → skipped
  - 2024-2025:     ECRS 100% active → skipped (plus rt_lmp null anyway)
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DATA_DIR, PARQUET_ENGINE, DATA_YEARS,
    TRAIN_FRAC, VAL_FRAC,
    TRAIN_YEAR, TRAIN_MONTHS, EVAL_MONTHS,   # kept for evaluate_revenue compat
    ENERGY_COL, AS_COLS, PRICE_ORDER, NUM_MARKETS,
    TIMESTEPS_PER_DAY, EMBED_DIM,
)


# ── 1. Parquet reader ─────────────────────────────────────────────────────────

def _read_parquet(filepath: str) -> pd.DataFrame:
    """Read a parquet file using the configured engine."""
    return pd.read_parquet(filepath, engine=PARQUET_ENGINE)


# ── 2. Build multi-year 5-column ERCOT price matrix ──────────────────────────

def build_full_price_matrix() -> pd.DataFrame:
    """
    Loads all DATA_YEARS, auto-skipping ECRS-active months.

    Returns DataFrame with columns [spot, RegUp, RegDn, RRS, NSRS]
    indexed by timestamp_utc, 5-minute resolution, covering all
    pre-ECRS complete months across DATA_YEARS.
    """
    dfs = []
    skipped_ecrs = []

    for yr in DATA_YEARS:
        for mo in range(1, 13):
            month_str = f"{yr}-{mo:02d}"
            e_f = os.path.join(DATA_DIR, "energy_prices", f"{month_str}.parquet")
            a_f = os.path.join(DATA_DIR, "as_prices",     f"{month_str}.parquet")

            if not os.path.exists(e_f) or not os.path.exists(a_f):
                continue

            try:
                df_e = _read_parquet(e_f)
                df_a = _read_parquet(a_f)
            except Exception as ex:
                print(f"[DataLoader] WARNING: Could not read {month_str}: {ex}")
                continue

            # Skip months with no energy data
            if df_e[ENERGY_COL].count() == 0:
                print(f"[DataLoader] Skipping {month_str}: no {ENERGY_COL} data")
                continue

            # Auto-skip ECRS-active months (>50% of intervals have non-zero ECRS)
            if "dam_as_ecrs" in df_a.columns:
                ecrs_frac = (df_a["dam_as_ecrs"].fillna(0) > 0).mean()
                if ecrs_frac > 0.5:
                    skipped_ecrs.append(month_str)
                    continue

            pdf = pd.DataFrame(index=df_e.index)
            pdf["spot"]  = df_e[ENERGY_COL]
            for market, col in AS_COLS.items():
                pdf[market] = df_a[col]

            dfs.append(pdf)

    if not dfs:
        raise RuntimeError("[DataLoader] No usable months found. Check DATA_YEARS and data files.")

    if skipped_ecrs:
        print(f"[DataLoader] Skipped {len(skipped_ecrs)} ECRS-active months: "
              f"{skipped_ecrs[0]} … {skipped_ecrs[-1]}")

    price_df = pd.concat(dfs).sort_index()
    price_df = price_df.ffill(limit=2).dropna()
    print(f"[DataLoader] Total rows after join+clean: {len(price_df):,}")
    return price_df[PRICE_ORDER]


# ── backward-compat single-year loader (used by evaluate_revenue.py) ──────────

def build_price_matrix(year: int = TRAIN_YEAR) -> pd.DataFrame:
    """
    Single-year price matrix. Used by evaluate_revenue.py.
    Kept for backward compatibility.
    """
    e_f_pattern = os.path.join(DATA_DIR, "energy_prices", f"{year}-*.parquet")
    a_f_pattern = os.path.join(DATA_DIR, "as_prices",     f"{year}-*.parquet")
    e_files = sorted(glob.glob(e_f_pattern))
    a_files = sorted(glob.glob(a_f_pattern))
    if not e_files:
        raise FileNotFoundError(f"No energy_prices parquet files for {year}. Pattern: {e_f_pattern}")

    df_e = pd.concat([_read_parquet(f) for f in e_files]).sort_index()
    df_a = pd.concat([_read_parquet(f) for f in a_files]).sort_index()

    price_df = pd.DataFrame(index=df_e.index)
    price_df["spot"]  = df_e[ENERGY_COL]
    for market, col in AS_COLS.items():
        price_df[market] = df_a[col]

    price_df = price_df.ffill(limit=2).dropna()
    return price_df[PRICE_ORDER]


# ── 3. State builder ──────────────────────────────────────────────────────────

def build_state(soc: float, prices_5: np.ndarray,
                ttfe_features: np.ndarray, timestep: int) -> np.ndarray:
    """
    Builds 72-dim state vector: [SoC(1), prices(5), TTFE(64), hour_sin_cos(2)]

    Args:
        soc:           scalar in [0, 1]
        prices_5:      (5,) raw prices [spot, RegUp, RegDn, RRS, NSRS]
        ttfe_features: (64,) TTFE embedding
        timestep:      0–287 (position in daily episode)
    """
    angle = 2.0 * np.pi * (timestep % 12) / 12.0
    hour_features = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
    return np.concatenate([[soc], prices_5, ttfe_features, hour_features]).astype(np.float32)


# ── 4. Chronological 70/10/20 split ──────────────────────────────────────────

def chronological_split(
    price_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits price_df into train/val/test by complete days, chronologically.

    Only days with exactly TIMESTEPS_PER_DAY rows are included.
    Split: first TRAIN_FRAC → train, next VAL_FRAC → val, remainder → test.

    Returns: (train_df, val_df, test_df)
    """
    # Get tz-naive date for grouping
    idx = price_df.index
    if hasattr(idx, 'tz') and idx.tz is not None:
        dates = idx.tz_convert("UTC").tz_localize(None).date
    else:
        dates = idx.date

    price_df = price_df.copy()
    price_df["_date"] = dates
    groups = price_df.groupby("_date")

    complete_days = sorted(
        [d for d, g in groups if len(g) == TIMESTEPS_PER_DAY]
    )
    n = len(complete_days)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    train_dates = set(complete_days[:n_train])
    val_dates   = set(complete_days[n_train:n_train + n_val])
    test_dates  = set(complete_days[n_train + n_val:])

    train_df = price_df[price_df["_date"].isin(train_dates)].drop(columns="_date")
    val_df   = price_df[price_df["_date"].isin(val_dates)].drop(columns="_date")
    test_df  = price_df[price_df["_date"].isin(test_dates)].drop(columns="_date")

    print(f"[DataLoader] Split: train={len(train_dates)} days "
          f"({100*len(train_dates)/n:.0f}%)  "
          f"val={len(val_dates)} days ({100*len(val_dates)/n:.0f}%)  "
          f"test={len(test_dates)} days ({100*len(test_dates)/n:.0f}%)")
    print(f"[DataLoader] Date ranges:")
    print(f"  Train: {complete_days[0]} → {complete_days[n_train-1]}")
    if val_dates:
        print(f"  Val:   {complete_days[n_train]} → {complete_days[n_train+n_val-1]}")
    if test_dates:
        print(f"  Test:  {complete_days[n_train+n_val]} → {complete_days[-1]}")

    return train_df, val_df, test_df


# ── backward-compat month-based split (used by evaluate_revenue.py) ───────────

def split_by_month(
    price_df: pd.DataFrame,
    train_months: list = TRAIN_MONTHS,
    eval_months:  list = EVAL_MONTHS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    month    = price_df.index.month
    train_df = price_df[month.isin(train_months)].copy()
    eval_df  = price_df[month.isin(eval_months)].copy()
    print(f"[DataLoader] Train rows: {len(train_df):,}  |  Eval rows: {len(eval_df):,}")
    return train_df, eval_df


# ── 5. Normalize ──────────────────────────────────────────────────────────────

def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df.values)
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(df.values).astype(np.float32)


# ── 6. Master loader ──────────────────────────────────────────────────────────

def load_all(year: int = None) -> Dict:
    """
    Full multi-year pipeline. Loads all DATA_YEARS, auto-skips ECRS-active months,
    splits chronologically 70/10/20, fits scaler on train only.

    Returns dict with keys:
        'train_raw'    : np.ndarray  (T_train, 5)
        'eval_raw'     : np.ndarray  (T_val,   5)  ← val set (used during training)
        'test_raw'     : np.ndarray  (T_test,  5)  ← held-out test set
        'train_scaled' : np.ndarray  (T_train, 5)
        'eval_scaled'  : np.ndarray  (T_val,   5)
        'test_scaled'  : np.ndarray  (T_test,  5)
        'train_index'  : pd.DatetimeIndex
        'eval_index'   : pd.DatetimeIndex
        'test_index'   : pd.DatetimeIndex
        'scaler'       : fitted StandardScaler
        'columns'      : ['spot', 'RegUp', 'RegDn', 'RRS', 'NSRS']
    """
    print(f"[DataLoader] Loading years {DATA_YEARS} ...")
    price_df = build_full_price_matrix()

    train_df, val_df, test_df = chronological_split(price_df)
    scaler = fit_scaler(train_df)

    return {
        "train_raw"    : train_df.values.astype(np.float32),
        "eval_raw"     : val_df.values.astype(np.float32),
        "test_raw"     : test_df.values.astype(np.float32),
        "train_scaled" : apply_scaler(train_df, scaler),
        "eval_scaled"  : apply_scaler(val_df,   scaler),
        "test_scaled"  : apply_scaler(test_df,  scaler),
        "train_index"  : train_df.index,
        "eval_index"   : val_df.index,
        "test_index"   : test_df.index,
        "scaler"       : scaler,
        "columns"      : PRICE_ORDER,
    }


# ── 7. Episode iterator ───────────────────────────────────────────────────────

def iter_daily_episodes(
    price_array: np.ndarray,
    timesteps_per_day: int = TIMESTEPS_PER_DAY,
) -> List[np.ndarray]:
    """
    Splits the price array into non-overlapping daily episodes of 288 steps.
    Drops the last incomplete day.
    Returns list of np.ndarray each of shape (288, 5).
    """
    n_days = len(price_array) // timesteps_per_day
    episodes = [
        price_array[i * timesteps_per_day:(i + 1) * timesteps_per_day]
        for i in range(n_days)
    ]
    print(f"[DataLoader] {n_days} complete daily episodes ({timesteps_per_day} steps each)")
    return episodes
