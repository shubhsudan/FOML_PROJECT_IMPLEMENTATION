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
    frames = [pd.read_parquet(f, engine="fastparquet") for f in files]
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
