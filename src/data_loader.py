"""
data_loader.py — Load, merge, clean, and split ERCOT 2022 price data.

Pipeline:
  1. Load monthly parquet files for energy_prices and as_prices
  2. Inner-join on timestamp_utc
  3. Build the 5-column price matrix ρ_t = [spot, RegUp, RegDn, RRS, NSRS]
  4. Forward-fill at most 2 steps for minor gaps; drop remaining NaNs
  5. StandardScaler fit on train, applied to train+eval
  6. Return aligned numpy arrays and metadata

ERCOT 2022 corrections vs prior NEM-proxy version:
  - ECRS (dam_as_ecrs) excluded — all-zero in 2022, not active until June 2023
  - RegDn used exactly once — was incorrectly duplicated as FL and SL
  - 5 columns total, not 7
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
    ENERGY_COL, AS_COLS, PRICE_ORDER, NUM_MARKETS,
    TIMESTEPS_PER_DAY, EMBED_DIM,
)


# ── 1. Load monthly parquet files ─────────────────────────────────────────────

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
            f"No parquet files found for {table} year={year}. Pattern: {pattern}"
        )
    return pd.concat([pd.read_parquet(f, engine="fastparquet") for f in files]).sort_index()


# ── 2. Build 5-column ERCOT price matrix ──────────────────────────────────────

def build_price_matrix(year: int = TRAIN_YEAR) -> pd.DataFrame:
    """
    Returns DataFrame with columns [spot, RegUp, RegDn, RRS, NSRS]
    indexed by timestamp_utc, 5-minute resolution.

    ECRS excluded — verified all-zero in 2022.
    RegDn appears exactly once (no duplication).
    """
    df_e = _load_table("energy_prices", year)
    df_a = _load_table("as_prices",     year)

    price_df = pd.DataFrame(index=df_e.index)
    price_df["spot"]  = df_e[ENERGY_COL]
    for market, col in AS_COLS.items():
        price_df[market] = df_a[col]

    price_df = price_df.ffill(limit=2).dropna()

    # Sanity-check: confirm ECRS is absent
    if "dam_as_ecrs" in df_a.columns:
        ecrs_nonzero = (df_a.loc[price_df.index, "dam_as_ecrs"].fillna(0) > 0).sum()
        if ecrs_nonzero > 0:
            print(f"[DataLoader] WARNING: {ecrs_nonzero} non-zero ECRS rows found in {year} "
                  f"— ECRS exclusion may need review for years after June 2023.")

    return price_df[PRICE_ORDER]   # shape: (T, 5)


# ── 3. State builder ──────────────────────────────────────────────────────────

def build_state(soc: float, prices_5: np.ndarray,
                ttfe_features: np.ndarray, timestep: int) -> np.ndarray:
    """
    Builds 72-dim state vector: [SoC(1), prices(5), TTFE(64), hour_sin_cos(2)]

    The hour_sin_cos pair encodes position within the current operating hour
    (0–59 min, repeating every 12 timesteps). The agent uses this to anticipate
    hour-boundary SOC floor enforcement.

    Args:
        soc:           scalar in [0, 1]
        prices_5:      (5,) raw prices [spot, RegUp, RegDn, RRS, NSRS]
        ttfe_features: (64,) TTFE embedding
        timestep:      0–287 (position in daily episode)
    """
    angle = 2.0 * np.pi * (timestep % 12) / 12.0
    hour_features = np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
    return np.concatenate([[soc], prices_5, ttfe_features, hour_features]).astype(np.float32)


# ── 4. Train / eval split ─────────────────────────────────────────────────────

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

def load_all(year: int = TRAIN_YEAR) -> Dict:
    """
    Full pipeline. Returns a dict with keys:
        'train_raw'    : np.ndarray  (T_train, 5)
        'eval_raw'     : np.ndarray  (T_eval,  5)
        'train_scaled' : np.ndarray  (T_train, 5)
        'eval_scaled'  : np.ndarray  (T_eval,  5)
        'train_index'  : pd.DatetimeIndex
        'eval_index'   : pd.DatetimeIndex
        'scaler'       : fitted StandardScaler
        'columns'      : ['spot', 'RegUp', 'RegDn', 'RRS', 'NSRS']
    """
    print(f"[DataLoader] Loading year {year} ...")
    price_df = build_price_matrix(year)
    print(f"[DataLoader] Total rows after cleaning: {len(price_df):,}")

    train_df, eval_df = split_by_month(price_df)
    scaler   = fit_scaler(train_df)

    return {
        "train_raw"    : train_df.values.astype(np.float32),
        "eval_raw"     : eval_df.values.astype(np.float32),
        "train_scaled" : apply_scaler(train_df, scaler),
        "eval_scaled"  : apply_scaler(eval_df,  scaler),
        "train_index"  : train_df.index,
        "eval_index"   : eval_df.index,
        "scaler"       : scaler,
        "columns"      : PRICE_ORDER,
    }


# ── 7. Episode iterator ───────────────────────────────────────────────────────

def iter_daily_episodes(
    price_array: np.ndarray,
    timesteps_per_day: int = TIMESTEPS_PER_DAY,
) -> list:
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
