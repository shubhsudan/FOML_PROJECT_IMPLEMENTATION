"""
data_loader_stage2.py — Post-RTC+B ERCOT data loader for TempDRL Stage 2.

Pipeline:
  1. Load monthly parquet files for energy_prices and as_prices (2025, 2026)
  2. Build 12-dim price matrix: [rt_lmp, rt_mcpc_*, dam_spp, dam_as_*]
     - For pre-RTC+B rows: fill rt_mcpc_* with dam_as_* as proxy
     - For post-RTC+B rows: use actual rt_mcpc_* values
  3. Load system condition features (total_load, wind, solar, etc.)
     - Falls back to zeros if syscond data not available
  4. Filter to post-RTC+B rows only (is_post_rtcb==True or date >= 2025-12-05)
  5. Chronological 70/15/15 train/val/test split by complete days (288 steps)
  6. StandardScaler fit on train, applied to all splits
  7. Return aligned arrays and metadata

State vector (78-dim):
  [TTFE_features(64), SysCond(7), CyclicalTime(6), SoC(1)]
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import PARQUET_ENGINE, TIMESTEPS_PER_DAY
from config_stage2 import (
    DATA_DIR, STAGE2_DATA_YEARS, STAGE2_START_DATE,
    PRICE_COLS_12, NUM_PRICE_DIMS,
    SYSCOND_COLS, NUM_SYSCOND_DIMS,
    STAGE2_TRAIN_FRAC, STAGE2_VAL_FRAC,
    TTFE_SEG_LEN_S2,
)


# ── Parquet reader ─────────────────────────────────────────────────────────────

def _read_parquet(filepath: str) -> pd.DataFrame:
    return pd.read_parquet(filepath, engine=PARQUET_ENGINE)


# ── 1. Build 12-dim price matrix ──────────────────────────────────────────────

def build_price_matrix_12() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads 2025-2026 energy and AS price parquets.
    Builds 12-dim price matrix with rt_mcpc_* proxy fill pre-RTC+B.

    Returns:
        price_df   : DataFrame with 12 price columns, 5-min resolution
        post_rtcb  : boolean Series (True = post Dec 5 2025 RTC+B launch)
    """
    dfs_price   = []
    dfs_syscond = []

    # dam_as_* → rt_mcpc_* proxy mapping (same service, DAM clearing price as proxy)
    proxy_map = {
        "rt_mcpc_regup": "dam_as_regup",
        "rt_mcpc_regdn": "dam_as_regdn",
        "rt_mcpc_rrs":   "dam_as_rrs",
        "rt_mcpc_ecrs":  "dam_as_ecrs",
        "rt_mcpc_nsrs":  "dam_as_nsrs",
    }

    for yr in STAGE2_DATA_YEARS:
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
                print(f"[DataLoaderS2] WARNING: Could not read {month_str}: {ex}")
                continue

            # Build price DataFrame for this month
            pdf = pd.DataFrame(index=df_e.index)

            # RT energy price
            pdf["rt_lmp"] = df_e["rt_lmp"] if "rt_lmp" in df_e.columns else np.nan

            # DAM energy price (always populated)
            pdf["dam_spp"] = df_e["dam_spp"] if "dam_spp" in df_e.columns else np.nan

            # DAM AS prices
            for col in ["dam_as_regup", "dam_as_regdn", "dam_as_rrs",
                        "dam_as_ecrs", "dam_as_nsrs"]:
                pdf[col] = df_a[col] if col in df_a.columns else 0.0

            # RT MCPC prices (RTC+B clearing prices — zeros pre-launch)
            for rt_col, dam_col in proxy_map.items():
                if rt_col in df_a.columns:
                    pdf[rt_col] = df_a[rt_col]
                elif rt_col in df_e.columns:
                    pdf[rt_col] = df_e[rt_col]
                else:
                    pdf[rt_col] = 0.0

            # System condition data (system_conditions directory)
            # Load first so we can use is_post_rtcb from syscond if available
            s_f = os.path.join(DATA_DIR, "system_conditions", f"{month_str}.parquet")
            df_s = None
            if os.path.exists(s_f):
                try:
                    df_s = _read_parquet(s_f)
                    sdf  = pd.DataFrame(index=df_e.index)
                    for col in SYSCOND_COLS:
                        sdf[col] = df_s[col] if col in df_s.columns else 0.0
                    dfs_syscond.append(sdf)
                except Exception:
                    df_s = None

            # Detect post-RTC+B: use is_post_rtcb flag if available, else date threshold
            if df_s is not None and "is_post_rtcb" in df_s.columns:
                pdf["is_post_rtcb"] = df_s["is_post_rtcb"].reindex(df_e.index).fillna(False).astype(bool)
            elif "is_post_rtcb" in df_e.columns:
                pdf["is_post_rtcb"] = df_e["is_post_rtcb"].fillna(False).astype(bool)
            elif "is_post_rtcb" in df_a.columns:
                pdf["is_post_rtcb"] = df_a["is_post_rtcb"].fillna(False).astype(bool)
            else:
                cutoff = pd.Timestamp(STAGE2_START_DATE, tz="UTC")
                if pdf.index.tz is None:
                    cutoff = cutoff.tz_localize(None)
                pdf["is_post_rtcb"] = pdf.index >= cutoff

            # Pre-RTC+B proxy fill: replace rt_mcpc_* zeros with dam_as_* values
            pre_mask = ~pdf["is_post_rtcb"]
            if pre_mask.any():
                for rt_col, dam_col in proxy_map.items():
                    pdf.loc[pre_mask, rt_col] = pdf.loc[pre_mask, dam_col]

            dfs_price.append(pdf)

    if not dfs_price:
        raise RuntimeError(
            f"[DataLoaderS2] No data found for years {STAGE2_DATA_YEARS}. "
            f"Check data/processed/energy_prices/ and as_prices/."
        )

    price_df = pd.concat(dfs_price).sort_index()

    # rt_lmp: forward-fill if missing (may be null outside operating hours)
    price_df["rt_lmp"] = price_df["rt_lmp"].ffill(limit=6).fillna(price_df["dam_spp"])

    # Final forward-fill and drop remaining NaN rows
    price_df = price_df.ffill(limit=2)
    post_rtcb = price_df.pop("is_post_rtcb")
    price_df  = price_df.dropna(subset=PRICE_COLS_12)

    post_rtcb = post_rtcb.reindex(price_df.index).fillna(False)

    print(f"[DataLoaderS2] Total rows loaded: {len(price_df):,}  "
          f"(post-RTC+B: {post_rtcb.sum():,})")

    # Syscond DataFrame (zeros if no files found)
    if dfs_syscond:
        syscond_df = pd.concat(dfs_syscond).sort_index().reindex(price_df.index).fillna(0.0)
    else:
        print("[DataLoaderS2] No syscond files found — using zeros for system conditions")
        syscond_df = pd.DataFrame(
            np.zeros((len(price_df), NUM_SYSCOND_DIMS), dtype=np.float32),
            index=price_df.index,
            columns=SYSCOND_COLS,
        )

    price_df = price_df[PRICE_COLS_12]
    return price_df, syscond_df, post_rtcb


# ── 2. Chronological split ────────────────────────────────────────────────────

def chronological_split_s2(
    price_df:   pd.DataFrame,
    syscond_df: pd.DataFrame,
    post_rtcb:  pd.Series,
) -> tuple:
    """
    Filters to post-RTC+B rows and splits into train/val/test by complete days.
    Split fractions: 70/15/15 (STAGE2_TRAIN_FRAC / STAGE2_VAL_FRAC / remainder).

    Returns 6 DataFrames: train/val/test for price and syscond.
    """
    # Keep only post-RTC+B rows
    mask        = post_rtcb.values
    price_post  = price_df[mask].copy()
    syscond_post = syscond_df[mask].copy()

    # Get UTC dates for grouping
    idx = price_post.index
    if hasattr(idx, 'tz') and idx.tz is not None:
        dates = idx.tz_convert("UTC").tz_localize(None).date
    else:
        dates = idx.date

    price_post["_date"]   = dates
    syscond_post["_date"] = dates

    groups = price_post.groupby("_date")
    complete_days = sorted([d for d, g in groups if len(g) == TIMESTEPS_PER_DAY])
    n = len(complete_days)

    if n < 10:
        raise RuntimeError(
            f"[DataLoaderS2] Only {n} complete post-RTC+B days found. "
            f"Need at least 10 for a meaningful split."
        )

    n_train = int(n * STAGE2_TRAIN_FRAC)
    n_val   = int(n * STAGE2_VAL_FRAC)

    train_dates = set(complete_days[:n_train])
    val_dates   = set(complete_days[n_train:n_train + n_val])
    test_dates  = set(complete_days[n_train + n_val:])

    def _split(df, date_set):
        return df[df["_date"].isin(date_set)].drop(columns="_date")

    train_p = _split(price_post,   train_dates)
    val_p   = _split(price_post,   val_dates)
    test_p  = _split(price_post,   test_dates)
    train_s = _split(syscond_post, train_dates)
    val_s   = _split(syscond_post, val_dates)
    test_s  = _split(syscond_post, test_dates)

    print(f"[DataLoaderS2] Post-RTC+B complete days: {n}  "
          f"(train={len(train_dates)}, val={len(val_dates)}, test={len(test_dates)})")
    if complete_days:
        print(f"[DataLoaderS2] Date range: {complete_days[0]} → {complete_days[-1]}")

    return train_p, val_p, test_p, train_s, val_s, test_s


# ── 3. Master loader ──────────────────────────────────────────────────────────

def load_stage2_data() -> Dict:
    """
    Full Stage 2 pipeline. Returns:
        'train_prices'   : np.ndarray (T_train, 12) raw
        'val_prices'     : np.ndarray (T_val,   12) raw
        'test_prices'    : np.ndarray (T_test,  12) raw
        'train_prices_sc': np.ndarray (T_train, 12) StandardScaler-normalized
        'val_prices_sc'  : np.ndarray (T_val,   12) normalized
        'test_prices_sc' : np.ndarray (T_test,  12) normalized
        'train_syscond'  : np.ndarray (T_train, 7)  raw
        'val_syscond'    : np.ndarray (T_val,   7)  raw
        'test_syscond'   : np.ndarray (T_test,  7)  raw
        'train_syscond_sc': normalized syscond
        'val_syscond_sc'  : normalized syscond
        'test_syscond_sc' : normalized syscond
        'price_scaler'   : fitted StandardScaler (12-dim)
        'syscond_scaler' : fitted StandardScaler (7-dim)
    """
    price_df, syscond_df, post_rtcb = build_price_matrix_12()
    train_p, val_p, test_p, train_s, val_s, test_s = \
        chronological_split_s2(price_df, syscond_df, post_rtcb)

    # Fit scalers on training data only
    price_scaler   = StandardScaler().fit(train_p.values)
    syscond_scaler = StandardScaler().fit(train_s.values)

    def _sc_price(df):
        return price_scaler.transform(df.values).astype(np.float32)

    def _sc_syscond(df):
        return syscond_scaler.transform(df.values).astype(np.float32)

    return {
        "train_prices"    : train_p.values.astype(np.float32),
        "val_prices"      : val_p.values.astype(np.float32),
        "test_prices"     : test_p.values.astype(np.float32),
        "train_prices_sc" : _sc_price(train_p),
        "val_prices_sc"   : _sc_price(val_p),
        "test_prices_sc"  : _sc_price(test_p),
        "train_syscond"   : train_s.values.astype(np.float32),
        "val_syscond"     : val_s.values.astype(np.float32),
        "test_syscond"    : test_s.values.astype(np.float32),
        "train_syscond_sc": _sc_syscond(train_s),
        "val_syscond_sc"  : _sc_syscond(val_s),
        "test_syscond_sc" : _sc_syscond(test_s),
        "price_scaler"    : price_scaler,
        "syscond_scaler"  : syscond_scaler,
        "train_index"     : train_p.index,
        "val_index"       : val_p.index,
        "test_index"      : test_p.index,
    }


# ── 4. Temporal segment builder (L=32, F=12) ──────────────────────────────────

def build_temporal_segment_12(
    price_array: np.ndarray,
    t:           int,
    L:           int = TTFE_SEG_LEN_S2,
) -> np.ndarray:
    """
    Constructs S_t = [ρ_{t-L+1}, ..., ρ_t] ∈ R^{L × 12}.
    Zero-pads at episode start if t < L-1.

    Args:
        price_array: (T, 12) price array for the episode (scaled)
        t:           current timestep (0-based)
        L:           segment length (default 32)

    Returns:
        segment: (L, 12)
    """
    F       = price_array.shape[1]
    segment = np.zeros((L, F), dtype=np.float32)
    start   = t - L + 1

    if start < 0:
        available = price_array[max(0, start):t + 1]
        segment[L - len(available):] = available
    else:
        segment = price_array[start:t + 1].copy()

    return segment   # (L, 12)


# ── 5. State builder (78-dim) ─────────────────────────────────────────────────

def build_state_78(
    soc:         float,
    syscond_7:   np.ndarray,
    time_6:      np.ndarray,
    ttfe_feat_64: np.ndarray,
) -> np.ndarray:
    """
    Builds 78-dim state: [TTFE(64), SysCond(7), Time(6), SoC(1)]

    Args:
        soc:           scalar in [0, 1]
        syscond_7:     (7,) normalized system condition features
        time_6:        (6,) cyclical time encoding
        ttfe_feat_64:  (64,) TTFE output features
    """
    return np.concatenate([
        ttfe_feat_64,
        syscond_7,
        time_6,
        [soc],
    ]).astype(np.float32)


def build_time_6(timestep: int, day_of_week: int = 0, month: int = 1) -> np.ndarray:
    """
    6-dim cyclical time encoding:
        [sin/cos(hour_of_day), sin/cos(day_of_week), sin/cos(month)]

    Args:
        timestep:    0–287 (5-min intervals within day)
        day_of_week: 0=Monday … 6=Sunday
        month:       1–12

    Returns:
        (6,) float32 array
    """
    hour_of_day = timestep / 12.0          # 0.0–23.917 (continuous)
    a_hour = 2.0 * np.pi * hour_of_day / 24.0
    a_dow  = 2.0 * np.pi * day_of_week / 7.0
    a_mon  = 2.0 * np.pi * (month - 1)   / 12.0

    return np.array([
        np.sin(a_hour), np.cos(a_hour),
        np.sin(a_dow),  np.cos(a_dow),
        np.sin(a_mon),  np.cos(a_mon),
    ], dtype=np.float32)


# ── 6. Episode iterator ───────────────────────────────────────────────────────

def iter_daily_episodes_s2(
    price_array:   np.ndarray,
    syscond_array: np.ndarray,
    index:         Optional[object] = None,
    timesteps_per_day: int = TIMESTEPS_PER_DAY,
) -> List[Dict]:
    """
    Splits price and syscond arrays into non-overlapping daily episodes.
    Returns list of dicts with keys: 'prices', 'syscond', 'day_of_week', 'month'.
    """
    n_days   = len(price_array) // timesteps_per_day
    episodes = []

    for i in range(n_days):
        s = i * timesteps_per_day
        e = s + timesteps_per_day
        ep = {
            "prices":  price_array[s:e],
            "syscond": syscond_array[s:e],
            "day_of_week": 0,
            "month": 1,
        }
        if index is not None:
            try:
                ts  = index[s]
                ep["day_of_week"] = ts.dayofweek
                ep["month"]       = ts.month
            except Exception:
                pass
        episodes.append(ep)

    print(f"[DataLoaderS2] {n_days} complete daily episodes "
          f"({timesteps_per_day} steps each)")
    return episodes
