# src/data_bridge_stage3.py
"""
Adapts load_stage2_data() output → Stage 3 dict-style interface.

load_stage2_data() returns split numpy arrays:
  train_prices  (T_train, 12), val_prices  (T_val, 12), test_prices  (T_test, 12)
  train_syscond (T_train,  7), val_syscond (T_val,  7), test_syscond (T_test,  7)
  train_index / val_index / test_index  (DatetimeIndex, 5-min resolution)

Stage 3 env expects dict-style:
  price_data["rt_lmp"][t], price_data["rt_mcpc_regup"][t], ...
  syscond_data["total_load_mw"][t], ...

Days are local-offset 0-based within each split (0, 288, 576, ...).
"""

import numpy as np

# Column order as built by build_price_matrix_12() in data_loader_stage2.py
PRICE_COLS = [
    "rt_lmp",
    "dam_spp",
    "dam_as_regup",
    "dam_as_regdn",
    "dam_as_rrs",
    "dam_as_ecrs",
    "dam_as_nsrs",
    "rt_mcpc_regup",
    "rt_mcpc_regdn",
    "rt_mcpc_rrs",
    "rt_mcpc_ecrs",
    "rt_mcpc_nsrs",
]

SYSCOND_COLS = [
    "total_load_mw",
    "load_forecast_mw",
    "wind_actual_mw",
    "wind_forecast_mw",
    "solar_actual_mw",
    "solar_forecast_mw",
    "net_load_mw",
]


def _arr_to_dict(arr: np.ndarray, cols: list) -> dict:
    return {col: arr[:, i] for i, col in enumerate(cols)}


def _day_starts(n_rows: int, steps_per_day: int = 288) -> list:
    return list(range(0, n_rows, steps_per_day))


def make_stage3_splits(raw: dict) -> dict:
    """
    Args:
        raw: return value of load_stage2_data()

    Returns:
        dict with keys "train", "val", "test", each containing:
            "prices"  : dict mapping column name → 1-D numpy array
            "syscond" : dict mapping column name → 1-D numpy array
            "days"    : list of local episode-start indices (0, 288, ...)
    """
    splits = {}
    for split in ("train", "val", "test"):
        prices_dict  = _arr_to_dict(raw[f"{split}_prices"],  PRICE_COLS)
        syscond_dict = _arr_to_dict(raw[f"{split}_syscond"], SYSCOND_COLS)
        # Store DatetimeIndex so the env can build cyclical time features
        prices_dict["_index"] = raw[f"{split}_index"]
        n_rows = len(raw[f"{split}_prices"])
        splits[split] = {
            "prices":  prices_dict,
            "syscond": syscond_dict,
            "days":    _day_starts(n_rows),
        }
    return splits
