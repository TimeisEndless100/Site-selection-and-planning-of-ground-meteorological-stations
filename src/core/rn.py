from __future__ import annotations

import numpy as np
import pandas as pd


def assign_sets(
    df: pd.DataFrame,
    rn_slope_thresh: float = 45.0,
    rn_water_thresh: float = 0.5,
    rn_strategy: str = "physics",
    rn_noise_rate: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Return dataframe with unified set labels in {'P','U','RN'}."""
    rng = np.random.RandomState(seed)
    out = df.copy()
    is_rn = np.zeros(len(out), dtype=bool)

    if rn_strategy != "none":
        if "slope_max" in out.columns:
            is_rn |= (out["slope_max"].values > rn_slope_thresh)
        if "water_ratio" in out.columns:
            is_rn |= (out["water_ratio"].values > rn_water_thresh)

    if rn_strategy == "random":
        rn_n = int(is_rn.sum())
        is_rn[:] = False
        candidates = np.where(out["is_P"].values == 0)[0]
        if rn_n > 0 and len(candidates) > 0:
            picks = rng.choice(candidates, size=min(rn_n, len(candidates)), replace=False)
            is_rn[picks] = True

    out["set"] = "U"
    out.loc[is_rn, "set"] = "RN"
    out.loc[out["is_P"] == 1, "set"] = "P"

    if rn_strategy == "noisy" and rn_noise_rate > 0:
        rn_idx = np.where(out["set"].values == "RN")[0]
        if len(rn_idx) > 0:
            flip_n = int(len(rn_idx) * rn_noise_rate)
            if flip_n > 0:
                flip_idx = rng.choice(rn_idx, size=flip_n, replace=False)
                out.loc[flip_idx, "set"] = "U"

    return out

