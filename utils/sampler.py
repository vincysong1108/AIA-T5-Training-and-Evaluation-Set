from __future__ import annotations

import numpy as np
import pandas as pd


def recency_weighted_sample(
    df: pd.DataFrame,
    n: int,
    date_col: str,
    recency_strength: float = 1.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()

    rng = np.random.default_rng(random_seed)
    sub = df.copy().sort_values(date_col)
    min_date = sub[date_col].min()
    max_date = sub[date_col].max()

    if pd.isna(min_date) or pd.isna(max_date) or min_date == max_date:
        weights = np.ones(len(sub), dtype=float)
    else:
        recency_days = (sub[date_col] - min_date).dt.days.astype(float)
        recency_scaled = recency_days / max(recency_days.max(), 1.0)
        weights = np.exp(recency_strength * recency_scaled)

    weights = weights / weights.sum()
    sampled_idx = rng.choice(sub.index, size=n, replace=False, p=weights)
    return sub.loc[sampled_idx].copy()
