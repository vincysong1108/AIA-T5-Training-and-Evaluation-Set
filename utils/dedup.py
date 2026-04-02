from __future__ import annotations

from typing import Iterable

import pandas as pd


def exclude_existing_ids(df: pd.DataFrame, exclude_ids: Iterable[str], item_id_col: str) -> pd.DataFrame:
    out = df.copy()
    out[item_id_col] = out[item_id_col].astype(str)
    exclude_set = set(map(str, exclude_ids))
    return out[~out[item_id_col].isin(exclude_set)].copy().reset_index(drop=True)


def append_deduplicated_history(history_df: pd.DataFrame, new_df: pd.DataFrame, item_id_col: str) -> tuple[pd.DataFrame, dict[str, int]]:
    history = history_df.copy()
    new = new_df.copy()
    if history.empty:
        appended = new.drop_duplicates(subset=[item_id_col]).reset_index(drop=True)
        return appended, {
            "history_rows": 0,
            "new_rows": len(new),
            "deduped_new_rows": len(appended),
            "final_rows": len(appended),
        }

    history[item_id_col] = history[item_id_col].astype(str)
    new[item_id_col] = new[item_id_col].astype(str)
    deduped_new = new[~new[item_id_col].isin(set(history[item_id_col]))].copy()
    appended = pd.concat([history, deduped_new], ignore_index=True, sort=False)
    appended = appended.drop_duplicates(subset=[item_id_col]).reset_index(drop=True)
    return appended, {
        "history_rows": len(history),
        "new_rows": len(new),
        "deduped_new_rows": len(deduped_new),
        "final_rows": len(appended),
    }
