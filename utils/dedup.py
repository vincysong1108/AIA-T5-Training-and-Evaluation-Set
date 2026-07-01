from __future__ import annotations

from typing import Iterable

import pandas as pd

from utils.id_utils import normalize_identifier_series


def keep_only_ids_present_in_reference(
    df: pd.DataFrame | None,
    reference_df: pd.DataFrame,
    item_id_col: str,
) -> tuple[pd.DataFrame | None, dict[str, int]]:
    if df is None:
        return None, {
            "input_rows": 0,
            "kept_rows": 0,
            "removed_missing_from_qa": 0,
        }

    history = df.copy()
    if history.empty or reference_df.empty or item_id_col not in history.columns or item_id_col not in reference_df.columns:
        return history.reset_index(drop=True), {
            "input_rows": len(history),
            "kept_rows": len(history),
            "removed_missing_from_qa": 0,
        }

    history[item_id_col] = normalize_identifier_series(history[item_id_col])
    reference_ids = set(normalize_identifier_series(reference_df[item_id_col]))
    kept = history[history[item_id_col].isin(reference_ids)].copy().reset_index(drop=True)
    return kept, {
        "input_rows": len(history),
        "kept_rows": len(kept),
        "removed_missing_from_qa": int(len(history) - len(kept)),
    }


def exclude_existing_ids(df: pd.DataFrame, exclude_ids: Iterable[str], item_id_col: str) -> pd.DataFrame:
    out = df.copy()
    out[item_id_col] = normalize_identifier_series(out[item_id_col])
    exclude_set = set(normalize_identifier_series(pd.Series(list(exclude_ids), dtype="object")))
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

    history[item_id_col] = normalize_identifier_series(history[item_id_col])
    new[item_id_col] = normalize_identifier_series(new[item_id_col])
    deduped_new = new[~new[item_id_col].isin(set(history[item_id_col]))].copy()
    appended = pd.concat([history, deduped_new], ignore_index=True, sort=False)
    appended = appended.drop_duplicates(subset=[item_id_col]).reset_index(drop=True)
    return appended, {
        "history_rows": len(history),
        "new_rows": len(new),
        "deduped_new_rows": len(deduped_new),
        "final_rows": len(appended),
    }
