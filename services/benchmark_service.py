from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from services.rolling_eval_service import (
    define_medium_longtail_classes,
    filter_non_training_rows,
    standardize_date_col,
    standardize_str_col,
)
from utils.dedup import append_deduplicated_history
from utils.logger import RunLogger
from utils.sampler import recency_weighted_sample


def _resolve_available_class_col(df: pd.DataFrame | None, candidate_cols: list[str]) -> str | None:
    if df is None:
        return None
    return next((col for col in candidate_cols if col and col in df.columns), None)


def _resolve_available_column(df: pd.DataFrame | None, candidate_cols: list[str]) -> str | None:
    if df is None:
        return None
    return next((col for col in candidate_cols if col and col in df.columns), None)


def _apply_fixed_eval_class_caps(
    df: pd.DataFrame,
    class_col: str,
    qa_date_col: str,
    p0_p1_classes: list[str],
    medium_classes: list[str],
    longtail_classes: list[str],
    class_caps: dict[str, int],
    protected_item_ids: set[str] | None = None,
    item_id_col: str | None = None,
) -> pd.DataFrame:
    if df.empty or class_col not in df.columns:
        return df.copy()

    out = standardize_str_col(df.copy(), class_col)
    out = standardize_date_col(out, qa_date_col)
    protected_item_ids = protected_item_ids or set()
    if item_id_col and item_id_col in out.columns:
        out[item_id_col] = out[item_id_col].astype(str)

    capped_parts: list[pd.DataFrame] = []
    p0_p1_set = set(p0_p1_classes)
    medium_set = set(medium_classes)
    longtail_set = set(longtail_classes)

    for class_name, group_df in out.groupby(class_col, dropna=False):
        if class_name in p0_p1_set:
            cap = class_caps["p0p1"]
        elif class_name in medium_set:
            cap = class_caps["medium"]
        elif class_name in longtail_set:
            cap = class_caps["longtail"]
        else:
            cap = class_caps["longtail"]

        protected_df = group_df.iloc[0:0].copy()
        candidate_df = group_df.copy()
        if item_id_col and item_id_col in group_df.columns and protected_item_ids:
            protected_df = group_df[group_df[item_id_col].isin(protected_item_ids)].copy()
            candidate_df = group_df[~group_df[item_id_col].isin(protected_item_ids)].copy()

        remaining_slots = max(cap - len(protected_df), 0)
        candidate_kept = candidate_df.sort_values(qa_date_col, ascending=False).head(remaining_slots).copy()
        kept = pd.concat([protected_df, candidate_kept], ignore_index=True, sort=False)
        capped_parts.append(kept)

    return pd.concat(capped_parts, ignore_index=True, sort=False).reset_index(drop=True)


def _backfill_class_column(df: pd.DataFrame, target_col: str, source_col: str) -> pd.DataFrame:
    out = df.copy()
    if source_col not in out.columns:
        return out
    if target_col not in out.columns:
        out[target_col] = out[source_col]
        return out

    target_as_str = out[target_col].astype(str).str.strip().str.lower()
    missing_mask = out[target_col].isna() | (target_as_str == "") | (target_as_str == "nan")
    out.loc[missing_mask, target_col] = out.loc[missing_mask, source_col]
    return out


def _overlay_original_qa_fields(
    eval_df: pd.DataFrame,
    qa_df: pd.DataFrame,
    item_id_col: str,
    qa_date_col: str,
    qa_tier3_col: str,
    qa_tier1_col: str | None = None,
) -> pd.DataFrame:
    if eval_df.empty or qa_df.empty or item_id_col not in eval_df.columns or item_id_col not in qa_df.columns:
        return eval_df.copy()

    qa_lookup_cols = [item_id_col, qa_date_col, qa_tier3_col]
    if qa_tier1_col and qa_tier1_col in qa_df.columns:
        qa_lookup_cols.append(qa_tier1_col)

    qa_lookup = qa_df[qa_lookup_cols].copy()
    qa_lookup[item_id_col] = qa_lookup[item_id_col].astype(str)
    qa_lookup = standardize_date_col(qa_lookup, qa_date_col)
    qa_lookup = qa_lookup.drop_duplicates(subset=[item_id_col], keep="last")
    rename_map = {col: f"__qa_overlay__{col}" for col in qa_lookup_cols if col != item_id_col}
    qa_lookup = qa_lookup.rename(columns=rename_map)

    out = eval_df.copy()
    out[item_id_col] = out[item_id_col].astype(str)
    if qa_date_col in out.columns:
        out = standardize_date_col(out, qa_date_col)
    out = out.merge(qa_lookup, on=item_id_col, how="left")

    for original_col, overlay_col in rename_map.items():
        if original_col not in out.columns:
            out[original_col] = out[overlay_col]
        else:
            if original_col == qa_date_col:
                out[original_col] = out[overlay_col].combine_first(out[original_col])
            else:
                original_as_str = out[original_col].astype(str).str.strip().str.lower()
                missing_mask = out[original_col].isna() | (original_as_str == "") | (original_as_str == "nan")
                out.loc[missing_mask, original_col] = out.loc[missing_mask, overlay_col]
        out = out.drop(columns=[overlay_col], errors="ignore")

    return out


def _compute_eval_take_with_reserve(target_n: int, available_n: int, reserve_ratio: float = 0.60) -> int:
    if available_n <= 0 or target_n <= 0:
        return 0
    if available_n >= target_n:
        return target_n
    usable_n = int(np.floor(available_n * (1.0 - reserve_ratio)))
    return max(0, min(target_n, usable_n))


def _refresh_historical_fixed_eval(
    historical_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    item_id_col: str,
    qa_date_col: str,
    historical_class_col: str,
    candidate_class_col: str,
    refresh_pct: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    history = historical_df.copy()
    candidates = candidate_df.copy()
    if history.empty:
        return candidate_df.copy(), {"historical_rows": 0, "removed_rows": 0, "added_rows": len(candidate_df), "final_rows": len(candidate_df)}

    if refresh_pct <= 0:
        preserved_history = history.drop_duplicates(subset=[item_id_col]).reset_index(drop=True)
        preserved_history = standardize_str_col(preserved_history, historical_class_col)
        candidates = standardize_str_col(candidates, candidate_class_col)
        appendable_rows = pd.DataFrame(columns=candidates.columns)
        appendable_rows = candidates[
            ~candidates[item_id_col].astype(str).isin(set(preserved_history[item_id_col].astype(str)))
        ].copy()
        if not appendable_rows.empty:
            preserved_history = pd.concat([preserved_history, appendable_rows], ignore_index=True, sort=False)
            preserved_history = preserved_history.drop_duplicates(subset=[item_id_col]).reset_index(drop=True)
        preserved_history = _backfill_class_column(
            preserved_history,
            target_col=candidate_class_col,
            source_col=historical_class_col,
        )
        return preserved_history, {
            "historical_rows": len(history),
            "removed_rows": 0,
            "added_rows": len(appendable_rows),
            "final_rows": len(preserved_history),
        }

    history = standardize_date_col(history, qa_date_col)
    history[item_id_col] = history[item_id_col].astype(str)
    candidates[item_id_col] = candidates[item_id_col].astype(str)
    history = standardize_str_col(history, historical_class_col)
    candidates = standardize_str_col(candidates, candidate_class_col)

    refreshed_parts: list[pd.DataFrame] = []
    removed_rows = 0
    added_rows = 0

    for class_name, history_group in history.groupby(historical_class_col, dropna=False):
        history_group = history_group.sort_values(qa_date_col, ascending=True).copy()
        target_size = len(history_group)

        candidate_group = candidates[candidates[candidate_class_col] == class_name].copy()
        candidate_group = candidate_group[
            ~candidate_group[item_id_col].isin(set(history_group[item_id_col]))
        ].copy()

        replace_n = max(1, int(target_size * refresh_pct / 100.0))
        replace_n = min(replace_n, len(candidate_group), target_size)

        if replace_n <= 0:
            refreshed_parts.append(history_group)
            continue

        retained_group = history_group.iloc[replace_n:].copy()
        replacement_rows = candidate_group.head(replace_n).copy()

        combined_group = pd.concat([retained_group, replacement_rows], ignore_index=True, sort=False)
        combined_group = combined_group.drop_duplicates(subset=[item_id_col]).copy()

        # Backfill from original history if dedup or lack of candidates shrank the class.
        if len(combined_group) < target_size:
            backfill = history_group[
                ~history_group[item_id_col].isin(set(combined_group[item_id_col]))
            ].copy()
            needed = target_size - len(combined_group)
            if needed > 0 and not backfill.empty:
                combined_group = pd.concat([combined_group, backfill.tail(needed)], ignore_index=True, sort=False)

        combined_group = combined_group.head(target_size).copy()
        refreshed_parts.append(combined_group)
        removed_rows += replace_n
        added_rows += len(replacement_rows)

    refreshed = pd.concat(refreshed_parts, ignore_index=True, sort=False)
    refreshed = refreshed.drop_duplicates(subset=[item_id_col]).reset_index(drop=True)

    if len(refreshed) < len(history):
        fallback_rows = history[~history[item_id_col].isin(set(refreshed[item_id_col]))].copy()
        needed = len(history) - len(refreshed)
        if needed > 0 and not fallback_rows.empty:
            refreshed = pd.concat([refreshed, fallback_rows.head(needed)], ignore_index=True, sort=False)
            refreshed = refreshed.drop_duplicates(subset=[item_id_col]).reset_index(drop=True)

    refreshed = _backfill_class_column(
        refreshed,
        target_col=candidate_class_col,
        source_col=historical_class_col,
    )

    return refreshed, {
        "historical_rows": len(history),
        "removed_rows": removed_rows,
        "added_rows": added_rows,
        "final_rows": len(refreshed),
    }


def build_fixed_benchmark_set(
    qa_df: pd.DataFrame,
    qa_date_col: str,
    qa_class_col: str,
    p0_p1_classes: list[str],
    medium_classes: list[str],
    longtail_classes: list[str],
    sample_sizes: dict[str, int] | None = None,
    recency_strength: float = 0.8,
    benchmark_start_date: str | None = None,
    benchmark_end_date: str | None = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if sample_sizes is None:
        sample_sizes = {"p0p1": 450, "medium": 80, "longtail": 25}

    qa = qa_df.copy()
    qa = standardize_str_col(qa, qa_class_col)
    qa = standardize_date_col(qa, qa_date_col)

    if benchmark_start_date is not None:
        qa = qa[qa[qa_date_col] >= pd.to_datetime(benchmark_start_date)].copy()
    if benchmark_end_date is not None:
        qa = qa[qa[qa_date_col] <= pd.to_datetime(benchmark_end_date)].copy()

    rows = []
    sampled = []
    group_specs = [
        ("p0p1", p0_p1_classes, sample_sizes["p0p1"]),
        ("medium", medium_classes, sample_sizes["medium"]),
        ("longtail", longtail_classes, sample_sizes["longtail"]),
    ]

    for group_name, class_list, target_n in group_specs:
        for class_name in class_list:
            sub = qa[qa[qa_class_col] == class_name].copy()
            available_n = len(sub)
            final_n = _compute_eval_take_with_reserve(target_n=target_n, available_n=available_n, reserve_ratio=0.60)

            rows.append(
                {
                    "class": class_name,
                    "group_name": group_name,
                    "target_n": target_n,
                    "available_historical_qa_cnt": available_n,
                    "reserved_for_training_n": int(np.ceil(available_n * 0.60)) if available_n < target_n else 0,
                    "final_target_n": final_n,
                }
            )

            if final_n <= 0:
                continue

            sampled.append(
                recency_weighted_sample(
                    df=sub,
                    n=final_n,
                    date_col=qa_date_col,
                    recency_strength=recency_strength,
                    random_seed=random_seed,
                )
            )

    allocation_df = pd.DataFrame(rows)
    fixed_eval_df = qa.iloc[0:0].copy() if not sampled else pd.concat(sampled).reset_index(drop=True)
    fixed_eval_df["eval_set_type"] = "fixed_benchmark"
    dropped_df = allocation_df[allocation_df["final_target_n"] < allocation_df["target_n"]].copy()
    return fixed_eval_df, allocation_df, dropped_df


def compare_benchmark_history(
    current_df: pd.DataFrame,
    previous_df: pd.DataFrame | None,
    current_class_col: str,
    previous_class_cols: list[str],
) -> pd.DataFrame:
    if previous_df is None or previous_df.empty:
        return pd.DataFrame(columns=[current_class_col, "current_sample_size", "previous_sample_size", "delta"])

    previous_class_col = next((col for col in previous_class_cols if col in previous_df.columns), None)
    if previous_class_col is None:
        return pd.DataFrame(columns=[current_class_col, "current_sample_size", "previous_sample_size", "delta"])

    current_counts = (
        current_df.groupby(current_class_col).size().reset_index(name="current_sample_size")
        if not current_df.empty
        else pd.DataFrame(columns=[current_class_col, "current_sample_size"])
    )
    previous_counts = (
        previous_df.groupby(previous_class_col)
        .size()
        .reset_index(name="previous_sample_size")
        .rename(columns={previous_class_col: current_class_col})
    )
    merged = current_counts.merge(previous_counts, on=current_class_col, how="outer").fillna(0)
    merged["delta"] = merged["current_sample_size"] - merged["previous_sample_size"]
    return merged.sort_values(current_class_col).reset_index(drop=True)


def run_monthly_benchmark_refresh(
    qa_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    config: dict[str, Any],
    historical_benchmark_df: pd.DataFrame | None = None,
    benchmark_start_date: str | None = None,
    benchmark_end_date: str | None = None,
    medium_cum_threshold: float = 0.60,
    fixed_p0p1_size: int = 450,
    fixed_medium_size: int = 80,
    fixed_longtail_size: int = 25,
    fixed_recency_strength: float = 0.8,
    refresh_pct: float = 5.0,
    random_seed: int = 42,
) -> dict[str, Any]:
    columns = config["columns"]
    p0_p1_classes = config.get("business_rules", {}).get("p0_p1_classes", [])
    eval_source_df = filter_non_training_rows(qa_df, columns.get("is_training"))
    logger = RunLogger("benchmark_refresh")
    logger.log_params(
        {
            "benchmark_start_date": benchmark_start_date,
            "benchmark_end_date": benchmark_end_date,
            "medium_cum_threshold": medium_cum_threshold,
            "fixed_p0p1_size": fixed_p0p1_size,
            "fixed_medium_size": fixed_medium_size,
            "fixed_longtail_size": fixed_longtail_size,
            "fixed_recency_strength": fixed_recency_strength,
            "refresh_pct": refresh_pct,
            "random_seed": random_seed,
            "eval_source_size": len(eval_source_df),
        }
    )

    class_groups = define_medium_longtail_classes(
        dist_df=benchmark_df,
        dist_class_col=columns["dist_class"],
        proportion_col=columns["dist_weight"],
        p0_p1_classes=p0_p1_classes,
        medium_cum_threshold=medium_cum_threshold,
    )

    new_fixed_eval_df, class_summary_df, dropped_classes_df = build_fixed_benchmark_set(
        qa_df=eval_source_df,
        qa_date_col=columns["qa_date"],
        qa_class_col=columns["qa_class"],
        p0_p1_classes=p0_p1_classes,
        medium_classes=class_groups["medium_classes"],
        longtail_classes=class_groups["longtail_classes"],
        sample_sizes={
            "p0p1": fixed_p0p1_size,
            "medium": fixed_medium_size,
            "longtail": fixed_longtail_size,
        },
        recency_strength=fixed_recency_strength,
        benchmark_start_date=benchmark_start_date,
        benchmark_end_date=benchmark_end_date,
        random_seed=random_seed,
    )

    benchmark_eval_df = new_fixed_eval_df.copy()
    refresh_summary = {
        "historical_rows": 0,
        "removed_rows": 0,
        "added_rows": len(new_fixed_eval_df),
        "final_rows": len(new_fixed_eval_df),
    }
    protected_item_ids: set[str] = set()
    if historical_benchmark_df is not None and not historical_benchmark_df.empty:
        if columns["item_id"] in historical_benchmark_df.columns:
            protected_item_ids = set(historical_benchmark_df[columns["item_id"]].astype(str).tolist())
        historical_class_col = _resolve_available_class_col(
            historical_benchmark_df,
            [columns["qa_class"], columns.get("benchmark_class", ""), columns.get("dist_class", "")],
        )
        benchmark_eval_df, refresh_summary = _refresh_historical_fixed_eval(
            historical_df=historical_benchmark_df,
            candidate_df=new_fixed_eval_df,
            item_id_col=columns["item_id"],
            qa_date_col=columns["qa_date"],
            historical_class_col=historical_class_col or columns["qa_class"],
            candidate_class_col=columns["qa_class"],
            refresh_pct=refresh_pct,
        )
        logger.info(f"Applied fixed eval refresh. Summary: {refresh_summary}")

    benchmark_eval_df = _apply_fixed_eval_class_caps(
        df=benchmark_eval_df,
        class_col=columns["qa_class"],
        qa_date_col=columns["qa_date"],
        p0_p1_classes=p0_p1_classes,
        medium_classes=class_groups["medium_classes"],
        longtail_classes=class_groups["longtail_classes"],
        class_caps={
            "p0p1": fixed_p0p1_size,
            "medium": fixed_medium_size,
            "longtail": fixed_longtail_size,
        },
        protected_item_ids=protected_item_ids,
        item_id_col=columns["item_id"],
    )
    qa_tier1_col = _resolve_available_column(
        qa_df,
        [
            "taxonomy5_tier1_top1_qa",
            "taxonomy_tier1_top1_qa",
            "taxonomy5_tier1_qa",
            "taxonomy_tier1_qa",
        ],
    )
    benchmark_eval_df = _overlay_original_qa_fields(
        eval_df=benchmark_eval_df,
        qa_df=qa_df,
        item_id_col=columns["item_id"],
        qa_date_col=columns["qa_date"],
        qa_tier3_col=columns["qa_class"],
        qa_tier1_col=qa_tier1_col,
    )

    history_comparison_df = compare_benchmark_history(
        current_df=benchmark_eval_df,
        previous_df=historical_benchmark_df,
        current_class_col=columns["qa_class"],
        previous_class_cols=[
            columns["qa_class"],
            columns.get("benchmark_class", ""),
            columns.get("dist_class", ""),
        ],
    )

    if not dropped_classes_df.empty:
        logger.warning(f"{len(dropped_classes_df)} classes were dropped due to insufficient QA data.")
    if historical_benchmark_df is not None and not historical_benchmark_df.empty and history_comparison_df.empty:
        logger.warning(
            "Historical fixed eval was uploaded, but no compatible class column was found for comparison."
        )

    benchmark_class_groups_df = pd.concat(
        [
            pd.DataFrame({"class": p0_p1_classes, "group_name": "p0p1"}),
            pd.DataFrame({"class": class_groups["medium_classes"], "group_name": "medium"}),
            pd.DataFrame({"class": class_groups["longtail_classes"], "group_name": "longtail"}),
        ],
        ignore_index=True,
    )

    summary = {
        "p0p1_class_count": len(p0_p1_classes),
        "medium_class_count": len(class_groups["medium_classes"]),
        "longtail_class_count": len(class_groups["longtail_classes"]),
        "benchmark_eval_size": len(benchmark_eval_df),
        "classes_dropped": len(dropped_classes_df),
        "fixed_eval_refresh_pct": refresh_pct,
        "fixed_eval_removed_rows": refresh_summary["removed_rows"],
        "fixed_eval_added_rows": refresh_summary["added_rows"],
    }

    return {
        "bottom_classes_df": benchmark_class_groups_df,
        "eval_source_df": eval_source_df,
        "new_fixed_eval_df": new_fixed_eval_df,
        "benchmark_eval_df": benchmark_eval_df,
        "class_summary_df": class_summary_df,
        "mom_comparison_df": history_comparison_df,
        "dropped_classes_df": dropped_classes_df,
        "summary": summary,
        "logger": logger,
    }
