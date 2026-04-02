from __future__ import annotations

from typing import Any

import pandas as pd

from services.rolling_eval_service import (
    define_medium_longtail_classes,
    standardize_date_col,
    standardize_str_col,
)
from utils.dedup import append_deduplicated_history
from utils.logger import RunLogger
from utils.sampler import recency_weighted_sample


def _refresh_historical_fixed_eval(
    historical_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    item_id_col: str,
    qa_date_col: str,
    refresh_pct: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    history = historical_df.copy()
    candidates = candidate_df.copy()
    if history.empty:
        return candidate_df.copy(), {"historical_rows": 0, "removed_rows": 0, "added_rows": len(candidate_df), "final_rows": len(candidate_df)}

    if refresh_pct <= 0:
        preserved_history = history.drop_duplicates(subset=[item_id_col]).reset_index(drop=True)
        return preserved_history, {
            "historical_rows": len(history),
            "removed_rows": 0,
            "added_rows": 0,
            "final_rows": len(preserved_history),
        }

    history = standardize_date_col(history, qa_date_col)
    history[item_id_col] = history[item_id_col].astype(str)
    candidates[item_id_col] = candidates[item_id_col].astype(str)

    replace_n = max(1, int(len(history) * refresh_pct / 100.0))
    oldest_history = history.sort_values(qa_date_col, ascending=True)
    removed_ids = set(oldest_history.head(replace_n)[item_id_col].tolist())
    retained_history = history[~history[item_id_col].isin(removed_ids)].copy()

    replacement_pool = candidates[~candidates[item_id_col].isin(set(retained_history[item_id_col]))].copy()
    replacement_rows = replacement_pool.head(replace_n).copy()

    refreshed = pd.concat([retained_history, replacement_rows], ignore_index=True, sort=False)
    refreshed = refreshed.drop_duplicates(subset=[item_id_col]).reset_index(drop=True)
    return refreshed, {
        "historical_rows": len(history),
        "removed_rows": len(removed_ids),
        "added_rows": len(replacement_rows),
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
            final_n = min(target_n, available_n)

            rows.append(
                {
                    "class": class_name,
                    "group_name": group_name,
                    "target_n": target_n,
                    "available_historical_qa_cnt": available_n,
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
    class_col: str,
) -> pd.DataFrame:
    if previous_df is None or previous_df.empty or class_col not in previous_df.columns:
        return pd.DataFrame(columns=[class_col, "current_sample_size", "previous_sample_size", "delta"])

    current_counts = (
        current_df.groupby(class_col).size().reset_index(name="current_sample_size")
        if not current_df.empty
        else pd.DataFrame(columns=[class_col, "current_sample_size"])
    )
    previous_counts = previous_df.groupby(class_col).size().reset_index(name="previous_sample_size")
    merged = current_counts.merge(previous_counts, on=class_col, how="outer").fillna(0)
    merged["delta"] = merged["current_sample_size"] - merged["previous_sample_size"]
    return merged.sort_values(class_col).reset_index(drop=True)


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
        qa_df=qa_df,
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
    if historical_benchmark_df is not None and not historical_benchmark_df.empty:
        benchmark_eval_df, refresh_summary = _refresh_historical_fixed_eval(
            historical_df=historical_benchmark_df,
            candidate_df=new_fixed_eval_df,
            item_id_col=columns["item_id"],
            qa_date_col=columns["qa_date"],
            refresh_pct=refresh_pct,
        )
        logger.info(f"Applied fixed eval refresh. Summary: {refresh_summary}")

    history_comparison_df = compare_benchmark_history(
        current_df=benchmark_eval_df,
        previous_df=historical_benchmark_df,
        class_col=columns["qa_class"],
    )

    if not dropped_classes_df.empty:
        logger.warning(f"{len(dropped_classes_df)} classes were dropped due to insufficient QA data.")

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
        "new_fixed_eval_df": new_fixed_eval_df,
        "benchmark_eval_df": benchmark_eval_df,
        "class_summary_df": class_summary_df,
        "mom_comparison_df": history_comparison_df,
        "dropped_classes_df": dropped_classes_df,
        "summary": summary,
        "logger": logger,
    }
