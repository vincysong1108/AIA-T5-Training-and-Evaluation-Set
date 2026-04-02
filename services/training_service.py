from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from services.rolling_eval_service import (
    define_medium_longtail_classes,
    run_confusion_pair_pipeline,
    standardize_date_col,
    standardize_str_col,
)
from utils.dedup import append_deduplicated_history, exclude_existing_ids
from utils.logger import RunLogger
from utils.sampler import recency_weighted_sample


def _filter_eval_only_rows(df: pd.DataFrame, is_training_col: str | None) -> pd.DataFrame:
    if not is_training_col or is_training_col not in df.columns:
        return df.copy()

    out = df.copy()
    numeric_flag = pd.to_numeric(out[is_training_col], errors="coerce")
    if numeric_flag.notna().any():
        return out[numeric_flag.fillna(1) == 0].copy()

    text_flag = out[is_training_col].astype(str).str.strip().str.lower()
    return out[text_flag.isin({"0", "false", "no"})].copy()


def assign_training_tier(class_name: str, p0_p1_classes: list[str], medium_classes: list[str], longtail_classes: list[str]) -> str:
    if class_name in set(p0_p1_classes):
        return "priority"
    if class_name in set(medium_classes):
        return "medium"
    if class_name in set(longtail_classes):
        return "longtail"
    return "other"


def build_training_candidate_pool(
    qa_df: pd.DataFrame,
    item_id_col: str,
    qa_date_col: str,
    qa_class_col: str,
    rolling_eval_df: pd.DataFrame | None = None,
    benchmark_eval_df: pd.DataFrame | None = None,
    train_start_date: str | None = None,
    train_end_date: str | None = None,
) -> pd.DataFrame:
    qa = qa_df.copy()
    qa[item_id_col] = qa[item_id_col].astype(str)
    qa = standardize_str_col(qa, qa_class_col)
    qa = standardize_date_col(qa, qa_date_col)

    if train_start_date is not None:
        qa = qa[qa[qa_date_col] >= pd.to_datetime(train_start_date)].copy()
    if train_end_date is not None:
        qa = qa[qa[qa_date_col] <= pd.to_datetime(train_end_date)].copy()

    eval_ids: set[str] = set()
    if rolling_eval_df is not None and not rolling_eval_df.empty:
        eval_ids.update(rolling_eval_df[item_id_col].astype(str).tolist())
    if benchmark_eval_df is not None and not benchmark_eval_df.empty:
        eval_ids.update(benchmark_eval_df[item_id_col].astype(str).tolist())

    train_pool = qa[~qa[item_id_col].isin(eval_ids)].copy()
    train_pool = train_pool[train_pool[qa_class_col].notna()].copy()
    train_pool = train_pool[train_pool[qa_class_col] != ""].copy()
    train_pool = train_pool[train_pool[qa_class_col].str.lower() != "nan"].copy()
    return train_pool.reset_index(drop=True)


def build_confusion_negative_set(
    train_pool_df: pd.DataFrame,
    qa_class_col: str,
    anchor_confusion_dict: dict[str, list[str]],
    max_per_anchor_confusion_class: int = 150,
    random_seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(random_seed)
    df = standardize_str_col(train_pool_df, qa_class_col)
    out = []
    for anchor_class, confusion_classes in anchor_confusion_dict.items():
        for confusion_class in confusion_classes:
            sub = df[df[qa_class_col] == confusion_class].copy()
            if sub.empty:
                continue
            picked = sub if len(sub) <= max_per_anchor_confusion_class else sub.sample(n=max_per_anchor_confusion_class, random_state=random_seed)
            picked["sample_source"] = "confusion_negative"
            picked["anchor_class"] = anchor_class
            picked["confusion_class"] = confusion_class
            out.append(picked)
    return df.iloc[0:0].copy() if not out else pd.concat(out).reset_index(drop=True)


def build_fp_hard_negative_set(
    train_pool_df: pd.DataFrame,
    qa_class_col: str,
    model_class_col: str,
    anchor_classes: list[str],
    max_per_anchor: int = 200,
    random_seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(random_seed)
    df = standardize_str_col(standardize_str_col(train_pool_df, qa_class_col), model_class_col)
    out = []
    for anchor in anchor_classes:
        sub = df[(df[model_class_col] == anchor) & (df[qa_class_col] != anchor)].copy()
        if sub.empty:
            continue
        picked = sub if len(sub) <= max_per_anchor else sub.sample(n=max_per_anchor, random_state=random_seed)
        picked["sample_source"] = "fp_hard_negative"
        picked["anchor_class"] = anchor
        picked["confusion_class"] = picked[qa_class_col]
        out.append(picked)
    return df.iloc[0:0].copy() if not out else pd.concat(out).reset_index(drop=True)


def build_disagreement_set(
    train_pool_df: pd.DataFrame,
    qa_class_col: str,
    model_class_col: str,
    max_total_size: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(random_seed)
    df = standardize_str_col(standardize_str_col(train_pool_df, qa_class_col), model_class_col)
    out = df[df[qa_class_col] != df[model_class_col]].copy()
    out["sample_source"] = "disagreement_case"
    if max_total_size is not None and len(out) > max_total_size:
        out = out.sample(n=max_total_size, random_state=random_seed)
    return out.reset_index(drop=True)


def build_edge_case_set(
    train_pool_df: pd.DataFrame,
    is_vague_col: str | None = None,
    is_appeal_success_col: str | None = None,
    max_total_size: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(random_seed)
    df = train_pool_df.copy()
    mask = pd.Series(False, index=df.index)
    if is_vague_col and is_vague_col in df.columns:
        mask = mask | (df[is_vague_col] == 1)
    if is_appeal_success_col and is_appeal_success_col in df.columns:
        mask = mask | (df[is_appeal_success_col] == 0)
    out = df[mask].copy()
    out["sample_source"] = "edge_case"
    if max_total_size is not None and len(out) > max_total_size:
        out = out.sample(n=max_total_size, random_state=random_seed)
    return out.reset_index(drop=True)


def build_hard_case_library(
    train_pool_df: pd.DataFrame,
    qa_class_col: str,
    model_class_col: str,
    anchor_confusion_dict: dict[str, list[str]],
    p0_p1_classes: list[str],
    total_size: int = 9000,
    max_confusion_per_pair: int = 150,
    max_fp_per_anchor: int = 200,
    max_disagreement_total: int = 3000,
    is_vague_col: str | None = None,
    is_appeal_success_col: str | None = None,
    max_edge_case_total: int = 2000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, int]]:
    confusion_neg_df = build_confusion_negative_set(
        train_pool_df=train_pool_df,
        qa_class_col=qa_class_col,
        anchor_confusion_dict=anchor_confusion_dict,
        max_per_anchor_confusion_class=max_confusion_per_pair,
        random_seed=random_seed,
    )
    fp_hard_neg_df = build_fp_hard_negative_set(
        train_pool_df=train_pool_df,
        qa_class_col=qa_class_col,
        model_class_col=model_class_col,
        anchor_classes=p0_p1_classes,
        max_per_anchor=max_fp_per_anchor,
        random_seed=random_seed,
    )
    disagreement_df = build_disagreement_set(
        train_pool_df=train_pool_df,
        qa_class_col=qa_class_col,
        model_class_col=model_class_col,
        max_total_size=max_disagreement_total,
        random_seed=random_seed,
    )
    edge_case_df = build_edge_case_set(
        train_pool_df=train_pool_df,
        is_vague_col=is_vague_col,
        is_appeal_success_col=is_appeal_success_col,
        max_total_size=max_edge_case_total,
        random_seed=random_seed,
    )

    hard_df = pd.concat([confusion_neg_df, fp_hard_neg_df, disagreement_df, edge_case_df], ignore_index=True, sort=False)
    if not hard_df.empty:
        hard_df = hard_df.drop_duplicates()
    if len(hard_df) > total_size:
        hard_df = hard_df.sample(n=total_size, random_state=random_seed)
    hard_df["library_type"] = "hard_case"

    summary = {
        "confusion_negative_size": len(confusion_neg_df),
        "fp_hard_negative_size": len(fp_hard_neg_df),
        "disagreement_size": len(disagreement_df),
        "edge_case_size": len(edge_case_df),
        "final_hard_case_size": len(hard_df),
    }
    return hard_df.reset_index(drop=True), summary


def _count_added_by_source_date(new_df: pd.DataFrame, sample_source_col: str, date_col: str) -> pd.DataFrame:
    if new_df.empty or sample_source_col not in new_df.columns or date_col not in new_df.columns:
        return pd.DataFrame(columns=[sample_source_col, date_col, "item_count"])
    out = (
        new_df.groupby([sample_source_col, date_col])
        .size()
        .reset_index(name="item_count")
        .sort_values("item_count", ascending=False)
        .reset_index(drop=True)
    )
    return out


def run_training_library_update(
    qa_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
    config: dict[str, Any],
    historical_training_df: pd.DataFrame | None = None,
    rolling_eval_df: pd.DataFrame | None = None,
    benchmark_eval_df: pd.DataFrame | None = None,
    train_start_date: str | None = None,
    train_end_date: str | None = None,
    hard_total_size: int = 9000,
    max_confusion_per_pair: int = 150,
    max_fp_per_anchor: int = 200,
    max_disagreement_total: int = 3000,
    max_edge_case_total: int = 2000,
    random_seed: int = 42,
    exclude_eval_overlap: bool = True,
    dedup_by_item_id: bool = True,
) -> dict[str, Any]:
    columns = config["columns"]
    p0_p1_classes = config.get("business_rules", {}).get("p0_p1_classes", [])
    logger = RunLogger("training_update")
    logger.log_params(
        {
            "train_start_date": train_start_date,
            "train_end_date": train_end_date,
            "hard_total_size": hard_total_size,
            "max_confusion_per_pair": max_confusion_per_pair,
            "max_fp_per_anchor": max_fp_per_anchor,
            "max_disagreement_total": max_disagreement_total,
            "max_edge_case_total": max_edge_case_total,
            "random_seed": random_seed,
            "exclude_eval_overlap": exclude_eval_overlap,
            "dedup_by_item_id": dedup_by_item_id,
        }
    )

    confusion_source_df = _filter_eval_only_rows(qa_df, columns.get("is_training"))
    confusion_result = run_confusion_pair_pipeline(
        df=confusion_source_df,
        gt_col=columns["qa_class"],
        pred_col=columns["model_class"],
        anchor_classes=p0_p1_classes,
        coverage_threshold=config["defaults"]["rolling_eval"]["coverage_threshold"],
        min_confusion_rate=config["defaults"]["rolling_eval"]["min_confusion_rate"],
        min_pair_confusion_cnt=config["defaults"]["rolling_eval"]["min_pair_confusion_cnt"],
    )
    top_confusion_pairs_df = (
        confusion_result["selected_pairs_df"].head(50).reset_index(drop=True)
        if "selected_pairs_df" in confusion_result
        else pd.DataFrame()
    )

    train_pool_df = build_training_candidate_pool(
        qa_df=qa_df,
        item_id_col=columns["item_id"],
        qa_date_col=columns["qa_date"],
        qa_class_col=columns["qa_class"],
        rolling_eval_df=rolling_eval_df if exclude_eval_overlap else None,
        benchmark_eval_df=benchmark_eval_df if exclude_eval_overlap else None,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
    )

    hard_case_df, hard_case_summary = build_hard_case_library(
        train_pool_df=train_pool_df,
        qa_class_col=columns["qa_class"],
        model_class_col=columns["model_class"],
        anchor_confusion_dict=confusion_result["anchor_confusion_dict"],
        p0_p1_classes=p0_p1_classes,
        total_size=hard_total_size,
        max_confusion_per_pair=max_confusion_per_pair,
        max_fp_per_anchor=max_fp_per_anchor,
        max_disagreement_total=max_disagreement_total,
        is_vague_col=columns.get("is_vague"),
        is_appeal_success_col=columns.get("is_appeal_success"),
        max_edge_case_total=max_edge_case_total,
        random_seed=random_seed,
    )

    new_hard_cases_df = hard_case_df.copy()
    if historical_training_df is None:
        historical_training_df = qa_df.iloc[0:0].copy()

    if dedup_by_item_id and not historical_training_df.empty:
        new_hard_cases_df = exclude_existing_ids(
            new_hard_cases_df,
            set(historical_training_df[columns["item_id"]].astype(str)),
            columns["item_id"],
        )

    updated_training_library_df, dedup_summary = append_deduplicated_history(
        history_df=historical_training_df,
        new_df=new_hard_cases_df,
        item_id_col=columns["item_id"],
    )

    count_added_by_class_df = (
        new_hard_cases_df.groupby(columns["qa_class"])
        .size()
        .reset_index(name="added_count")
        .sort_values("added_count", ascending=False)
        .reset_index(drop=True)
        if not new_hard_cases_df.empty
        else pd.DataFrame(columns=[columns["qa_class"], "added_count"])
    )

    count_added_by_source_date_df = _count_added_by_source_date(
        new_hard_cases_df,
        "sample_source",
        columns["qa_date"],
    )

    dedup_summary_df = pd.DataFrame([dedup_summary])
    summary = {
        "confusion_source_size": len(confusion_source_df),
        "train_pool_size": len(train_pool_df),
        "new_hard_cases_added": len(new_hard_cases_df),
        "updated_training_library_size": len(updated_training_library_df),
    }
    summary.update(hard_case_summary)

    if len(new_hard_cases_df) < len(hard_case_df):
        logger.warning("Some hard cases were removed during append-mode dedup.")

    return {
        "train_pool_df": train_pool_df,
        "new_hard_cases_df": new_hard_cases_df,
        "updated_training_library_df": updated_training_library_df,
        "top_confusion_pairs_df": top_confusion_pairs_df,
        "dedup_summary_df": dedup_summary_df,
        "count_added_by_class_df": count_added_by_class_df,
        "count_added_by_source_date_df": count_added_by_source_date_df,
        "summary": summary,
        "logger": logger,
    }
