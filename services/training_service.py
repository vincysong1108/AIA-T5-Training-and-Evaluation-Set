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
from utils.data_loader import ensure_unique_columns
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


def _resolve_filter_token_to_column(df: pd.DataFrame, config: dict[str, Any], token: str) -> str | None:
    stripped = token.strip()
    if not stripped:
        return None

    columns_cfg = config.get("columns", {})
    if stripped in columns_cfg and columns_cfg[stripped] in df.columns:
        return columns_cfg[stripped]
    if stripped in df.columns:
        return stripped
    return None


def _parse_filter_literal(token: str) -> Any:
    stripped = token.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1]

    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(stripped)
    except ValueError:
        pass

    try:
        return float(stripped)
    except ValueError:
        pass

    return stripped


def _normalized_compare_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _build_filter_mask(
    df: pd.DataFrame,
    config: dict[str, Any],
    filter_expression: str,
) -> tuple[pd.Series, str]:
    expression = (filter_expression or "").strip()
    if not expression:
        expression = "qa_class = model_class"

    normalized_expression = expression.replace("==", "=")
    if "=" not in normalized_expression:
        raise ValueError("Stable core filter must use '='. Examples: 'qa_class = model_class' or 'consistent = 1'.")

    left_token, right_token = [part.strip() for part in normalized_expression.split("=", 1)]
    if not left_token or not right_token:
        raise ValueError("Stable core filter must include both left and right operands.")

    left_col = _resolve_filter_token_to_column(df, config, left_token)
    if left_col is None:
        raise ValueError(f"Stable core filter left operand '{left_token}' was not found in the training pool.")

    right_col = _resolve_filter_token_to_column(df, config, right_token)
    left_series = df[left_col]
    left_norm = _normalized_compare_series(left_series)

    if right_col is not None:
        right_series = df[right_col]
        mask = left_series.eq(right_series) | (
            left_series.notna()
            & right_series.notna()
            & left_norm.eq(_normalized_compare_series(right_series))
        )
        resolved_expression = f"{left_col} = {right_col}"
        return mask.fillna(False), resolved_expression

    literal = _parse_filter_literal(right_token)
    literal_norm = str(literal).strip().lower()
    mask = left_series.eq(literal) | (left_series.notna() & left_norm.eq(literal_norm))
    resolved_expression = f"{left_col} = {literal!r}"
    return mask.fillna(False), resolved_expression


def build_stable_core_source_pool(
    train_pool_df: pd.DataFrame,
    item_id_col: str,
    qa_class_col: str,
    config: dict[str, Any],
    filter_expression: str,
) -> tuple[pd.DataFrame, str]:
    data = train_pool_df.copy()
    data[item_id_col] = data[item_id_col].astype(str)
    data = standardize_str_col(data, qa_class_col)
    filter_mask, resolved_expression = _build_filter_mask(
        df=data,
        config=config,
        filter_expression=filter_expression,
    )

    out = data[filter_mask].copy()
    out = out[out[qa_class_col].notna()].copy()
    out = out[out[qa_class_col] != ""].copy()
    out = out[out[qa_class_col].str.lower() != "nan"].copy()
    out["sample_source"] = "stable_core_filter_match"
    out["library_type"] = "stable_core"
    return out.reset_index(drop=True), resolved_expression


def build_stable_core_library(
    stable_core_pool_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
    qa_class_col: str,
    dist_class_col: str,
    dist_weight_col: str,
    qa_date_col: str,
    p0_p1_classes: list[str],
    medium_classes: list[str],
    longtail_classes: list[str],
    total_size: int,
    tier_mix: dict[str, float] | None = None,
    min_per_class: dict[str, int] | None = None,
    recency_strength: float = 0.8,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if tier_mix is None:
        tier_mix = {"priority": 0.35, "medium": 0.45, "longtail": 0.20}
    if min_per_class is None:
        min_per_class = {"priority": 220, "medium": 55, "longtail": 25}

    pool_df = standardize_str_col(stable_core_pool_df.copy(), qa_class_col)
    pool_df = standardize_date_col(pool_df, qa_date_col)
    dist_df = standardize_str_col(distribution_df.copy(), dist_class_col)
    dist_df = dist_df.rename(columns={dist_class_col: qa_class_col}).copy()

    pool_df["training_tier"] = pool_df[qa_class_col].apply(
        lambda x: assign_training_tier(x, p0_p1_classes, medium_classes, longtail_classes)
    )
    dist_df["training_tier"] = dist_df[qa_class_col].apply(
        lambda x: assign_training_tier(x, p0_p1_classes, medium_classes, longtail_classes)
    )

    sampled: list[pd.DataFrame] = []
    alloc_rows: list[dict[str, Any]] = []

    for tier in ["priority", "medium", "longtail"]:
        tier_pool = pool_df[pool_df["training_tier"] == tier].copy()
        tier_dist = dist_df[dist_df["training_tier"] == tier].copy()
        if tier_pool.empty or tier_dist.empty or total_size <= 0:
            continue

        tier_budget = int(total_size * tier_mix[tier])
        available_classes = set(tier_pool[qa_class_col].unique())
        tier_dist = tier_dist[tier_dist[qa_class_col].isin(available_classes)].copy()
        if tier_dist.empty:
            continue

        tier_dist["tier_weight"] = tier_dist[dist_weight_col] / tier_dist[dist_weight_col].sum()
        tier_dist["target_n_raw"] = tier_dist["tier_weight"] * tier_budget
        tier_dist["target_n"] = tier_dist["target_n_raw"].round().astype(int)
        tier_dist["target_n"] = tier_dist["target_n"].clip(lower=min_per_class[tier])

        for _, row in tier_dist.iterrows():
            class_name = row[qa_class_col]
            target_n = int(row["target_n"])
            sub = tier_pool[tier_pool[qa_class_col] == class_name].copy()
            available_n = len(sub)
            final_n = min(target_n, available_n)

            alloc_rows.append(
                {
                    "library_type": "stable_core",
                    "training_tier": tier,
                    "class": class_name,
                    "distribution_weight": row[dist_weight_col],
                    "tier_weight": row["tier_weight"],
                    "target_n": target_n,
                    "available_n": available_n,
                    "final_n": final_n,
                }
            )

            if final_n <= 0:
                continue

            picked = recency_weighted_sample(
                df=sub,
                n=final_n,
                date_col=qa_date_col,
                recency_strength=recency_strength,
                random_seed=random_seed,
            )
            picked["library_type"] = "stable_core"
            sampled.append(picked)

    alloc_df = pd.DataFrame(alloc_rows)
    if not sampled:
        return pool_df.iloc[0:0].copy(), alloc_df
    stable_core_df = pd.concat(sampled, ignore_index=True, sort=False).reset_index(drop=True)
    if total_size > 0 and len(stable_core_df) > total_size:
        stable_core_df = recency_weighted_sample(
            df=stable_core_df,
            n=total_size,
            date_col=qa_date_col,
            recency_strength=recency_strength,
            random_seed=random_seed,
        ).reset_index(drop=True)
    return stable_core_df, alloc_df


def build_confusion_negative_set(
    train_pool_df: pd.DataFrame,
    qa_class_col: str,
    anchor_confusion_dict: dict[str, list[str]],
    max_per_anchor_confusion_class: int | None = 150,
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
            if max_per_anchor_confusion_class is None or len(sub) <= max_per_anchor_confusion_class:
                picked = sub
            else:
                picked = sub.sample(n=max_per_anchor_confusion_class, random_state=random_seed)
            picked["sample_source"] = "confusion_negative"
            picked["anchor_class"] = anchor_class
            picked["confusion_class"] = confusion_class
            out.append(picked)
    return df.iloc[0:0].copy() if not out else pd.concat(out).reset_index(drop=True)


def build_fp_hard_negative_set(
    train_pool_df: pd.DataFrame,
    qa_class_col: str,
    model_class_col: str | None,
    anchor_classes: list[str],
    max_per_anchor: int | None = 200,
    random_seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(random_seed)
    if not model_class_col or model_class_col not in train_pool_df.columns:
        return train_pool_df.iloc[0:0].copy()
    df = standardize_str_col(standardize_str_col(train_pool_df, qa_class_col), model_class_col)
    out = []
    for anchor in anchor_classes:
        sub = df[(df[model_class_col] == anchor) & (df[qa_class_col] != anchor)].copy()
        if sub.empty:
            continue
        if max_per_anchor is None or len(sub) <= max_per_anchor:
            picked = sub
        else:
            picked = sub.sample(n=max_per_anchor, random_state=random_seed)
        picked["sample_source"] = "fp_hard_negative"
        picked["anchor_class"] = anchor
        picked["confusion_class"] = picked[qa_class_col]
        out.append(picked)
    return df.iloc[0:0].copy() if not out else pd.concat(out).reset_index(drop=True)


def build_disagreement_set(
    train_pool_df: pd.DataFrame,
    qa_class_col: str,
    model_class_col: str | None,
    max_total_size: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(random_seed)
    if not model_class_col or model_class_col not in train_pool_df.columns:
        return train_pool_df.iloc[0:0].copy()
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
    model_class_col: str | None,
    anchor_confusion_dict: dict[str, list[str]],
    p0_p1_classes: list[str],
    total_size: int = 9000,
    max_confusion_per_pair: int | None = 150,
    max_fp_per_anchor: int | None = 200,
    max_disagreement_total: int | None = 3000,
    is_vague_col: str | None = None,
    is_appeal_success_col: str | None = None,
    max_edge_case_total: int | None = 2000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, int]]:
    confusion_neg_df = (
        build_confusion_negative_set(
            train_pool_df=train_pool_df,
            qa_class_col=qa_class_col,
            anchor_confusion_dict=anchor_confusion_dict,
            max_per_anchor_confusion_class=max_confusion_per_pair,
            random_seed=random_seed,
        )
        if anchor_confusion_dict
        else train_pool_df.iloc[0:0].copy()
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


def merge_training_library_parts(
    stable_core_df: pd.DataFrame,
    hard_case_df: pd.DataFrame,
    item_id_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = pd.concat([stable_core_df, hard_case_df], ignore_index=True, sort=False)
    if merged.empty:
        return merged.copy(), pd.DataFrame(columns=[item_id_col, "sample_source_joined", "library_type_joined"])

    merged[item_id_col] = merged[item_id_col].astype(str)
    source_summary_df = (
        merged.groupby(item_id_col)
        .agg(
            {
                "sample_source": lambda x: sorted(set(i for i in x.dropna().astype(str))),
                "library_type": lambda x: sorted(set(i for i in x.dropna().astype(str))),
            }
        )
        .reset_index()
    )
    source_summary_df["sample_source_joined"] = source_summary_df["sample_source"].apply(lambda x: "|".join(x))
    source_summary_df["library_type_joined"] = source_summary_df["library_type"].apply(lambda x: "|".join(x))

    final_training_df = merged.drop_duplicates(subset=[item_id_col]).copy()
    final_training_df = final_training_df.drop(columns=["sample_source", "library_type"], errors="ignore")
    final_training_df = final_training_df.merge(
        source_summary_df[[item_id_col, "sample_source_joined", "library_type_joined"]],
        on=item_id_col,
        how="left",
    )
    return final_training_df.reset_index(drop=True), source_summary_df.reset_index(drop=True)


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
    max_confusion_per_pair: int | None = 150,
    max_fp_per_anchor: int | None = 200,
    max_disagreement_total: int | None = 3000,
    max_edge_case_total: int | None = 2000,
    bootstrap_when_no_history: bool = False,
    target_total_training_size: int = 30000,
    stable_core_filter_expression: str = "qa_class = model_class",
    random_seed: int = 42,
    exclude_eval_overlap: bool = True,
    dedup_by_item_id: bool = True,
) -> dict[str, Any]:
    qa_df, qa_duplicate_columns = ensure_unique_columns(qa_df)
    distribution_df, _ = ensure_unique_columns(distribution_df)
    if historical_training_df is not None:
        historical_training_df, _ = ensure_unique_columns(historical_training_df)
    if rolling_eval_df is not None:
        rolling_eval_df, _ = ensure_unique_columns(rolling_eval_df)
    if benchmark_eval_df is not None:
        benchmark_eval_df, _ = ensure_unique_columns(benchmark_eval_df)
    columns = config["columns"]
    model_class_col = (columns.get("model_class") or "").strip() or None
    labeler_class_col = (columns.get("labeler_class") or "").strip() or None
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
            "bootstrap_when_no_history": bootstrap_when_no_history,
            "target_total_training_size": target_total_training_size,
            "stable_core_filter_expression": stable_core_filter_expression,
            "model_class_enabled": bool(model_class_col),
            "labeler_class_enabled": bool(labeler_class_col),
            "random_seed": random_seed,
            "exclude_eval_overlap": exclude_eval_overlap,
            "dedup_by_item_id": dedup_by_item_id,
        }
    )
    if qa_duplicate_columns:
        logger.warning(
            "Duplicate QA columns were detected and auto-renamed on load/runtime: "
            + ", ".join(sorted(set(qa_duplicate_columns)))
        )
    if model_class_col and columns["qa_class"] == model_class_col:
        raise ValueError(
            f"Training update requires different config values for QA class column and Model class column, "
            f"but both are currently set to '{columns['qa_class']}'. "
            f"Update the Shared Config on the home page before running training update."
        )

    confusion_source_df = _filter_eval_only_rows(qa_df, columns.get("is_training"))
    confusion_result: dict[str, Any] = {"anchor_confusion_dict": {}, "selected_pairs_df": pd.DataFrame()}
    top_confusion_pairs_df = pd.DataFrame()
    if model_class_col and model_class_col in confusion_source_df.columns:
        confusion_result = run_confusion_pair_pipeline(
            df=confusion_source_df,
            gt_col=columns["qa_class"],
            pred_col=model_class_col,
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
    else:
        logger.info("Model class column is blank or unavailable, so confusion-pair, FP, and disagreement hard-case logic was skipped.")

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
        model_class_col=model_class_col,
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

    stable_core_pool_df = qa_df.iloc[0:0].copy()
    stable_core_df = qa_df.iloc[0:0].copy()
    stable_alloc_df = pd.DataFrame()
    source_summary_df = pd.DataFrame()
    selected_rows_df = hard_case_df.copy()
    new_hard_cases_df = hard_case_df.copy()
    resolved_stable_core_filter_expression = ""

    should_bootstrap = bootstrap_when_no_history and (historical_training_df is None or historical_training_df.empty)
    if bootstrap_when_no_history and not should_bootstrap:
        logger.warning("Fresh bootstrap mode was selected, but a historical training library is already loaded. Falling back to append mode.")

    if should_bootstrap:
        class_groups = define_medium_longtail_classes(
            dist_df=distribution_df,
            dist_class_col=columns["dist_class"],
            proportion_col=columns["dist_weight"],
            p0_p1_classes=p0_p1_classes,
            medium_cum_threshold=config["defaults"]["benchmark_refresh"]["medium_cum_threshold"],
        )
        stable_core_pool_df, resolved_stable_core_filter_expression = build_stable_core_source_pool(
            train_pool_df=train_pool_df,
            item_id_col=columns["item_id"],
            qa_class_col=columns["qa_class"],
            config=config,
            filter_expression=stable_core_filter_expression,
        )
        stable_total_size = max(target_total_training_size - len(hard_case_df), 0)
        stable_core_df, stable_alloc_df = build_stable_core_library(
            stable_core_pool_df=stable_core_pool_df,
            distribution_df=distribution_df,
            qa_class_col=columns["qa_class"],
            dist_class_col=columns["dist_class"],
            dist_weight_col=columns["dist_weight"],
            qa_date_col=columns["qa_date"],
            p0_p1_classes=p0_p1_classes,
            medium_classes=class_groups["medium_classes"],
            longtail_classes=class_groups["longtail_classes"],
            total_size=stable_total_size,
            random_seed=random_seed,
        )
        updated_training_library_df, source_summary_df = merge_training_library_parts(
            stable_core_df=stable_core_df,
            hard_case_df=hard_case_df,
            item_id_col=columns["item_id"],
        )
        if len(updated_training_library_df) < target_total_training_size:
            missing_n = target_total_training_size - len(updated_training_library_df)
            selected_item_ids = set(updated_training_library_df[columns["item_id"]].astype(str))
            stable_backfill_pool_df = stable_core_pool_df[
                ~stable_core_pool_df[columns["item_id"]].astype(str).isin(selected_item_ids)
            ].copy()
            if not stable_backfill_pool_df.empty:
                stable_backfill_df = recency_weighted_sample(
                    df=stable_backfill_pool_df,
                    n=min(missing_n, len(stable_backfill_pool_df)),
                    date_col=columns["qa_date"],
                    recency_strength=0.8,
                    random_seed=random_seed,
                )
                stable_backfill_df["library_type"] = "stable_core"
                stable_core_df = pd.concat([stable_core_df, stable_backfill_df], ignore_index=True, sort=False)
                updated_training_library_df, source_summary_df = merge_training_library_parts(
                    stable_core_df=stable_core_df,
                    hard_case_df=hard_case_df,
                    item_id_col=columns["item_id"],
                )
        selected_rows_df = pd.concat([stable_core_df, hard_case_df], ignore_index=True, sort=False)
        dedup_summary = {
            "history_rows": 0,
            "new_rows": len(selected_rows_df),
            "deduped_new_rows": len(updated_training_library_df),
            "final_rows": len(updated_training_library_df),
        }
        if len(updated_training_library_df) < target_total_training_size:
            logger.warning("Fresh bootstrap mode could not fully reach the target training size with the available stable-core pool after dedup.")
    else:
        if historical_training_df is None:
            historical_training_df = qa_df.iloc[0:0].copy()

        if dedup_by_item_id and not historical_training_df.empty:
            new_hard_cases_df = exclude_existing_ids(
                new_hard_cases_df,
                set(historical_training_df[columns["item_id"]].astype(str)),
                columns["item_id"],
            )
        selected_rows_df = new_hard_cases_df.copy()
        updated_training_library_df, dedup_summary = append_deduplicated_history(
            history_df=historical_training_df,
            new_df=new_hard_cases_df,
            item_id_col=columns["item_id"],
        )

    count_added_by_class_df = (
        selected_rows_df.groupby(columns["qa_class"])
        .size()
        .reset_index(name="added_count")
        .sort_values("added_count", ascending=False)
        .reset_index(drop=True)
        if not selected_rows_df.empty
        else pd.DataFrame(columns=[columns["qa_class"], "added_count"])
    )

    count_added_by_source_date_df = _count_added_by_source_date(
        selected_rows_df,
        "sample_source",
        columns["qa_date"],
    )

    dedup_summary_df = pd.DataFrame([dedup_summary])
    summary = {
        "training_update_mode": "bootstrap_no_history" if should_bootstrap else "append_hard_cases",
        "confusion_source_size": len(confusion_source_df),
        "train_pool_size": len(train_pool_df),
        "new_hard_cases_added": len(new_hard_cases_df),
        "updated_training_library_size": len(updated_training_library_df),
    }
    summary.update(hard_case_summary)
    if should_bootstrap:
        summary["target_total_training_size"] = int(target_total_training_size)
        summary["stable_core_target_size"] = int(max(target_total_training_size - len(hard_case_df), 0))
        summary["stable_core_pool_size"] = int(len(stable_core_pool_df))
        summary["stable_core_library_size"] = int(len(stable_core_df))
        summary["stable_core_filter"] = resolved_stable_core_filter_expression or stable_core_filter_expression

    if len(new_hard_cases_df) < len(hard_case_df):
        logger.warning("Some hard cases were removed during append-mode dedup.")

    return {
        "train_pool_df": train_pool_df,
        "stable_core_pool_df": stable_core_pool_df,
        "stable_core_df": stable_core_df,
        "stable_alloc_df": stable_alloc_df,
        "new_hard_cases_df": new_hard_cases_df,
        "updated_training_library_df": updated_training_library_df,
        "source_summary_df": source_summary_df,
        "top_confusion_pairs_df": top_confusion_pairs_df,
        "dedup_summary_df": dedup_summary_df,
        "count_added_by_class_df": count_added_by_class_df,
        "count_added_by_source_date_df": count_added_by_source_date_df,
        "summary": summary,
        "logger": logger,
    }
