from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from utils.dedup import append_deduplicated_history
from utils.logger import RunLogger
from utils.sampler import recency_weighted_sample


def _load_default_p0_p1_classes() -> list[str]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        return list(config.get("business_rules", {}).get("p0_p1_classes", []))
    except Exception:
        return []


def filter_non_training_rows(df: pd.DataFrame, is_training_col: str | None) -> pd.DataFrame:
    if not is_training_col or is_training_col not in df.columns:
        return df.copy()

    out = df.copy()
    numeric_flag = pd.to_numeric(out[is_training_col], errors="coerce")
    if numeric_flag.notna().any():
        return out[numeric_flag.fillna(1) == 0].copy()

    text_flag = out[is_training_col].astype(str).str.strip().str.lower()
    return out[text_flag.isin({"0", "false", "no"})].copy()


def standardize_str_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = df.copy()
    out[col] = out[col].astype(str).str.strip()
    return out


def standardize_date_col(df: pd.DataFrame, date_col: str, date_format: str = "%Y%m%d") -> pd.DataFrame:
    out = df.copy()
    raw = out[date_col]
    raw_str = raw.astype(str).str.strip()

    # Prefer explicit yyyyMMdd parsing for 8-digit numeric-like values so
    # values like 20260219 do not get interpreted as Unix-style timestamps.
    looks_like_compact_date = raw_str.str.fullmatch(r"\d{8}")
    parsed = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")

    if looks_like_compact_date.any():
        parsed.loc[looks_like_compact_date] = pd.to_datetime(
            raw_str.loc[looks_like_compact_date],
            format=date_format,
            errors="coerce",
        )

    remaining_mask = parsed.isna()
    if remaining_mask.any():
        parsed.loc[remaining_mask] = pd.to_datetime(
            raw_str.loc[remaining_mask],
            errors="coerce",
        )

    out[date_col] = parsed
    return out


def compute_pair_confusion_rate(df: pd.DataFrame, gt_col: str, pred_col: str) -> pd.DataFrame:
    data = df[[gt_col, pred_col]].dropna().copy()
    data[gt_col] = data[gt_col].astype(str)
    data[pred_col] = data[pred_col].astype(str)

    gt_counts = data.groupby(gt_col).size().to_dict()
    confusion_matrix = data.groupby([gt_col, pred_col]).size().to_dict()
    classes = list(gt_counts.keys())

    results: list[dict[str, Any]] = []
    for i, j in itertools.combinations(classes, 2):
        c_i_j = confusion_matrix.get((i, j), 0)
        c_j_i = confusion_matrix.get((j, i), 0)
        n_i = gt_counts.get(i, 0)
        n_j = gt_counts.get(j, 0)
        if n_i == 0 or n_j == 0:
            continue
        results.append(
            {
                "class_i": i,
                "class_j": j,
                "C_i_to_j": c_i_j,
                "C_j_to_i": c_j_i,
                "N_i": n_i,
                "N_j": n_j,
                "confusion_rate": 0.5 * ((c_i_j / n_i) + (c_j_i / n_j)),
            }
        )

    pair_df = pd.DataFrame(results)
    if pair_df.empty:
        return pair_df
    return pair_df.sort_values("confusion_rate", ascending=False).reset_index(drop=True)


def add_priority_score(pair_df: pd.DataFrame) -> pd.DataFrame:
    df = pair_df.copy()
    if df.empty:
        return df
    df["pair_confusion_cnt"] = df["C_i_to_j"] + df["C_j_to_i"]
    df["avg_pair_volume"] = (df["N_i"] + df["N_j"]) / 2
    df["priority_score"] = df["confusion_rate"] * df["avg_pair_volume"]
    return df


def filter_pairs_related_to_anchor_classes(pair_df: pd.DataFrame, anchor_classes: list[str]) -> pd.DataFrame:
    anchor_set = set(map(str, anchor_classes))
    df = pair_df.copy()
    if df.empty:
        return df
    return df[df["class_i"].isin(anchor_set) | df["class_j"].isin(anchor_set)].reset_index(drop=True)


def select_pairs_by_cumulative_coverage(
    pair_df: pd.DataFrame,
    coverage_threshold: float = 0.80,
    min_confusion_rate: float = 0.0,
    min_pair_confusion_cnt: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    df = pair_df.copy()
    if df.empty:
        summary = {
            "total_pair_count": 0,
            "selected_pair_count": 0,
            "total_priority_score": 0.0,
            "selected_priority_score": 0.0,
            "selected_coverage": 0.0,
        }
        return df, df, summary

    df = df[
        (df["confusion_rate"] >= min_confusion_rate)
        & (df["pair_confusion_cnt"] >= min_pair_confusion_cnt)
    ].copy()
    if df.empty:
        summary = {
            "total_pair_count": 0,
            "selected_pair_count": 0,
            "total_priority_score": 0.0,
            "selected_priority_score": 0.0,
            "selected_coverage": 0.0,
        }
        return df, df, summary

    df = df.sort_values(
        ["priority_score", "confusion_rate", "pair_confusion_cnt"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    total_priority_score = float(df["priority_score"].sum())
    if total_priority_score <= 0:
        df["priority_score_pct"] = 0.0
        df["cumulative_priority_score"] = 0.0
        df["cumulative_coverage"] = 0.0
        summary = {
            "total_pair_count": int(len(df)),
            "selected_pair_count": 0,
            "total_priority_score": total_priority_score,
            "selected_priority_score": 0.0,
            "selected_coverage": 0.0,
        }
        return df.iloc[0:0].copy(), df, summary

    df["priority_score_pct"] = df["priority_score"] / total_priority_score
    df["cumulative_priority_score"] = df["priority_score"].cumsum()
    df["cumulative_coverage"] = df["cumulative_priority_score"] / total_priority_score

    selected_df = df[df["cumulative_coverage"] <= coverage_threshold].copy()
    if len(selected_df) < len(df):
        selected_df = pd.concat([selected_df, df.iloc[[len(selected_df)]]], ignore_index=True)

    selected_priority_score = float(selected_df["priority_score"].sum())
    summary = {
        "total_pair_count": int(len(df)),
        "selected_pair_count": int(len(selected_df)),
        "total_priority_score": total_priority_score,
        "selected_priority_score": selected_priority_score,
        "selected_coverage": float(selected_priority_score / total_priority_score),
    }
    return selected_df, df, summary


def build_anchor_confusion_dict(selected_df: pd.DataFrame, anchor_classes: list[str]) -> dict[str, list[str]]:
    anchor_set = set(map(str, anchor_classes))
    result: dict[str, list[str]] = {}
    if selected_df.empty:
        return {anchor: [] for anchor in sorted(anchor_set)}

    for anchor in sorted(anchor_set):
        related = selected_df[(selected_df["class_i"] == anchor) | (selected_df["class_j"] == anchor)].copy()
        related_classes = set(related["class_i"]).union(set(related["class_j"]))
        related_classes.discard(anchor)
        result[anchor] = sorted(related_classes)
    return result


def run_confusion_pair_pipeline(
    df: pd.DataFrame,
    gt_col: str,
    pred_col: str,
    anchor_classes: list[str],
    coverage_threshold: float = 0.80,
    min_confusion_rate: float = 0.03,
    min_pair_confusion_cnt: int = 10,
) -> dict[str, Any]:
    pair_df = compute_pair_confusion_rate(df=df, gt_col=gt_col, pred_col=pred_col)
    pair_df = add_priority_score(pair_df)
    pair_df_anchor = filter_pairs_related_to_anchor_classes(pair_df=pair_df, anchor_classes=anchor_classes)
    selected_pairs_df, ranked_pairs_df, selection_summary = select_pairs_by_cumulative_coverage(
        pair_df=pair_df_anchor,
        coverage_threshold=coverage_threshold,
        min_confusion_rate=min_confusion_rate,
        min_pair_confusion_cnt=min_pair_confusion_cnt,
    )
    anchor_confusion_dict = build_anchor_confusion_dict(selected_pairs_df, anchor_classes)
    return {
        "pair_df_all": pair_df,
        "pair_df_anchor": pair_df_anchor,
        "ranked_pairs_df": ranked_pairs_df,
        "selected_pairs_df": selected_pairs_df,
        "anchor_confusion_dict": anchor_confusion_dict,
        "summary": selection_summary,
    }


def define_medium_longtail_classes(
    dist_df: pd.DataFrame,
    dist_class_col: str,
    proportion_col: str = "proportion",
    p0_p1_classes: list[str] | None = None,
    medium_cum_threshold: float = 0.60,
) -> dict[str, Any]:
    df = standardize_str_col(dist_df, dist_class_col)
    p0_p1_set = set(p0_p1_classes or [])
    df_non_p0 = df[~df[dist_class_col].isin(p0_p1_set)].copy()
    df_non_p0 = df_non_p0.sort_values(proportion_col, ascending=False).reset_index(drop=True)
    df_non_p0["cum_prop"] = df_non_p0[proportion_col].cumsum()
    medium_df = df_non_p0[df_non_p0["cum_prop"] <= medium_cum_threshold].copy()
    longtail_df = df_non_p0[df_non_p0["cum_prop"] > medium_cum_threshold].copy()
    return {
        "medium_classes": medium_df[dist_class_col].tolist(),
        "longtail_classes": longtail_df[dist_class_col].tolist(),
        "medium_df": medium_df,
        "longtail_df": longtail_df,
    }


def _compute_eval_take_with_reserve(target_n: int, available_n: int, reserve_ratio: float = 0.60) -> int:
    if available_n <= 0 or target_n <= 0:
        return 0
    if available_n >= target_n:
        return target_n
    usable_n = int(np.floor(available_n * (1.0 - reserve_ratio)))
    return max(0, min(target_n, usable_n))


def build_rolling_fresh_set(
    qa_df: pd.DataFrame,
    labeler_dist_df: pd.DataFrame,
    qa_date_col: str,
    qa_class_col: str,
    dist_class_col: str,
    proportion_col: str = "proportion",
    recent_start_date: str | None = None,
    recent_end_date: str | None = None,
    total_size: int = 30000,
    p0_p1_classes: list[str] | None = None,
    min_per_p0p1_class: int = 50,
    recency_strength: float = 2.0,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qa = standardize_str_col(qa_df, qa_class_col)
    qa = standardize_date_col(qa, qa_date_col)
    dist = standardize_str_col(labeler_dist_df, dist_class_col)
    dist = dist.rename(columns={dist_class_col: qa_class_col}).copy()

    if recent_start_date is not None:
        qa = qa[qa[qa_date_col] >= pd.to_datetime(recent_start_date)].copy()
    if recent_end_date is not None:
        qa = qa[qa[qa_date_col] <= pd.to_datetime(recent_end_date)].copy()

    p0_p1_set = set(p0_p1_classes or [])
    dist["target_n_raw"] = dist[proportion_col] * total_size
    dist["target_n"] = np.floor(dist["target_n_raw"]).astype(int)
    dist["is_p0p1"] = dist[qa_class_col].isin(p0_p1_set)
    dist.loc[dist["is_p0p1"], "target_n"] = dist.loc[dist["is_p0p1"], "target_n"].clip(lower=min_per_p0p1_class)

    available_cnt = qa.groupby(qa_class_col).size().reset_index(name="available_recent_qa_cnt")
    allocation_df = dist.merge(available_cnt, on=qa_class_col, how="left")
    allocation_df["available_recent_qa_cnt"] = allocation_df["available_recent_qa_cnt"].fillna(0).astype(int)
    allocation_df["reserved_for_training_n"] = np.where(
        allocation_df["available_recent_qa_cnt"] < allocation_df["target_n"],
        np.ceil(allocation_df["available_recent_qa_cnt"] * 0.60).astype(int),
        0,
    )
    allocation_df["final_target_n"] = allocation_df.apply(
        lambda row: _compute_eval_take_with_reserve(
            target_n=int(row["target_n"]),
            available_n=int(row["available_recent_qa_cnt"]),
            reserve_ratio=0.60,
        ),
        axis=1,
    )

    sampled = []
    for _, row in allocation_df.iterrows():
        class_name = row[qa_class_col]
        n = int(row["final_target_n"])
        if n <= 0:
            continue
        sub = qa[qa[qa_class_col] == class_name].copy()
        if sub.empty:
            continue
        sampled.append(
            recency_weighted_sample(
                df=sub,
                n=n,
                date_col=qa_date_col,
                recency_strength=recency_strength,
                random_seed=random_seed,
            )
        )

    rolling_eval_df = qa.iloc[0:0].copy() if not sampled else pd.concat(sampled).reset_index(drop=True)
    rolling_eval_df["eval_set_type"] = "rolling_fresh"
    return rolling_eval_df, allocation_df


def summarize_eval_set(eval_df: pd.DataFrame, class_col: str, date_col: str | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {"total_rows": len(eval_df), "unique_classes": eval_df[class_col].nunique() if class_col in eval_df.columns else 0}
    if date_col and date_col in eval_df.columns and not eval_df.empty:
        date_series = pd.to_datetime(eval_df[date_col], errors="coerce").dropna()
        if not date_series.empty:
            summary["min_date"] = str(date_series.min())
            summary["max_date"] = str(date_series.max())
    return summary


def summarize_eval_by_class(eval_df: pd.DataFrame, class_col: str) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame(columns=[class_col, "sample_size"])
    return (
        eval_df.groupby(class_col)
        .size()
        .reset_index(name="sample_size")
        .sort_values("sample_size", ascending=False)
        .reset_index(drop=True)
    )


def run_weekly_rolling_eval(
    qa_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
    config: dict[str, Any],
    historical_eval_df: pd.DataFrame | None = None,
    recent_start_date: str | None = None,
    recent_end_date: str | None = None,
    total_size: int = 30000,
    min_per_class: int = 50,
    recency_strength: float = 2.0,
    random_seed: int = 42,
    mix_with_history: bool = True,
) -> dict[str, Any]:
    columns = config["columns"]
    eval_source_df = filter_non_training_rows(qa_df, columns.get("is_training"))
    logger = RunLogger("rolling_eval")
    logger.log_params(
        {
            "recent_start_date": recent_start_date,
            "recent_end_date": recent_end_date,
            "total_size": total_size,
            "min_per_class": min_per_class,
            "recency_strength": recency_strength,
            "random_seed": random_seed,
            "mix_with_history": mix_with_history,
            "eval_source_size": len(eval_source_df),
        }
    )

    rolling_eval_df, allocation_df = build_rolling_fresh_set(
        qa_df=eval_source_df,
        labeler_dist_df=distribution_df,
        qa_date_col=columns["qa_date"],
        qa_class_col=columns["qa_class"],
        dist_class_col=columns["dist_class"],
        proportion_col=columns["dist_weight"],
        recent_start_date=recent_start_date,
        recent_end_date=recent_end_date,
        total_size=total_size,
        p0_p1_classes=config.get("business_rules", {}).get("p0_p1_classes", []) or _load_default_p0_p1_classes(),
        min_per_p0p1_class=min_per_class,
        recency_strength=recency_strength,
        random_seed=random_seed,
    )

    new_rolling_eval_df = rolling_eval_df.copy()
    new_rows_added = len(new_rolling_eval_df)
    if mix_with_history and historical_eval_df is not None and not historical_eval_df.empty:
        rolling_eval_df, dedup_summary = append_deduplicated_history(
            history_df=historical_eval_df,
            new_df=new_rolling_eval_df,
            item_id_col=columns["item_id"],
        )
        logger.info(f"Mixed with historical rolling eval. Dedup summary: {dedup_summary}")

    class_distribution_df = summarize_eval_by_class(rolling_eval_df, columns["qa_class"])
    coverage_summary_df = allocation_df[
        [columns["qa_class"], "target_n", "available_recent_qa_cnt", "final_target_n"]
    ].copy()
    insufficient_classes_df = allocation_df[
        allocation_df["final_target_n"] < allocation_df["target_n"]
    ][[columns["qa_class"], "target_n", "available_recent_qa_cnt", "final_target_n"]].copy()

    if not insufficient_classes_df.empty:
        logger.warning(f"{len(insufficient_classes_df)} classes have insufficient QA coverage for requested rolling eval allocation.")

    summary = summarize_eval_set(rolling_eval_df, columns["qa_class"], columns["qa_date"])
    summary["new_rows_added"] = new_rows_added

    return {
        "eval_source_df": eval_source_df,
        "new_rolling_eval_df": new_rolling_eval_df,
        "rolling_eval_df": rolling_eval_df,
        "allocation_df": allocation_df,
        "class_distribution_df": class_distribution_df,
        "coverage_summary_df": coverage_summary_df,
        "insufficient_classes_df": insufficient_classes_df,
        "summary": summary,
        "logger": logger,
    }
