import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
from vis import render_dataframe
import itertools

#P0/P1 List

p0_p1_classes = [
    "Mobile Phones",
    "Gardening",
    "Painting Tutorials",
    "Anime/Animation Narrative",
    "Anime/Animation Edits",
    "TV series Edits",
    "Entertainment News",
    "Movie Narrative",
    "Movie Edits",
    "Celebrity Edits",
    "Crime&Police Scenes",
    "Natural disaster Scenes",
    "Current affairs News",
    "Video Games",
    "Professional Dance",
    "Live Performances",
    "Motorcycles",
    "Body Recomposition",
    "Soccer Celebrities",
    "Soccer Commentary",
    "Wrestling",
    "Makeup",
    "Pet Cats/Dogs",
    "Hair",
    "Food Preparation",
    "Instrumental Performance",
    "Other ice and snow sports",
    "Skiing & Snowboarding",
    "Air limit",
    "Badminton",
    "Billiards",
    "Boxing",
    "Cryptocurrency",
    "Equestrian",
    "Farm Animals",
    "Geography Knowledge",
    "Golf",
    "Gymnastics",
    "Ice Hockey",
    "Magic Tricks",
    "Movie Relevance",
    "Music Teaching",
    "Other Language Learning",
    "Other Pets",
    "Pickleball",
    "Rock climbing",
    "Roller skating",
    "Skateboard",
    "Stocks",
    "Surfing",
    "Table Tennis",
    "Taekwondo",
    "Yoga & Pilates"
]

#Confusion Pair
def compute_pair_confusion_rate(df, gt_col, pred_col):
    """
    Compute symmetric confusion rate R_ij for all class pairs.
    R_ij = 0.5 * (C_{i->j}/N_i + C_{j->i}/N_j)
    """

    data = df[[gt_col, pred_col]].dropna().copy()
    data[gt_col] = data[gt_col].astype(str)
    data[pred_col] = data[pred_col].astype(str)

    # N_i
    gt_counts = data.groupby(gt_col).size().to_dict()

    # C_{i->j}
    confusion_matrix = (
        data.groupby([gt_col, pred_col])
        .size()
        .to_dict()
    )

    classes = list(gt_counts.keys())
    pairs = list(itertools.combinations(classes, 2))

    results = []

    for i, j in pairs:
        C_i_j = confusion_matrix.get((i, j), 0)
        C_j_i = confusion_matrix.get((j, i), 0)

        N_i = gt_counts.get(i, 0)
        N_j = gt_counts.get(j, 0)

        if N_i == 0 or N_j == 0:
            continue

        R_ij = 0.5 * ((C_i_j / N_i) + (C_j_i / N_j))

        results.append({
            "class_i": i,
            "class_j": j,
            "C_i_to_j": C_i_j,
            "C_j_to_i": C_j_i,
            "N_i": N_i,
            "N_j": N_j,
            "confusion_rate": R_ij
        })

    pair_df = pd.DataFrame(results)
    pair_df = pair_df.sort_values("confusion_rate", ascending=False)

    return pair_df

def add_priority_score(pair_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pair_confusion_cnt, avg_pair_volume, priority_score
    """
    df = pair_df.copy()

    df["pair_confusion_cnt"] = df["C_i_to_j"] + df["C_j_to_i"]
    df["avg_pair_volume"] = (df["N_i"] + df["N_j"]) / 2
    df["priority_score"] = df["confusion_rate"] * df["avg_pair_volume"]

    return df

def filter_pairs_related_to_anchor_classes(pair_df: pd.DataFrame, anchor_classes: list) -> pd.DataFrame:
    """
    Keep pairs where either class_i or class_j is in anchor_classes
    """
    anchor_set = set(map(str, anchor_classes))
    df = pair_df.copy()

    df = df[
        df["class_i"].isin(anchor_set) | df["class_j"].isin(anchor_set)
    ].copy()

    return df.reset_index(drop=True)


def select_pairs_by_cumulative_coverage(
    pair_df: pd.DataFrame,
    coverage_threshold: float = 0.80,
    min_confusion_rate: float = 0.0,
    min_pair_confusion_cnt: int = 1,
):
    """
    Select confusion pairs by cumulative priority coverage.

    Parameters
    ----------
    pair_df : pd.DataFrame
    coverage_threshold : float
        e.g. 0.80 means keep pairs until cumulative priority coverage reaches 80%
    min_confusion_rate : float
    min_pair_confusion_cnt : int

    Returns
    -------
    selected_df : pd.DataFrame
    ranked_df : pd.DataFrame
    summary : dict
    """
    df = pair_df.copy()

    # optional guardrails
    df = df[
        (df["confusion_rate"] >= min_confusion_rate) &
        (df["pair_confusion_cnt"] >= min_pair_confusion_cnt)
    ].copy()

    if df.empty:
        summary = {
            "total_pair_count": 0,
            "selected_pair_count": 0,
            "total_priority_score": 0.0,
            "selected_priority_score": 0.0,
            "selected_coverage": 0.0
        }
        return df, df, summary

    df = df.sort_values(
        ["priority_score", "confusion_rate", "pair_confusion_cnt"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    total_priority_score = df["priority_score"].sum()

    if total_priority_score <= 0:
        df["priority_score_pct"] = 0.0
        df["cumulative_priority_score"] = 0.0
        df["cumulative_coverage"] = 0.0

        summary = {
            "total_pair_count": int(len(df)),
            "selected_pair_count": 0,
            "total_priority_score": float(total_priority_score),
            "selected_priority_score": 0.0,
            "selected_coverage": 0.0
        }
        return df.iloc[0:0].copy(), df, summary

    df["priority_score_pct"] = df["priority_score"] / total_priority_score
    df["cumulative_priority_score"] = df["priority_score"].cumsum()
    df["cumulative_coverage"] = df["cumulative_priority_score"] / total_priority_score

    selected_df = df[df["cumulative_coverage"] <= coverage_threshold].copy()

    # make sure we include the first row that crosses the threshold
    if len(selected_df) < len(df):
        next_row = df.iloc[[len(selected_df)]]
        selected_df = pd.concat([selected_df, next_row], ignore_index=True)

    selected_priority_score = selected_df["priority_score"].sum()

    summary = {
        "total_pair_count": int(len(df)),
        "selected_pair_count": int(len(selected_df)),
        "total_priority_score": float(total_priority_score),
        "selected_priority_score": float(selected_priority_score),
        "selected_coverage": float(selected_priority_score / total_priority_score if total_priority_score > 0 else 0.0)
    }

    return selected_df, df, summary

def compute_selected_class_count(selected_df: pd.DataFrame, anchor_classes: list) -> dict:
    """
    Compute selected class count summary
    """
    anchor_set = set(map(str, anchor_classes))

    all_selected_classes = set(selected_df["class_i"]).union(set(selected_df["class_j"]))
    selected_anchor_classes = all_selected_classes.intersection(anchor_set)
    selected_non_anchor_classes = all_selected_classes - anchor_set

    return {
        "selected_class_count": len(all_selected_classes),
        "selected_anchor_class_count": len(selected_anchor_classes),
        "selected_non_anchor_class_count": len(selected_non_anchor_classes),
        "selected_classes": sorted(all_selected_classes),
        "selected_anchor_classes": sorted(selected_anchor_classes),
        "selected_non_anchor_classes": sorted(selected_non_anchor_classes),
    }

def compute_per_anchor_selected_confusion_count(selected_df: pd.DataFrame, anchor_classes: list) -> pd.DataFrame:
    """
    For each anchor class, count how many selected confusion classes it connects to
    """
    anchor_set = set(map(str, anchor_classes))
    rows = []

    for anchor in sorted(anchor_set):
        related = selected_df[
            (selected_df["class_i"] == anchor) | (selected_df["class_j"] == anchor)
        ].copy()

        related_classes = set(related["class_i"]).union(set(related["class_j"]))
        related_classes.discard(anchor)

        rows.append({
            "anchor_class": anchor,
            "selected_confusion_class_count": len(related_classes),
            "selected_confusion_classes": sorted(related_classes)
        })

    out = pd.DataFrame(rows).sort_values(
        ["selected_confusion_class_count", "anchor_class"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return out

def build_anchor_confusion_dict(selected_df: pd.DataFrame, anchor_classes: list) -> dict:
    """
    Build:
        {
            anchor_class_1: [confusion_class_a, confusion_class_b, ...],
            anchor_class_2: [...],
            ...
        }
    """
    anchor_set = set(map(str, anchor_classes))
    result = {}

    for anchor in sorted(anchor_set):
        related = selected_df[
            (selected_df["class_i"] == anchor) | (selected_df["class_j"] == anchor)
        ].copy()

        related_classes = set(related["class_i"]).union(set(related["class_j"]))
        related_classes.discard(anchor)

        result[anchor] = sorted(related_classes)

    return result

def run_confusion_pair_pipeline(
    df: pd.DataFrame,
    gt_col: str,
    pred_col: str,
    anchor_classes: list,
    coverage_threshold: float = 0.80,
    min_confusion_rate: float = 0.03,
    min_pair_confusion_cnt: int = 10,
):
    """
    End-to-end pipeline:
    1. compute pair confusion rate
    2. add priority score
    3. keep pairs related to anchor classes
    4. select by cumulative coverage
    5. summarize selected class count
    6. build per-anchor table and dict

    Returns
    -------
    result_dict : dict
    """
    # step 1
    pair_df = compute_pair_confusion_rate(df=df, gt_col=gt_col, pred_col=pred_col)

    # step 2
    pair_df = add_priority_score(pair_df)

    # step 3
    pair_df_anchor = filter_pairs_related_to_anchor_classes(
        pair_df=pair_df,
        anchor_classes=anchor_classes
    )

    # step 4
    selected_pairs_df, ranked_pairs_df, selection_summary = select_pairs_by_cumulative_coverage(
        pair_df=pair_df_anchor,
        coverage_threshold=coverage_threshold,
        min_confusion_rate=min_confusion_rate,
        min_pair_confusion_cnt=min_pair_confusion_cnt
    )

    # step 5
    class_summary = compute_selected_class_count(
        selected_df=selected_pairs_df,
        anchor_classes=anchor_classes
    )

    # step 6
    per_anchor_df = compute_per_anchor_selected_confusion_count(
        selected_df=selected_pairs_df,
        anchor_classes=anchor_classes
    )

    # step 7
    anchor_confusion_dict = build_anchor_confusion_dict(
        selected_df=selected_pairs_df,
        anchor_classes=anchor_classes
    )

    summary = {**selection_summary, **class_summary}

    return {
        "pair_df_all": pair_df,
        "pair_df_anchor": pair_df_anchor,
        "ranked_pairs_df": ranked_pairs_df,
        "selected_pairs_df": selected_pairs_df,
        "per_anchor_df": per_anchor_df,
        "anchor_confusion_dict": anchor_confusion_dict,
        "summary": summary
    }

result = run_confusion_pair_pipeline(
    df=df[df['is_training_data'] == 0],
    gt_col="taxonomy5_tier3_top1_qa",  
    pred_col="taxonomy_tier3_model",            
    anchor_classes=p0_p1_classes,
    coverage_threshold=0.90,
    min_confusion_rate=0.01,
    min_pair_confusion_cnt=10
)

anchor_confusion_dict = result["anchor_confusion_dict"]

#Evaluation Set
def standardize_str_col(df, col):
    out = df.copy()
    out[col] = out[col].astype(str).str.strip()
    return out


def standardize_date_col(df, date_col, date_format="%Y%m%d"):
    out = df.copy()
    out[date_col] = pd.to_datetime(
        out[date_col].astype(str).str.strip(),
        format=date_format,
        errors="coerce"
    )
    return out

def define_medium_longtail_classes(
    dist_df,
    dist_class_col,
    proportion_col="proportion",
    p0_p1_classes=None,
    medium_cum_threshold=0.60
):
    df = dist_df.copy()
    df = standardize_str_col(df, dist_class_col)

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
        "longtail_df": longtail_df
    }

def recency_weighted_sample(
    df,
    n,
    date_col,
    recency_strength=1.0,
    random_seed=42
):
    """
    Random sample with preference toward newer rows.

    Parameters
    ----------
    df : DataFrame
    n : int
    date_col : str
    recency_strength : float
        0.0 = uniform random
        larger = stronger preference toward newer data
    """
    if len(df) <= n:
        return df.copy()

    rng = np.random.default_rng(random_seed)

    sub = df.copy().sort_values(date_col)
    min_date = sub[date_col].min()
    max_date = sub[date_col].max()

    if min_date == max_date:
        weights = np.ones(len(sub), dtype=float)
    else:
        recency_days = (sub[date_col] - min_date).dt.days.astype(float)
        recency_scaled = recency_days / max(recency_days.max(), 1.0)
        weights = np.exp(recency_strength * recency_scaled)

    weights = weights / weights.sum()

    sampled_idx = rng.choice(sub.index, size=n, replace=False, p=weights)
    return sub.loc[sampled_idx].copy()


def build_rolling_fresh_set(
    qa_df,
    labeler_dist_df,
    qa_date_col,
    qa_class_col,
    dist_class_col,
    proportion_col="proportion",
    recent_start_date=None,
    recent_end_date=None,
    total_size=30000,
    p0_p1_classes=None,
    min_per_p0p1_class=50,
    recency_strength=2.0,
    random_seed=42
):
    qa = qa_df.copy()
    dist = labeler_dist_df.copy()

    qa = standardize_str_col(qa, qa_class_col)
    qa = standardize_date_col(qa, qa_date_col)
    dist = standardize_str_col(dist, dist_class_col)

    # unify dist class col name -> qa_class_col
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
    allocation_df["final_target_n"] = allocation_df[["target_n", "available_recent_qa_cnt"]].min(axis=1)

    sampled = []

    for _, row in allocation_df.iterrows():
        c = row[qa_class_col]
        n = int(row["final_target_n"])
        if n <= 0:
            continue

        sub = qa[qa[qa_class_col] == c].copy()
        if len(sub) == 0:
            continue

        sampled_sub = recency_weighted_sample(
            df=sub,
            n=n,
            date_col=qa_date_col,
            recency_strength=recency_strength,
            random_seed=random_seed
        )
        sampled.append(sampled_sub)

    if len(sampled) == 0:
        rolling_eval_df = qa.iloc[0:0].copy()
    else:
        rolling_eval_df = pd.concat(sampled).reset_index(drop=True)

    rolling_eval_df["eval_set_type"] = "rolling_fresh"

    return rolling_eval_df, allocation_df

def build_fixed_benchmark_set(
    qa_df,
    qa_date_col,
    qa_class_col,
    p0_p1_classes,
    medium_classes,
    longtail_classes,
    sample_sizes=None,
    recency_strength=0.8,
    benchmark_start_date=None,
    benchmark_end_date=None,
    random_seed=42
):
    if sample_sizes is None:
        sample_sizes = {
            "p0p1": 450,
            "medium": 80,
            "longtail": 25
        }

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
        for c in class_list:
            sub = qa[qa[qa_class_col] == c].copy()
            available_n = len(sub)
            final_n = min(target_n, available_n)

            rows.append({
                "class": c,
                "group_name": group_name,
                "target_n": target_n,
                "available_historical_qa_cnt": available_n,
                "final_target_n": final_n
            })

            if final_n <= 0:
                continue

            sampled_sub = recency_weighted_sample(
                df=sub,
                n=final_n,
                date_col=qa_date_col,
                recency_strength=recency_strength,
                random_seed=random_seed
            )
            sampled.append(sampled_sub)

    allocation_df = pd.DataFrame(rows)

    if len(sampled) == 0:
        fixed_eval_df = qa.iloc[0:0].copy()
    else:
        fixed_eval_df = pd.concat(sampled).reset_index(drop=True)

    fixed_eval_df["eval_set_type"] = "fixed_benchmark"

    return fixed_eval_df, allocation_df


def summarize_eval_set(eval_df, class_col, date_col=None):
    summary = {
        "total_rows": len(eval_df),
        "unique_classes": eval_df[class_col].nunique()
    }

    if date_col and date_col in eval_df.columns:
        summary["min_date"] = eval_df[date_col].min()
        summary["max_date"] = eval_df[date_col].max()

    return summary


def summarize_eval_by_class(eval_df, class_col):
    return (
        eval_df.groupby(class_col)
        .size()
        .reset_index(name="sample_size")
        .sort_values("sample_size", ascending=False)
    )

def build_unified_evaluation_sets(
    qa_df,
    last_week_dist_df,
    qa_date_col,
    qa_class_col,
    dist_class_col,
    proportion_col="proportion",
    p0_p1_classes=None,
    medium_cum_threshold=0.60,

    # rolling config
    rolling_recent_start_date=None,
    rolling_recent_end_date=None,
    rolling_total_size=30000,
    rolling_min_per_p0p1_class=50,
    rolling_recency_strength=2.0,

    # fixed config
    fixed_sample_sizes=None,
    fixed_recency_strength=0.8,
    fixed_start_date=None,
    fixed_end_date=None,

    random_seed=42
):
    if fixed_sample_sizes is None:
        fixed_sample_sizes = {
            "p0p1": 450,
            "medium": 80,
            "longtail": 25
        }

    # class groups from distribution table
    class_groups = define_medium_longtail_classes(
        dist_df=last_week_dist_df,
        dist_class_col=dist_class_col,
        proportion_col=proportion_col,
        p0_p1_classes=p0_p1_classes,
        medium_cum_threshold=medium_cum_threshold
    )

    medium_classes = class_groups["medium_classes"]
    longtail_classes = class_groups["longtail_classes"]

    # rolling
    rolling_eval_df, rolling_alloc_df = build_rolling_fresh_set(
        qa_df=qa_df,
        labeler_dist_df=last_week_dist_df,
        qa_date_col=qa_date_col,
        qa_class_col=qa_class_col,
        dist_class_col=dist_class_col,
        proportion_col=proportion_col,
        recent_start_date=rolling_recent_start_date,
        recent_end_date=rolling_recent_end_date,
        total_size=rolling_total_size,
        p0_p1_classes=p0_p1_classes,
        min_per_p0p1_class=rolling_min_per_p0p1_class,
        recency_strength=rolling_recency_strength,
        random_seed=random_seed
    )

    # fixed
    fixed_eval_df, fixed_alloc_df = build_fixed_benchmark_set(
        qa_df=qa_df,
        qa_date_col=qa_date_col,
        qa_class_col=qa_class_col,
        p0_p1_classes=p0_p1_classes,
        medium_classes=medium_classes,
        longtail_classes=longtail_classes,
        sample_sizes=fixed_sample_sizes,
        recency_strength=fixed_recency_strength,
        benchmark_start_date=fixed_start_date,
        benchmark_end_date=fixed_end_date,
        random_seed=random_seed
    )

    # summaries
    rolling_summary = summarize_eval_set(
    rolling_eval_df,
    class_col=qa_class_col,
    date_col=qa_date_col)

    fixed_summary = summarize_eval_set(
        fixed_eval_df,
        class_col=qa_class_col,
        date_col=qa_date_col)

    rolling_by_class = summarize_eval_by_class(rolling_eval_df, class_col=qa_class_col)
    fixed_by_class = summarize_eval_by_class(fixed_eval_df, class_col=qa_class_col)

    return {
        "medium_classes": medium_classes,
        "longtail_classes": longtail_classes,
        "medium_df": class_groups["medium_df"],
        "longtail_df": class_groups["longtail_df"],

        "rolling_eval_df": rolling_eval_df,
        "rolling_alloc_df": rolling_alloc_df,
        "rolling_summary": rolling_summary,
        "rolling_by_class": rolling_by_class,

        "fixed_eval_df": fixed_eval_df,
        "fixed_alloc_df": fixed_alloc_df,
        "fixed_summary": fixed_summary,
        "fixed_by_class": fixed_by_class
    }

eval_result = build_unified_evaluation_sets(
    qa_df=df[df['is_training_data'] == 0],
    last_week_dist_df=lw_dist,
    qa_date_col="tcs_finish_date_qa",
    qa_class_col="taxonomy5_tier3_top1_qa",
    dist_class_col="taxonomy5_tier3_top1",
    proportion_col="item_id%",
    p0_p1_classes=p0_p1_classes,
    medium_cum_threshold=0.60,

    # rolling
    rolling_recent_start_date="2026-02-01",
    rolling_recent_end_date="2026-03-23",
    rolling_total_size=30000,
    rolling_min_per_p0p1_class=50,
    rolling_recency_strength=2.0,

    # fixed
    fixed_sample_sizes={"p0p1": 450, "medium": 80, "longtail": 25},
    fixed_recency_strength=0.8,
    fixed_start_date="2026-01-01",
    fixed_end_date="2026-03-23",

    random_seed=42
)

# Training 
def standardize_str_col(df, col):
    out = df.copy()
    out[col] = out[col].astype(str).str.strip()
    return out


def parse_date_col(df, date_col, date_format="%Y%m%d"):
    out = df.copy()
    out[date_col] = pd.to_datetime(
        out[date_col].astype(str).str.strip(),
        format=date_format,
        errors="coerce"
    )
    return out


def recency_weighted_sample(
    df,
    n,
    date_col,
    recency_strength=1.0,
    random_seed=42
):
    """
    Random sample with preference toward newer rows.
    """
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

def assign_training_tier(
    class_name,
    p0_p1_classes,
    medium_classes,
    longtail_classes
):
    if class_name in set(p0_p1_classes):
        return "priority"
    elif class_name in set(medium_classes):
        return "medium"
    elif class_name in set(longtail_classes):
        return "longtail"
    else:
        return "other"

def build_training_candidate_pool(
    qa_df,
    item_id_col,
    qa_date_col,
    qa_class_col,
    rolling_eval_df=None,
    fixed_eval_df=None,
    train_start_date=None,
    train_end_date=None,
    date_format="%Y%m%d"
):
    qa = qa_df.copy()
    qa[item_id_col] = qa[item_id_col].astype(str)
    qa = standardize_str_col(qa, qa_class_col)
    qa = parse_date_col(qa, qa_date_col, date_format=date_format)

    if train_start_date is not None:
        qa = qa[qa[qa_date_col] >= pd.to_datetime(train_start_date)].copy()
    if train_end_date is not None:
        qa = qa[qa[qa_date_col] <= pd.to_datetime(train_end_date)].copy()

    eval_ids = set()

    if rolling_eval_df is not None and len(rolling_eval_df) > 0:
        eval_ids.update(rolling_eval_df[item_id_col].astype(str).tolist())

    if fixed_eval_df is not None and len(fixed_eval_df) > 0:
        eval_ids.update(fixed_eval_df[item_id_col].astype(str).tolist())

    train_pool = qa[~qa[item_id_col].isin(eval_ids)].copy()

    # keep valid QA label only
    train_pool = train_pool[train_pool[qa_class_col].notna()].copy()
    train_pool = train_pool[train_pool[qa_class_col] != ""].copy()
    train_pool = train_pool[train_pool[qa_class_col].str.lower() != "nan"].copy()

    return train_pool.reset_index(drop=True)

def build_stable_core_source_pool(
    df,
    item_id_col,
    qa_class_col,
    model_class_col,
    rolling_eval_df=None,
    fixed_eval_df=None,
    qa_date_col=None,
    start_date=None,
    end_date=None,
    date_format="%Y%m%d"
):
    data = df.copy()

    data[item_id_col] = data[item_id_col].astype(str)
    data[qa_class_col] = data[qa_class_col].astype(str).str.strip()
    data[model_class_col] = data[model_class_col].astype(str).str.strip()

    # parse date
    if qa_date_col is not None and qa_date_col in data.columns:
        data[qa_date_col] = pd.to_datetime(
            data[qa_date_col].astype(str).str.strip(),
            format=date_format,
            errors="coerce"
        )

        if start_date is not None:
            data = data[data[qa_date_col] >= pd.to_datetime(start_date)].copy()
        if end_date is not None:
            data = data[data[qa_date_col] <= pd.to_datetime(end_date)].copy()

    # remove eval ids
    eval_ids = set()
    if rolling_eval_df is not None and len(rolling_eval_df) > 0:
        eval_ids.update(rolling_eval_df[item_id_col].astype(str).tolist())
    if fixed_eval_df is not None and len(fixed_eval_df) > 0:
        eval_ids.update(fixed_eval_df[item_id_col].astype(str).tolist())

    data = data[~data[item_id_col].isin(eval_ids)].copy()

    # keep qa = model
    out = data[data[qa_class_col] == data[model_class_col]].copy()

    # keep valid qa label
    out = out[out[qa_class_col].notna()]
    out = out[out[qa_class_col] != ""]
    out = out[out[qa_class_col].str.lower() != "nan"]

    out["sample_source"] = "historical_qa_model_agree"

    return out.reset_index(drop=True)

def build_stable_core_library(
    stable_core_pool_df,
    lw_dist,
    qa_class_col,
    dist_class_col,
    qa_date_col,
    p0_p1_classes,
    medium_classes,
    longtail_classes,
    proportion_col="proportion",
    total_size=21000,
    tier_mix=None,
    min_per_class=None,
    recency_strength=0.8,
    random_seed=42
):
    if tier_mix is None:
        tier_mix = {
            "priority": 0.35,
            "medium": 0.45,
            "longtail": 0.20
        }

    if min_per_class is None:
        min_per_class = {
            "priority": 220,
            "medium": 55,
            "longtail": 25
        }

    pool_df = stable_core_pool_df.copy()
    pool_df = standardize_str_col(pool_df, qa_class_col)

    dist_df = lw_dist.copy()
    dist_df = standardize_str_col(dist_df, dist_class_col)
    dist_df = dist_df.rename(columns={dist_class_col: qa_class_col}).copy()

    # assign tier using class names
    pool_df["training_tier"] = pool_df[qa_class_col].apply(
        lambda x: assign_training_tier(
            x,
            p0_p1_classes=p0_p1_classes,
            medium_classes=medium_classes,
            longtail_classes=longtail_classes
        )
    )

    dist_df["training_tier"] = dist_df[qa_class_col].apply(
        lambda x: assign_training_tier(
            x,
            p0_p1_classes=p0_p1_classes,
            medium_classes=medium_classes,
            longtail_classes=longtail_classes
        )
    )

    sampled = []
    alloc_rows = []

    for tier in ["priority", "medium", "longtail"]:
        tier_pool = pool_df[pool_df["training_tier"] == tier].copy()
        tier_dist = dist_df[dist_df["training_tier"] == tier].copy()

        if len(tier_pool) == 0 or len(tier_dist) == 0:
            continue

        tier_budget = int(total_size * tier_mix[tier])

        # only keep classes that actually exist in stable pool
        available_classes = set(tier_pool[qa_class_col].unique())
        tier_dist = tier_dist[tier_dist[qa_class_col].isin(available_classes)].copy()

        if len(tier_dist) == 0:
            continue

        # normalize weights within the tier using lw_dist
        tier_dist["tier_weight"] = tier_dist[proportion_col] / tier_dist[proportion_col].sum()

        tier_dist["target_n_raw"] = tier_dist["tier_weight"] * tier_budget
        tier_dist["target_n"] = tier_dist["target_n_raw"].round().astype(int)
        tier_dist["target_n"] = tier_dist["target_n"].clip(lower=min_per_class[tier])

        for _, row in tier_dist.iterrows():
            c = row[qa_class_col]
            target_n = int(row["target_n"])

            sub = tier_pool[tier_pool[qa_class_col] == c].copy()
            available_n = len(sub)
            final_n = min(target_n, available_n)

            alloc_rows.append({
                "library_type": "stable_core",
                "training_tier": tier,
                "class": c,
                "lw_class_weight": row[proportion_col],
                "tier_weight": row["tier_weight"],
                "target_n": target_n,
                "available_n": available_n,
                "final_n": final_n
            })

            if final_n <= 0:
                continue

            picked = recency_weighted_sample(
                df=sub,
                n=final_n,
                date_col=qa_date_col,
                recency_strength=recency_strength,
                random_seed=random_seed
            )
            picked["library_type"] = "stable_core"
            sampled.append(picked)

    alloc_df = pd.DataFrame(alloc_rows)

    if len(sampled) == 0:
        stable_core_df = pool_df.iloc[0:0].copy()
    else:
        stable_core_df = pd.concat(sampled).reset_index(drop=True)

    return stable_core_df, alloc_df

def build_stable_core_supplement_pool(
    df,
    qa_class_col,
    labeler_class_col,
    model_class_col,
    qa_date_col=None,
    start_date=None,
    end_date=None,
    date_format="%Y%m%d"
):
    data = df.copy()

    if qa_date_col is not None and qa_date_col in data.columns:
        data = parse_date_col(data, qa_date_col, date_format=date_format)

        if start_date is not None:
            data = data[data[qa_date_col] >= pd.to_datetime(start_date)].copy()
        if end_date is not None:
            data = data[data[qa_date_col] <= pd.to_datetime(end_date)].copy()

    for col in [qa_class_col, labeler_class_col, model_class_col]:
        data = standardize_str_col(data, col)

    mask_no_qa = (
        data[qa_class_col].isna() |
        (data[qa_class_col] == "") |
        (data[qa_class_col].str.lower() == "nan")
    )

    out = data[
        mask_no_qa &
        (data[labeler_class_col] == data[model_class_col])
    ].copy()

    out["sample_source"] = "historical_labeler_model_agree"

    return out.reset_index(drop=True)

def build_stable_core_library(
    stable_core_pool_df,
    lw_dist_df,
    qa_class_col,
    dist_class_col,
    qa_date_col,
    p0_p1_classes,
    medium_classes,
    longtail_classes,
    proportion_col="proportion",
    total_size=21000,
    tier_mix=None,
    min_per_class=None,
    recency_strength=0.8,
    random_seed=42
):
    if tier_mix is None:
        tier_mix = {
            "priority": 0.35,
            "medium": 0.45,
            "longtail": 0.20
        }

    if min_per_class is None:
        min_per_class = {
            "priority": 220,
            "medium": 55,
            "longtail": 25
        }

    pool_df = stable_core_pool_df.copy()
    pool_df = standardize_str_col(pool_df, qa_class_col)

    dist_df = lw_dist_df.copy()
    dist_df = standardize_str_col(dist_df, dist_class_col)
    dist_df = dist_df.rename(columns={dist_class_col: qa_class_col}).copy()

    # assign tier using class names
    pool_df["training_tier"] = pool_df[qa_class_col].apply(
        lambda x: assign_training_tier(
            x,
            p0_p1_classes=p0_p1_classes,
            medium_classes=medium_classes,
            longtail_classes=longtail_classes
        )
    )

    dist_df["training_tier"] = dist_df[qa_class_col].apply(
        lambda x: assign_training_tier(
            x,
            p0_p1_classes=p0_p1_classes,
            medium_classes=medium_classes,
            longtail_classes=longtail_classes
        )
    )

    sampled = []
    alloc_rows = []

    for tier in ["priority", "medium", "longtail"]:
        tier_pool = pool_df[pool_df["training_tier"] == tier].copy()
        tier_dist = dist_df[dist_df["training_tier"] == tier].copy()

        if len(tier_pool) == 0 or len(tier_dist) == 0:
            continue

        tier_budget = int(total_size * tier_mix[tier])

        # only keep classes that actually exist in stable pool
        available_classes = set(tier_pool[qa_class_col].unique())
        tier_dist = tier_dist[tier_dist[qa_class_col].isin(available_classes)].copy()

        if len(tier_dist) == 0:
            continue

        # normalize weights within the tier using lw_dist
        tier_dist["tier_weight"] = tier_dist[proportion_col] / tier_dist[proportion_col].sum()

        tier_dist["target_n_raw"] = tier_dist["tier_weight"] * tier_budget
        tier_dist["target_n"] = tier_dist["target_n_raw"].round().astype(int)
        tier_dist["target_n"] = tier_dist["target_n"].clip(lower=min_per_class[tier])

        for _, row in tier_dist.iterrows():
            c = row[qa_class_col]
            target_n = int(row["target_n"])

            sub = tier_pool[tier_pool[qa_class_col] == c].copy()
            available_n = len(sub)
            final_n = min(target_n, available_n)

            alloc_rows.append({
                "library_type": "stable_core",
                "training_tier": tier,
                "class": c,
                "lw_class_weight": row[proportion_col],
                "tier_weight": row["tier_weight"],
                "target_n": target_n,
                "available_n": available_n,
                "final_n": final_n
            })

            if final_n <= 0:
                continue

            picked = recency_weighted_sample(
                df=sub,
                n=final_n,
                date_col=qa_date_col,
                recency_strength=recency_strength,
                random_seed=random_seed
            )
            picked["library_type"] = "stable_core"
            sampled.append(picked)

    alloc_df = pd.DataFrame(alloc_rows)

    if len(sampled) == 0:
        stable_core_df = pool_df.iloc[0:0].copy()
    else:
        stable_core_df = pd.concat(sampled).reset_index(drop=True)

    return stable_core_df, alloc_df

def build_confusion_negative_set(
    train_pool_df,
    qa_class_col,
    anchor_confusion_dict,
    max_per_anchor_confusion_class=150,
    random_seed=42
):
    np.random.seed(random_seed)

    df = train_pool_df.copy()
    df = standardize_str_col(df, qa_class_col)

    out = []

    for anchor_class, confusion_classes in anchor_confusion_dict.items():
        for confusion_class in confusion_classes:
            sub = df[df[qa_class_col] == confusion_class].copy()
            if len(sub) == 0:
                continue

            if len(sub) <= max_per_anchor_confusion_class:
                picked = sub.copy()
            else:
                picked = sub.sample(
                    n=max_per_anchor_confusion_class,
                    random_state=random_seed
                )

            picked["sample_source"] = "confusion_negative"
            picked["anchor_class"] = anchor_class
            picked["confusion_class"] = confusion_class
            out.append(picked)

    if len(out) == 0:
        return df.iloc[0:0].copy()

    return pd.concat(out).reset_index(drop=True)

def build_fp_hard_negative_set(
    train_pool_df,
    qa_class_col,
    model_class_col,
    anchor_classes,
    max_per_anchor=200,
    random_seed=42
):
    np.random.seed(random_seed)

    df = train_pool_df.copy()
    df = standardize_str_col(df, qa_class_col)
    df = standardize_str_col(df, model_class_col)

    out = []

    for anchor in anchor_classes:
        sub = df[
            (df[model_class_col] == anchor) &
            (df[qa_class_col] != anchor)
        ].copy()

        if len(sub) == 0:
            continue

        if len(sub) <= max_per_anchor:
            picked = sub.copy()
        else:
            picked = sub.sample(n=max_per_anchor, random_state=random_seed)

        picked["sample_source"] = "fp_hard_negative"
        picked["anchor_class"] = anchor
        picked["confusion_class"] = picked[qa_class_col]
        out.append(picked)

    if len(out) == 0:
        return df.iloc[0:0].copy()

    return pd.concat(out).reset_index(drop=True)

def build_disagreement_set(
    train_pool_df,
    qa_class_col,
    model_class_col,
    max_total_size=None,
    random_seed=42
):
    np.random.seed(random_seed)

    df = train_pool_df.copy()
    df[qa_class_col] = df[qa_class_col].astype(str).str.strip()
    df[model_class_col] = df[model_class_col].astype(str).str.strip()

    out = df[df[qa_class_col] != df[model_class_col]].copy()
    out["sample_source"] = "disagreement_case"

    if max_total_size is not None and len(out) > max_total_size:
        out = out.sample(n=max_total_size, random_state=random_seed)

    return out.reset_index(drop=True)


def build_edge_case_set(
    train_pool_df,
    is_vague_col=None,
    is_appeal_success_col=None,
    max_total_size=None,
    random_seed=42
):
    np.random.seed(random_seed)

    df = train_pool_df.copy()

    mask = pd.Series(False, index=df.index)

    if is_vague_col is not None and is_vague_col in df.columns:
        mask = mask | (df[is_vague_col] == 1)

    if is_appeal_success_col is not None and is_appeal_success_col in df.columns:
        mask = mask | (df[is_appeal_success_col] == 0)

    out = df[mask].copy()
    out["sample_source"] = "edge_case"

    if max_total_size is not None and len(out) > max_total_size:
        out = out.sample(n=max_total_size, random_state=random_seed)

    return out.reset_index(drop=True)


def build_hard_case_library(
    train_pool_df,
    qa_class_col,
    model_class_col,
    anchor_confusion_dict,
    p0_p1_classes,
    total_size=9000,
    max_confusion_per_pair=150,
    max_fp_per_anchor=200,
    max_disagreement_total=3000,
    is_vague_col=None,
    is_appeal_success_col=None,
    max_edge_case_total=2000,
    random_seed=42
):
    confusion_neg_df = build_confusion_negative_set(
        train_pool_df=train_pool_df,
        qa_class_col=qa_class_col,
        anchor_confusion_dict=anchor_confusion_dict,
        max_per_anchor_confusion_class=max_confusion_per_pair,
        random_seed=random_seed
    )

    fp_hard_neg_df = build_fp_hard_negative_set(
        train_pool_df=train_pool_df,
        qa_class_col=qa_class_col,
        model_class_col=model_class_col,
        anchor_classes=p0_p1_classes,
        max_per_anchor=max_fp_per_anchor,
        random_seed=random_seed
    )

    disagreement_df = build_disagreement_set(
        train_pool_df=train_pool_df,
        qa_class_col=qa_class_col,
        model_class_col=model_class_col,
        max_total_size=max_disagreement_total,
        random_seed=random_seed
    )

    edge_case_df = build_edge_case_set(
        train_pool_df=train_pool_df,
        is_vague_col=is_vague_col,
        is_appeal_success_col=is_appeal_success_col,
        max_total_size=max_edge_case_total,
        random_seed=random_seed
    )

    hard_df = pd.concat(
        [confusion_neg_df, fp_hard_neg_df, disagreement_df, edge_case_df],
        ignore_index=True,
        sort=False
    )

    if len(hard_df) > 0:
        hard_df = hard_df.drop_duplicates()

    if len(hard_df) > total_size:
        hard_df = hard_df.sample(n=total_size, random_state=random_seed)

    hard_df["library_type"] = "hard_case"

    summary = {
        "confusion_negative_size": len(confusion_neg_df),
        "fp_hard_negative_size": len(fp_hard_neg_df),
        "disagreement_size": len(disagreement_df),
        "edge_case_size": len(edge_case_df),
        "final_hard_case_size": len(hard_df)
    }

    return hard_df.reset_index(drop=True), summary

def merge_final_training_set(
    stable_core_df,
    hard_case_df,
    item_id_col
):
    merged = pd.concat(
        [stable_core_df, hard_case_df],
        ignore_index=True,
        sort=False
    )

    merged[item_id_col] = merged[item_id_col].astype(str)

    source_summary = (
        merged.groupby(item_id_col)
        .agg({
            "sample_source": lambda x: sorted(set([i for i in x.dropna().astype(str)])),
            "library_type": lambda x: sorted(set([i for i in x.dropna().astype(str)]))
        })
        .reset_index()
    )

    source_summary["sample_source_joined"] = source_summary["sample_source"].apply(lambda x: "|".join(x))
    source_summary["library_type_joined"] = source_summary["library_type"].apply(lambda x: "|".join(x))

    final_training_df = merged.drop_duplicates(subset=[item_id_col]).copy()
    final_training_df = final_training_df.drop(columns=["sample_source", "library_type"], errors="ignore")

    final_training_df = final_training_df.merge(
        source_summary[[item_id_col, "sample_source_joined", "library_type_joined"]],
        on=item_id_col,
        how="left"
    )

    return final_training_df.reset_index(drop=True), source_summary

def summarize_training_set(
    final_training_df,
    qa_class_col,
    source_summary_df
):
    summary = {
        "total_training_size": int(len(final_training_df)),
        "unique_class_count": int(final_training_df[qa_class_col].nunique()) if len(final_training_df) > 0 else 0,
        "source_combo_count": int(source_summary_df["sample_source_joined"].nunique()) if len(source_summary_df) > 0 else 0,
        "library_combo_count": int(source_summary_df["library_type_joined"].nunique()) if len(source_summary_df) > 0 else 0
    }

    by_class_df = (
        final_training_df.groupby(qa_class_col)
        .size()
        .reset_index(name="training_sample_size")
        .sort_values(["training_sample_size", qa_class_col], ascending=[False, True])
        .reset_index(drop=True)
    )

    by_source_combo_df = (
        source_summary_df.groupby("sample_source_joined")
        .size()
        .reset_index(name="item_count")
        .sort_values("item_count", ascending=False)
        .reset_index(drop=True)
    )

    by_library_combo_df = (
        source_summary_df.groupby("library_type_joined")
        .size()
        .reset_index(name="item_count")
        .sort_values("item_count", ascending=False)
        .reset_index(drop=True)
    )

    return summary, by_class_df, by_source_combo_df, by_library_combo_df

def build_unified_training_set(
    qa_df,
    eval_result,
    lw_dist_df,
    item_id_col,
    qa_date_col,
    qa_class_col,
    model_class_col,
    dist_class_col,
    p0_p1_classes,
    anchor_confusion_dict,
    medium_classes,
    longtail_classes,

    # optional cols
    is_training_col=None,
    labeler_class_col=None,
    is_vague_col=None,
    is_appeal_success_col=None,

    # date config
    train_start_date="2026-01-01",
    train_end_date=None,
    date_format="%Y%m%d",

    # stable core config
    stable_total_size=21000,
    stable_tier_mix=None,
    stable_min_per_class=None,
    stable_recency_strength=0.8,
    stable_history_start_date=None,
    stable_history_end_date=None,
    proportion_col="proportion",

    # hard case config
    hard_total_size=9000,
    max_confusion_per_pair=150,
    max_fp_per_anchor=200,
    max_disagreement_total=3000,
    max_edge_case_total=2000,

    # random seed
    random_seed=42
):
    rolling_eval_df = eval_result.get("rolling_eval_df")
    fixed_eval_df = eval_result.get("fixed_eval_df")

    # 1. training candidate pool: exclude BOTH eval sets
    train_pool_df = build_training_candidate_pool(
        qa_df=qa_df,
        item_id_col=item_id_col,
        qa_date_col=qa_date_col,
        qa_class_col=qa_class_col,
        rolling_eval_df=rolling_eval_df,
        fixed_eval_df=fixed_eval_df,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        date_format=date_format
    )

    # 2. stable core source pool
    if is_training_col is None:
        raise ValueError("is_training_col is required for stable core source pool.")

    stable_core_pool_df = build_stable_core_source_pool(
        df=qa_df,  
        item_id_col=item_id_col,
        qa_class_col=qa_class_col,
        model_class_col=model_class_col,
        rolling_eval_df=rolling_eval_df,
        fixed_eval_df=fixed_eval_df,
        qa_date_col=qa_date_col,
        start_date=stable_history_start_date,
        end_date=stable_history_end_date,
        date_format=date_format
    )

    # optional supplement
    if labeler_class_col is not None and labeler_class_col in qa_df.columns:
        stable_supplement_df = build_stable_core_supplement_pool(
            df=qa_df,
            qa_class_col=qa_class_col,
            labeler_class_col=labeler_class_col,
            model_class_col=model_class_col,
            qa_date_col=qa_date_col,
            start_date=stable_history_start_date,
            end_date=stable_history_end_date,
            date_format=date_format
        )

        # still exclude eval ids from supplement
        eval_ids = set()
        if rolling_eval_df is not None and len(rolling_eval_df) > 0:
            eval_ids.update(rolling_eval_df[item_id_col].astype(str).tolist())
        if fixed_eval_df is not None and len(fixed_eval_df) > 0:
            eval_ids.update(fixed_eval_df[item_id_col].astype(str).tolist())

        stable_supplement_df[item_id_col] = stable_supplement_df[item_id_col].astype(str)
        stable_supplement_df = stable_supplement_df[
            ~stable_supplement_df[item_id_col].isin(eval_ids)
        ].copy()

        stable_core_pool_all_df = pd.concat(
            [stable_core_pool_df, stable_supplement_df],
            ignore_index=True,
            sort=False
        ).drop_duplicates(subset=[item_id_col])
    else:
        stable_supplement_df = train_pool_df.iloc[0:0].copy()
        stable_core_pool_all_df = stable_core_pool_df.copy()

    # 3. stable core library: use lw_dist_df weights
    stable_core_df, stable_alloc_df = build_stable_core_library(
        stable_core_pool_df=stable_core_pool_all_df,
        lw_dist_df=lw_dist_df,
        qa_class_col=qa_class_col,
        dist_class_col=dist_class_col,
        qa_date_col=qa_date_col,
        p0_p1_classes=p0_p1_classes,
        medium_classes=medium_classes,
        longtail_classes=longtail_classes,
        proportion_col=proportion_col,
        total_size=stable_total_size,
        tier_mix=stable_tier_mix,
        min_per_class=stable_min_per_class,
        recency_strength=stable_recency_strength,
        random_seed=random_seed
    )

    # 4. hard case library
    hard_case_df, hard_case_summary = build_hard_case_library(
        train_pool_df=train_pool_df,
        qa_class_col=qa_class_col,
        model_class_col=model_class_col,
        anchor_confusion_dict=anchor_confusion_dict,
        p0_p1_classes=p0_p1_classes,
        total_size=hard_total_size,
        max_confusion_per_pair=max_confusion_per_pair,
        max_fp_per_anchor=max_fp_per_anchor,
        max_disagreement_total=max_disagreement_total,
        is_vague_col=is_vague_col,
        is_appeal_success_col=is_appeal_success_col,
        max_edge_case_total=max_edge_case_total,
        random_seed=random_seed
    )

    # 5. merge final
    final_training_df, source_summary_df = merge_final_training_set(
        stable_core_df=stable_core_df,
        hard_case_df=hard_case_df,
        item_id_col=item_id_col
    )

    # 6. summaries
    summary, by_class_df, by_source_combo_df, by_library_combo_df = summarize_training_set(
        final_training_df=final_training_df,
        qa_class_col=qa_class_col,
        source_summary_df=source_summary_df
    )

    summary["stable_core_pool_size"] = int(len(stable_core_pool_df))
    summary["stable_supplement_pool_size"] = int(len(stable_supplement_df))
    summary["stable_core_library_size"] = int(len(stable_core_df))
    summary["hard_case_library_size"] = int(len(hard_case_df))
    summary.update(hard_case_summary)

    return {
        "train_pool_df": train_pool_df,

        "stable_core_pool_df": stable_core_pool_df,
        "stable_supplement_df": stable_supplement_df,
        "stable_core_pool_all_df": stable_core_pool_all_df,
        "stable_core_df": stable_core_df,
        "stable_alloc_df": stable_alloc_df,

        "hard_case_df": hard_case_df,

        "final_training_df": final_training_df,
        "source_summary_df": source_summary_df,

        "summary": summary,
        "by_class_df": by_class_df,
        "by_source_combo_df": by_source_combo_df,
        "by_library_combo_df": by_library_combo_df
    }

train_result = build_unified_training_set(
    qa_df=df,
    eval_result=eval_result,
    lw_dist_df=lw_dist,
    item_id_col="item_id",
    qa_date_col="tcs_finish_date_qa",
    qa_class_col="taxonomy5_tier3_top1_qa",
    model_class_col="taxonomy_tier3_model",
    dist_class_col="taxonomy5_tier3_top1",
    p0_p1_classes=p0_p1_classes,
    anchor_confusion_dict=anchor_confusion_dict,
    medium_classes=eval_result["medium_classes"],
    longtail_classes=eval_result["longtail_classes"],

    is_training_col="is_training_data",
    labeler_class_col="taxonomy5_tier3_top1_label",
    is_vague_col="is_vague",
    is_appeal_success_col="is_appeal_success",

    train_start_date="2026-01-01",
    train_end_date=None,
    date_format="%Y%m%d",

    stable_total_size=21000,
    stable_tier_mix={
        "priority": 0.35,
        "medium": 0.45,
        "longtail": 0.20
    },
    stable_min_per_class={
        "priority": 220,
        "medium": 55,
        "longtail": 25
    },
    stable_recency_strength=0.8,
    stable_history_start_date="2025-01-01",
    stable_history_end_date="2026-03-23",
    proportion_col="item_id%",

    hard_total_size=9000,
    max_confusion_per_pair=150,
    max_fp_per_anchor=200,
    max_disagreement_total=3000,
    max_edge_case_total=2000,

    random_seed=42
)