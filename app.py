from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, Response, redirect, render_template, request, url_for
import pandas as pd

from services.benchmark_service import run_monthly_benchmark_refresh
from services.rolling_eval_service import run_weekly_rolling_eval
from services.training_service import build_training_candidate_pool, run_training_library_update
from utils.config import load_app_config
from utils.data_loader import build_dataset_profile, load_tabular_file
from utils.logger import export_run_log
from utils.validators import build_schema_check_report


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"

app = Flask(__name__, template_folder="templates")


STATE: dict[str, Any] = {
    "config": load_app_config(DEFAULT_CONFIG_PATH),
    "qa_df": None,
    "distribution_df": None,
    "training_library_df": None,
    "rolling_eval_history_df": None,
    "benchmark_eval_history_df": None,
    "rolling_eval_result": None,
    "training_update_result": None,
    "training_pool_result": None,
    "benchmark_result": None,
    "form_values": {
        "rolling_eval": {},
        "training_update": {},
        "training_pool": {},
        "benchmark_refresh": {},
    },
}


def _to_table(df: pd.DataFrame | None, max_rows: int | None = 10, wrapper_class: str = "table-wrap") -> str:
    if df is None or df.empty:
        return "<p class='muted'>No data available.</p>"

    display_df = df.copy() if max_rows is None else df.head(max_rows).copy()
    for col in display_df.columns:
        if pd.api.types.is_datetime64_any_dtype(display_df[col]):
            display_df[col] = display_df[col].dt.strftime("%Y-%m-%d").fillna("")

    return (
        f'<div class="{wrapper_class}">'
        + display_df.to_html(classes="data-table", index=False, border=0)
        + "</div>"
    )


def _profile_for(session_key: str, required_aliases: list[str], optional_aliases: list[str]) -> dict[str, Any]:
    df = STATE.get(session_key)
    if df is None:
        return {"loaded": False}

    config = STATE["config"]
    report = build_schema_check_report(df, config, required_aliases, optional_aliases)
    profile = build_dataset_profile(df)
    return {
        "loaded": True,
        "profile": profile,
        "schema_report": report,
        "preview_html": _to_table(df, max_rows=10),
    }


def _save_upload(file_storage, session_key: str) -> None:
    if file_storage and file_storage.filename:
        STATE[session_key] = load_tabular_file(file_storage)


def _coerce_int(value: str | None, fallback: int) -> int:
    try:
        return int(value) if value not in (None, "") else fallback
    except (TypeError, ValueError):
        return fallback


def _coerce_optional_int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _coerce_float(value: str | None, fallback: float) -> float:
    try:
        return float(value) if value not in (None, "") else fallback
    except (TypeError, ValueError):
        return fallback


def _update_config_from_form(form) -> None:
    config = STATE["config"]
    columns = config["columns"]
    app_cfg = config["app"]
    rolling = config["defaults"]["rolling_eval"]
    training = config["defaults"]["training_update"]
    benchmark = config["defaults"]["benchmark_refresh"]

    columns["item_id"] = form.get("cfg_item_id", columns["item_id"])
    columns["qa_date"] = form.get("cfg_qa_date", columns["qa_date"])
    columns["qa_class"] = form.get("cfg_qa_class", columns["qa_class"])
    columns["model_class"] = form.get("cfg_model_class", columns["model_class"])
    columns["labeler_class"] = form.get("cfg_labeler_class", columns["labeler_class"])
    columns["dist_class"] = form.get("cfg_dist_class", columns["dist_class"])
    columns["dist_weight"] = form.get("cfg_dist_weight", columns["dist_weight"])
    columns["benchmark_class"] = form.get("cfg_benchmark_class", columns["benchmark_class"])
    columns["is_vague"] = form.get("cfg_is_vague", columns["is_vague"])
    columns["is_appeal_success"] = form.get("cfg_is_appeal_success", columns["is_appeal_success"])

    app_cfg["random_seed"] = _coerce_int(form.get("cfg_random_seed"), app_cfg["random_seed"])
    app_cfg["date_format_preference"] = form.get("cfg_date_format", app_cfg["date_format_preference"])

    rolling["total_size"] = _coerce_int(form.get("cfg_rolling_total_size"), rolling["total_size"])
    rolling["min_per_p0p1_class"] = _coerce_int(form.get("cfg_rolling_min_per_class"), rolling["min_per_p0p1_class"])
    rolling["recency_strength"] = _coerce_float(form.get("cfg_rolling_recency_strength"), rolling["recency_strength"])
    rolling["coverage_threshold"] = _coerce_float(form.get("cfg_rolling_coverage_threshold"), rolling["coverage_threshold"])
    rolling["min_confusion_rate"] = _coerce_float(form.get("cfg_rolling_min_confusion_rate"), rolling["min_confusion_rate"])
    rolling["min_pair_confusion_cnt"] = _coerce_int(form.get("cfg_rolling_min_pair_confusion_cnt"), rolling["min_pair_confusion_cnt"])
    rolling["mix_with_history"] = form.get("cfg_rolling_mix_with_history") == "on"

    training["hard_total_size"] = _coerce_int(form.get("cfg_training_hard_total_size"), training["hard_total_size"])
    training["max_confusion_per_pair"] = _coerce_int(form.get("cfg_training_max_confusion_per_pair"), training["max_confusion_per_pair"])
    training["max_fp_per_anchor"] = _coerce_int(form.get("cfg_training_max_fp_per_anchor"), training["max_fp_per_anchor"])
    training["max_disagreement_total"] = _coerce_int(form.get("cfg_training_max_disagreement_total"), training["max_disagreement_total"])
    training["max_edge_case_total"] = _coerce_int(form.get("cfg_training_max_edge_case_total"), training["max_edge_case_total"])
    training["bootstrap_when_no_history"] = form.get("cfg_training_bootstrap_when_no_history") == "on"
    training["target_total_training_size"] = _coerce_int(form.get("cfg_training_target_total_training_size"), training.get("target_total_training_size", 30000))
    training["stable_core_filter_expression"] = form.get(
        "cfg_training_stable_core_filter_expression",
        training.get("stable_core_filter_expression", "qa_class = model_class"),
    )
    training["exclude_eval_overlap"] = form.get("cfg_training_exclude_eval_overlap") == "on"
    training["dedup_by_item_id"] = form.get("cfg_training_dedup_by_item_id") == "on"

    benchmark["medium_cum_threshold"] = _coerce_float(form.get("cfg_benchmark_medium_cum_threshold"), benchmark["medium_cum_threshold"])
    benchmark["fixed_p0p1_size"] = _coerce_int(form.get("cfg_benchmark_fixed_p0p1_size"), benchmark["fixed_p0p1_size"])
    benchmark["fixed_medium_size"] = _coerce_int(form.get("cfg_benchmark_fixed_medium_size"), benchmark["fixed_medium_size"])
    benchmark["fixed_longtail_size"] = _coerce_int(form.get("cfg_benchmark_fixed_longtail_size"), benchmark["fixed_longtail_size"])
    benchmark["fixed_recency_strength"] = _coerce_float(form.get("cfg_benchmark_fixed_recency_strength"), benchmark["fixed_recency_strength"])

    raw_p0 = form.get("cfg_p0_p1_classes", "")
    parsed_p0 = [line.strip() for line in raw_p0.splitlines() if line.strip()]
    if parsed_p0:
        config["business_rules"]["p0_p1_classes"] = parsed_p0


def _result_df(result_key: str, df_key: str) -> pd.DataFrame | None:
    result = STATE.get(result_key)
    if not result:
        return None
    return result.get(df_key)


def _merged_form_values(defaults: dict[str, Any], stored: dict[str, Any]) -> dict[str, Any]:
    merged = dict(defaults)
    merged.update(stored or {})
    return merged


def _download_csv(df: pd.DataFrame, filename: str) -> Response:
    return Response(
        df.to_csv(index=False),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _download_text(content: str, filename: str) -> Response:
    return Response(
        content.encode("utf-8"),
        mimetype="text/plain",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _has_any_result() -> bool:
    return any(
        STATE.get(key) is not None
        for key in ["rolling_eval_result", "training_update_result", "benchmark_result"]
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        _update_config_from_form(request.form)

        _save_upload(request.files.get("qa_file"), "qa_df")
        _save_upload(request.files.get("distribution_file"), "distribution_df")
        _save_upload(request.files.get("training_library_file"), "training_library_df")
        _save_upload(request.files.get("rolling_eval_history_file"), "rolling_eval_history_df")
        _save_upload(request.files.get("benchmark_eval_history_file"), "benchmark_eval_history_df")
        return redirect(url_for("index"))

    datasets = {
        "qa": _profile_for(
            "qa_df",
            required_aliases=["item_id", "qa_date", "qa_class"],
            optional_aliases=["model_class", "labeler_class", "is_vague", "is_appeal_success"],
        ),
        "distribution": _profile_for(
            "distribution_df",
            required_aliases=["dist_class", "dist_weight"],
            optional_aliases=["benchmark_class"],
        ),
        "training_library": _profile_for(
            "training_library_df",
            required_aliases=["item_id"],
            optional_aliases=["qa_class"],
        ),
        "rolling_eval_history": _profile_for(
            "rolling_eval_history_df",
            required_aliases=["item_id"],
            optional_aliases=["qa_class"],
        ),
        "benchmark_eval_history": _profile_for(
            "benchmark_eval_history_df",
            required_aliases=["item_id"],
            optional_aliases=["qa_class"],
        ),
    }

    downloads = {
        "rolling_eval": _result_df("rolling_eval_result", "rolling_eval_df") is not None,
        "training_library": _result_df("training_update_result", "updated_training_library_df") is not None,
        "training_pool": _result_df("training_pool_result", "training_pool_df") is not None,
        "appended_hard_cases": _result_df("training_update_result", "new_hard_cases_df") is not None,
        "benchmark_eval": _result_df("benchmark_result", "benchmark_eval_df") is not None,
        "class_summary": _result_df("benchmark_result", "class_summary_df") is not None,
        "run_log": _has_any_result(),
        "html_report": _has_any_result(),
    }

    return render_template("index.html", datasets=datasets, downloads=downloads, config=STATE["config"])


@app.route("/rolling-eval", methods=["GET", "POST"])
def rolling_eval():
    if STATE["qa_df"] is None or STATE["distribution_df"] is None:
        return redirect(url_for("index"))

    defaults = STATE["config"]["defaults"]["rolling_eval"]
    form_values = _merged_form_values(defaults, STATE["form_values"]["rolling_eval"])
    if request.method == "POST":
        form_values = {
            "recent_start_date": request.form.get("recent_start_date", ""),
            "recent_end_date": request.form.get("recent_end_date", ""),
            "total_size": int(request.form.get("total_size", defaults["total_size"])),
            "min_per_class": int(request.form.get("min_per_class", defaults["min_per_p0p1_class"])),
            "recency_strength": float(request.form.get("recency_strength", defaults["recency_strength"])),
            "random_seed": int(request.form.get("random_seed", STATE["config"]["app"]["random_seed"])),
            "mix_with_history": request.form.get("mix_with_history") == "on",
        }
        STATE["form_values"]["rolling_eval"] = form_values
        result = run_weekly_rolling_eval(
            qa_df=STATE["qa_df"],
            distribution_df=STATE["distribution_df"],
            historical_eval_df=STATE["rolling_eval_history_df"],
            config=STATE["config"],
            recent_start_date=form_values["recent_start_date"] or None,
            recent_end_date=form_values["recent_end_date"] or None,
            total_size=form_values["total_size"],
            min_per_class=form_values["min_per_class"],
            recency_strength=form_values["recency_strength"],
            random_seed=form_values["random_seed"],
            mix_with_history=form_values["mix_with_history"],
        )
        STATE["rolling_eval_result"] = result

    result = STATE.get("rolling_eval_result")
    return render_template(
        "rolling_eval.html",
        defaults=defaults,
        result=result,
        summary=(result or {}).get("summary"),
        form_values=form_values,
        rolling_eval_html=_to_table((result or {}).get("rolling_eval_df")),
        class_distribution_html=_to_table((result or {}).get("class_distribution_df")),
        coverage_summary_html=_to_table((result or {}).get("coverage_summary_df")),
        insufficient_classes_html=_to_table((result or {}).get("insufficient_classes_df")),
        log_text=export_run_log((result or {}).get("logger")),
        seed=STATE["config"]["app"]["random_seed"],
    )


@app.route("/training-update", methods=["GET", "POST"])
def training_update():
    if STATE["qa_df"] is None or STATE["distribution_df"] is None:
        return redirect(url_for("index"))

    defaults = STATE["config"]["defaults"]["training_update"]
    form_values = _merged_form_values(defaults, STATE["form_values"]["training_update"])
    for optional_key in [
        "max_confusion_per_pair",
        "max_fp_per_anchor",
        "max_disagreement_total",
        "max_edge_case_total",
    ]:
        form_values[optional_key] = STATE["form_values"]["training_update"].get(optional_key, "")
    if request.method == "POST":
        form_values = {
            "train_start_date": request.form.get("train_start_date", ""),
            "train_end_date": request.form.get("train_end_date", ""),
            "hard_total_size": int(request.form.get("hard_total_size", defaults["hard_total_size"])),
            "max_confusion_per_pair": _coerce_optional_int(request.form.get("max_confusion_per_pair")),
            "max_fp_per_anchor": _coerce_optional_int(request.form.get("max_fp_per_anchor")),
            "max_disagreement_total": _coerce_optional_int(request.form.get("max_disagreement_total")),
            "max_edge_case_total": _coerce_optional_int(request.form.get("max_edge_case_total")),
            "bootstrap_when_no_history": request.form.get("bootstrap_when_no_history") == "on",
            "target_total_training_size": int(
                request.form.get(
                    "target_total_training_size",
                    defaults.get("target_total_training_size", 30000),
                )
            ),
            "stable_core_filter_expression": request.form.get(
                "stable_core_filter_expression",
                defaults.get("stable_core_filter_expression", "qa_class = model_class"),
            ).strip() or defaults.get("stable_core_filter_expression", "qa_class = model_class"),
            "random_seed": int(request.form.get("random_seed", STATE["config"]["app"]["random_seed"])),
            "exclude_eval_overlap": request.form.get("exclude_eval_overlap") == "on",
            "dedup_by_item_id": request.form.get("dedup_by_item_id") == "on",
        }
        STATE["form_values"]["training_update"] = form_values
        result = run_training_library_update(
            qa_df=STATE["qa_df"],
            distribution_df=STATE["distribution_df"],
            historical_training_df=STATE["training_library_df"],
            rolling_eval_df=_result_df("rolling_eval_result", "rolling_eval_df"),
            benchmark_eval_df=_result_df("benchmark_result", "benchmark_eval_df"),
            config=STATE["config"],
            train_start_date=form_values["train_start_date"] or None,
            train_end_date=form_values["train_end_date"] or None,
            hard_total_size=form_values["hard_total_size"],
            max_confusion_per_pair=form_values["max_confusion_per_pair"],
            max_fp_per_anchor=form_values["max_fp_per_anchor"],
            max_disagreement_total=form_values["max_disagreement_total"],
            max_edge_case_total=form_values["max_edge_case_total"],
            bootstrap_when_no_history=form_values["bootstrap_when_no_history"],
            target_total_training_size=form_values["target_total_training_size"],
            stable_core_filter_expression=form_values["stable_core_filter_expression"],
            random_seed=form_values["random_seed"],
            exclude_eval_overlap=form_values["exclude_eval_overlap"],
            dedup_by_item_id=form_values["dedup_by_item_id"],
        )
        STATE["training_update_result"] = result

    result = STATE.get("training_update_result")
    return render_template(
        "training_update.html",
        defaults=defaults,
        result=result,
        summary=(result or {}).get("summary"),
        form_values=form_values,
        top_confusion_pairs_html=_to_table(
            (result or {}).get("top_confusion_pairs_df"),
            max_rows=None,
            wrapper_class="table-wrap table-wrap-tall",
        ),
        stable_core_alloc_html=_to_table((result or {}).get("stable_alloc_df")),
        stable_core_html=_to_table((result or {}).get("stable_core_df")),
        new_hard_cases_html=_to_table((result or {}).get("new_hard_cases_df")),
        updated_training_html=_to_table((result or {}).get("updated_training_library_df")),
        dedup_summary_html=_to_table((result or {}).get("dedup_summary_df")),
        count_added_by_class_html=_to_table((result or {}).get("count_added_by_class_df")),
        count_added_by_source_date_html=_to_table((result or {}).get("count_added_by_source_date_df")),
        log_text=export_run_log((result or {}).get("logger")),
        seed=STATE["config"]["app"]["random_seed"],
    )


@app.route("/training-pool", methods=["GET", "POST"])
def training_pool():
    if STATE["qa_df"] is None or STATE["distribution_df"] is None:
        return redirect(url_for("index"))

    defaults = {"train_start_date": "", "train_end_date": "", "exclude_eval_overlap": True}
    form_values = _merged_form_values(defaults, STATE["form_values"]["training_pool"])
    if request.method == "POST":
        form_values = {
            "train_start_date": request.form.get("train_start_date", ""),
            "train_end_date": request.form.get("train_end_date", ""),
            "exclude_eval_overlap": request.form.get("exclude_eval_overlap") == "on",
        }
        STATE["form_values"]["training_pool"] = form_values

        training_pool_df = build_training_candidate_pool(
            qa_df=STATE["qa_df"],
            item_id_col=STATE["config"]["columns"]["item_id"],
            qa_date_col=STATE["config"]["columns"]["qa_date"],
            qa_class_col=STATE["config"]["columns"]["qa_class"],
            rolling_eval_df=_result_df("rolling_eval_result", "rolling_eval_df") if form_values["exclude_eval_overlap"] else None,
            benchmark_eval_df=_result_df("benchmark_result", "benchmark_eval_df") if form_values["exclude_eval_overlap"] else None,
            train_start_date=form_values["train_start_date"] or None,
            train_end_date=form_values["train_end_date"] or None,
        )

        summary = {
            "training_pool_size": len(training_pool_df),
            "unique_class_count": training_pool_df[STATE["config"]["columns"]["qa_class"]].nunique() if not training_pool_df.empty else 0,
            "excluded_new_rolling_eval": _result_df("rolling_eval_result", "new_rolling_eval_df") is not None and form_values["exclude_eval_overlap"],
            "excluded_new_fixed_eval": _result_df("benchmark_result", "new_fixed_eval_df") is not None and form_values["exclude_eval_overlap"],
        }

        STATE["training_pool_result"] = {
            "training_pool_df": training_pool_df,
            "summary": summary,
        }

    result = STATE.get("training_pool_result")
    return render_template(
        "training_pool.html",
        form_values=form_values,
        result=result,
        summary=(result or {}).get("summary"),
        training_pool_html=_to_table((result or {}).get("training_pool_df")),
    )


@app.route("/benchmark-refresh", methods=["GET", "POST"])
def benchmark_refresh():
    if STATE["qa_df"] is None or STATE["distribution_df"] is None:
        return redirect(url_for("index"))

    defaults = STATE["config"]["defaults"]["benchmark_refresh"]
    form_values = _merged_form_values(defaults, STATE["form_values"]["benchmark_refresh"])
    if request.method == "POST":
        form_values = {
            "benchmark_start_date": request.form.get("benchmark_start_date", ""),
            "benchmark_end_date": request.form.get("benchmark_end_date", ""),
            "medium_cum_threshold": float(request.form.get("medium_cum_threshold", defaults["medium_cum_threshold"])),
            "fixed_p0p1_size": int(request.form.get("fixed_p0p1_size", defaults["fixed_p0p1_size"])),
            "fixed_medium_size": int(request.form.get("fixed_medium_size", defaults["fixed_medium_size"])),
            "fixed_longtail_size": int(request.form.get("fixed_longtail_size", defaults["fixed_longtail_size"])),
            "fixed_recency_strength": float(request.form.get("fixed_recency_strength", defaults["fixed_recency_strength"])),
            "refresh_pct": float(request.form.get("refresh_pct", defaults["refresh_pct"])),
            "random_seed": int(request.form.get("random_seed", STATE["config"]["app"]["random_seed"])),
        }
        STATE["form_values"]["benchmark_refresh"] = form_values
        result = run_monthly_benchmark_refresh(
            qa_df=STATE["qa_df"],
            benchmark_df=STATE["distribution_df"],
            historical_benchmark_df=STATE["benchmark_eval_history_df"],
            config=STATE["config"],
            benchmark_start_date=form_values["benchmark_start_date"] or None,
            benchmark_end_date=form_values["benchmark_end_date"] or None,
            medium_cum_threshold=form_values["medium_cum_threshold"],
            fixed_p0p1_size=form_values["fixed_p0p1_size"],
            fixed_medium_size=form_values["fixed_medium_size"],
            fixed_longtail_size=form_values["fixed_longtail_size"],
            fixed_recency_strength=form_values["fixed_recency_strength"],
            refresh_pct=form_values["refresh_pct"],
            random_seed=form_values["random_seed"],
        )
        STATE["benchmark_result"] = result

    result = STATE.get("benchmark_result")
    return render_template(
        "benchmark_refresh.html",
        defaults=defaults,
        form_values=form_values,
        result=result,
        summary=(result or {}).get("summary"),
        bottom_classes_html=_to_table((result or {}).get("bottom_classes_df")),
        benchmark_eval_html=_to_table((result or {}).get("benchmark_eval_df")),
        class_summary_html=_to_table((result or {}).get("class_summary_df")),
        mom_comparison_html=_to_table((result or {}).get("mom_comparison_df")),
        dropped_classes_html=_to_table((result or {}).get("dropped_classes_df")),
        log_text=export_run_log((result or {}).get("logger")),
        seed=STATE["config"]["app"]["random_seed"],
    )


@app.route("/download/<artifact>")
def download(artifact: str):
    if artifact == "rolling_eval":
        df = _result_df("rolling_eval_result", "rolling_eval_df")
        if df is not None:
            return _download_csv(df, "rolling_eval.csv")
    if artifact == "training_library":
        df = _result_df("training_update_result", "updated_training_library_df")
        if df is not None:
            return _download_csv(df, "training_library.csv")
    if artifact == "training_pool":
        df = _result_df("training_pool_result", "training_pool_df")
        if df is not None:
            return _download_csv(df, "training_pool.csv")
    if artifact == "appended_hard_cases":
        df = _result_df("training_update_result", "new_hard_cases_df")
        if df is not None:
            return _download_csv(df, "appended_hard_cases.csv")
    if artifact == "benchmark_eval":
        df = _result_df("benchmark_result", "benchmark_eval_df")
        if df is not None:
            return _download_csv(df, "benchmark_eval.csv")
    if artifact == "class_summary":
        df = _result_df("benchmark_result", "class_summary_df")
        if df is not None:
            return _download_csv(df, "class_summary.csv")
    if artifact == "run_log":
        run_log = "\n\n".join(
            [
                export_run_log((STATE.get("rolling_eval_result") or {}).get("logger")),
                export_run_log((STATE.get("training_update_result") or {}).get("logger")),
                export_run_log((STATE.get("benchmark_result") or {}).get("logger")),
            ]
        ).strip()
        if run_log:
            return _download_text(run_log, "run_log.txt")
    if artifact == "html_report":
        html = render_template(
            "report.html",
            rolling_result=STATE.get("rolling_eval_result"),
            training_result=STATE.get("training_update_result"),
            benchmark_result=STATE.get("benchmark_result"),
            rolling_eval_html=_to_table(_result_df("rolling_eval_result", "rolling_eval_df")),
            rolling_class_distribution_html=_to_table((STATE.get("rolling_eval_result") or {}).get("class_distribution_df")),
            training_new_hard_cases_html=_to_table(_result_df("training_update_result", "new_hard_cases_df")),
            training_updated_library_html=_to_table(_result_df("training_update_result", "updated_training_library_df")),
            benchmark_eval_html=_to_table(_result_df("benchmark_result", "benchmark_eval_df")),
            benchmark_class_summary_html=_to_table(_result_df("benchmark_result", "class_summary_df")),
            run_log="\n\n".join(
                [
                    export_run_log((STATE.get("rolling_eval_result") or {}).get("logger")),
                    export_run_log((STATE.get("training_update_result") or {}).get("logger")),
                    export_run_log((STATE.get("benchmark_result") or {}).get("logger")),
                ]
            ).strip(),
        )
        return Response(
            html,
            mimetype="text/html",
            headers={"Content-Disposition": "attachment; filename=dataset_ops_report.html"},
        )
    return redirect(url_for("index"))


if __name__ == "__main__":
    host = os.environ.get("DATASET_OPS_HOST", "0.0.0.0")
    port = int(os.environ.get("DATASET_OPS_PORT", "8501"))
    debug = os.environ.get("DATASET_OPS_DEBUG", "true").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug)
