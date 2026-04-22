from __future__ import annotations

import pandas as pd


def _aliases_to_columns(config: dict, aliases: list[str]) -> list[str]:
    columns = config.get("columns", {})
    return [columns.get(alias, alias) for alias in aliases]


def build_schema_check_report(
    df: pd.DataFrame,
    config: dict,
    required_aliases: list[str],
    optional_aliases: list[str],
) -> dict[str, list[str]]:
    required_columns = _aliases_to_columns(config, required_aliases)
    optional_columns = _aliases_to_columns(config, optional_aliases)
    existing = set(df.columns)
    duplicate_columns = sorted(df.columns[df.columns.duplicated()].astype(str).tolist())
    return {
        "missing_columns": [column for column in required_columns if column not in existing],
        "missing_optional_columns": [column for column in optional_columns if column not in existing],
        "duplicate_columns": duplicate_columns,
    }
