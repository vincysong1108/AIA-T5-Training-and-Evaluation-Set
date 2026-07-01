from __future__ import annotations

from decimal import Decimal, InvalidOperation

import pandas as pd


def normalize_identifier_value(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    lowered = text.lower()
    if lowered in {"nan", "none", "null"}:
        return ""

    try:
        numeric = Decimal(text)
    except (InvalidOperation, ValueError):
        return text

    if numeric == numeric.to_integral_value():
        return format(numeric.quantize(Decimal("1")), "f")
    return text


def normalize_identifier_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_identifier_value)
