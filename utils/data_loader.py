from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd


def ensure_unique_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    seen: dict[str, int] = {}
    renamed_duplicates: list[str] = []
    unique_columns: list[str] = []

    for column in map(str, out.columns):
        count = seen.get(column, 0)
        if count == 0:
            unique_columns.append(column)
        else:
            unique_columns.append(f"{column}__dup{count + 1}")
            renamed_duplicates.append(column)
        seen[column] = count + 1

    out.columns = unique_columns
    return out, renamed_duplicates


def load_tabular_file(uploaded_file) -> pd.DataFrame:
    filename = getattr(uploaded_file, "filename", None) or getattr(uploaded_file, "name", "")
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(uploaded_file)
        return ensure_unique_columns(df)[0]
    if suffix == ".xlsx":
        df = pd.read_excel(uploaded_file)
        return ensure_unique_columns(df)[0]
    raise ValueError(f"Unsupported file type: {suffix}")


def build_dataset_profile(df: pd.DataFrame) -> dict[str, object]:
    _, duplicate_columns = ensure_unique_columns(df)
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "duplicate_columns": duplicate_columns,
    }


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
