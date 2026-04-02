from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd


def load_tabular_file(uploaded_file) -> pd.DataFrame:
    filename = getattr(uploaded_file, "filename", None) or getattr(uploaded_file, "name", "")
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix == ".xlsx":
        return pd.read_excel(uploaded_file)
    raise ValueError(f"Unsupported file type: {suffix}")


def build_dataset_profile(df: pd.DataFrame) -> dict[str, object]:
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
    }


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
