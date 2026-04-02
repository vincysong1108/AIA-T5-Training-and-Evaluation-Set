from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO

import yaml


def load_app_config(source: str | Path | BinaryIO) -> dict[str, Any]:
    if hasattr(source, "read"):
        content = source.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return yaml.safe_load(content) or {}

    with open(source, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
