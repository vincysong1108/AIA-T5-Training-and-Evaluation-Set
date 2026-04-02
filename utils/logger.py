from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class RunLogger:
    run_name: str
    events: list[str] = field(default_factory=list)

    def _stamp(self, level: str, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.events.append(f"[{timestamp}] [{self.run_name}] [{level}] {message}")

    def info(self, message: str) -> None:
        self._stamp("INFO", message)

    def warning(self, message: str) -> None:
        self._stamp("WARN", message)

    def log_params(self, params: dict[str, Any]) -> None:
        self.info(f"Parameters: {params}")


def export_run_log(logger: RunLogger | None) -> str:
    if logger is None:
        return ""
    return "\n".join(logger.events)
