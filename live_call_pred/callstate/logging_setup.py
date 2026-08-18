"""
Logging: a readable console stream plus a machine-readable JSONL trace.

Both exist because they answer different questions. The console line is for a
person watching a call go by and asking "what is it doing right now"; the
JSONL trace is for reconstructing, after the fact, exactly which evidence
produced a state change on a call that went wrong. The trace carries the
per-feature contributions from the state model, so a bad call can be debugged
without re-running it.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional


class JsonlHandler(logging.Handler):
    """Writes structured records; anything attached as `record.extra_fields`."""

    def __init__(self, path: str):
        super().__init__()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.stream = open(path, "a", encoding="utf-8")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = {
                "ts": record.created,
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            extra = getattr(record, "extra_fields", None)
            if isinstance(extra, dict):
                payload.update(extra)
            self.stream.write(json.dumps(payload, default=str) + "\n")
            self.stream.flush()
        except Exception:  # pragma: no cover - logging must never crash a call
            self.handleError(record)

    def close(self) -> None:
        try:
            self.stream.close()
        finally:
            super().close()


class _ConsoleFormatter(logging.Formatter):
    COLORS = {"DEBUG": "\033[90m", "INFO": "", "WARNING": "\033[33m",
              "ERROR": "\033[31m", "CRITICAL": "\033[1;31m"}
    RESET = "\033[0m"

    def __init__(self, color: bool = True):
        super().__init__("%(asctime)s %(levelname)-7s %(name)-22s %(message)s", "%H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        if not self.color:
            return base
        c = self.COLORS.get(record.levelname, "")
        return f"{c}{base}{self.RESET}" if c else base


def setup_logging(level: str = "INFO", jsonl_path: Optional[str] = None,
                  quiet_console: bool = False) -> logging.Logger:
    root = logging.getLogger("callstate")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    if not quiet_console:
        ch = logging.StreamHandler(sys.stderr)
        ch.setFormatter(_ConsoleFormatter(color=sys.stderr.isatty()))
        root.addHandler(ch)

    if jsonl_path:
        root.addHandler(JsonlHandler(jsonl_path))

    root.propagate = False
    return root


def log_kv(logger: logging.Logger, level: int, msg: str, **fields) -> None:
    """Log a human line and the structured fields that back it, together."""
    logger.log(level, msg, extra={"extra_fields": fields})
