"""Structured logging.

A small structured logger over :mod:`logging` — records are emitted as
``message key=value`` pairs so they stay readable locally and parseable in
log pipelines. No third-party dependency.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_CONFIGURED = False


def configure_logging(level: int | str = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger("aire")
    root.addHandler(handler)
    root.setLevel(level)
    _CONFIGURED = True


class StructuredLogger:
    """Thin wrapper adding keyword-argument structure to stdlib logging."""

    def __init__(self, name: str) -> None:
        configure_logging()
        self._logger = logging.getLogger(name)

    def _format(self, message: str, fields: dict[str, Any]) -> str:
        if not fields:
            return message
        parts = " ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"{message} {parts}"

    def debug(self, message: str, **fields: Any) -> None:
        self._logger.debug(self._format(message, fields))

    def info(self, message: str, **fields: Any) -> None:
        self._logger.info(self._format(message, fields))

    def warning(self, message: str, **fields: Any) -> None:
        self._logger.warning(self._format(message, fields))

    def error(self, message: str, **fields: Any) -> None:
        self._logger.error(self._format(message, fields))


def get_logger(name: str) -> StructuredLogger:
    return StructuredLogger(name)
