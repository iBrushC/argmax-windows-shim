"""Shared error types and logging helpers."""

from __future__ import annotations

import logging

_warned: set[str] = set()


class MissingDependencyError(ImportError):
    """Raised when a backend's optional dependency is not installed."""

    def __init__(self, package: str, extra: str) -> None:
        super().__init__(
            f"This feature needs the optional package '{package}'. Install it with:\n"
            f'    pip install "argmax-shim[{extra}]"'
        )
        self.package = package
        self.extra = extra


def warn_once(logger: logging.Logger, key: str, message: str) -> None:
    """Log a warning only the first time ``key`` is seen in this process."""
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message)
