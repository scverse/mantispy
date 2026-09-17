"""The package logger, and the one rule for when a drop is worth a warning."""

from __future__ import annotations

import logging

_LOGGER = logging.getLogger("mantispy")


def get_logger() -> logging.Logger:
    """Return the package logger."""
    return _LOGGER


def report_drop(what: str, dropped: int, total: int, remedy: str = "", escalate: bool = True) -> None:
    """Log that rows or features were discarded, as a warning when half or more are dropped.

    Dropping 2 features of 3634 is routine and logged at info level.
    Dropping half or more is logged as a warning, since it can leave an object too small to use.

    ``escalate=False`` keeps the message at info level however much of the input is removed.
    It is for exclusions that are the documented default rather than a loss: reading a CellProfiler export drops whole-field ``Image_`` measurements by design, and a normal four-image export can have more of those than per-cell ones.
    A warning there would teach users to ignore warnings.
    """
    if dropped <= 0:
        return
    message = "dropped %d of %d %s" + (f"; {remedy}" if remedy else "")
    heavy = escalate and total and dropped >= max(total // 2, 1)
    log = get_logger().warning if heavy else get_logger().info
    log(message, dropped, total, what)
