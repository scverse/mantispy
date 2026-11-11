"""The package logger, and the one rule for when a drop is worth a warning."""

from __future__ import annotations

from scverse_misc.logging import TimedLogger
from scverse_misc.logging import get_logger as _scverse_logger

_LOGGER = _scverse_logger("mantispy", timed=True)


def get_logger() -> TimedLogger:
    """Return ``scverse.mantispy``, whose level follows :data:`scverse_misc.logging.config`."""
    return _LOGGER


def report_drop(what: str, dropped: int, total: int, remedy: str = "", escalate: bool = True) -> None:
    """Log that rows or features were discarded, as a warning when half or more are dropped.

    Dropping 2 features of 3634 is routine and logged at info level.
    Dropping half or more is logged as a warning, since it can leave an object too small to use.
    The ``remedy`` is shown only when the verbosity is more detailed than the message's level.

    ``escalate=False`` keeps the message at info level however much of the input is removed.
    It is for exclusions that are the documented default rather than a loss: reading a CellProfiler export drops whole-field ``Image_`` measurements by design, and a normal four-image export can have more of those than per-cell ones.
    """
    if dropped <= 0:
        return
    heavy = escalate and total and dropped >= max(total // 2, 1)
    log = get_logger().warning if heavy else get_logger().info
    log("dropped %d of %d %s", dropped, total, what, deep=remedy or None)
