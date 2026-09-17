"""Package-level settings.

The package logger has no handler of its own, so diagnostics such as "dropped 2 of 3634 feature(s)" or "dropped 960 of 960 cells" would not be shown.
This module installs a handler and sets its level from ``verbosity``, on the same 0-3 scale as scanpy.

The class comes from scverse-misc, so every setting is also read from a ``MANTISPY_`` environment variable and can be changed for a block with ``settings.override``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

from platformdirs import user_cache_dir
from pydantic import Field, field_validator
from scverse_misc import Settings

from mantispy._core.logging import get_logger

#: Verbosity to logging level, matching ``scanpy.settings.verbosity``.
LEVELS = {0: logging.ERROR, 1: logging.WARNING, 2: logging.INFO, 3: logging.DEBUG}


class _CurrentStderr:
    """A stream that forwards to whatever ``sys.stderr`` is at write time.

    ``logging.StreamHandler`` binds its stream at construction, which happens at import.
    Jupyter, ``contextlib.redirect_stderr``, pytest and captured subprocesses replace ``sys.stderr`` later, and a bound handler would keep writing to the replaced stream.
    """

    def write(self, text: str) -> int:
        """Write `text` to whatever ``sys.stderr`` is bound to now.

        Args:
            text: The text to write.

        Returns:
            How many characters were written.
        """
        return sys.stderr.write(text)

    def flush(self) -> None:
        """Flush whatever ``sys.stderr`` is bound to now."""
        sys.stderr.flush()


class MantispySettings(Settings):
    """Settings of mantispy, overridable through ``MANTISPY_`` environment variables."""

    cache_dir: Path = Path(user_cache_dir("mantispy"))
    """Where :mod:`mantispy.ds` keeps the datasets it downloads."""

    verbosity: Annotated[int, Field(ge=0, le=3)] = 1
    """How much mantispy says: 0 errors, 1 warnings, 2 info, 3 debug."""

    @field_validator("verbosity")
    @classmethod
    def _set_level(cls, value: int) -> int:
        get_logger().setLevel(LEVELS[value])
        return value


_handler = logging.StreamHandler(_CurrentStderr())
_handler.setFormatter(logging.Formatter("%(message)s"))
get_logger().handlers = [_handler]
# Without this, calling logging.basicConfig(), as notebook tutorials often do, prints every mantispy line twice.
get_logger().propagate = False

settings = MantispySettings()
