from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir
from scverse_misc import Settings


def _default_stream_bytes() -> int:
    """A sixteenth of the machine's memory, between 256 MB and 4 GB."""
    try:
        memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        memory = 0
    return min(max(memory // 16, 256_000_000), 4_000_000_000)


class MantispySettings(Settings):
    """Settings of mantispy, overridable through ``MANTISPY_`` environment variables.

    The log level is shared by all scverse packages, and ``scverse_misc.logging.config.verbosity`` sets it.
    """

    cache_dir: Path = Path(user_cache_dir("mantispy"))
    """Where :mod:`mantispy.ds` keeps the datasets it downloads."""

    stream_bytes: int = _default_stream_bytes()
    """How many bytes of a matrix that is not in memory a grouped operation reads at once.

    The default is a sixteenth of the machine's memory, between 256 MB and 4 GB.
    Groups are packed into reads of about this size, and a group larger than that is read a block of columns at a time, so a smaller value lowers the peak memory and a larger one reads the file in fewer passes.
    :func:`mantispy.io.read` with ``lazy=True`` also sizes its chunks so that one per dask worker fits in this many bytes.
    """


settings = MantispySettings()
