from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from platformdirs import user_cache_dir
from pydantic import Field
from scverse_misc import Settings


def _machine_memory() -> int:
    """The machine's memory in bytes, or 0 when the system does not say."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return 0


def _default_stream_bytes() -> int:
    """A sixteenth of the machine's memory, between 256 MB and 4 GB."""
    return min(max(_machine_memory() // 16, 256_000_000), 4_000_000_000)


def _default_map_table_bytes() -> int:
    """Half of the machine's memory, or 8 GiB when the system does not say."""
    return _machine_memory() // 2 or 2**33


class MantispySettings(Settings):
    """Settings of mantispy, overridable through ``MANTISPY_`` environment variables.

    The log level is shared by all scverse packages, and ``scverse_misc.logging.config.verbosity`` sets it.
    """

    cache_dir: Path = Path(user_cache_dir("mantispy"))
    """Where :mod:`mantispy.ds` keeps the datasets it downloads."""

    stream_bytes: Annotated[int, Field(gt=0)] = _default_stream_bytes()
    """How many bytes of a matrix that is not in memory a grouped operation reads at once.

    The default is a sixteenth of the machine's memory, between 256 MB and 4 GB.
    Groups are packed into reads of about this size, and a group larger than that is read a block of columns at a time, so a smaller value lowers the peak memory and a larger one reads the file in fewer passes.
    :func:`mantispy.io.read` with ``lazy=True`` also sizes its chunks so that one per dask worker fits in this many bytes.
    It must be positive.
    """

    map_table_bytes: int = _default_map_table_bytes()
    """How many bytes of sorted similarities and ranks the permutation null of :func:`mantispy.tl.map` may hold.

    The default is half of the machine's memory.
    The null refuses a screen whose sorted similarities would exceed it.
    When a table of each similarity's rank fits beside them, the draws read the ranks, which is faster; otherwise they search the sorted similarities and give the same p-values.
    """


settings = MantispySettings()
