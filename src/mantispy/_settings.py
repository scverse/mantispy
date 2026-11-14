from __future__ import annotations

from pathlib import Path

from platformdirs import user_cache_dir
from scverse_misc import Settings


class MantispySettings(Settings):
    """Settings of mantispy, overridable through ``MANTISPY_`` environment variables.

    The log level is shared by all scverse packages, and ``scverse_misc.logging.config.verbosity`` sets it.
    """

    cache_dir: Path = Path(user_cache_dir("mantispy"))
    """Where :mod:`mantispy.ds` keeps the datasets it downloads."""

    stream_bytes: int = 256_000_000
    """How many bytes of a matrix that is not in memory a grouped operation reads at once.

    Groups are packed into reads of about this size, so a smaller value lowers the peak memory and a larger one reads the file in fewer passes.
    A single group larger than this is still read whole.
    """


settings = MantispySettings()
