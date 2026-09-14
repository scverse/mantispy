from __future__ import annotations

from pathlib import Path

from platformdirs import user_cache_dir
from scverse_misc import Settings


class MantispySettings(Settings):
    """Settings of mantispy, overridable through ``MANTISPY_`` environment variables."""

    cache_dir: Path = Path(user_cache_dir("mantispy"))
    """Where :mod:`mantispy.ds` keeps the datasets it downloads."""


settings = MantispySettings()
