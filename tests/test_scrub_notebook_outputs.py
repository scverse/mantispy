"""Tests for the notebook-output path scrubber (.scripts/ci/scrub_notebook_outputs.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRUBBER = Path(__file__).resolve().parents[1] / ".scripts" / "ci" / "scrub_notebook_outputs.py"
_spec = importlib.util.spec_from_file_location("scrub_notebook_outputs", _SCRUBBER)
assert _spec is not None and _spec.loader is not None
scrub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scrub)
