"""The stamp reference page documents every core ``uns["mantispy"]`` key."""

from pathlib import Path

import pytest

from mantispy._core.schema import UNS_KEYS

PAGE = Path(__file__).resolve().parents[1] / "docs" / "data_model" / "stamp_reference.md"


@pytest.mark.parametrize("key", UNS_KEYS)
def test_stamp_reference_lists_key(key: str) -> None:
    assert f"`{key}`" in PAGE.read_text()
