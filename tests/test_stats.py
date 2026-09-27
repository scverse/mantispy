import numpy as np
import pytest

from mantispy._core._stats import (
    split_reference,
)


def test_split_reference_refuses_a_reference_it_cannot_halve():
    """One row per side is the least that can carry a null at all."""
    with pytest.raises(ValueError, match="at least four rows"):
        split_reference(np.arange(3), np.random.default_rng(0))
