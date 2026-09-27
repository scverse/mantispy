import numpy as np
import pytest

from mantispy._core._numba import (
    MEDIAN,
    grouped_median_spread,
)

# --- kernels ---------------------------------------------------------------


def test_median_spread_refuses_a_stat_that_is_not_a_spread():
    """MEDIAN reads as a valid selector everywhere else, and would silently give the IQR here."""
    with pytest.raises(ValueError, match="MAD or IQR"):
        grouped_median_spread(np.ones((4, 2), dtype=np.float32), np.zeros(4, dtype=np.int32), 1, MEDIAN)


# --- the seam --------------------------------------------------------------
