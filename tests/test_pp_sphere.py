"""Sphering on reference sets that are too small to define a covariance."""

import numpy as np
import pytest

import mantispy as mt


def test_a_plate_with_one_control_well_raises_rather_than_zeroing_it(wells):
    """With a single reference row the rank guard `rank == n_obs - 1` reads `0 == 0`, the
    padded singular values are all zero, and the plate's X would come back all zero."""
    control = np.zeros(wells.n_obs, dtype=bool)
    plate = wells.obs["Metadata_Plate"].to_numpy()
    control[np.flatnonzero(plate == plate[0])[:4]] = True
    control[np.flatnonzero(plate != plate[0])[0]] = True  # a single control well on the other plate
    wells.obs["Metadata_Control"] = control

    with pytest.raises(ValueError, match="at least 2 reference rows"):
        mt.pp.sphere(wells, method="ZCA", by="Metadata_Plate")
