"""Rank-based inverse normal transformation, against the reference implementation."""

import numpy as np
import pytest
import scipy.stats as ss

import mantispy as mt
from mantispy.pp._transform import BLOM, rank_inverse_normal


def reference(array: np.ndarray, c: float = BLOM, stochastic: bool = True, seed: int = 0) -> np.ndarray:
    """rank_int_array from broadinstitute/jump-profiling-recipe, with identical behaviour."""
    rng = np.random.default_rng(seed=seed)
    if stochastic:
        order = rng.permutation(len(array))
        rank = ss.rankdata(array[order], method="ordinal")[np.argsort(order)]
    else:
        rank = ss.rankdata(array, method="average")
    return ss.norm.ppf((rank - c) / (len(rank) - 2 * c + 1))


@pytest.mark.parametrize("stochastic", [True, False])
def test_it_matches_the_jump_recipe_exactly(stochastic):
    rng = np.random.default_rng(0)
    values = rng.normal(size=(500, 6))
    values[:, 3] = np.round(values[:, 3], 1)  # heavy ties, where the two tie modes differ

    ours = rank_inverse_normal(values, stochastic=stochastic)
    theirs = np.column_stack([reference(values[:, j], stochastic=stochastic) for j in range(values.shape[1])])
    np.testing.assert_allclose(ours, theirs, atol=1e-12)


def test_one_missing_well_does_not_erase_the_feature():
    """scipy's rankdata returns all-NaN for a column holding any NaN, so the reference
    implementation loses the whole feature to one unmeasured well."""
    values = np.array([[3.0], [1.0], [np.nan], [2.0]])

    assert np.isnan(reference(values.ravel())).all()

    ours = rank_inverse_normal(values)
    assert np.isnan(ours[2, 0])
    assert np.isfinite(ours[[0, 1, 3], 0]).all()
    assert ours[0, 0] > ours[3, 0] > ours[1, 0]  # the order of the measured values survives


def test_the_output_is_standard_normal_per_feature(cells):
    mt.pp.rank_int(cells)
    values = np.asarray(cells.X, dtype=float)
    np.testing.assert_allclose(values.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(values.std(axis=0), 1.0, atol=0.02)


def test_grouping_ranks_within_each_group(cells):
    mt.pp.rank_int(cells, by="Metadata_Plate", key_added="ranked")
    for plate in cells.obs["Metadata_Plate"].unique():
        rows = (cells.obs["Metadata_Plate"] == plate).to_numpy()
        block = np.asarray(cells.layers["ranked"], dtype=float)[rows]
        np.testing.assert_allclose(block.mean(axis=0), 0.0, atol=1e-6)
    assert not np.allclose(np.asarray(cells.X), np.asarray(cells.layers["ranked"]))


def test_it_is_reproducible(cells):
    first = mt.pp.rank_int(cells, copy=True)
    second = mt.pp.rank_int(cells, copy=True)
    np.testing.assert_array_equal(np.asarray(first.X), np.asarray(second.X))


def test_a_single_measured_value_is_not_erased():
    """The fewer-than-two-finite-values guard skipped the column without writing, so the
    NaN prefill was returned and a measured value came back missing, although the transform
    is well defined at n=1: ndtri((1 - 0.375) / (1 - 0.75 + 1)) = 0.0."""
    values = np.array([[1.0, np.nan], [np.nan, np.nan], [np.nan, 3.0], [np.nan, 4.0], [np.nan, 5.0]])

    out = rank_inverse_normal(values)
    assert out[0, 0] == pytest.approx(0.0)
    assert np.isnan(out[1:, 0]).all(), "only missing entries come back missing"
    assert np.isfinite(out[2:, 1]).all()


def test_rank_int_keeps_a_feature_measured_once_per_group():
    """Ranking within a group hit the same guard, so a feature measured in exactly one well
    per plate lost all three of its values under by='Metadata_Plate' while keeping them
    under by=None."""
    from mantispy.ds import synthetic_plate

    adata = synthetic_plate(n_plates=3, n_wells=6, n_cells=1, n_features=8, seed=0)
    plate = adata.obs["Metadata_Plate"].astype(str).to_numpy()
    values = np.asarray(adata.X, dtype=np.float64).copy()
    values[:, 1] = np.nan
    for index, name in enumerate(sorted(set(plate))):
        values[np.flatnonzero(plate == name)[0], 1] = 1.0 + index
    adata.X = values.astype(np.float32)

    mt.pp.rank_int(adata, by="Metadata_Plate", key_added="ranked")
    assert int(np.isfinite(np.asarray(adata.layers["ranked"])[:, 1]).sum()) == 3
