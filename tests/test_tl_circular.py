"""Circular-covariate helpers: correlation, cycle orientation, neighbour alignment, shift, feature association."""

import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy.tl._circular import (
    _wrap,
    circular_correlation,
    circular_neighbor_error,
    orient_cycle,
)


def _object_obs(columns, n):
    """Object-resolution metadata carrying the ids from_embedding's stamp requires."""
    return pd.DataFrame(
        {**columns, "Metadata_ImageID": ["img"] * n, "Metadata_ObjectNumber": list(range(n))},
        index=[str(i) for i in range(n)],
    )


def _cycling(n, seed, theta=None):
    """An object with one cycling feature (peak at 0.5 rad), one flat feature, and the angle in obs."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, size=n) if theta is None else theta
    cycling = np.cos(theta - 0.5) + 0.3 * rng.normal(size=n)
    flat = rng.normal(size=n)
    obs = _object_obs({"Metadata_Plate": ["P"] * n, "Metadata_Well": ["A01"] * n, "theta": theta}, n)
    return mt.io.from_embedding(np.c_[cycling, flat], obs)


def _association(adata, **kwargs):
    mt.tl.feature_cycle_association(adata, angle_key="theta", **kwargs)
    return adata.uns["mantispy"]["feature_cycle_association"].set_index("feature")


def test_circular_correlation_is_wrap_invariant_and_detects_a_relationship():
    rng = np.random.default_rng(0)
    theta = rng.uniform(-np.pi, np.pi, size=500)
    phi = _wrap(theta + 0.1 * rng.normal(size=500))
    r = circular_correlation(theta, phi)
    assert r > 0.8
    assert abs(r - circular_correlation(theta + 2 * np.pi, phi)) < 1e-9
    assert abs(circular_correlation(theta, rng.uniform(-np.pi, np.pi, size=500))) < 0.2
    assert circular_correlation(np.degrees(theta), np.degrees(phi), period=360) == pytest.approx(r)


def test_orient_cycle_puts_the_peak_at_zero_and_orders_direction():
    theta = np.linspace(-np.pi, np.pi, 361, endpoint=False)
    peak = np.cos(theta - 1.0)  # S-score peak at angle 1.0
    direction = np.sin(theta - 1.0)  # rises just after the peak
    oriented = orient_cycle(theta, peak, direction)
    assert abs(oriented[np.argmax(peak)]) < 0.05
    mask = (oriented > 0) & (oriented < 1.0)
    assert np.corrcoef(oriented[mask], direction[mask])[0, 1] > 0


def test_orient_cycle_ignores_a_score_offset_where_cells_crowd():
    """Cells crowd near angle -2 and the score carries a large offset; the zero still lands on the score's peak."""
    rng = np.random.default_rng(3)
    theta = _wrap(np.r_[rng.vonmises(-2.0, 3.0, 3000), rng.uniform(-np.pi, np.pi, 1000)])
    peak = 10.0 + np.cos(theta - 1.0)
    direction = 10.0 + np.sin(theta - 1.0)
    oriented = orient_cycle(theta, peak, direction)
    assert abs(_wrap(theta[np.argmin(np.abs(oriented))] - 1.0)) < 0.05


def test_orient_cycle_skips_missing_values_and_keeps_units():
    theta = np.linspace(-np.pi, np.pi, 360, endpoint=False)
    direction = np.sin(theta - 1.0)
    peak = np.cos(theta - 1.0)
    peak[5] = np.nan
    gap = theta.copy()
    gap[7] = np.nan
    oriented = orient_cycle(gap, peak, direction)
    assert np.isnan(oriented[7])
    assert np.isfinite(np.delete(oriented, 7)).all()
    clean = np.cos(theta - 1.0)
    degrees = orient_cycle(np.degrees(theta), clean, direction, period=360)
    np.testing.assert_allclose(degrees, np.degrees(orient_cycle(theta, clean, direction)), atol=1e-9)


def test_circular_neighbor_error_is_small_for_adjacent_neighbours():
    theta = _wrap(np.linspace(0, 2 * np.pi, 200, endpoint=False))
    idx = np.arange(200)
    adjacent = np.c_[(idx - 1) % 200, (idx + 1) % 200]
    random = np.random.default_rng(0).integers(0, 200, size=(200, 2))
    assert circular_neighbor_error(theta, adjacent) < 0.2
    assert circular_neighbor_error(theta, adjacent) < circular_neighbor_error(theta, random)


def test_feature_cycle_association_recovers_a_planted_cycle_feature():
    out = _association(_cycling(800, seed=1))
    assert out.loc["embedding_0", "r2"] > 0.6
    assert out.loc["embedding_1", "r2"] < 0.1
    assert abs(_wrap(out.loc["embedding_0", "peak_angle"] - 0.5)) < 0.05
    assert out.loc["embedding_0", "split_half_r"] > 0.95
    assert out.loc["embedding_1", "split_half_r"] < 0.9


def test_feature_cycle_association_reads_turns():
    adata = _cycling(800, seed=1)
    radians = _association(adata.copy())
    adata.obs["theta"] = adata.obs["theta"] / (2 * np.pi)
    turns = _association(adata, period=1.0)
    np.testing.assert_allclose(turns["r2"], radians["r2"])
    expected = radians.loc["embedding_0", "peak_angle"] / (2 * np.pi)
    assert turns.loc["embedding_0", "peak_angle"] == pytest.approx(expected)


def test_feature_cycle_association_needs_more_harmonics_for_a_sharp_drop():
    """DNA rises through the cycle and halves at division, a sawtooth one sinusoid follows only in part."""
    rng = np.random.default_rng(4)
    theta = rng.uniform(-np.pi, np.pi, size=2000)
    adata = _cycling(2000, seed=4, theta=theta)
    adata.X = adata.X.copy()
    adata.X[:, 0] = theta + 0.1 * rng.normal(size=2000)  # sawtooth: jumps from +pi to -pi
    one = _association(adata.copy()).loc["embedding_0", "r2"]
    three = _association(adata, n_harmonics=3).loc["embedding_0", "r2"]
    assert three > one + 0.1


def test_feature_cycle_association_handles_missing_values():
    """A NaN angle (one cell) or a NaN feature value must be dropped, not crash or silently zero the feature."""
    adata = _cycling(800, seed=5)
    adata.X = adata.X.copy()
    adata.X[1, 0] = np.nan
    adata.obs.loc[adata.obs_names[0], "theta"] = np.nan
    out = _association(adata)
    assert np.isfinite(out.loc["embedding_0", "r2"])
    assert out.loc["embedding_0", "r2"] > 0.5
    assert out.loc["embedding_1", "r2"] < 0.1


def _shift_object(n_wells, shift, cells_per_guide=40, guides=4, seed=2):
    """NTC and KO guides in every well; KO cells sit ``shift`` radians further round the cycle."""
    rng = np.random.default_rng(seed)
    rows = []
    for well in range(n_wells):
        for gene, centre in (("NTC", 0.0), ("KO", shift)):
            for guide in range(guides):
                # Each guide has its own small offset, so the guide, not the cell, is the replicate.
                offset = rng.normal(0.0, 0.1)
                for angle in rng.vonmises(centre + offset, 4.0, cells_per_guide):
                    rows.append((f"A{well + 1:02d}", gene, f"{gene}_g{guide}", angle))
    frame = pd.DataFrame(rows, columns=["Metadata_Well", "Metadata_Perturbation", "Metadata_Guide", "theta"])
    obs = _object_obs(frame.to_dict("list"), len(frame))
    return mt.io.from_embedding(np.zeros((len(frame), 1)), obs)


def _shift(adata, **kwargs):
    mt.tl.circular_shift(
        adata, angle_key="theta", by="Metadata_Perturbation", reference="NTC", block="Metadata_Well", **kwargs
    )
    return adata.uns["mantispy"]["circular_shift"].set_index("group")


def test_circular_shift_flags_a_shift_and_drops_the_reference_row():
    with pytest.warns(UserWarning, match="fewer than 6 blocks"):
        out = _shift(_shift_object(2, 1.0))
    assert "NTC" not in out.index
    assert out.loc["KO", "shift"] > 0.8
    assert {"pvalue", "qvalue", "n_blocks"} <= set(out.columns)


def test_circular_shift_permutation_over_guides_works_with_three_wells():
    """With three wells the signed-rank test cannot reach 0.05; shuffling guides within wells can."""
    moved = _shift(_shift_object(3, 1.0), unit="Metadata_Guide", n_permutations=199)
    assert moved.loc["KO", "pvalue"] < 0.05
    null = _shift(_shift_object(3, 0.0, seed=7), unit="Metadata_Guide", n_permutations=199)
    assert null.loc["KO", "pvalue"] > 0.05


def test_circular_shift_reports_in_the_units_it_was_given():
    adata = _shift_object(6, 1.0)
    radians = _shift(adata.copy()).loc["KO", "shift"]
    adata.obs["theta"] = np.degrees(adata.obs["theta"])
    assert _shift(adata, period=360).loc["KO", "shift"] == pytest.approx(np.degrees(radians))


def test_circular_shift_refuses_a_guide_in_two_groups():
    adata = _shift_object(3, 1.0)
    adata.obs["Metadata_Guide"] = "one_guide"
    with pytest.raises(ValueError, match="span more than one group"):
        _shift(adata, unit="Metadata_Guide", n_permutations=9)
