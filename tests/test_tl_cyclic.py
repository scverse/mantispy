"""Circular-covariate helpers: correlation, cycle orientation, neighbour alignment, feature association."""

import numpy as np
import pandas as pd

import mantispy as mt
from mantispy.tl._cyclic import (
    _wrap,
    circular_correlation,
    circular_neighbor_error,
    cyclic_shift,
    feature_cycle_association,
    orient_cycle,
)


def _object_obs(columns, n):
    """Object-resolution metadata carrying the ids from_embedding's stamp requires."""
    return pd.DataFrame(
        {**columns, "Metadata_ImageID": ["img"] * n, "Metadata_ObjectNumber": list(range(n))},
        index=[str(i) for i in range(n)],
    )


def test_circular_correlation_is_wrap_invariant_and_detects_a_relationship():
    rng = np.random.default_rng(0)
    theta = rng.uniform(-np.pi, np.pi, size=500)
    phi = _wrap(theta + 0.1 * rng.normal(size=500))
    r = circular_correlation(theta, phi)
    assert r > 0.8
    assert abs(r - circular_correlation(theta + 2 * np.pi, phi)) < 1e-9
    assert abs(circular_correlation(theta, rng.uniform(-np.pi, np.pi, size=500))) < 0.2


def test_orient_cycle_puts_the_peak_at_zero_and_orders_direction():
    theta = np.linspace(-np.pi, np.pi, 361, endpoint=False)
    peak = np.cos(theta - 1.0)  # S-score peak at angle 1.0
    direction = np.sin(theta - 1.0)  # rises just after the peak
    oriented = orient_cycle(theta, peak, direction)
    assert abs(oriented[np.argmax(peak)]) < 0.05
    mask = (oriented > 0) & (oriented < 1.0)
    assert np.corrcoef(oriented[mask], direction[mask])[0, 1] > 0


def test_circular_neighbor_error_is_small_for_adjacent_neighbours():
    theta = _wrap(np.linspace(0, 2 * np.pi, 200, endpoint=False))
    idx = np.arange(200)
    adjacent = np.c_[(idx - 1) % 200, (idx + 1) % 200]
    random = np.random.default_rng(0).integers(0, 200, size=(200, 2))
    assert circular_neighbor_error(theta, adjacent) < 0.2
    assert circular_neighbor_error(theta, adjacent) < circular_neighbor_error(theta, random)


def test_feature_cycle_association_recovers_a_planted_cycle_feature():
    rng = np.random.default_rng(1)
    n = 800
    theta = rng.uniform(-np.pi, np.pi, size=n)
    cycling = np.cos(theta - 0.5) + 0.3 * rng.normal(size=n)
    flat = rng.normal(size=n)
    adata = mt.io.from_embedding(
        np.c_[cycling, flat],
        _object_obs({"Metadata_Plate": ["P"] * n, "Metadata_Well": ["A01"] * n, "theta": theta}, n),
    )
    out = feature_cycle_association(adata, angle_key="theta").set_index("feature")
    assert out.loc["embedding_0", "r2"] > 0.6
    assert out.loc["embedding_1", "r2"] < 0.1
    assert abs(_wrap(out.loc["embedding_0", "peak_angle"] - 0.5)) < 0.2


def test_cyclic_shift_is_near_zero_for_the_reference_and_flags_a_shift():
    rng = np.random.default_rng(2)
    theta = np.r_[rng.vonmises(0.0, 4, 300), rng.vonmises(1.0, 4, 300)]
    obs = _object_obs(
        {
            "Metadata_Well": (["A01"] * 150 + ["A02"] * 150) * 2,
            "Metadata_Perturbation": ["NTC"] * 300 + ["KO"] * 300,
            "theta": theta,
        },
        600,
    )
    adata = mt.io.from_embedding(rng.normal(size=(600, 2)), obs)
    out = cyclic_shift(
        adata, angle_key="theta", by="Metadata_Perturbation", reference="NTC", block="Metadata_Well"
    ).set_index("group")
    assert abs(out.loc["NTC", "shift"]) < 1e-6
    assert out.loc["KO", "shift"] > 0.5
