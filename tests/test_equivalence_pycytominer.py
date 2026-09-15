"""Equivalence against pycytominer, which is a test-only dependency.

pycytominer is pinned in the ``test`` dependency group. It is never imported from
``src/``; these tests check that the numba reimplementations give the same results.
"""

import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy.ds import synthetic_plate

pycytominer = pytest.importorskip("pycytominer")

TOLERANCE = {"rtol": 1e-6, "atol": 1e-6}


@pytest.fixture
def cells():
    return synthetic_plate(n_plates=2, n_wells=24, n_cells=15, n_features=20, seed=1)


def _features(adata):
    return list(adata.var_names)


def _meta(frame):
    return [column for column in frame.columns if column.startswith("Metadata_")]


def _pycytominer_normalize(frame, features, method, samples="all"):
    """pycytominer normalizes one plate at a time; mantispy does it with by=."""
    return (
        frame.groupby("Metadata_Plate", group_keys=False)[frame.columns.tolist()]
        .apply(
            lambda block: pycytominer.normalize(
                profiles=block,
                features=features,
                meta_features=_meta(block),
                method=method,
                samples=samples,
            )
        )
        .loc[frame.index]
    )


@pytest.mark.parametrize("method", ["mad_robustize", "standardize", "robustize"])
@pytest.mark.parametrize("with_nan", [False, True], ids=["clean", "with_nan"])
def test_normalize_equivalence(cells, method, with_nan):
    if with_nan:
        values = cells.X.copy()
        values[0, 0] = np.nan
        cells.X = values

    frame = mt.get.to_dataframe(cells)
    features = _features(cells)
    expected = _pycytominer_normalize(frame, features, method)

    mt.pp.normalize(cells, method=method, by="Metadata_Plate")
    np.testing.assert_allclose(cells.X, expected[features].to_numpy(dtype=np.float32), **TOLERANCE)


def test_normalize_on_controls_equivalence(cells):
    frame = mt.get.to_dataframe(cells)
    features = _features(cells)
    expected = _pycytominer_normalize(frame, features, "mad_robustize", samples="Metadata_Perturbation == 'DMSO'")

    mt.pp.normalize(cells, method="mad_robustize", by="Metadata_Plate", reference="negcon")
    np.testing.assert_allclose(cells.X, expected[features].to_numpy(dtype=np.float32), **TOLERANCE)


@pytest.mark.parametrize("func", ["median", "mean"])
@pytest.mark.parametrize("with_nan", [False, True], ids=["clean", "with_nan"])
def test_aggregate_equivalence(cells, func, with_nan):
    if with_nan:
        values = cells.X.copy()
        values[0, 0] = np.nan
        cells.X = values

    frame = mt.get.to_dataframe(cells)
    features = _features(cells)
    expected = (
        pycytominer.aggregate(
            population_df=frame,
            strata=["Metadata_Plate", "Metadata_Well"],
            features=features,
            operation=func,
        )
        .sort_values(["Metadata_Plate", "Metadata_Well"])
        .reset_index(drop=True)
    )

    wells = mt.tl.aggregate(cells, func=func, min_cells=0)
    actual = mt.get.to_dataframe(wells).sort_values(["Metadata_Plate", "Metadata_Well"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        actual[["Metadata_Plate", "Metadata_Well"]].astype(str),
        expected[["Metadata_Plate", "Metadata_Well"]].astype(str),
    )
    np.testing.assert_allclose(
        actual[features].to_numpy(dtype=np.float64),
        expected[features].to_numpy(dtype=np.float64),
        **TOLERANCE,
    )


def test_normalize_then_aggregate_pipeline_equivalence(cells):
    frame = mt.get.to_dataframe(cells)
    features = _features(cells)
    normalized = _pycytominer_normalize(frame, features, "mad_robustize")
    expected = (
        pycytominer.aggregate(
            population_df=normalized,
            strata=["Metadata_Plate", "Metadata_Well"],
            features=features,
            operation="median",
        )
        .sort_values(["Metadata_Plate", "Metadata_Well"])
        .reset_index(drop=True)
    )

    mt.pp.normalize(cells, method="mad_robustize", by="Metadata_Plate")
    wells = mt.tl.aggregate(cells, func="median", min_cells=0)
    actual = mt.get.to_dataframe(wells).sort_values(["Metadata_Plate", "Metadata_Well"]).reset_index(drop=True)
    np.testing.assert_allclose(
        actual[features].to_numpy(dtype=np.float64),
        expected[features].to_numpy(dtype=np.float64),
        **TOLERANCE,
    )
