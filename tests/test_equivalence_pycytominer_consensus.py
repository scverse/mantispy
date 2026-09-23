"""consensus equivalence against pycytominer, which is where the weighting rule comes from."""

import numpy as np
import pandas as pd
import pytest

import mantispy as mt

pycytominer = pytest.importorskip("pycytominer")


@pytest.mark.parametrize("correlation", ["spearman", "pearson"])
def test_modz_matches_pycytominer(correlation):
    cells = mt.ds.synthetic_plate(n_plates=2, n_wells=48, n_cells=10, n_features=12, seed=5)
    wells = mt.tl.aggregate(cells, min_cells=0)
    features = list(wells.var_names)
    frame = mt.get.to_dataframe(wells)

    expected = (
        pycytominer.consensus(
            profiles=frame,
            replicate_columns=["Metadata_Perturbation"],
            operation="modz",
            features=features,
            modz_args={"method": correlation},
        )
        .sort_values("Metadata_Perturbation")
        .reset_index(drop=True)
    )
    actual = (
        mt.get.to_dataframe(mt.tl.consensus(wells, method="modz", correlation=correlation))
        .sort_values("Metadata_Perturbation")
        .reset_index(drop=True)
    )
    np.testing.assert_allclose(
        actual[features].to_numpy(np.float64), expected[features].to_numpy(np.float64), rtol=1e-5, atol=1e-5
    )


def test_default_matches_pycytominer_median():
    """Both default to median, so the default profiles must match: the parity the docstring claims."""
    cells = mt.ds.synthetic_plate(n_plates=2, n_wells=48, n_cells=10, n_features=12, seed=5)
    wells = mt.tl.aggregate(cells, min_cells=0)
    features = list(wells.var_names)
    frame = mt.get.to_dataframe(wells)

    expected = (
        pycytominer.consensus(profiles=frame, replicate_columns=["Metadata_Perturbation"], features=features)
        .sort_values("Metadata_Perturbation")
        .reset_index(drop=True)
    )
    actual = mt.get.to_dataframe(mt.tl.consensus(wells)).sort_values("Metadata_Perturbation").reset_index(drop=True)
    np.testing.assert_allclose(
        actual[features].to_numpy(np.float64), expected[features].to_numpy(np.float64), rtol=1e-5, atol=1e-5
    )


def test_a_shared_gap_weights_the_replicates_as_pycytominer_does():
    """pycytominer correlates pairwise-complete, so a shared gap is neutral there; zero-filling the ranks diverged 0.33 in weight.

    pycytominer is the reference for the weighting rule, and its correlation comes from
    ``pandas.DataFrame.corr``, which drops a pair's missing features rather than filling them.
    """
    from pycytominer.cyto_utils.util import get_pairwise_correlation

    from mantispy.tl._consensus import modz_weights

    rng = np.random.default_rng(0)
    block = rng.normal(size=(4, 60))  # four replicates that agree on nothing
    block[np.ix_([0, 1], np.arange(20))] = np.nan  # the same features undefined in two of them

    # pycytominer's own steps on pycytominer's own correlation: samples are the columns, the
    # diagonal is dropped, anticorrelation counts as uninformative, and the weights sum to one.
    correlation = get_pairwise_correlation(pd.DataFrame(block).transpose(), method="spearman")[0]
    correlation = correlation.to_numpy(dtype=np.float64).copy()
    np.fill_diagonal(correlation, np.nan)
    expected = np.clip(np.nanmean(np.clip(correlation, 0.0, None), axis=1), 0.01, None)
    expected = np.round(expected / expected.sum(), 4)

    np.testing.assert_allclose(modz_weights(block), expected, atol=0.05)
