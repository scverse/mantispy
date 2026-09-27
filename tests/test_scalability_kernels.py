"""Correctness of the column-blocked kernels: block iteration and float32 correlation.

Chunking is a re-association of the same math, so the result must equal the whole-matrix
reference. float32 correlation only diverges near the threshold, which these tests keep clear of.
"""

import numpy as np
import pandas as pd
import pytest

from mantispy._core._corr import correlated_pairs


def test_windowed_rejects_bad_window_and_stride():
    """The fast path validates its knobs: window and stride must be positive integers."""
    rng = np.random.default_rng(13)
    X = rng.standard_normal((50, 6))
    for bad_window in (0, -1):
        with pytest.raises(ValueError):
            correlated_pairs(X, 0.9, window=bad_window)
    for bad_stride in (0, -2):
        with pytest.raises(ValueError):
            correlated_pairs(X, 0.9, window=3, stride=bad_stride)


def test_feature_select_corr_window_drops_one_of_a_pair():
    """The windowed correlation path actually drops a redundant feature and writes a boolean var column."""
    from anndata import AnnData

    import mantispy as mt

    rng = np.random.default_rng(7)
    n_obs = 60
    X = rng.standard_normal((n_obs, 6)).astype(np.float32)
    X[:, 1] = X[:, 0] + rng.standard_normal(n_obs).astype(np.float32) * 0.01
    names = [f"Cells_Intensity_{i}" for i in range(6)]
    adata = AnnData(X=X, var=pd.DataFrame(index=names))

    mt.pp.feature_select(adata, operations=("correlation_threshold",), corr_window=3, corr_stride=1)
    selected = adata.var["selected"]
    assert selected.dtype == bool
    assert selected.shape == (6,)
    assert int(selected.sum()) == 5  # one of the correlated pair is dropped
    assert not (bool(selected.iloc[0]) and bool(selected.iloc[1]))  # not both members kept
