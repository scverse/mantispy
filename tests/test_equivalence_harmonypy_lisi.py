"""LISI equivalence against harmonypy, the Python port of the Harmony and LISI code of :cite:t:`Korsunsky_2019`.

Published LISI values come from this code, so an iLISI of 1.49 is only comparable with a
paper if the values agree. A variant with the right monotonicity but different values
would pass every internal sanity check.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantispy as mt

harmonypy = pytest.importorskip("harmonypy")
from harmonypy.lisi import compute_lisi  # noqa: E402


def _embedded(n_categories: float, separation: float, n_obs: int = 300, seed: int = 0) -> ad.AnnData:
    """An embedding whose batches are separated by ``separation`` standard deviations."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, int(n_categories), n_obs)
    values = rng.normal(size=(n_obs, 10)) + separation * np.eye(10)[codes]
    adata = ad.AnnData(
        X=values.astype(np.float32),
        obs=pd.DataFrame({"Metadata_Batch": codes.astype(str)}, index=[str(i) for i in range(n_obs)]),
    )
    adata.obsm["X_pca"] = values
    return adata


# A squared-distance kernel fails in every regime here. A fixture with no batch overlap
# would agree under either kernel, so none is used.
@pytest.mark.parametrize(
    ("n_categories", "separation"),
    [(3, 0.0), (2, 1.5), (3, 2.0), (5, 4.0)],
    ids=["mixed", "partly-mixed", "overlapping", "separated"],
)
def test_lisi_matches_harmonypy(n_categories, separation):
    adata = _embedded(n_categories, separation)
    expected = float(np.median(compute_lisi(adata.obsm["X_pca"], adata.obs, ["Metadata_Batch"], perplexity=30)[:, 0]))

    value = float(mt.metrics.lisi(adata, key="Metadata_Batch")["value"].iloc[0])
    assert value == pytest.approx(expected, rel=1e-9)
