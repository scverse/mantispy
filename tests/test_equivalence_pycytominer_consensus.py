"""modz equivalence against pycytominer, which is where the weighting rule comes from."""

import numpy as np
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
