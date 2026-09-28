"""Mechanism-of-action retrieval and neighbourhood enrichment."""

import pytest

import mantispy as mt

MECHANISMS = {"pert00": "A", "pert01": "A", "pert02": "B", "pert03": "B", "pert04": "C", "pert05": "C"}


@pytest.fixture
def labelled():
    """Perturbations grouped into mechanisms, two compounds per mechanism."""
    cells = mt.ds.synthetic_plate(
        n_plates=2, n_wells=96, n_cells=10, n_features=30, n_perturbations=6, effect_size=4.0, seed=0
    )
    wells = mt.tl.aggregate(cells, min_cells=0)
    perturbation = wells.obs["Metadata_Perturbation"].astype(str)
    wells.obs["Metadata_Compound"] = perturbation.to_numpy()
    wells.obs["Metadata_MOA"] = perturbation.map(lambda name: MECHANISMS.get(name, "DMSO")).to_numpy()
    # synthetic_plate puts both plates in one batch; nscb needs two to have anywhere to look.
    wells.obs["Metadata_Batch"] = wells.obs["Metadata_Plate"].astype(str).to_numpy()
    return wells[wells.obs["Metadata_MOA"] != "DMSO"].copy()


def test_nscb_on_a_single_batch_classifies_nothing_and_says_so(labelled):
    """Every neighbour shares the batch, so there is nowhere to look.

    Returning an accuracy of 0.0 without a warning would read as a failed method rather than an impossible split.
    """
    single = labelled.copy()
    single.obs["Metadata_Batch"] = "one"
    with pytest.warns(UserWarning, match="excluded every neighbour"):
        mt.tl.nn_moa_classify(single, scheme="nscb")
    assert single.uns["mantispy"]["moa"]["n_excluded"] == single.n_obs
    assert single.obs["moa_predicted"].isna().all()
