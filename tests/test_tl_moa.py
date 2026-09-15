"""Mechanism-of-action retrieval and neighbourhood enrichment."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy._core.schema import stamp

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


@pytest.mark.parametrize("scheme", ["nn", "nsc", "nscb"])
def test_classification_beats_chance(labelled, scheme):
    mt.tl.nn_moa_classify(labelled, scheme=scheme)
    summary = labelled.uns["mantispy"]["moa"]
    assert summary["scheme"] == scheme
    assert summary["accuracy"] > 1 / labelled.obs["Metadata_MOA"].nunique()
    assert summary["n_classified"] + summary["n_excluded"] == labelled.n_obs
    assert "moa_predicted" in labelled.obs


def test_not_same_compound_is_the_harder_task(labelled):
    """Excluding the same compound is what stops the benchmark being trivial."""
    mt.tl.nn_moa_classify(labelled, scheme="nn", key_added="nn")
    mt.tl.nn_moa_classify(labelled, scheme="nsc", key_added="nsc")
    assert labelled.uns["mantispy"]["nn"]["accuracy"] >= labelled.uns["mantispy"]["nsc"]["accuracy"]


def test_the_confusion_table_is_tidy_and_totals_correctly(labelled):
    mt.tl.nn_moa_classify(labelled)
    confusion = labelled.uns["mantispy"]["moa_confusion"]
    assert {"true", "predicted", "count"} <= set(confusion.columns)
    assert confusion["count"].sum() == labelled.uns["mantispy"]["moa"]["n_classified"]


def test_nscb_on_a_single_batch_classifies_nothing_and_says_so(labelled):
    """Every neighbour shares the batch, so there is nowhere to look. Returning an
    accuracy of 0.0 without a warning would read as a failed method rather than an impossible split."""
    single = labelled.copy()
    single.obs["Metadata_Batch"] = "one"
    with pytest.warns(UserWarning, match="excluded every neighbour"):
        mt.tl.nn_moa_classify(single, scheme="nscb")
    assert single.uns["mantispy"]["moa"]["n_excluded"] == single.n_obs
    assert single.obs["moa_predicted"].isna().all()


def test_enrichment_finds_the_right_neighbourhood(labelled):
    mt.tl.moa_enrichment(labelled, k=5)
    table = labelled.uns["mantispy"]["moa_enrichment"]
    assert {"group", "moa", "n_neighbours", "pvalue", "qvalue"} <= set(table.columns)

    truth = labelled.obs.groupby("Metadata_Perturbation", observed=True)["Metadata_MOA"].first().astype(str)
    best = table.loc[table.groupby("group")["pvalue"].idxmin()]
    assert (best["group"].map(truth).to_numpy() == best["moa"].to_numpy()).mean() > 0.5


def test_round_trip(labelled, tmp_path):
    mt.tl.nn_moa_classify(labelled)
    mt.tl.moa_enrichment(labelled, k=5)
    mt.io.write(labelled, tmp_path / "moa.h5ad")
    loaded = mt.io.read(tmp_path / "moa.h5ad")
    assert loaded.uns["mantispy"]["moa"]["accuracy"] > 0
    assert len(loaded.uns["mantispy"]["moa_enrichment"]) > 0


@pytest.fixture
def partly_annotated():
    """The normal state of a compound screen: most compounds have no mechanism on file.

    Six annotated compounds that cluster together and twenty-four unannotated ones that do
    not. ``labelled`` annotates every row, so it cannot exercise unannotated compounds.
    """
    n = 30
    values = np.random.default_rng(0).standard_normal((n, 6)).astype(np.float32)
    values[:6] += 5.0
    labels = np.where(np.arange(n) < 6, "tubulin", None)
    adata = ad.AnnData(
        values,
        obs=pd.DataFrame(
            {
                "Metadata_Perturbation": [f"c{i}" for i in range(n)],
                "Metadata_MOA": pd.array(labels, dtype="string"),
                "Metadata_Plate": "P1",
                "Metadata_Well": [f"A{i:03d}" for i in range(n)],
            },
            index=[str(i) for i in range(n)],
        ),
        var=pd.DataFrame(index=[f"Cells_AreaShape_f{i}" for i in range(6)]),
    )
    stamp(adata, resolution="perturbation")
    return adata


def test_moa_enrichment_survives_unannotated_compounds(partly_annotated):
    """On pandas 3, pd.Categorical(x.astype(str)) leaves NaN as NaN with code -1, and
    np.bincount then raises an opaque "'list' argument must have no negative elements"."""
    mt.tl.moa_enrichment(partly_annotated, moa_key="Metadata_MOA", k=3)
    table = partly_annotated.uns["mantispy"]["moa_enrichment"]
    assert "nan" not in set(table["moa"].astype(str))
    assert len(table) > 0
    # `top` draws k neighbours from every other profile, unannotated ones included, so the
    # null must size its population the same way. The six tubulin compounds cluster and
    # their k=3 neighbours are always each other, which the null must call unlikely. A null
    # sized on annotated rows alone sees an all-tubulin population and returns 1.0.
    assert (table["pvalue"] < 0.01).all(), table


def test_nn_moa_classify_scores_only_the_annotated_rows(partly_annotated):
    """nan == nan is False, so scoring unannotated rows would count each as a wrong answer
    (accuracy 0.2 over 30 rows on this fixture, where the metric is defined on 6)."""
    mt.tl.nn_moa_classify(partly_annotated, moa_key="Metadata_MOA", scheme="nn")
    # key_added defaults to "moa", so the summary lands at uns["mantispy"]["moa"].
    result = partly_annotated.uns["mantispy"]["moa"]
    assert result["n_classified"] == 6
    assert result["n_excluded"] == 24
    assert result["accuracy"] == pytest.approx(1.0)


@pytest.fixture
def unannotated_nearest_neighbour():
    """Four profiles: an annotated compound whose nearest neighbour is unannotated, a
    second annotated compound of the same mechanism one place further down, and an
    unannotated filler far from everyone.

    Row 0 (tubulin) is closest to row 1 (unannotated) and second-closest to row 2
    (tubulin). Row 2's own nearest neighbour is row 0, so masking the unannotated columns
    fixes row 0 without changing row 2. Masking rows instead of columns gets this fixture
    wrong.
    """

    def _direction(degrees: float) -> list[float]:
        radians = np.radians(degrees)
        return [float(np.cos(radians)), float(np.sin(radians))]

    values = np.array(
        [
            _direction(0),  # 0: tubulin
            _direction(3),  # 1: unannotated, nearest to 0
            _direction(-37),  # 2: tubulin, row 0's second-nearest and row 2's nearest
            _direction(150),  # 3: unannotated, far from everyone
        ],
        dtype=np.float32,
    )
    labels = ["tubulin", None, "tubulin", None]
    adata = ad.AnnData(
        values,
        obs=pd.DataFrame(
            {
                "Metadata_Perturbation": [f"c{i}" for i in range(4)],
                "Metadata_MOA": pd.array(labels, dtype="string"),
                "Metadata_Plate": "P1",
                "Metadata_Well": [f"A{i:02d}" for i in range(4)],
            },
            index=[str(i) for i in range(4)],
        ),
        var=pd.DataFrame(index=[f"Cells_AreaShape_f{i}" for i in range(2)]),
    )
    stamp(adata, resolution="perturbation")
    return adata


def test_an_unannotated_nearest_neighbour_does_not_poison_the_vote(unannotated_nearest_neighbour):
    """Unannotated profiles are left out of the neighbour search as well as the scoring.

    If argmax runs over every column, an annotated compound whose nearest profile is
    unannotated predicts NA and is scored wrong, although an annotated compound of the same
    mechanism is its next neighbour.
    """
    mt.tl.nn_moa_classify(unannotated_nearest_neighbour, moa_key="Metadata_MOA", scheme="nn")
    result = unannotated_nearest_neighbour.uns["mantispy"]["moa"]
    assert result["n_classified"] == 2
    assert result["accuracy"] == pytest.approx(1.0)
    predicted = unannotated_nearest_neighbour.obs["moa_predicted"]
    assert predicted.iloc[[0, 2]].notna().all()
