"""Does an effect survive the move to another setting, and is the answer calibrated."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import mantispy as mt


@pytest.fixture
def two_batches():
    """Four plates over two batches, with a real effect present in both."""
    cells = mt.ds.synthetic_plate(
        n_plates=4,
        n_wells=96,
        n_cells=10,
        n_features=30,
        n_batches=2,
        batch_effect=3.0,
        n_perturbations=9,
        effect_size=3.0,
    )
    wells = mt.tl.aggregate(cells, min_cells=0)
    mt.pp.normalize(wells, by="Metadata_Plate", reference="negcon")
    return wells


def test_transport_refuses_a_setting_that_does_not_vary(two_batches):
    two_batches.obs["Metadata_OnlyOne"] = "single"
    with pytest.raises(ValueError, match="has one level"):
        mt.tl.transport(two_batches, by="Metadata_OnlyOne")


def test_transport_refuses_a_setting_without_its_own_controls(two_batches):
    """The effect is measured against each setting's own reference wells."""
    plates = two_batches.obs["Metadata_Plate"].astype(str)
    orphan = plates == sorted(plates.unique())[0]
    drop = orphan & two_batches.obs["Metadata_Control"].to_numpy()
    two_batches.obs.loc[drop, "Metadata_Control"] = False
    two_batches.obs.loc[drop, "Metadata_Control_Type"] = "treatment"
    mt.tl.transport(two_batches, by="Metadata_Plate")
    matrix = two_batches.uns["mantispy"]["transport_units"]
    assert len(matrix) == 3, "a setting with no reference rows is dropped, not guessed at"


def test_activity_weighting_changes_the_answer(two_batches):
    """An inactive perturbation has no effect vector worth correlating.

    On JUMP, activity weighting moves the median agreement from +0.33 to +0.75, because the unweighted number is dominated by inactive compounds.
    """
    mt.tl.transport(two_batches, by="Metadata_Plate", weight="activity", key_added="weighted")
    mt.tl.transport(two_batches, by="Metadata_Plate", weight="equal", key_added="flat")
    weighted = two_batches.uns["mantispy"]["weighted"].set_index("group")["agreement"]
    flat = two_batches.uns["mantispy"]["flat"].set_index("group")["agreement"]
    assert not np.allclose(weighted.to_numpy(), flat.reindex(weighted.index).to_numpy())

    with pytest.raises(ValueError, match="weight must be one of"):
        mt.tl.transport(two_batches, by="Metadata_Plate", weight="inverse")


def test_both_plots_draw_what_the_table_holds(two_batches):
    """An axes that drew nothing is still not None, so a plot scrambling its rows passed these checks."""
    mt.tl.transport(two_batches, by=["Metadata_Batch", "Metadata_Plate"])
    table = two_batches.uns["mantispy"]["transport"]

    for level in ("Metadata_Plate", "Metadata_Batch"):
        block = table[table["level"] == level].sort_values("agreement", ascending=False)
        assert len(block) <= 25, "beyond top= only the extremes are drawn, and this compares them all"
        _, ax = plt.subplots()
        mt.pl.transport(two_batches, level=level, ax=ax)
        assert [label.get_text() for label in ax.get_yticklabels()] == list(block["group"].astype(str))
        widths = [patch.get_width() for patch in ax.patches]
        np.testing.assert_allclose(widths, block["agreement"].to_numpy(), rtol=1e-6)
        assert ax.get_title() == f"{int(block['transports'].sum())} of {len(block)} reproduce"

    matrix = two_batches.uns["mantispy"]["transport_units"]
    _, ax = plt.subplots()
    mt.pl.setting_agreement(two_batches, ax=ax)
    labels = [label.get_text() for label in ax.get_xticklabels()]
    assert sorted(labels) == sorted(str(name) for name in matrix.columns)
    assert [label.get_text() for label in ax.get_yticklabels()] == labels
    drawn = np.asarray(ax.get_images()[0].get_array(), dtype=float)
    np.testing.assert_allclose(drawn, matrix.loc[labels, labels].to_numpy(dtype=float), equal_nan=True)

    _, ax = plt.subplots()
    mt.pl.setting_agreement(two_batches, by="Metadata_Batch", ax=ax)
    labels = [label.get_text() for label in ax.get_xticklabels()]
    obs = two_batches.obs
    batch_of = obs.groupby(obs["Metadata_Plate"].astype(str), observed=True)["Metadata_Batch"].first().astype(str)
    annotation = [batch_of[label] for label in labels]
    assert annotation == sorted(annotation), f"plates are grouped by batch, got {annotation}"
    assert len(ax.lines) == 2, "one horizontal and one vertical line at the single batch boundary"

    with pytest.raises(KeyError, match="no level"):
        mt.pl.transport(two_batches, level="Metadata_Nonsense")
    plt.close("all")


def test_an_empty_transport_table_says_so(two_batches):
    """Regression test for #55: pl.transport failed on ``levels[-1]``, where its siblings name the empty table."""
    mt.tl.transport(two_batches, by="Metadata_Plate")
    store = two_batches.uns["mantispy"]
    store["transport"] = store["transport"].iloc[:0]

    with pytest.raises(ValueError, match="is empty"):
        mt.pl.transport(two_batches)
