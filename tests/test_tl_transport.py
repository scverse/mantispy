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


def test_transport_finds_an_effect_present_in_both_batches(two_batches):
    mt.tl.transport(two_batches, by="Metadata_Plate")
    table = two_batches.uns["mantispy"]["transport"]

    assert set(table.columns) >= {"group", "level", "n_pairs", "agreement", "pvalue", "qvalue", "transports"}
    assert table["transports"].all(), "every perturbation carries the same injected effect in both batches"
    assert "transport_agreement" in two_batches.obs


def test_transport_is_calibrated_on_a_screen_where_nothing_happened():
    """On a screen with no effect, transport calls fewer than 15% of groups.

    The statistic is an average over unit pairs, so the null averages over the same unit
    pairs. A null of one pair per permutation has far more variance than a ten-pair average
    and makes the test roughly eightfold conservative.
    """
    called = total = 0
    for seed in range(4):
        cells = mt.ds.synthetic_plate(
            n_plates=4,
            n_wells=96,
            n_cells=10,
            n_features=30,
            n_batches=2,
            batch_effect=3.0,
            n_perturbations=9,
            effect_size=0.0,
            seed=seed,
        )
        wells = mt.tl.aggregate(cells, min_cells=0)
        mt.pp.normalize(wells, by="Metadata_Plate", reference="negcon")
        mt.tl.transport(wells, by=["Metadata_Batch", "Metadata_Plate"])
        table = wells.uns["mantispy"]["transport"]
        called += int(table["transports"].sum())
        total += len(table)
    assert total >= 40
    assert called / total < 0.15, f"{called}/{total} groups called on a screen with no effect in it"


def test_a_hierarchy_reports_each_level_and_partitions_the_pairs(two_batches):
    """Every unit pair belongs to one level, the coarsest one on which it disagrees."""
    mt.tl.transport(two_batches, by=["Metadata_Batch", "Metadata_Plate"])
    table = two_batches.uns["mantispy"]["transport"]
    assert set(table["level"]) == {"Metadata_Batch", "Metadata_Plate"}

    # Four plates over two batches: two within-batch pairs, four across-batch pairs.
    per_level = table.groupby("level", observed=True)["n_pairs"].max()
    assert per_level["Metadata_Plate"] == 2
    assert per_level["Metadata_Batch"] == 4
    assert per_level.sum() == 6 == 4 * 3 // 2, "the levels partition the pairs rather than sharing them"


def test_transport_refuses_a_setting_that_does_not_vary(two_batches):
    two_batches.obs["Metadata_OnlyOne"] = "single"
    with pytest.raises(ValueError, match="has one level"):
        mt.tl.transport(two_batches, by="Metadata_OnlyOne")


def test_transport_refuses_a_setting_without_its_own_controls(two_batches):
    """The effect is measured against each setting's own reference wells."""
    plates = two_batches.obs["Metadata_Plate"].astype(str)
    orphan = plates == sorted(plates.unique())[0]
    two_batches.obs.loc[orphan & two_batches.obs["Metadata_Control"].to_numpy(), "Metadata_Control"] = False
    # One plate now has no controls; the other three still do, so this must survive.
    mt.tl.transport(two_batches, by="Metadata_Plate")
    matrix = two_batches.uns["mantispy"]["transport_units"]
    assert len(matrix) == 3, "a setting with no reference rows is dropped, not guessed at"


def test_activity_weighting_changes_the_answer(two_batches):
    """An inactive perturbation has no effect vector worth correlating.

    On JUMP, activity weighting moves the median agreement from +0.33 to +0.75, because
    the unweighted number is dominated by inactive compounds.
    """
    mt.tl.transport(two_batches, by="Metadata_Plate", weight="activity", key_added="weighted")
    mt.tl.transport(two_batches, by="Metadata_Plate", weight="equal", key_added="flat")
    weighted = two_batches.uns["mantispy"]["weighted"].set_index("group")["agreement"]
    flat = two_batches.uns["mantispy"]["flat"].set_index("group")["agreement"]
    assert not np.allclose(weighted.to_numpy(), flat.reindex(weighted.index).to_numpy())

    with pytest.raises(ValueError, match="weight must be one of"):
        mt.tl.transport(two_batches, by="Metadata_Plate", weight="inverse")


def test_the_units_matrix_is_symmetric_with_a_unit_diagonal(two_batches):
    mt.tl.transport(two_batches, by="Metadata_Plate")
    matrix = two_batches.uns["mantispy"]["transport_units"].to_numpy(dtype=float)
    np.testing.assert_allclose(matrix, matrix.T, equal_nan=True)
    np.testing.assert_allclose(np.diag(matrix), 1.0)


def test_results_survive_a_round_trip(two_batches, tmp_path):
    mt.tl.transport(two_batches, by="Metadata_Plate")
    mt.io.write(two_batches, tmp_path / "transport.h5ad")
    loaded = mt.io.read(tmp_path / "transport.h5ad")
    assert len(loaded.uns["mantispy"]["transport"]) == len(two_batches.uns["mantispy"]["transport"])
    assert loaded.uns["mantispy"]["transport_units"].shape == (4, 4)


def test_both_plots_draw(two_batches):
    mt.tl.transport(two_batches, by=["Metadata_Batch", "Metadata_Plate"])
    assert mt.pl.transport(two_batches) is not None
    assert mt.pl.transport(two_batches, level="Metadata_Batch") is not None
    assert mt.pl.setting_agreement(two_batches) is not None
    assert mt.pl.setting_agreement(two_batches, by="Metadata_Batch") is not None
    with pytest.raises(KeyError, match="no level"):
        mt.pl.transport(two_batches, level="Metadata_Nonsense")
    plt.close("all")


def test_there_is_no_sampling_parameter_to_get_wrong(two_batches):
    """The null is the complete set of mismatched pairs, so there is nothing to sample.

    A sampled null of n draws cannot go below p = 1/(n+1), and BH across every
    (group, level) row needs smaller p-values, so the number of groups called would depend
    on the permutation budget rather than the data.
    """
    import inspect

    parameters = inspect.signature(mt.tl.transport).parameters
    assert "n_permutations" not in parameters, "a complete null has no budget to tune"
    assert "seed" not in parameters, "and nothing to seed"

    first = mt.tl.transport(two_batches, by="Metadata_Plate", copy=True)
    second = mt.tl.transport(two_batches, by="Metadata_Plate", copy=True)
    left = first.uns["mantispy"]["transport"].set_index("group")
    right = second.uns["mantispy"]["transport"].set_index("group")
    np.testing.assert_array_equal(left["pvalue"].to_numpy(), right["pvalue"].reindex(left.index).to_numpy())


def test_the_null_is_every_mismatched_pair_not_a_sample(two_batches):
    """The p-value floor is set by how many mismatched pairs exist, not by a budget.

    With g groups and the complete null the smallest expressible p is 1/(g*(g-1)+1). A
    sampled null of n draws floors at 1/(n+1) regardless of how much data there is.
    """
    mt.tl.transport(two_batches, by="Metadata_Plate")
    table = two_batches.uns["mantispy"]["transport"]
    n_groups = table["group"].nunique()

    smallest = float(table["pvalue"].min())
    floor = 1.0 / (n_groups * (n_groups - 1) + 1)
    assert smallest >= floor
    assert smallest <= 1.0 / n_groups, "a complete null over g groups resolves finer than 1/g"
