"""Replicate power, and telling a phenotype from cell death."""

import numpy as np
import pytest

import mantispy as mt
from mantispy._core.schema import stamp


@pytest.fixture
def profiles():
    cells = mt.ds.synthetic_plate(
        n_plates=2, n_wells=96, n_cells=20, n_features=25, n_perturbations=5, effect_size=4.0, seed=0
    )
    mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon")
    return mt.tl.aggregate(cells, min_cells=0)


def test_saturation_improves_with_replicates(profiles):
    """The curve is the answer to 'how many replicates do I need'."""
    mt.tl.replicate_saturation(profiles, n_draws=3, max_replicates=4)
    table = profiles.uns["mantispy"]["replicate_saturation"]
    assert set(table["n_replicates"]) == {1, 2, 3, 4}
    assert table.sort_values("n_replicates")["mean"].is_monotonic_increasing


def test_saturation_is_reproducible_and_stops_when_it_runs_out(profiles):
    mt.tl.replicate_saturation(profiles, n_draws=3, seed=1)
    first = profiles.uns["mantispy"]["replicate_saturation"]
    mt.tl.replicate_saturation(profiles, n_draws=3, seed=1)
    np.testing.assert_allclose(first["mean"].to_numpy(), profiles.uns["mantispy"]["replicate_saturation"]["mean"])
    # Two disjoint subsets are needed, so the deepest depth is half the largest group.
    assert first["n_replicates"].max() <= profiles.obs["Metadata_Perturbation"].value_counts().max() // 2


def test_a_custom_metric_is_accepted(profiles):
    def always(profiles_, codes, depth, generator):
        return float(depth)

    mt.tl.replicate_saturation(profiles, metric=always, n_draws=2, max_replicates=3)
    assert profiles.uns["mantispy"]["replicate_saturation"]["mean"].tolist() == [1.0, 2.0, 3.0]


def test_cytotoxicity_flags_cell_loss_not_morphology(profiles):
    """A group that lost most of its cells is suspect even when it scores as a hit."""
    counts = profiles.obs["Metadata_CellCount"].to_numpy().copy()
    toxic = (profiles.obs["Metadata_Perturbation"] == "pert00").to_numpy()
    counts[toxic] = counts[toxic] // 5
    profiles.obs["Metadata_CellCount"] = counts
    profiles.obs["hits_row_distance"] = np.where(toxic, 10.0, 1.0)

    mt.tl.cytotoxicity(profiles)
    table = profiles.uns["mantispy"]["cytotoxicity"].set_index("group")
    assert table.loc["pert00", "viability"] < 0.5
    assert bool(table.loc["pert00", "suspect"])
    assert not bool(table.loc["DMSO", "suspect"])
    assert profiles.obs["cytotoxicity_suspect"].to_numpy()[toxic].all()


def test_cell_loss_alone_is_not_suspect(profiles):
    """Losing cells is a phenotype; only cell loss together with a large distance is suspect."""
    counts = profiles.obs["Metadata_CellCount"].to_numpy().copy()
    quiet = (profiles.obs["Metadata_Perturbation"] == "pert01").to_numpy()
    counts[quiet] = counts[quiet] // 5
    profiles.obs["Metadata_CellCount"] = counts
    profiles.obs["hits_row_distance"] = 1.0

    mt.tl.cytotoxicity(profiles)
    table = profiles.uns["mantispy"]["cytotoxicity"].set_index("group")
    assert table.loc["pert01", "viability"] < 0.5
    assert not bool(table.loc["pert01", "suspect"])


def test_cytotoxicity_medians_the_rows_rather_than_a_group_statistic(profiles):
    """Regression for #84.

    ``hits_distance`` is one number repeated over a group's rows, so the median this function
    documents returned the value it was handed, and no genuinely per-row response could be given.
    """
    toxic = (profiles.obs["Metadata_Perturbation"] == "pert00").to_numpy()
    counts = profiles.obs["Metadata_CellCount"].to_numpy().copy()
    counts[toxic] = counts[toxic] // 5
    profiles.obs["Metadata_CellCount"] = counts
    # A quiet group with one well far out, and the group statistic that would hide the difference.
    rows = np.ones(profiles.n_obs)
    rows[np.flatnonzero(toxic)[0]] = 100.0
    profiles.obs["hits_row_distance"] = rows
    profiles.obs["hits_distance"] = np.where(toxic, 100.0, 1.0)

    mt.tl.cytotoxicity(profiles)
    table = profiles.uns["mantispy"]["cytotoxicity"].set_index("group")
    assert table.loc["pert00", "viability"] < 0.5
    assert table.loc["pert00", "distance"] == 1.0, "one well far out does not move the group's median"
    assert not bool(table.loc["pert00", "suspect"])


def test_cytotoxicity_says_what_to_run_first(profiles):
    with pytest.raises(KeyError, match="mt.tl.hit_calling"):
        mt.tl.cytotoxicity(profiles, distance_key="not_computed")


def test_a_missing_count_names_where_to_get_one(profiles):
    del profiles.obs["Metadata_CellCount"]
    profiles.obs["hits_row_distance"] = 1.0
    with pytest.raises(KeyError, match="mt.tl.aggregate.*mt.ds.*count_key="):
        mt.tl.cytotoxicity(profiles)


def test_viability_compares_cells_per_field(profiles):
    """A well imaged at half its fields holds half the cells without having lost any."""
    halved = (profiles.obs["Metadata_Perturbation"] == "pert01").to_numpy()
    profiles.obs["Metadata_SiteCount"] = np.where(halved, 2.0, 4.0)
    profiles.obs["Metadata_CellCount"] = np.where(halved, 10.0, 20.0)
    profiles.obs["hits_row_distance"] = 1.0

    mt.tl.cytotoxicity(profiles)
    assert profiles.uns["mantispy"]["cytotoxicity"].set_index("group").loc["pert01", "viability"] == 1.0
    mt.tl.cytotoxicity(profiles, site_key=None)
    assert profiles.uns["mantispy"]["cytotoxicity"].set_index("group").loc["pert01", "viability"] == 0.5


def test_convergence_reaches_deeper_than_disjoint_halves(profiles):
    """Convergence reaches n - 1 replicates where disjoint halves stop at n // 2, so three
    replicates give a curve of two points instead of one."""
    mt.tl.replicate_saturation(profiles, metric="convergence", n_draws=3, key_added="converging")
    mt.tl.replicate_saturation(profiles, n_draws=3)
    deepest = profiles.obs["Metadata_Perturbation"].value_counts().max()
    assert profiles.uns["mantispy"]["converging"]["n_replicates"].max() == deepest - 1
    assert profiles.uns["mantispy"]["replicate_saturation"]["n_replicates"].max() == deepest // 2
    curve = profiles.uns["mantispy"]["converging"].sort_values("n_replicates")["mean"]
    # A sampled estimate at three draws is not strictly monotonic, but it rises.
    assert curve.iloc[-1] > curve.iloc[0]


def test_an_unknown_metric_lists_the_known_ones(profiles):
    with pytest.raises(ValueError, match="signature_stability"):
        mt.tl.replicate_saturation(profiles, metric="nope")


def test_saturation_depth_comes_from_the_replicates_not_the_controls():
    """A screen's largest group is its negative control, and it must not set the range.

    JUMP TARGET2 has 8505 DMSO wells against a median group of 132. Taking the deepest
    depth from the largest group would ask for 4252 wells, and past about 66 only DMSO can
    supply them, so the reported median would be one group correlated with itself.
    """
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(0)
    # Twelve treatments with six wells each, and one enormous control group.
    labels = ["negcon"] * 400 + [f"pert{i:02d}" for i in range(12) for _ in range(6)]
    obs = pd.DataFrame(
        {
            "Metadata_Plate": "P1",
            "Metadata_Well": [f"{chr(65 + i // 24)}{i % 24 + 1:02d}" for i in range(len(labels))],
            "Metadata_Perturbation": labels,
            "Metadata_Control": [label == "negcon" for label in labels],
        },
        index=[str(i) for i in range(len(labels))],
    )
    adata = ad.AnnData(X=rng.normal(size=(len(labels), 12)).astype(np.float32), obs=obs)
    stamp(adata, resolution="well")

    mt.tl.replicate_saturation(adata, n_draws=2)
    depths = adata.uns["mantispy"]["replicate_saturation"]["n_replicates"]
    assert depths.max() == 3, "six-well treatments support depth 3, not the controls' 200"

    mt.tl.replicate_saturation(adata, n_draws=2, min_groups=1, key_added="everything")
    everything = adata.uns["mantispy"]["everything"]["n_replicates"]
    assert everything.max() == 200, "min_groups=1 restores the range the controls set"
