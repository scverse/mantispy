"""mAP, similarity, percent replicating and grit."""

import warnings
from importlib.util import find_spec

import numpy as np
import pytest

import mantispy as mt
from mantispy.ds import synthetic_plate

# tl.map runs copairs, which declares requires-python <3.13 until that is lifted upstream.
requires_copairs = pytest.mark.skipif(find_spec("copairs") is None, reason="copairs does not install on this Python")


@pytest.fixture
def profiles():
    cells = synthetic_plate(
        n_plates=2, n_wells=96, n_cells=10, n_features=20, n_perturbations=5, effect_size=4.0, seed=0
    )
    mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon")
    return mt.tl.aggregate(cells, min_cells=0)


# --- mAP -------------------------------------------------------------------


@requires_copairs
def test_activity_scores_treatments_against_the_controls(profiles):
    """Phenotypic activity as the field defines it: replicates retrieved against control
    profiles only, with the controls themselves dropped from the result."""
    mt.tl.map(profiles, mode="activity", null_size=200)
    table = profiles.uns["mantispy"]["map"]

    assert "DMSO" not in set(table["Metadata_Perturbation"])
    assert table["mean_average_precision"].median() > 0.9  # every injected effect is real
    assert {"map", "map_qvalue"} <= set(profiles.obs.columns)


@requires_copairs
def test_activity_and_replicability_are_different_questions(profiles):
    """They differ by an order of magnitude on real data (JUMP: 0.93 against 0.09), so they
    have separate names."""
    mt.tl.map(profiles, mode="activity", null_size=200, key_added="activity")
    mt.tl.map(profiles, mode="replicability", null_size=200, key_added="replicability")

    activity = profiles.uns["mantispy"]["activity"].set_index("Metadata_Perturbation")
    replicability = profiles.uns["mantispy"]["replicability"].set_index("Metadata_Perturbation")

    # The controls are a group like any other under replicability, and the thing every
    # other group is measured against under activity.
    assert "DMSO" in replicability.index
    assert "DMSO" not in activity.index
    assert set(activity.index) == set(replicability.index) - {"DMSO"}


@requires_copairs
def test_activity_needs_controls_and_says_so(profiles):
    treated = profiles[~profiles.obs["Metadata_Control"].to_numpy()].copy()
    with pytest.raises(ValueError, match="mode='replicability'"):
        mt.tl.map(treated, mode="activity", null_size=100)


@requires_copairs
def test_consistency_groups_by_an_annotation(profiles):
    """Phenotypic consistency :cite:p:`Kalinin_2025`: do perturbations sharing a mechanism look
    alike, against those that do not?"""
    with pytest.raises(ValueError, match="annotation_key"):
        mt.tl.map(profiles, mode="consistency", null_size=100)

    profiles.obs["Metadata_MOA"] = np.where(
        profiles.obs["Metadata_Perturbation"].astype(str).isin(["pert00", "pert01"]), "A", "B"
    )
    with pytest.warns(UserWarning, match="one profile per perturbation"):
        mt.tl.map(profiles, mode="consistency", annotation_key="Metadata_MOA", null_size=200)
    assert set(profiles.uns["mantispy"]["map"]["Metadata_MOA"]) == {"A", "B"}


@requires_copairs
def test_consistency_does_not_count_a_perturbation_agreeing_with_itself(profiles):
    """Two wells of one treatment share its annotation trivially.

    Counting those pairs makes replicate retrieval look like mechanism coherence, more so
    the more wells a perturbation has. A consensus object has one row per perturbation, so
    excluding same-perturbation pairs must change nothing there.
    """
    profiles.obs["Metadata_MOA"] = np.where(
        profiles.obs["Metadata_Perturbation"].astype(str).isin(["pert00", "pert01"]), "A", "B"
    )
    treated = profiles[~profiles.obs["Metadata_Control"].to_numpy()].copy()

    wells = treated.copy()
    with pytest.warns(UserWarning, match="one profile per perturbation"):
        mt.tl.map(wells, mode="consistency", annotation_key="Metadata_MOA", null_size=200, seed=0)
    loose = wells.copy()
    mt.tl.map(
        loose,
        pos_sameby=["Metadata_MOA"],
        pos_diffby=[],
        neg_diffby=["Metadata_MOA"],
        null_size=200,
        seed=0,
    )
    strict = wells.uns["mantispy"]["map"]["mean_average_precision"].mean()
    assert strict < loose.uns["mantispy"]["map"]["mean_average_precision"].mean()

    # On one profile per perturbation the exclusion removes no pair, so the two agree.
    signatures = mt.tl.consensus(treated, method="median", min_replicates=1)
    signatures.obs["Metadata_MOA"] = np.where(
        signatures.obs["Metadata_Perturbation"].astype(str).isin(["pert00", "pert01"]), "A", "B"
    )
    collapsed = signatures.copy()
    mt.tl.map(collapsed, mode="consistency", annotation_key="Metadata_MOA", null_size=200, seed=0)
    reference = signatures.copy()
    mt.tl.map(
        reference,
        pos_sameby=["Metadata_MOA"],
        pos_diffby=[],
        neg_diffby=["Metadata_MOA"],
        null_size=200,
        seed=0,
    )
    np.testing.assert_allclose(
        collapsed.uns["mantispy"]["map"]["mean_average_precision"].to_numpy(),
        reference.uns["mantispy"]["map"]["mean_average_precision"].to_numpy(),
    )


@requires_copairs
def test_map_result_survives_a_round_trip(profiles, tmp_path):
    """copairs returns a ragged 'indices' column that h5ad cannot store."""
    mt.tl.map(profiles, mode="activity", null_size=200)
    assert "indices" not in profiles.uns["mantispy"]["map"].columns
    mt.io.write(profiles, tmp_path / "with_map.h5ad")
    assert "map" in mt.io.read(tmp_path / "with_map.h5ad").uns["mantispy"]


@requires_copairs
def test_map_is_reproducible_and_supports_a_representation(profiles):
    mt.tl.map(profiles, mode="activity", null_size=200, seed=1)
    first = profiles.uns["mantispy"]["map"]["corrected_p_value"].to_numpy()
    mt.tl.map(profiles, mode="activity", null_size=200, seed=1)
    np.testing.assert_allclose(first, profiles.uns["mantispy"]["map"]["corrected_p_value"].to_numpy())

    import scanpy as sc

    sc.pp.pca(profiles, n_comps=10)
    mt.tl.map(profiles, mode="replicability", use_rep="X_pca", null_size=200, key_added="map_pca")
    assert len(profiles.uns["mantispy"]["map_pca"]) > 0


@requires_copairs
def test_map_argument_errors(profiles):
    with pytest.raises(ValueError, match="not both"):
        mt.tl.map(profiles, mode="activity", pos_sameby=["Metadata_Perturbation"])
    with pytest.raises(ValueError, match="pos_sameby"):
        mt.tl.map(profiles)
    with pytest.raises(KeyError, match="Metadata_Nope"):
        mt.tl.map(profiles, pos_sameby=["Metadata_Nope"])


@requires_copairs
def test_map_refuses_missing_values(profiles):
    values = profiles.X.copy()
    values[0, 0] = np.nan
    profiles.X = values
    with pytest.raises(ValueError, match="missing values"):
        mt.tl.map(profiles, mode="activity", null_size=100)


# --- similarity and the older readouts -------------------------------------


@pytest.mark.parametrize("metric", ["cosine", "pearson"])
def test_similarity_is_symmetric_with_unit_diagonal(profiles, metric):
    mt.tl.similarity(profiles, metric=metric)
    matrix = np.asarray(profiles.obsp["similarity"])
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-5)
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-5)


def test_pearson_similarity_matches_numpy(profiles):
    mt.tl.similarity(profiles, metric="pearson")
    np.testing.assert_allclose(np.asarray(profiles.obsp["similarity"]), np.corrcoef(profiles.X), rtol=1e-4, atol=1e-5)


def test_percent_replicating_finds_strong_perturbations(profiles):
    mt.tl.percent_replicating(profiles, null_size=200)
    table = profiles.uns["mantispy"]["percent_replicating"].set_index("group")
    assert table.drop(index="DMSO")["is_replicating"].mean() > 0.5
    assert 0.0 <= profiles.uns["mantispy"]["percent_replicating_summary"]["fraction_replicating"] <= 1.0


def test_grit_is_higher_for_treated_than_controls(profiles):
    mt.tl.grit(profiles)
    table = profiles.uns["mantispy"]["grit"].set_index("group")
    assert table.drop(index="DMSO")["grit"].median() > table.loc["DMSO", "grit"]
    assert "grit" in profiles.obs


def test_grit_needs_controls(profiles):
    profiles.obs["Metadata_Control"] = False
    with pytest.raises(ValueError, match="no reference rows"):
        mt.tl.grit(profiles)


def test_cell_resolution_warns_but_perturbation_level_does_not(profiles):
    """Consensus profiles are the canonical input, so a warning on them would teach users
    to ignore the one at cell resolution."""
    cells = synthetic_plate(n_wells=24, n_cells=5, n_features=10, seed=0)
    with pytest.warns(UserWarning, match="resolution"):
        mt.tl.similarity(cells)

    consensus = mt.tl.aggregate(profiles, by=("Metadata_Perturbation",), min_cells=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mt.tl.similarity(consensus)
        mt.tl.grit(consensus)


def test_a_quadratic_similarity_is_refused_with_the_fix_named():
    """50 640 JUMP wells would need 30 GB, so the error names tl.consensus as the fix."""
    from mantispy.tl._similarity import SIMILARITY_BYTES, similarity_matrix

    too_many = int(np.sqrt(SIMILARITY_BYTES / 8)) + 1
    with pytest.raises(ValueError, match="mt.tl.consensus"):
        similarity_matrix(np.empty((too_many, 2), dtype=np.float32))
