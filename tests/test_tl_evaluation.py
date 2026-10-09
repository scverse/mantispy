"""mAP, similarity, percent replicating and grit."""

import warnings
from importlib.util import find_spec

import anndata as ad
import numpy as np
import pandas as pd
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


@requires_copairs
def test_activity_scores_treatments_against_the_controls(profiles):
    """Phenotypic activity as the field defines it: replicates retrieved against control profiles only, with the controls themselves dropped from the result."""
    mt.tl.map(profiles, mode="activity", null_size=200)
    table = profiles.uns["mantispy"]["map"]

    assert "DMSO" not in set(table["Metadata_Perturbation"])
    assert table["mean_average_precision"].median() > 0.9
    assert {"map", "map_qvalue"} <= set(profiles.obs.columns)


def _inert_p_value(pure_noise_screen, controls_elsewhere: int) -> float:
    """Score a compound drawn exactly as its plate's controls are, beside a second plate of nothing but controls.
    Each plate has its own seed, so the first is identical whatever the second holds."""
    here = pure_noise_screen(n_control=24, n_groups=1, per_group=6, n_features=12, seed=0)
    there = pure_noise_screen(n_control=controls_elsewhere, n_groups=0, n_features=12, seed=1)
    there.obs["Metadata_Plate"] = "P2"
    here.X[:, 0] += 10
    there.X[:, 1] += 10
    wells = ad.concat([here, there], keys=["P1", "P2"], index_unique=":")
    mt.io.stamp(wells, resolution="well", feature_kind="measurement")
    mt.tl.map(wells, mode="activity", null_size=2000)
    return float(wells.uns["mantispy"]["map"]["p_value"].item())


@requires_copairs
def test_activity_does_not_depend_on_another_plates_controls(pure_noise_screen):
    """Phenotypic activity asks whether a perturbation stands apart from the controls it was plated with.

    Pooling every plate's controls lets another plate in.
    Its controls sit far away and are beaten trivially, but they still count in the permutation null, which then expects the replicates to compete with all of them.
    So the more controls another plate has, the more active an inert compound on this plate looks.
    """
    few, many = _inert_p_value(pure_noise_screen, 24), _inert_p_value(pure_noise_screen, 96)

    assert few == many
    assert few > 0.05


@requires_copairs
def test_a_plate_without_controls_calls_nothing_active(pure_noise_screen):
    """A query with replicates and nothing to rank them against has a perfect rank list by construction.
    Scored, the inert compound on the plate without controls would come out active at the smallest p."""
    here = pure_noise_screen(n_control=24, n_groups=1, per_group=6, n_features=12, seed=0)
    there = pure_noise_screen(n_control=0, n_groups=1, per_group=6, n_features=12, seed=1)
    there.obs["Metadata_Plate"], there.obs["Metadata_Perturbation"] = "P2", "stranded"
    wells = ad.concat([here, there], keys=["P1", "P2"], index_unique=":")
    mt.io.stamp(wells, resolution="well", feature_kind="measurement")

    with pytest.warns(UserWarning, match="no negative pair"):
        mt.tl.map(wells, mode="activity", null_size=200)

    assert set(wells.uns["mantispy"]["map"]["Metadata_Perturbation"]) == {"p00"}


@requires_copairs
def test_a_null_too_small_for_the_correction_says_so(pure_noise_screen):
    """No p-value falls below 1 / (null_size + 1), so over enough groups the correction calls nothing however strong a lone effect is, and a screen scored that way would report no hits without saying why."""
    screen = pure_noise_screen()
    with pytest.warns(UserWarning, match="at least 3 reach that floor"):
        mt.tl.map(screen, mode="activity", null_size=100)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mt.tl.map(screen, mode="activity", null_size=1000)


@requires_copairs
def test_a_p_value_does_not_depend_on_earlier_calls(pure_noise_screen, tmp_path, monkeypatch):
    """copairs draws each null with a seed that depends on the other nulls in the same call.
    A cache shared across calls hands the second object a null drawn for the first, so its p-values depend on what ran before it."""
    alone = pure_noise_screen(n_control=0, n_groups=6, per_group=4, seed=0)
    other_plate = pure_noise_screen(n_control=0, n_groups=3, per_group=2, seed=1)
    other_plate.obs["Metadata_Plate"] = "P2"
    other_plate.obs["Metadata_Perturbation"] = "q" + other_plate.obs["Metadata_Perturbation"].astype(str)
    both = ad.concat([alone, other_plate], index_unique=":", keys=["P1", "P2"])
    mt.io.stamp(both, resolution="well", feature_kind="measurement")

    def p_values(*objects):
        for adata in objects:
            mt.tl.map(adata, mode="replicability", reference=None, null_size=200)
        return objects[-1].uns["mantispy"]["map"]["p_value"].to_numpy()

    monkeypatch.setenv("HOME", str(tmp_path / "fresh"))
    fresh = p_values(alone.copy())
    monkeypatch.setenv("HOME", str(tmp_path / "after"))
    after = p_values(both, alone.copy())

    np.testing.assert_array_equal(fresh, after)
    assert not (tmp_path / "after" / ".copairs").exists()


@requires_copairs
def test_activity_and_replicability_are_different_questions(profiles):
    mt.tl.map(profiles, mode="activity", null_size=200, key_added="activity")
    mt.tl.map(profiles, mode="replicability", null_size=200, key_added="replicability")

    activity = profiles.uns["mantispy"]["activity"].set_index("Metadata_Perturbation")
    replicability = profiles.uns["mantispy"]["replicability"].set_index("Metadata_Perturbation")

    # Both score the treatments; the controls are what activity retrieves against, and replicability leaves them out.
    assert "DMSO" not in activity.index
    assert set(activity.index) == set(replicability.index)


@requires_copairs
def test_replicability_is_the_map_nonrep_of_arevalo(profiles):
    """Regression test for #70: as in the paper's code, negatives come from the query's plate and the controls are left out."""
    mt.tl.map(profiles, mode="replicability", null_size=200)
    treated = profiles[~profiles.obs["Metadata_Control"].to_numpy()].copy()
    mt.tl.map(
        treated,
        pos_sameby=["Metadata_Perturbation"],
        neg_sameby=["Metadata_Plate"],
        neg_diffby=["Metadata_Perturbation"],
        null_size=200,
    )
    pd.testing.assert_frame_equal(
        profiles.uns["mantispy"]["map"], treated.uns["mantispy"]["map"], check_categorical=False
    )

    mt.tl.map(profiles, mode="replicability", reference=None, null_size=200, key_added="with_controls")
    assert "DMSO" in set(profiles.uns["mantispy"]["with_controls"]["Metadata_Perturbation"])


@requires_copairs
def test_activity_needs_controls_and_says_so(profiles):
    treated = profiles[~profiles.obs["Metadata_Control"].to_numpy()].copy()
    with pytest.raises(ValueError, match="mode='replicability'"):
        mt.tl.map(treated, mode="activity", null_size=100)


@requires_copairs
def test_consistency_groups_by_an_annotation(profiles):
    """Phenotypic consistency :cite:p:`Kalinin_2025`: do perturbations sharing a mechanism look alike, against those that do not?"""
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

    Counting those pairs makes replicate retrieval look like mechanism coherence, more so the more wells a perturbation has.
    A consensus object has one row per perturbation, so excluding same-perturbation pairs must change nothing there.
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


def _consensus_with_moa(profiles, assignment):
    """One row per perturbation, with ``Metadata_MOA`` set from a {perturbation: value} map."""
    treated = profiles[~profiles.obs["Metadata_Control"].to_numpy()].copy()
    signatures = mt.tl.consensus(treated, method="median", min_replicates=1)
    perturbations = signatures.obs["Metadata_Perturbation"].astype(str)
    signatures.obs["Metadata_MOA"] = pd.Series(
        [assignment[p] for p in perturbations], index=signatures.obs.index, dtype=object
    )
    return signatures


@requires_copairs
def test_multilabel_scores_a_perturbation_toward_each_of_its_labels(profiles):
    """A list of labels per row is multilabel: a pair is positive if the lists intersect, and a perturbation with
    two labels is scored toward both classes."""
    both = ["pert00", "pert01", "pert02", "pert03", "pert04"]
    assignment = {"pert00": ["A"], "pert01": ["A"], "pert02": ["A", "B"], "pert03": ["B"], "pert04": ["B"]}
    signatures = _consensus_with_moa(profiles, assignment)
    assert set(signatures.obs["Metadata_Perturbation"].astype(str)) == set(both)

    mt.tl.map(signatures, mode="consistency", annotation_key="Metadata_MOA", null_size=200, seed=0)
    table = signatures.uns["mantispy"]["map"]

    # Both classes are scored, each over its three members (pert02 is in both).
    assert set(table["Metadata_MOA"]) == {"A", "B"}
    # A list-valued annotation has no single per-row score, so obs is left untouched.
    assert "map" not in signatures.obs.columns


@requires_copairs
def test_single_element_lists_match_the_single_label_column(profiles):
    """Wrapping each scalar label in a one-element list must score exactly as the scalar column does."""
    scalar = {"pert00": "A", "pert01": "A", "pert02": "A", "pert03": "B", "pert04": "B"}
    as_list = {perturbation: [label] for perturbation, label in scalar.items()}

    single = _consensus_with_moa(profiles, scalar)
    mt.tl.map(single, mode="consistency", annotation_key="Metadata_MOA", null_size=200, seed=0)

    listed = _consensus_with_moa(profiles, as_list)
    mt.tl.map(listed, mode="consistency", annotation_key="Metadata_MOA", null_size=200, seed=0)

    single_scores = single.uns["mantispy"]["map"].set_index("Metadata_MOA")["mean_average_precision"].sort_index()
    listed_scores = listed.uns["mantispy"]["map"].set_index("Metadata_MOA")["mean_average_precision"].sort_index()
    np.testing.assert_allclose(single_scores.to_numpy(), listed_scores.to_numpy())


@requires_copairs
def test_a_label_column_mixing_lists_and_scalars_is_rejected(profiles):
    mixed = {"pert00": ["A"], "pert01": "A", "pert02": ["A", "B"], "pert03": "B", "pert04": "B"}
    signatures = _consensus_with_moa(profiles, mixed)
    with pytest.raises(ValueError, match="mixes list-valued and scalar"):
        mt.tl.map(signatures, mode="consistency", annotation_key="Metadata_MOA", null_size=200)


@requires_copairs
def test_label_sep_splits_a_delimited_string_into_the_same_scores(profiles):
    """A stored 'A|B' string split by label_sep must score exactly as the equivalent list column, since a list
    column cannot be written to h5ad and a delimited string can."""
    delimited = {"pert00": "A", "pert01": "A", "pert02": "A|B", "pert03": "B", "pert04": "B"}
    as_list = {"pert00": ["A"], "pert01": ["A"], "pert02": ["A", "B"], "pert03": ["B"], "pert04": ["B"]}

    strings = _consensus_with_moa(profiles, delimited)
    mt.tl.map(strings, mode="consistency", annotation_key="Metadata_MOA", label_sep="|", null_size=200, seed=0)

    listed = _consensus_with_moa(profiles, as_list)
    mt.tl.map(listed, mode="consistency", annotation_key="Metadata_MOA", null_size=200, seed=0)

    split_scores = strings.uns["mantispy"]["map"].set_index("Metadata_MOA")["mean_average_precision"].sort_index()
    list_scores = listed.uns["mantispy"]["map"].set_index("Metadata_MOA")["mean_average_precision"].sort_index()
    np.testing.assert_allclose(split_scores.to_numpy(), list_scores.to_numpy())


@requires_copairs
def test_a_missing_multilabel_entry_is_left_unscored(profiles):
    """A row with no annotation becomes an empty label set and joins no class, rather than erroring."""
    assignment = {"pert00": ["A"], "pert01": ["A"], "pert02": ["A", "B"], "pert03": ["B"], "pert04": None}
    signatures = _consensus_with_moa(profiles, assignment)
    mt.tl.map(signatures, mode="consistency", annotation_key="Metadata_MOA", null_size=200, seed=0)
    assert set(signatures.uns["mantispy"]["map"]["Metadata_MOA"]) == {"A", "B"}


@requires_copairs
def test_multilabel_clears_a_stale_single_label_obs_column(profiles):
    """The single-label path writes obs[key]; a later multilabel run must not leave those per-row columns behind to
    disagree with the per-class table."""
    single = {"pert00": "A", "pert01": "A", "pert02": "A", "pert03": "B", "pert04": "B"}
    signatures = _consensus_with_moa(profiles, single)
    mt.tl.map(signatures, mode="consistency", annotation_key="Metadata_MOA", null_size=200, seed=0)
    assert "map" in signatures.obs.columns

    perturbations = signatures.obs["Metadata_Perturbation"].astype(str)
    as_list = {"pert00": ["A"], "pert01": ["A"], "pert02": ["A", "B"], "pert03": ["B"], "pert04": ["B"]}
    signatures.obs["Metadata_MOA"] = pd.Series(
        [as_list[p] for p in perturbations], index=signatures.obs.index, dtype=object
    )
    mt.tl.map(signatures, mode="consistency", annotation_key="Metadata_MOA", null_size=200, seed=0)
    assert "map" not in signatures.obs.columns
    assert "map_qvalue" not in signatures.obs.columns


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


@pytest.mark.parametrize("infinity", [np.inf, -np.inf])
def test_an_infinite_value_is_compared_as_missing(infinity):
    """Regression test for #65: filled as the largest float, one infinity made its profile orthogonal to every other."""
    from mantispy.tl._similarity import similarity_matrix

    values = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.1], [1.0, infinity, 3.0]])
    matrix = similarity_matrix(values)
    np.testing.assert_array_equal(matrix, similarity_matrix(np.where(np.isinf(values), np.nan, values)))
    assert matrix[2, :2].min() > 0.8


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
    profiles.obs["Metadata_Control_Type"] = "treatment"
    with pytest.raises(ValueError, match="no reference rows"):
        mt.tl.grit(profiles)


def test_cell_resolution_warns_but_perturbation_level_does_not(profiles):
    """Consensus profiles are the canonical input, so a warning on them would teach users to ignore the one at cell resolution."""
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
