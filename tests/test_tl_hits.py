"""Hit calling and energy distance."""

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

import mantispy as mt
from mantispy._core.schema import stamp
from mantispy.tl._hits import ks_statistic


@pytest.fixture
def scored():
    cells = mt.ds.synthetic_plate(n_wells=48, n_cells=40, n_features=15, n_perturbations=3, effect_size=3.0, seed=0)
    mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon")
    return cells


@pytest.mark.parametrize("method", ["mahalanobis", "ks"])
def test_treated_groups_are_hits_and_sit_further_out_than_the_controls(scored, method):
    """Whether the reference row is called is not asserted here.

    It is an honest draw from the null, so it is called at about the nominal rate by
    construction, and a single seed that happens not to call it would pin the bias rather
    than the behaviour. Its distribution is pinned across seeds by
    test_the_reference_row_is_not_scored_on_the_rows_that_fitted_the_covariance.
    """
    mt.tl.hit_calling(scored, method=method, n_permutations=200)
    table = scored.uns["mantispy"]["hits"].set_index("group")
    assert table.drop(index="DMSO")["is_hit"].all()
    assert table.drop(index="DMSO")["distance"].min() > table.loc["DMSO", "distance"]


def test_hit_calling_corrects_and_joins_back(scored):
    mt.tl.hit_calling(scored, n_permutations=200)
    table = scored.uns["mantispy"]["hits"]
    assert (table["qvalue"] >= table["pvalue"] - 1e-12).all()
    assert (table["pvalue"] > 0).all()  # a permutation p-value is never zero
    assert {"hits_distance", "hits_qvalue"} <= set(scored.obs.columns)
    assert np.isfinite(scored.obs["hits_distance"]).all()


def test_the_controls_that_fitted_the_transform_are_marked(scored):
    """Regression for #83: anything reading a scale off the controls needs to know which half is honest."""
    mt.tl.hit_calling(scored, n_permutations=200, seed=0)
    held_out = scored.obs["hits_reference_held_out"].to_numpy(dtype=bool)
    is_control = scored.obs["Metadata_Control"].to_numpy(dtype=bool)

    assert not held_out[~is_control].any(), "a treated row was never a candidate for the reference"
    # The split halves the controls, and the other half is the one that placed the centroid.
    assert 0 < held_out.sum() < is_control.sum()


def test_hit_calling_is_reproducible(scored):
    mt.tl.hit_calling(scored, n_permutations=200, seed=3)
    first = scored.uns["mantispy"]["hits"]["pvalue"].to_numpy()
    mt.tl.hit_calling(scored, n_permutations=200, seed=3)
    np.testing.assert_allclose(first, scored.uns["mantispy"]["hits"]["pvalue"].to_numpy())


def test_more_features_than_controls_warns_and_names_the_way_out(scored):
    """rohban reached Mahalanobis distances of 3e17 this way."""
    few = scored[scored.obs["Metadata_Perturbation"] != "DMSO"].copy()
    few.obs["Metadata_Control"] = np.arange(few.n_obs) < 5
    with pytest.warns(UserWarning, match="use_rep"):
        mt.tl.hit_calling(few, n_permutations=50)


def test_the_vectorised_ks_matches_scipy():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=300)
    samples = rng.normal(0.4, 1.5, size=(20, 40))
    expected = [ks_2samp(row, reference).statistic for row in samples]
    np.testing.assert_allclose(ks_statistic(samples, reference), expected, rtol=1e-12)


def test_edistance_against_the_controls_ranks_the_perturbations_first(scored):
    mt.tl.edistance(scored, n_permutations=200)
    table = scored.uns["mantispy"]["edistance"].set_index("group")
    assert table.drop(index="DMSO")["distance"].min() > table.loc["DMSO", "distance"]
    assert (table["qvalue"] >= table["pvalue"] - 1e-12).all()
    assert not bool(table.loc["DMSO", "is_hit"])


def test_pairwise_edistance_is_symmetric_with_a_zero_diagonal(scored):
    mt.tl.edistance(scored, reference=None)
    matrix = scored.uns["mantispy"]["edistance_pairwise"]
    assert list(matrix.index) == sorted(scored.obs["Metadata_Perturbation"].astype(str).unique())
    values = matrix.to_numpy()
    np.testing.assert_allclose(values, values.T, atol=1e-6)
    np.testing.assert_allclose(np.diag(values), 0.0, atol=1e-6)


def test_results_round_trip(scored, tmp_path):
    mt.tl.hit_calling(scored, n_permutations=100)
    mt.tl.edistance(scored, n_permutations=100)
    mt.io.write(scored, tmp_path / "hits.h5ad")
    loaded = mt.io.read(tmp_path / "hits.h5ad")
    assert len(loaded.uns["mantispy"]["hits"]) == 4
    assert len(loaded.uns["mantispy"]["edistance"]) == 4


def test_hit_calling_does_not_call_a_screen_of_pure_noise(pure_noise_screen):
    """The null is drawn from control rows held out of the centroid and covariance fit, so
    it is out-of-sample like every group. An in-sample null calls 12 of 12 groups here.

    The screen also warns, because splitting halves the rows that fit the covariance and
    leaves 60 of them against 80 features.
    """
    adata = pure_noise_screen()
    with pytest.warns(UserWarning, match="reference rows"):
        mt.tl.hit_calling(adata, n_permutations=1000, seed=0)
    table = adata.uns["mantispy"]["hits"]
    treated = table[table["group"] != "DMSO"]
    assert int(treated["is_hit"].sum()) <= 1, treated[["group", "pvalue", "qvalue"]]


@pytest.mark.filterwarnings("ignore:the covariance is estimated")
def test_hit_calling_still_finds_a_real_effect(pure_noise_screen):
    """A guard against fixing the false positives by refusing to call anything."""
    adata = pure_noise_screen()
    moved = adata.obs["Metadata_Perturbation"].to_numpy() == "p00"
    values = adata.X.copy()
    values[moved] += 6.0
    adata.X = values
    mt.tl.hit_calling(adata, n_permutations=1000, seed=0)
    table = adata.uns["mantispy"]["hits"].set_index("group")
    assert bool(table.loc["p00", "is_hit"])


@pytest.mark.filterwarnings("ignore:the covariance is estimated")
def test_a_missing_feature_does_not_make_a_group_maximally_control_like(pure_noise_screen):
    """nan_to_num after the matmul would turn one NaN feature into a whitened row of NaN and
    then zeros, making the strongest hit the most control-like group."""
    adata = pure_noise_screen()
    moved = adata.obs["Metadata_Perturbation"].to_numpy() == "p00"
    values = adata.X.copy()
    values[moved] += 6.0
    adata.X = values
    mt.tl.hit_calling(adata, n_permutations=1000, seed=0)
    intact = adata.uns["mantispy"]["hits"].set_index("group").loc["p00", "distance"]

    values = adata.X.copy()
    values[moved, 7] = np.nan
    adata.X = values
    mt.tl.hit_calling(adata, n_permutations=1000, seed=0)
    with_gap = adata.uns["mantispy"]["hits"].set_index("group").loc["p00", "distance"]

    assert with_gap > 0.5 * intact, f"one NaN feature took the distance from {intact} to {with_gap}"


def test_the_reference_row_is_not_scored_on_the_rows_that_fitted_the_covariance(pure_noise_screen):
    """The controls carry a perturbation label of their own, so one row of the table is the reference against itself.

    Scoring that row on all of the control rows puts the covariance-fitting half, which sits
    closer to the centroid than any other group's rows can, on the tested side. Its median
    then comes out below the null it is compared with, and the row cannot reach significance
    however the controls fall: its p-value is pinned near one instead of being a draw from
    the null like every other group's.

    The mean is bounded on both sides, so a row that is systematically small fails here too.
    The bound is wide enough that a calibrated row passes it however the seeds fall; the
    rate in the tail is what the row is for, and separating a 5% tail from a 1% one needs
    more draws than a unit test can afford.
    """
    pvalues = []
    for seed in range(24):
        adata = pure_noise_screen(n_control=192, n_groups=4, per_group=48, n_features=10, seed=seed)
        mt.tl.hit_calling(adata, n_permutations=200, seed=seed)
        table = adata.uns["mantispy"]["hits"].set_index("group")
        assert table.loc["DMSO", "n_obs"] == 48, "a quarter of the controls: half held out, and half of those tested"
        pvalues.append(float(table.loc["DMSO", "pvalue"]))
    assert 0.25 < float(np.mean(pvalues)) < 0.75, f"a draw from the null is uniform, got {pvalues}"


def test_the_reference_row_is_not_halved_along_the_plate_order(well_profiles):
    """The reference group's held-out rows are halved at random, not by row order.

    Rows arrive ordered by plate and well, so the first half in row order is one set of plates
    and the second is another. With the controls carrying a between-plate offset, that makes
    the reference row a comparison between plates rather than a draw from the null, and it is
    called far above the nominal rate. Both methods draw that half from the same generator;
    ``ks`` is used here because it reads the two halves against each other directly, which
    separates a leaking split from a calibrated one in the fewest seeds.
    """
    called = 0
    for seed in range(30):
        adata = well_profiles(n_plates=8, per_plate=128, n_features=10, plate_sd=3.0, seed=seed)
        mt.tl.hit_calling(adata, method="ks", seed=seed)
        table = adata.uns["mantispy"]["hits"].set_index("group")
        called += int(float(table.loc["DMSO", "pvalue"]) < 0.05)
    assert called <= 4, f"{called}/30 seeds called the reference row at p < 0.05, where 0.05 is calibrated"


def test_edistance_does_not_call_a_screen_of_pure_noise(pure_noise_screen):
    """Pure noise calls at most one of twelve groups.

    The null is too narrow, and calls most groups on this fixture, if its between-group
    term includes each row's zero self-distance or if a group as large as the control set
    draws every control on each permutation.
    """
    adata = pure_noise_screen()
    mt.tl.edistance(adata, n_permutations=500, seed=0)
    table = adata.uns["mantispy"]["edistance"]
    treated = table[table["group"] != "DMSO"]
    assert int(treated["is_hit"].sum()) <= 1, treated[["group", "pvalue", "qvalue"]]


def test_edistance_scores_a_balanced_screen(pure_noise_screen):
    """Every group gets a p-value when groups are as large as the control set.

    That is the normal shape of a screen with equal wells per treatment. A null drawn from
    a subset of the controls cannot be formed there, while permuting labels over the pooled
    rows can.
    """
    adata = pure_noise_screen(n_control=60, n_groups=3, per_group=60, n_features=20)
    mt.tl.edistance(adata, n_permutations=500, seed=0)
    table = adata.uns["mantispy"]["edistance"]
    assert table["pvalue"].notna().all(), table
    # A zero-variance null would put every group at 1/(n_permutations + 1), which passes
    # the check above but calls every group.
    treated = table[table["group"] != "DMSO"]
    assert int(treated["is_hit"].sum()) == 0, treated[["group", "pvalue", "qvalue"]]


def test_edistance_still_finds_a_real_effect(pure_noise_screen):
    adata = pure_noise_screen()
    moved = adata.obs["Metadata_Perturbation"].to_numpy() == "p00"
    values = adata.X.copy()
    values[moved] += 2.0
    adata.X = values
    mt.tl.edistance(adata, n_permutations=500, seed=0)
    assert bool(adata.uns["mantispy"]["edistance"].set_index("group").loc["p00", "is_hit"])


def test_the_controls_are_not_a_hit_against_themselves(pure_noise_screen):
    """The reference group has no other reference to be compared with, so it is scored
    against half of its own rows and must come out as an ordinary non-hit."""
    adata = pure_noise_screen()
    mt.tl.edistance(adata, n_permutations=500, seed=0)
    row = adata.uns["mantispy"]["edistance"].set_index("group").loc["DMSO"]
    assert not bool(row["is_hit"])
    assert float(row["pvalue"]) > 0.05


def test_a_reference_too_small_to_halve_is_unscored_rather_than_a_crash(pure_noise_screen):
    """`split_reference` raises below four rows. With two or three control rows every
    treated group can still be scored against the reference; only the reference against
    itself cannot, so that row is left unscored.
    """
    adata = pure_noise_screen(n_control=3, n_groups=1, per_group=20)
    with pytest.warns(UserWarning, match="fewer than four rows"):
        mt.tl.edistance(adata, n_permutations=200, seed=0)
    table = adata.uns["mantispy"]["edistance"].set_index("group")
    assert np.isfinite(table.loc["p00", "pvalue"])
    assert np.isnan(table.loc["DMSO", "pvalue"])
    assert not bool(table.loc["DMSO", "is_hit"])


def test_single_row_groups_are_named_in_a_warning_not_silently_nan(pure_noise_screen):
    """A group of one well has no within-group term for energy distance, so its p-value is
    NaN. Those groups are named in a warning, since this is the shape of a single-well
    primary screen or a tl.consensus output.
    """
    adata = pure_noise_screen(n_control=40, n_groups=5, per_group=1, n_features=10)
    moved = adata.obs["Metadata_Perturbation"].to_numpy() == "p00"
    values = adata.X.copy()
    values[moved] += 8.0
    adata.X = values

    with pytest.warns(UserWarning, match="fewer than two rows"):
        mt.tl.edistance(adata, n_permutations=200, seed=0)
    table = adata.uns["mantispy"]["edistance"].set_index("group")
    for group in [f"p{i:02d}" for i in range(5)]:
        assert np.isnan(table.loc[group, "pvalue"]), group
        assert not bool(table.loc[group, "is_hit"]), group
    assert np.isfinite(table.loc["DMSO", "pvalue"])


def test_the_ks_method_is_calibrated_on_pure_noise():
    """Pseudo-groups for the null must not come from the rows that form their reference.

    If they do, every draw is a subset of its own reference, the null statistic is too
    small, and nearly half of pure-noise groups are called.
    """
    called = total = 0
    for seed in range(8):
        cells = mt.ds.synthetic_plate(
            n_plates=2, n_wells=96, n_cells=12, n_features=10, n_perturbations=3, effect_size=0.0, seed=seed
        )
        mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon")
        wells = mt.tl.aggregate(cells, min_cells=0)
        mt.tl.hit_calling(wells, method="ks", n_permutations=200, key_added="k")
        table = wells.uns["mantispy"]["k"]
        treated = table[table["group"] != "DMSO"]
        called += int((treated["qvalue"] < 0.05).sum())
        total += len(treated)
    assert total >= 20
    assert called / total < 0.15, f"{called}/{total} pure-noise groups called at q < 0.05"


def test_the_ks_method_still_sees_a_partial_responder():
    """The calibrated KS test still calls a group in which a fifth of the cells respond."""
    cells = mt.ds.synthetic_plate(n_wells=96, n_cells=40, n_features=25, n_perturbations=2, effect_size=0.0, seed=0)
    mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon")
    rng = np.random.default_rng(0)
    target = np.flatnonzero((cells.obs["Metadata_Perturbation"] == "pert00").to_numpy())
    responders = rng.choice(target, size=int(0.2 * target.size), replace=False)
    values = cells.X.copy()
    values[responders] += 8.0
    cells.X = values

    mt.tl.hit_calling(cells, method="ks", n_permutations=200, key_added="k")
    table = cells.uns["mantispy"]["k"].set_index("group")
    assert bool(table.loc["pert00", "is_hit"]), "a fifth of the cells moved a long way"
    assert not bool(table.loc["pert01", "is_hit"]), "and nothing happened to this one"


def test_edistance_n_obs_reports_the_rows_the_statistic_used(scored):
    """n_obs counts the rows the statistic used.

    The reference group has no other reference, so it is scored against half of itself. A
    60-vs-60 comparison reported as n_obs=120 would overstate its power twofold.
    """
    wells = mt.tl.aggregate(scored, min_cells=0)
    mt.tl.edistance(wells, n_permutations=50, key_added="d")
    table = wells.uns["mantispy"]["d"].set_index("group")
    counts = wells.obs["Metadata_Perturbation"].astype(str).value_counts()

    assert table.loc["DMSO", "n_obs"] == counts["DMSO"] // 2, "the reference is split against itself"
    for group in set(counts.index) - {"DMSO"}:
        assert table.loc[group, "n_obs"] == counts[group], "an ordinary group uses all of its rows"


def test_edistance_caps_both_sides_and_says_so(scored, caplog):
    """max_reference caps both the group and the reference, and sampling down is logged."""
    import logging

    wells = mt.tl.aggregate(scored, min_cells=0)
    with caplog.at_level(logging.INFO, logger="mantispy"):
        mt.tl.edistance(wells, n_permutations=20, max_reference=6, key_added="capped")

    table = wells.uns["mantispy"]["capped"]
    assert (table["n_obs"] <= 6).all(), "no group may enter the statistic above the cap"
    messages = " ".join(record.message for record in caplog.records)
    assert "sampled" in messages, "sampling down changes the answer and must be logged"


def test_the_pairwise_edistance_caps_its_groups_too(scored):
    """max_reference and seed were accepted and ignored on the reference=None path, where memory is quadratic in the two largest groups."""

    def pairwise(**kwargs):
        out = mt.tl.edistance(scored, reference=None, copy=True, **kwargs)
        return out.uns["mantispy"]["edistance_pairwise"].to_numpy(dtype=float)

    capped = pairwise(max_reference=20)
    assert not np.allclose(capped, pairwise()), "20 of ~480 rows per group must change the distances"
    np.testing.assert_array_equal(capped, pairwise(max_reference=20, seed=0), "and the same seed must repeat them")
    assert not np.allclose(capped, pairwise(max_reference=20, seed=1)), "while another seed samples other rows"


def test_the_null_is_calibrated_on_pure_noise():
    """On data with no effect, p < 0.05 should happen about 5% of the time.

    The null is the other ways to draw a group of this size from the group and the held-out controls
    pooled. Bootstrapping the controls alone centres the null on that sample's own median and leaves
    its error out of the spread, which called ~9% of nothing at n=24 against 48 held-out controls.
    """
    rng = np.random.default_rng(0)
    pvalues = []
    for seed in range(25):
        n_controls, n_groups, group_size = 96, 6, 24
        n = n_controls + n_groups * group_size
        adata = ad.AnnData(X=rng.normal(size=(n, 12)).astype(np.float32))
        adata.obs["Metadata_Plate"] = "P1"
        adata.obs["Metadata_Well"] = [f"{chr(65 + i // 24)}{i % 24 + 1:02d}" for i in range(n)]
        adata.obs["Metadata_Perturbation"] = ["DMSO"] * n_controls + [
            f"p{g:02d}" for g in range(n_groups) for _ in range(group_size)
        ]
        adata.obs["Metadata_Control"] = adata.obs["Metadata_Perturbation"] == "DMSO"
        mt.tl.hit_calling(adata, groupby="Metadata_Perturbation", n_permutations=500, seed=seed)
        table = adata.uns["mantispy"]["hits"]
        pvalues += list(table.loc[table["group"] != "DMSO", "pvalue"])

    rate = float(np.mean(np.asarray(pvalues) < 0.05))
    assert rate < 0.08, f"called {rate:.1%} of pure noise at a nominal 5%"


def _plate(group_size: int, n_groups: int = 20, n_controls: int = 24, seed: int = 0):
    """A plate of pure noise where every treated group holds `group_size` wells."""
    rng = np.random.default_rng(seed)
    n = n_controls + n_groups * group_size
    adata = ad.AnnData(X=rng.normal(size=(n, 8)).astype(np.float32))
    adata.obs["Metadata_Plate"] = "P1"
    adata.obs["Metadata_Well"] = [f"{chr(65 + i // 24)}{i % 24 + 1:02d}" for i in range(n)]
    adata.obs["Metadata_Perturbation"] = ["DMSO"] * n_controls + [
        f"p{g:02d}" for g in range(n_groups) for _ in range(group_size)
    ]
    adata.obs["Metadata_Control"] = np.arange(n) < n_controls
    return adata


def test_a_screen_without_replication_says_so():
    """A group of one or two wells has a median no permutation null can rescue, so say so rather than call it."""
    with pytest.warns(UserWarning, match="fewer than three rows"):
        mt.tl.hit_calling(_plate(group_size=2), n_permutations=200)


def test_a_replicated_screen_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mt.tl.hit_calling(_plate(group_size=6), n_permutations=200)


def test_a_robust_covariance_still_calls_a_hit_that_stray_controls_would_hide():
    """A few wild control wells widen the empirical covariance along their own direction.

    A real effect in that direction is then measured in units the outliers set and reads as
    ordinary. The minimum covariance determinant subset ignores them, so the hit survives.
    """
    rng = np.random.default_rng(0)
    n_features = 6
    direction = np.zeros(n_features)
    direction[0] = 1.0

    controls = rng.standard_normal((60, n_features))
    # Six control wells blown out along one feature, as a pipetting or focus artifact does.
    controls[:6] += 12.0 * direction
    treated = rng.standard_normal((12, n_features)) + 4.0 * direction

    values = np.vstack([controls, treated]).astype(np.float32)
    obs = pd.DataFrame(
        {
            "Metadata_Plate": "P1",
            "Metadata_Well": [f"W{index:03d}" for index in range(len(values))],
            "Metadata_Perturbation": ["DMSO"] * len(controls) + ["pert"] * len(treated),
            "Metadata_Control": [True] * len(controls) + [False] * len(treated),
        },
        index=[str(index) for index in range(len(values))],
    )
    adata = ad.AnnData(X=values, obs=obs, var=pd.DataFrame(index=[f"Cells_AreaShape_f{i}" for i in range(n_features)]))
    stamp(adata, resolution="well")

    empirical = mt.tl.hit_calling(adata, covariance="empirical", n_permutations=500, copy=True)
    robust = mt.tl.hit_calling(adata, covariance="robust", n_permutations=500, copy=True)

    def distance_of(scored, group):
        table = scored.uns["mantispy"]["hits"]
        return float(table.loc[table["group"] == group, "distance"].iloc[0])

    assert distance_of(robust, "pert") > distance_of(empirical, "pert"), (
        "the outlying controls should stop inflating the scale the treatment is measured on"
    )
    # The control row is the reference against itself, so it is the scale every other group is read on.
    assert distance_of(empirical, "pert") < distance_of(empirical, "DMSO"), (
        "with the outliers in the covariance the treatment reads as no further out than the controls"
    )
    assert distance_of(robust, "pert") > distance_of(robust, "DMSO")

    def pvalue_of(scored):
        table = scored.uns["mantispy"]["hits"]
        return float(table.loc[table["group"] == "pert", "pvalue"].iloc[0])

    assert pvalue_of(robust) < 0.05 < pvalue_of(empirical)


def test_a_robust_covariance_needs_more_control_rows_than_features():
    """Well-level profiles have far fewer rows than features, which is what use_rep is for."""
    cells = mt.ds.synthetic_plate(n_wells=48, n_cells=40, n_features=60, n_perturbations=3, effect_size=3.0, seed=0)
    wells = mt.tl.aggregate(cells)
    with pytest.raises(ValueError, match="more complete reference rows than features"):
        mt.tl.hit_calling(wells, covariance="robust", n_permutations=50)


def _well_effect_cells(n_control_wells=32, n_groups=10, wells_per_group=2, cells_per_well=200, n_features=12, seed=0):
    """A cell-level screen with a real well random effect and no treatment effect at all.

    Every well carries a shared offset (sd 1) applied to all of its cells, plus per-cell noise
    (sd 1), so the design's unit is the well and the cells within it are pseudoreplicates of it.
    Whole wells are relabelled as pseudo-treatments against a DMSO reference; no group differs
    from any other, so every hit called on this object is a false positive. Thirty-two control
    wells give the well-block null enough exchangeable units on each half of the split to
    calibrate (issue #68 was measured to need this many; twenty leave it anti-conservative).
    """
    rng = np.random.default_rng(seed)
    n_wells = n_control_wells + n_groups * wells_per_group
    offsets = rng.normal(0.0, 1.0, size=(n_wells, n_features))

    values, plate, well, perturbation, control = [], [], [], [], []
    for w in range(n_wells):
        values.append(offsets[w] + rng.normal(0.0, 1.0, size=(cells_per_well, n_features)))
        well += [f"{chr(65 + w // 24)}{w % 24 + 1:02d}"] * cells_per_well
        plate += ["P1"] * cells_per_well
        treated = w >= n_control_wells
        perturbation += [f"p{(w - n_control_wells) // wells_per_group:02d}" if treated else "DMSO"] * cells_per_well
        control += [not treated] * cells_per_well

    obs = pd.DataFrame(
        {
            "Metadata_Plate": plate,
            "Metadata_Well": well,
            "Metadata_Perturbation": perturbation,
            "Metadata_Control": control,
        },
        index=[str(i) for i in range(len(well))],
    )
    adata = ad.AnnData(
        X=np.vstack(values).astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"Cells_AreaShape_f{i}" for i in range(n_features)]),
    )
    stamp(adata, resolution="cell")
    return adata


def test_hit_calling_block_null_fixes_cell_resolution_pseudoreplication():
    """Regression for #68: cells within a well are pseudoreplicates, not independent draws.

    A null that shuffles cells gives the group median far too little spread, since the shrink
    goes as the cell count rather than the well count, so pure-null pseudo-treatments are
    called well above nominal. Resampling whole wells (block="Metadata_Well") matches the
    null's exchangeable unit to the design and brings the rate back to nominal.
    """
    block_hits = block_total = cell_hits = cell_total = 0
    for seed in range(12):
        adata = _well_effect_cells(seed=seed)
        mt.tl.hit_calling(adata, block="Metadata_Well", n_permutations=400, seed=seed, key_added="block")

        # Drop the well column so the cell-shuffle null (the bug) runs on the same feature matrix.
        buggy = adata.copy()
        del buggy.obs["Metadata_Well"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mt.tl.hit_calling(buggy, n_permutations=400, seed=seed, key_added="cell")

        block = adata.uns["mantispy"]["block"]
        cell = buggy.uns["mantispy"]["cell"]
        block_hits += int((block[block["group"] != "DMSO"]["qvalue"] < 0.05).sum())
        block_total += int((block["group"] != "DMSO").sum())
        cell_hits += int((cell[cell["group"] != "DMSO"]["qvalue"] < 0.05).sum())
        cell_total += int((cell["group"] != "DMSO").sum())

    block_rate = block_hits / block_total
    cell_rate = cell_hits / cell_total
    assert block_rate <= 0.08, f"the well-block null called {block_rate:.1%} of pure-null pseudo-treatments"
    assert cell_rate >= 0.2, f"the cell-shuffle null called {cell_rate:.1%}, so the bug is not reproduced"


def test_hit_calling_defaults_to_the_well_block_at_cell_resolution():
    """At cell resolution with a well column present, block=None resolves to Metadata_Well."""
    adata = _well_effect_cells(seed=0)
    default = mt.tl.hit_calling(adata, n_permutations=200, seed=0, copy=True)
    explicit = mt.tl.hit_calling(adata, block="Metadata_Well", n_permutations=200, seed=0, copy=True)
    pd.testing.assert_frame_equal(default.uns["mantispy"]["hits"], explicit.uns["mantispy"]["hits"])


def test_hit_calling_warns_without_a_well_column_at_cell_resolution():
    """No well column and no explicit block leaves the anti-conservative cell-shuffle null, so warn."""
    adata = _well_effect_cells(seed=0)
    del adata.obs["Metadata_Well"]
    with pytest.warns(UserWarning, match="not independent replicates"):
        mt.tl.hit_calling(adata, n_permutations=100, seed=0)


def test_the_block_null_reference_group_is_not_systematically_called():
    """Regression for #68: the reference group's well-block null must not degenerate to a constant.

    hit_calling scores the controls against themselves too. When the fit and null halves are split by
    cell, the reference group's tested cells span essentially every control well, so the null draws the
    whole pool on every permutation and collapses to the two values {1/(n+1), 1.0}: about half the seeds
    then call the control a hit. Splitting whole wells into the two halves makes the reference draw a
    strict subset of the wells like any other group, so its p-value is a genuine draw from the null.
    """
    called = 0
    pvalues = []
    for seed in range(12):
        adata = _well_effect_cells(seed=seed)
        mt.tl.hit_calling(adata, n_permutations=400, seed=seed)
        row = adata.uns["mantispy"]["hits"].set_index("group").loc["DMSO"]
        called += int(bool(row["is_hit"]))
        pvalues.append(float(row["pvalue"]))
    assert called <= 2, f"the reference group was called on {called}/12 pure-null seeds, p={pvalues}"
    # A constant null pins every p-value at 1/(n+1) or 1.0; a genuine draw sits away from both extremes.
    assert 0.2 < float(np.mean(pvalues)) < 0.8, f"the reference p-values are not a draw from the null: {pvalues}"


def _two_plate_cells(cells_per_well=150, n_features=12, seed=0):
    """A cell-level screen on two plates that reuse the same well names, with a per-well random effect.

    Sixteen control wells and eight treated wells per plate, the well names identical across the two
    plates, so a block keyed on the bare well name would fold each pair of physically distinct wells
    into one and halve the block count.
    """
    rng = np.random.default_rng(seed)
    names = [f"{chr(65 + w // 24)}{w % 24 + 1:02d}" for w in range(24)]
    offsets = rng.normal(0.0, 1.0, size=(2, 24, n_features))

    values, plate, well, perturbation, control = [], [], [], [], []
    for p in range(2):
        for w in range(24):
            values.append(offsets[p, w] + rng.normal(0.0, 1.0, size=(cells_per_well, n_features)))
            well += [names[w]] * cells_per_well
            plate += [f"P{p + 1}"] * cells_per_well
            treated = w >= 16
            perturbation += [f"g{(w - 16) // 2:02d}" if treated else "DMSO"] * cells_per_well
            control += [not treated] * cells_per_well

    obs = pd.DataFrame(
        {
            "Metadata_Plate": plate,
            "Metadata_Well": well,
            "Metadata_Perturbation": perturbation,
            "Metadata_Control": control,
        },
        index=[str(i) for i in range(len(well))],
    )
    adata = ad.AnnData(
        X=np.vstack(values).astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"Cells_AreaShape_f{i}" for i in range(n_features)]),
    )
    stamp(adata, resolution="cell")
    return adata


def test_hit_calling_blocks_on_the_physical_well_across_plates():
    """The default block is the (plate, well) pair, so the same well name on two plates is two blocks."""
    from mantispy._core._reduce import group_codes

    adata = _two_plate_cells(seed=0)
    by_name = int(group_codes(adata, "Metadata_Well")[0].max()) + 1
    physical = int(group_codes(adata, ["Metadata_Plate", "Metadata_Well"])[0].max()) + 1
    assert (by_name, physical) == (24, 48), "the fixture must reuse well names across the two plates"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        default = mt.tl.hit_calling(adata, n_permutations=200, seed=0, copy=True)
        physical_block = mt.tl.hit_calling(
            adata, block=["Metadata_Plate", "Metadata_Well"], n_permutations=200, seed=0, copy=True
        )
        by_name_block = mt.tl.hit_calling(adata, block="Metadata_Well", n_permutations=200, seed=0, copy=True)

    pd.testing.assert_frame_equal(default.uns["mantispy"]["hits"], physical_block.uns["mantispy"]["hits"])
    assert not default.uns["mantispy"]["hits"].equals(by_name_block.uns["mantispy"]["hits"]), (
        "folding two plates' wells onto one name must change the null"
    )


def test_hit_calling_does_not_crash_on_missing_wells_at_cell_resolution():
    """A gap in Metadata_Well must warn and run the cell path, not crash a default hit_calling(adata).

    group_codes raises on a missing value, so auto-detecting the block off the well column would turn a
    call that worked before #68 into a crash. The block is left off and the cell-shuffle null runs instead.
    """
    adata = _well_effect_cells(seed=0)
    wells = adata.obs["Metadata_Well"].astype(object).to_numpy().copy()
    wells[: 5 * 200] = np.nan  # the first few wells go missing, as an incomplete platemap leaves them
    adata.obs["Metadata_Well"] = wells
    with pytest.warns(UserWarning, match="not independent replicates"):
        mt.tl.hit_calling(adata, n_permutations=100, seed=0)
    table = adata.uns["mantispy"]["hits"].set_index("group")
    assert np.isfinite(table.loc["DMSO", "pvalue"]), "every group is still scored on the cell path"
