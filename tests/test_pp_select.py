"""Behaviour of feature_select beyond equivalence, which test_equivalence_* covers."""

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantispy as mt


def test_writes_a_bool_column_and_a_per_operation_count(wells):
    mt.pp.feature_select(wells)
    assert wells.var["selected"].dtype == bool
    counts = wells.uns["mantispy"]["feature_select"]
    assert set(counts) == set(mt.pp._select.DEFAULT_OPERATIONS)
    assert all(isinstance(value, int) for value in counts.values())


def test_counts_are_what_each_operation_removes_on_its_own(wells):
    """A feature two operations both remove is counted by both, in whichever order they run."""
    values = wells.X.copy()
    values[:, 0] = 1.0  # Constant, so variance_threshold removes it.
    values[: int(0.6 * wells.n_obs), 0] = np.nan  # And mostly missing, so drop_na_columns removes it too.
    wells.X = values

    alone = {}
    for operation in ("drop_na_columns", "variance_threshold"):
        trial = wells.copy()
        mt.pp.feature_select(trial, operations=(operation,))
        alone[operation] = int((~trial.var["selected"].to_numpy()).sum())
    assert min(alone.values()) > 0, "neither operation removes anything; the test proves nothing"

    for operations in (("drop_na_columns", "variance_threshold"), ("variance_threshold", "drop_na_columns")):
        run = wells.copy()
        mt.pp.feature_select(run, operations=operations)
        assert dict(run.uns["mantispy"]["feature_select"]) == alone, f"counts changed with order {operations}"


def _one_feature_per_operation():
    """Four replicate groups in which drop_na_columns, drop_outliers and noise_removal each have exactly one feature to remove."""
    rng = np.random.default_rng(0)
    n_obs, n_groups = 20, 4
    per_group = n_obs // n_groups

    plain = rng.normal(scale=0.2, size=n_obs)  # no NaN, small, quiet within a group: every operation keeps it
    jitter = rng.normal(scale=5.0, size=n_obs)  # within-group standard deviation above the 0.8 cutoff
    ratio = np.repeat(np.arange(1, n_groups + 1) * 1000.0, per_group)  # above the 500 cutoff, flat within a group
    sparse = np.full((n_groups, per_group), np.nan)  # missing in 60% of rows, two per group so no group is all-NaN
    sparse[:, :2] = (np.arange(n_groups) * 0.5)[:, None] + [0.0, 0.1]

    obs = pd.DataFrame(
        {"Metadata_Perturbation": np.repeat([f"g{group}" for group in range(n_groups)], per_group)},
        index=[str(index) for index in range(n_obs)],
    )
    # Bound to their names here, so a column cannot drift away from the name the assertion reads.
    features = {
        "Cells_AreaShape_Plain": plain,
        "Cells_AreaShape_Sparse": sparse.ravel(),
        "Cells_Intensity_Ratio": ratio,
        "Cells_AreaShape_Jitter": jitter,
    }
    adata = ad.AnnData(X=np.column_stack(list(features.values())).astype(np.float32), obs=obs)
    adata.var_names = list(features)
    return adata


def test_each_operation_counts_the_features_it_actually_removed():
    """Each operation meets a case built to give it exactly one feature to remove, so turning it into a no-op that selects everything fails on the count and misattributing its removal fails on the name.

    ``drop_outliers`` and ``noise_removal`` have no other cover; ``drop_na_columns`` is also caught by the order test above, and is kept here because this states the tighter property.
    """
    adata = _one_feature_per_operation()
    operations = ("drop_na_columns", "drop_outliers", "noise_removal")

    # Passed rather than defaulted, so a change to the defaults cannot fail this for an unrelated reason.
    mt.pp.feature_select(
        adata, operations=operations, na_cutoff=0.05, outlier_cutoff=500.0, noise_removal_stdev_cutoff=0.8
    )

    assert adata.uns["mantispy"]["feature_select"] == dict.fromkeys(operations, 1)
    assert set(adata.var_names[~adata.var["selected"]]) == {
        "Cells_AreaShape_Sparse",
        "Cells_Intensity_Ratio",
        "Cells_AreaShape_Jitter",
    }


def test_nothing_is_dropped_or_reordered(wells):
    before = list(wells.var_names)
    mt.pp.feature_select(wells)
    assert list(wells.var_names) == before


def test_key_added_and_unknown_operation(wells):
    mt.pp.feature_select(wells, key_added="my_key")
    assert "my_key" in wells.var
    with pytest.raises(ValueError, match="unknown operation"):
        mt.pp.feature_select(wells, operations=("nonsense",))


def test_noise_removal_needs_its_grouping_column(wells):
    wells.obs = wells.obs.drop(columns="Metadata_Perturbation")
    with pytest.raises(KeyError, match="group replicates by"):
        mt.pp.feature_select(wells, operations=("noise_removal",))


def test_subset_features_requires_the_key(wells):
    with pytest.raises(KeyError, match="feature_select"):
        mt.pp.subset_features(wells, key="missing")


def test_correlated_copies_are_broken_up(cells):
    """The synthetic plate injects near-duplicate features; selection must split them."""
    from mantispy.ds import synthetic_plate

    plate = synthetic_plate(n_wells=48, n_cells=10, n_features=20, n_correlated_pairs=3, seed=0)
    profiles = mt.tl.aggregate(plate, min_cells=0)
    mt.pp.feature_select(profiles, operations=("correlation_threshold",), corr_threshold=0.9)
    selected = set(profiles.var_names[profiles.var["selected"]])
    for original, copy in profiles.uns["mantispy"]["truth"]["correlated_pairs"]:
        assert not ({original, copy} <= selected), f"{original} and {copy} both survived"


def test_an_unknown_blocklist_name_is_an_error(wells):
    """Falling back to the bundled list would return the wrong feature set."""
    with pytest.raises(ValueError, match="unknown blocklist"):
        mt.pp.feature_select(wells, operations=("blocklist",), blocklist="not_a_real_list")
    with pytest.raises(ValueError, match="unknown blocklist"):
        mt.pp.filter_features(wells, blocklist="not_a_real_list")


def test_the_same_input_always_gives_the_same_mask(wells):
    """No operation draws a random number, so a staged selection stays reproducible."""
    import numpy as np

    masks = []
    for _ in range(2):
        trial = wells.copy()
        mt.pp.feature_select(trial)
        masks.append(trial.var["selected"].to_numpy())

    assert np.array_equal(*masks)


def test_an_all_nan_feature_is_selected_on_without_a_warning():
    """Regression test for #57: the nan-functions warn through ``warnings``, which ``np.errstate`` does not reach.

    A measurement that failed on every cell looks like this, and it left three RuntimeWarnings in the caller's output.
    """
    adata = _one_feature_per_operation()
    adata.X[:, adata.var_names.get_loc("Cells_AreaShape_Plain")] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        mt.pp.feature_select(adata, operations=("variance_threshold", "drop_outliers", "noise_removal"), na_cutoff=0.05)

    assert not adata.var["selected"]["Cells_AreaShape_Plain"], "an all-NaN feature has no variance to keep it"


def test_selecting_nothing_warns_rather_than_emptying_the_object_silently(wells):
    """noise_removal's cutoff is an absolute threshold on the scale normalize left the values on, so against
    control-normalized values it drops every feature, and on its own that surfaces as an empty matrix in
    whatever runs next."""
    with pytest.warns(UserWarning, match="flagged none of the"):
        mt.pp.feature_select(
            wells,
            operations=("noise_removal",),
            noise_removal_perturb_groups="Metadata_Perturbation",
            noise_removal_stdev_cutoff=0.0,
        )

    assert not wells.var["selected"].any()
    assert wells.uns["mantispy"]["feature_select"]["noise_removal"] == wells.n_vars


def _multiple_r(X: np.ndarray) -> np.ndarray:
    """The multiple correlation of each column with all the others, for the redundancy invariant."""
    Xs = (X - X.mean(0)) / X.std(0)
    out = np.empty(Xs.shape[1])
    for j in range(Xs.shape[1]):
        others = np.delete(Xs, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, Xs[:, j], rcond=None)
        out[j] = np.sqrt(max(1 - (Xs[:, j] - others @ beta).var() / Xs[:, j].var(), 0.0))
    return out


def _collinear_profiles():
    """a, b, c independent, d = a + b + c (collinear with no large pairwise correlation), and an independent e."""
    rng = np.random.default_rng(0)
    n = 500
    a, b, c, e = (rng.standard_normal(n) for _ in range(4))
    d = a + b + c + 0.02 * rng.standard_normal(n)
    names = [f"Cells_Intensity_{name}" for name in ("a", "b", "c", "d", "e")]
    adata = ad.AnnData(np.column_stack([a, b, c, d, e]).astype(np.float64), var=pd.DataFrame(index=names))
    return adata


def test_decorrelate_removes_multivariate_redundancy_correlation_threshold_misses():
    """Regression test for #131: d = a + b + c has no pairwise |r| > 0.9 with any single feature, so
    correlation_threshold keeps it, but it carries no new information and decorrelate removes it."""
    adata = _collinear_profiles()

    kept_corr = adata.copy()
    mt.pp.feature_select(kept_corr, operations=("correlation_threshold",), corr_threshold=0.9)
    assert kept_corr.var["selected"].all(), "no pair exceeds 0.9, so correlation_threshold removes nothing"

    kept_decorr = adata.copy()
    mt.pp.feature_select(kept_decorr, operations=(), decorrelate=True, decorr_threshold=0.9)
    selected = kept_decorr.var["selected"].to_numpy()
    assert int((~selected).sum()) == 1, "exactly one of the four collinear features is redundant"
    assert (_multiple_r(adata.X[:, selected]) < 0.9).all(), "a survivor is still predictable from the others"


def test_absolute_correlation_threshold_reduces_an_anticorrelated_pair():
    """correlation_threshold thresholds signed correlation, so r = -1 keeps both; corr_absolute drops one."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal(400)
    values = np.column_stack([a, -a + 1e-6 * rng.standard_normal(400), rng.standard_normal(400)])
    adata = ad.AnnData(values.astype(np.float64), var=pd.DataFrame(index=[f"Cells_Intensity_{i}" for i in range(3)]))

    signed = adata.copy()
    mt.pp.feature_select(signed, operations=("correlation_threshold",), corr_threshold=0.9)
    assert signed.var["selected"][:2].all(), "signed threshold keeps an anti-correlated pair"

    absolute = adata.copy()
    mt.pp.feature_select(absolute, operations=("correlation_threshold",), corr_threshold=0.9, corr_absolute=True)
    assert int(absolute.var["selected"][:2].sum()) == 1, "absolute threshold drops one of the pair"


def test_iterative_correlation_drop_removes_the_shared_feature():
    """On the chain 0-1-2 the single-pass rule can drop both leaves; iterative drops the connector 1,
    keeping the two features that are not correlated with each other."""
    from mantispy.pp._select import _iterative_correlation_drop

    pairs = np.array([[0, 1], [1, 2]])
    total = np.array([10.0, 1.0, 10.0])  # leaves rank higher, so a by-total rule would drop them
    keep = _iterative_correlation_drop(pairs, total, n_vars=3)
    assert keep.tolist() == [True, False, True]


def test_iterative_correlation_threshold_keeps_at_least_as_many_and_leaves_no_pair():
    """On a correlated screen the iterative absolute path removes all over-threshold pairs and keeps at
    least as many features as the single pass."""
    from mantispy._core._corr import correlated_pairs

    rng = np.random.default_rng(2)
    latent = rng.standard_normal((300, 5))
    blocks = [latent[:, k : k + 1] + 0.1 * rng.standard_normal((300, 3)) for k in range(5)]
    values = np.column_stack(blocks + [rng.standard_normal((300, 4))])
    adata = ad.AnnData(
        values.astype(np.float64), var=pd.DataFrame(index=[f"Cells_f{i}" for i in range(values.shape[1])])
    )

    single, iterative = adata.copy(), adata.copy()
    mt.pp.feature_select(single, operations=("correlation_threshold",), corr_threshold=0.9, corr_absolute=True)
    mt.pp.feature_select(
        iterative, operations=("correlation_threshold",), corr_threshold=0.9, corr_absolute=True, corr_iterative=True
    )
    assert int(iterative.var["selected"].sum()) >= int(single.var["selected"].sum())

    kept = iterative.var["selected"].to_numpy()
    remaining, _ = correlated_pairs(adata.X[:, kept], 0.9, absolute=True)
    assert remaining.size == 0, "the iterative path left an over-threshold pair"


def test_decorrelate_is_deterministic():
    """The QR selection draws no random numbers, so the same input always gives the same mask.

    (The kept set is not order-independent: column-pivoted QR breaks near-ties by column order.)
    """
    adata = _collinear_profiles()
    masks = []
    for _ in range(2):
        trial = adata.copy()
        mt.pp.feature_select(trial, operations=(), decorrelate=True)
        masks.append(trial.var["selected"].to_numpy())
    assert np.array_equal(*masks)


def test_decorrelate_rejects_a_threshold_outside_the_unit_interval():
    """A threshold >= 1 would make sqrt(1 - t**2) NaN and silently drop everything; reject it."""
    adata = _collinear_profiles()
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        mt.pp.feature_select(adata, operations=(), decorrelate=True, decorr_threshold=1.0)


def test_decorrelate_runs_as_a_post_step_and_only_removes(wells):
    """decorrelate=True adds a count and can only shrink the set the operations already chose."""
    without = wells.copy()
    mt.pp.feature_select(without)
    with_decorr = wells.copy()
    mt.pp.feature_select(with_decorr, decorrelate=True)

    assert "decorrelate" not in without.uns["mantispy"]["feature_select"]
    assert "decorrelate" in with_decorr.uns["mantispy"]["feature_select"]
    # A subset of what the operations kept: never adds a feature back.
    assert set(with_decorr.var_names[with_decorr.var["selected"]]) <= set(without.var_names[without.var["selected"]])


def test_features_normalize_could_not_scale_are_dropped_before_the_rest_are_judged():
    """drop_degenerate drops what normalize flagged, and the other operations never see it: judged alongside
    the rest, the flagged feature here would push a healthy one out through the correlation ranking."""
    rng = np.random.default_rng(2)
    values = rng.normal(size=(60, 4)) @ rng.normal(size=(4, 4))
    adata = ad.AnnData(values.astype(np.float32), var=pd.DataFrame(index=[f"Cells_Intensity_f{i}" for i in range(4)]))
    adata.var["degenerate_scale"] = [False, False, False, True]

    mt.pp.feature_select(adata)

    assert adata.var["selected"].tolist() == [True, True, True, False]
    assert adata.uns["mantispy"]["feature_select"]["drop_degenerate"] == 1
