"""Effect sizes and Wasserstein distances against the controls."""

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

import mantispy as mt
from mantispy.tl._effect import _wasserstein_columns


@pytest.fixture
def perturbed():
    return mt.ds.synthetic_plate(n_wells=48, n_cells=40, n_features=20, n_perturbations=3, effect_size=3.0, seed=0)


def test_effect_size_recovers_the_injected_features(perturbed):
    mt.tl.effect_size(perturbed)
    truth = perturbed.uns["mantispy"]["truth"]["affected_features"]
    group = next(name for name, features in truth.items() if features)

    table = perturbed.uns["mantispy"]["effect"]
    scores = table[table["group"] == group].set_index("feature")["effect"].abs()
    assert scores[truth[group]].min() > scores.drop(index=truth[group]).max()


def test_effect_size_writes_a_matrix_and_a_tidy_table(perturbed):
    mt.tl.effect_size(perturbed)
    groups = perturbed.uns["mantispy"]["effect_groups"]
    assert perturbed.varm["effect"].shape == (perturbed.n_vars, len(groups))

    table = perturbed.uns["mantispy"]["effect"]
    assert {"group", "feature", "effect", "pvalue", "qvalue"} <= set(table.columns)
    assert (table["qvalue"] >= table["pvalue"] - 1e-12).all()
    # The tidy table and the matrix are the same numbers in the same order.
    first = table[table["group"] == groups[0]]["effect"].to_numpy()
    np.testing.assert_allclose(first, perturbed.varm["effect"][:, 0], rtol=1e-6)


@pytest.mark.parametrize("method", ["cohens_d", "robust_z"])
def test_the_controls_have_no_effect_against_themselves(perturbed, method):
    """The calibration check: whatever the estimator, DMSO against DMSO is nothing."""
    mt.tl.effect_size(perturbed, method=method)
    table = perturbed.uns["mantispy"]["effect"]
    assert table[table["group"] == "DMSO"]["effect"].abs().median() < 0.3


def test_wasserstein_sees_a_change_in_spread_that_a_mean_shift_misses(perturbed):
    rows = (perturbed.obs["Metadata_Perturbation"] == "pert00").to_numpy()
    values = perturbed.X.copy()
    values[rows, 0] = np.random.default_rng(0).standard_normal(int(rows.sum())) * 6.0
    perturbed.X = values

    mt.tl.effect_size(perturbed)
    mt.tl.wasserstein_features(perturbed)
    position = perturbed.uns["mantispy"]["effect_groups"].index("pert00")
    assert abs(float(perturbed.varm["effect"][0, position])) < 0.5
    assert float(perturbed.varm["wasserstein"][0, position]) > 1.0


def test_the_vectorised_wasserstein_matches_scipy_with_and_without_gaps():
    rng = np.random.default_rng(0)
    treated, control = rng.normal(1, 2, (60, 8)), rng.normal(size=(90, 8))
    treated[3, 5] = np.nan  # one ragged column takes the scalar path

    ours = _wasserstein_columns(treated, control)
    theirs = [
        wasserstein_distance(treated[np.isfinite(treated[:, j]), j], control[:, j]) for j in range(treated.shape[1])
    ]
    np.testing.assert_allclose(ours, theirs, rtol=1e-10)


def test_a_group_too_small_to_score_stays_missing_rather_than_zero(perturbed):
    mt.tl.effect_size(perturbed, min_obs=10**6)
    assert np.isnan(perturbed.varm["effect"]).all()


def test_results_survive_a_round_trip(perturbed, tmp_path):
    mt.tl.effect_size(perturbed)
    mt.tl.wasserstein_features(perturbed)
    mt.io.write(perturbed, tmp_path / "effects.h5ad")
    loaded = mt.io.read(tmp_path / "effects.h5ad")
    assert loaded.varm["effect"].shape == perturbed.varm["effect"].shape
    assert len(loaded.uns["mantispy"]["wasserstein"]) == perturbed.n_vars * 4


def test_the_pvalues_can_be_skipped(perturbed):
    """They dominate the runtime (179 s against under a second on a JUMP-sized screen), and
    at cell resolution nearly everything is significant."""
    mt.tl.effect_size(perturbed, pvalues=False)
    table = perturbed.uns["mantispy"]["effect"]
    assert table["pvalue"].isna().all()
    assert np.isfinite(perturbed.varm["effect"]).any()

    with_values = mt.tl.effect_size(perturbed, copy=True)
    np.testing.assert_allclose(with_values.varm["effect"], perturbed.varm["effect"], rtol=1e-6)


@pytest.mark.parametrize(
    ("kind", "n_treated"),
    [("continuous", 14), ("ties", 14), ("missing", 14), ("small group", 4)],
)
def test_the_fast_mann_whitney_matches_scipy(kind, n_treated):
    """The searchsorted path must reproduce scipy's p-values.

    Groups of eight or fewer take scipy's exact branch and are checked against
    ``method="auto"``; larger ones take the normal approximation, as scipy does there.
    """
    from scipy.stats import mannwhitneyu

    from mantispy._core._stats import mannwhitney_pvalues, sorted_control

    rng = np.random.default_rng(0)
    if kind == "ties":
        control = rng.integers(0, 4, (300, 40)).astype(float)
        treated = rng.integers(0, 4, (n_treated, 40)).astype(float)
    else:
        control = rng.standard_normal((300, 40))
        treated = rng.standard_normal((n_treated, 40))
    if kind == "missing":
        control[rng.random(control.shape) < 0.05] = np.nan
        treated[rng.random(treated.shape) < 0.05] = np.nan

    expected = mannwhitneyu(treated, control, axis=0, method="asymptotic", nan_policy="omit").pvalue
    np.testing.assert_array_equal(mannwhitney_pvalues(treated, sorted_control(control)), expected)


def test_effect_size_pvalues_are_unchanged_by_the_fast_path(perturbed):
    """End to end: the table must hold what a per-group scipy call would have written."""
    from scipy.stats import mannwhitneyu

    from mantispy._core._reduce import get_matrix, group_codes
    from mantispy._core.masks import reference_mask

    mt.tl.effect_size(perturbed)
    values = get_matrix(perturbed).astype(np.float64)
    control = values[reference_mask(perturbed, "negcon")]
    codes, keys = group_codes(perturbed, "Metadata_Perturbation")
    table = perturbed.uns["mantispy"]["effect"]

    for index, key in enumerate(keys):
        treated = values[codes == index]
        if treated.shape[0] < 2:
            continue
        expected = mannwhitneyu(treated, control, axis=0, nan_policy="omit").pvalue
        actual = table.loc[table["group"] == str(key), "pvalue"].to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("kind", ["continuous", "ties", "single replicate"])
def test_the_pre_sorted_wasserstein_matches_scipy(kind):
    """Walking a reference sorted once must reproduce scipy's distance."""
    from mantispy._core._numba import _wasserstein_against

    rng = np.random.default_rng(0)
    n_treated = 1 if kind == "single replicate" else 11
    if kind == "ties":
        control = rng.integers(0, 4, (6, 400)).astype(float)
        treated = rng.integers(0, 4, (6, n_treated)).astype(float)
    else:
        control = rng.standard_normal((6, 400))
        treated = rng.standard_normal((6, n_treated))

    ranked = np.ascontiguousarray(np.sort(control, axis=1))
    actual = _wasserstein_against(np.ascontiguousarray(treated), ranked, np.full(6, control.shape[1], np.int64))
    expected = [wasserstein_distance(treated[row], control[row]) for row in range(6)]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_wasserstein_features_is_unchanged_by_the_pre_sorted_path(perturbed):
    """End to end: the same distances the per-group path would have written."""
    from mantispy._core._reduce import get_matrix, group_codes
    from mantispy._core.masks import reference_mask

    mt.tl.wasserstein_features(perturbed)
    values = get_matrix(perturbed)
    control = values[reference_mask(perturbed, "negcon")]
    codes, keys = group_codes(perturbed, "Metadata_Perturbation")

    for index in range(len(keys)):
        expected = _wasserstein_columns(values[codes == index], control)
        np.testing.assert_allclose(perturbed.varm["wasserstein"][:, index], expected, rtol=1e-5, atol=1e-6)


def test_a_features_pvalue_does_not_depend_on_its_neighbours_in_the_array():
    """One tied feature must not drag the others onto the normal approximation.

    scipy picks exact-or-asymptotic once for a whole 2-D call and looks for ties across
    every column at once, so a single repeated value in one feature changes every other
    feature's p-value. At three treated wells the shift is four orders of magnitude, on the
    low-replicate screens that have no resolution to spare.
    """
    import anndata as ad
    import pandas as pd
    from scipy.stats import mannwhitneyu

    from mantispy._core.schema import stamp

    rng = np.random.default_rng(0)
    control = rng.standard_normal((60, 3))
    treated = rng.standard_normal((3, 3)) + 50.0  # every treated value above every control
    control[0, 0] = control[1, 0]  # one tie, in feature 0 only

    values = np.vstack([treated, control])
    labels = ["pert"] * 3 + ["DMSO"] * 60
    adata = ad.AnnData(
        X=values.astype(np.float32),
        obs=pd.DataFrame(
            {"Metadata_Perturbation": labels, "Metadata_Control": [n == "DMSO" for n in labels]},
            index=[str(i) for i in range(len(labels))],
        ),
    )
    adata.var_names = [f"Cells_AreaShape_F{i}" for i in range(3)]
    stamp(adata, resolution="well")

    mt.tl.effect_size(adata, key_added="e")
    table = adata.uns["mantispy"]["e"]
    got = table[table["group"] == "pert"].set_index("feature")["pvalue"]

    alone = {
        name: float(mannwhitneyu(values[:3, i].astype(np.float32), values[3:, i].astype(np.float32)).pvalue)
        for i, name in enumerate(adata.var_names)
    }
    for name, expected in alone.items():
        assert got[name] == pytest.approx(expected, rel=1e-9), name

    # The untied features reach the exact null and land an order of magnitude below the
    # tied one, which keeps the approximation.
    assert got.iloc[1] * 10 < got.iloc[0]


def test_an_infinity_is_dropped_like_a_missing_value_rather_than_returned_as_a_distance(perturbed):
    """wasserstein_features splits usable columns on isfinite and ragged ones on isnan, so an inf column fell between the two and came back inf."""
    rng = np.random.default_rng(0)
    treated, control = rng.normal(1, 2, (60, 8)), rng.normal(size=(90, 8))
    treated[3, 5] = np.inf

    expected = wasserstein_distance(treated[np.isfinite(treated[:, 5]), 5], control[:, 5])
    assert _wasserstein_columns(treated, control)[5] == pytest.approx(expected, rel=1e-10)
    # The mask wasserstein_features passes: complete columns are left to the pre-sorted path.
    usable = np.isfinite(control).all(axis=0) & np.isfinite(treated).all(axis=0)
    assert _wasserstein_columns(treated, control, only=~usable)[5] == pytest.approx(expected, rel=1e-10)

    values = perturbed.X.copy()
    values[np.flatnonzero(perturbed.obs["Metadata_Control"].to_numpy())[0], 0] = np.inf
    perturbed.X = values
    mt.tl.wasserstein_features(perturbed)
    assert np.isfinite(perturbed.varm["wasserstein"]).all()


def _one_inf_control(n_control: int, n_treated: int, n_features: int = 3, seed: int = 1):
    """Well profiles whose controls hold one ``-inf``, in one feature, with nothing missing."""
    import anndata as ad
    import pandas as pd

    from mantispy._core.schema import stamp

    rng = np.random.default_rng(seed)
    values = np.vstack([rng.standard_normal((n_control, n_features)), rng.standard_normal((n_treated, n_features))])
    values[2, 1] = -np.inf  # a control row, in one feature only
    labels = ["DMSO"] * n_control + ["pert"] * n_treated
    adata = ad.AnnData(
        X=values,
        obs=pd.DataFrame(
            {"Metadata_Perturbation": labels, "Metadata_Control": [label == "DMSO" for label in labels]},
            index=[str(index) for index in range(len(labels))],
        ),
    )
    adata.var_names = [f"Cells_AreaShape_F{index}" for index in range(n_features)]
    stamp(adata, resolution="well")
    return adata, values[n_control:], values[:n_control]


# An infinite control value gives the pooled spread of that feature no value, so the effect
# estimate is NaN and numpy says so. This test is about the p-values.
@pytest.mark.filterwarnings("ignore:invalid value encountered")
def test_an_infinity_does_not_choose_a_different_branch_than_scipy_would():
    """An infinity counted as unmeasured put nine controls at eight, which takes scipy's exact branch where its own auto takes the asymptotic one."""
    from scipy.stats import mannwhitneyu

    adata, treated, control = _one_inf_control(n_control=9, n_treated=12)
    mt.tl.effect_size(adata, key_added="e")

    table = adata.uns["mantispy"]["e"]
    got = table[table["group"] == "pert"].set_index("feature")["pvalue"]
    expected = mannwhitneyu(treated, control, axis=0, nan_policy="omit").pvalue
    np.testing.assert_allclose(got.to_numpy(), expected, rtol=1e-9)


def test_an_infinite_control_value_does_not_shift_the_pre_sorted_reference():
    """Counting only the finite values left the reference one short, which cuts the largest control off its end."""
    adata, treated, control = _one_inf_control(n_control=30, n_treated=12)
    mt.tl.wasserstein_features(adata)

    column = adata.uns["mantispy"]["wasserstein_groups"].index("pert")
    expected = [
        wasserstein_distance(treated[np.isfinite(treated[:, j]), j], control[np.isfinite(control[:, j]), j])
        for j in range(treated.shape[1])
    ]
    np.testing.assert_allclose(adata.varm["wasserstein"][:, column], expected, rtol=1e-5)
