"""Effect sizes and Wasserstein distances against the controls."""

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

import mantispy as mt
from mantispy.tl._effect import _wasserstein_columns


@pytest.fixture
def perturbed():
    return mt.ds.synthetic_plate(n_wells=48, n_cells=40, n_features=20, n_perturbations=3, effect_size=3.0, seed=0)


@pytest.mark.parametrize("method", ["cohens_d", "robust_z"])
def test_the_controls_have_no_effect_against_themselves(perturbed, method):
    mt.tl.effect_size(perturbed, method=method)
    table = perturbed.uns["mantispy"]["effect"]
    assert table[table["group"] == "DMSO"]["effect"].abs().median() < 0.3


def test_a_group_too_small_to_score_stays_missing_rather_than_zero(perturbed):
    mt.tl.effect_size(perturbed, min_obs=10**6)
    assert np.isnan(perturbed.varm["effect"]).all()


def test_the_pvalues_can_be_skipped(perturbed):
    """They dominate the runtime (179 s against under a second on a JUMP-sized screen), and at cell resolution nearly everything is significant."""
    mt.tl.effect_size(perturbed, pvalues=False)
    table = perturbed.uns["mantispy"]["effect"]
    assert table["pvalue"].isna().all()
    assert np.isfinite(perturbed.varm["effect"]).any()

    with_values = mt.tl.effect_size(perturbed, copy=True)
    np.testing.assert_allclose(with_values.varm["effect"], perturbed.varm["effect"], rtol=1e-6)


def test_a_features_pvalue_does_not_depend_on_its_neighbours_in_the_array():
    """One tied feature must not drag the others onto the normal approximation.

    scipy picks exact-or-asymptotic once for a whole 2-D call and looks for ties across every column at once, so a single repeated value in one feature changes every other feature's p-value.
    At three treated wells the shift is four orders of magnitude, on the low-replicate screens that have no resolution to spare.
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
            {
                "Metadata_Perturbation": labels,
                "Metadata_Control": [n == "DMSO" for n in labels],
                "Metadata_Control_Type": ["negcon" if n == "DMSO" else "treatment" for n in labels],
            },
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

    # The untied features reach the exact null and land an order of magnitude below the tied one.
    assert got.iloc[1] * 10 < got.iloc[0]


def _untied_screen(n_treated, n_control, n_features=4, seed=0):
    """A well-level screen whose treated values all sit above the controls, so no pooled sample has a tie."""
    import anndata as ad
    import pandas as pd

    from mantispy._core.schema import stamp

    rng = np.random.default_rng(seed)
    control = rng.standard_normal((n_control, n_features))
    treated = rng.standard_normal((n_treated, n_features)) + 50.0
    values = np.vstack([treated, control]).astype(np.float64)
    labels = ["pert"] * n_treated + ["DMSO"] * n_control
    adata = ad.AnnData(
        X=values,
        obs=pd.DataFrame(
            {
                "Metadata_Perturbation": labels,
                "Metadata_Control": [n == "DMSO" for n in labels],
                "Metadata_Control_Type": ["negcon" if n == "DMSO" else "treatment" for n in labels],
            },
            index=[str(i) for i in range(len(labels))],
        ),
    )
    adata.var_names = [f"Cells_AreaShape_F{i}" for i in range(n_features)]
    stamp(adata, resolution="well")
    return adata, values, n_treated


def test_exact_false_uses_the_normal_approximation_even_on_small_samples():
    """exact=False sends even a small, untied group to the fast normal approximation, not the exact null."""
    from scipy.stats import mannwhitneyu

    adata, values, n_t = _untied_screen(n_treated=4, n_control=40, seed=1)
    mt.tl.effect_size(adata, exact=False, key_added="e")
    got = adata.uns["mantispy"]["e"].query("group == 'pert'").set_index("feature")["pvalue"]
    for i, name in enumerate(adata.var_names):
        expected = float(mannwhitneyu(values[:n_t, i], values[n_t:, i], method="asymptotic").pvalue)
        assert got[name] == pytest.approx(expected, rel=1e-9), name


def test_exact_true_uses_the_exact_null_even_on_large_samples():
    """exact=True takes the exact null where auto would have used the approximation (both samples past eight)."""
    from scipy.stats import mannwhitneyu

    adata, values, n_t = _untied_screen(n_treated=12, n_control=30, seed=2)
    mt.tl.effect_size(adata, exact=True, key_added="e")
    exact = adata.uns["mantispy"]["e"].query("group == 'pert'").set_index("feature")["pvalue"]
    for i, name in enumerate(adata.var_names):
        expected = float(mannwhitneyu(values[:n_t, i], values[n_t:, i], method="exact").pvalue)
        assert exact[name] == pytest.approx(expected, rel=1e-9), name

    mt.tl.effect_size(adata, exact=False, key_added="a")
    asymptotic = adata.uns["mantispy"]["a"].query("group == 'pert'").set_index("feature")["pvalue"]
    assert not np.allclose(exact.to_numpy(), asymptotic.to_numpy())


def test_the_auto_default_matches_scipy_exact_on_a_small_untied_group():
    """The default must still match scipy's own exact p-value for a small untied group."""
    from scipy.stats import mannwhitneyu

    adata, values, n_t = _untied_screen(n_treated=5, n_control=6, seed=3)
    mt.tl.effect_size(adata, key_added="e")
    got = adata.uns["mantispy"]["e"].query("group == 'pert'").set_index("feature")["pvalue"]
    for i, name in enumerate(adata.var_names):
        expected = float(mannwhitneyu(values[:n_t, i], values[n_t:, i], method="exact").pvalue)
        assert got[name] == pytest.approx(expected, rel=1e-9), name


def test_exact_rejects_an_unknown_value(perturbed):
    with pytest.raises(ValueError, match="exact must be"):
        mt.tl.effect_size(perturbed, exact="sometimes")


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
