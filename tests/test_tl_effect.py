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

    # The untied features reach the exact null and land an order of magnitude below the tied one.
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
