import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy._core.plate import well_col, well_row
from mantispy.ds import synthetic_plate


@pytest.fixture
def gradient_cells():
    return synthetic_plate(n_plates=2, n_wells=96, n_cells=8, n_features=10, row_gradient=3.0, col_gradient=2.0, seed=0)


def _position_correlation(adata, feature=0):
    rows = np.array([well_row(w) for w in adata.obs["Metadata_Well"]])
    cols = np.array([well_col(w) for w in adata.obs["Metadata_Well"]])
    values = adata.X[:, feature]
    return abs(np.corrcoef(rows, values)[0, 1]), abs(np.corrcoef(cols, values)[0, 1])


def test_median_polish_removes_row_and_column_gradients(gradient_cells):
    wells = mt.tl.aggregate(gradient_cells, min_cells=0)
    before = _position_correlation(wells)
    assert before[0] > 0.5 and before[1] > 0.3

    mt.pp.correct_plate_position(wells)
    after = _position_correlation(wells)
    assert after[0] < 0.15 and after[1] < 0.15


def test_polish_records_effects_and_honours_key_added(gradient_cells):
    wells = mt.tl.aggregate(gradient_cells, min_cells=0)
    before = wells.X.copy()
    mt.pp.correct_plate_position(wells, key_added="positioned")
    np.testing.assert_array_equal(wells.X, before)
    assert wells.layers["positioned"].shape == wells.shape
    assert set(wells.uns["mantispy"]["plate_position"]) == set(wells.obs["Metadata_Plate"].astype(str))


def test_regress_out_removes_cell_count_dependence():
    cells = synthetic_plate(n_wells=96, n_cells=30, n_features=10, confounder_effect=3.0, seed=0)
    wells = mt.tl.aggregate(cells, min_cells=0)
    confounded = wells.uns["mantispy"]["truth"]["confounded_features"][0]
    position = wells.var_names.get_loc(confounded)
    counts = wells.obs["Metadata_CellCount"].to_numpy(dtype=float)

    assert abs(np.corrcoef(counts, wells.X[:, position])[0, 1]) > 0.3
    mt.pp.regress_out(wells, keys=("Metadata_CellCount",), by=None)
    assert abs(np.corrcoef(counts, wells.X[:, position])[0, 1]) < 0.05


def test_regress_out_handles_categorical_keys_and_missing_columns(gradient_cells):
    wells = mt.tl.aggregate(gradient_cells, min_cells=0)
    plate = wells.obs["Metadata_Plate"].astype(str).to_numpy()

    def between_plate_spread(adata):
        values = np.asarray(adata.X)
        return float(np.std([values[plate == p].mean(0) for p in sorted(set(plate))], axis=0).mean())

    before = between_plate_spread(wells)
    mt.pp.regress_out(wells, keys=("Metadata_Plate",), by=None)
    # Regressing out plate identity globally must remove the between-plate difference.
    # A finiteness check alone would also pass for a function that does nothing.
    assert between_plate_spread(wells) < 0.01 * before
    assert np.isfinite(wells.X).all()

    with pytest.raises(KeyError, match="Metadata_Nope"):
        mt.pp.regress_out(wells, keys=("Metadata_Nope",))


def test_harmony_says_so_when_it_corrects_nothing(cells):
    """harmonypy reports convergence and returns the input unchanged when the batches are
    perfectly separated in the embedding, because every soft cluster is then single-batch.
    The wrapper warns in that case."""
    pytest.importorskip("harmonypy")
    import scanpy as sc

    wells = mt.tl.aggregate(cells, min_cells=0)
    wells.obs["Metadata_Site"] = np.where(wells.obs["Metadata_Plate"].astype(str) == "Plate01", "north", "south")
    values = np.asarray(wells.X, dtype=float).copy()
    values[(wells.obs["Metadata_Site"] == "north").to_numpy()] += 5.0  # completely separable
    wells.X = values.astype(np.float32)
    sc.pp.pca(wells, n_comps=10)

    with pytest.warns(UserWarning, match="corrected nothing"):
        mt.pp.harmony(wells, batch_key="Metadata_Site", use_rep="X_pca")
    np.testing.assert_array_equal(wells.obsm["X_harmony"], wells.obsm["X_pca"])


def test_harmony_needs_something_to_correct(cells):
    pytest.importorskip("harmonypy")
    import scanpy as sc

    wells = mt.tl.aggregate(cells, min_cells=0)
    sc.pp.pca(wells, n_comps=5)
    with pytest.raises(KeyError, match="sc.pp.pca"):
        mt.pp.harmony(wells, batch_key="Metadata_Plate", use_rep="X_absent")
    wells.obs["Metadata_OneSite"] = "only"
    with pytest.raises(ValueError, match="one level"):
        mt.pp.harmony(wells, batch_key="Metadata_OneSite")


def test_one_infinity_does_not_spread_across_features(gradient_cells):
    """A single inf affects only its own value.

    ``np.linalg.lstsq`` with several right-hand sides returns NaN coefficients for all of
    them when any one column holds an infinity, which would turn the whole plate into NaN.
    Three JUMP plates carry one inf each. Regression test for #66: the feature holding it
    was then fitted with the inf and lost on its plate.
    """
    adata = gradient_cells.copy()
    adata.obs["Metadata_CellCount"] = np.arange(adata.n_obs) % 37 + 10
    values = np.asarray(adata.X, dtype=np.float32).copy()
    values[0, 2] = np.inf
    adata.X = values
    gapped = adata.copy()
    gapped.X[0, 2] = np.nan

    for each in (adata, gapped):
        mt.pp.regress_out(each, keys=("Metadata_CellCount",), key_added="regressed")
    corrected, expected = adata.layers["regressed"], gapped.layers["regressed"]

    assert np.isinf(corrected[0, 2]), "the infinity stays where it was"
    corrected[0, 2] = expected[0, 2] = 0.0
    np.testing.assert_array_equal(corrected, expected)  # everything else is fitted as if it were missing
    assert np.isfinite(corrected).all()


def _two_plates(count_means=(1500.0, 1500.0), slope=0.0, n=192, n_features=30, seed=0):
    """Two plates with identical biology, differing only in how confluent they are."""
    from mantispy.ds import synthetic_plate

    adata = synthetic_plate(n_plates=2, n_wells=n, n_cells=2, n_features=n_features, seed=seed)
    adata = mt.tl.aggregate(adata, min_cells=0)
    plate = adata.obs["Metadata_Plate"].astype(str).to_numpy()
    generator = np.random.default_rng(seed)
    first, second = sorted(set(plate))
    counts = np.where(
        plate == first,
        generator.normal(count_means[0], 120, adata.n_obs),
        generator.normal(count_means[1], 120, adata.n_obs),
    )
    adata.obs["Metadata_Count"] = counts
    values = generator.normal(0.0, 1.13, adata.shape) + slope * (counts - 1500.0)[:, None]
    adata.X = values.astype(np.float32)
    return adata


def test_regress_out_says_when_a_plate_never_reaches_the_pooled_density():
    """On pki two plates hold a third of the others' cells; fitted per plate and re-expressed at the pooled
    mean, the result correlated with the cell count more than the input had."""
    adata = _two_plates(count_means=(600.0, 1800.0), slope=0.002)
    with pytest.warns(UserWarning, match="extrapolating"):
        mt.pp.regress_out(adata, keys=["Metadata_Count"], by="Metadata_Plate")


def _treated_and_thinned(seed=0):
    """Controls whose features follow density for technical reasons, and a treatment that both thins the wells
    and has a phenotype of its own, so a fit over every well mistakes the phenotype for density."""
    import anndata as ad

    from mantispy._core.schema import stamp

    generator = np.random.default_rng(seed)
    control = np.arange(200) < 100
    count = np.where(control, generator.normal(1500, 150, 200), generator.normal(900, 150, 200))
    values = 0.002 * (count - 1500)[:, None] + np.where(control, 0.0, 3.0)[:, None] + generator.normal(0, 0.1, (200, 4))
    obs = pd.DataFrame(
        {
            "Metadata_Plate": "P1",
            "Metadata_Well": [f"W{index:03d}" for index in range(200)],
            "Metadata_Control": control,
            "Metadata_CellCount": count,
        },
        index=[str(index) for index in range(200)],
    )
    adata = ad.AnnData(values.astype(np.float32), obs=obs, var=pd.DataFrame(index=[f"Cells_F{i}" for i in range(4)]))
    stamp(adata, resolution="well")
    return adata, control


def test_regress_out_on_the_controls_removes_density_and_keeps_the_phenotype():
    adata, control = _treated_and_thinned()
    before = np.asarray(adata.X).copy()
    on_controls = mt.pp.regress_out(adata, reference="negcon", copy=True)
    on_everything = mt.pp.regress_out(adata, copy=True)

    fixed = np.asarray(on_controls.X)
    count = adata.obs["Metadata_CellCount"].to_numpy()
    # The technical slope is gone among the controls, and they stay where normalization put them.
    assert abs(np.corrcoef(count[control], fixed[control, 0])[0, 1]) < 0.2
    assert np.allclose(fixed[control].mean(axis=0), before[control].mean(axis=0), atol=1e-4)
    # The treatment keeps most of its phenotype; a fit over every well takes it away.
    assert fixed[~control].mean() - fixed[control].mean() > 2.0
    assert np.asarray(on_everything.X)[~control].mean() - np.asarray(on_everything.X)[control].mean() < 1.0


def test_regress_out_on_the_controls_refuses_a_categorical_covariate(gradient_cells):
    wells = mt.tl.aggregate(gradient_cells, min_cells=0)
    with pytest.raises(ValueError, match="numeric covariates only"):
        mt.pp.regress_out(wells, keys=("Metadata_Plate",), by=None, reference="negcon")


def _operator_wells(missing_label: bool):
    """24 wells whose only structure is a per-operator offset of 0, 5 or 10.

    Args:
        missing_label: Leave row 0's operator label missing, as an unmatched platemap row does.

    Returns:
        ``(adata, operator)``, the object and the true operator of every row.
    """
    import anndata as ad

    from mantispy._core.schema import stamp

    operator = np.array(["carol", "alice", "bob"] * 8)
    offset = {"alice": 0.0, "bob": 5.0, "carol": 10.0}
    generator = np.random.default_rng(0)
    values = np.array([offset[name] for name in operator])[:, None] + generator.normal(0.0, 0.1, (24, 3))

    labels = operator.astype(object).copy()
    if missing_label:
        labels[0] = None
    obs = pd.DataFrame(
        {
            "Metadata_Plate": "P1",
            "Metadata_Well": [f"A{index + 1:02d}" for index in range(24)],
            "Metadata_Operator": pd.Categorical(labels),
        },
        index=[str(index) for index in range(24)],
    )
    adata = ad.AnnData(
        X=values.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"Cells_AreaShape_F{index}" for index in range(3)]),
    )
    stamp(adata, resolution="well")
    return adata, operator


def test_regress_out_refuses_a_covariate_with_a_missing_label():
    """A NaN category encodes all-zero, which is the level drop_first removed, so a row
    whose label is missing is fitted as the reference level: row 0 came out at 13.85
    instead of 5.06, and the 23 correctly labelled rows moved with it (per-operator means
    6.42/6.42/6.42 became 5.17/6.42/7.39)."""
    adata, _ = _operator_wells(missing_label=True)
    with pytest.raises(ValueError, match="Metadata_Operator"):
        mt.pp.regress_out(adata, keys=["Metadata_Operator"], by=None)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_regress_out_says_so_when_a_missing_covariate_value_disables_it(value):
    """``np.ptp`` is NaN for a covariate holding a NaN and ``NaN > 0`` is False, so the
    covariate is read as non-varying and dropped: the correction becomes a bitwise no-op
    (corr 0.98 before and after) while the only log line claims the covariate was removed.
    An infinite value is dropped the same way, where it was fitted without a warning (#66)."""
    adata, _ = _operator_wells(missing_label=False)
    generator = np.random.default_rng(1)
    counts = generator.normal(1500.0, 200.0, adata.n_obs)
    values = np.asarray(adata.X, dtype=np.float64).copy()
    values[:, 0] = 0.05 * counts + generator.normal(0.0, 1.0, adata.n_obs)
    adata.X = values.astype(np.float32)
    counts[3] = value
    adata.obs["Metadata_CellCount"] = counts

    before = np.asarray(adata.X).copy()
    with pytest.warns(UserWarning, match="Metadata_CellCount"):
        mt.pp.regress_out(adata, keys=["Metadata_CellCount"], by=None)
    # Leaving the group uncorrected is the conservative choice; doing it silently is not.
    assert np.array_equal(np.asarray(adata.X), before)
