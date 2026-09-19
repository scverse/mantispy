import warnings

import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy._core.plate import well_col, well_row
from mantispy.ds import synthetic_plate
from mantispy.pp._batch import _median_polish_stack


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


def test_polish_works_at_cell_resolution(gradient_cells):
    """Cells share grid positions, and assigning them into the grid directly keeps only the last."""
    before = _position_correlation(gradient_cells)
    assert before[0] > 0.5

    mt.pp.correct_plate_position(gradient_cells)
    after = _position_correlation(gradient_cells)
    assert after[0] < 0.15
    assert gradient_cells.X.dtype == np.float32


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


def test_regress_out_leaves_missing_values_missing():
    """Filling NaN to make the fit run and writing the fitted value back would turn
    'not measured' into a number indistinguishable from a real one."""
    cells = synthetic_plate(n_wells=48, n_cells=10, n_features=8, seed=0)
    wells = mt.tl.aggregate(cells, min_cells=0)
    values = wells.X.copy()
    values[0, 0] = np.nan
    wells.X = values

    mt.pp.regress_out(wells, keys=("Metadata_CellCount",), by=None)
    assert np.isnan(wells.X[0, 0])
    assert int(np.isnan(wells.X).sum()) == 1


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


def test_regress_out_refuses_a_collinear_design(gradient_cells):
    """A confounder constant within a group makes the design rank-deficient; lstsq would
    still return a minimum-norm solution that rescales or wipes the group."""
    wells = mt.tl.aggregate(gradient_cells, min_cells=0)
    wells.X = np.full((wells.n_obs, wells.n_vars), 10.0, dtype=np.float32)
    # Metadata_CellCount is constant within each plate here, so the design is collinear.
    mt.pp.regress_out(wells, keys=("Metadata_CellCount",), by="Metadata_Plate")
    np.testing.assert_allclose(wells.X, 10.0, rtol=1e-5)


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


def _reference_polish(matrix, max_iter=10, tol=1e-4):
    """Tukey median polish of one grid, written as the textbook loop to compare against.

    The shipped implementation polishes every feature in one kernel because per-slice
    dispatch dominates at plate sizes.
    """
    residual = np.array(matrix, dtype=np.float64, copy=True)
    row_effects = np.zeros(residual.shape[0])
    column_effects = np.zeros(residual.shape[1])
    previous = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an entirely unmeasured row
        for _ in range(max_iter):
            medians = np.nan_to_num(np.nanmedian(residual, axis=1), nan=0.0)
            residual -= medians[:, None]
            row_effects += medians
            row_effects -= np.median(row_effects)

            medians = np.nan_to_num(np.nanmedian(residual, axis=0), nan=0.0)
            residual -= medians[None, :]
            column_effects += medians
            column_effects -= np.median(column_effects)

            current = np.nansum(np.abs(residual))
            if abs(previous - current) < tol:
                break
            previous = current
    return row_effects, column_effects


@pytest.mark.parametrize("gaps", ["none", "one well", "a whole row", "sparse"])
def test_the_polish_kernel_matches_a_plain_median_polish(gaps):
    """Bit-for-bit, including the plate layouts where a median has nothing to work with."""
    rng = np.random.default_rng(0)
    grids = rng.normal(size=(8, 12, 40))
    grids += rng.normal(size=(8, 1, 40)) * 3  # a row gradient to remove
    grids += rng.normal(size=(1, 12, 40)) * 2
    if gaps == "one well":
        grids[2, 5, :] = np.nan
    elif gaps == "a whole row":
        grids[4, :, :] = np.nan
    elif gaps == "sparse":
        grids[rng.random(grids.shape) < 0.2] = np.nan

    rows, columns = _median_polish_stack(grids, max_iter=10, tol=1e-4)
    for feature in range(grids.shape[2]):
        expected_rows, expected_columns = _reference_polish(grids[:, :, feature])
        np.testing.assert_array_equal(rows[:, feature], expected_rows)
        np.testing.assert_array_equal(columns[:, feature], expected_columns)


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

    mt.pp.regress_out(adata, keys=("Metadata_CellCount",), key_added="regressed")
    corrected = adata.layers["regressed"]

    # The fit is per plate, so only the plate holding the infinity can be affected.
    plates = adata.obs["Metadata_Plate"].to_numpy()
    spoiled = plates == plates[0]
    others = np.delete(np.arange(adata.n_vars), 2)

    gapped = adata.copy()
    gapped.X[0, 2] = np.nan
    mt.pp.regress_out(gapped, keys=("Metadata_CellCount",), key_added="regressed")
    assert np.isinf(corrected[0, 2]), "the infinity stays where it was"
    # and its feature is fitted without it
    np.testing.assert_array_equal(corrected[1:, 2], gapped.layers["regressed"][1:, 2])
    assert np.isfinite(corrected[np.ix_(spoiled, others)]).all(), "and takes no neighbour with it"
    assert np.isfinite(corrected[~spoiled]).all(), "nor any other plate"


def test_one_nan_in_a_regression_key_does_not_spread_across_groups():
    """A missing covariate value in one group does not affect the other groups.

    A plain ``.mean()`` over the pooled design makes the anchor NaN for the whole column
    when any group's covariate has a missing value. The group holding the NaN drops the
    column and fits intercept-only, but every other group anchored at that NaN mean would
    turn entirely NaN. Here one row of plate 1's covariate is NaN, and plates 2 and 3 must
    stay finite.
    """
    adata = synthetic_plate(n_plates=3, n_wells=48, n_cells=2, n_features=8, seed=0)
    adata = mt.tl.aggregate(adata, min_cells=0)
    plate = adata.obs["Metadata_Plate"].astype(str).to_numpy()
    plates_sorted = sorted(set(plate))
    generator = np.random.default_rng(0)
    adata.obs["Metadata_Count"] = generator.normal(1500.0, 120, adata.n_obs)
    first_row = adata.obs.index[np.flatnonzero(plate == plates_sorted[0])[0]]
    adata.obs.loc[first_row, "Metadata_Count"] = np.nan

    mt.pp.regress_out(adata, keys=["Metadata_Count"], by="Metadata_Plate", key_added="regressed")
    corrected = np.asarray(adata.layers["regressed"])

    for p in plates_sorted[1:]:
        assert np.isfinite(corrected[plate == p]).all(), f"{p} was destroyed by a NaN it never carried"
    # The plate holding the NaN is allowed to lose the covariate's effect but must stay
    # finite everywhere the underlying data is finite.
    holder = plate == plates_sorted[0]
    assert np.isfinite(corrected[holder]).all(), "the plate that carries the NaN must still be finite itself"


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


def _plate_gap(adata):
    plate = adata.obs["Metadata_Plate"].astype(str).to_numpy()
    first, second = sorted(set(plate))
    values = np.asarray(adata.X)
    return float(np.abs(values[plate == first].mean(0) - values[plate == second].mean(0)).mean())


def test_regress_out_actually_runs_on_a_categorical_key():
    """pd.get_dummies on a Categorical emits one column per declared level, so a level
    absent from a plate gives an all-zero column that the rank guard reads as collinearity,
    and the plate is skipped. obs is categorical after tl.aggregate and any h5ad round trip,
    so this is the common case."""
    adata = _two_plates()
    generator = np.random.default_rng(0)
    plate = adata.obs["Metadata_Plate"].astype(str).to_numpy()
    # Operator nested in plate: six declared levels, two observed per plate.
    operator = [f"op{2 * sorted(set(plate)).index(p) + generator.integers(2)}" for p in plate]
    adata.obs["Metadata_Op"] = pd.Categorical(operator, categories=[f"op{i}" for i in range(6)])

    before = np.asarray(adata.X).copy()
    mt.pp.regress_out(adata, keys=["Metadata_Op"], by="Metadata_Plate")
    assert np.nanmax(np.abs(np.asarray(adata.X) - before)) > 1e-3


def test_regress_out_leaves_identical_plates_identical():
    """Adding back the raw intercept re-expresses every feature at covariate=0 and turns
    each plate's slope noise into a constant offset between plates that started identical."""
    adata = _two_plates()
    before = _plate_gap(adata)
    mt.pp.regress_out(adata, keys=["Metadata_Count"], by="Metadata_Plate")
    assert _plate_gap(adata) < 1.2 * before


def test_regress_out_removes_a_real_confluency_difference():
    adata = _two_plates(count_means=(1200.0, 1800.0), slope=0.002)
    before = _plate_gap(adata)
    mt.pp.regress_out(adata, keys=["Metadata_Count"], by="Metadata_Plate")
    assert _plate_gap(adata) < 0.4 * before


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


def test_regress_out_levels_a_fully_labelled_covariate():
    """The guard above must refuse the missing label without disabling the correction itself."""
    adata, operator = _operator_wells(missing_label=False)
    mt.pp.regress_out(adata, keys=["Metadata_Operator"], by=None)
    means = [np.asarray(adata.X)[operator == name, 0].mean() for name in ("alice", "bob", "carol")]
    assert max(means) - min(means) < 0.05


def test_regress_out_says_so_when_a_missing_covariate_value_disables_it():
    """``np.ptp`` is NaN for a covariate holding a NaN and ``NaN > 0`` is False, so the
    covariate is read as non-varying and dropped: the correction becomes a bitwise no-op
    (corr 0.98 before and after) while the only log line claims the covariate was removed."""
    adata, _ = _operator_wells(missing_label=False)
    generator = np.random.default_rng(1)
    counts = generator.normal(1500.0, 200.0, adata.n_obs)
    values = np.asarray(adata.X, dtype=np.float64).copy()
    values[:, 0] = 0.05 * counts + generator.normal(0.0, 1.0, adata.n_obs)
    adata.X = values.astype(np.float32)
    counts[3] = np.nan
    adata.obs["Metadata_CellCount"] = counts

    before = np.asarray(adata.X).copy()
    with pytest.warns(UserWarning, match="Metadata_CellCount"):
        mt.pp.regress_out(adata, keys=["Metadata_CellCount"], by=None)
    # Leaving the group uncorrected is the conservative choice; doing it silently is not.
    assert np.array_equal(np.asarray(adata.X), before)
