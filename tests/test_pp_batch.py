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

    mt.pp.correct_plate_position(wells, method="median_polish")
    after = _position_correlation(wells)
    assert after[0] < 0.15 and after[1] < 0.15


def test_b_score_removes_gradient_and_standardizes(gradient_cells):
    wells = mt.tl.aggregate(gradient_cells, min_cells=0)
    before = _position_correlation(wells)
    assert before[0] > 0.5

    mt.pp.correct_plate_position(wells)  # default method="b_score"
    after = _position_correlation(wells)
    assert after[0] < 0.15 and after[1] < 0.15
    # the B-score is a robust z-score, so each feature's per-plate MAD is about one.
    values = np.asarray(wells.X, dtype=float)
    mad = np.median(np.abs(values - np.median(values, axis=0)), axis=0)
    assert np.allclose(np.median(mad), 1.0, atol=0.3)


def test_b_score_recovers_hits_a_gradient_masks():
    import anndata as ad

    from mantispy._core.plate import PLATE_FORMATS, well_name
    from mantispy._core.schema import stamp

    n_rows, n_cols = PLATE_FORMATS[96]
    names = [well_name(r, c) for r in range(n_rows) for c in range(n_cols)]
    rows = np.array([well_row(w) for w in names], dtype=float)
    cols = np.array([well_col(w) for w in names], dtype=float)
    rng = np.random.default_rng(0)
    gradient = 6.0 * rows / rows.max() + 4.0 * cols / cols.max()
    value = gradient + rng.normal(0.0, 0.05, len(names))
    # Plant the hits in the low-gradient wells, where the gradient hides them in the raw signal.
    hits = np.argsort(gradient)[:6]
    value[hits] += 3.0  # real signal, smaller than the full gradient span

    adata = ad.AnnData(
        X=value.reshape(-1, 1).astype(np.float32),
        obs=pd.DataFrame(
            {
                "Metadata_Plate": "P1",
                "Metadata_Well": names,
                "Metadata_Control": True,
                "Metadata_Control_Type": "negcon",
            },
            index=[str(i) for i in range(len(names))],
        ),
        var=pd.DataFrame(index=["Cells_Intensity_F0"]),
    )
    stamp(adata, resolution="well")

    def precision_at_k(values, k=6):
        top = set(np.argsort(values)[::-1][:k])
        return len(top & set(hits.tolist())) / k

    raw = precision_at_k(value)
    corrected = mt.pp.correct_plate_position(adata, copy=True)
    b = precision_at_k(np.asarray(corrected.X, dtype=float)[:, 0])
    # the gradient hides hits in the raw signal; the B-score recovers them.
    assert raw < 0.5 and b == 1.0


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
    assert between_plate_spread(wells) < 0.01 * before
    assert np.isfinite(wells.X).all()

    with pytest.raises(KeyError, match="Metadata_Nope"):
        mt.pp.regress_out(wells, keys=("Metadata_Nope",))


def test_harmony_says_so_when_it_corrects_nothing(cells):
    """harmonypy reports convergence and returns the input unchanged when the batches are perfectly separated in the embedding, because every soft cluster is then single-batch.

    The wrapper warns in that case.
    """
    pytest.importorskip("harmonypy")
    import scanpy as sc

    wells = mt.tl.aggregate(cells, min_cells=0)
    wells.obs["Metadata_Site"] = np.where(wells.obs["Metadata_Plate"].astype(str) == "Plate01", "north", "south")
    values = np.asarray(wells.X, dtype=float).copy()
    values[(wells.obs["Metadata_Site"] == "north").to_numpy()] += 5.0
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

    ``np.linalg.lstsq`` with several right-hand sides returns NaN coefficients for all of them when any one column holds an infinity, which would turn the whole plate into NaN.
    Three JUMP plates carry one inf each.
    Regression test for #66: the feature holding it was then fitted with the inf and lost on its plate.
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


def _control_plate(n_wells, n_features, gradient, seed, noise=0.0, plate="P1"):
    """A plate of only control wells, carrying either a smooth row/column gradient or iid noise.

    Args:
        n_wells: Number of control wells, laid out on the standard grid.
        n_features: Number of features.
        gradient: Plant a smooth row/column gradient shared across features; otherwise iid noise.
        noise: Standard deviation of the per-well noise added on top of a gradient.
        seed: Seed for reproducibility.
        plate: Value for the ``Metadata_Plate`` column, so several plates can be concatenated.
    """
    import anndata as ad

    from mantispy._core.plate import PLATE_FORMATS, well_name
    from mantispy._core.schema import stamp

    n_rows, n_cols = PLATE_FORMATS[next(size for size in sorted(PLATE_FORMATS) if size >= n_wells)]
    wells = [well_name(row, col) for row in range(n_rows) for col in range(n_cols)][:n_wells]
    rows = np.array([well_row(well) for well in wells], dtype=float)
    cols = np.array([well_col(well) for well in wells], dtype=float)

    generator = np.random.default_rng(seed)
    if gradient:
        # A plate that fits in one row or one column has no span to divide by.
        smooth = (3.0 * rows / max(rows.max(), 1.0) + 2.0 * cols / max(cols.max(), 1.0))[:, None]
        values = smooth + generator.normal(0.0, noise, (n_wells, n_features))
    else:
        values = generator.normal(0.0, 1.0, (n_wells, n_features))

    obs = pd.DataFrame(
        {"Metadata_Plate": plate, "Metadata_Well": wells, "Metadata_Control": True, "Metadata_Control_Type": "negcon"},
        index=[str(index) for index in range(n_wells)],
    )
    adata = ad.AnnData(
        X=values.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"Cells_AreaShape_F{index}" for index in range(n_features)]),
    )
    stamp(adata, resolution="well")
    return adata


def test_detect_plate_position_fires_on_a_smooth_gradient():
    adata = _control_plate(n_wells=96, n_features=12, gradient=True, noise=0.1, seed=0)
    mt.pp.detect_plate_position(adata)
    row = adata.uns["mantispy"]["plate_position_detection"].set_index("plate").loc["P1"]
    assert row["cv_r2_median"] > 0
    assert row["frac_features_positive"] > 0.8
    assert row["reason"] == ""


def test_detect_plate_position_does_not_fire_on_noise():
    adata = _control_plate(n_wells=96, n_features=12, gradient=False, seed=0)
    mt.pp.detect_plate_position(adata)
    row = adata.uns["mantispy"]["plate_position_detection"].set_index("plate").loc["P1"]
    assert row["cv_r2_median"] <= 0
    assert row["frac_features_positive"] < 0.5
    assert row["reason"] == ""


def test_detect_plate_position_scores_each_plate_independently():
    import anndata as ad

    gradient = _control_plate(n_wells=96, n_features=12, gradient=True, noise=0.1, seed=0, plate="P1")
    noise = _control_plate(n_wells=96, n_features=12, gradient=False, seed=1, plate="P2")
    adata = ad.concat([gradient, noise], index_unique="-")

    mt.pp.detect_plate_position(adata)
    table = adata.uns["mantispy"]["plate_position_detection"].set_index("plate")

    assert set(table.index) == {"P1", "P2"}
    assert table.loc["P1", "cv_r2_median"] > 0
    assert table.loc["P1", "frac_features_positive"] > 0.8
    assert table.loc["P2", "cv_r2_median"] <= 0
    assert (table["reason"] == "").all()


def test_detect_plate_position_records_too_few_controls_without_raising():
    adata = _control_plate(n_wells=12, n_features=12, gradient=True, noise=0.1, seed=0)
    mt.pp.detect_plate_position(adata, min_controls=20)
    row = adata.uns["mantispy"]["plate_position_detection"].set_index("plate").loc["P1"]
    assert np.isnan(row["cv_r2_median"])
    assert np.isnan(row["frac_features_positive"])
    assert "control wells" in row["reason"]
    assert row["n_controls"] == 12


def test_detect_pass_coincides_with_the_correction_flattening_the_grid():
    adata = _control_plate(n_wells=96, n_features=12, gradient=True, noise=0.1, seed=0)
    mt.pp.detect_plate_position(adata)
    assert adata.uns["mantispy"]["plate_position_detection"].set_index("plate").loc["P1", "cv_r2_median"] > 0

    before = _position_correlation(adata)
    mt.pp.correct_plate_position(adata)
    after = _position_correlation(adata)
    assert after[0] < before[0] and after[1] < before[1]


def test_correct_plate_position_plates_subset_leaves_others_untouched():
    import anndata as ad

    first = _control_plate(n_wells=96, n_features=12, gradient=True, noise=0.1, seed=0, plate="P1")
    second = _control_plate(n_wells=96, n_features=12, gradient=True, noise=0.1, seed=1, plate="P2")
    adata = ad.concat([first, second], index_unique="-")
    before = adata.X.copy()
    is_second = adata.obs["Metadata_Plate"].to_numpy() == "P2"

    mt.pp.correct_plate_position(adata, plates=["P1"])

    assert set(adata.uns["mantispy"]["plate_position"]) == {"P1"}
    np.testing.assert_array_equal(adata.X[is_second], before[is_second])
    assert not np.array_equal(adata.X[~is_second], before[~is_second])


def test_correct_plate_position_rejects_unknown_plates():
    adata = _control_plate(n_wells=96, n_features=12, gradient=True, noise=0.1, seed=0)
    with pytest.raises(ValueError, match="plates not found"):
        mt.pp.correct_plate_position(adata, plates=["P1", "nope"])


def _plate(x, names, control, var_names):
    import anndata as ad

    from mantispy._core.schema import stamp

    adata = ad.AnnData(
        X=np.asarray(x, dtype=np.float32),
        obs=pd.DataFrame(
            {
                "Metadata_Plate": "P",
                "Metadata_Well": names,
                "Metadata_Control": control,
                "Metadata_Control_Type": np.where(np.asarray(control, dtype=bool), "negcon", "treatment"),
            },
            index=[str(i) for i in range(len(names))],
        ),
        var=pd.DataFrame(index=var_names),
    )
    stamp(adata, resolution="well")
    return adata


def test_detect_cv_r2_is_offset_invariant():
    from mantispy._core.plate import PLATE_FORMATS, well_name

    n_rows, n_cols = PLATE_FORMATS[96]
    names, control = [], []
    for r in range(n_rows):
        for c in range(n_cols):
            names.append(well_name(r, c))
            control.append(c in (0, n_cols - 1))  # controls in two edge columns only
    rows = np.array([well_row(w) for w in names], dtype=float)
    cols = np.array([well_col(w) for w in names], dtype=float)
    rng = np.random.default_rng(0)
    base = (3.0 * rows / rows.max() + 2.0 * cols / cols.max())[:, None] + rng.normal(0.0, 0.1, (len(names), 8))
    var = [f"Cells_AreaShape_F{j}" for j in range(8)]

    def score(matrix):
        adata = _plate(matrix, names, control, var)
        mt.pp.detect_plate_position(adata, reference="negcon", min_controls=8, n_splits=8)
        return adata.uns["mantispy"]["plate_position_detection"].set_index("plate").loc["P", "cv_r2_median"]

    # Adding a large grand level must not change an additive model's cross-validated R^2 (the bug sent it
    # to ~-1e9); the tiny residual that remains is only float32 round-off from storing base + 1e5.
    base_score, shifted = score(base), score(base + 1e5)
    assert base_score > 0.5 and shifted > 0.5
    assert abs(base_score - shifted) < 1e-2


def test_b_score_keeps_treatment_wells_when_a_feature_is_blank_on_controls():
    from mantispy._core.plate import PLATE_FORMATS, well_name

    n_rows, n_cols = PLATE_FORMATS[96]
    names = [well_name(r, c) for r in range(n_rows) for c in range(n_cols)]
    control = np.array([i % 2 == 0 for i in range(len(names))])
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, (len(names), 2))
    x[control, 1] = np.nan  # feature 1 is blank on the controls but measured on the treated wells
    adata = _plate(x, names, control.tolist(), ["Cells_AreaShape_F0", "Cells_AreaShape_F1"])

    mt.pp.correct_plate_position(adata, method="b_score", reference="negcon")
    assert np.isfinite(np.asarray(adata.X)[~control, 1]).all()


def test_detect_records_a_reason_when_no_feature_is_scorable():
    from mantispy._core.plate import PLATE_FORMATS, well_name

    n_rows, n_cols = PLATE_FORMATS[96]
    names = [well_name(r, c) for r in range(n_rows) for c in range(n_cols)]
    adata = _plate(np.ones((len(names), 4)), names, [True] * len(names), [f"Cells_AreaShape_F{j}" for j in range(4)])

    mt.pp.detect_plate_position(adata, reference="negcon", min_controls=8)
    row = adata.uns["mantispy"]["plate_position_detection"].set_index("plate").loc["P"]
    assert np.isnan(row["cv_r2_median"]) and row["reason"] != ""


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
    """On pki two plates hold a third of the others' cells; fitted per plate and re-expressed at the pooled mean, the result correlated with the cell count more than the input had."""
    adata = _two_plates(count_means=(600.0, 1800.0), slope=0.002)
    with pytest.warns(UserWarning, match="extrapolating"):
        mt.pp.regress_out(adata, keys=["Metadata_Count"], by="Metadata_Plate")


def _treated_and_thinned(seed=0):
    """Controls whose features follow density for technical reasons, and a treatment that both thins the wells and has a phenotype of its own, so a fit over every well mistakes the phenotype for density."""
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
            "Metadata_Control_Type": np.where(control, "negcon", "treatment"),
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
    assert abs(np.corrcoef(count[control], fixed[control, 0])[0, 1]) < 0.2
    assert np.allclose(fixed[control].mean(axis=0), before[control].mean(axis=0), atol=1e-4)
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
    """A NaN category encodes all-zero, which is the level drop_first removed, so a row whose label is missing is fitted as the reference level: row 0 came out at 13.85 instead of 5.06, and the 23 correctly labelled rows moved with it (per-operator means 6.42/6.42/6.42 became 5.17/6.42/7.39)."""
    adata, _ = _operator_wells(missing_label=True)
    with pytest.raises(ValueError, match="Metadata_Operator"):
        mt.pp.regress_out(adata, keys=["Metadata_Operator"], by=None)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_regress_out_says_so_when_a_missing_covariate_value_disables_it(value):
    """``np.ptp`` is NaN for a covariate holding a NaN and ``NaN > 0`` is False, so the covariate is read as non-varying and dropped: the correction becomes a bitwise no-op (corr 0.98 before and after) while the only log line claims the covariate was removed.

    An infinite value is dropped the same way, where it was fitted without a warning (#66).
    """
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
    assert np.array_equal(np.asarray(adata.X), before)


def _crispr_arm_plate(arm_background, genes, arms, unexpressed_bio, treated_bio, n_features=3):
    """One well per gene: X = arm background + a per-gene offset, so a known background sits on each arm."""
    import anndata as ad

    from mantispy._core.schema import stamp

    rows = []
    for gene, arm in zip(genes, arms, strict=True):
        base = np.array(arm_background.get(arm, [0.0] * n_features), dtype=float)
        offset = unexpressed_bio.get(gene, treated_bio.get(gene, 0.0))
        rows.append(base + offset)
    adata = ad.AnnData(
        X=np.array(rows, dtype=np.float32),
        obs=pd.DataFrame(
            {"Metadata_Gene": genes, "Metadata_ChromosomeArm": arms},
            index=[f"w{i}" for i in range(len(genes))],
        ),
        var=pd.DataFrame(index=[f"Cells_Intensity_F{i}" for i in range(n_features)]),
    )
    stamp(adata, resolution="well")
    return adata


def test_correct_chromosome_arm_matches_the_hand_computed_background():
    # 1q has 22 unexpressed genes (> min_genes), so its background is estimated and removed; 1p has too few.
    unexp_q = [f"q{i}" for i in range(22)]
    trt_q = ["QT1", "QT2"]
    genes = unexp_q + trt_q + ["p0", "p1", "PT"]
    arms = ["1q"] * (len(unexp_q) + len(trt_q)) + ["1p"] * 3
    # Give the unexpressed wells spread so the background is a real mean, not a constant.
    unexpressed_bio = {g: float(i % 3) for i, g in enumerate(unexp_q)} | {"p0": 0.0, "p1": 0.0}
    treated_bio = {"QT1": 7.0, "QT2": -4.0, "PT": 2.0}
    arm_background = {"1q": [10.0, 10.0, 10.0], "1p": [5.0, 5.0, 5.0]}
    adata = _crispr_arm_plate(arm_background, genes, arms, unexpressed_bio, treated_bio)

    raw = adata.X.copy()
    on_q = adata.obs["Metadata_ChromosomeArm"].to_numpy() == "1q"
    is_unexp = adata.obs["Metadata_Gene"].isin(unexp_q).to_numpy()
    expected_background = raw[on_q & is_unexp].mean(axis=0)

    mt.pp.correct_chromosome_arm(adata, unexpressed=set(unexp_q))

    np.testing.assert_allclose(adata.X[on_q], raw[on_q] - expected_background, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(adata.X[~on_q], raw[~on_q])  # 1p has too few unexpressed genes
    assert adata.uns["mantispy"]["chromosome_arm"] == {"1q": 22}


def test_correct_chromosome_arm_leaves_unmapped_wells_untouched():
    genes = [f"q{i}" for i in range(22)] + ["NOARM1", "NOARM2"]
    arms = ["1q"] * 22 + [None, None]
    adata = _crispr_arm_plate({"1q": [3.0, 3.0]}, genes, arms, dict.fromkeys(genes[:22], 0.0), {}, n_features=2)
    raw = adata.X.copy()
    mt.pp.correct_chromosome_arm(adata, unexpressed=set(genes[:22]))
    unmapped = adata.obs["Metadata_ChromosomeArm"].isna().to_numpy()
    np.testing.assert_array_equal(adata.X[unmapped], raw[unmapped])


def test_correct_chromosome_arm_requires_its_columns():
    adata = _crispr_arm_plate({"1q": [1.0]}, ["q0"], ["1q"], {"q0": 0.0}, {}, n_features=1)
    del adata.obs["Metadata_Gene"]
    with pytest.raises(KeyError, match="Metadata_Gene"):
        mt.pp.correct_chromosome_arm(adata, unexpressed=set())


def test_correct_chromosome_arm_reads_unexpressed_from_depmap():
    """With no unexpressed set it reads the DepMap matrix for the given model id."""
    unexp_q = [f"q{i}" for i in range(22)]
    genes = unexp_q + ["QT1", "p0"]
    arms = ["1q"] * 23 + ["1p"]
    adata = _crispr_arm_plate(
        {"1q": [10.0, 10.0], "1p": [5.0, 5.0]}, genes, arms, dict.fromkeys(unexp_q, 0.0), {"QT1": 7.0}, n_features=2
    )

    # DepMap-shaped matrix: the 22 q-genes are zero-TPM in this model, QT1 is expressed.
    expression = pd.DataFrame({**{f"{g} (1)": [0.0] for g in unexp_q}, "QT1 (2)": [7.3]}, index=["ACH-000001"])

    raw = adata.X.copy()
    mt.pp.correct_chromosome_arm(adata, expression=expression, cell_line="ACH-000001")
    on_q = adata.obs["Metadata_ChromosomeArm"].to_numpy() == "1q"
    assert adata.uns["mantispy"]["chromosome_arm"] == {"1q": 22}
    np.testing.assert_array_equal(adata.X[~on_q], raw[~on_q])


def test_correct_chromosome_arm_needs_a_reference():
    """Without a ready-made set, a DepMap matrix and model id are required; nothing is hard-coded."""
    adata = _crispr_arm_plate({"1q": [1.0]}, ["q0"], ["1q"], {"q0": 0.0}, {}, n_features=1)
    with pytest.raises(ValueError, match="DepMap"):
        mt.pp.correct_chromosome_arm(adata)


def test_correct_chromosome_arm_rejects_a_bare_string():
    """A bare string as `unexpressed` would silently match its characters; isin raises instead."""
    adata = _crispr_arm_plate({"1q": [1.0]}, ["q0"], ["1q"], {"q0": 0.0}, {}, n_features=1)
    with pytest.raises(TypeError):
        mt.pp.correct_chromosome_arm(adata, unexpressed="q0")
