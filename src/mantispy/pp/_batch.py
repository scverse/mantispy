"""Plate-position, confounder and batch corrections."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._numba import MEAN, MEDIAN, _polish_planes, grouped_stat
from mantispy._core._reduce import get_matrix, group_codes
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy
from mantispy._core.plate import well_col, well_row

METHODS = ("median_polish",)


def _median_polish_stack(grids: np.ndarray, max_iter: int, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Median polish every feature of a ``(rows, columns, features)`` stack.

    Returns the fitted ``(row_effects, column_effects)``, both ``(positions, features)``.
    The grand level is not included, so subtracting the effects keeps each feature's level, as :func:`regress_out` does.
    """
    planes = np.ascontiguousarray(np.moveaxis(np.asarray(grids, dtype=np.float64), 2, 0))
    rows, columns = _polish_planes(planes, int(max_iter), float(tol))
    return rows.T, columns.T


@inplace_or_copy()
def correct_plate_position(
    adata: AnnData,
    method: str = "median_polish",
    by: str = "Metadata_Plate",
    reference: str | None = None,
    max_iter: int = 10,
    tol: float = 1e-4,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Remove row and column position effects, per plate and per feature.

    Args:
        adata: Object to correct, at cell or well resolution.
        method: Only ``"median_polish"`` (Tukey), which is robust to a few extreme wells.
        by: Column identifying the plate.
        reference: Fit the row and column effects on these rows only.
            ``"negcon"`` is the usual choice, so that treatments laid out in particular columns are not absorbed into a column effect.
            ``None`` fits on every well.
        max_iter: Maximum number of median-polish iterations.
        tol: Convergence tolerance of the median polish.
        key_added: Write to ``layers[key_added]`` instead of overwriting ``X``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``X`` or ``layers[key_added]``, and the fitted effects per plate to ``uns["mantispy"]["plate_position"]``.

    Raises:
        ValueError: If ``method`` is unknown, or a plate holds no reference rows.

    Notes:
        The polish is fitted on the well grid.
        At cell resolution each well is first reduced to its median, and the fitted effect is then subtracted from every cell of that well.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    X = get_matrix(adata)
    wells = adata.obs["Metadata_Well"].to_numpy()
    rows = np.array([well_row(well) for well in wells])
    columns = np.array([well_col(well) for well in wells])

    fit_mask = reference_mask(adata, reference)
    codes, keys = group_codes(adata, by)
    # Promote one plate at a time: float64 copies of the input and output would take four times the matrix in memory.
    out = np.array(X, dtype=np.float32)
    effects: dict[str, dict[str, list]] = {}

    for group, key in enumerate(keys):
        selected = np.flatnonzero(codes == group)
        fit_rows = selected[fit_mask[selected]]
        if fit_rows.size == 0:
            raise ValueError(f"no reference rows in group {key!r}")

        # Each plate gets its own extent; a shared or padded grid would fit the effects on empty wells.
        n_rows, n_columns = rows[selected].max() + 1, columns[selected].max() + 1

        # One value per well, so several cells in a well cannot overwrite each other.
        well_index = rows[fit_rows] * n_columns + columns[fit_rows]
        occupied, codes_in_plate = np.unique(well_index, return_inverse=True)

        grid = np.full((n_rows * n_columns, adata.n_vars), np.nan)
        grid[occupied] = grouped_stat(X[fit_rows], codes_in_plate, occupied.size, MEDIAN)

        row_effect, column_effect = _median_polish_stack(grid.reshape(n_rows, n_columns, adata.n_vars), max_iter, tol)
        adjustment = row_effect[rows[selected]] + column_effect[columns[selected]]
        out[selected] = (X[selected].astype(np.float64) - adjustment).astype(np.float32)
        effects[str(key)] = {"row": row_effect.T.tolist(), "col": column_effect.T.tolist()}

    if key_added is None:
        adata.X = out
    else:
        adata.layers[key_added] = out
    adata.uns.setdefault("mantispy", {})["plate_position"] = effects
    return None


def _smooth_grid(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth a ``(rows, columns, features)`` grid over its two spatial axes only.

    Pads with zeros so a position with no data borrows only from its neighbours.
    """
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(grid, sigma=(sigma, sigma, 0.0), mode="constant", cval=0.0)


@inplace_or_copy()
def detect_plate_position(
    adata: AnnData,
    by: str = "Metadata_Plate",
    reference: str | None = "negcon",
    sigma: float = 1.5,
    n_splits: int = 5,
    min_controls: int = 20,
    seed: int = 0,
    key_added: str = "plate_position_detection",
    copy: bool = False,
) -> AnnData | None:
    """Test whether a plate carries a position artifact that generalizes, to gate the correction on it.

    A per-plate position effect is often overfit noise rather than a real artifact: with few controls
    on a large grid, the apparent structure is the controls' own scatter, and :func:`correct_plate_position`
    would then subtract signal. This detector fits a smooth position map on the control wells under
    cross-validation and scores how well it predicts held-out controls, so a plate is corrected only when
    the position structure is learnable.

    Per plate, on the control wells selected by ``reference``, each well is reduced to one value per grid
    position. Across cross-validation folds a position map is fitted on the training controls as their
    per-position mean, Gaussian-smoothed over the plate grid: the summed values and the well counts are
    smoothed separately and divided, so an empty position borrows from its neighbours, and a position no
    neighbour reaches takes the training grand mean. Each held-out control is predicted from its own
    position's cell of the map. A cross-validated coefficient of determination is then formed per feature
    from the held-out residuals against the plate's control grand mean, and summarized per plate.

    Args:
        adata: Object to inspect, at cell or well resolution.
        by: Column identifying the plate.
        reference: Wells to fit and score on, defaulting to the controls (``"negcon"``), unlike
            :func:`correct_plate_position`, which fits on every well by default; see there.
        sigma: Width of the Gaussian that smooths the position map over the plate grid.
        n_splits: Number of cross-validation folds over the control wells.
        min_controls: Fewest control wells a plate needs to be scored; a plate below it is recorded unscored.
            Clamped to the two the cross-validation needs at minimum.
        seed: Seed for the fold assignment.
        key_added: Key under ``uns["mantispy"]`` for the result table.
        copy: Return a modified copy instead of writing in place.

    Returns:
        ``None``, or the modified copy.
        Writes a per-plate table to ``uns["mantispy"][key_added]`` with one row per plate and the columns
        ``plate``, ``n_controls``, ``cv_r2_median``, ``frac_features_positive`` and ``reason``.
        A positive ``cv_r2_median`` means the position structure generalizes and the plate has a learnable
        artifact worth correcting; a value at or below zero means the apparent structure does not generalize,
        so :func:`correct_plate_position` would remove noise and should be skipped. A plate with too few
        controls is recorded with missing scores and a ``reason``, and is neither scored nor, by this reading,
        a candidate for correction.

    Raises:
        KeyError: If ``reference`` names a column that is not present.
        ValueError: If the reference column has missing values.
        TypeError: If the reference column is not boolean.

    Notes:
        This pairs with :func:`correct_plate_position` as its gate: detect first, then correct only the
        plates whose artifact is learnable.
    """
    wells = adata.obs["Metadata_Well"].to_numpy()
    all_rows = np.array([well_row(well) for well in wells])
    all_columns = np.array([well_col(well) for well in wells])
    is_control = reference_mask(adata, reference)
    codes, keys = group_codes(adata, by)
    # Two folds need two controls; np.array_split(order, 0) would also raise.
    min_controls = max(min_controls, 2)

    table: dict[str, list] = {
        "plate": [],
        "n_controls": [],
        "cv_r2_median": [],
        "frac_features_positive": [],
        "reason": [],
    }

    def record(key: object, n_controls: int, cv_r2_median: float, frac_positive: float, reason: str) -> None:
        table["plate"].append(str(key))
        table["n_controls"].append(int(n_controls))
        table["cv_r2_median"].append(float(cv_r2_median))
        table["frac_features_positive"].append(float(frac_positive))
        table["reason"].append(reason)

    for group, key in enumerate(keys):
        selected = np.flatnonzero(codes == group)
        n_grid_rows = all_rows[selected].max() + 1
        n_grid_columns = all_columns[selected].max() + 1
        control = selected[is_control[selected]]

        # One grid position per control well, so several cells in a well are not separate observations.
        linear = all_rows[control] * n_grid_columns + all_columns[control]
        positions, inverse = np.unique(linear, return_inverse=True)
        n_controls = positions.size
        well_rows, well_columns = positions // n_grid_columns, positions % n_grid_columns

        if n_controls < min_controls:
            record(key, n_controls, float("nan"), float("nan"), f"only {n_controls} control wells")
            continue
        splits = min(n_splits, n_controls)
        if splits < 2:
            record(key, n_controls, float("nan"), float("nan"), "need at least two cross-validation folds")
            continue

        # grouped_stat skips NaN and inf, so a control well carrying one still contributes its finite features.
        values = grouped_stat(get_matrix(adata, rows=control), inverse, n_controls, MEAN).astype(np.float64)
        grand_mean = values.mean(axis=0)
        order = np.random.default_rng(seed).permutation(n_controls)
        ss_res = np.zeros(adata.n_vars)
        ss_tot = np.zeros(adata.n_vars)
        for held_out in np.array_split(order, splits):
            train = np.setdiff1d(order, held_out, assume_unique=True)

            # Each position is a distinct grid cell, so the scatter indices are unique: plain assignment.
            sums = np.zeros((n_grid_rows, n_grid_columns, adata.n_vars))
            counts = np.zeros((n_grid_rows, n_grid_columns, 1))
            sums[well_rows[train], well_columns[train]] = values[train]
            counts[well_rows[train], well_columns[train], 0] = 1.0
            smooth_sums = _smooth_grid(sums, sigma)
            smooth_counts = _smooth_grid(counts, sigma)[..., 0]

            position_map = np.broadcast_to(
                values[train].mean(axis=0), (n_grid_rows, n_grid_columns, adata.n_vars)
            ).copy()
            reached = smooth_counts > 1e-9
            position_map[reached] = smooth_sums[reached] / smooth_counts[reached, None]

            predicted = position_map[well_rows[held_out], well_columns[held_out]]
            ss_res += ((values[held_out] - predicted) ** 2).sum(axis=0)
            ss_tot += ((values[held_out] - grand_mean) ** 2).sum(axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            cv_r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
        scored = np.isfinite(cv_r2)
        record(
            key,
            n_controls,
            float(np.median(cv_r2[scored])) if scored.any() else float("nan"),
            float((cv_r2[scored] > 0).mean()) if scored.any() else float("nan"),
            "",
        )

    adata.uns.setdefault("mantispy", {})[key_added] = pd.DataFrame(table)
    get_logger().info("detect_plate_position scored position artifacts per %s into uns['mantispy'][%r]", by, key_added)
    return None


@inplace_or_copy()
def regress_out(
    adata: AnnData,
    keys: Sequence[str] = ("Metadata_CellCount",),
    by: str | None = "Metadata_Plate",
    reference: str | None = None,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Regress confounders out of every feature, within groups.

    Args:
        adata: Object to correct.
        keys: ``obs`` columns to regress out.
            Numeric columns enter directly; categorical ones are one-hot encoded with the first level dropped.
        by: Fit separately within each group of this column, usually the plate, which :func:`scanpy.pp.regress_out` cannot do.
            ``None`` fits one model globally.
        reference: Rows to fit on: ``None`` for all, ``"negcon"`` for ``Metadata_Control``, or the name of a boolean ``obs`` column.
            With a reference, each feature is re-expressed at the reference rows' mean covariate, and a covariate beyond the range the reference rows span is clipped to it, so no row is corrected by extrapolating the fit.
            Numeric covariates only.
        key_added: Write to ``layers[key_added]`` instead of overwriting ``X``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``X`` or ``layers[key_added]``, where each feature is replaced by its residual plus the fitted value at an anchor: the mean over the whole object for a numeric covariate, and the group's own mean for a categorical one, which keeps the units of the data.
        With ``reference``, the anchor is the reference rows' mean within each group.

    Raises:
        KeyError: If any of ``keys`` is not an ``obs`` column.
        ValueError: If a categorical covariate has missing values, or ``reference`` selects no rows or is given with a categorical covariate.

    Notes:
        Missing and infinite values stay as they are, and a feature holding one is fitted on its finite rows.
        A group with no more rows than design columns is left uncorrected and logged.

        Without a reference, every group is re-expressed at the pooled mean covariate, which removes a density difference between plates when every plate spans that value.
        A plate that does not span it is corrected by extrapolating its own fit, and a warning names it.

        With ``reference="negcon"`` the slope is estimated where density varies for technical reasons only.
        Fitted on every well of a screen whose treatments change density, the slope also carries the treatments' own effects, so the correction erodes their phenotypes and moves the controls away from the centre a control-referenced normalization put them at.
        With few control wells per plate, pool them with ``by=None`` and a covariate that is comparable between plates, such as the log of each well's cell count over its plate's control median.

        Whether the cell count is a confounder at all depends on the screen.
        In the ORF and CRISPR arms of JUMP, whose plate layouts were not randomized, it is largely technical, and the recipe regresses it out :cite:p:`Chandrasekaran_2023`.
        In a compound screen it is partly a treatment effect (a compound that kills cells is supposed to lower it), so regressing it out removes part of the phenotype along with the nuisance, and the recipe does not.
        Measure both ways before adopting either; :func:`~mantispy.metrics.batch_variance_explained` scores exactly this, a covariate at a time, and :func:`~mantispy.tl.cytotoxicity` asks the question directly.

        A numeric covariate with a missing or infinite value is dropped from that group's design and nothing is regressed out for it there, with a warning.
        A categorical covariate with a missing label is refused instead: the all-zero encoding of a missing category is also the encoding of the level ``drop_first`` removed, so those rows would be corrected as the reference level and take every other row with them.
    """
    missing = [key for key in keys if key not in adata.obs]
    if missing:
        raise KeyError(f"obs is missing regression key(s): {missing}")

    X = get_matrix(adata)
    codes, keys_index = group_codes(adata, by)
    out = np.array(X, dtype=np.float32)

    design_all, is_numeric, sources = _design_matrix(adata, keys)
    fitted_on = reference_mask(adata, reference)
    if reference is not None:
        if not fitted_on.any():
            raise ValueError(f"no reference rows selected by reference={reference!r}")
        if not is_numeric[1:].all():
            raise ValueError(
                "reference= fits numeric covariates only; a categorical level absent from the reference rows has no slope to apply"
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # "Mean of empty slice"
        pooled = np.nanmean(design_all, axis=0)
    pooled_ok = np.isfinite(pooled)

    extrapolated = []
    for group in range(len(keys_index)):
        rows = np.flatnonzero(codes == group)
        fit = fitted_on[rows]
        block_design = design_all[rows]
        unusable = ~np.isfinite(block_design).all(axis=0)
        if not fit.any():
            get_logger().warning(
                "regress_out: no reference rows within %s=%r, so nothing is regressed out there", by, keys_index[group]
            )
            continue
        # A column constant within the group carries no within-group information; its effect stays in the intercept.
        varying = (np.ptp(block_design[fit], axis=0) > 0) & ~unusable
        varying[0] = True
        incomplete = sorted({str(name) for name in sources[unusable] if name})
        if incomplete:
            scope = f" within {by}={keys_index[group]!r}" if by is not None else ""
            warnings.warn(
                f"regress_out: {incomplete} has missing or infinite values{scope}, so it is dropped from "
                "the design; nothing is regressed out for it there. Fill the column or drop those rows to "
                "correct that group.",
                UserWarning,
                stacklevel=3,
            )
        selected = np.flatnonzero(varying)[_independent(block_design[fit][:, varying])]
        design = block_design[:, selected]
        if fit.sum() <= design.shape[1]:
            get_logger().warning(
                "regress_out: %s has too few wells to fit within %s=%r, so nothing is regressed out there",
                list(keys),
                by,
                keys_index[group],
            )
            continue

        # Never anchored at zero: the bare intercept lies outside the data and turns each group's slope noise into an offset between groups.
        anchor = np.where(is_numeric[selected] & pooled_ok[selected], pooled[selected], design.mean(axis=0))
        applied = design
        if reference is not None:
            anchor = design[fit].mean(axis=0)
            applied = np.clip(design, design[fit].min(axis=0), design[fit].max(axis=0))
        elif ((anchor < design.min(axis=0)) | (anchor > design.max(axis=0)))[is_numeric[selected]].any():
            extrapolated.append(str(keys_index[group]))

        # A single inf in the batched block makes lstsq return NaN coefficients for every feature.
        block = X[rows].astype(np.float64)
        gaps = ~np.isfinite(block).all(axis=0)
        clean = np.flatnonzero(~gaps)
        if clean.size:
            values = block[:, clean]
            coefficients, *_ = np.linalg.lstsq(design[fit], values[fit], rcond=None)
            residual = values - applied @ coefficients
            out[np.ix_(rows, clean)] = (residual + anchor @ coefficients).astype(np.float32)

        for feature in np.flatnonzero(gaps):
            values = block[:, feature]
            observed = np.isfinite(values)
            if (observed & fit).sum() <= design.shape[1]:
                continue
            coefficients, *_ = np.linalg.lstsq(design[observed & fit], values[observed & fit], rcond=None)
            residual = values[observed] - applied[observed] @ coefficients
            gap_anchor = np.where(
                is_numeric[selected] & pooled_ok[selected], pooled[selected], design[observed].mean(axis=0)
            )
            if reference is not None:
                gap_anchor = design[observed & fit].mean(axis=0)
            out[rows[observed], feature] = (residual + gap_anchor @ coefficients).astype(np.float32)

    if extrapolated:
        warnings.warn(
            f"regress_out: the pooled mean of {list(keys)} lies outside the range {by}={extrapolated} span, so those "
            "groups are corrected by extrapolating their own fit, which can leave the result more dependent on the "
            "covariate than it was. Pass reference='negcon' to anchor each group at its own controls.",
            UserWarning,
            stacklevel=3,
        )
    if key_added is None:
        adata.X = out
    else:
        adata.layers[key_added] = out
    get_logger().info("regress_out removed %s within %s", list(keys), by)
    return None


def _design_matrix(adata: AnnData, keys: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the design over every row, a numeric mask over its columns, and each column's key.

    Built once for the whole object, so a group's design is a row slice and each column means the same in every group, which lets residuals be re-expressed at a common covariate value.
    The mask is needed because numeric columns are anchored at their pooled mean and dummies at the group's own mean.

    Raises:
        ValueError: If a categorical covariate has missing values, which :func:`pandas.get_dummies` encodes as all-zero: the same encoding as the level ``drop_first`` removes.
    """
    obs = as_frame(adata.obs)
    # The intercept is not a covariate, and belongs to no key.
    columns, numeric, sources = [np.ones(adata.n_obs)], [False], [""]
    for key in keys:
        values = obs[key]
        if pd.api.types.is_numeric_dtype(values) and not isinstance(values.dtype, pd.CategoricalDtype):
            columns.append(values.to_numpy(dtype=float))
            numeric.append(True)
            sources.append(key)
        else:
            missing = int(values.isna().sum())
            if missing:
                raise ValueError(
                    f"obs[{key!r}] has {missing} missing value(s), and a missing category is encoded "
                    "all-zero: exactly the encoding of the reference level that drop_first removes. "
                    "Those rows would be corrected as if they carried the reference level, and every "
                    "correctly labelled row would move with them. Fill the column (an unmatched "
                    "platemap row is the usual cause), drop those rows, or leave the key out."
                )
            # get_dummies emits an all-zero column for every unobserved declared level, which matrix_rank reads as collinearity.
            if isinstance(values.dtype, pd.CategoricalDtype):
                values = values.cat.remove_unused_categories()
            dummies = pd.get_dummies(values, drop_first=True, dtype=float).to_numpy().T
            columns.extend(dummies)
            numeric.extend([False] * len(dummies))
            sources.extend([key] * len(dummies))
    return np.column_stack(columns), np.array(numeric), np.array(sources)


def _independent(design: np.ndarray) -> np.ndarray:
    """Return the indices of a maximal linearly independent set of columns, intercept first.

    Dummies drop the first category of the whole object, so in a group that observed only the other levels they sum to the intercept.
    Dropping the redundant column lets the fit proceed without changing the fitted values.
    """
    keep: list[int] = []
    rank = 0
    for column in range(design.shape[1]):
        trial = [*keep, column]
        if np.linalg.matrix_rank(design[:, trial]) > rank:
            keep, rank = trial, rank + 1
    return np.array(keep, dtype=int)


@inplace_or_copy(expects=("well", "perturbation"))
def harmony(
    adata: AnnData,
    batch_key: str = "Metadata_Batch",
    use_rep: str = "X_pca",
    key_added: str = "X_harmony",
    max_iter: int = 20,
    seed: int = 0,
    copy: bool = False,
    **harmony_kwargs: Any,
) -> AnnData | None:
    """Correct an embedding for batch with Harmony.

    Harmony ranked in the top three in every scenario of the batch-correction benchmark of :cite:t:`Arevalo_2024`, as did Seurat RPCA.
    It is the last step of the `JUMP profiling recipe <https://github.com/broadinstitute/jump-profiling-recipe>`_ for compound and ORF profiles.
    It iterates soft clustering and per-cluster linear correction on an embedding, so it writes a corrected ``obsm`` and leaves ``X`` unchanged.

    Args:
        adata: Object holding the embedding to correct.
        batch_key: ``obs`` column naming the nuisance grouping, such as the batch, plate or imaging site.
        use_rep: Embedding to correct, normally the output of :func:`scanpy.pp.pca`.
        key_added: ``obsm`` key for the corrected embedding.
        max_iter: Harmony iterations.
        seed: Seed, passed through as ``random_state``.
        copy: Return a modified copy instead of writing in place.
        harmony_kwargs: Passed to ``harmonypy.run_harmony``, for example ``theta``, ``nclust`` or ``sigma``.

    Returns:
        ``None``, or the modified copy.
        Writes ``obsm[key_added]``.

    Raises:
        ImportError: If ``harmonypy`` is not installed.
        KeyError: If ``use_rep`` is not in ``obsm``, or ``batch_key`` is not an ``obs`` column.
        ValueError: If ``batch_key`` has a single level, leaving nothing to correct for.

    Notes:
        Requires ``harmonypy``: ``pip install 'mantispy[harmony]'``.

        harmonypy can report convergence and return the embedding unchanged; this wrapper warns when it does.
        What decides this is separability rather than size: it corrected the embedding of 50 640 JUMP TARGET2 wells across ten imaging sites, and returns the input untouched when the batches occupy disjoint regions of the embedding, because every soft cluster then holds a single batch.

        Check the result with more than one metric.
        On those JUMP wells Harmony moved the site centroids 36% closer together (mean separation 300 to 192, with unchanged overall spread) but lowered iLISI from 2.01 to 1.02, so the sites moved together globally while neighborhoods stayed site-pure.
        Batch metrics often disagree like this, which is why :func:`~mantispy.metrics.evaluate_integration` reports the whole scib-metrics panel rather than one number.
    """
    try:
        import harmonypy
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "Harmony needs harmonypy, which is an optional dependency: pip install 'mantispy[harmony]'"
        ) from error

    if use_rep not in adata.obsm:
        raise KeyError(f"obsm has no {use_rep!r}; compute it first, e.g. sc.pp.pca(adata, n_comps=50)")
    if batch_key not in adata.obs:
        raise KeyError(f"obs has no column {batch_key!r} to correct for")
    if as_frame(adata.obs)[batch_key].nunique() < 2:
        raise ValueError(f"{batch_key!r} has one level, so there is nothing to correct for")

    values = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    result = harmonypy.run_harmony(
        values,
        as_frame(adata.obs)[[batch_key]],
        [batch_key],
        max_iter_harmony=max_iter,
        random_state=seed,
        verbose=False,
        **harmony_kwargs,
    )
    # harmonypy returned Z_corr as (dims, rows) up to 0.0.10 and as (rows, dims) from 0.1.0, the change scanpy's sce.pp.harmony_integrate still does not handle (scverse/scanpy#3940).
    corrected = np.asarray(result.Z_corr)
    if corrected.shape[0] != adata.n_obs:
        corrected = corrected.T

    if np.array_equal(corrected, values):
        warnings.warn(
            f"harmony reported convergence but corrected nothing and returned {use_rep!r} unchanged. "
            "This happens when the batches are fully separated in the embedding, so every soft cluster "
            "holds a single batch. Check that the batches overlap, for example with mt.pl.metrics or an "
            f"embedding colored by {batch_key!r}.",
            UserWarning,
            stacklevel=3,
        )

    adata.obsm[key_added] = corrected.astype(np.float32)
    get_logger().info("harmony corrected %s for %s into %s", use_rep, batch_key, key_added)
    return None
