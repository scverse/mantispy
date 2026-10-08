"""Plate-position, confounder and batch corrections."""

from __future__ import annotations

import warnings
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._numba import MEDIAN, _polish_planes, grouped_stat
from mantispy._core._reduce import get_matrix, group_codes
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy
from mantispy._core.plate import well_col, well_row

METHODS = ("b_score", "median_polish")


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
    method: str = "b_score",
    by: str = "Metadata_Plate",
    reference: str | None = None,
    plates: Sequence[object] | None = None,
    max_iter: int = 10,
    tol: float = 1e-4,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Remove row and column position effects, per plate and per feature.

    Both methods fit the row and column effects with Tukey's two-way median polish, which is robust to a few extreme wells.

    Args:
        adata: Object to correct, at cell or well resolution.
        method: ``"b_score"`` (default) is the B-score of :cite:t:`Brideau_2003`: the median-polish residual, divided by the plate's median absolute deviation, so each feature becomes a robust, position-corrected z-score per plate.
            ``"median_polish"`` subtracts the row and column effects only, keeping each feature's original level and units.
            The B-score is the screening standard for calling hits against a positional gradient; it also re-scales each plate, so it doubles as a normalization and should not be followed by a second per-plate scaling.
        by: Column identifying the plate.
        reference: Fit the row and column effects on these rows only.
            ``"negcon"`` is the usual choice, so that treatments laid out in particular columns are not absorbed into a column effect.
            ``None`` fits on every well.
        plates: Correct only these ``by`` values and leave every other plate untouched.
            ``None`` corrects every plate. Pair it with :func:`detect_plate_position` to correct only the plates whose artifact is learnable.
        max_iter: Maximum number of median-polish iterations.
        tol: Convergence tolerance of the median polish.
        key_added: Write to ``layers[key_added]`` instead of overwriting ``X``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``X`` or ``layers[key_added]``, and the fitted effects per corrected plate to ``uns["mantispy"]["plate_position"]``.

    Raises:
        ValueError: If ``method`` is unknown, a corrected plate holds no reference rows, or ``plates`` names a value absent from ``by``.

    Notes:
        The polish is fitted on the well grid.
        At cell resolution each well is first reduced to its median, and the fit is then applied to every cell of that well.
        The B-score's level and scale are taken from the fitted wells, so with a ``reference`` they are the reference wells' median and spread.
        That scale is a per-well statistic; at cell resolution the per-cell B-score is therefore standardized only on average, not to an exact unit MAD.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    X = get_matrix(adata)
    wells = adata.obs["Metadata_Well"].to_numpy()
    rows = np.array([well_row(well) for well in wells])
    columns = np.array([well_col(well) for well in wells])

    fit_mask = reference_mask(adata, reference)
    codes, keys = group_codes(adata, by)
    chosen = None if plates is None else {str(plate) for plate in plates}
    if chosen is not None:
        unknown = chosen - {str(key) for key in keys}
        if unknown:
            raise ValueError(f"plates not found in {by!r}: {sorted(unknown)}")
    # Promote one plate at a time: float64 copies of the input and output would take four times the matrix in memory.
    out = np.array(X, dtype=np.float32)
    effects: dict[str, dict[str, list]] = {}

    for group, key in enumerate(keys):
        if chosen is not None and str(key) not in chosen:
            continue
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
        corrected = X[selected].astype(np.float64) - adjustment
        if method == "b_score":
            # The B-score standardizes the median-polish residual against the plate's own spread.
            fit_residual = grid[occupied] - (row_effect[occupied // n_columns] + column_effect[occupied % n_columns])
            with warnings.catch_warnings():  # a feature all-NaN on the fit wells has no level or spread
                warnings.simplefilter("ignore", RuntimeWarning)
                level = np.nanmedian(fit_residual, axis=0)
                scale = np.nanmedian(np.abs(fit_residual - level), axis=0)
            # Leave such a feature as the median-polish residual rather than wiping it to NaN.
            level = np.where(np.isfinite(level), level, 0.0)
            corrected = (corrected - level) / np.where(scale > 0, scale, 1.0)
        out[selected] = corrected.astype(np.float32)
        effects[str(key)] = {"row": row_effect.T.tolist(), "col": column_effect.T.tolist()}

    if key_added is None:
        adata.X = out
    else:
        adata.layers[key_added] = out
    adata.uns.setdefault("mantispy", {})["plate_position"] = effects
    return None


@inplace_or_copy()
def correct_chromosome_arm(
    adata: AnnData,
    *,
    expression: str | Path | pd.DataFrame | None = None,
    cell_line: str | None = None,
    gene: str = "Metadata_Gene",
    arm: str = "Metadata_ChromosomeArm",
    unexpressed: Collection[str] | None = None,
    tpm_cutoff: float = 0.5,
    min_genes: int = 20,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Remove the chromosome-arm (proximity-bias) background from CRISPR knockout profiles.

    A CRISPR cut can change the copy number along the gene's chromosome arm, so knockouts on the same arm tend to share a background that is not their biology. Following :cite:t:`Chandrasekaran_2023`'s profiling recipe, this subtracts, from every well on an arm, the mean profile of that arm's wells whose gene is not expressed in the screened cell line, since an unexpressed gene's knockout carries only the arm background.

    The unexpressed genes are read from DepMap for the screened cell line, so the correction is tied to the line rather than to any hard-coded reference. Pass the DepMap expression matrix and the cell line's model id, or a ready-made set of unexpressed genes.

    This is for CRISPR knockout data. ORF overexpression wells carry no such arm background, so do not run it on them even though :func:`mantispy.pp.annotate_jump` also gives them a chromosome arm.

    Args:
        adata: Object to correct, with one perturbed gene per well.
        expression: A DepMap expression matrix, as a path or a loaded frame, read by :func:`mantispy.io.unexpressed_genes` when `unexpressed` is not given.
        cell_line: The DepMap model id of the screened line, such as ``"ACH-000364"`` for U2OS.
        gene: Column holding each well's gene symbol.
        arm: Column holding each well's chromosome arm, such as ``"1p"``. Wells with no arm are left untouched. :func:`mantispy.ds.jump_crispr` writes both columns.
        unexpressed: Gene symbols counted as unexpressed. When ``None``, they are read from `expression` for `cell_line`.
        tpm_cutoff: The log2(TPM+1) value at or below which a gene counts as unexpressed, used when `unexpressed` is ``None``. The default admits lowly-expressed genes, not only those at zero TPM, so that enough genes sit on each arm to estimate its background, matching the permissive threshold the recipe uses.
        min_genes: Correct an arm only when more than this many of its unexpressed genes are present, so the background is estimated from enough wells.
        key_added: Write to ``layers[key_added]`` instead of overwriting ``X``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``X`` or ``layers[key_added]``, and the corrected arms with the count of unexpressed genes behind each to ``uns["mantispy"]["chromosome_arm"]``.

    Raises:
        KeyError: `gene` or `arm` is not a column of ``adata.obs``.
        ValueError: neither `unexpressed` nor both `expression` and `cell_line` are given.
    """
    for column in (gene, arm):
        if column not in adata.obs:
            raise KeyError(f"adata.obs has no {column!r} column")

    if unexpressed is None:
        if expression is None or cell_line is None:
            raise ValueError(
                "pass unexpressed=, or both expression= (a DepMap matrix) and cell_line= (its model id), "
                "so the unexpressed genes come from the screened cell line rather than a hard-coded reference"
            )
        from mantispy.io._jump import unexpressed_genes

        unexpressed = unexpressed_genes(expression, cell_line, tpm_cutoff=tpm_cutoff)

    obs = as_frame(adata.obs)
    gene_symbols = obs[gene].astype(str)
    arms = obs[arm]
    # isin over the gene column, not a Python membership loop: it is the codebase idiom, and a bare
    # string passed as `unexpressed` raises here instead of silently matching its characters.
    is_unexpressed = gene_symbols.isin(unexpressed).to_numpy()
    genes = gene_symbols.to_numpy()

    X = get_matrix(adata)
    out = np.array(X, dtype=np.float32)
    corrected: dict[str, int] = {}

    for key in pd.unique(arms[arms.notna()]):
        on_arm = (arms == key).to_numpy()
        unexpressed_on_arm = on_arm & is_unexpressed
        n_genes = np.unique(genes[unexpressed_on_arm]).size
        if n_genes <= min_genes:
            continue
        with warnings.catch_warnings():  # a feature all-NaN across the unexpressed wells has no background
            warnings.simplefilter("ignore", RuntimeWarning)
            background = np.nanmean(X[unexpressed_on_arm].astype(np.float64), axis=0)
        background = np.where(np.isfinite(background), background, 0.0)
        out[on_arm] = (X[on_arm].astype(np.float64) - background).astype(np.float32)
        corrected[str(key)] = int(n_genes)

    get_logger().info("chromosome-arm correction: corrected %d arm(s)", len(corrected))
    if key_added is None:
        adata.X = out
    else:
        adata.layers[key_added] = out
    adata.uns.setdefault("mantispy", {})["chromosome_arm"] = corrected
    return None


@inplace_or_copy()
def detect_plate_position(
    adata: AnnData,
    by: str = "Metadata_Plate",
    reference: str | None = "negcon",
    n_splits: int = 5,
    min_controls: int = 20,
    max_iter: int = 10,
    tol: float = 1e-4,
    seed: int = 0,
    key_added: str = "plate_position_detection",
    copy: bool = False,
) -> AnnData | None:
    """Test whether the plate-position correction generalizes, to gate it per plate.

    A per-plate row and column effect is often mis-estimated rather than a real artifact: with few
    controls on a large grid, the apparent gradient is the controls' own scatter, and
    :func:`correct_plate_position` would then subtract signal. This is a correctability gate, not an
    artifact detector: it cross-validates the very model the corrector applies -- the additive row and
    column median polish -- and reports whether it predicts held-out controls better than their plate
    level. A plate is worth correcting only when that model generalizes.

    Per plate, on the control wells selected by ``reference``, each well is reduced to one value per grid
    position. Across cross-validation folds the row and column effects are fitted on the training controls
    by the same median polish as :func:`correct_plate_position`, and a plate level is taken as the median
    of the training controls once their effects are removed. Each held-out control is predicted from that
    level plus its own row and column effect, and a cross-validated coefficient of determination is formed
    per feature from the held-out residuals against the level alone, then summarized per plate.

    Args:
        adata: Object to inspect, at cell or well resolution.
        by: Column identifying the plate.
        reference: Wells to fit and score on, defaulting to the controls (``"negcon"``), unlike
            :func:`correct_plate_position`, which fits on every well by default; see there. Pass the same
            ``reference`` to both so the gate validates what the correction will remove.
        n_splits: Number of cross-validation folds over the control wells.
        min_controls: Fewest control wells a plate needs to be scored; a plate below it is recorded unscored.
        max_iter: Maximum number of median-polish iterations, as in :func:`correct_plate_position`.
        tol: Convergence tolerance of the median polish, as in :func:`correct_plate_position`.
        seed: Seed for the fold assignment.
        key_added: Key under ``uns["mantispy"]`` for the result table.
        copy: Return a modified copy instead of writing in place.

    Returns:
        ``None``, or the modified copy.
        Writes a per-plate table to ``uns["mantispy"][key_added]`` with one row per plate and the columns
        ``plate``, ``n_controls``, ``cv_r2_median``, ``frac_features_positive`` and ``reason``.
        A positive ``cv_r2_median`` means the correction generalizes to held-out controls and the plate is
        worth correcting; a value at or below zero means it predicts them no better than their plate level,
        so :func:`correct_plate_position` would remove scatter and should be skipped. A value at or below
        zero is a decision to skip, not a verdict that the plate is clean, and a real but non-additive or
        weakly-estimated effect can score this way. A plate with too few controls is recorded with missing
        scores and a ``reason``; unscored means unknown, so inspect it rather than read it as negative.

    Raises:
        KeyError: If ``reference`` names a column that is not present.
        ValueError: If the reference column has missing values.
        TypeError: If the reference column is not boolean.

    Notes:
        This pairs with :func:`correct_plate_position` as its gate: detect first, then pass the passing
        plates to the corrector's ``plates`` argument so only they are corrected.
        It is also a quality-control signal on its own: a plate whose position structure is strong and
        generalizing can be excluded or inspected rather than corrected, which on multivariate profiles,
        where correcting rarely helps, is often the better choice. Whether correcting helps is a question
        for the downstream metric, not for this score.
    """
    wells = adata.obs["Metadata_Well"].to_numpy()
    all_rows = np.array([well_row(well) for well in wells])
    all_columns = np.array([well_col(well) for well in wells])
    is_control = reference_mask(adata, reference)
    codes, keys = group_codes(adata, by)

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

        # One value per control well; non-finite entries become NaN so later nan-aware steps skip them
        # per feature, as the comment promises. MEDIAN matches the reduction correct_plate_position fits on.
        values = grouped_stat(get_matrix(adata, rows=control), inverse, n_controls, MEDIAN).astype(np.float64)
        values[~np.isfinite(values)] = np.nan
        order = np.random.default_rng(seed).permutation(n_controls)
        ss_res = np.zeros(adata.n_vars)
        ss_tot = np.zeros(adata.n_vars)
        for held_out in np.array_split(order, splits):
            train = np.setdiff1d(order, held_out, assume_unique=True)

            # Fit the same additive row+col polish the corrector applies, on the training controls only.
            grid = np.full((n_grid_rows * n_grid_columns, adata.n_vars), np.nan)
            grid[positions[train]] = values[train]
            row_effect, column_effect = _median_polish_stack(
                grid.reshape(n_grid_rows, n_grid_columns, adata.n_vars), max_iter, tol
            )
            # A row or column with no training control has no estimable effect: zero it, so the polish's
            # grand-level bookkeeping does not leak a spurious offset and the score stays offset-invariant.
            seen_row = np.zeros(n_grid_rows, dtype=bool)
            seen_row[well_rows[train]] = True
            seen_column = np.zeros(n_grid_columns, dtype=bool)
            seen_column[well_columns[train]] = True
            row_effect[~seen_row] = 0.0
            column_effect[~seen_column] = 0.0

            # The polish centres its effects, so the plate level is what remains once they are removed.
            adjustment = row_effect[well_rows] + column_effect[well_columns]
            with warnings.catch_warnings():  # a feature all-NaN on the training controls has no level
                warnings.simplefilter("ignore", RuntimeWarning)
                level = np.nanmedian(values[train] - adjustment[train], axis=0)
            predicted = level + adjustment[held_out]
            ss_res += np.nansum((values[held_out] - predicted) ** 2, axis=0)
            ss_tot += np.nansum((values[held_out] - level) ** 2, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            cv_r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)
        scored = np.isfinite(cv_r2)
        if not scored.any():
            record(key, n_controls, float("nan"), float("nan"), "no feature had the variance to be scored")
            continue
        record(key, n_controls, float(np.median(cv_r2[scored])), float((cv_r2[scored] > 0).mean()), "")

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


@inplace_or_copy(expects=("well", "aggregate"))
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
