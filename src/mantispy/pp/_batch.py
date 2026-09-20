"""Plate-position, confounder and batch corrections.

Median polish corrects plate position, regression removes a measured confounder, and :func:`harmony` corrects the batch.
:func:`~mantispy.pp.sphere`, in its own module, whitens by the control covariance.

:func:`harmony` needs ``harmonypy`` 2.0 or later, installed with the ``harmony`` extra.
Processing a single laboratory's data does not need it.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
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
from mantispy._core.plate import plate_grid, well_col, well_row

METHODS = ("median_polish",)


def _median_polish_stack(grids: np.ndarray, max_iter: int, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Median polish every feature of a ``(rows, columns, features)`` stack.

    Each feature's grid is independent, so the stack goes to one numba kernel that polishes a plane per thread.
    A per-feature Python loop took 20 minutes on 132 JUMP plates, almost all of it interpreter and pandas overhead.

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
        reference: Fit the row and column effects on these rows only. ``"negcon"`` is the usual choice, so that treatments laid out in particular columns are not absorbed into a column effect. ``None`` fits on every well.
        max_iter: Maximum number of median-polish iterations.
        tol: Convergence tolerance of the median polish.
        key_added: Write to ``layers[key_added]`` instead of overwriting ``X``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes ``X`` or ``layers[key_added]``, and the fitted effects per plate to ``uns["mantispy"]["plate_position"]``.

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
    # Keep the output float32 and promote one plate at a time; float64 copies of the input
    # and output would take four times the matrix in memory.
    out = np.array(X, dtype=np.float32)
    effects: dict[str, dict[str, list]] = {}

    for group, key in enumerate(keys):
        selected = np.flatnonzero(codes == group)
        fit_rows = selected[fit_mask[selected]]
        if fit_rows.size == 0:
            raise ValueError(f"no reference rows in group {key!r}")

        # Each plate is polished on its own format's grid; one grid for the object would drop a 384-well plate
        # into a corner of a 1536-well one and fit the row and column effects on three quarters of nothing.
        n_rows, n_columns = plate_grid(wells[selected])

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


@inplace_or_copy()
def regress_out(
    adata: AnnData,
    keys: Sequence[str] = ("Metadata_CellCount",),
    by: str | None = "Metadata_Plate",
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Regress confounders out of every feature, within groups.

    Args:
        adata: Object to correct.
        keys: ``obs`` columns to regress out. Numeric columns enter directly; categorical ones are one-hot encoded with the first level dropped.
        by: Fit separately within each group of this column, usually the plate, which ``sc.pp.regress_out`` cannot do. ``None`` fits one model globally.
        key_added: Write to ``layers[key_added]`` instead of overwriting ``X``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes ``X`` or ``layers[key_added]``, where each feature is replaced by its residual plus the fitted value at an anchor: the mean over the whole object for a numeric covariate, and the group's own mean for a categorical one, which keeps the units of the data.

    Raises:
        KeyError: If any of ``keys`` is not an ``obs`` column.
        ValueError: If a categorical covariate has missing values.

    Notes:
        Missing and infinite values stay as they are, and a feature holding one is fitted on its finite rows.
        A group with no more rows than design columns is left uncorrected and logged.

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
    # nanmean, so one missing covariate value does not make the pooled anchor NaN. A column
    # whose pooled mean is still non-finite (all missing, or holding an inf) falls back to
    # each group's own mean, as dummies do.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # "Mean of empty slice"
        pooled = np.nanmean(design_all, axis=0)
    pooled_ok = np.isfinite(pooled)

    for group in range(len(keys_index)):
        rows = np.flatnonzero(codes == group)
        block_design = design_all[rows]
        # A column that does not vary inside this group carries no within-group
        # information; its effect stays in the intercept.
        # A covariate with a missing or infinite value cannot be fitted, so it is dropped too,
        # which leaves the group uncorrected. Leaving it alone is the conservative choice; doing
        # so silently is not, and the log line below reports the covariate as removed either way.
        unusable = ~np.isfinite(block_design).all(axis=0)
        varying = (np.ptp(block_design, axis=0) > 0) & ~unusable
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
        selected = np.flatnonzero(varying)[_independent(block_design[:, varying])]
        design = block_design[:, selected]
        if rows.size <= design.shape[1]:
            get_logger().warning(
                "regress_out: %s has too few wells to fit within %s=%r, so nothing is regressed out there",
                list(keys),
                by,
                keys_index[group],
            )
            continue

        # The covariate value the corrected values are re-expressed at. Anchoring at zero (the
        # bare intercept) lies outside the data and turns each group's slope noise into an
        # offset between groups. A numeric covariate is anchored at its pooled mean, so every
        # group ends at the same value. A dummy is anchored at the group's own mean, so a group
        # is not extrapolated onto a level it never observed; that part of
        # `anchor @ coefficients` is the group's mean fitted value, which is unique even where
        # the least squares solution is not.
        anchor = np.where(is_numeric[selected] & pooled_ok[selected], pooled[selected], design.mean(axis=0))

        # Complete features share one design, so lstsq solves them together; a feature with
        # gaps is fitted alone on its own rows. The split tests for non-finite values because a
        # single inf in the batched block makes lstsq return NaN coefficients for every feature.
        block = X[rows].astype(np.float64)
        gaps = ~np.isfinite(block).all(axis=0)
        clean = np.flatnonzero(~gaps)
        if clean.size:
            values = block[:, clean]
            coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
            residual = values - design @ coefficients
            out[np.ix_(rows, clean)] = (residual + anchor @ coefficients).astype(np.float32)

        for feature in np.flatnonzero(gaps):
            values = block[:, feature]
            observed = np.isfinite(values)
            if observed.sum() <= design.shape[1]:
                continue  # too few observations to fit; leave the feature alone
            coefficients, *_ = np.linalg.lstsq(design[observed], values[observed], rcond=None)
            residual = values[observed] - design[observed] @ coefficients
            # Same anchor rule, over the rows where this feature was measured.
            gap_anchor = np.where(
                is_numeric[selected] & pooled_ok[selected], pooled[selected], design[observed].mean(axis=0)
            )
            out[rows[observed], feature] = (residual + gap_anchor @ coefficients).astype(np.float32)

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
    The key names let a column that cannot be fitted be reported under the name the caller passed.

    Raises:
        ValueError: If a categorical covariate has missing values, which ``pd.get_dummies`` encodes as all-zero: the same encoding as the level ``drop_first`` removes.
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
            # get_dummies emits a column for every declared level, observed or not, and an
            # all-zero column looks like collinearity to np.linalg.matrix_rank.
            if isinstance(values.dtype, pd.CategoricalDtype):
                values = values.cat.remove_unused_categories()
            dummies = pd.get_dummies(values, drop_first=True, dtype=float).to_numpy().T
            columns.extend(dummies)
            numeric.extend([False] * len(dummies))
            sources.extend([key] * len(dummies))
    return np.column_stack(columns), np.array(numeric), np.array(sources)


def _independent(design: np.ndarray) -> np.ndarray:
    """Return the indices of a maximal linearly independent set of columns, intercept first.

    Dummies drop the first category of the whole object, so in a group that observed only the other levels they sum to the intercept (a plate run by operators B and C, with A as the reference, gives rank 2 with 3 columns).
    Dropping the redundant column lets the fit proceed without changing the fitted values.

    Makes one rank call per column on one group's block, which is cheap for a few keys on a plate's wells.
    Switch to a pivoted QR if this runs on cells with many keys.
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
        use_rep: Embedding to correct, normally ``sc.pp.pca``'s output.
        key_added: ``obsm`` key for the corrected embedding.
        max_iter: Harmony iterations.
        seed: Seed, passed through as ``random_state``.
        copy: Return a modified copy instead of writing in place.
        harmony_kwargs: Passed to ``harmonypy.run_harmony``, for example ``theta``, ``nclust`` or ``sigma``.

    Returns:
        ``None``, or the modified copy. Writes ``obsm[key_added]``.

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
        Batch metrics often disagree like this, which is why :func:`~mantispy.metrics.evaluate_correction` reports several and takes a ``map_key``.
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
