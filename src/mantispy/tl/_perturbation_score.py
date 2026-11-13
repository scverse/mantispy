"""Per-cell perturbation score: how well a classifier tells each group's cells from the reference cells."""

from __future__ import annotations

import os
import warnings
import zlib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._reduce import _reads_from_disk, _source, get_matrix, group_codes
from mantispy._core._stats import _is_object_resolution
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy


def _roc_auc(y: np.ndarray, proba: np.ndarray) -> float:
    """ROC AUC from the out-of-fold probabilities, equal to :func:`sklearn.metrics.roc_auc_score` for a binary label.

    The AUC is the Mann-Whitney statistic, the mean rank of the positive scores standardised by the class sizes,
    which is :func:`~sklearn.metrics.roc_auc_score` to floating-point precision, ties shared by the average rank as
    that function also does, without its per-call validation of the label and the curve it never needs here.
    """
    from scipy.stats import rankdata

    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(proba)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _score_group(
    x: np.ndarray,
    y: np.ndarray,
    replicates: np.ndarray | None,
    n_folds: int,
    n_features: int,
    seed: int,
) -> tuple[np.ndarray, float, int]:
    """Out-of-fold probability of the group class for every row of ``x``, the ROC AUC, and the folds used.

    The fold loop is :func:`~sklearn.model_selection.cross_val_predict` of a feature-selection-then-logistic-regression
    pipeline, written out so each of the many small groups skips its per-call input validation. The splits, the
    refitted in-fold feature selection and the solver are unchanged, so the probabilities match it to the bit.
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    # A feature constant over the sample has no F statistic and only draws a warning from f_classif.
    x = x[:, np.ptp(x, axis=0) > 0]
    if replicates is None:
        folds = min(n_folds, int(np.bincount(y).min()))
    else:
        folds = min(n_folds, *(np.unique(replicates[y == side]).size for side in (0, 1)))
    if folds < 2 or x.shape[1] == 0:
        return np.full(int(y.sum()), np.nan), np.nan, 0

    splitter = StratifiedKFold if replicates is None else StratifiedGroupKFold
    cv = splitter(n_splits=folds, shuffle=True, random_state=seed)
    k = min(n_features, x.shape[1])
    select = k < x.shape[1]  # with every feature kept, SelectKBest is a no-op on the columns, in their own order

    proba = np.empty(x.shape[0], dtype=np.float64)
    for train, test in cv.split(x, y, replicates):
        x_train, y_train = x[train], y[train]
        if select:
            # The same f_classif ranking cross_val_predict refits per fold, so the held-out rows never pick features.
            columns = SelectKBest(score_func=f_classif, k=k).fit(x_train, y_train).get_support()
            x_train, x_test = x_train[:, columns], x[test][:, columns]
        else:
            x_test = x[test]
        model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
        model.fit(x_train, y_train)
        proba[test] = model.predict_proba(x_test)[:, 1]
    return proba[y == 1], _roc_auc(y, proba), folds


@inplace_or_copy(expects="object")
def perturbation_score(
    adata: AnnData,
    groupby: str = "Metadata_Gene",
    reference: str = "negcon",
    use_rep: str | None = None,
    layer: str | None = None,
    replicate: str | Sequence[str] | None = "Metadata_sgRNA",
    by: str | None = None,
    n_features: int = 200,
    n_folds: int = 5,
    min_cells: int = 50,
    n_jobs: int | None = None,
    seed: int = 0,
    key_added: str = "perturbation_score",
    copy: bool = False,
) -> AnnData | None:
    """Score each cell for how clearly it shows its group's phenotype, by a classifier against the reference cells.

    In an optical pooled screen many cells carry a guide but show no phenotype, because the knockout did not take or the cell escaped it, and averaging them with the responders dilutes a real hit.
    This is the perturbation score of Brieflow :cite:p:`Di_Bernardo_2026`: for each group, its cells and an equal-sized random sample of the reference cells are told apart by a logistic regression on the ``n_features`` features with the largest ANOVA F statistic.
    Each cell's score is its out-of-fold probability of belonging to the group, from a model that never saw it, and the group's ROC AUC over those probabilities says how separable the group is at all.
    Feature selection is refitted inside every fold, so the held-out cells never inform which features are used.

    The features should already be centred and scaled on the reference cells within each batch, which :func:`~mantispy.pp.normalize` does once for the whole screen with ``method="standardize"``, ``reference="negcon"`` and ``by`` the plate and well.
    Brieflow scales each group's sample on its own sampled controls instead, which fits the scale to the very controls the model then sees, so its AUCs run slightly higher.

    To drop the non-responders, subset on the score, for example ``adata[~(adata.obs["perturbation_score"] < 0.5)]``, which keeps the reference cells and the unscored groups, whose score is missing.

    Args:
        adata: Cells by features, with the group label and the reference class in ``obs``.
        groupby: ``obs`` column holding the group each cell is scored as, usually the gene.
        reference: The cells every group is told apart from: ``"negcon"`` for the rows whose ``Metadata_Control_Type`` is ``"negcon"``, or the name of a boolean ``obs`` column.
        use_rep: Read ``obsm[use_rep]`` instead of ``X``; cannot be combined with `layer`.
        layer: Read this layer instead of ``X``.
        replicate: ``obs`` column, or columns, whose groups are kept whole within a cross-validation fold, the guide by default (``"Metadata_sgRNA"``, as :func:`~mantispy.tl.composition_shift` also defaults to).
            The model is then always tested on guides it was not trained on, so a phenotype one guide carries alone, such as an off-target effect, does not raise the group's score.
            A named column that ``obs`` lacks raises, naming the argument.
            ``None`` draws folds over cells, as Brieflow does, and warns on an object stamped object resolution, since non-targeting guides grouped into pseudo-genes can then score above one half.
        by: ``obs`` column whose levels are scored apart, for example ``"Metadata_Class"`` to score interphase and mitotic cells separately, as Brieflow does.
            Each group is scored within each level against a reference sample drawn from that level, so a cell's score and its group's AUC come from its own level; a group too small within a level is left unscored there.
            ``None`` scores every cell in one pass.
        n_features: How many features, ranked by the ANOVA F statistic within each fold, the model reads.
        n_folds: Cross-validation folds, capped by the cells, or with `replicate` the replicates, on the smaller side.
        min_cells: Groups with fewer cells than this are left unscored.
            Brieflow's minimum of 100 counts the group and its equal-sized reference sample together.
        n_jobs: Groups scored in parallel threads; ``None`` scores them one at a time and ``-1`` uses every core.
        seed: Seeds the reference sample and the fold split, together with the group's name, so a group's score does not depend on which other groups are present.
        key_added: ``obs`` column for the per-cell score and the ``uns["mantispy"]`` key for the per-group table.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``obs[key_added]``, each cell's out-of-fold probability of its group, missing for the reference cells and for unscored groups.
        Writes ``uns["mantispy"][key_added]``, one row per group with ``group``, ``n_cells``, ``n_reference`` (the size of its reference sample), ``n_folds`` and ``auc``, which is missing for an unscored group; with `by`, one row per group and level and a column named for `by` holding the level.

    Raises:
        ValueError: `reference` selects no cells, or a scored cell or a reference cell has a missing feature value.
        KeyError: `groupby`, `replicate` or `by` names a column that ``obs`` lacks.

    Notes:
        The reference cells set the null each group is tested against and are not scored themselves.
        Cells of a reference class score as a group only when they are not in `reference`, for example intergenic guides scored against non-targeting ones, which is a check of the calibration; intergenic guides cut the genome, so they can carry a mild phenotype of their own.

        Cells in one well share its plate position, seeding and focus, but in a pooled screen every well holds every group, so the well does not separate a group from the reference and needs no grouping here.
        The guide does: random folds let the model learn what a guide's own cells share, and pass that off as the gene's phenotype.

        The cost grows with the number of cells, not with the number of groups, since each group's fit sees its own cells and as many reference cells.
        Each group reads only its own rows and its reference sample, so a backed object is never loaded whole.
    """
    is_reference = reference_mask(adata, reference)
    if not is_reference.any():
        raise ValueError(f"reference={reference!r} selects no cells")
    if replicate is not None:
        columns = [replicate] if isinstance(replicate, str) else list(replicate)
        missing = [column for column in columns if column not in adata.obs]
        if missing:
            raise KeyError(
                f"perturbation_score: replicate column(s) {missing} are not in obs; pass an existing column "
                "as replicate=, or replicate=None for Brieflow's random folds over cells"
            )
    if replicate is None and _is_object_resolution(adata):
        warnings.warn(
            "replicate=None draws folds over cells, so the model can learn what one guide's cells share and score it "
            "as the group's phenotype. Pass replicate= the guide column, e.g. replicate='Metadata_sgRNA'.",
            UserWarning,
            stacklevel=3,
        )
    codes, keys = group_codes(adata, groupby)
    replicate_codes = group_codes(adata, replicate)[0] if replicate is not None else None
    # One level per distinct value of `by`, scored on its own cells and its own reference sample; without
    # `by` the whole object is one level, with the seed and reference set it had before `by` existed.
    level_codes, level_keys = group_codes(adata, by) if by is not None else (None, None)

    # Read every row once into a dense block when the matrix is in memory, so each group slices it instead of
    # re-reading and re-casting its rows; a backed matrix keeps reading one group at a time, never loaded whole.
    # The block keeps its own float dtype rather than being widened to float64: liblinear casts each fold back up
    # to float64 from the same values, so a float32 screen scores identically from half the memory and bandwidth.
    source = _source(adata, layer, use_rep)
    matrix = None if _reads_from_disk(source) else np.asarray(get_matrix(adata, layer=layer, use_rep=use_rep))

    def read(both: np.ndarray) -> np.ndarray:
        if matrix is not None:
            return matrix[both]
        return np.asarray(get_matrix(adata, layer=layer, rows=both, use_rep=use_rep))

    def group_rows(pool: np.ndarray) -> dict[int, np.ndarray]:
        """Map each group's code to its non-reference rows within ``pool``, in increasing row order."""
        scored = pool[~is_reference[pool]]
        scored = scored[np.argsort(codes[scored], kind="stable")]
        if scored.size == 0:
            return {}
        scored_codes = codes[scored]
        starts = np.r_[0, np.flatnonzero(np.diff(scored_codes)) + 1]
        return {
            int(scored_codes[start]): rows for start, rows in zip(starts, np.split(scored, starts[1:]), strict=True)
        }

    # (level key, group code, the group's rows, the level's reference rows); without `by`, one level over all rows.
    Task = tuple[Any, int, np.ndarray, np.ndarray]
    tasks: list[Task] = []
    if by is None:
        rows_for = group_rows(np.arange(adata.n_obs))
        reference_rows = np.flatnonzero(is_reference)
        tasks = [(None, code, rows, reference_rows) for code, rows in rows_for.items()]
    else:
        assert level_codes is not None and level_keys is not None
        for level, level_key in enumerate(level_keys):
            in_level = level_codes == level
            rows_for = group_rows(np.flatnonzero(in_level))
            reference_rows = np.flatnonzero(is_reference & in_level)
            tasks += [(level_key, code, rows, reference_rows) for code, rows in rows_for.items()]

    def run(task: Task) -> tuple[np.ndarray, np.ndarray, float, int, int]:
        level_key, code, rows, reference_rows = task
        # Seed on the group's name, and on the level's when `by` splits, so a score depends on neither which
        # other groups nor which other levels are present; without `by` the entropy is what it was before.
        entropy = [seed, zlib.crc32(str(keys[code]).encode())]
        if by is not None:
            entropy.insert(1, zlib.crc32(str(level_key).encode()))
        rng = np.random.default_rng(entropy)
        drawn = np.sort(rng.choice(reference_rows, size=min(rows.size, reference_rows.size), replace=False))
        if rows.size < min_cells:
            return rows, np.full(rows.size, np.nan), np.nan, drawn.size, 0
        # Rows in object order, as Brieflow takes them, so a seeded fold split matches its split.
        both = np.union1d(rows, drawn)
        x = read(both)
        if not np.isfinite(x).all():
            raise ValueError(
                f"perturbation_score: group {keys[code]!r} or its reference sample has missing or infinite feature "
                "values; drop or fill them first, for example with mt.pp.feature_select"
            )
        y = np.isin(both, rows).astype(int)
        groups = replicate_codes[both] if replicate_codes is not None else None
        proba, auc, folds = _score_group(x, y, groups, n_folds, n_features, seed)
        return both[y == 1], proba, auc, drawn.size, folds

    if n_jobs is None or n_jobs == 1:
        results = [run(task) for task in tasks]
    else:
        # liblinear and the BLAS calls release the GIL, so threads share the matrix instead of copying it.
        with ThreadPoolExecutor(max_workers=os.cpu_count() if n_jobs == -1 else n_jobs) as pool:
            results = list(pool.map(run, tasks))

    score = np.full(adata.n_obs, np.nan)
    records = []
    for (level_key, code, _, _), (rows, proba, auc, n_reference, folds) in zip(tasks, results, strict=True):
        score[rows] = proba
        row = (str(keys[code]), rows.size, n_reference, folds, auc)
        records.append(row if by is None else (row[0], level_key, *row[1:]))
    columns = ["group", "n_cells", "n_reference", "n_folds", "auc"]
    if by is not None:
        columns.insert(1, by)
    table = pd.DataFrame(records, columns=columns)

    small = int((table["n_cells"] < min_cells).sum())
    unfolded = int(((table["n_cells"] >= min_cells) & (table["n_folds"] == 0)).sum())
    if small or unfolded:
        get_logger().warning(
            "perturbation_score: %d of %d groups left unscored, %d below min_cells=%d and %d without two folds",
            small + unfolded,
            len(table),
            small,
            min_cells,
            unfolded,
            deep="a group needs two replicates on each side to be split by replicate=; lower min_cells or pass replicate=None",
        )

    adata.obs[key_added] = score
    adata.uns.setdefault("mantispy", {})[key_added] = table
    get_logger().info(
        "perturbation_score scored %d of %d groups, median AUC %.2f",
        int(table["auc"].notna().sum()),
        len(table),
        float(table["auc"].median()) if table["auc"].notna().any() else float("nan"),
    )
    return None
