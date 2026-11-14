"""Per-cell perturbation score: how well a classifier tells each group's cells from the reference cells."""

from __future__ import annotations

import warnings
import zlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from anndata import AnnData
from numba import njit

from mantispy._core._liblinear import fit_l2_logistic
from mantispy._core._numba import _kernel_input
from mantispy._core._reduce import _reads_from_disk, _source, get_matrix, group_codes
from mantispy._core._stats import _is_object_resolution
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy

if TYPE_CHECKING:
    # numba's prange iterates like range, but its own type says nothing about that.
    from builtins import range as prange
else:
    from numba import prange

#: A backed matrix is read this many rows at a time, a batch of whole groups and their reference samples.
_BACKED_BATCH_ROWS = 1 << 20


@njit(cache=True, nogil=True, error_model="numpy")
def _top_columns(
    x: np.ndarray, rows: np.ndarray, y: np.ndarray, train: np.ndarray, columns: np.ndarray, k: int
) -> np.ndarray:
    """The ``k`` of ``columns`` with the largest ANOVA F statistic over the training rows, in column order.

    The statistic is :func:`~sklearn.feature_selection.f_classif`'s sums-of-squares formula, a missing one ranks
    lowest as in :func:`~sklearn.feature_selection.SelectKBest`, and ties keep the later column, as its stable sort does.
    """
    scores = np.empty(columns.size)
    for c in range(columns.size):
        j = columns[c]
        n = np.zeros(2)
        sums = np.zeros(2)
        squares = 0.0
        for i in range(rows.size):
            if train[i]:
                value = np.float64(x[rows[i], j])
                n[y[i]] += 1
                sums[y[i]] += value
                squares += value * value
        total = sums[0] + sums[1]
        n_all = n[0] + n[1]
        between = sums[0] ** 2 / n[0] + sums[1] ** 2 / n[1] - total**2 / n_all
        within = (squares - total**2 / n_all) - between
        f = between / (within / (n_all - 2))
        scores[c] = np.finfo(np.float64).min if np.isnan(f) else f
    keep = np.zeros(columns.size, dtype=np.bool_)
    keep[np.argsort(scores, kind="mergesort")[columns.size - k :]] = True
    return columns[keep]


@njit(cache=True, nogil=True)
def _fit_fold(
    x: np.ndarray, rows: np.ndarray, y: np.ndarray, fold: np.ndarray, held_out: int, n_features: int, proba: np.ndarray
) -> bool:
    """Fit one fold of a group and write the held-out rows' probability of the group class into ``proba``.

    The fit is ``SelectKBest(f_classif, k=n_features)`` then ``LogisticRegression(class_weight="balanced",
    solver="liblinear", max_iter=2000)`` on the rows outside ``held_out``, over the features that vary across the
    group's rows. Returns ``False``, leaving ``proba`` untouched, when no feature varies or the training rows lack a class.
    """
    n_columns = x.shape[1]
    varying = np.zeros(n_columns, dtype=np.bool_)
    for j in range(n_columns):
        first = x[rows[0], j]
        for i in range(1, rows.size):
            if x[rows[i], j] != first:
                varying[j] = True
                break
    columns = np.flatnonzero(varying)
    train = fold != held_out
    n_train = 0
    n_pos = 0
    for i in range(rows.size):
        if train[i]:
            n_train += 1
            n_pos += y[i]
    if columns.size == 0 or n_pos == 0 or n_pos == n_train:
        return False
    k = min(n_features, columns.size)
    if k < columns.size:
        columns = _top_columns(x, rows, y, train, columns, k)

    # liblinear's layout: the negative class's rows first, each class in row order, and a bias column of ones.
    weight = (n_train / (2.0 * (n_train - n_pos)), n_train / (2.0 * n_pos))  # class_weight="balanced"
    xt = np.empty((n_train, k + 1))
    yt = np.empty(n_train)
    cost = np.empty(n_train)
    r = 0
    for side in range(2):
        for i in range(rows.size):
            if train[i] and y[i] == side:
                for c in range(k):
                    xt[r, c] = x[rows[i], columns[c]]
                xt[r, k] = 1.0
                yt[r] = 2.0 * side - 1.0
                cost[r] = weight[side]
                r += 1
    w = fit_l2_logistic(xt, yt, cost, 1e-4, 2000)
    for i in range(rows.size):
        if not train[i]:
            margin = 0.0
            for c in range(k):
                margin += x[rows[i], columns[c]] * w[c]
            proba[i] = 1 / (1 + np.exp(-(margin + w[k])))
    return True


@njit(cache=True, nogil=True)
def _roc_auc(y: np.ndarray, proba: np.ndarray) -> float:
    """ROC AUC as the Mann-Whitney statistic, ties sharing the average rank, as :func:`sklearn.metrics.roc_auc_score`."""
    order = np.argsort(proba, kind="mergesort")
    n_pos = 0
    rank_sum = 0.0
    start = 0
    while start < order.size:
        stop = start + 1
        while stop < order.size and proba[order[stop]] == proba[order[start]]:
            stop += 1
        rank = (start + 1 + stop) / 2.0
        for i in range(start, stop):
            if y[order[i]]:
                n_pos += 1
                rank_sum += rank
        start = stop
    n_neg = order.size - n_pos
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


@njit(parallel=True, cache=True, nogil=True)
def _score_groups(
    x: np.ndarray,
    rows: np.ndarray,
    y: np.ndarray,
    fold: np.ndarray,
    starts: np.ndarray,
    n_folds: np.ndarray,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold probabilities over the concatenated groups and each group's AUC, every fold of every group a thread task.

    Group ``g`` holds ``rows[starts[g]:starts[g + 1]]`` of ``x``, its labels ``y`` and fold numbers ``fold``. A group
    with a fold that could not be fitted is left missing throughout.
    """
    n_groups = n_folds.size
    jobs = np.zeros(n_groups + 1, dtype=np.int64)
    for g in range(n_groups):
        jobs[g + 1] = jobs[g] + n_folds[g]
    proba = np.full(rows.size, np.nan)
    fitted = np.zeros(jobs[n_groups], dtype=np.bool_)
    for job in prange(jobs[n_groups]):
        group = np.searchsorted(jobs, job, side="right") - 1
        a, b = starts[group], starts[group + 1]
        fitted[job] = _fit_fold(x, rows[a:b], y[a:b], fold[a:b], job - jobs[group], n_features, proba[a:b])
    auc = np.full(n_groups, np.nan)
    for g in prange(n_groups):
        a, b = starts[g], starts[g + 1]
        if fitted[jobs[g] : jobs[g + 1]].all():
            auc[g] = _roc_auc(y[a:b], proba[a:b])
        else:
            proba[a:b] = np.nan
    return proba, auc


@njit(parallel=True, cache=True, nogil=True)
def _finite_rows(x: np.ndarray) -> np.ndarray:
    """Whether every value of each row is finite."""
    out = np.ones(x.shape[0], dtype=np.bool_)
    for i in prange(x.shape[0]):
        for j in range(x.shape[1]):
            if not np.isfinite(x[i, j]):
                out[i] = False
                break
    return out


def _replicate_codes(obs: pd.DataFrame, replicate: str | Sequence[str]) -> np.ndarray:
    """Each row's replicate code, numbered by the sorted string labels whatever the column's dtype.

    The group splitter orders the replicates by their codes before it shuffles them, so a string column and a
    categorical one, whose categories may come in any order, would otherwise draw different folds from the same guides.
    """
    columns = [replicate] if isinstance(replicate, str) else list(replicate)
    ranks = []
    for column in columns:
        values = pd.Categorical(obs[column])
        if (values.codes < 0).any():
            raise ValueError(f"replicate column {column!r} contains missing values. Filter or fill them first.")
        rank = np.unique(np.asarray(values.categories.astype(str), dtype=str), return_inverse=True)[1]
        ranks.append(rank[values.codes])
    if len(ranks) == 1:
        return ranks[0]
    return np.unique(np.column_stack(ranks), axis=0, return_inverse=True)[1].ravel()


def _folds(y: np.ndarray, replicates: np.ndarray | None, n_folds: int, seed: int) -> tuple[np.ndarray, int]:
    """Each row's cross-validation fold, from the splitter :func:`~sklearn.model_selection.cross_val_predict` would use."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    if replicates is None:
        folds = min(n_folds, int(np.bincount(y).min()))
    else:
        folds = min(n_folds, *(np.unique(replicates[y == side]).size for side in (0, 1)))
    fold = np.empty(y.size, dtype=np.int8)
    if folds < 2:
        return fold, 0
    splitter = StratifiedKFold if replicates is None else StratifiedGroupKFold
    cv = splitter(n_splits=folds, shuffle=True, random_state=seed)
    for number, (_, test) in enumerate(cv.split(np.empty((y.size, 1)), y, replicates)):
        fold[test] = number
    return fold, folds


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
            Replicates are ordered by their labels as strings, so a string column and a categorical one draw the same folds.
            ``None`` draws folds over cells, as Brieflow does, and warns on an object stamped object resolution, since non-targeting guides grouped into pseudo-genes can then score above one half.
        by: ``obs`` column whose levels are scored apart, for example ``"Metadata_Class"`` to score interphase and mitotic cells separately, as Brieflow does.
            Each group is scored within each level against a reference sample drawn from that level, so a cell's score and its group's AUC come from its own level; a group too small within a level is left unscored there.
            ``None`` scores every cell in one pass.
        n_features: How many features, ranked by the ANOVA F statistic within each fold, the model reads.
        n_folds: Cross-validation folds, capped by the cells, or with `replicate` the replicates, on the smaller side.
        min_cells: Groups with fewer cells than this are left unscored.
            Brieflow's minimum of 100 counts the group and its equal-sized reference sample together.
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
        Every fold of every group is fitted in parallel on numba's threads, which :func:`numba.set_num_threads` caps.
        The fit is a port of the liblinear solver ``LogisticRegression(solver="liblinear")`` runs, so the scores match scikit-learn's to rounding.
        A backed object is read a batch of groups at a time, each group's rows and its reference sample, so it is never loaded whole.
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
    replicate_codes = _replicate_codes(as_frame(adata.obs), replicate) if replicate is not None else None
    # One level per distinct value of `by`, scored on its own cells and its own reference sample; without
    # `by` the whole object is one level, with the seed and reference set it had before `by` existed.
    level_codes, level_keys = group_codes(adata, by) if by is not None else (None, None)

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

    # Python draws each group's reference sample and its folds; the fits run in one numba kernel over every
    # (group, fold), reading the matrix in place in its own float dtype, as liblinear casts each fold up to float64.
    Prepared = tuple[np.ndarray, np.ndarray, np.ndarray, int]  # rows, labels, folds, number of folds
    prepared: dict[int, Prepared] = {}
    n_reference = np.empty(len(tasks), dtype=np.int64)
    for t, (level_key, code, rows, reference_rows) in enumerate(tasks):
        # Seed on the group's name, and on the level's when `by` splits, so a score depends on neither which
        # other groups nor which other levels are present; without `by` the entropy is what it was before.
        entropy = [seed, zlib.crc32(str(keys[code]).encode())]
        if by is not None:
            entropy.insert(1, zlib.crc32(str(level_key).encode()))
        rng = np.random.default_rng(entropy)
        drawn = np.sort(rng.choice(reference_rows, size=min(rows.size, reference_rows.size), replace=False))
        n_reference[t] = drawn.size
        if rows.size < min_cells:
            continue
        # Rows in object order, as Brieflow takes them, so a seeded fold split matches its split.
        both = np.union1d(rows, drawn)
        y = np.isin(both, rows).astype(np.int8)
        fold, folds = _folds(y, replicate_codes[both] if replicate_codes is not None else None, n_folds, seed)
        if folds:
            prepared[t] = (both, y, fold, folds)

    score = np.full(adata.n_obs, np.nan)
    auc = np.full(len(tasks), np.nan)
    source = _source(adata, layer, use_rep)
    backed = _reads_from_disk(source)
    # In memory the whole matrix is one batch, read in place; a backed one is read a batch of groups at a time.
    batches, batch, size = [], [], 0
    for t, (both, *_) in prepared.items():
        batch.append(t)
        size += both.size
        if backed and size >= _BACKED_BATCH_ROWS:
            batches.append(batch)
            batch, size = [], 0
    if batch:
        batches.append(batch)
    matrix = None if backed else np.asarray(get_matrix(adata, layer=layer, use_rep=use_rep))
    for batch in batches:
        parts = [prepared[t] for t in batch]
        rows = np.concatenate([p[0] for p in parts])
        if matrix is None:
            needed = np.unique(rows)
            x = np.asarray(get_matrix(adata, layer=layer, rows=needed, use_rep=use_rep))
            local = np.searchsorted(needed, rows)
        else:
            x, local = matrix, rows
        x = _kernel_input(x)
        finite = _finite_rows(x)
        starts = np.r_[0, np.cumsum([p[0].size for p in parts])]
        for t, a, b in zip(batch, starts[:-1], starts[1:], strict=True):
            if not finite[local[a:b]].all():
                raise ValueError(
                    f"perturbation_score: group {keys[tasks[t][1]]!r} or its reference sample has missing or "
                    "infinite feature values; drop or fill them first, for example with mt.pp.feature_select"
                )
        proba, batch_auc = _score_groups(
            x,
            local.astype(np.int64),
            np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]),
            starts.astype(np.int64),
            np.array([p[3] for p in parts], dtype=np.int64),
            int(n_features),
        )
        auc[batch] = batch_auc
        for (both, y, _, _), a, b in zip(parts, starts[:-1], starts[1:], strict=True):
            score[both[y == 1]] = proba[a:b][y == 1]

    records = []
    for t, (level_key, code, rows, _) in enumerate(tasks):
        # A group whose folds could not all be fitted, for want of a varying feature or a class, has no AUC.
        folds = prepared[t][3] if t in prepared and np.isfinite(auc[t]) else 0
        row = (str(keys[code]), rows.size, int(n_reference[t]), folds, auc[t])
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
