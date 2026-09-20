"""Recall of known relationships between perturbation profiles.

Two perturbations that act on the same complex or pathway should look more alike — or more opposed — than an arbitrary pair. The benchmark scores a map by the share of annotated pairs whose similarity falls in either tail of the similarity distribution over all pairs of the map :cite:p:`Celik_2024`.

Both tails count, because two perturbations with opposite effects on the same process are as related as two with the same effect. A map that carries no information recovers twice ``percentile`` of its annotated pairs, so the default 5 leaves a baseline of 10%.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mantispy._core._reduce import representation
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy.metrics._common import tidy
from mantispy.tl._similarity import similarity_matrix

if TYPE_CHECKING:
    from anndata import AnnData

#: Columns of a frame of sets, where two perturbations are related when they share a set.
SET_COLUMNS = ("source", "target")

#: Columns of a frame of pairs, which is how the reference relationship sets ship.
PAIR_COLUMNS = ("entity1", "entity2")

#: Largest number of annotated pairs a frame of sets is expanded into. A set of n members
#: contributes n(n-1)/2 pairs, so one very large set can dominate the recall as well as the memory.
MAX_PAIRS = 20_000_000


def _recall(null: np.ndarray, query: np.ndarray, tail: float) -> float:
    """Share of ``query`` that ranks in either tail of ``null``.

    The rank of a query value is its position in the sorted null rather than a value read off an interpolated quantile, which is what the reference implementation does and what makes ties fall on the conservative side: a value tied with much of the null is credited to neither tail.

    Args:
        null: The comparison distribution, in any order.
        query: The values to rank against it.
        tail: Size of each tail, as a fraction.

    Returns:
        The share of ``query`` at or below the lower tail, or at or above the upper one.
    """
    ordered = np.sort(null)
    at_or_below = np.searchsorted(ordered, query, side="right") / ordered.size
    strictly_below = np.searchsorted(ordered, query, side="left") / ordered.size
    return float(np.mean((at_or_below <= tail) | (strictly_below >= 1.0 - tail)))


def _pairs_from_sets(net: pd.DataFrame, codes: dict[str, int], n_labels: int) -> np.ndarray:
    """Pair codes for every two profiled labels that share a set."""
    blocks, total = [], 0
    for _, block in net.groupby(SET_COLUMNS[0], observed=True):
        members = np.unique([codes[name] for name in set(block[SET_COLUMNS[1]].astype(str)) if name in codes])
        total += members.size * (members.size - 1) // 2
        if total > MAX_PAIRS:
            raise ValueError(
                f"the sets in net expand into more than {MAX_PAIRS} pairs; drop the largest ones, e.g. "
                "net = net.groupby('source').filter(lambda block: len(block) <= 500)"
            )
        if members.size >= 2:
            rows, columns = np.triu_indices(members.size, 1)
            blocks.append(members[rows].astype(np.int64) * n_labels + members[columns])
    return np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64)


def _pairs_from_edges(net: pd.DataFrame, codes: dict[str, int], n_labels: int) -> np.ndarray:
    """Pair codes for every row naming two profiled labels."""
    left = net[PAIR_COLUMNS[0]].astype(str).map(codes).to_numpy(dtype=float)
    right = net[PAIR_COLUMNS[1]].astype(str).map(codes).to_numpy(dtype=float)
    both = np.isfinite(left) & np.isfinite(right)
    first = np.minimum(left[both], right[both]).astype(np.int64)
    second = np.maximum(left[both], right[both]).astype(np.int64)
    return (first * n_labels + second)[first != second]


def _pair_keys(net: pd.DataFrame, codes: dict[str, int], n_labels: int) -> np.ndarray:
    """Every unordered pair of profiled labels the annotation relates, as one integer each.

    A pair of row positions ``i < j`` is held as ``i * n_labels + j``, so deduplicating across sets, and collapsing a pair given in both directions, is one :func:`numpy.unique`, and the pairs of a large screen stay integers rather than tuples.

    Args:
        net: Either sets with :data:`SET_COLUMNS`, or pairs with :data:`PAIR_COLUMNS`.
        codes: Perturbation label to row position, for the labels that were profiled.
        n_labels: Number of profiled labels, which is the base of the encoding.

    Returns:
        The sorted, deduplicated pair codes, without self-pairs.

    Raises:
        ValueError: ``net`` has neither pair of columns, or the sets expand past :data:`MAX_PAIRS`.
    """
    columns = set(net.columns)
    if set(PAIR_COLUMNS) <= columns:
        keys = _pairs_from_edges(net, codes, n_labels)
    elif set(SET_COLUMNS) <= columns:
        keys = _pairs_from_sets(net, codes, n_labels)
    else:
        raise ValueError(
            f"net needs either {SET_COLUMNS} naming a set and one of its members, or {PAIR_COLUMNS} naming "
            f"two related perturbations, and has {sorted(columns)}"
        )
    return np.unique(keys)


def known_relationships(
    adata: AnnData,
    net: pd.DataFrame,
    label_key: str = "Metadata_Perturbation",
    metric: str = "cosine",
    use_rep: str | None = None,
    percentile: float = 5.0,
) -> pd.DataFrame:
    """Share of annotated pairs that land in either tail of the similarity distribution :cite:p:`Celik_2024`.

    Args:
        adata: One profile per perturbation, normally the output of :func:`~mantispy.tl.consensus`.
        net: The annotation, in either of two shapes. A frame with :data:`SET_COLUMNS`, as :func:`~mantispy.tl.gene_sets` returns, relates two perturbations that share a set, which also expresses a mechanism of action shared by several compounds. A frame with :data:`PAIR_COLUMNS` relates the two of each row, which is how the reference gene sets — CORUM, hu.MAP, Reactome, SIGNOR, StringDB — are distributed. Either way a pair is counted once, in either direction, and nothing is paired with itself.
        label_key: ``obs`` column holding the perturbation label, normally a gene symbol. Labels are matched to the annotation exactly, as the reference implementation matches them, so a screen that writes its symbols in another case recalls nothing.
        metric: Similarity between profiles, ``"cosine"`` or ``"pearson"``.
        use_rep: Measure in ``obsm[use_rep]`` instead of ``X``.
        percentile: Size of each tail, in percent. The comparison distribution is every pair of profiles, so the tails adapt to how similar the map is overall.

    Returns:
        A one-row tidy frame with ``metric``, ``representation``, ``key`` and ``value``, so it stacks with the other metrics.
        ``value`` is the recall, between 0 and 1.

    Raises:
        KeyError: ``obs`` has no column ``label_key``.
        ValueError: ``label_key`` repeats a label, so a pair of labels would not be a pair of profiles. Aggregate first with ``adata = mt.tl.consensus(adata)``.
        ValueError: ``net`` has neither shape of columns, or relates no two perturbations that were both profiled.
        ValueError: The sets expand into more than :data:`MAX_PAIRS` pairs.

    Notes:
        Read this against the 2 × ``percentile`` baseline, not against 100%. Annotated pairs are noisy — two genes share a complex and still do different things — so published maps recover a minority of them, and the number ranks pipelines against each other rather than standing on its own :cite:p:`Celik_2024`.

        Which annotation is supplied matters more than any argument here. Broad sets, such as the hallmark programs, call hundreds of genes related and pull the recall toward the baseline; curated complexes are the stricter test.

        The comparison distribution contains the annotated pairs themselves, as in the reference implementation. They are a small minority of all pairs in a real screen, and holding them out would score each source against a different distribution.
    """
    obs = as_frame(adata.obs)
    if label_key not in obs:
        raise KeyError(f"obs has no column {label_key!r} holding the perturbation label")

    labels = obs[label_key].astype(str).to_numpy()
    repeated = pd.Index(labels)[pd.Index(labels).duplicated()].unique()
    if len(repeated):
        raise ValueError(
            f"{label_key!r} repeats {len(repeated)} label(s), such as {sorted(repeated)[:3]}; this expects one "
            "profile per perturbation, so aggregate first with adata = mt.tl.consensus(adata)"
        )

    codes = {label: index for index, label in enumerate(labels)}
    keys = _pair_keys(net, codes, len(labels))
    if not keys.size:
        raise ValueError(
            f"net relates no two of the {len(labels)} perturbations in obs[{label_key!r}]; check that the "
            f"labels match, e.g. {sorted(labels)[:3]} against "
            f"{sorted(set(net[net.columns[-1]].astype(str)))[:3]}"
        )

    matrix = similarity_matrix(representation(adata, use_rep), metric)
    similarities = matrix[keys // len(labels), keys % len(labels)]
    # Every off-diagonal entry is the comparison distribution. Both triangles are kept because
    # duplicating a symmetric matrix leaves every rank fraction unchanged, and dropping one of
    # them costs a pair of index arrays larger than the matrix itself.
    np.fill_diagonal(matrix, np.nan)
    null = np.sort(matrix, axis=None)[: matrix.size - matrix.shape[0]]

    recall = _recall(null, similarities, percentile / 100.0)
    get_logger().info(
        "known_relationships: %.1f%% of %d annotated pair(s) in the tails, against a %.1f%% baseline",
        100 * recall,
        keys.size,
        2 * percentile,
    )
    return tidy("known_relationships", use_rep or "X", label_key, recall)
