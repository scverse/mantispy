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
from mantispy.tl._similarity import _non_replicate_pool, similarity_matrix

if TYPE_CHECKING:
    from anndata import AnnData

#: Columns of the annotation, where two perturbations are related when they share a set.
#: This is what :func:`~mantispy.tl.gene_sets` returns and what :func:`~mantispy.tl.pathway_coherence` reads.
SET_COLUMNS = ("source", "target")

#: Largest number of annotated pairs a frame of sets is expanded into. A set of n members
#: contributes n(n-1)/2 pairs, so one very large set can dominate the recall as well as the memory.
MAX_PAIRS = 20_000_000


def _recall(null: np.ndarray, query: np.ndarray, tail: float) -> float:
    """Share of ``query`` that ranks in either tail of ``null``.

    The rank of a query value is its position in the sorted null rather than a value read off an interpolated quantile, which is what the reference implementation does and what makes ties fall on the conservative side: a value tied with much of the null is credited to neither tail.

    Args:
        null: The comparison distribution, which is sorted in place.
        query: The values to rank against it.
        tail: Size of each tail, as a fraction.

    Returns:
        The share of ``query`` at or below the lower tail, or at or above the upper one.
    """
    # Sorted in place: the caller builds this pool for this call and it is the largest array the
    # benchmark holds, so a sorted copy of it would be the function's peak.
    null.sort()
    at_or_below = np.searchsorted(null, query, side="right") / null.size
    strictly_below = np.searchsorted(null, query, side="left") / null.size
    return float(np.mean((at_or_below <= tail) | (strictly_below >= 1.0 - tail)))


def _pairs_from_sets(net: pd.DataFrame, codes: dict[str, int]) -> np.ndarray:
    """Row positions of every two profiled labels that share a set, as ``(n, 2)`` with ``i < j``.

    Raises:
        ValueError: The sets expand into more than :data:`MAX_PAIRS` pairs.
    """
    blocks, total = [], 0
    for _, block in net.groupby("source", observed=True):
        # codes is injective, so deduplicating the positions deduplicates the names too.
        members = np.unique([codes[name] for name in block["target"].astype(str) if name in codes])
        if members.size < 2:
            continue
        total += members.size * (members.size - 1) // 2
        if total > MAX_PAIRS:
            raise ValueError(
                f"the sets in net expand into more than {MAX_PAIRS} pairs; drop the largest ones, e.g. "
                "net = net.groupby('source').filter(lambda block: len(block) <= 500)"
            )
        rows, columns = np.triu_indices(members.size, 1)
        blocks.append(np.column_stack([members[rows], members[columns]]))
    return np.concatenate(blocks) if blocks else np.empty((0, 2), dtype=np.int64)


def _pair_keys(net: pd.DataFrame, codes: dict[str, int], n_labels: int) -> np.ndarray:
    """Every unordered pair of profiled labels the annotation relates, as one integer each.

    A pair of row positions ``i < j`` is held as ``i * n_labels + j``, so deduplicating across sets, and collapsing a pair given in both directions, is one :func:`numpy.unique`, and the pairs of a large screen stay integers rather than tuples.

    Args:
        net: The annotation, with :data:`SET_COLUMNS`.
        codes: Perturbation label to row position, for the labels that were profiled.
        n_labels: Number of profiled labels, which is the base of the encoding.

    Returns:
        The sorted, deduplicated pair codes, without self-pairs.

    Raises:
        ValueError: ``net`` lacks :data:`SET_COLUMNS`, or the sets expand past :data:`MAX_PAIRS`.
    """
    if not set(SET_COLUMNS) <= set(net.columns):
        raise ValueError(f"net needs {SET_COLUMNS}, naming a set and one of its members, and has {sorted(net.columns)}")
    pairs = _pairs_from_sets(net, codes)
    return np.unique(pairs[:, 0] * n_labels + pairs[:, 1])


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
        net: The annotation, with a ``source`` column naming a set and a ``target`` column naming one of its members, as :func:`~mantispy.tl.gene_sets` returns it: two perturbations are related when they share a set, which also expresses a mechanism of action shared by several compounds. A pair is counted once whichever way round it appears, and nothing is paired with itself. The reference gene sets — CORUM, hu.MAP, Reactome, SIGNOR, StringDB — are distributed as one pair per row, which becomes this shape with ``pairs.assign(source=pairs.index.astype(str)).melt(id_vars="source", value_name="target")[["source", "target"]]``.
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
        ValueError: ``net`` lacks ``source`` or ``target``, or relates no two perturbations that were both profiled.
        ValueError: The sets expand into more pairs than the module's ``MAX_PAIRS`` cap allows.

    Notes:
        Read this against the 2 × ``percentile`` baseline, not against 100%. Annotated pairs are noisy — two genes share a complex and still do different things — so published maps recover a minority of them, and the number ranks pipelines against each other rather than standing on its own :cite:p:`Celik_2024`.

        Which annotation is supplied matters more than any argument here. Broad sets, such as the hallmark programs, call hundreds of genes related and pull the recall toward the baseline; curated complexes are the stricter test.

        The comparison distribution contains the annotated pairs themselves, as in the reference implementation. They are a small minority of all pairs in a real screen, and holding them out would score each source against a different distribution.
    """
    obs = as_frame(adata.obs)
    if label_key not in obs:
        raise KeyError(f"obs has no column {label_key!r} holding the perturbation label")

    labels = obs[label_key].astype(str).to_numpy()
    index = pd.Index(labels)
    repeated = index[index.duplicated()].unique()
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
            f"labels match, e.g. {sorted(labels)[:3]} against {sorted(set(net['target'].astype(str)))[:3]}"
        )

    matrix = similarity_matrix(representation(adata, use_rep), metric)
    similarities = matrix[keys // len(labels), keys % len(labels)]
    # Every pair of distinct profiles is the comparison distribution. Giving each row its own
    # group makes every pair a non-replicate pair, so this is the strict upper triangle, read a
    # row block at a time rather than materialized as index arrays larger than the matrix.
    null = _non_replicate_pool(matrix, np.arange(len(labels)))

    recall = _recall(null, similarities, percentile / 100.0)
    get_logger().info(
        "known_relationships: %.1f%% of %d annotated pair(s) in the tails, against a %.1f%% baseline",
        100 * recall,
        keys.size,
        2 * percentile,
    )
    return tidy("known_relationships", use_rep or "X", label_key, recall)
