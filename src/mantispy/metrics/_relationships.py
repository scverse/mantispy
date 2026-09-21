"""Recall of known relationships between perturbation profiles.

Two perturbations that act on the same complex or pathway should look more alike — or more opposed — than an arbitrary pair. The benchmark scores a map by the share of annotated pairs whose similarity falls in either tail of the similarity distribution over all pairs of the map :cite:p:`Celik_2024`.

Both tails count, because two perturbations with opposite effects on the same process are as related as two with the same effect. When every perturbation belongs to the same number of sets, a map that carries no information recovers twice ``percentile`` of its annotated pairs, 10% at the default 5. When some belong to many sets, as compounds with several targets do, their pairs are overrepresented and chance moves with where the map puts them, so it has to be measured by shuffling the annotation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mantispy._core._reduce import representation
from mantispy._core._stats import permutation_pvalue
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy.metrics._common import tidy

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

    The rank of a query value is its position in the sorted null, as the reference implementation takes it, rather than a comparison against an interpolated quantile. That is what puts ties on the conservative side: a value tied with much of the null is credited to neither tail.

    Args:
        null: The comparison distribution, sorted.
        query: The values to rank against it.
        tail: Size of each tail, as a fraction.

    Returns:
        The share of ``query`` at or below the lower tail, or at or above the upper one.
    """
    at_or_below = np.searchsorted(null, query, side="right") / null.size
    strictly_below = np.searchsorted(null, query, side="left") / null.size
    return float(np.mean((at_or_below <= tail) | (strictly_below >= 1.0 - tail)))


def _pairs_from_sets(net: pd.DataFrame, codes: dict[str, int]) -> np.ndarray:
    """Row positions of every two profiled labels that share a set.

    Args:
        net: The annotation, with a ``source`` column naming a set and a ``target`` column naming one of its members.
        codes: Perturbation label to row position, for the labels that were profiled.

    Returns:
        One row per pair, as ``(n, 2)`` with ``i < j``. A member listed twice in one set yields one pair, not three.

    Raises:
        ValueError: The sets expand into more than :data:`MAX_PAIRS` pairs.
    """
    # Mapped once for the whole frame rather than per set, because the reference relationship files are
    # distributed as one pair per row, which makes every set a group of two.
    positions = net["target"].astype(str).map(codes)
    members = pd.DataFrame({"source": np.asarray(net["source"]), "position": positions})
    members = members.dropna(subset=["position"]).astype({"position": np.int64}).drop_duplicates()

    # Sorted by set and then by position, so each set's rows are adjacent and ascending. That is what lets
    # the pairs be laid out by arithmetic below, and it means a pair comes out as (i, j) with i < j already.
    members = members.sort_values(["source", "position"], kind="stable")
    sets = members["source"].to_numpy()
    positions_by_set = members["position"].to_numpy()

    if positions_by_set.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    starts = np.flatnonzero(np.r_[True, sets[1:] != sets[:-1]])
    sizes = np.diff(np.r_[starts, positions_by_set.size])
    total = int((sizes * (sizes - 1) // 2).sum())
    if total > MAX_PAIRS:
        raise ValueError(
            f"the sets in net expand into more than {MAX_PAIRS} pairs; drop the largest ones, e.g. "
            "net = net.groupby('source').filter(lambda block: len(block) <= 500)"
        )

    # Each member pairs with the members after it in its own set, so it opens that many pairs. Expanding by
    # those counts walks every upper-triangle pair of every set at once, with no per-set Python loop: this is
    # the whole cost of the benchmark when the annotation is resampled to build a null.
    within_set = np.arange(positions_by_set.size) - np.repeat(starts, sizes)
    opened = np.repeat(sizes, sizes) - 1 - within_set
    first = np.repeat(np.arange(positions_by_set.size), opened)
    # The partners of one member are the rows straight after it, so the n-th pair it opens is n rows along.
    step = np.arange(total) - np.repeat(np.cumsum(opened) - opened, opened)
    return np.column_stack([positions_by_set[first], positions_by_set[first + 1 + step]])


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
    name: str | None = None,
    n_permutations: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """Share of annotated pairs that land in either tail of the similarity distribution :cite:p:`Celik_2024`.

    Args:
        adata: One profile per perturbation, normally the output of :func:`~mantispy.tl.consensus`.
        net: The annotation, with a ``source`` column naming a set and a ``target`` column naming one of its members, as :func:`~mantispy.tl.gene_sets` returns it: two perturbations are related when they share a set, which also expresses a mechanism of action shared by several compounds. A pair is counted once whichever way round it appears, and nothing is paired with itself. The reference gene sets — CORUM, hu.MAP, Reactome, SIGNOR, StringDB — are distributed as one pair per row, which becomes this shape with ``pairs.assign(source=pairs.index.astype(str)).melt(id_vars="source", value_name="target")[["source", "target"]]``.
        label_key: ``obs`` column holding the perturbation label, normally a gene symbol. Labels are matched to the annotation exactly, as the reference implementation matches them, so a screen that writes its symbols in another case recalls nothing.
        metric: Similarity between profiles, ``"cosine"`` or ``"pearson"``.
        use_rep: Measure in ``obsm[use_rep]`` instead of ``X``.
        percentile: Size of each tail, in percent, between 0 and 50. The comparison distribution is every pair of profiles, so the tails adapt to how similar the map is overall.
        name: What to call this annotation in the ``metric`` column, as ``known_relationships:name``. Each source is scored separately, and two rows both called ``known_relationships`` would collide when :func:`~mantispy.pl.metrics` pivots the table.
        n_permutations: How many times to shuffle which perturbation each annotation row names, to measure the recall this map gives by chance. Each shuffle keeps the size of every set and the number of sets every perturbation belongs to. ``0`` skips it.
        seed: Seed for the shuffles.

    Returns:
        A one-row tidy frame with ``metric``, ``representation``, ``key`` and ``value``, so it stacks with the other metrics.
        ``value`` is the recall, between 0 and 1.
        With ``n_permutations``, also ``null``, the mean recall over the shuffles, and ``p_value``, the share of shuffles recalling at least as much, counted as ``(k + 1) / (n + 1)``.

    Raises:
        KeyError: ``obs`` has no column ``label_key``.
        ValueError: ``label_key`` repeats a label, so a pair of labels would not be a pair of profiles. Aggregate first with ``adata = mt.tl.consensus(adata)``.
        ValueError: ``net`` lacks ``source`` or ``target``, or relates no two perturbations that were both profiled.
        ValueError: ``percentile`` is not between 0 and 50, or the sets expand into more pairs than the module's ``MAX_PAIRS`` cap allows.

    Notes:
        Read this against chance, not against 100%: 2 × ``percentile`` when every perturbation belongs to the same number of sets, and the ``null`` of ``n_permutations`` otherwise. Annotated pairs are noisy — two genes share a complex and still do different things — so published maps recover a minority of them, and the number ranks pipelines against each other rather than standing on its own :cite:p:`Celik_2024`.

        Which annotation is supplied matters more than any argument here. Broad sets, such as the hallmark programs, call hundreds of genes related and pull the recall toward the baseline; curated complexes are the stricter test. Score each source on its own, under its own ``name``, rather than concatenating them: a pair two sources agree on would otherwise be counted once and a source with more pairs would decide the number.

        The comparison distribution contains the annotated pairs themselves, as in the reference implementation. They are a small minority of all pairs in a real screen, and holding them out would score each source against a different distribution.
    """
    from mantispy.tl._similarity import _non_replicate_pool, similarity_matrix

    if not 0 < percentile < 50:
        raise ValueError(f"percentile is the size of one tail in percent and must be in (0, 50), got {percentile}")

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
    pool = _non_replicate_pool(matrix, np.arange(len(labels)))
    # Sorted in place, once for the observed recall and every shuffle: this function built the pool and nothing
    # else reads it, so a sorted copy would be one more array the size of the upper triangle.
    pool.sort()

    tail = percentile / 100.0
    recall = _recall(pool, similarities, tail)
    get_logger().info("known_relationships: %.1f%% of %d annotated pair(s) in the tails", 100 * recall, keys.size)
    result = tidy(
        "known_relationships" if name is None else f"known_relationships:{name}", use_rep or "X", label_key, recall
    )
    if not n_permutations:
        return result

    rng = np.random.default_rng(seed)
    members = net["target"].to_numpy()
    shuffled = []
    for _ in range(n_permutations):
        # Moving members between sets keeps every set's size and every perturbation's number of sets, which is
        # what decides how often its pairs are drawn.
        permuted = _pair_keys(net.assign(target=rng.permutation(members)), codes, len(labels))
        shuffled.append(_recall(pool, matrix[permuted // len(labels), permuted % len(labels)], tail))
    null = np.asarray(shuffled)
    return result.assign(null=float(null.mean()), p_value=float(permutation_pvalue(np.array([recall]), null)[0]))
