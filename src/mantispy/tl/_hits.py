"""Hit calling: which perturbations moved away from the controls, and by how much."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._distance import mahalanobis_transform
from mantispy._core._reduce import group_codes, group_offsets, representation
from mantispy._core._stats import benjamini_hochberg, permutation_pvalue, split_reference
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy
from mantispy._core.schema import get_resolution

METHODS = ("mahalanobis", "ks")

#: Scatter estimators :func:`hit_calling` can measure the Mahalanobis distance in.
COVARIANCES = ("empirical", "robust")

#: Blocks (wells) needed on either side of the split before a well-level permutation null can
#: calibrate finely rather than run coarse and conservative. Below this :func:`hit_calling` warns.
_BLOCK_MIN = 16


def ks_statistic(samples: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Two-sample KS statistic of each row of ``samples`` against ``reference``.

    Vectorized over rows, where ``scipy.stats.ks_2samp`` takes one pair at a time.
    The supremum of ``|F_s - F_r|`` is attained at a jump of either step function, and both one-sided differences evaluated at the points of ``samples`` cover every such jump.
    With the reference sorted once, all rows take two ``searchsorted`` calls.

    Args:
        samples: One sample per row; a one-dimensional array is treated as a single row.
        reference: The sample to compare against, sorted here rather than by the caller.

    Returns:
        One statistic per row of ``samples``, or ``NaN`` for every row when either side is empty.
    """
    samples = np.atleast_2d(samples)
    reference = np.sort(reference)
    n, m = samples.shape[1], reference.size
    if n == 0 or m == 0:
        return np.full(samples.shape[0], np.nan)

    ordered = np.sort(samples, axis=1)
    below = np.searchsorted(reference, ordered, side="left") / m
    at_or_below = np.searchsorted(reference, ordered, side="right") / m
    steps = np.arange(1, n + 1) / n
    upper = np.max(steps[None, :] - at_or_below, axis=1)
    lower = np.max(below - (steps - 1.0 / n)[None, :], axis=1)
    return np.maximum(np.maximum(upper, lower), 0.0)


def _statistic(distances: np.ndarray, control_distances: np.ndarray, method: str) -> np.ndarray:
    """One number per row of ``distances``: a median, or a KS statistic."""
    if method == "mahalanobis":
        return np.median(np.atleast_2d(distances), axis=1)
    return ks_statistic(distances, control_distances)


def _block_null(
    to_control: np.ndarray,
    block_codes: np.ndarray,
    pool: np.ndarray,
    tested: np.ndarray,
    against: np.ndarray,
    method: str,
    n_permutations: int,
    seed: int,
    index: int,
) -> np.ndarray:
    """Permutation null that resamples whole blocks (wells) rather than cells.

    Cells within a well share the well, its plate position, seeding and focus, so they are not
    independent replicates: the design's exchangeable unit is the well, and a null that draws
    cells gives the statistic far too little spread. This draws whole blocks instead. The pool is
    the blocks the tested group and the held-out controls span; each permutation takes as many
    blocks as the tested group spans, without replacement, and all of the pool's rows in them.
    """
    pool_blocks = block_codes[pool]
    uniq = np.unique(pool_blocks)
    n_draw = int(np.unique(block_codes[tested]).size)
    # The pool's rows grouped by block, so a draw is the concatenation of a few of these.
    rows_by_block = [pool[pool_blocks == code] for code in uniq]
    generator = np.random.default_rng([seed, index])
    null = np.empty(n_permutations)
    for permutation in range(n_permutations):
        chosen = generator.choice(uniq.size, size=n_draw, replace=False)
        drawn = np.concatenate([rows_by_block[position] for position in chosen])
        null[permutation] = _statistic(to_control[drawn], against, method)[0]
    return null


@inplace_or_copy()
def hit_calling(
    adata: AnnData,
    groupby: str = "Metadata_Perturbation",
    reference: str | None = "negcon",
    method: str = "mahalanobis",
    covariance: str = "empirical",
    use_rep: str | None = None,
    block: str | None = None,
    n_permutations: int = 1000,
    threshold: float = 0.05,
    seed: int = 0,
    key_added: str = "hits",
    copy: bool = False,
) -> AnnData | None:
    """Call hits by testing each group's distance from the controls.

    Each group is scored against the half of the reference rows that did not fit the centroid and
    covariance, and its null is the other ways to draw a group of its size from the group and those
    controls pooled. Under the null the two are exchangeable, so the null is calibrated; bootstrapping
    the controls alone is not, because it is centred on that sample's own median rather than the
    population's and leaves the error in that centre out of its spread.

    This wants a well-replicated design. The statistic is a group's median distance, so a group of one
    or two wells is dominated by whichever wells it holds and no number of permutations recovers that;
    :func:`mantispy.tl.map` with ``mode="activity"`` ranks replicate pairs instead and is the usual
    readout on screens with little replication. JUMP-Target-2 read as a single plate gives every
    compound one well and is the common way to land in that regime, while the same plate map read
    across several of them gives one well per plate.

    Distance from the controls also rises when a treatment kills cells. Read the calls beside a cell
    count, or beside :func:`mantispy.tl.cytotoxicity`, before taking them for morphology.

    Args:
        adata: Object to score, at cell or well resolution.
        groupby: Column defining the groups to test.
        reference: Which rows are the controls. They are split in half, one half to fit the covariance and the other to form the null (see Notes).
        method: ``"mahalanobis"`` scores the median distance of the group's rows from the control centroid, measured in the controls' covariance so that directions the controls already vary in count for less. ``"ks"`` scores the Kolmogorov-Smirnov statistic between the group's and the controls' distance distributions, which detects a shifted subpopulation that leaves the median unchanged. Use it at cell resolution. Its p-value comes from ``scipy.stats.ks_2samp``, so ``n_permutations`` does not apply.
        covariance: Scatter the Mahalanobis distance is measured in. ``"empirical"`` uses every fitting control row. ``"robust"`` uses the minimum covariance determinant subset, so a few stray control wells stop widening the covariance in their own direction and masking real hits there; it needs more control rows than features, so pair it with ``use_rep``.
        use_rep: Score ``obsm[use_rep]`` instead of ``X``. When the covariance-fitting half of the controls has no more rows than there are features, the covariance is singular and a warning suggests a PCA representation.
        block: ``obs`` column whose groups are the design's exchangeable unit, normally the well. The permutation null then draws whole blocks rather than cells, since cells within a well are not independent replicates (see Notes). Left ``None``, it defaults to ``"Metadata_Well"`` on a cell-resolution object that has replicated wells, and stays off otherwise; a well-resolution object already has one row per well and needs no block.
        n_permutations: Size of the permutation null. Applies to ``method="mahalanobis"`` only.
        threshold: q-value below which a group is called a hit in ``is_hit``.
        seed: Seed for the control split and the permutation null.
        key_added: Name for the outputs.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with ``group``, ``n_obs``, ``distance``, ``pvalue``, ``qvalue`` and ``is_hit``, where ``n_obs`` counts the rows the statistic used rather than the rows the group has, and joins ``obs[key_added + "_distance"]`` and ``obs[key_added + "_qvalue"]`` back onto the rows.
        Also writes ``obs[key_added + "_row_distance"]``, each row's own distance from the control centroid rather than its group's.
        That is the column a dose-response fit wants: the group statistic is one number repeated over the group's rows, so the controls show no spread and nothing downstream can read a scale off them.
        Rows that fitted the centroid sit a little closer to it than the held-out controls do, by the same split the Notes describe.
        Which rows those are is written to ``obs[key_added + "_reference_held_out"]``, true for the controls that did not fit, so that a scale taken from the controls can be taken from the honest half.

    Raises:
        ValueError: ``method`` is not one of ``METHODS``, or ``reference`` selects fewer than four rows.

    Notes:
        The controls are split in half.
        One half estimates the centroid and the covariance, and the other supplies the null.
        A null drawn from the rows that defined the centroid would be in-sample while every tested group is out-of-sample, so it would come out too small and call pure noise as hits.
        Because only half the controls fit the covariance, the singular-covariance warning fires when there are fewer than about twice as many controls as features.

        A row that fitted the centroid and the covariance sits closer to the centroid than any other group's row can, so it stays off the tested side of every group.
        The controls carry a perturbation label of their own, and that group is therefore left with the held-out half, which is split once more, at random, so that the rows tested and the rows they are tested against are different rows.
        Its row of the table is a draw from the null rather than a sample compared with part of itself measured against a centroid half of it placed, and ``n_obs`` reports about a quarter of the controls for it.
        Its null is the other ways to halve those rows rather than a bootstrap of them, which would be too wide because the sample is part of what it is drawn from.
        That row is therefore an honest draw from the null: its p-value is uniform and falls below any cutoff about as often as the cutoff says, so a control group appearing in a hit list is this test working rather than a fault.

        The null is drawn from the controls only, and asks whether a group is further out than the same number of control rows would be.
        Drawing from every row would put real hits into the null, and a screen with many hits would look like one with none.

        Calibration degrades with few controls.
        On pure-noise screens of 12 groups of 12 rows with 10 features (the ``pure_noise_screen`` test fixture), the false positive rate at a nominal 0.05 was 0.10 with 48 controls, 0.077 with 192 and 0.052 with 384.
        Across four configurations from 20 to 80 features and 48 to 200 controls, ten seeds each, it was 2.1% overall.
        Neither figure is a bound for another screen, and both were measured with ``method="mahalanobis"``.
        Under ``method="ks"`` there is no permutation null; each group's distances are compared with those of the null half by ``scipy.stats.ks_2samp``.

        At cell resolution the calibration above assumes a null unit that matches the design. Cells within a well share the well, its plate position, seeding and focus, so they are not independent replicates: the exchangeable unit is the well, and a null that shuffles cells shrinks the statistic's spread by the cell count rather than the well count and calls pure noise far above nominal (issue #68). ``block`` fixes this by drawing whole wells; the held-out fit and null split is unchanged, so only the resampling unit moves. It defaults to ``"Metadata_Well"`` on a replicated cell-resolution object, and warns when a cell-resolution object carries no well column. With fewer than about sixteen wells to draw from on either half of the split the well-level null is coarse and runs conservative rather than anti-conservative; for an exact well-level test, aggregate with :func:`~mantispy.tl.aggregate` first.

        To check the rate on your own screen, :func:`~mantispy.metrics.diagnose_testing` relabels control wells as pseudo-treatments of your group sizes and reports the fraction called.
    """
    from scipy.stats import ks_2samp

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if covariance not in COVARIANCES:
        raise ValueError(f"covariance must be one of {COVARIANCES}, got {covariance!r}")

    # At cell resolution the permutation null's exchangeable unit must be the well, not the cell
    # (see Notes). Default it to the well column when the object has one and its wells are replicated;
    # a single cell per well is already well-level, so blocking would be a no-op and stays off.
    if block is None and get_resolution(adata) == "cell":
        if "Metadata_Well" not in adata.obs:
            warnings.warn(
                "the object is at cell resolution and no block was given, so the permutation null draws "
                "single cells. Cells within a well share the well, its plate position, seeding and focus, "
                "so they are not independent replicates and the null is anti-conservative. Pass block= a "
                "well column, or aggregate to wells with mt.tl.aggregate before testing.",
                UserWarning,
                stacklevel=3,
            )
        elif np.bincount(group_codes(adata, "Metadata_Well")[0]).max() > 1:
            block = "Metadata_Well"

    block_codes = group_codes(adata, block)[0] if block is not None else None

    values = representation(adata, use_rep)
    is_control = reference_mask(adata, reference)
    n_control = int(is_control.sum())
    if n_control < 4:
        raise ValueError(
            f"hit calling needs at least four reference rows, and reference={reference!r} selects {n_control}; "
            "half fit the covariance and half form the null. On a consensus object the controls were probably "
            "dropped before mt.tl.consensus; keep them through consensus and drop the control row afterwards."
        )

    generator = np.random.default_rng(seed)
    fit_rows, null_rows = split_reference(np.flatnonzero(is_control), generator)
    if fit_rows.size <= values.shape[1]:
        warnings.warn(
            f"the covariance is estimated from {fit_rows.size} of {n_control} reference rows (half the "
            f"controls) for {values.shape[1]} features, so it is singular and the Mahalanobis distances are "
            "dominated by the regularization. Reduce the features with mt.pp.feature_select, or score a PCA "
            "representation with use_rep='X_pca'.",
            UserWarning,
            stacklevel=3,
        )

    centre, whitening = mahalanobis_transform(values[fit_rows], robust=covariance == "robust", seed=seed)
    # After whitening, the Mahalanobis distance to the control centroid is the norm of the row.
    # Fill gaps before whitening, which mixes columns; one NaN would otherwise zero the row's distance.
    to_control = np.linalg.norm(np.nan_to_num(values - centre) @ whitening, axis=1)
    fitted = np.zeros(adata.n_obs, dtype=bool)
    fitted[fit_rows] = True
    # The split partitions the control rows, so the held-out half is the controls that did not fit.
    held_out = is_control & ~fitted

    if block_codes is not None:
        fit_blocks = np.unique(block_codes[fit_rows]).size
        null_blocks = np.unique(block_codes[null_rows]).size
        if min(fit_blocks, null_blocks) < _BLOCK_MIN:
            warnings.warn(
                f"the well-block null draws from {fit_blocks} block(s) in the fitting half and {null_blocks} "
                "in the null half, too few to calibrate finely, so the test there is conservative rather than "
                "anti-conservative. For an exact well-level test, aggregate to wells with mt.tl.aggregate "
                "before testing.",
                UserWarning,
                stacklevel=3,
            )

    codes, keys = group_codes(adata, groupby)

    observed = np.empty(len(keys))
    sizes = np.empty(len(keys), dtype=int)
    pvalues = np.empty(len(keys))
    null = np.empty((len(keys), n_permutations))
    order, offsets = group_offsets(codes, len(keys))
    for index in range(len(keys)):
        rows = order[offsets[index] : offsets[index + 1]]
        # A row that fitted the centroid and the covariance sits closer to the centroid than one that did not, so it stays off the tested side of every group.
        keep = ~fitted[rows]
        # The controls carry a perturbation label of their own, so one group is the reference against itself.
        # Half of its held-out rows are the sample and half are what it is tested against, drawn at random because the rows are ordered by plate and well.
        shared = np.flatnonzero(held_out[rows])
        keep[generator.permutation(shared)[: shared.size // 2]] = False
        tested = rows[keep]
        # A row is never on both sides: the reference group's sample comes out of the held-out rows, so it is tested against the rest of them, and every other group is tested against all of them.
        in_sample = np.zeros(adata.n_obs, dtype=bool)
        in_sample[tested] = True
        against_rows = null_rows[~in_sample[null_rows]]
        against = to_control[against_rows]
        sizes[index] = tested.size
        observed[index] = _statistic(to_control[tested], against, method)[0]
        if method == "ks":
            # The KS statistic has a known null distribution.
            # A permutation null drawn from the reference rows is too small, since each draw is a subset of its own reference, and a further split cannot supply pseudo-groups as large as the real ones.
            pvalues[index] = float(ks_2samp(to_control[tested], against).pvalue)
            continue
        # Under the null this group is exchangeable with the controls it is measured against, so the null is the other ways to draw a group of its size from the two of them pooled.
        # Bootstrapping the controls alone instead centres the null on that sample's own median rather than the population's, and leaves the error in that centre out of the spread, so the observed lands in the tail more often than it should: the rate goes as the one-sided tail of z / sqrt(1 + tested/held-out), which called 9 to 11% of pure noise at 24 rows against 48 held-out controls.
        # Drawing without replacement from the controls alone is worse still, since it narrows the null further.
        pool = np.concatenate([tested, against_rows])
        if block_codes is not None:
            # Cells within a well are pseudoreplicates, so the null resamples whole wells (issue #68).
            null[index] = _block_null(
                to_control, block_codes, pool, tested, against, method, n_permutations, seed, index
            )
            continue
        spread = np.random.default_rng([seed, index]).random((n_permutations, pool.size))
        draws = pool[np.argsort(spread, axis=1)[:, : max(tested.size, 1)]]
        null[index] = _statistic(to_control[draws], against, method)

    # A group of one or two rows has a median dominated by whichever well it happens to hold, and the null
    # cannot separate that from the controls however many permutations it draws. JUMP-Target-2 read as a
    # single plate is the common way to land here: every compound has one well, and the calls that come back
    # track cell loss rather than morphology.
    thin = int(np.sum(sizes < 3))
    if thin > len(keys) // 2:
        warnings.warn(
            f"{thin} of {len(keys)} groups have fewer than three rows, so their statistic is the median of "
            "one or two wells. Aggregate more replicates, or score activity with mt.tl.map(mode='activity'), "
            "which ranks replicate pairs and is built for screens with little replication.",
            UserWarning,
            stacklevel=3,
        )

    if method != "ks":
        pvalues = permutation_pvalue(observed, null)
    qvalues = benjamini_hochberg(pvalues)
    table = pd.DataFrame(
        {
            "group": [str(key) for key in keys],
            "n_obs": sizes,
            "distance": observed,
            "pvalue": pvalues,
            "qvalue": qvalues,
            "is_hit": qvalues < threshold,
        }
    )

    adata.uns.setdefault("mantispy", {})[key_added] = table
    lookup = table.set_index("group")
    labels = as_frame(adata.obs)[groupby].astype(str)
    adata.obs[f"{key_added}_distance"] = lookup["distance"].reindex(labels).to_numpy()
    adata.obs[f"{key_added}_qvalue"] = lookup["qvalue"].reindex(labels).to_numpy()
    # The group statistic broadcast over its rows says nothing about how one well differs from another, so anything
    # that reads a response per row, a dose-response fit above all, has no spread to work with. The row's own
    # distance is already computed here.
    adata.obs[f"{key_added}_row_distance"] = to_control
    adata.obs[f"{key_added}_reference_held_out"] = held_out
    get_logger().info("hit_calling(%s) called %d of %d groups", method, int(table["is_hit"].sum()), len(table))
    return None
