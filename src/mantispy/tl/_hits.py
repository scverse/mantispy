"""Hit calling: which perturbations moved away from the controls, and by how much."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._distance import mahalanobis_transform
from mantispy._core._reduce import group_codes, group_offsets, representation
from mantispy._core._stats import (
    _default_replicate,
    _is_object_resolution,
    _split_wells,
    benjamini_hochberg,
    permutation_pvalue,
    split_reference,
)
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy

METHODS = ("mahalanobis", "ks", "hellinger", "mp_value")

COVARIANCES = ("empirical", "robust")

_REPLICATE_MIN = 16


def ks_statistic(samples: np.ndarray, reference: np.ndarray, *, reference_sorted: bool = False) -> np.ndarray:
    """Two-sample KS statistic of each row of ``samples`` against ``reference``.

    Vectorized over rows, where ``scipy.stats.ks_2samp`` takes one pair at a time.
    The supremum of ``|F_s - F_r|`` is attained at a jump of either step function, and both one-sided differences evaluated at the points of ``samples`` cover every such jump.
    With the reference sorted once, all rows take two ``searchsorted`` calls.

    Args:
        samples: One sample per row; a one-dimensional array is treated as a single row.
        reference: The sample to compare against, sorted here rather than by the caller.
        reference_sorted: Skip sorting ``reference``, for a caller that passes the same sorted reference many times (a permutation null).

    Returns:
        One statistic per row of ``samples``, or ``NaN`` for every row when either side is empty.
    """
    samples = np.atleast_2d(samples)
    reference = reference if reference_sorted else np.sort(reference)
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


def _replicate_null(
    to_control: np.ndarray,
    replicate_codes: np.ndarray,
    pool: np.ndarray,
    tested: np.ndarray,
    n_permutations: int,
    seed: int,
    index: int,
) -> np.ndarray:
    """Permutation null that resamples whole replicates (normally wells) rather than cells.

    The pool is the replicates the tested group and the held-out controls span; each permutation draws as many whole replicates as the tested group spans, without replacement, and takes all of the pool's rows in them.
    Only ``method="mahalanobis"`` reaches here, so the statistic is the median distance.
    """
    pool_replicates = replicate_codes[pool]
    uniq = np.unique(pool_replicates)
    n_draw = int(np.unique(replicate_codes[tested]).size)
    dist_by_replicate = [to_control[pool[pool_replicates == code]] for code in uniq]
    generator = np.random.default_rng([seed, index])
    null = np.empty(n_permutations)
    for permutation in range(n_permutations):
        chosen = generator.choice(uniq.size, size=n_draw, replace=False)
        null[permutation] = float(np.median(np.concatenate([dist_by_replicate[position] for position in chosen])))
    return null


def _ks_replicate_null(
    values: np.ndarray,
    replicate_codes: np.ndarray,
    pool: np.ndarray,
    tested: np.ndarray,
    reference: np.ndarray,
    n_permutations: int,
    seed: int,
    index: int,
) -> np.ndarray:
    """Replicate permutation null for the KS statistic.

    Like :func:`_replicate_null` the pool is the replicates the tested group and the held-out controls span, and each permutation draws as many whole replicates as the tested group spans; the statistic recomputed per draw is the KS distance of the drawn rows against the fixed ``reference`` rather than a median.
    """
    pool_replicates = replicate_codes[pool]
    uniq = np.unique(pool_replicates)
    n_draw = int(np.unique(replicate_codes[tested]).size)
    value_by_replicate = [values[pool[pool_replicates == code]] for code in uniq]
    reference = np.sort(reference)  # fixed across draws, so sort it once rather than inside ks_statistic
    generator = np.random.default_rng([seed, index])
    null = np.empty(n_permutations)
    for permutation in range(n_permutations):
        chosen = generator.choice(uniq.size, size=n_draw, replace=False)
        drawn = np.concatenate([value_by_replicate[position] for position in chosen])
        null[permutation] = float(ks_statistic(drawn[None, :], reference, reference_sorted=True)[0])
    return null


def _hellinger_null(
    x: np.ndarray,
    tested: np.ndarray,
    against: np.ndarray,
    replicate_codes: np.ndarray | None,
    settings: tuple,
    min_against: int,
    n_permutations: int,
    seed: int,
    index: int,
) -> tuple[float, np.ndarray]:
    """Observed Hellinger distance or mp-value statistic and its null, from the same kernel so that the observed draw ties with itself.

    The null draws as many whole replicates as the tested group spans, or as many rows without a replicate, from the tested and reference rows pooled, against the rest of them.
    """
    import numba

    from mantispy._core import _hellinger

    pool = np.concatenate([tested, against])
    if replicate_codes is not None:
        units = np.unique(replicate_codes[pool], return_inverse=True)[1].astype(np.int64)
        observed_draw = np.unique(units[: tested.size])
    else:
        units, observed_draw = np.arange(pool.size, dtype=np.int64), np.arange(tested.size)
    n_units, n_draw = (int(units.max()) + 1 if pool.size else 0), observed_draw.size
    # A side too small for its covariance leaves the group unscored, so there is no null to draw.
    min_rows = settings[-1]
    if n_draw == 0 or n_draw >= n_units or tested.size < min_rows or against.size < min_against:
        return np.nan, np.full(n_permutations, np.nan)
    generator = np.random.default_rng([seed, index])
    draws = np.stack(
        [observed_draw] + [generator.choice(n_units, n_draw, replace=False) for _ in range(n_permutations)]
    )
    threads = numba.get_num_threads()
    # ponytail: wide matrices run the permutations on one thread, because concurrent LAPACK calls on them crashed the
    # BLAS with numba's threads; a reduced use_rep, which the method wants anyway, keeps the parallel path.
    if x.shape[1] > 64:
        numba.set_num_threads(1)
    try:
        out = _hellinger.null(np.ascontiguousarray(x[pool]), units, draws, *settings)
    finally:
        numba.set_num_threads(threads)
    return float(out[0]), out[1:]


@inplace_or_copy()
def hit_calling(
    adata: AnnData,
    groupby: str = "Metadata_Perturbation",
    reference: str | None = "negcon",
    method: str = "mahalanobis",
    covariance: str = "empirical",
    use_rep: str | None = None,
    replicate: str | Sequence[str] | None = None,
    n_permutations: int = 1000,
    threshold: float = 0.05,
    seed: int = 0,
    key_added: str = "hits",
    copy: bool = False,
) -> AnnData | None:
    """Call hits by testing each group's distance from the controls.

    Each group is scored against the half of the reference rows that did not fit the centroid and covariance, and its null is the other ways to draw a group of its size from the group and those controls pooled.
    Under the null the two are exchangeable, so the null is calibrated; bootstrapping the controls alone is not, because it is centred on that sample's own median rather than the population's and leaves the error in that centre out of its spread.

    This calibrates each group against control wells; :func:`mantispy.tl.empirical_fdr` instead calibrates a per-gene score against control genes that cannot respond, for a genetic screen where non-expressed genes are the better null.

    This wants a well-replicated design.
    The statistic is a group's median distance, so a group of one or two wells is dominated by whichever wells it holds and no number of permutations recovers that; :func:`mantispy.tl.map` with ``mode="activity"`` ranks replicate pairs instead and is the usual readout on screens with little replication.
    JUMP-Target-2 read as a single plate gives every compound one well and is the common way to land in that regime, while the same plate map read across several of them gives one well per plate.

    Distance from the controls also rises when a treatment kills cells.
    Read the calls beside a cell count, or beside :func:`mantispy.tl.cytotoxicity`, before taking them for morphology.

    Args:
        adata: Object to score, at cell or well resolution.
        groupby: Column defining the groups to test.
        reference: Which rows are the controls.
            They are split in half, one half to fit the covariance and the other to form the null (see Notes).
        method: ``"mahalanobis"`` scores the median distance of the group's rows from the control centroid, measured in the controls' covariance so that directions the controls already vary in count for less.
            ``"ks"`` scores the Kolmogorov-Smirnov statistic between the group's and the controls' distance distributions, which detects a shifted subpopulation that leaves the median unchanged.
            Use it at cell resolution.
            Its p-value comes from a permutation null over whole replicates when one is in use (see ``replicate``), and otherwise from ``scipy.stats.ks_2samp``.
            ``"hellinger"`` fits a Gaussian to the group's rows and one to the controls', and scores the Hellinger distance between the two, so a change in the group's own spread or shape counts as well as a shift of its centre; a treatment that sends only some cells into a new state widens the group without moving its median.
            It needs each side to have at least twice as many rows as features, so use it at cell resolution on a reduced ``use_rep``; a group short of that is left unscored with a warning.
            It is opt-in; the default stays ``"mahalanobis"``.
            A single Gaussian describes a cloud with several cell states poorly, so on a real screen it can call fewer groups than ``"mahalanobis"`` or ``"ks"``; read it beside them rather than instead of them.
            With ``covariance="robust"`` it is the robust Hellinger distance of BioProfiling.jl (see Notes), which resists stray cells but trims a partial responder and runs far slower.
            ``"mp_value"`` scores the squared Mahalanobis distance between the group's centroid and the controls', measured in the average of the two sides' covariances, the multidimensional perturbation value of Hutz et al. (see Notes).
            Unlike ``"mahalanobis"`` it counts the group's own spread, so a group whose replicates disagree is measured in a wider covariance and moves less.
            The average is positive definite whenever the controls' covariance is, so the group needs only two rows and the controls more rows than features; it works on well profiles with a few replicates, where ``"hellinger"`` cannot.
            A group with fewer rows is left unscored with a warning.
        covariance: Scatter the Mahalanobis distance is measured in.
            ``"empirical"`` uses every fitting control row.
            ``"robust"`` uses the minimum covariance determinant subset, so a few stray control wells stop widening the covariance in their own direction and masking real hits there; it needs more control rows than features, so pair it with ``use_rep``.
            Under ``method="hellinger"`` and ``method="mp_value"`` it is the location and scatter both sides are fitted with, refitted on every permutation; under ``method="mp_value"`` the group then needs more rows than features too.
        use_rep: Score ``obsm[use_rep]`` instead of ``X``.
            When the covariance-fitting half of the controls has no more rows than there are features, the covariance is singular and a warning suggests a PCA representation.
        replicate: ``obs`` column, or sequence of columns, whose groups are the design's exchangeable unit: the well in an arrayed screen, the guide (``"Metadata_sgRNA"``) in a pooled one.
            The permutation null then draws whole replicates rather than cells, since cells within a well are not independent replicates (see Notes).
            Left ``None``, it defaults to the physical well, the ``(Metadata_Plate, Metadata_Well)`` pair, on an object explicitly stamped cell resolution that has replicated wells; a same well name on two plates is then two replicates, not one.
            On such an object where some well holds both reference and tested cells, a pooled design, there is no default and it raises instead.
            It stays off when the object is not stamped cell resolution and when it has no complete well column, in which case the cell path warns; under ``method="ks"`` without a replicate the p-value then falls back to ``scipy.stats.ks_2samp``.
        n_permutations: Size of the permutation null.
            Applies to ``method="mahalanobis"``, ``method="hellinger"`` and ``method="mp_value"``, and to ``method="ks"`` when a replicate is in use.
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
        ValueError: ``method`` is not one of ``METHODS``, ``reference`` selects fewer than four rows, or ``replicate`` is not given on a pooled design at cell resolution.

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
        On pure-noise screens the false positive rate at a nominal 0.05 is inflated with few controls and approaches the nominal rate as controls are added.
        This is not a bound for another screen, and it was checked with ``method="mahalanobis"``.
        Under ``method="ks"`` with a replicate the null draws whole replicates for the pseudo-treated side and recomputes the KS statistic, the same unit ``replicate`` fixes for the median; without one each group's distances are compared with those of the null half by ``scipy.stats.ks_2samp``.

        At cell resolution the calibration above assumes a null unit that matches the design.
        Cells within a well share the well, its plate position, seeding and focus, so they are not independent replicates: the exchangeable unit is the well, and a null that shuffles cells shrinks the statistic's spread by the cell count rather than the well count and calls pure noise far above nominal.
        ``replicate`` fixes this by drawing whole wells.
        The controls are then split into whole wells too, one half fitting the centroid and covariance and the other forming the null, so every well in the null pool is complete and the reference group draws whole wells like any other group.
        ``replicate`` defaults to the physical well, the ``(Metadata_Plate, Metadata_Well)`` pair, on an object explicitly stamped cell resolution with replicated wells; it warns and leaves the cell-level test in place when no complete well column is present.
        In a pooled screen every well holds every perturbation, so drawing whole wells cannot separate a group from the reference; the exchangeable unit is the guide, and the call raises until ``replicate`` names it.
        With fewer than about sixteen wells in the null half the well-level null is coarse and runs conservative rather than anti-conservative; for an exact well-level test, aggregate with :func:`~mantispy.tl.aggregate` first.

        To check the rate on your own screen, :func:`~mantispy.metrics.diagnose_testing` relabels control wells as pseudo-treatments of your group sizes and reports the fraction called.

        ``method="hellinger"`` follows the robust morphological perturbation value of BioProfiling.jl (Vulliard et al., Bioinformatics, 2022).
        It does not split the controls, because both Gaussians are refitted on every draw and no row is spent on a fixed centroid: a group is tested against every control row it does not hold, and its null draws whole replicates, or rows without a replicate, from the two pooled.
        Rows with a missing value are left out of both fits.
        With ``covariance="robust"`` each fit is the reweighted minimum covariance determinant estimate with the defaults of R robustbase's ``covMcd``, which BioProfiling.jl calls, so the distance agrees with BioProfiling.jl's wherever the two searches reach the same subset.
        That robustness costs a partial responder: the MCD fits the most concentrated half of a side's rows, so a responding subpopulation smaller than half of the group is trimmed as outlying, where the empirical fit keeps it.
        The robust fit is also refitted on every permutation, which makes it far slower than the empirical one; the empirical fit is the default for that reason.

        ``method="mp_value"`` follows the multidimensional perturbation value of Hutz et al. (J Biomol Screen, 2013), with the covariance averaged over the group and the controls as its Methods describe; cytominer-eval's ``mp_value`` uses the controls' covariance alone.
        Like ``method="hellinger"`` it does not split the controls, and its null draws whole replicates, or rows without a replicate, from the group and the controls pooled.
        Hutz et al. first project each group and the controls onto the principal components that carry 90% of their variance; that step is left out, and ``use_rep`` sets the representation as for the other methods.
    """
    from scipy.stats import ks_2samp

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if covariance not in COVARIANCES:
        raise ValueError(f"covariance must be one of {COVARIANCES}, got {covariance!r}")

    is_control = reference_mask(adata, reference)
    replicate_codes = _default_replicate(adata, replicate=replicate, reference=is_control)
    if replicate_codes is None and replicate is None and _is_object_resolution(adata):
        warnings.warn(
            "replicate=None at cell resolution treats cells as independent replicates, which makes p-values too "
            "small. Pass replicate= a well column, or aggregate with mt.tl.aggregate.",
            UserWarning,
            stacklevel=3,
        )

    values = representation(adata, use_rep)
    n_control = int(is_control.sum())
    if n_control < 4:
        raise ValueError(
            f"hit calling needs at least four reference rows, and reference={reference!r} selects {n_control}; "
            "half fit the covariance and half form the null. On a consensus object the controls were probably "
            "dropped before mt.tl.consensus; keep them through consensus and drop the control row afterwards."
        )

    generator = np.random.default_rng(seed)
    control_rows = np.flatnonzero(is_control)
    control_replicates = (
        np.unique(replicate_codes[control_rows]) if replicate_codes is not None else np.empty(0, dtype=int)
    )
    refit = method in ("hellinger", "mp_value")
    if refit:
        # Both sides are refitted on every draw, so no control is spent on a fixed centroid.
        fit_rows = null_rows = control_rows
    elif replicate_codes is not None and control_replicates.size >= 4:
        fit_replicates = split_reference(control_replicates, generator)[0]
        in_fit = np.isin(replicate_codes[control_rows], fit_replicates)
        fit_rows, null_rows = control_rows[in_fit], control_rows[~in_fit]
    else:
        fit_rows, null_rows = split_reference(control_rows, generator)
        replicate_codes = None
    if not refit and fit_rows.size <= values.shape[1]:
        get_logger().warning(
            "hit_calling: covariance from %d of %d reference rows for %d features is singular",
            fit_rows.size,
            n_control,
            values.shape[1],
            deep="reduce features with mt.pp.feature_select, or score use_rep='X_pca'",
        )

    if refit and covariance == "robust":
        # The row distances use the same MCD the statistic does, rather than scikit-learn's.
        from mantispy._core._mcd import cov_mcd

        reference_rows = values[fit_rows]
        centre, scatter = cov_mcd(reference_rows[~np.isnan(reference_rows).any(axis=1)], seed=seed)
        eigenvalues, eigenvectors = np.linalg.eigh(scatter)
        whitening = eigenvectors @ np.diag(1.0 / np.sqrt(np.clip(eigenvalues, 1e-6, None))) @ eigenvectors.T
    else:
        centre, whitening = mahalanobis_transform(values[fit_rows], robust=covariance == "robust", seed=seed)
    # Fill gaps before whitening, which mixes columns; one NaN would otherwise zero the row's distance.
    to_control = np.linalg.norm(np.nan_to_num(values - centre) @ whitening, axis=1)
    fitted = np.zeros(adata.n_obs, dtype=bool)
    if not refit:
        fitted[fit_rows] = True
    held_out = is_control & ~fitted
    if refit:
        from scipy.stats import chi2

        from mantispy._core._mcd import N_SAMP, mcd_constants

        x = np.ascontiguousarray(values, dtype=np.float64)
        complete = ~np.isnan(x).any(axis=1)
        robust = covariance == "robust"
        # MCD constants by sample size, looked up inside numba where scipy is unavailable; unused for the empirical fit.
        sizes_up_to = np.arange(adata.n_obs + 1) if robust else np.zeros(1, dtype=np.int64)
        tables = mcd_constants(sizes_up_to, x.shape[1])
        n_features = x.shape[1]
        if method == "hellinger":
            # BioProfiling leaves a side with fewer than twice as many rows as features unscored.
            min_rows = min_against = 2 * n_features
        else:
            # The pooled covariance only needs the controls' to be full rank; the group needs two rows for a covariance,
            # or more rows than features for an MCD fit.
            min_rows, min_against = (n_features + 1 if robust else 2), n_features + 1
        cutoff = float(chi2.ppf(0.975, n_features))
        settings = (method == "mp_value", robust, *tables, cutoff, N_SAMP, seed, min_rows)

    if replicate_codes is not None:
        # Only the null half enters the permutation pool, so it alone sets how finely the null calibrates.
        null_replicates = np.unique(replicate_codes[null_rows]).size
        if null_replicates < _REPLICATE_MIN:
            where = "the reference" if refit else "the null half"
            get_logger().warning(
                "hit_calling: the permutation test draws from only %d replicate(s) in %s, so its p-values can take only a few values",
                null_replicates,
                where,
                deep="the test is conservative, and aggregating to wells with mt.tl.aggregate gives an exact test",
            )

    codes, keys = group_codes(adata, groupby)

    observed = np.empty(len(keys))
    sizes = np.empty(len(keys), dtype=int)
    pvalues = np.empty(len(keys))
    null = np.empty((len(keys), n_permutations))
    order, offsets = group_offsets(codes, len(keys))
    for index in range(len(keys)):
        rows = order[offsets[index] : offsets[index + 1]]
        keep = ~fitted[rows]
        # The reference group's held-out rows are halved at random because rows are ordered by plate and well.
        shared = np.flatnonzero(held_out[rows])
        if replicate_codes is not None:
            keep[shared[_split_wells(replicate_codes[rows[shared]], generator)]] = False
        else:
            keep[generator.permutation(shared)[: shared.size // 2]] = False
        tested = rows[keep]
        # A row is never on both sides: the reference's sample comes out of the held-out rows.
        in_sample = np.zeros(adata.n_obs, dtype=bool)
        in_sample[tested] = True
        against_rows = null_rows[~in_sample[null_rows]]
        against = to_control[against_rows]
        sizes[index] = tested.size
        if refit:
            tested, against_rows = tested[complete[tested]], against_rows[complete[against_rows]]
            sizes[index] = tested.size
            observed[index], null[index] = _hellinger_null(
                x, tested, against_rows, replicate_codes, settings, min_against, n_permutations, seed, index
            )
            continue
        observed[index] = _statistic(to_control[tested], against, method)[0]
        # The pool must include the tested rows; bootstrapping the controls alone is anti-conservative.
        pool = np.concatenate([tested, against_rows])
        if method == "ks":
            if replicate_codes is not None:
                null[index] = _ks_replicate_null(
                    to_control, replicate_codes, pool, tested, against, n_permutations, seed, index
                )
                pvalues[index] = float(permutation_pvalue(observed[[index]], null[[index]])[0])
            else:
                # A permutation null from the reference rows is too small here, since each draw is a subset of its own reference.
                pvalues[index] = float(ks_2samp(to_control[tested], against).pvalue)
            continue
        if replicate_codes is not None:
            null[index] = _replicate_null(to_control, replicate_codes, pool, tested, n_permutations, seed, index)
            continue
        spread = np.random.default_rng([seed, index]).random((n_permutations, pool.size))
        draws = pool[np.argsort(spread, axis=1)[:, : max(tested.size, 1)]]
        null[index] = _statistic(to_control[draws], against, method)

    if refit and np.isnan(observed).any():
        unscored = [str(key) for key, value in zip(keys, observed, strict=True) if np.isnan(value)]
        if method == "hellinger":
            reason = "a side had fewer than twice as many complete rows as features, or a singular covariance"
        else:
            reason = (
                f"the group had fewer than {min_rows} complete rows, the reference no more than features, or the "
                "averaged covariance was singular"
            )
        get_logger().warning(
            "hit_calling: %d group(s) left unscored (%s)",
            len(unscored),
            ", ".join(unscored[:5]) + (", ..." if len(unscored) > 5 else ""),
            deep=f"{reason}, so score a reduced representation with use_rep='X_pca'",
        )
    thin = int(np.sum(sizes < 3))
    if not refit and thin > len(keys) // 2:
        get_logger().warning(
            "hit_calling: %d of %d groups have fewer than three rows",
            thin,
            len(keys),
            deep="their statistic is the median of one or two wells, so aggregate more replicates or use "
            "mt.tl.map(mode='activity')",
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
    adata.obs[f"{key_added}_row_distance"] = to_control
    adata.obs[f"{key_added}_reference_held_out"] = held_out
    get_logger().info("hit_calling(%s) called %d of %d groups", method, int(table["is_hit"].sum()), len(table))
    return None
