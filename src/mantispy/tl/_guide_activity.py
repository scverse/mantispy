"""Score each guide for phenotypic activity against a reference control class."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._reduce import get_matrix
from mantispy._core._stats import MAD_TO_SIGMA
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.mutation import inplace_or_copy


@inplace_or_copy()
def guide_activity(
    adata: AnnData,
    *,
    reference: str,
    group: str = "Metadata_Gene",
    use_rep: str | None = None,
    layer: str | None = None,
    key_added: str = "guide_activity",
    copy: bool = False,
) -> AnnData | None:
    """Score each guide for how far it sits from a reference control class, as a one-sided p-value.

    This produces the per-guide score :func:`~mantispy.tl.aggregate_guides` reads: a one-sided p-value, small when the guide shows a phenotype.
    Each feature is standardized by the reference guides' median and median absolute deviation, so a guide's activity is the size of its standardized profile, how far it moves from the reference centre in control units.
    The p-value is the share of reference guides whose activity reaches the guide's or beyond, with one added to the count and the total, so it is one-sided and small for a strong phenotype.

    The reference class sets the scale and is not scored, so for :func:`~mantispy.tl.aggregate_guides` the reference here and the ``control`` null there must be two different control classes.
    A pooled screen that carries both intergenic and non-targeting guides can set the scale with one and keep the other as the null, so the calibration is not circular.

    Args:
        adata: One row per guide, with the class label in ``obs`` and the profiles in ``X`` (or `use_rep`/`layer`).
        reference: The value of `group` that marks the reference control guides, such as ``"intergenic"``.
        group: ``obs`` column holding the class label, the gene or control name per guide.
        use_rep: Read ``obsm[use_rep]`` instead of ``X``; cannot be combined with `layer`.
        layer: Read this layer instead of ``X``.
        key_added: ``obs`` column the per-guide p-value is written to.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``obs[key_added]``, the one-sided per-guide p-value, with the reference guides left missing since they set the scale rather than being scored.

    Raises:
        ValueError: `reference` is absent from ``obs[group]``, or every feature is constant across the reference guides.
        KeyError: `group` is not an ``obs`` column.

    Notes:
        A feature that does not vary across the reference guides carries no information and is dropped, so a profile of all-constant features cannot be scored.
    """
    obs = as_frame(adata.obs)
    if group not in obs:
        raise KeyError(f"group={group!r} is not an obs column")
    labels = obs[group].astype(str).to_numpy()
    is_reference = labels == str(reference)
    if not is_reference.any():
        raise ValueError(f"reference={reference!r} is absent from obs[{group!r}]")

    x = np.asarray(get_matrix(adata, layer=layer, use_rep=use_rep), dtype=np.float64)
    ref = x[is_reference]
    median = np.nanmedian(ref, axis=0)
    mad = MAD_TO_SIGMA * np.nanmedian(np.abs(ref - median), axis=0)
    varying = mad > 1e-9
    if not varying.any():
        raise ValueError(f"every feature is constant across the {reference!r} guides, so there is no scale to score against")

    z = (x[:, varying] - median[varying]) / mad[varying]
    activity = np.sqrt(np.nanmean(z**2, axis=1))

    ref_sorted = np.sort(activity[is_reference])
    at_or_above = ref_sorted.size - np.searchsorted(ref_sorted, activity, side="left")
    pvalue = (at_or_above + 1) / (ref_sorted.size + 1)
    pvalue[is_reference] = np.nan  # the reference sets the scale, it is not scored

    adata.obs[key_added] = pvalue
    get_logger().info(
        "guide_activity scored %d guides against %d %r reference guides",
        int((~is_reference).sum()),
        int(is_reference.sum()),
        reference,
    )
    return None
