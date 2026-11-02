"""Centre or standardise each feature within a group, such as a well."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from mantispy._core._reduce import get_matrix, group_codes
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy


@inplace_or_copy()
def center_within(
    adata: AnnData,
    by: str = "Metadata_Well",
    reference: str | None = None,
    method: str = "subtract",
    layer: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Centre each feature within each group, optionally on a reference subset.

    Well-to-well offsets move every feature together, so subtracting each well's own mean (or the mean of
    its controls) puts the cells of every well on a common origin before they are compared. With
    ``method="zscore"`` each feature is also divided by its spread within the group.

    Args:
        adata: The object to transform.
        by: ``obs`` column whose groups are centred separately, normally the well.
        reference: Which rows set each group's centre: ``None`` for every row of the group, ``"negcon"`` or
            ``"poscon"`` for a control class, or the name of a boolean ``obs`` column.
        method: ``"subtract"`` to centre only, or ``"zscore"`` to also divide by the within-group spread.
        layer: Transform this layer instead of ``X``.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. A group with fewer than two reference rows, and a constant feature
        under ``"zscore"``, are left unchanged rather than divided by zero.

    Raises:
        ValueError: ``method`` is neither ``"subtract"`` nor ``"zscore"``.
    """
    if method not in ("subtract", "zscore"):
        raise ValueError(f"method must be 'subtract' or 'zscore', got {method!r}")

    values = get_matrix(adata, layer).astype(np.float64)
    reference_rows = reference_mask(adata, reference)
    codes, keys = group_codes(adata, by)
    for index in range(len(keys)):
        rows = np.flatnonzero(codes == index)
        anchor = rows[reference_rows[rows]]
        if anchor.size < 2:
            continue
        values[rows] -= np.nanmean(values[anchor], axis=0)
        if method == "zscore":
            spread = np.nanstd(values[anchor], axis=0)
            values[rows] /= np.where(spread > 0, spread, 1.0)

    result = values.astype(np.float32)
    if layer is None:
        adata.X = result
    else:
        adata.layers[layer] = result
    return None
