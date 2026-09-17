"""Well-level quality control."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._reduce import get_matrix, group_codes
from mantispy._core.logging import get_logger
from mantispy._core.masks import feature_mask, reference_mask
from mantispy._core.mutation import inplace_or_copy


@inplace_or_copy()
def well_qc(
    adata: AnnData,
    min_cells: int = 50,
    max_nan_fraction: float = 0.1,
    max_control_cv: float | None = None,
    key: str | None = "selected",
    copy: bool = False,
) -> AnnData | None:
    """Flag wells with too few cells, too much missing data, or unstable controls.

    Args:
        adata: Single-cell object to summarize per well.
        min_cells: Wells with fewer cells than this fail.
        max_nan_fraction: Wells missing more than this fraction of their values fail.
        max_control_cv: For control wells only, the largest acceptable median coefficient of variation across features. ``None`` skips the check, which is the default because a sensible value depends on the assay.
        key: Restrict the statistics to features flagged by this boolean ``var`` column.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes ``uns["mantispy"]["well_qc"]``, a frame with the columns ``Metadata_Plate``, ``Metadata_Well``, ``n_cells``, ``nan_fraction``, ``control_cv`` and ``qc_well_pass``, and broadcasts ``obs["qc_well_pass"]``.

    Notes:
        The table keeps plate and well as columns because a MultiIndex in ``uns`` cannot be written to h5ad.
    """
    selected = feature_mask(adata, key)
    X = get_matrix(adata)[:, selected]
    codes, keys = group_codes(adata, ["Metadata_Plate", "Metadata_Well"])
    is_control = (
        reference_mask(adata, "negcon") if "Metadata_Control" in adata.obs else np.zeros(adata.n_obs, dtype=bool)
    )

    records = []
    for group in range(len(keys)):
        rows = codes == group
        block = X[rows]
        control_cv = np.nan
        if is_control[rows].any() and block.shape[0] > 1:
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.nanmean(block, axis=0)
                deviation = np.nanstd(block, axis=0, ddof=1)
                control_cv = float(np.nanmedian(np.abs(deviation / np.where(mean == 0, np.nan, mean))))
        records.append(
            {
                "n_cells": int(rows.sum()),
                "nan_fraction": float(np.isnan(block).mean()) if block.size else 1.0,
                "control_cv": control_cv,
            }
        )

    table = pd.DataFrame(records)
    table.insert(0, "Metadata_Well", keys.get_level_values(1))
    table.insert(0, "Metadata_Plate", keys.get_level_values(0))

    passed = (table["n_cells"] >= min_cells) & (table["nan_fraction"] <= max_nan_fraction)
    if max_control_cv is not None:
        passed &= table["control_cv"].isna() | (table["control_cv"] <= max_control_cv)
    table["qc_well_pass"] = passed

    adata.uns.setdefault("mantispy", {})["well_qc"] = table
    adata.obs["qc_well_pass"] = passed.to_numpy()[codes]
    get_logger().info("well_qc failed %d of %d wells", int((~passed).sum()), len(table))
    return None
