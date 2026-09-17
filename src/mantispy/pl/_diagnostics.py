"""Diagnostic plots for plate position, image quality, control drift and outliers.

These plots only diagnose; the corrections are in :mod:`mantispy.pp`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mantispy._core._reduce import get_matrix
from mantispy._core._utils import as_frame, reference_mask
from mantispy._core.plate import well_col, well_row

if TYPE_CHECKING:
    from anndata import AnnData


def _feature_values(adata: AnnData, feature: str | None) -> np.ndarray:
    """One number per row: a named feature, or the mean across features."""
    matrix = get_matrix(adata)
    if feature is not None:
        return matrix[:, adata.var_names.get_loc(feature)].astype(float)
    with np.errstate(invalid="ignore"):
        return np.nanmean(matrix, axis=1)


def plate_effects(adata: AnnData, feature: str | None = None, axes: np.ndarray | None = None):
    """Row and column medians per plate, for spotting plate position artifacts.

    Args:
        adata: Object to draw, at any resolution.
        feature: A single feature, or ``None`` for the mean across features.
        axes: A ``(n_plates, 2)`` array of axes to draw into.

    Returns:
        The axes array, one row per plate, with the row marginal on the left and the column marginal on the
        right, each with the plate median drawn as a reference line.
    """
    values = _feature_values(adata, feature)
    frame = pd.DataFrame(
        {
            "plate": adata.obs["Metadata_Plate"].astype(str).to_numpy(),
            "row": [well_row(well) for well in adata.obs["Metadata_Well"]],
            "col": [well_col(well) for well in adata.obs["Metadata_Well"]],
            "value": values,
        }
    )
    plates = sorted(frame["plate"].unique())
    if axes is None:
        _, axes = plt.subplots(len(plates), 2, figsize=(9, 3 * len(plates)), squeeze=False)

    for index, plate in enumerate(plates):
        block = frame[frame["plate"] == plate]
        reference = block["value"].median()
        for position, axis_name in enumerate(("row", "col")):
            axis = axes[index, position]
            marginal = block.groupby(axis_name)["value"].median()
            axis.plot(marginal.index, marginal.to_numpy(), marker="o", ms=3)
            axis.axhline(reference, color="grey", ls="--", lw=1)
            axis.set_xlabel(f"plate {axis_name}")
            axis.set_title(f"{plate} by {axis_name}", fontsize=9)
    axes[0, 0].set_ylabel(feature or "mean feature value")
    return axes


def image_qc(adata: AnnData, ax: plt.Axes | None = None):
    """Image quality score per image, with the flagged images marked."""
    if "image_qc" not in adata.uns.get("mantispy", {}):
        raise KeyError("uns['mantispy']['image_qc'] is missing; run mt.pp.image_qc first")
    table = pd.DataFrame(adata.uns["mantispy"]["image_qc"])

    ax = ax or plt.subplots(figsize=(8, 4))[1]
    failed = ~table["qc_image_pass"].to_numpy(dtype=bool)
    positions = np.arange(len(table))
    ax.scatter(positions[~failed], table["qc_image_score"].to_numpy()[~failed], s=6, label="pass")
    ax.scatter(positions[failed], table["qc_image_score"].to_numpy()[failed], s=18, color="crimson", label="flagged")
    ax.set_xlabel("image")
    ax.set_ylabel("quality score")
    ax.legend(fontsize=7)
    return ax


def control_drift(
    adata: AnnData,
    groupby: str = "Metadata_Plate",
    n_components: int = 2,
    ax: plt.Axes | None = None,
):
    """Control wells projected onto principal components fitted on the controls alone.

    Fitting on the controls alone shows how the reference moves between plates or batches,
    which is the drift normalization should remove.
    """
    from sklearn.decomposition import PCA

    is_control = reference_mask(adata, "negcon")
    if is_control.sum() < n_components + 1:
        raise ValueError(f"need more than {n_components} control rows, found {int(is_control.sum())}")

    controls = np.nan_to_num(get_matrix(adata)[is_control], nan=0.0)
    embedding = PCA(n_components=n_components).fit_transform(controls)
    labels = adata.obs[groupby].astype(str).to_numpy()[is_control]

    ax = ax or plt.subplots(figsize=(5, 4))[1]
    for group in pd.unique(labels):
        selected = labels == group
        ax.scatter(embedding[selected, 0], embedding[selected, 1], s=12, label=str(group))
    ax.set_xlabel("control PC1")
    ax.set_ylabel("control PC2")
    ax.legend(title=groupby, fontsize=6, title_fontsize=7)
    return ax


def outliers(adata: AnnData, key: str = "qc_outlier", axes: np.ndarray | None = None):
    """Outlier score distribution, and the flagged fraction per plate."""
    if key not in adata.obs:
        raise KeyError(f"obs has no {key!r}; run mt.pp.outliers first")
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    scores = as_frame(adata.obs)[f"{key}_score"].to_numpy(dtype=float)
    flagged = as_frame(adata.obs)[key].to_numpy(dtype=bool)
    axes[0].hist(scores[~flagged], bins=50, label="kept")
    axes[0].hist(scores[flagged], bins=50, color="crimson", label="flagged")
    axes[0].set_xlabel("outlier score")
    axes[0].legend(fontsize=7)

    per_plate = as_frame(adata.obs).groupby("Metadata_Plate", observed=True)[key].mean()
    axes[1].bar(np.arange(len(per_plate)), per_plate.to_numpy())
    axes[1].set_xticks(np.arange(len(per_plate)))
    axes[1].set_xticklabels([str(name) for name in per_plate.index], rotation=45, fontsize=7)
    axes[1].set_ylabel("fraction flagged")
    return axes
