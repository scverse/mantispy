"""Quality-control plots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mantispy._core._reduce import get_matrix, group_codes
from mantispy._core._utils import as_frame
from mantispy.pl._common import axes

if TYPE_CHECKING:
    from anndata import AnnData


#: Same name the other plot modules use for :func:`mantispy.pl._common.axes`.
_axes = axes


def cell_counts(adata: AnnData, groupby: str = "Metadata_Plate", ax: plt.Axes | None = None):
    """Distribution of cells per well, split by ``groupby``."""
    ax = _axes(ax, (6, 4))
    codes, keys = group_codes(adata, ["Metadata_Plate", "Metadata_Well"])
    counts = np.bincount(codes, minlength=len(keys))
    labels = as_frame(adata.obs).groupby(codes, observed=True)[groupby].first()

    groups = list(dict.fromkeys(labels))
    ax.boxplot([counts[labels.to_numpy() == group] for group in groups], tick_labels=[str(g) for g in groups])
    ax.set_ylabel("cells per well")
    ax.set_xlabel(groupby)
    ax.tick_params(axis="x", rotation=45)
    return ax


def feature_distributions(
    adata: AnnData,
    features: Sequence[str],
    groupby: str = "Metadata_Plate",
    layer_before: str | None = "raw",
    kind: str = "ecdf",
):
    """Per-feature distributions, before and after normalization when ``layer_before`` exists.

    ``kind`` is ``"ecdf"``, ``"hist"`` or ``"ridge"`` (one offset filled density per group,
    easier to read with many groups).

    Returns a 2-D array of axes with one row per layer shown and one column per feature.
    """
    if kind not in {"ecdf", "hist", "ridge"}:
        raise ValueError(f"kind must be 'ecdf', 'hist' or 'ridge', got {kind!r}")
    features = list(features)
    show_before = layer_before is not None and layer_before in adata.layers
    layers = [layer_before, None] if show_before else [None]

    figure, axes = plt.subplots(len(layers), len(features), figsize=(4 * len(features), 3 * len(layers)), squeeze=False)
    for row, layer in enumerate(layers):
        matrix = get_matrix(adata, layer)
        for column, feature in enumerate(features):
            axis = axes[row, column]
            values = matrix[:, adata.var_names.get_loc(feature)]
            for offset, group in enumerate(dict.fromkeys(adata.obs[groupby])):
                selected = values[(adata.obs[groupby] == group).to_numpy()]
                selected = selected[~np.isnan(selected)]
                if not selected.size:
                    continue
                if kind == "ecdf":
                    axis.plot(np.sort(selected), np.linspace(0, 1, selected.size), lw=1, label=str(group))
                elif kind == "ridge":
                    _ridge(axis, selected, offset, str(group))
                else:
                    axis.hist(selected, bins=50, histtype="step", density=True, label=str(group))
            axis.set_title(f"{feature}\n{'raw' if layer else 'current'}", fontsize=8)
    axes[0, 0].legend(fontsize=6)
    figure.tight_layout()
    return axes


def _ridge(axis: plt.Axes, values: np.ndarray, offset: int, label: str) -> None:
    """One filled density curve, raised by ``offset`` so the groups stack rather than overlap."""
    grid = np.linspace(values.min(), values.max(), 128)
    if values.size < 2 or np.ptp(values) == 0:
        return
    from scipy.stats import gaussian_kde

    density = gaussian_kde(values)(grid)
    density = density / density.max() * 0.9
    axis.fill_between(grid, offset, offset + density, alpha=0.7, lw=0.6, edgecolor="black", label=label)


def nan_matrix(adata: AnnData, max_features: int = 200, ax: plt.Axes | None = None):
    """Fraction of missing values per feature, per plate."""
    ax = _axes(ax, (8, 4))
    missing = np.isnan(get_matrix(adata))
    codes, keys = group_codes(adata, "Metadata_Plate")
    fractions = np.stack([missing[codes == index].mean(axis=0) for index in range(len(keys))])

    image = ax.imshow(fractions[:, :max_features], aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([str(key) for key in keys], fontsize=7)
    ax.set_xlabel("feature")
    ax.figure.colorbar(image, ax=ax, label="NaN fraction")
    return ax


def qc(adata: AnnData, figsize: tuple[float, float] = (12, 8)):
    """Two-by-two summary of the QC metrics :func:`~mantispy.pp.calculate_qc_metrics` writes."""
    figure, axes = plt.subplots(2, 2, figsize=figsize)
    cell_counts(adata, ax=axes[0, 0])

    if "qc_n_nan" in adata.var:
        axes[0, 1].hist(adata.var["qc_n_nan"], bins=40)
        axes[0, 1].set_xlabel("cells missing this feature")

    flags = [column for column in ("qc_is_border", "qc_area_outlier", "qc_pass") if column in adata.obs]
    if flags:
        obs = as_frame(adata.obs)
        obs.groupby("Metadata_Plate", observed=True)[flags].mean().plot.bar(ax=axes[1, 0])
        axes[1, 0].set_ylabel("fraction of cells")
        axes[1, 0].legend(fontsize=6)

    if "qc_variance" in adata.var:
        variance = as_frame(adata.var)["qc_variance"].to_numpy(dtype=float)
        # All-NaN features have NaN variance, which log10 cannot take.
        variance = variance[np.isfinite(variance) & (variance > 0)]
        if variance.size:
            axes[1, 1].hist(np.log10(variance), bins=40)
        axes[1, 1].set_xlabel("log10 feature variance")

    figure.tight_layout()
    return axes


def replicate_saturation(adata: AnnData, key: str = "replicate_saturation", ax: plt.Axes | None = None):
    """The saturation curve with its spread across draws.

    A curve still rising at the right edge means the screen is under-replicated, which
    informs the design of the next experiment.
    """
    store = adata.uns.get("mantispy", {})
    if key not in store:
        raise KeyError(f"uns['mantispy'][{key!r}] is missing; run mt.tl.replicate_saturation first")

    table = pd.DataFrame(store[key])
    ax = _axes(ax, (5, 4))
    ax.errorbar(table["n_replicates"], table["mean"], yerr=table["std"], marker="o", capsize=3)
    ax.set_xticks(table["n_replicates"].to_numpy())
    ax.set_xlabel("replicates per perturbation")
    ax.set_ylabel("signature agreement")
    return ax


def cytotoxicity(adata: AnnData, key: str = "cytotoxicity", label_top: int = 8, ax: plt.Axes | None = None):
    """Distance from the controls against viability, with the suspect groups marked.

    Groups in the upper left are far from the controls and have lost most of their cells.
    """
    store = adata.uns.get("mantispy", {})
    if key not in store:
        raise KeyError(f"uns['mantispy'][{key!r}] is missing; run mt.tl.cytotoxicity first")

    table = pd.DataFrame(store[key])
    suspect = table["suspect"].to_numpy(dtype=bool)
    ax = _axes(ax, (5.5, 4.5))
    ax.scatter(table["viability"][~suspect], table["distance"][~suspect], s=16, color="tab:blue", label="ok")
    ax.scatter(table["viability"][suspect], table["distance"][suspect], s=16, color="crimson", label="suspect")

    threshold = adata.uns.get("mantispy", {}).get("params", {}).get("cytotoxicity", {}).get("min_viability", 0.7)
    ax.axvline(threshold, color="grey", ls="--", lw=1, label=f"viability = {threshold}")
    for _, row in table[suspect].nlargest(label_top, "distance").iterrows():
        ax.annotate(str(row["group"]), (row["viability"], row["distance"]), fontsize=6)

    ax.set_xlabel("viability, relative to the controls")
    ax.set_ylabel("distance from the controls")
    ax.legend(fontsize=7)
    return ax
