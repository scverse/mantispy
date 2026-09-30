from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mantispy.metrics._common import embedding, tidy
from mantispy.metrics._variance import pc_regression

if TYPE_CHECKING:
    from anndata import AnnData

BETTER = {
    "silhouette_label": "higher",
    "silhouette_batch": "higher",
    "ilisi": "higher",
    "clisi": "lower",
    "pc_regression": "lower",
    "mean_average_precision": "higher",
}


def _map_row(adata: AnnData, map_key: str, label_key: str) -> pd.DataFrame:
    """The mean mAP of a table :func:`~mantispy.tl.map` wrote, under the representation that run scored.

    Args:
        adata: Object holding the table and the provenance of the run that wrote it.
        map_key: Name of that table in ``uns["mantispy"]``.
        label_key: ``obs`` column to name in the row's ``key``.

    Returns:
        A one-row tidy frame holding ``mean_average_precision``.

    Raises:
        KeyError: There is no such table.
        KeyError: Nothing recorded which representation :func:`~mantispy.tl.map` scored.
    """
    store = adata.uns.get("mantispy", {})
    table = store.get(map_key)
    if table is None:
        raise KeyError(f"uns['mantispy'] has no {map_key!r}; run mt.tl.map first")

    # Provenance is keyed by function name, and mt.tl.map records use_rep=None when it scores X.
    recorded = store.get("params", {}).get("map")
    if recorded is None:
        raise KeyError(
            f"uns['mantispy']['params'] holds no record of mt.tl.map, so the representation behind "
            f"{map_key!r} is unknown; rerun mt.tl.map to record it"
        )
    rep = recorded.get("use_rep") or "X"
    value = float(pd.DataFrame(table)["mean_average_precision"].mean())
    return tidy("mean_average_precision", rep, label_key, value)


def _lisi_row(scib_metrics: Any, adata: AnnData, key: str, use_rep: str, perplexity: float, kind: str) -> pd.DataFrame:
    """Median LISI over rows :cite:p:`Korsunsky_2019`, from scib-metrics, or NaN when the row count cannot support ``perplexity``."""
    from scib_metrics.nearest_neighbors import NeighborsResults
    from sklearn.neighbors import NearestNeighbors

    values = embedding(adata, use_rep)
    labels = adata.obs[key].to_numpy()
    missing = int(pd.isna(labels).sum())
    if missing:
        raise ValueError(f"obs[{key!r}] has {missing} missing value(s); drop those rows or fill the column.")

    metric = "ilisi" if kind == "batch" else "clisi"
    n_neighbors = int(perplexity * 3)
    if n_neighbors >= adata.n_obs:
        # Each neighborhood needs n_neighbors + 1 rows; with fewer, scib cannot calibrate the kernel and LISI is undefined.
        supported = (adata.n_obs - 1) // 3
        remedy = f"pass a perplexity of at most {supported}" if supported >= 2 else "measure on a larger object"
        warnings.warn(
            f"LISI over obs[{key!r}] is undefined at perplexity={perplexity}: the kernel is calibrated over "
            f"{n_neighbors} neighbors, which needs {n_neighbors + 1} rows, and this object has {adata.n_obs}. "
            f"A 48-well plate, or a consensus object with one row per perturbation, is the usual cause; "
            f"to measure it, {remedy}. Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return tidy(metric, use_rep, key, np.nan)

    # Self is kept in the graph (index 0); scib masks it out when calibrating the kernel.
    distances, indices = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(values).kneighbors(values)
    neighbors = NeighborsResults(indices=indices, distances=distances)
    # scale=False returns the raw median LISI, so ilisi stays "higher is better" and clisi "lower is better".
    knn = scib_metrics.ilisi_knn if kind == "batch" else scib_metrics.clisi_knn
    value = knn(neighbors, labels, perplexity=perplexity, scale=False)
    return tidy(metric, use_rep, key, float(value))


def _silhouette_label_row(scib_metrics: Any, adata: AnnData, label_key: str, use_rep: str) -> pd.DataFrame:
    """Label separation rescaled to ``[0, 1]``, from scib-metrics, or NaN when separation is undefined for the object."""
    values = embedding(adata, use_rep)
    labels = adata.obs[label_key].to_numpy()

    n_labels = len(pd.unique(labels))
    if not 2 <= n_labels <= adata.n_obs - 1:
        # The silhouette needs between 2 and n_obs - 1 distinct labels; scib would raise here without naming the cause.
        warnings.warn(
            f"the label silhouette is undefined for obs[{label_key!r}]: it needs between 2 and "
            f"n_obs - 1 distinct labels, and this object has {n_labels} over {adata.n_obs} rows. "
            "One row per label, as a consensus object has, is the usual cause. Returning NaN.",
            UserWarning,
            stacklevel=2,
        )
        return tidy("silhouette_label", use_rep, label_key, np.nan)

    # rescale=True (the default) maps the average silhouette width into [0, 1].
    value = scib_metrics.silhouette_label(values, labels)
    return tidy("silhouette_label", use_rep, label_key, float(value))


def _silhouette_batch_row(
    scib_metrics: Any, adata: AnnData, label_key: str, batch_key: str, use_rep: str
) -> pd.DataFrame:
    """Batch mixing within each label, from scib-metrics, or NaN when every label is unscorable."""
    values = embedding(adata, use_rep)
    labels = adata.obs[label_key].to_numpy()
    batches = adata.obs[batch_key].to_numpy()

    # scib raises when every label group is undefined; check first so the panel gets NaN, not an error.
    # A group is scored only when it holds more than one batch but fewer batches than rows.
    def _scorable(label: object) -> bool:
        rows = labels == label
        return 1 < len(np.unique(batches[rows])) < int(rows.sum())

    if not any(_scorable(label) for label in pd.unique(labels)):
        return tidy("silhouette_batch", use_rep, batch_key, np.nan)

    value = scib_metrics.silhouette_batch(values, labels, batches)
    return tidy("silhouette_batch", use_rep, batch_key, float(value))


def _scib_panel(
    scib_metrics: Any, adata: AnnData, reps: Sequence[str], label_key: str, batch_key: str, perplexity: float
) -> list[pd.DataFrame]:
    """The batch-mixing rows scib-metrics owns, one block per representation.

    iLISI/cLISI and the batch and label silhouettes come from scib-metrics :cite:p:`Korsunsky_2019`.
    A metric that is undefined for the object (too few rows for the perplexity, one label, no scorable batch group) is a NaN row rather than a scib traceback.
    """
    frames = []
    for rep in reps:
        frames += [
            _silhouette_label_row(scib_metrics, adata, label_key=label_key, use_rep=rep),
            _silhouette_batch_row(scib_metrics, adata, label_key=label_key, batch_key=batch_key, use_rep=rep),
            _lisi_row(scib_metrics, adata, key=batch_key, use_rep=rep, perplexity=perplexity, kind="batch"),
            _lisi_row(scib_metrics, adata, key=label_key, use_rep=rep, perplexity=perplexity, kind="label"),
        ]
    return frames


def evaluate_correction(
    adata: AnnData,
    *,
    reps: Sequence[str] = ("X_pca",),
    label_key: str = "Metadata_Perturbation",
    batch_key: str = "Metadata_Batch",
    covariates: Sequence[str] = (),
    map_key: str | None = None,
    perplexity: float = 30,
) -> pd.DataFrame:
    """Run the correction panel over every representation and stack the results.

    The native rows always run: PC-regression on the batch, PC-regression on each covariate, and the mean mAP row when ``map_key`` is given.
    The batch-mixing rows (iLISI, cLISI, the batch and label silhouettes) come from scib-metrics, an optional dependency.
    When scib-metrics is installed those rows are added; when it is not, they are skipped and one message names them and how to add them, so the function returns the native rows rather than raising.

    Args:
        adata: Object holding the representations in ``obsm``.
        reps: Representations to compare, e.g. ``("X_pca", "X_pca_harmony")``.
        label_key: ``obs`` column with the biological grouping.
        batch_key: ``obs`` column with the nuisance grouping.
        covariates: Further ``obs`` columns to measure each representation against, numeric or categorical, one row each.
            A representation can be dominated by something that is neither the batch nor the label, such as the cell count, and nothing else here would report it.
        map_key: Name of a table written by :func:`~mantispy.tl.map`, to add its mean mAP as one more row.
            That table is read rather than recomputed, so the row appears once, under the representation that run scored, and not once per entry of ``reps``.
        perplexity: Perplexity for both LISI rows (iLISI and cLISI), used only when scib-metrics is installed.
            The default needs more than 90 rows, and on a smaller object those two rows are NaN unless a smaller value is passed.

    Returns:
        A tidy frame with ``metric``, ``representation``, ``key``, ``value`` and ``better``, the last saying which direction is an improvement for that metric.
        Without scib-metrics the frame holds the native rows only; with it, the batch-mixing rows are added.
        A covariate's row has no ``better``: whether its share of the variance should be small depends on what the covariate is.
        A cell count is a nuisance in a genetic screen, where the layout was not randomized, and partly a treatment effect in a compound screen, where a compound that kills cells is supposed to lower it.
        A metric that is undefined for this object, such as a LISI whose perplexity the row count cannot support or a silhouette over one row per label, is NaN in that frame rather than an error, so one undefined metric still leaves the others readable.

    Raises:
        KeyError: ``obsm`` holds nothing under one of ``reps``, or ``obs`` no column under one of ``covariates``.
        KeyError: ``map_key`` names no table, or nothing recorded the representation behind it.
    """
    frames = []
    for rep in reps:
        frames.append(pc_regression(adata, key=batch_key, use_rep=rep))
        # Two rows called "pc_regression" would collide when the table is pivoted on the metric.
        for covariate in covariates:
            measured = pc_regression(adata, key=covariate, use_rep=rep)
            frames.append(measured.assign(metric=f"pc_regression:{covariate}"))
    if map_key is not None:
        frames.append(_map_row(adata, map_key, label_key))

    try:
        import scib_metrics
    except ImportError:
        warnings.warn(
            "scib-metrics is not installed, so the batch-mixing metrics (iLISI, cLISI, the batch and label "
            "silhouettes) are omitted; only the native PC-regression rows are returned. "
            "Install it with pip install 'mantispy[integration]' to add them.",
            UserWarning,
            stacklevel=2,
        )
    else:
        frames += _scib_panel(
            scib_metrics, adata, reps, label_key=label_key, batch_key=batch_key, perplexity=perplexity
        )

    result = pd.concat(frames, ignore_index=True)
    result["better"] = result["metric"].map(BETTER)
    return result
