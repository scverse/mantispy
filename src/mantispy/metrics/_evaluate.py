"""The integration benchmark: scib-metrics' standard panel, with a copairs mAP column when copairs is present."""

from __future__ import annotations

import tempfile
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mantispy.metrics._common import embedding

if TYPE_CHECKING:
    from anndata import AnnData
    from plottable import Table


def _has_scib() -> bool:
    """Whether scib-metrics can be imported, the engine behind the benchmark panel."""
    try:
        import scib_metrics  # noqa: F401
    except ImportError:
        return False
    return True


def _has_copairs() -> bool:
    """Whether copairs can be imported, which adds the mean-average-precision column (it needs Python < 3.13)."""
    try:
        from copairs import map as _  # noqa: F401
    except ImportError:
        return False
    return True


def _mean_map(adata: AnnData, use_rep: str, label_key: str) -> float:
    """Mean of the per-group mean average precision over ``obsm[use_rep]``, scored with copairs.

    This runs the same copairs path as :func:`~mantispy.tl.map` on the representation, with positive
    pairs sharing ``label_key`` and negative pairs differing in it, and returns the mean of the
    ``mean_average_precision`` column. It writes nothing into ``adata``.

    Args:
        adata: Object holding the representation in ``obsm`` and the label in ``obs``.
        use_rep: ``obsm`` key of the representation to score.
        label_key: ``obs`` column that defines a group of replicates.

    Returns:
        The mean of the per-group mean average precision.
    """
    from copairs import map as copairs_map

    from mantispy._core.frames import as_frame

    features = embedding(adata, use_rep).astype(np.float32)
    obs = as_frame(adata.obs)
    meta = obs[[column for column in obs.columns if column.startswith("Metadata_")]].reset_index(drop=True)
    if label_key not in meta.columns:
        meta[label_key] = obs[label_key].reset_index(drop=True).to_numpy()

    settings = {"pos_sameby": [label_key], "pos_diffby": [], "neg_sameby": [], "neg_diffby": [label_key]}
    precision = copairs_map.average_precision(meta, features, **settings, progress_bar=False)
    # copairs keys its on-disk null cache without the seed, so a shared cache leaks nulls between calls.
    # The mean average precision itself does not depend on the null, so a small one keeps this fast.
    with tempfile.TemporaryDirectory() as cache:
        table = copairs_map.mean_average_precision(
            precision, sameby=[label_key], null_size=100, threshold=0.05, seed=0, progress_bar=False, cache_dir=cache
        )
    return float(table["mean_average_precision"].mean())


def _build(adata: AnnData, *, reps: Sequence[str], label_key: str, batch_key: str, with_map: bool) -> Any:
    """A scib-metrics :class:`~scib_metrics.benchmark.Benchmarker`, benchmarked, with the mAP row injected when asked.

    Kept as a seam so a test can read the benchmarked ``_results`` frame directly. scib-metrics is imported
    here and only here, so the native PC-regression path never pulls it in.

    Args:
        adata: Object holding the representations in ``obsm`` and the label and batch in ``obs``.
        reps: ``obsm`` keys to compare.
        label_key: ``obs`` column with the biological grouping.
        batch_key: ``obs`` column with the nuisance grouping.
        with_map: Inject a ``mean_average_precision`` row tagged as bio conservation before the caller plots.

    Returns:
        The benchmarked ``Benchmarker``.
    """
    from scib_metrics.benchmark import Benchmarker

    # progress_bar=False keeps the tqdm bars out of executed notebooks and test logs.
    bm = Benchmarker(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        embedding_obsm_keys=list(reps),
        n_jobs=-1,
        progress_bar=False,
    )
    bm.benchmark()
    if with_map:
        # mAP is higher-better and scib min-max scales higher-better metrics, so the colours and the
        # bio-weighted Total stay correct once the row is tagged as bio conservation.
        for rep in reps:
            bm._results.loc["mean_average_precision", rep] = _mean_map(adata, rep, label_key)
        bm._results.loc["mean_average_precision", "Metric Type"] = "Bio conservation"
    return bm


def evaluate_integration(
    adata: AnnData,
    *,
    reps: Sequence[str] = ("X_pca",),
    label_key: str = "Metadata_Perturbation",
    batch_key: str = "Metadata_Batch",
) -> pd.DataFrame | Table:
    """Score one or more representations against a batch, with the integration benchmark scib-metrics owns.

    The return type follows what is installed:

    - With ``mantispy[integration]`` (scib-metrics), this returns scib-metrics' standard benchmark table:
      bio conservation, batch correction and the weighted Total, as a :class:`plottable.Table` that draws
      its own figure. When copairs is also installed, a ``mean_average_precision`` row is added to it.
    - With only copairs, this returns a native :class:`pandas.DataFrame` of the per-representation mean
      average precision, one row per entry of ``reps``.
    - With neither, this raises :class:`ImportError`.

    This function computes no native PC-regression of its own. To audit what a representation spends its
    variance on besides the label, such as the cell count or the plate position, whose better direction is
    context-dependent and so is deliberately not in this table, use
    :func:`~mantispy.metrics.batch_variance_explained` or :func:`~mantispy.metrics.pc_regression`.

    Args:
        adata: Object holding the representations in ``obsm`` and the label and batch in ``obs``.
        reps: ``obsm`` keys to compare, e.g. ``("X_pca", "X_pca_harmony")``.
        label_key: ``obs`` column with the biological grouping.
        batch_key: ``obs`` column with the nuisance grouping.

    Returns:
        A :class:`plottable.Table` when scib-metrics is installed, or a per-representation mean-average-precision
        :class:`pandas.DataFrame` when only copairs is.

    Raises:
        ImportError: Neither scib-metrics nor copairs is installed.
    """
    have_scib = _has_scib()
    have_copairs = _has_copairs()

    if not have_scib and not have_copairs:
        raise ImportError(
            "evaluate_integration needs mantispy[integration] (scib-metrics, and copairs for the mAP "
            "column). Install it, or use mt.metrics.batch_variance_explained / pc_regression for a "
            "native batch-variance check."
        )

    if not have_scib:
        warnings.warn(
            "scib-metrics is not installed, so evaluate_integration returns only the per-representation "
            "mean average precision. Install mantispy[integration] to add the full scib-metrics benchmark "
            "panel (bio conservation, batch correction and the Total).",
            UserWarning,
            stacklevel=2,
        )
        return pd.DataFrame(
            {"mean_average_precision": [_mean_map(adata, rep, label_key) for rep in reps]},
            index=pd.Index(list(reps), name="representation"),
        )

    bm = _build(adata, reps=reps, label_key=label_key, batch_key=batch_key, with_map=have_copairs)
    return bm.plot_results_table(show=False)
