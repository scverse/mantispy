"""The integration benchmark: scib-metrics' standard panel drawn as a mantispy heatmap, with a copairs mAP block when copairs is present."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mantispy.metrics._common import embedding

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes


def _has_scib() -> bool:
    """Whether scib-metrics can be imported, the engine behind the benchmark panel."""
    try:
        import scib_metrics  # noqa: F401
    except ImportError:
        return False
    return True


def _has_copairs() -> bool:
    """Whether copairs can be imported, which adds the mean-average-precision block (it needs Python < 3.13)."""
    try:
        from copairs import map as _  # noqa: F401
    except ImportError:
        return False
    return True


def _map_settings(
    map_mode: str, label_key: str, batch_key: str, map_kwargs: dict[str, Any] | None
) -> dict[str, list[str]]:
    """The four copairs pair arguments for the per-representation mAP, generic over the key names.

    ``map_mode="replicability"`` is the ``mAP-nonrep`` of :cite:t:`Arevalo_2024`: a label's replicates retrieve
    each other against the other labels in the same batch, controls left out. ``map_mode="cross_plate"`` counts
    only replicates in a different batch, which separates reproducible biology from batch effects. ``map_kwargs``
    overrides any of the four arguments on top of the chosen preset for full control.

    Args:
        map_mode: The preset, ``"replicability"`` or ``"cross_plate"``.
        label_key: ``obs`` column with the biological grouping.
        batch_key: ``obs`` column with the nuisance grouping.
        map_kwargs: Overrides for any of ``pos_sameby``, ``pos_diffby``, ``neg_sameby``, ``neg_diffby``, or ``None``.

    Returns:
        The pair-argument dict copairs takes.

    Raises:
        ValueError: ``map_mode`` is neither preset and ``map_kwargs`` did not fully specify the pairing.
    """
    presets = {
        "replicability": {
            "pos_sameby": [label_key],
            "pos_diffby": [],
            "neg_sameby": [batch_key],
            "neg_diffby": [label_key],
        },
        "cross_plate": {
            "pos_sameby": [label_key],
            "pos_diffby": [batch_key],
            "neg_sameby": [],
            "neg_diffby": [label_key],
        },
    }
    if map_mode not in presets:
        if map_kwargs is None:
            raise ValueError(f"map_mode must be one of {tuple(presets)}, or pass map_kwargs=; got {map_mode!r}")
        settings = dict(presets["replicability"])
    else:
        settings = {key: list(value) for key, value in presets[map_mode].items()}
    if map_kwargs is not None:
        settings.update({key: list(value) for key, value in map_kwargs.items()})
    return settings


def _rep_map(adata: AnnData, use_rep: str, settings: dict[str, list[str]], *, null_size: int = 100) -> float:
    """Mean of the per-group mean average precision over ``obsm[use_rep]``, scored with copairs.

    Shares :func:`~mantispy.tl.map`'s scoring path, so the two never drift, and writes nothing into ``adata``.
    The controls are left out when ``obs`` marks them, matching the treated-only ``mAP-nonrep``. The null size
    is kept small because only the mean of ``mean_average_precision`` is read here, not the per-group p-values.

    Args:
        adata: Object holding the representation in ``obsm`` and the pairing columns in ``obs``.
        use_rep: ``obsm`` key of the representation to score.
        settings: The copairs pair arguments from :func:`_map_settings`.
        null_size: Size of the permutation null, small because only the scores are used.

    Returns:
        The mean of the per-group mean average precision.
    """
    from mantispy._core.frames import as_frame
    from mantispy.tl._map import _score_map

    features = embedding(adata, use_rep).astype(np.float32)
    obs = as_frame(adata.obs)
    needed = sorted({column for group in settings.values() for column in group})
    meta = pd.DataFrame({column: obs[column].reset_index(drop=True).to_numpy() for column in needed})
    if "Metadata_Control" in obs.columns:
        treated = ~obs["Metadata_Control"].to_numpy(dtype=bool)
        meta, features = meta[treated].reset_index(drop=True), features[treated]

    table = _score_map(meta, features, settings, null_size=null_size, threshold=0.05, seed=0, warn=False)
    return float(table["mean_average_precision"].mean())


def _benchmark(
    adata: AnnData, *, reps: Sequence[str], label_key: str, batch_key: str, min_max_scale: bool
) -> pd.DataFrame:
    """scib-metrics' public ``get_results`` frame for the representations.

    Kept as a seam so a test can read the benchmarked frame directly. scib-metrics is imported here and only
    here, so the native PC-regression path never pulls it in. Only the public ``get_results`` is read, never
    the private ``_results``, so scib's own Total stays the number scib computes.

    Args:
        adata: Object holding the representations in ``obsm`` and the label and batch in ``obs``.
        reps: ``obsm`` keys to compare.
        label_key: ``obs`` column with the biological grouping.
        batch_key: ``obs`` column with the nuisance grouping.
        min_max_scale: Min-max scale each column across the representations, as scib does by default.

    Returns:
        The ``get_results`` frame: one row per representation plus a ``Metric Type`` row tagging each column.
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
    return bm.get_results(min_max_scale=min_max_scale)


def evaluate_integration(
    adata: AnnData,
    *,
    reps: Sequence[str] = ("X_pca",),
    label_key: str = "Metadata_Perturbation",
    batch_key: str = "Metadata_Batch",
    min_max_scale: bool = False,
    map_mode: str = "replicability",
    map_kwargs: dict[str, Any] | None = None,
    ax: Axes | None = None,
) -> pd.DataFrame:
    """Score one or more representations against a batch and draw the integration benchmark as a heatmap.

    The numbers come from scib-metrics, the field's implementation of the integration panel, which mantispy
    reads through its public ``get_results`` rather than reimplements. mantispy renders its own heatmap from
    them and, when copairs is installed, adds the cross-replicate mean average precision (mAP) as its own block.
    The columns are grouped left to right into bio conservation, batch correction, retrieval (mAP) and the
    aggregate scores, each block under its own header.

    The bio-conservation metrics measure whether a representation keeps the biology (cLISI is how label-pure a
    well's neighbourhood is); the batch-correction metrics whether it mixes the batches (iLISI is how
    batch-mixed that neighbourhood is, PCR comparison how much less of the variance the batch explains after
    correction). ``Total`` is scib's own weighted score, ``0.4`` batch correction and ``0.6`` bio conservation,
    left exactly as scib computes it. ``Total+mAP`` is mantispy's: it folds mAP into the bio group as one more
    bio signal, ``bio' = mean(bio metrics + mAP)``, then reweights ``0.4`` batch correction and ``0.6`` bio'.

    The return type is uniform across install states, so the numbers are always reachable:

    - With ``mantispy[integration]`` (scib-metrics) and copairs, the frame holds every metric, the mAP column
      and both totals; the heatmap has all four blocks.
    - With only scib-metrics, the frame and heatmap drop the retrieval block and ``Total+mAP``.
    - With only copairs, the frame and heatmap hold the per-representation mAP alone, and a warning notes that
      scib-metrics adds the full panel.
    - With neither, this raises :class:`ImportError`.

    This function computes no native PC-regression of its own. To audit what a representation spends its
    variance on besides the label, such as the cell count or the plate position, whose better direction is
    context-dependent and so is deliberately not in this table, use :func:`~mantispy.metrics.pc_regression`
    for a single covariate or :func:`~mantispy.metrics.batch_variance_explained` to stack several.

    Args:
        adata: Object holding the representations in ``obsm`` and the label and batch in ``obs``.
        reps: ``obsm`` keys to compare, e.g. ``("X_pca", "X_pca_harmony")``.
        label_key: ``obs`` column with the biological grouping.
        batch_key: ``obs`` column with the nuisance grouping.
        min_max_scale: Colour and score each metric column scaled across the representations, as scib does by
            default. Off by default so a single representation still has honest absolute values to colour by.
        map_mode: The copairs pairing for mAP, ``"replicability"`` (the Arevalo ``mAP-nonrep`` default) or
            ``"cross_plate"``. See :func:`_map_settings`.
        map_kwargs: Overrides for any of the four copairs pair arguments, on top of ``map_mode``.
        ax: Axes to draw the heatmap on, or ``None`` for a new figure.

    Returns:
        The numeric results, one row per representation and one column per metric, the mAP and both totals.

    Raises:
        ImportError: Neither scib-metrics nor copairs is installed.
        ValueError: The object has one row per perturbation, which has no replicate pairs to score.
        KeyError: ``obs`` is missing ``label_key`` or ``batch_key``.
    """
    from mantispy.pl._evaluation import _integration_heatmap

    have_scib = _has_scib()
    have_copairs = _has_copairs()

    if not have_scib and not have_copairs:
        raise ImportError(
            'evaluate_integration needs scib-metrics and copairs; install pip install "mantispy[integration,map]". '
            "For a native batch-variance check without them, use mt.metrics.batch_variance_explained (stack "
            "several covariates) or mt.metrics.pc_regression (a single covariate)."
        )

    missing = [key for key in (label_key, batch_key) if key not in adata.obs.columns]
    if missing:
        raise KeyError(f"obs is missing the column(s) the benchmark needs: {missing}")

    counts = adata.obs[label_key].value_counts()
    if counts.empty or int(counts.max()) < 2:
        raise ValueError(
            "evaluate_integration needs replicate profiles across batches, and this object looks like one row "
            f"per perturbation (no value of obs[{label_key!r}] has two or more profiles). It scores the "
            "integration of well-level replicates; for consensus profiles use mt.tl.map(mode='consistency') or "
            "mt.metrics.known_relationships instead."
        )

    if have_copairs:
        settings = _map_settings(map_mode, label_key, batch_key, map_kwargs)
        map_values = [_rep_map(adata, rep, settings) for rep in reps]

    if not have_scib:
        warnings.warn(
            "scib-metrics is not installed, so evaluate_integration draws only the per-representation mean "
            'average precision. Install pip install "mantispy[integration,map]" to add the full scib-metrics '
            "panel (bio conservation, batch correction and the Total).",
            UserWarning,
            stacklevel=2,
        )
        frame = pd.DataFrame(
            {"mean_average_precision": map_values},
            index=pd.Index(list(reps), name="representation"),
        )
        _integration_heatmap(frame, {"Retrieval": ["mean_average_precision"]}, ax=ax)
        return frame

    if min_max_scale and len(reps) == 1:
        warnings.warn(
            "min_max_scale scales each metric across the representations, and with a single representation there "
            "is nothing to scale against, so every scaled column collapses. Pass more than one representation, "
            "or keep the default min_max_scale=False for honest absolute values.",
            UserWarning,
            stacklevel=2,
        )

    results = _benchmark(adata, reps=reps, label_key=label_key, batch_key=batch_key, min_max_scale=min_max_scale)
    metric_type = results.loc["Metric Type"]
    data = results.drop(index="Metric Type").astype(float).loc[list(reps)]

    bio_cols = [column for column in data.columns if metric_type[column] == "Bio conservation"]
    batch_cols = [column for column in data.columns if metric_type[column] == "Batch correction"]
    aggregate_cols = ["Batch correction", "Bio conservation", "Total"]

    blocks: dict[str, list[str]] = {"Bio conservation": bio_cols, "Batch correction": batch_cols}
    if have_copairs:
        data["mean_average_precision"] = map_values
        bio_prime = (data[bio_cols].sum(axis=1) + data["mean_average_precision"]) / (len(bio_cols) + 1)
        data["Total+mAP"] = 0.4 * data["Batch correction"] + 0.6 * bio_prime
        blocks["Retrieval"] = ["mean_average_precision"]
        aggregate_cols = [*aggregate_cols, "Total+mAP"]
    blocks["Aggregate"] = aggregate_cols

    ordered = bio_cols + batch_cols + (["mean_average_precision"] if have_copairs else []) + aggregate_cols
    data = data[ordered]
    _integration_heatmap(data, blocks, ax=ax)
    return data
