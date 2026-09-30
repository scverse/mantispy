"""The integration benchmark: scib-metrics' standard panel drawn as a mantispy heatmap, with a copairs mAP block when copairs is present."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mantispy._core.masks import reference_mask
from mantispy.metrics._common import embedding

if TYPE_CHECKING:
    from anndata import AnnData
    from matplotlib.axes import Axes

#: The copairs pair-argument keys a caller must fully specify to bypass a preset.
_PAIR_KEYS = ("pos_sameby", "pos_diffby", "neg_sameby", "neg_diffby")


def _has_scib() -> bool:
    """Whether scib-metrics, the engine behind the benchmark panel, can be imported."""
    try:
        import scib_metrics  # noqa: F401
    except ImportError:
        return False
    return True


def _has_copairs() -> bool:
    """Whether copairs, which adds the mean-average-precision block, can be imported."""
    try:
        from copairs import map as _  # noqa: F401
    except ImportError:
        return False
    return True


def _map_settings(
    map_mode: str, label_key: str, batch_key: str, map_kwargs: dict[str, Any] | None
) -> dict[str, list[str]]:
    """The four copairs pair arguments for the mAP, from the ``replicability``/``cross_plate`` preset plus overrides.

    Raises:
        ValueError: ``map_mode`` is neither preset and ``map_kwargs`` did not fully specify the four pair arguments.
    """
    from mantispy.tl._map import MODES, _resolve_mode

    if map_mode in MODES and map_mode not in ("activity", "consistency"):
        settings = _resolve_mode(map_mode, label=label_key, batch=batch_key)
        if map_kwargs is not None:
            settings.update({key: list(value) for key, value in map_kwargs.items()})
        return settings
    if map_kwargs is None or not set(_PAIR_KEYS) <= set(map_kwargs):
        raise ValueError(
            f"map_mode must be 'replicability' or 'cross_plate', or pass map_kwargs with all of {_PAIR_KEYS}; "
            f"got {map_mode!r}"
        )
    return {key: list(map_kwargs[key]) for key in _PAIR_KEYS}


def _map_meta(adata: AnnData, settings: dict[str, list[str]]) -> tuple[pd.DataFrame, np.ndarray | None]:
    """The copairs metadata frame and the treated-row mask, built once so every representation reuses them."""
    from mantispy._core.frames import as_frame

    obs = as_frame(adata.obs)
    needed = sorted({column for group in settings.values() for column in group})
    meta = pd.DataFrame({column: obs[column].reset_index(drop=True).to_numpy() for column in needed})
    treated = None
    if "Metadata_Control" in adata.obs.columns:
        # reference_mask is the hardened control read: it handles the "True"/"False" categorical an h5ad round trip
        # leaves and raises on a NaN control rather than silently dropping it, matching mt.tl.map.
        treated = ~reference_mask(adata, "negcon")
        meta = meta[treated].reset_index(drop=True)
    return meta, treated


def _rep_map(
    adata: AnnData,
    use_rep: str,
    settings: dict[str, list[str]],
    prepared: tuple[pd.DataFrame, np.ndarray | None] | None = None,
) -> float:
    """Mean of the per-group mean average precision over ``obsm[use_rep]``, scored with copairs, controls left out."""
    from mantispy.tl._map import _score_map

    meta, treated = prepared if prepared is not None else _map_meta(adata, settings)
    features = embedding(adata, use_rep).astype(np.float32)
    if treated is not None:
        features = features[treated]

    # compute_null=False skips the permutation null: only the mean point estimate is read here, not the p-values.
    table, _ = _score_map(meta, features, settings, null_size=0, threshold=0.05, seed=0, warn=False, compute_null=False)
    return float(table["mean_average_precision"].mean())


def _benchmark(
    adata: AnnData, *, reps: Sequence[str], label_key: str, batch_key: str, min_max_scale: bool
) -> pd.DataFrame:
    """scib-metrics' public ``get_results`` frame for the representations, kept as a seam a test can read directly.

    scib-metrics is imported here and only here, and only the public ``get_results`` is read, never the private
    ``_results``, so scib's own Total stays the number scib computes.
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
    The mAP is measured over the treated wells only (controls left out), while scib's bio, batch and ``Total``
    columns use every well, so ``Total`` and ``Total+mAP`` are not strictly apples-to-apples on a control-heavy
    screen.

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
            ``"cross_plate"``. See ``_map_settings``.
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

    reps = tuple(dict.fromkeys(reps))  # a duplicate rep would draw two identically-labelled rows

    missing = [key for key in (label_key, batch_key) if key not in adata.obs.columns]
    if missing:
        raise KeyError(f"obs is missing the column(s) the benchmark needs: {missing}")

    # Count the treated labels, since the mAP drops the controls; value_counts drops NaN labels too.
    treated_labels = adata.obs[label_key]
    if "Metadata_Control" in adata.obs.columns:
        treated_labels = treated_labels[~reference_mask(adata, "negcon")]
    counts = treated_labels.value_counts()
    if counts.empty:
        raise ValueError(
            f"evaluate_integration found no treated profiles to score: obs[{label_key!r}] is empty or all missing "
            "once the controls are dropped."
        )
    if int(counts.max()) < 2:
        raise ValueError(
            "evaluate_integration needs replicate profiles across batches, and this object looks like one row "
            f"per perturbation (no value of obs[{label_key!r}] has two or more profiles). It scores the "
            "integration of well-level replicates; for consensus profiles use mt.tl.map(mode='consistency') or "
            "mt.metrics.known_relationships instead."
        )

    if have_copairs:
        settings = _map_settings(map_mode, label_key, batch_key, map_kwargs)
        prepared = _map_meta(adata, settings)
        map_values = [_rep_map(adata, rep, settings, prepared) for rep in reps]

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
        map_column = pd.Series(map_values, index=data.index)
        if min_max_scale:
            # Fold the mAP in on the same worst->0, best->1 scale as the bio metrics it joins, so Total+mAP and
            # the heatmap colour are not a scaled/raw mix.
            span = map_column.max() - map_column.min()
            map_column = (map_column - map_column.min()) / span
        data["mean_average_precision"] = map_column
        blocks["Retrieval"] = ["mean_average_precision"]
        if bio_cols:
            bio_prime = (data[bio_cols].sum(axis=1) + data["mean_average_precision"]) / (len(bio_cols) + 1)
            data["Total+mAP"] = 0.4 * data["Batch correction"] + 0.6 * bio_prime
            aggregate_cols = [*aggregate_cols, "Total+mAP"]
        else:
            warnings.warn(
                "scib-metrics returned no bio-conservation columns, so Total+mAP is undefined and is left out; the "
                "mAP is still shown on its own. This usually means a scib-metrics version change.",
                UserWarning,
                stacklevel=2,
            )
    blocks["Aggregate"] = aggregate_cols

    data = data[[column for columns in blocks.values() for column in columns]]
    _integration_heatmap(data, blocks, ax=ax)
    return data
