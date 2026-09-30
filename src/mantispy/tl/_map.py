from __future__ import annotations

import math
import tempfile
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._reduce import representation
from mantispy._core.frames import as_frame
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy

#: Row index for each control and -1 elsewhere, so under ``mode="activity"`` replicates retrieve against controls only.
REFERENCE_COLUMN = "Metadata_reference_index"

#: The copairs pairings, generic over the key names so :func:`map` and ``evaluate_integration`` share one source.
#: ``__label__``/``__batch__``/``__annotation__`` are filled in by :func:`_resolve_mode`.
MODES = {
    # Other plates' controls are beaten trivially yet count in the null, so pooling them inflates activity.
    "activity": {
        "pos_sameby": ["__label__", REFERENCE_COLUMN],
        "pos_diffby": [],
        "neg_sameby": ["__batch__"],
        "neg_diffby": ["__label__", REFERENCE_COLUMN],
    },
    "consistency": {
        "pos_sameby": ["__annotation__"],
        "pos_diffby": ["__label__"],
        "neg_sameby": [],
        "neg_diffby": ["__annotation__"],
    },
    "replicability": {
        "pos_sameby": ["__label__"],
        "pos_diffby": [],
        "neg_sameby": ["__batch__"],
        "neg_diffby": ["__label__"],
    },
    "cross_plate": {
        "pos_sameby": ["__label__"],
        "pos_diffby": ["__batch__"],
        "neg_sameby": [],
        "neg_diffby": ["__label__"],
    },
}

#: The ragged per-group row indices cannot be written to h5ad.
_UNWRITABLE = ("indices",)


def _resolve_mode(mode: str, *, label: str, batch: str, annotation: str | None = None) -> dict[str, list[str]]:
    """Substitute the ``__label__``/``__batch__``/``__annotation__`` placeholders in a :data:`MODES` preset."""
    substitutions = {"__label__": label, "__batch__": batch}
    if annotation is not None:
        substitutions["__annotation__"] = annotation
    return {key: [substitutions.get(column, column) for column in value] for key, value in MODES[mode].items()}


def _score_map(
    meta: pd.DataFrame,
    features: np.ndarray,
    settings: dict[str, list[str]],
    *,
    null_size: int,
    threshold: float,
    seed: int,
    distance: str = "cosine",
    drop_positions: np.ndarray | None = None,
    warn: bool = True,
    compute_null: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """The copairs scoring shared by :func:`map` and :func:`~mantispy.metrics.evaluate_integration`.

    Returns the per-group ``mean_average_precision`` table (without its ragged ``indices`` column) and the
    group columns, writing nothing into any object. With ``compute_null=False`` it skips the permutation null
    and returns just the per-group mean of ``average_precision``, the point estimate the null does not affect,
    which ``evaluate_integration`` uses so scoring a representation does not pay for discarded permutations.

    Raises:
        ValueError: No profile has a negative pair, so there is nothing to score.
    """
    from copairs import map as copairs_map

    precision = copairs_map.average_precision(meta, features, **settings, distance=distance, progress_bar=False)
    if drop_positions is not None:
        precision = precision[~precision.index.isin(drop_positions)]
    group_columns = [column for column in settings["pos_sameby"] if column != REFERENCE_COLUMN]

    # copairs scores a query with replicates but no negatives AP = 1 at the smallest p-value.
    queries = precision["n_pos_pairs"] > 0
    stranded = queries & (precision["n_total_pairs"] == precision["n_pos_pairs"])
    if stranded.any():
        if stranded.sum() == queries.sum():
            raise ValueError(
                "no profile has a negative pair to rank its replicates against, so there is nothing to score"
            )
        if warn:
            scope = settings["neg_sameby"] or group_columns
            warnings.warn(
                f"{int(stranded.sum())} profile(s) have replicates but no negative pair to rank them against, and are "
                f"left out. They are in {precision.loc[stranded, scope].drop_duplicates().to_dict('records')}.",
                UserWarning,
                stacklevel=4,
            )
        precision = precision[~stranded]

    if not compute_null:
        scored = precision[precision["n_pos_pairs"] > 0]
        table = scored.groupby(group_columns, as_index=False)["average_precision"].mean()
        return table.rename(columns={"average_precision": "mean_average_precision"}), group_columns

    if warn:
        groups = len(precision.loc[precision["n_pos_pairs"] > 0, group_columns].drop_duplicates())
        sharing = groups / ((null_size + 1) * threshold)
        if sharing >= 1:
            warnings.warn(
                f"with null_size={null_size} no p-value can fall below 1/{null_size + 1}, so the correction over "
                f"{groups} groups calls none of them unless at least {math.floor(sharing) + 1} reach that floor "
                f"together. Raise null_size above {groups / threshold:.0f} for one group to be callable on its own.",
                UserWarning,
                stacklevel=4,
            )

    # copairs keys its on-disk null cache without the seed, so a shared cache leaks nulls between calls.
    with tempfile.TemporaryDirectory() as cache:
        table = copairs_map.mean_average_precision(
            precision,
            sameby=group_columns,
            null_size=null_size,
            threshold=threshold,
            seed=seed,
            progress_bar=False,
            cache_dir=cache,
        ).drop(columns=list(_UNWRITABLE), errors="ignore")
    return table, group_columns


@inplace_or_copy(expects=("well", "perturbation"))
def map(
    adata: AnnData,
    pos_sameby: Sequence[str] | None = None,
    pos_diffby: Sequence[str] = (),
    neg_sameby: Sequence[str] = (),
    neg_diffby: Sequence[str] = (),
    mode: str | None = None,
    annotation_key: str | None = None,
    reference: str | None = "negcon",
    use_rep: str | None = None,
    null_size: int = 10_000,
    threshold: float = 0.05,
    seed: int = 0,
    distance: str = "cosine",
    key_added: str = "map",
    copy: bool = False,
) -> AnnData | None:
    """Mean average precision per group, with a permutation null.

    Args:
        adata: Profiles to score, normally well-level.
        pos_sameby: ``obs`` columns a positive pair must share, in copairs' terms.
            Pass the four pair arguments or ``mode``, not both.
        pos_diffby: ``obs`` columns in which a positive pair must differ.
        neg_sameby: ``obs`` columns a negative pair must share.
        neg_diffby: ``obs`` columns in which a negative pair must differ.
        mode: A preset for the pair definitions, one of the following.

            ``"activity"``
                Is this perturbation distinguishable from the negative controls it was plated with?
                Its replicates are retrieved against the control profiles on the query's own plate only.
                This is the phenotypic activity of :cite:t:`Kalinin_2025`.
                Needs ``reference`` and ``Metadata_Plate``; queries on a plate without controls are left out, with a warning.
            ``"consistency"``
                Do perturbations sharing an annotation look more alike than those that do not?
                This is the phenotypic consistency of :cite:t:`Kalinin_2025`.
                Needs ``annotation_key`` (a mechanism, target or gene column) and is meant for consensus profiles of perturbations already known to be active.
            ``"replicability"``
                Do a perturbation's replicates retrieve each other against the other perturbations on the query's plate?
                This is the ``mAP-nonrep`` of :cite:t:`Arevalo_2024`, which leaves the controls out; ``reference=None`` keeps them in.
            ``"cross_plate"``
                Do a perturbation's replicates on other plates retrieve each other against all other profiles?
                A replicate counts only if it is on a different plate, which separates reproducible biology from plate effects.
        annotation_key: The ``obs`` column ``mode="consistency"`` groups by.
        reference: Which rows are the negative controls, which ``mode="activity"`` retrieves against and ``mode="replicability"`` leaves out: ``"negcon"``, the name of a boolean ``obs`` column, or ``None`` for none.
        use_rep: Score ``obsm[use_rep]`` instead of ``X``.
        null_size: Size of the permutation null.
            No p-value falls below ``1 / (null_size + 1)``, so the correction over many groups needs a large one, and a warning says when it is too small to call a group on its own.
        threshold: Significance threshold passed to copairs.
        seed: Seed for the permutation null.
        distance: Distance copairs ranks by.
        key_added: Where to store results.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes the per-group table to ``uns["mantispy"][key_added]`` and joins ``obs[key_added]`` and ``obs[key_added + "_qvalue"]`` back onto the rows.

    Raises:
        ImportError: copairs is not installed, which it is not by default because it needs Python < 3.13.
        ValueError: ``mode`` was passed together with explicit pair arguments or neither was passed, ``mode`` is not one of ``MODES``, ``mode="consistency"`` came without ``annotation_key``, ``mode="activity"`` found no controls, no profile has a negative pair to be ranked against, or the profiles hold missing values, which cannot be ranked.
        KeyError: ``obs`` is missing a column the pair definitions or ``reference`` name.
    """
    try:
        from copairs import map as _  # noqa: F401 - the scoring imports it again; this is the friendly early guard
    except ImportError as error:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "mt.tl.map needs copairs, an optional extra because it requires Python < 3.13. "
            "Install it with pip install 'mantispy[map]' on Python 3.12 or older."
        ) from error

    explicit = any([pos_sameby, pos_diffby, neg_sameby, neg_diffby])
    if mode is not None and explicit:
        raise ValueError("pass either mode= or explicit pos_*/neg_* arguments, not both")
    if mode is not None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {tuple(MODES)}, got {mode!r}")
        if mode == "consistency" and annotation_key is None:
            raise ValueError(
                "mode='consistency' needs annotation_key=, the obs column holding the mechanism, "
                "target or gene. For replicate retrieval, use mode='replicability'."
            )
        settings = _resolve_mode(mode, label="Metadata_Perturbation", batch="Metadata_Plate", annotation=annotation_key)
        if mode == "consistency":
            perturbations = adata.obs["Metadata_Perturbation"].nunique() if "Metadata_Perturbation" in adata.obs else 0
            if perturbations and adata.n_obs > perturbations:
                warnings.warn(
                    f"mode='consistency' expects one profile per perturbation, and this object has "
                    f"{adata.n_obs} rows for {perturbations} perturbation(s). Same-perturbation pairs are "
                    "excluded, but perturbations with more wells still weigh more. Collapse first with "
                    "mt.tl.consensus().",
                    UserWarning,
                    stacklevel=3,
                )
    elif pos_sameby:
        settings = {
            "pos_sameby": list(pos_sameby),
            "pos_diffby": list(pos_diffby),
            "neg_sameby": list(neg_sameby),
            "neg_diffby": list(neg_diffby),
        }
    else:
        raise ValueError("pass pos_sameby= or mode=")

    needed = {column for group in settings.values() for column in group} - {REFERENCE_COLUMN}
    missing = sorted(needed - set(adata.obs.columns))
    if missing:
        raise KeyError(f"obs is missing the column(s) mAP needs: {missing}")

    features = representation(adata, use_rep).astype(np.float32)
    if np.isnan(features).any():
        raise ValueError(
            "mAP cannot rank profiles with missing values. Drop the affected features "
            "with mt.pp.feature_select(adata, na_cutoff=0.0); the default na_cutoff=0.05 "
            "keeps features with up to 5% missing values."
        )
    obs = as_frame(adata.obs)
    meta = obs[[c for c in obs.columns if c.startswith("Metadata_")]].reset_index(drop=True)
    drop_positions = None
    if mode == "activity":
        is_control = reference_mask(adata, reference)
        if not is_control.any():
            raise ValueError(
                f"mode='activity' needs negative controls, and reference={reference!r} selects none. "
                "Run mt.pp.annotate_controls, or use mode='replicability', which needs no controls."
            )
        meta[REFERENCE_COLUMN] = np.where(is_control, np.arange(adata.n_obs), -1)
        drop_positions = np.flatnonzero(is_control)
    elif mode == "replicability" and reference is not None:
        treated = ~reference_mask(adata, reference)
        meta, features = meta[treated].reset_index(drop=True), features[treated]

    table, group_columns = _score_map(
        meta,
        features,
        settings,
        null_size=null_size,
        threshold=threshold,
        seed=seed,
        distance=distance,
        drop_positions=drop_positions,
    )

    if mode == "activity":
        control_groups = set(np.asarray(meta.loc[is_control, "Metadata_Perturbation"], dtype=object))
        table = table[~table["Metadata_Perturbation"].isin(control_groups)].reset_index(drop=True)
        table = table.drop(columns=[REFERENCE_COLUMN], errors="ignore")

    adata.uns.setdefault("mantispy", {})[key_added] = table

    lookup = table.set_index(group_columns)
    index = pd.MultiIndex.from_frame(obs[group_columns]) if len(group_columns) > 1 else pd.Index(obs[group_columns[0]])
    adata.obs[key_added] = lookup["mean_average_precision"].reindex(index).to_numpy()
    adata.obs[f"{key_added}_qvalue"] = lookup["corrected_p_value"].reindex(index).to_numpy()
    return None
