"""Mean average precision, computed with copairs.

mAP measures retrieval.
Profiles are ranked by similarity to a query profile, and the score is high when its positive pairs, such as replicates, rank above its negative pairs.
It is rank-based, so it needs no correlation threshold, and a permutation null gives each group a p-value.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._reduce import representation
from mantispy._core.frames import as_frame
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy

#: Column added under ``mode="activity"``, holding the row index for each control and -1 for every other row.
#: It keeps controls out of the queries and makes a perturbation's replicates retrieve against controls only.
REFERENCE_COLUMN = "Metadata_reference_index"

#: copairs pair definitions for each ``mode`` of :func:`map`.
MODES = {
    # Phenotypic activity :cite:p:`Kalinin_2025`, as in copairs' own example.
    # Is this perturbation distinguishable from the negative controls?
    "activity": {
        "pos_sameby": ["Metadata_Perturbation", REFERENCE_COLUMN],
        "pos_diffby": [],
        "neg_sameby": [],
        "neg_diffby": ["Metadata_Perturbation", REFERENCE_COLUMN],
    },
    # Phenotypic consistency :cite:p:`Kalinin_2025`.
    # Do perturbations sharing an annotation, such as a mechanism, target or gene, look more alike than those that do not?
    # Needs `annotation_key`; meant for consensus profiles of active perturbations.
    "consistency": {
        "pos_sameby": ["__annotation__"],
        # A positive pair must be two different perturbations, so replicate wells of one treatment do not count as annotation agreement.
        # On consensus input this excludes nothing.
        "pos_diffby": ["Metadata_Perturbation"],
        "neg_sameby": [],
        "neg_diffby": ["__annotation__"],
    },
    # Do a perturbation's replicates retrieve each other against all other profiles?
    # Like the "mAP-nonrep" of :cite:t:`Arevalo_2024`, which also restricts the negatives to the query's plate.
    "replicability": {
        "pos_sameby": ["Metadata_Perturbation"],
        "pos_diffby": [],
        "neg_sameby": [],
        "neg_diffby": ["Metadata_Perturbation"],
    },
    # The same, but a replicate only counts if it comes from a different plate.
    "cross_plate": {
        "pos_sameby": ["Metadata_Perturbation"],
        "pos_diffby": ["Metadata_Plate"],
        "neg_sameby": [],
        "neg_diffby": ["Metadata_Perturbation"],
    },
}

#: copairs output columns that are dropped.
#: The ragged per-group row indices cannot be written to h5ad and can be recomputed from the inputs.
_UNWRITABLE = ("indices",)


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
        pos_sameby: ``obs`` columns a positive pair must share, in copairs' terms. Pass the four pair arguments or ``mode``, not both.
        pos_diffby: ``obs`` columns in which a positive pair must differ.
        neg_sameby: ``obs`` columns a negative pair must share.
        neg_diffby: ``obs`` columns in which a negative pair must differ.
        mode: A preset for the pair definitions, one of the following.

            ``"activity"``
                Is this perturbation distinguishable from the negative controls?
                Its replicates are retrieved against control profiles only.
                This is the phenotypic activity of :cite:t:`Kalinin_2025`.
                Needs ``reference``.
            ``"consistency"``
                Do perturbations sharing an annotation look more alike than those that do not?
                This is the phenotypic consistency of :cite:t:`Kalinin_2025`.
                Needs ``annotation_key`` (a mechanism, target or gene column) and is meant for consensus profiles of perturbations already known to be active.
            ``"replicability"``
                Do a perturbation's replicates retrieve each other against all other profiles?
                Like the ``mAP-nonrep`` of the batch-correction benchmark of :cite:t:`Arevalo_2024`, which also restricts the negatives to the query's plate.
            ``"cross_plate"``
                As ``"replicability"``, but a replicate counts only if it is on a different plate, which separates reproducible biology from plate effects.
        annotation_key: The ``obs`` column ``mode="consistency"`` groups by.
        reference: Which rows are the negative controls for ``mode="activity"``, either ``"negcon"`` or the name of a boolean ``obs`` column.
        use_rep: Score ``obsm[use_rep]`` instead of ``X``.
        null_size: Size of the permutation null.
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
        ValueError: ``mode`` was passed together with explicit pair arguments or neither was passed, ``mode`` is not one of ``MODES``, ``mode="consistency"`` came without ``annotation_key``, ``mode="activity"`` found no controls, or the profiles hold missing values, which cannot be ranked.
        KeyError: ``obs`` is missing a column the pair definitions name.
    """
    try:
        from copairs import map as copairs_map
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
        settings = {key: list(value) for key, value in MODES[mode].items()}
        if mode == "consistency":
            if annotation_key is None:
                raise ValueError(
                    "mode='consistency' needs annotation_key=, the obs column holding the mechanism, "
                    "target or gene. For replicate retrieval, use mode='replicability'."
                )
            settings = {
                key: [annotation_key if column == "__annotation__" else column for column in value]
                for key, value in settings.items()
            }
            # Consistency is defined on consensus profiles.
            # With replicate rows, perturbations with more wells dominate their annotation groups.
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

    # The reference column is built below, not supplied by the caller.
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
    if mode == "activity":
        is_control = reference_mask(adata, reference)
        if not is_control.any():
            raise ValueError(
                f"mode='activity' needs negative controls, and reference={reference!r} selects none. "
                "Run mt.pp.annotate_controls, or use mode='replicability', which needs no controls."
            )
        meta[REFERENCE_COLUMN] = np.where(is_control, np.arange(adata.n_obs), -1)

    precision = copairs_map.average_precision(meta, features, **settings, distance=distance, progress_bar=False)
    if mode == "activity":
        precision = precision[~precision.index.isin(np.flatnonzero(is_control))]
    table = copairs_map.mean_average_precision(
        precision,
        sameby=[c for c in settings["pos_sameby"] if c != REFERENCE_COLUMN],
        null_size=null_size,
        threshold=threshold,
        seed=seed,
        progress_bar=False,
    ).drop(columns=list(_UNWRITABLE), errors="ignore")

    if mode == "activity":
        # Activity is reported for treatments only.
        control_groups = set(np.asarray(meta.loc[is_control, "Metadata_Perturbation"], dtype=object))
        table = table[~table["Metadata_Perturbation"].isin(control_groups)].reset_index(drop=True)
        table = table.drop(columns=[REFERENCE_COLUMN], errors="ignore")

    adata.uns.setdefault("mantispy", {})[key_added] = table

    group_columns = [c for c in settings["pos_sameby"] if c != REFERENCE_COLUMN]
    lookup = table.set_index(group_columns)
    index = pd.MultiIndex.from_frame(obs[group_columns]) if len(group_columns) > 1 else pd.Index(obs[group_columns[0]])
    adata.obs[key_added] = lookup["mean_average_precision"].reindex(index).to_numpy()
    adata.obs[f"{key_added}_qvalue"] = lookup["corrected_p_value"].reindex(index).to_numpy()
    return None
