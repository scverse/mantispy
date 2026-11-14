from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._reduce import representation
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy
from mantispy._core.schema import resolve_perturbation_column

#: Row index for each control and -1 elsewhere, so under ``mode="activity"`` replicates retrieve against controls only.
# A transient per-row index of the reference set, added for the pairing and dropped before return. It is
# not metadata, so it does not wear the Metadata_ prefix (cf. the "__label__" helper column).
REFERENCE_COLUMN = "__reference_index__"

#: The pair definitions, generic over the key names so :func:`map` and ``evaluate_integration`` share one source.
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

#: Types that mark an annotation cell as a list of labels rather than one scalar label.
_LIST_LIKE = (list, tuple, set, np.ndarray)

#: The nulls of a group's mAP, and the samplers of the copied one.
NULLS = ("copied", "permutation", "joint")
NULL_SAMPLERS = ("exact", "floyd")


def _resolve_mode(mode: str, *, label: str, batch: str, annotation: str | None = None) -> dict[str, list[str]]:
    """Substitute the ``__label__``/``__batch__``/``__annotation__`` placeholders in a :data:`MODES` preset."""
    substitutions = {"__label__": label, "__batch__": batch}
    if annotation is not None:
        substitutions["__annotation__"] = annotation
    return {key: [substitutions.get(column, column) for column in value] for key, value in MODES[mode].items()}


def _list_valued(series: pd.Series) -> bool:
    """Whether a label column holds a list of labels per row rather than one scalar label.

    A perturbation with several annotations (targets, mechanisms) is a list; a string is one label, not a list of
    characters. A column that mixes the two cannot be read either way and is rejected.
    """
    # Lists can only live in an object column, so a numeric, boolean or categorical column is scalar without a scan.
    if series.dtype != object:
        return False
    present = series.dropna()
    if present.empty:
        return False
    is_list = present.map(lambda value: isinstance(value, _LIST_LIKE))
    if bool(is_list.all()):
        return True
    if bool(is_list.any()):
        raise ValueError(
            f"label column {series.name!r} mixes list-valued and scalar entries, so it is neither single-label "
            "nor multilabel. Give every row a list, or every row one scalar label."
        )
    return False


def _refuse_unrankable(features: np.ndarray, names: pd.Index, distance: str) -> None:
    """Raise when a profile holds an infinite value or, for a cosine distance, has a zero or overflowing norm.

    Such a profile is zeroed by the cosine, so every similarity it has ties and its replicates rank first, scoring an
    average precision of one that inflates the mAP without a sign that anything went wrong.
    """
    bad = ~np.isfinite(features).all(axis=1)
    if "cosine" in distance:
        with np.errstate(over="ignore"):
            norm = np.linalg.norm(features, axis=1)
        bad |= ~np.isfinite(norm) | (norm == 0)
    if bad.any():
        raise ValueError(
            f"mAP cannot rank {int(bad.sum())} profile(s) with an infinite value, or a zero or overflowing norm "
            f"under distance={distance!r} (e.g. {', '.join(str(name) for name in names[bad][:5])}). Drop the features with no "
            "spread that normalize flags in var['degenerate_scale'], or features with extreme values, first."
        )


def _check_null(null: str, null_sampler: str, seed: int) -> None:
    if null not in NULLS:
        raise ValueError(f"null must be one of {NULLS}, got {null!r}")
    if null_sampler not in NULL_SAMPLERS:
        raise ValueError(f"null_sampler must be one of {NULL_SAMPLERS}, got {null_sampler!r}")
    if null_sampler == "floyd" and null != "copied":
        raise ValueError(f"null_sampler='floyd' draws the copied null; it does not apply to null={null!r}")
    if int(seed) != seed or not 0 <= seed < 2**64:
        raise ValueError(f"seed must be an integer in [0, 2**64), got {seed!r}")


def _score_map(
    meta: pd.DataFrame,
    features: np.ndarray,
    settings: dict[str, list[str]],
    *,
    null_size: int,
    threshold: float,
    seed: int,
    distance: str = "cosine",
    null: str = "copied",
    null_sampler: str = "exact",
    similarity_solver: str = "exact",
    drop_positions: np.ndarray | None = None,
    multilabel_col: str | None = None,
    warn: bool = True,
    compute_null: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """The mAP scoring shared by :func:`map` and :func:`~mantispy.metrics.evaluate_integration`.

    Returns the per-group table (its ``score`` column is the mean average precision) and the group columns, writing
    nothing into any object. With ``compute_null=False`` it skips the null and returns just the per-group mean of
    ``average_precision``, the point estimate the null does not affect, which ``evaluate_integration`` uses so
    scoring a representation does not pay for discarded draws.

    With ``multilabel_col`` set, a positive pair shares at least one of that column's labels, and a profile with
    several labels is scored once per label, so it counts toward each of its classes.

    Raises:
        ValueError: No profile has a positive pair, or none has a negative pair, so there is nothing to score.
    """
    from numba import get_num_threads

    from mantispy._core import _average_precision as ap_kernel
    from mantispy._core import _map_null, _map_permutation
    from mantispy._core._stats import benjamini_hochberg

    _check_null(null, null_sampler, seed)
    lay = ap_kernel.layout(
        meta, settings["pos_sameby"], settings["pos_diffby"], settings["neg_sameby"], settings["neg_diffby"],
        multilabel_col,
    )  # fmt: skip
    result = ap_kernel.average_precision(lay, features, distance, similarity_solver)
    precision = ap_kernel.ap_frame(meta, lay, result, multilabel_col)
    if drop_positions is not None:
        precision = precision[~precision[ap_kernel.PROFILE].isin(drop_positions)]
    group_columns = [column for column in settings["pos_sameby"] if column != REFERENCE_COLUMN]

    queries = precision["n_pos_pairs"] > 0
    if not queries.any():
        raise ValueError("no profile has a positive pair, so there is nothing to score")
    # A query with replicates but no negatives cannot be ranked, so it is left out rather than scored AP = 1.
    stranded = queries & (precision["n_total_pairs"] == precision["n_pos_pairs"])
    if stranded.any():
        if stranded.sum() == queries.sum():
            raise ValueError(
                "no profile has a negative pair to rank its replicates against, so there is nothing to score"
            )
        if warn:
            scope = settings["neg_sameby"] or group_columns
            get_logger().warning(
                "map: %d profile(s) have replicates but no negative pair, and are left out",
                int(stranded.sum()),
                deep=f"they are in {precision.loc[stranded, scope].drop_duplicates().to_dict('records')}",
            )
        precision = precision[~stranded]

    if not compute_null:
        scored = precision[precision["n_pos_pairs"] > 0]
        # Canonical schema names (§15.1): the per-group mean average precision is the method score.
        return scored.groupby(group_columns, as_index=False).agg(score=("average_precision", "mean")), group_columns

    if warn:
        groups = len(precision.loc[precision["n_pos_pairs"] > 0, group_columns].drop_duplicates())
        sharing = groups / ((null_size + 1) * threshold)
        if sharing >= 1:
            get_logger().warning(
                "map: null_size is too small to call any of %d groups unless at least %d of them reach the smallest p-value together",
                groups,
                math.floor(sharing) + 1,
                deep=f"raise null_size above {groups / threshold:.0f} for one group to be callable on its own",
            )

    groups_table, members, tested = _map_null.group_scores(precision, group_columns)
    scores = groups_table["mean_average_precision"].to_numpy()
    n_pos = tested["n_pos_pairs"].to_numpy()
    n_total = tested["n_total_pairs"].to_numpy()
    if null_size == 0:
        pvalues = np.full(len(groups_table), np.nan)
    elif null == "copied":
        order = np.concatenate(members)
        ptr = np.concatenate([[0], np.cumsum([len(ix) for ix in members])]).astype(np.int64)
        pvalues = _map_null.copied_pvalues(
            scores, ptr, n_pos[order], n_total[order], null_size, seed, null_sampler, get_num_threads()
        )
    elif null == "joint":
        edges = _map_null.group_edges(members, tested.index.to_numpy(), result)
        pvalues = _map_null.joint_pvalues(scores, members, n_pos, n_total - n_pos, edges, null_size, seed)
    else:
        pvalues = _map_permutation.permutation_pvalues(
            lay, result, features, distance, scores, members, tested.index.to_numpy(), null_size, seed
        )
    qvalues = benjamini_hochberg(pvalues)
    # Canonical schema names (§15.1): score/pvalue/qvalue/is_significant. below_p (the raw-p decision) stays as a
    # tool-specific column beside the canonical is_significant.
    groups_table = groups_table.rename(columns={"mean_average_precision": "score"})
    groups_table["pvalue"] = pvalues
    groups_table["qvalue"] = qvalues
    for column, values in (("below_p", pvalues), ("is_significant", qvalues)):
        flags = pd.Series(values < threshold)
        groups_table[column] = flags.astype("boolean").mask(np.isnan(values)) if np.isnan(values).any() else flags
    return groups_table, group_columns


@inplace_or_copy(expects=("well", "aggregate"))
def map(
    adata: AnnData,
    pos_sameby: Sequence[str] | None = None,
    pos_diffby: Sequence[str] = (),
    neg_sameby: Sequence[str] = (),
    neg_diffby: Sequence[str] = (),
    mode: str | None = None,
    label: str | None = None,
    batch: str = "Metadata_Plate",
    annotation_key: str | None = None,
    label_sep: str | None = None,
    reference: str | None = "negcon",
    use_rep: str | None = None,
    null_size: int = 10_000,
    threshold: float = 0.05,
    seed: int = 0,
    distance: str = "cosine",
    null: Literal["copied", "permutation", "joint"] = "copied",
    null_sampler: Literal["exact", "floyd"] = "exact",
    similarity_solver: Literal["exact", "blas"] = "exact",
    key_added: str = "map",
    copy: bool = False,
) -> AnnData | None:
    """Mean average precision per group, with a null to test it against.

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
                Needs ``reference`` and the ``batch`` column; queries on a plate without controls are left out, with a warning.
                In a CRISPR screen, non-targeting controls do not cut DNA, so nearly every knockout can come out active from the cutting response alone; guides against genes the cell line does not express (:func:`~mantispy.io.unexpressed_genes`) are a cutting control to retrieve against instead, and :func:`~mantispy.tl.empirical_fdr` calibrates the calls.
            ``"consistency"``
                Do perturbations sharing an annotation look more alike than those that do not?
                This is the phenotypic consistency of :cite:t:`Kalinin_2025`.
                Needs ``annotation_key`` (a mechanism, target or gene column) and is meant for consensus profiles of perturbations already known to be active.
                When ``annotation_key`` holds a list of labels per row, a pair is positive if the two lists share any label, and a perturbation with several labels is scored toward each of its classes. This multilabel scoring is supported through ``mode="consistency"``.
            ``"replicability"``
                Do a perturbation's replicates retrieve each other against the other perturbations on the query's plate?
                This is the ``mAP-nonrep`` of :cite:t:`Arevalo_2024`, which leaves the controls out; ``reference=None`` keeps them in.
            ``"cross_plate"``
                Do a perturbation's replicates on other plates retrieve each other against all other profiles?
                A replicate counts only if it is on a different plate, which separates reproducible biology from plate effects.
        label: The ``obs`` column naming the perturbation, which the presets group replicates by.
            Defaults to the perturbation column the object records, else ``Metadata_Perturbation``.
        batch: The ``obs`` column naming the plate or batch, which the presets draw negatives from.
        annotation_key: The ``obs`` column ``mode="consistency"`` groups by. A column holding a list of labels per row is scored as multilabel (see the mode description); a scalar column is a single label. A list-valued ``obs`` column does not survive ``write_h5ad``, so store the labels as a delimited string, which does survive, and split it with ``label_sep``.
        label_sep: Split a string ``annotation_key`` on this separator into a list of labels per row, so a stored ``"MEK|ERK"`` is scored as multilabel. ``None`` leaves the column as it is. An entry that is already a list, or is missing, is left untouched.
        reference: Which rows are the negative controls, which ``mode="activity"`` retrieves against and ``mode="replicability"`` leaves out: ``"negcon"``, the name of a boolean ``obs`` column, or ``None`` for none.
        use_rep: Score ``obsm[use_rep]`` instead of ``X``.
        null_size: Number of draws of the null.
            No p-value falls below ``1 / (null_size + 1)``, so the correction over many groups needs a large one, and a warning says when it is too small to call a group on its own.
        threshold: Significance threshold of the p- and q-values.
        seed: Seed of the null.
        distance: Distance to rank by: ``"cosine"`` (the default), ``"abs_cosine"``, ``"correlation"``,
            ``"euclidean"``, ``"manhattan"``, ``"chebyshev"``, or a :func:`scipy.spatial.distance.cdist` metric name,
            each turned into a similarity as copairs does.
        null: The null model of the group mAP, as copairs defines it: ``"copied"``, ``"permutation"`` or ``"joint"``.
            ``"copied"`` is the model of :cite:t:`Kalinin_2025` and copairs before 0.6.
            ``"permutation"`` relabels profiles within the blocks they are ranked in and recomputes the mAP from their similarities. It needs every profile holding a label in the scored table, so it fails on a multilabel annotation where a profile has no negative pair.
            ``"joint"`` draws the members' APs together, sharing each positive pair between its two members.
        null_sampler: How the copied null is drawn. ``"exact"`` replays copairs' random stream, so p-values equal copairs' for the same seed; ``"floyd"`` draws only the positions of the positives, which is faster and has the same distribution but different draws.
        similarity_solver: How similarities are computed. ``"exact"`` reproduces copairs' float32 values, so rankings and scores equal copairs'; ``"blas"`` computes them as matrix products, which is faster for the cosine, correlation and euclidean distances but can order near-tied pairs differently.
        key_added: Where to store results.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes the per-group table to ``uns["mantispy"][key_added]`` and joins ``obs[key_added]`` and ``obs[key_added + "_qvalue"]`` back onto the rows.
        A group with one member has no within-group pair to score, so it is absent from the table.
        A multilabel annotation has no single score per row, so it writes the per-class table to ``uns`` only and clears ``obs[key_added]`` and ``obs[key_added + "_qvalue"]``.

    Raises:
        ValueError: ``mode`` was passed together with explicit pair arguments or neither was passed, ``mode`` is not one of ``MODES``, ``mode="consistency"`` came without ``annotation_key``, ``mode="activity"`` found no controls, no profile has a positive pair or a negative pair to be ranked against, the profiles hold missing or infinite values or a cosine norm of zero or overflow, which cannot be ranked, a label column mixes list-valued and scalar entries, more than one positive-pair column is list-valued, or the null, sampler, solver or distance is not supported.
        KeyError: ``obs`` is missing a column the pair definitions or ``reference`` name.

    Notes:
        The pairs are never materialized: a query's positives and negatives follow from the group codes of the pair columns, so memory does not grow with the number of pairs, while time does. An explicit pair query that pairs across a high-cardinality column (every compound against every other) is best run on consensus profiles (:func:`~mantispy.tl.consensus`).
    """
    label = resolve_perturbation_column(adata, label)
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
        settings = _resolve_mode(mode, label=label, batch=batch, annotation=annotation_key)
        if mode == "consistency":
            perturbations = adata.obs[label].nunique() if label in adata.obs else 0
            if perturbations and adata.n_obs > perturbations:
                get_logger().warning(
                    "map: mode='consistency' expects one profile per perturbation, but the object has %d rows for %d perturbations",
                    adata.n_obs,
                    perturbations,
                    deep="perturbations with more wells weigh more, so collapse them first with mt.tl.consensus()",
                )
    elif pos_sameby:
        settings = {
            "pos_sameby": list(pos_sameby),
            "pos_diffby": list(pos_diffby),
            "neg_sameby": list(neg_sameby),
            "neg_diffby": list(neg_diffby),
        }
    else:
        raise ValueError(
            "mt.tl.map needs mode='replicability', 'activity' or 'consistency', or explicit pos_sameby= and neg_sameby="
        )

    needed = {column for group in settings.values() for column in group} - {REFERENCE_COLUMN}
    missing = sorted(needed - set(adata.obs.columns))
    if missing:
        raise KeyError(f"obs is missing the column(s) mAP needs: {missing}")

    features = representation(adata, use_rep).astype(np.float32)
    if np.isnan(features).any():
        raise ValueError(
            "mAP cannot rank profiles with missing values. Fill them with mt.pp.impute(adata), or drop the "
            "affected features with mt.pp.feature_select(adata, na_cutoff=0.0, subset=True); the default "
            "na_cutoff=0.05 keeps features with up to 5% missing values."
        )
    _refuse_unrankable(features, adata.obs_names, distance)
    obs = as_frame(adata.obs)
    # The scoring reads the Metadata_ columns plus every obs column the pair spec or label names, so a column
    # without the prefix (batch="source", a user-added label) is matched like any other.
    passthrough = needed | {label}
    meta = obs[[c for c in obs.columns if c.startswith("Metadata_") or c in passthrough]].reset_index(drop=True)
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

    # A positive-pair column holding a list per row is a multilabel annotation: a pair is positive when the two
    # label sets intersect. A profile with several labels is scored once per label, toward each of its classes.
    label_columns = [c for c in settings["pos_sameby"] if c != REFERENCE_COLUMN and c in meta.columns]
    if label_sep is not None:
        # Split a stored delimited string into the list of labels the detection below looks for.
        for column in label_columns:
            meta[column] = meta[column].map(lambda value: value.split(label_sep) if isinstance(value, str) else value)
    multilabel = [c for c in label_columns if _list_valued(meta[c])]
    if len(multilabel) > 1:
        raise ValueError(
            f"mAP scores against one multilabel column at a time, but these positive-pair columns are all "
            f"list-valued: {multilabel}. Keep one as the label and flatten the rest."
        )
    multilabel_col = multilabel[0] if multilabel else None
    if multilabel_col is not None:
        # A missing annotation becomes an empty label set: the row joins no class and is left unscored.
        meta[multilabel_col] = meta[multilabel_col].map(
            lambda value: list(value) if isinstance(value, _LIST_LIKE) else []
        )

    table, group_columns = _score_map(
        meta,
        features,
        settings,
        null_size=null_size,
        threshold=threshold,
        seed=seed,
        distance=distance,
        null=null,
        null_sampler=null_sampler,
        similarity_solver=similarity_solver,
        drop_positions=drop_positions,
        multilabel_col=multilabel_col,
    )

    if mode == "activity":
        control_groups = set(np.asarray(meta.loc[is_control, label], dtype=object))
        table = table[~table[label].isin(control_groups)].reset_index(drop=True)
        table = table.drop(columns=[REFERENCE_COLUMN], errors="ignore")

    adata.uns.setdefault("mantispy", {})[key_added] = table

    if multilabel_col is not None:
        # A row carries several labels, so there is no single per-row score to join back; the per-class table in
        # uns is the result, keyed by the label the score belongs to. Clear any per-row columns a single-label run
        # left behind, so obs cannot disagree with the table just written.
        stale = [c for c in (key_added, f"{key_added}_qvalue") if c in obs.columns]
        if stale:
            adata.obs = obs.drop(columns=stale)
        return None

    lookup = table.set_index(group_columns)
    index = pd.MultiIndex.from_frame(obs[group_columns]) if len(group_columns) > 1 else pd.Index(obs[group_columns[0]])
    adata.obs[key_added] = lookup["score"].reindex(index).to_numpy()
    adata.obs[f"{key_added}_qvalue"] = lookup["qvalue"].reindex(index).to_numpy()
    return None
