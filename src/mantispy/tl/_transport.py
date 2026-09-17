"""Per-perturbation reproducibility across plates, batches or laboratories.

Aggregate reproducibility describes a whole screen. This module tests each perturbation
separately, so the result says which perturbations reproduced and which did not.

The unit of comparison is the effect, a group's median profile minus the median of the
controls in the same setting. Using each setting's own controls keeps a baseline offset
between settings from counting as disagreement.
``pp.normalize(by="Metadata_Plate", reference="negcon")`` usually removes that offset
already, so the comparison is about whether features respond the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._reduce import group_codes, representation
from mantispy._core._stats import benjamini_hochberg
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.masks import reference_mask
from mantispy._core.mutation import inplace_or_copy

WEIGHTS = ("activity", "equal")


def _effects(
    values: np.ndarray, units: np.ndarray, codes: np.ndarray, keys: pd.Index, is_control: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """Per setting, a groups-by-features effect matrix and each group's effect magnitude.

    Groups missing from a setting get NaN rows, so comparing two settings is one matrix
    product.
    """
    effect: dict[str, np.ndarray] = {}
    activity: dict[str, np.ndarray] = {}
    usable = []
    for unit in sorted(set(units.tolist())):
        here = units == unit
        if (here & is_control).sum() < 2:
            continue
        usable.append(unit)
        centre = np.nanmedian(values[here & is_control], axis=0)
        block = np.full((len(keys), values.shape[1]), np.nan)
        for index in range(len(keys)):
            rows = here & (codes == index) & ~is_control
            if rows.any():
                block[index] = np.nanmedian(values[rows], axis=0) - centre
        effect[unit] = block
        activity[unit] = np.linalg.norm(np.nan_to_num(block), axis=1)
        activity[unit][~np.isfinite(block).any(axis=1)] = np.nan
    return effect, activity, usable


def _quiet(block: np.ndarray) -> np.ndarray:
    """``block`` with all-NaN rows replaced by zeros, so ``nanmean`` does not warn."""
    empty = ~np.isfinite(block).any(axis=1)
    if not empty.any():
        return block
    out = block.copy()
    out[empty] = 0.0
    return out


def _standardize(block: np.ndarray) -> np.ndarray:
    """Rows centered and scaled to unit norm, so the dot product of two rows is a correlation.

    Missing values are filled with zero after centering, as :func:`~mantispy.tl.hit_calling`
    does. Without missing values the result is the Pearson correlation.
    """
    # Rows for groups missing from this setting are all-NaN and masked out downstream.
    with np.errstate(invalid="ignore"):
        centre = np.where(
            np.isfinite(block).any(axis=1, keepdims=True), np.nanmean(_quiet(block), axis=1, keepdims=True), 0.0
        )
    centred = np.nan_to_num(block - centre)
    norm = np.sqrt(np.einsum("ij,ij->i", centred, centred))[:, None]
    return np.divide(centred, norm, out=np.zeros_like(centred), where=norm > 0)


def _agreement_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Correlation of every group's effect in one setting with every group's effect in another.

    Entry ``[i, j]`` correlates group ``i``'s effect at the left setting with group ``j``'s
    at the right. The diagonal holds each group's statistic and the off-diagonal entries
    form the null, so the complete null costs one matrix product.
    """
    return _standardize(left) @ _standardize(right).T


def _level_pairs(frame: pd.DataFrame, levels: list[str], unit_key: str) -> dict[str, list[tuple[str, str]]]:
    """Unit pairs assigned to each level, keyed in the order of ``levels``.

    Each pair of units belongs to the coarsest level at which the two differ. Two plates of
    one laboratory separate at the plate level, and two plates of different laboratories at
    the laboratory level. Each pair is counted at one level only.
    """
    # drop=False: with a single level, the unit column is also the level column.
    lookup = frame.drop_duplicates(unit_key).set_index(unit_key, drop=False)
    units = list(lookup.index)
    buckets: dict[str, list[tuple[str, str]]] = {level: [] for level in levels}
    for i, one in enumerate(units):
        for other in units[:i]:
            for level in levels:  # coarsest first, so the first difference wins
                if lookup.loc[one, level] != lookup.loc[other, level]:
                    buckets[level].append((one, other))
                    break
    return buckets


@inplace_or_copy(expects=("well", "perturbation"))
def transport(
    adata: AnnData,
    by: str | list[str] = "Metadata_Plate",
    groupby: str = "Metadata_Perturbation",
    reference: str | None = "negcon",
    use_rep: str | None = None,
    weight: str = "activity",
    min_shared: int = 3,
    threshold: float = 0.05,
    key_added: str = "transport",
    copy: bool = False,
) -> AnnData | None:
    """Test whether each perturbation's effect reproduces across settings.

    Args:
        adata: Well-level profiles. Each setting needs its own reference wells, since effects are
            measured against them; settings with fewer than two are left out.
        by: ``obs`` column defining the setting, or a list of columns from coarsest to finest.
            ``["Metadata_Source", "Metadata_Plate"]`` reports agreement between plates of one
            source separately from agreement between sources, and the difference shows what a
            change of laboratory costs beyond a change of plate. The finest level defines the
            units that are compared.
        groupby: The perturbation column.
        reference: Which rows are the negative controls, per setting.
        use_rep: Score ``obsm[use_rep]`` instead of ``X``.
        weight: ``"activity"`` weights each comparison by the smaller of the two effect magnitudes,
            since the correlation of an inactive perturbation is noise. ``"equal"`` weights all
            comparisons the same.
        min_shared: Minimum number of shared perturbations for a pair of units to be compared.
        threshold: q-value cutoff for ``transports``. The null uses every mismatched pair of
            perturbations, so there is no null size or seed to set.
        key_added: Name for the outputs.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy. Writes ``uns["mantispy"][key_added]`` with one row per
        ``group`` and ``level`` and the columns ``n_pairs``, ``agreement``, ``pvalue``,
        ``qvalue`` and ``transports``. Writes ``uns["mantispy"][key_added + "_units"]``, the
        units-by-units agreement matrix drawn by :func:`~mantispy.pl.setting_agreement`, and
        joins the finest level's agreement back onto the rows as
        ``obs[key_added + "_agreement"]``.

    Notes:
        The null pairs a perturbation at one unit with a different perturbation at another, so
        a screen in which all effects look alike does not count as reproducing. This matters
        on screens with few mechanisms. The mismatched null centers at +0.26 on BBBC021 and at
        +0.03 on JUMP.

        The null is computed in full rather than sampled. Standardizing the effect vectors
        turns all group-against-group correlations into one matrix product, so 301 groups over
        48 unit pairs give 90,300 null values in about a second. A sampled null would floor the
        p-values at ``1/(n + 1)``, too coarse for Benjamini-Hochberg over every
        ``(group, level)`` row, and the number of groups called would depend on the number of
        draws.

        This is an observational measure. It shows whether an effect reproduced at another
        site, not what the effect would have been there.
    """
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}, got {weight!r}")
    levels = [by] if isinstance(by, str) else list(by)
    obs = as_frame(adata.obs)
    if missing := [level for level in levels if level not in obs.columns]:
        raise KeyError(f"obs has no column(s) {missing}; by= names the setting(s) to compare across")

    unit_key = levels[-1]
    units = obs[unit_key].astype(str).to_numpy()
    if len(set(units.tolist())) < 2:
        raise ValueError(
            f"transport compares settings, and obs[{unit_key!r}] has one level. Pass by= a column that "
            "varies, such as the plate, batch or laboratory."
        )

    values = representation(adata, use_rep)
    is_control = reference_mask(adata, reference)
    codes, keys = group_codes(adata, groupby)
    effect, activity, usable = _effects(values, units, codes, keys, is_control)
    if len(usable) < 2:
        raise ValueError(
            f"only {len(usable)} level(s) of {unit_key!r} have at least two reference rows. Effects are "
            "measured against each setting's own controls, so at least two settings need them."
        )

    frame = obs.loc[:, levels].astype(str).assign(**{unit_key: units})
    buckets = _level_pairs(frame, levels, unit_key)
    buckets = {
        level: [pair for pair in pairs if pair[0] in usable and pair[1] in usable] for level, pairs in buckets.items()
    }

    records: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    for level, pairs in buckets.items():
        if not pairs:
            get_logger().info("transport has no unit pair separating at %r; skipping that level", level)
            continue

        # Average the agreement matrices over unit pairs. The diagonal is each group's
        # statistic and the off-diagonal entries are the null.
        total = np.zeros((len(keys), len(keys)))
        weights = np.zeros_like(total)
        counted = np.zeros(len(keys), dtype=int)
        for one, other in pairs:
            left, right = effect[one], effect[other]
            here = np.isfinite(left).any(axis=1)
            there = np.isfinite(right).any(axis=1)
            if int((here & there).sum()) < min_shared:
                continue
            block = _agreement_matrix(left, right)
            if weight == "activity":
                pair_weight = np.minimum(activity[one][:, None], activity[other][None, :])
            else:
                pair_weight = np.ones_like(block)
            pair_weight = np.where(here[:, None] & there[None, :], np.nan_to_num(pair_weight), 0.0)
            total += np.nan_to_num(block) * pair_weight
            weights += pair_weight
            counted += (here & there).astype(int)
            shared = np.flatnonzero(here & there)
            matrix_rows.extend(
                {
                    "left": one,
                    "right": other,
                    "group": str(keys[int(index)]),
                    "agreement": float(block[index, index]),
                    "weight": float(pair_weight[index, index]),
                }
                for index in shared
            )

        with np.errstate(invalid="ignore", divide="ignore"):
            combined = np.where(weights > 0, total / np.where(weights > 0, weights, 1.0), np.nan)
        observed = np.diag(combined)
        off_diagonal = combined[~np.eye(len(keys), dtype=bool)]
        null = off_diagonal[np.isfinite(off_diagonal)]

        for index in range(len(keys)):
            if not counted[index] or not np.isfinite(observed[index]):
                continue
            pvalue = float((np.sum(null >= observed[index]) + 1) / (null.size + 1)) if null.size else np.nan
            records.append(
                {
                    "group": str(keys[int(index)]),
                    "level": level,
                    "n_pairs": int(counted[index]),
                    "agreement": float(observed[index]),
                    "pvalue": pvalue,
                }
            )

    table = pd.DataFrame(records, columns=["group", "level", "n_pairs", "agreement", "pvalue"])
    if len(table):
        table["qvalue"] = benjamini_hochberg(table["pvalue"].to_numpy())
        table["transports"] = table["qvalue"] < threshold
    else:
        table["qvalue"] = []
        table["transports"] = []

    adata.uns.setdefault("mantispy", {})[key_added] = table
    pairs_frame = pd.DataFrame(matrix_rows, columns=["left", "right", "group", "agreement", "weight"])
    adata.uns["mantispy"][f"{key_added}_units"] = _unit_matrix(pairs_frame, usable, weight)

    finest = table[table["level"] == unit_key] if unit_key in set(table["level"]) else table
    lookup = finest.drop_duplicates("group").set_index("group")["agreement"]
    labels = obs[groupby].astype(str)
    adata.obs[f"{key_added}_agreement"] = lookup.reindex(labels).to_numpy()
    for level in buckets:
        called = table[(table["level"] == level) & table["transports"]]
        get_logger().info(
            "transport(%s): %d of %d group(s) reproduce", level, len(called), int((table["level"] == level).sum())
        )
    return None


def _unit_matrix(pairs: pd.DataFrame, units: list[str], weight: str) -> pd.DataFrame:
    """Units by units: how well two settings agree over the perturbations they share."""
    values = np.full((len(units), len(units)), np.nan)
    np.fill_diagonal(values, 1.0)
    matrix = pd.DataFrame(values, index=units, columns=units, dtype=float)
    if not len(pairs):
        return matrix
    for (one, other), block in pairs.groupby(["left", "right"], observed=True):
        weights = block["weight"].to_numpy() if weight == "activity" else np.ones(len(block))
        if weights.sum() <= 0:
            weights = np.ones(len(block))
        value = float(np.average(block["agreement"].to_numpy(), weights=weights))
        matrix.loc[one, other] = matrix.loc[other, one] = value
    return matrix
