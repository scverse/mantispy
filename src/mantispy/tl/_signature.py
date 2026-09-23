"""Per-feature statistics collapsed into feature families.

A differential table has one row per perturbation and feature, and a screen has thousands of features.
A family of features, such as intensity in the tubulin channel or texture in the nucleus, is easier to interpret.

On BBBC021, mechanism retrieval from 19 families scores 0.631, against 0.769 from all 344 individual features, which keeps about 82% of the signal.
Dropping the channel from the grouping leaves four families and scores 0.340.
"""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core.features import annotation
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.schema import stamp

#: Default ``var`` columns that define a family; see the module docstring for its BBBC021 score.
DEFAULT_BY = ("feature_group", "channel", "object")

#: What joins a family's components into its name.
SEPARATOR = " | "


def feature_signature(
    adata: AnnData,
    key: str = "differential",
    by: Sequence[str] = DEFAULT_BY,
    statistic: str = "t",
) -> AnnData:
    """Collapse a differential table into a perturbation-by-feature-family matrix.

    Args:
        adata: Object carrying ``uns["mantispy"][key]``, as written by :func:`~mantispy.tl.differential_features`.
        key: Which table to collapse.
        by: ``var`` columns whose combination names a family. The default is the feature group, the channel it was measured in, and the compartment it was measured on.
        statistic: Column of the table to average within each family, such as ``"t"`` or ``"difference"`` (the raw contrast).

    Returns:
        A new :class:`~anndata.AnnData` of perturbations by families, with each family's ``by`` columns and ``n_features`` in ``var``, beside the schema's remaining annotation columns, left empty because a family is not a measurement they describe.
        A ``by`` column carries the family's own value, in the dtype ``var`` held it in; the ``"none"`` that names a family whose features had no value appears in the family's name, not in the column.
        It is a perturbation-level profile object, so ``sc.pp.neighbors``, ``sc.tl.leiden`` and :func:`~mantispy.tl.nn_moa_classify` accept it.

    Raises:
        KeyError: ``uns["mantispy"][key]`` is missing, that table has no ``statistic`` column, or ``var`` is missing one of the ``by`` columns.
        ValueError: ``by`` names the same column more than once, which would give one family two copies of a component; every ``by`` column is empty, so there is no family to name; or two different sets of values name the same family.

    Notes:
        Signed statistics are averaged, so a family that decreased stays distinct from one that increased.
        On BBBC021, microtubule stabilizers score +2.2 on ``Intensity | CorrTub | Nuclei``, while destabilizers score +1.4 on ``Intensity | CorrActin | Nuclei`` with no tubulin signal; as the microtubules come apart the cells round up, which is the largest remaining change.
    """
    store = adata.uns.get("mantispy", {})
    if key not in store:
        raise KeyError(f"uns['mantispy'][{key!r}] is missing; run mt.tl.differential_features first")
    table = pd.DataFrame(store[key])
    if statistic not in table.columns:
        raise KeyError(f"the {key!r} table has no column {statistic!r}; it holds {list(table.columns)}")

    var = as_frame(adata.var)
    missing = [column for column in by if column not in var.columns]
    if missing:
        raise KeyError(f"var is missing the column(s) that name a feature family: {missing}")
    if pd.Index(by).has_duplicates:
        raise ValueError(f"by names the same column more than once: {list(by)}")

    components = var[list(by)]
    if components.isna().all().all():
        raise ValueError(
            f"var's {list(by)} name no family: every one is empty, so every feature would join into "
            "a single column averaging the whole object. mt.io.read_profiles writes the parsed "
            "annotation; an object whose feature names carry no structure has no families to collapse."
        )
    # Masked on the original frame rather than filled after astype: pandas 3 keeps a missing value
    # through astype(str) and pandas 2 turns it into the string "nan", which fillna cannot see.
    labels = components.astype(str).mask(components.isna(), "none")
    # str.cat, not a per-row apply: same bytes, and the apply is a Python loop over every feature.
    family = labels.iloc[:, 0].str.cat(labels.iloc[:, 1:], sep=SEPARATOR) if len(by) > 1 else labels.iloc[:, 0]
    # Compared on the names, not the raw components: a missing value and a literal "none" are the
    # same family, and only a component that itself holds SEPARATOR can make two different names
    # collide. Nothing the parser writes can -- it joins multi-channel values with a bare "|" --
    # so this catches a var assembled by hand.
    distinct = labels.assign(__family__=family).drop_duplicates()["__family__"]
    if distinct.duplicated().any():
        clash = sorted(set(distinct[distinct.duplicated(keep=False)]))[:3]
        raise ValueError(
            f"different {list(by)} values name the same family: {clash}. A value holds the "
            f"{SEPARATOR!r} that joins them, so one family's columns would be averaged with another's; "
            "rename those values, or group on columns that do not hold it"
        )
    table = table.join(family.rename("__family__"), on="feature")

    wide = table.pivot_table(index="group", columns="__family__", values=statistic, aggfunc="mean", observed=True)
    # Take the components from var rather than splitting the joined name, which a value holding the
    # separator would break, and from `components` rather than `labels`, so that a by column
    # that is also a schema column keeps its own dtype and its true missing values instead of the strings and
    # the "none" sentinel that name the family.
    parts = components.assign(__family__=family).drop_duplicates("__family__").set_index("__family__")
    parts = parts.reindex(wide.columns)
    parts["n_features"] = family.value_counts().reindex(wide.columns)

    var = annotation(
        wide.columns.rename(None),
        **{column: parts[column] for column in by},
        n_features=parts["n_features"],
    )

    signature = ad.AnnData(
        X=wide.to_numpy(dtype=np.float32),
        # rename, not pd.Index(..., name=None): None is pandas' "keep the name", so the pivot's key
        # rode into obs and onto the file's obs index.
        obs=pd.DataFrame(index=wide.index.astype(str).rename(None)),
        var=var,
    )
    signature.obs["Metadata_Perturbation"] = signature.obs_names.to_numpy()
    # Copy the other per-perturbation Metadata_ columns so the result can be scored.
    obs = as_frame(adata.obs)
    if "Metadata_Perturbation" in obs.columns:
        per_group = obs.drop_duplicates("Metadata_Perturbation").set_index("Metadata_Perturbation")
        for column in per_group.columns:
            if column.startswith("Metadata_") and column != "Metadata_Perturbation":
                values = per_group[column].reindex(signature.obs_names)
                if values.notna().any():
                    signature.obs[column] = values.to_numpy()
    stamp(signature, resolution="perturbation")
    get_logger().info(
        "feature_signature: %d perturbations by %d families, from %d features",
        signature.n_obs,
        signature.n_vars,
        adata.n_vars,
    )
    return signature
