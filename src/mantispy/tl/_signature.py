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

from mantispy._core.features import empty_annotation
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.schema import stamp

#: Default ``var`` columns that define a family; see the module docstring for its BBBC021 score.
DEFAULT_BY = ("feature_group", "channel", "object")


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
        A new :class:`~anndata.AnnData` of perturbations by families, with each family's ``by`` columns and ``n_features`` in ``var``, beside the schema's annotation columns left empty for the families they do not describe.
        It is a perturbation-level profile object, so ``sc.pp.neighbors``, ``sc.tl.leiden`` and :func:`~mantispy.tl.nn_moa_classify` accept it.

    Raises:
        KeyError: ``uns["mantispy"][key]`` is missing, that table has no ``statistic`` column, or ``var`` is missing one of the ``by`` columns.

    Notes:
        Signed statistics are averaged, so a family that decreased stays distinct from one that increased.
        On BBBC021, microtubule stabilizers score +2.2 on ``Intensity|CorrTub|Nuclei``, while destabilizers score +1.4 on ``Intensity|CorrActin|Nuclei`` with no tubulin signal; as the microtubules come apart the cells round up, which is the largest remaining change.
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

    labels = var[list(by)].astype(str).fillna("none")
    family = labels.apply(lambda row: " | ".join(row), axis=1)
    table = table.join(family.rename("__family__"), on="feature")

    wide = table.pivot_table(index="group", columns="__family__", values=statistic, aggfunc="mean", observed=True)
    # Take the components from var rather than splitting the joined name, since a feature group or channel may contain the separator (rohban2017's do).
    parts = labels.assign(__family__=family).drop_duplicates("__family__").set_index("__family__")
    parts = parts.reindex(wide.columns)
    parts["n_features"] = family.value_counts().reindex(wide.columns).to_numpy()

    # A family is not a CellProfiler measurement, so the schema's annotation columns are supplied empty
    # rather than guessed at, and the columns that name the family are written over them.
    annotation = empty_annotation(wide.columns)
    for column in by:
        annotation[column] = parts[column].to_numpy()
    annotation["n_features"] = parts["n_features"].to_numpy()

    signature = ad.AnnData(
        X=wide.to_numpy(dtype=np.float32),
        obs=pd.DataFrame(index=pd.Index(wide.index.astype(str), name=None)),
        var=annotation,
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
