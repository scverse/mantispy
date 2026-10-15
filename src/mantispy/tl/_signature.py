from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core.features import annotation
from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.provenance import level_change_source, record_history
from mantispy._core.schema import stamp

DEFAULT_BY = ("feature_group", "channel", "object")

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
        by: ``var`` columns whose combination names a family.
            The default is the feature group, the channel it was measured in, and the compartment it was measured on.
        statistic: Column of the table to average within each family, such as ``"t"`` or ``"difference"`` (the raw contrast).

    Returns:
        A new :class:`~anndata.AnnData` of perturbations by families, with each family's ``by`` columns and ``n_features`` in ``var``, beside the schema's remaining annotation columns, left empty because a family is not a measurement they describe.
        A ``by`` column carries the family's own value, in the dtype ``var`` held it in; the ``"none"`` that names a family whose features had no value appears in the family's name, not in the column.
        It is a perturbation-level profile object, so :func:`~scanpy.pp.neighbors`, :func:`~scanpy.tl.leiden` and :func:`~mantispy.tl.nn_moa_classify` accept it.

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
    gaps = components.isna()
    if gaps.all().all():
        raise ValueError(
            f"var's {list(by)} name no family: every one is empty, so every feature would join into "
            "a single column averaging the whole object. mt.io.read_profiles writes the parsed "
            "annotation; an object whose feature names carry no structure has no families to collapse."
        )
    # Masked before astype: pandas 2 turns a missing value into the string "nan", which fillna cannot see.
    labels = components.astype(str).mask(gaps, "none")
    family = labels.iloc[:, 0].str.cat(labels.iloc[:, 1:], sep=SEPARATOR) if len(by) > 1 else labels.iloc[:, 0]
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
    # From components, not labels or the split name, so a by column keeps its dtype and true missing values.
    parts = (
        components.assign(__family__=family).drop_duplicates("__family__").set_index("__family__").reindex(wide.columns)
    )

    var = annotation(
        wide.columns.rename(None),
        **{column: parts[column] for column in by},
        n_features=family.value_counts().reindex(wide.columns),
        feature_kind="derived",
    )

    signature = ad.AnnData(
        X=wide.to_numpy(dtype=np.float32),
        # rename, not pd.Index(..., name=None), which pandas reads as "keep the name".
        obs=pd.DataFrame(index=wide.index.astype(str).rename(None)),
        var=var,
    )
    signature.obs["Metadata_Perturbation"] = signature.obs_names.to_numpy()
    obs = as_frame(adata.obs)
    if "Metadata_Perturbation" in obs.columns:
        # Carry every column constant within a perturbation (spec §13.1), not just Metadata_ ones, and
        # drop a column that varies within a perturbation rather than copy an arbitrary first value.
        carried = [column for column in obs.columns if column != "Metadata_Perturbation"]
        grouped = obs[carried].groupby(obs["Metadata_Perturbation"], observed=True)
        constant = grouped.nunique(dropna=False).le(1).all()
        firsts = grouped.first()
        for column in carried:
            if not constant[column]:
                get_logger().debug("feature_signature dropped non-constant column %s", column)
                continue
            values = firsts[column].reindex(signature.obs_names)
            if values.notna().any():
                signature.obs[column] = values.to_numpy()
    store = adata.uns.get("mantispy", {})
    stamp(signature, resolution="aggregate", grouped_by=["Metadata_Perturbation"], history=store.get("history"))
    record_history(
        signature,
        "feature_signature",
        params={"key": key, "by": list(by), "statistic": statistic},
        source=level_change_source(adata, "aggregate", ["Metadata_Perturbation"]),
    )
    get_logger().info(
        "feature_signature: %d perturbations by %d families, from %d features",
        signature.n_obs,
        signature.n_vars,
        adata.n_vars,
    )
    return signature
