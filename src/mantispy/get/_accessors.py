"""Accessors that turn a mantispy AnnData into plain Python objects."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core._reduce import get_matrix
from mantispy._core.features import canonical_channel, unparsed
from mantispy._core.frames import as_frame
from mantispy._core.masks import feature_mask, reference_mask


def features(
    adata: AnnData,
    object: str | None = None,
    feature_group: str | None = None,
    channel: str | None = None,
    key: str | None = None,
    canonical_channels: bool = False,
) -> list[str]:
    """Names of the features matching every filter given.

    Args:
        adata: Object to query.
        object: Exact matches against the parsed ``var`` columns of the same name.
        feature_group: Exact matches against the parsed ``var`` columns of the same name.
        channel: Matches any component of a multi-channel value, so ``"DNA"`` selects ``"DNA|ER"`` colocalization features too.
        key: Restrict to features where this boolean ``var`` column is true, e.g. ``"selected"``.
        canonical_channels: Compare channels through ``canonical_channel``, so that ``channel="DNA"`` also matches a dataset whose nuclear stain is named ``Hoechst`` or ``DAPI``.

    Returns:
        Feature names, in ``var`` order.

    Raises:
        KeyError: `key` names a column ``var`` does not have, or a column filtered on is absent or entirely empty.
    """
    filters = {"object": object, "feature_group": feature_group, "channel": channel}
    # A column supplied empty is present, so filtering on it silently matched nothing.
    if unusable := unparsed(as_frame(adata.var), [name for name, value in filters.items() if value is not None]):
        raise KeyError(
            f"var has no usable column(s) {unusable} to filter on; they are absent, or present with "
            "nothing in them. mt.io.read_profiles writes the parsed annotation, and mt.io.stamp "
            "supplies these columns empty for an object whose names carry no structure."
        )

    mask = np.ones(adata.n_vars, dtype=bool)
    if object is not None:
        mask &= (adata.var["object"] == object).to_numpy()
    if feature_group is not None:
        mask &= (adata.var["feature_group"] == feature_group).to_numpy()
    if channel is not None:
        wanted = canonical_channel(channel) if canonical_channels else channel
        mask &= np.array(
            [
                wanted in str(canonical_channel(value) if canonical_channels else value).split("|")
                for value in adata.var["channel"]
            ],
            dtype=bool,
        )
    if key is not None:
        if key not in adata.var:
            raise KeyError(f"var has no column {key!r}")
        mask &= feature_mask(adata, key)
    return adata.var_names[mask].tolist()


def to_dataframe(
    adata: AnnData,
    layer: str | None = None,
    metadata: bool = True,
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Wide, pycytominer-shaped frame: ``Metadata_`` columns first, then features.

    Use it to pass profiles to pycytominer, cytominer-eval and other tools that expect a flat table.

    Args:
        adata: Object to read.
        layer: Layer to read instead of ``X``.
        metadata: Put the ``Metadata_`` columns of ``obs`` in front of the feature columns.
        features: Features to take, in the order given, defaulting to every feature in ``var`` order.

    Returns:
        A frame indexed by ``obs_names``, with one column per feature and, unless `metadata` is off, the ``Metadata_`` columns of ``obs`` before them.

    Raises:
        KeyError: A name in `features` is not in ``var_names``.
    """
    names = list(adata.var_names) if features is None else list(features)
    matrix = get_matrix(adata, layer)
    if features is not None:
        positions = np.asarray(adata.var_names.get_indexer(pd.Index(names)))
        if (positions < 0).any():
            unknown = [name for name, position in zip(names, positions, strict=True) if position < 0]
            raise KeyError(f"features not in var_names: {unknown[:5]}")
        matrix = matrix[:, positions]

    values = pd.DataFrame(matrix, index=adata.obs_names, columns=names)
    if not metadata:
        return values
    obs = as_frame(adata.obs)
    meta_columns = [column for column in obs.columns if column.startswith("Metadata_")]
    return pd.concat([obs[meta_columns], values], axis=1)


def controls(adata: AnnData, kind: str = "negcon") -> np.ndarray:
    """Boolean mask of control rows.

    Args:
        adata: Object to read.
        kind: ``"negcon"`` reads ``Metadata_Control``; ``"poscon"`` reads ``Metadata_Control_Type == "poscon"`` and is all-``False`` when that column is absent.

    Returns:
        A boolean mask over ``obs``, true on the control rows.

    Raises:
        ValueError: `kind` is neither ``"negcon"`` nor ``"poscon"``, or ``Metadata_Control`` has missing values.
        KeyError: ``"negcon"`` was asked for and ``obs`` has no ``Metadata_Control`` column.
        TypeError: ``Metadata_Control`` is not boolean, so it would select every row.
    """
    if kind == "negcon":
        return reference_mask(adata, "negcon")
    if kind == "poscon":
        if "Metadata_Control_Type" not in adata.obs:
            return np.zeros(adata.n_obs, dtype=bool)
        return (as_frame(adata.obs)["Metadata_Control_Type"] == "poscon").to_numpy(dtype=bool)
    raise ValueError(f"kind must be 'negcon' or 'poscon', got {kind!r}")
