"""Recording how an object was made: result parameters under uns["mantispy"]["params"] and an ordered, append-only log under uns["mantispy"]["history"]."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData


def _mantispy_version() -> str:
    """The installed mantispy version, or ``"unknown"`` when the package is not installed."""
    try:
        return version("mantispy")
    except PackageNotFoundError:
        return "unknown"


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of a parameter value to something JSON can hold."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple | set | frozenset):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


def record_params(adata: AnnData, func_name: str, params: dict[str, Any]) -> None:
    """Record a call's parameters under ``uns['mantispy']['params'][func_name]``.

    This is the latest-call store a plot or downstream function reads for result-specific parameters, so a
    second call of the same function overwrites the first. The ordered record of every call lives in
    :func:`record_history`.
    """
    store = adata.uns.setdefault("mantispy", {}).setdefault("params", {})
    store[func_name] = {key: _jsonable(value) for key, value in params.items()}


def record_history(
    adata: AnnData,
    operation: str,
    *,
    params: dict[str, Any] | None = None,
    input: str = "X",
    output: str = "X",
    source: dict[str, Any] | None = None,
) -> None:
    """Append one record to the ordered, append-only ``uns['mantispy']['history']`` log (spec §14).

    Unlike :func:`record_params`, this never overwrites: running the same operation twice appends two
    records, so the list is the execution order. A level-changing operation passes ``source`` to record
    the resolution/grouping it came from and produced.

    Args:
        adata: The object whose history to extend.
        operation: The mantispy function, e.g. ``"pp.normalize"`` or ``"tl.aggregate"``.
        params: The call's parameters, coerced to JSON-safe values.
        input: Where the operation read from (``"X"`` or a layer name).
        output: Where it wrote to (``"X"`` or a layer name).
        source: For a level-changing operation, the source and target resolution/grouping.
    """
    record: dict[str, Any] = {
        "operation": operation,
        "params": {key: _jsonable(value) for key, value in (params or {}).items()},
        "input": input,
        "output": output,
        "mantispy_version": _mantispy_version(),
    }
    if source is not None:
        record["source"] = {key: _jsonable(value) for key, value in source.items()}
    store = adata.uns.setdefault("mantispy", {})
    # Each record is stored as a JSON string: anndata writes a list of strings to h5ad cleanly but cannot
    # serialize a list of dicts. A loaded history comes back as a numpy array, so coerce to a list first.
    existing = store.get("history")
    history = list(existing) if existing is not None else []
    history.append(json.dumps(record))
    store["history"] = history


def read_history(adata: AnnData) -> list[dict[str, Any]]:
    """The ordered history records of ``adata`` as dicts (decoding the stored JSON strings)."""
    return [json.loads(record) for record in adata.uns.get("mantispy", {}).get("history", [])]


def level_change_source(adata: AnnData, to_resolution: str, to_grouped_by: list[str]) -> dict[str, Any]:
    """The ``source`` record for a level-changing operation: the resolution and grouping it came from and produced."""
    from mantispy._core.schema import get_resolution

    store = adata.uns.get("mantispy", {})
    grouped_by = store.get("grouped_by")
    return {
        "resolution": get_resolution(adata, default="object"),
        "grouped_by": list(grouped_by) if grouped_by is not None else [],
        "to_resolution": to_resolution,
        "to_grouped_by": list(to_grouped_by),
    }
