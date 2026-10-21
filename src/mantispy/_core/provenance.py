"""Recording how an object was made: result parameters under uns["mantispy"]["params"]."""

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
    second call of the same function overwrites the first.
    """
    store = adata.uns.setdefault("mantispy", {}).setdefault("params", {})
    store[func_name] = {key: _jsonable(value) for key, value in params.items()}
