"""Shared plumbing: logging, provenance, and the in-place/copy contract."""

from __future__ import annotations

import functools
import inspect
import json
import logging
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from anndata import AnnData

_LOGGER = logging.getLogger("mantispy")

F = TypeVar("F", bound=Callable[..., Any])


def get_logger() -> logging.Logger:
    """Return the package logger."""
    return _LOGGER


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
    """Record a call's parameters under ``uns['mantispy']['params'][func_name]``."""
    store = adata.uns.setdefault("mantispy", {}).setdefault("params", {})
    store[func_name] = {key: _jsonable(value) for key, value in params.items()}


def as_frame(obj: Any) -> pd.DataFrame:
    """Narrow ``adata.obs``/``adata.var`` to a DataFrame.

    anndata types these as ``DataFrame | Dataset2D`` because a backed object can hold a
    lazy table. mantispy works on in-memory tables, so the cast is made here once instead
    of at every call site.
    """
    return cast("pd.DataFrame", obj)


def feature_mask(adata: AnnData, key: str | None) -> np.ndarray:
    """Boolean mask over ``var``: the features flagged by ``key``, or all of them.

    A missing column is not an error. Callers pass ``key="selected"`` by default, so running
    after :func:`~mantispy.pp.feature_select` uses the selection and running before it uses
    every feature.
    """
    if key is not None and key in adata.var:
        return as_frame(adata.var)[key].to_numpy(dtype=bool)
    if key is not None:
        get_logger().debug("var has no column %r; using every feature", key)
    return np.ones(adata.n_vars, dtype=bool)


def reference_mask(adata: AnnData, reference: str | None) -> np.ndarray:
    """Boolean mask over ``obs``: the rows a transform should be fitted on.

    ``None`` fits on everything, ``"negcon"`` on ``Metadata_Control``, and anything else
    names a boolean ``obs`` column.
    """
    if reference is None:
        return np.ones(adata.n_obs, dtype=bool)

    column = "Metadata_Control" if reference == "negcon" else reference
    if column not in adata.obs:
        extra = " Run mt.pp.annotate_controls to create it." if column == "Metadata_Control" else ""
        raise KeyError(f"obs has no column {column!r} to use as reference.{extra}")

    values = pd.Series(adata.obs[column])
    missing = int(values.isna().sum())
    if missing:
        raise ValueError(
            f"obs[{column!r}] has {missing} missing value(s) and cannot be used as a reference flag, "
            "because NaN coerces to True and would mark those rows as controls. Fill them, or check "
            "that the platemap covers every well."
        )

    known = set(values.unique())
    # An h5ad round trip can bring a bool column back as a category of "True"/"False".
    if known <= {"True", "False"}:
        return (values == "True").to_numpy()
    if not (pd.api.types.is_bool_dtype(values) or known <= {0, 1}):
        raise TypeError(
            f"obs[{column!r}] must be boolean to select reference rows, got dtype {values.dtype}. "
            "A string column would select every row."
        )
    return values.to_numpy(dtype=bool)


def categorize_metadata(obs: pd.DataFrame) -> pd.DataFrame:
    """Convert low-cardinality string ``obs`` columns to ``category``.

    Plate, well, perturbation and batch repeat across millions of rows. As objects they
    cost a pointer plus a string each; as categories, one int8 or int16 code. It also
    stops anndata printing "storing X as categorical" on every construction.
    """
    for column in obs.columns:
        values = obs[column]
        if isinstance(values.dtype, pd.CategoricalDtype):
            continue
        # pandas 3 infers StringDtype where pandas 2 gave object, so check both.
        if values.dtype == object or isinstance(values.dtype, pd.StringDtype):
            obs[column] = values.astype("category")
    return obs


def warn_resolution(adata: AnnData, expected: str | tuple[str, ...]) -> None:
    """Warn, without raising, when the recorded resolution is not one of ``expected``.

    Resolution is advisory, since a user may run an aggregated-profile tool on single cells
    on purpose.
    """
    accepted = (expected,) if isinstance(expected, str) else tuple(expected)
    actual = adata.uns.get("mantispy", {}).get("resolution")
    if actual is not None and actual not in accepted:
        wanted = " or ".join(repr(name) for name in accepted)
        warnings.warn(
            f"this function expects {wanted} resolution but the object is annotated "
            f"{actual!r}; results may not mean what you expect",
            UserWarning,
            stacklevel=3,
        )


def inplace_or_copy(expects: str | tuple[str, ...] | None = None) -> Callable[[F], F]:
    """Give a function the standard mantispy mutation contract.

    The wrapped function receives the object to mutate and mutates it. The decorator
    handles the rest:

    * ``copy=True`` hands the function a copy and returns it; ``copy=False`` mutates
      the caller's object and returns ``None``.
    * The call's parameters are recorded into ``uns["mantispy"]["params"]``, read from
      the signature rather than a hand-written dict, so provenance cannot drift from the
      arguments the function accepts.
    * ``expects`` optionally emits the advisory resolution warning.

    Only for functions that mutate in place. Functions that return a new object
    (``tl.aggregate``, ``pp.subset_features``) do not use it.
    """

    def decorator(func: F) -> F:
        signature = inspect.signature(func)
        parameters = list(signature.parameters.values())
        if not parameters:
            raise TypeError(f"{func.__name__} must take an AnnData as its first argument")
        first = parameters[0].name
        var_keyword = next((p.name for p in parameters if p.kind is p.VAR_KEYWORD), None)
        # `key_added: str | None = None` means "write to this layer instead of X", so two
        # calls with different layers produce two different matrices and need two
        # provenance entries. `key_added: str = "something"` names a column and does not.
        layer_key = signature.parameters.get("key_added")
        writes_layer = layer_key is not None and layer_key.default is None
        if "copy" not in signature.parameters:
            raise TypeError(
                f"{func.__name__} must declare `copy: bool = False` so that it shows up in "
                "its signature and documentation"
            )

        @functools.wraps(func)
        def wrapper(adata: AnnData, *args: Any, **kwargs: Any) -> AnnData | None:
            bound = signature.bind(adata, *args, **kwargs)
            bound.apply_defaults()
            params = dict(bound.arguments)
            params.pop(first)
            # Read `copy` from the bound arguments, so passing it positionally works too.
            copy = bool(params.pop("copy", False))
            extra = params.pop(var_keyword, {}) if var_keyword else {}

            if not copy and getattr(adata, "is_view", False):
                raise ValueError(
                    f"{func.__name__} cannot modify a view in place, because anndata would materialize "
                    "the view and the parent object would keep none of the result. Pass copy=True, or "
                    "call .copy() on the subset first."
                )
            backed = bool(getattr(adata, "isbacked", False))
            if backed and not copy and writes_layer:
                raise ValueError(
                    f"{func.__name__} rewrites X, which a backed object holds read-only on disk. "
                    "Pass copy=True to get the result in memory, or call adata.to_memory() first."
                )
            # Copying a backed object needs a filename, so copy=True loads it into memory.
            target = (adata.to_memory() if backed else adata.copy()) if copy else adata
            if expects is not None:
                warn_resolution(target, expects)

            func(target, **params, **extra)

            written = params.get("key_added")
            name = f"{func.__name__}:{written}" if writes_layer and written else func.__name__
            record_params(target, name, {**params, **extra})
            return target if copy else None

        return wrapper  # type: ignore[return-value]

    return decorator


def report_drop(what: str, dropped: int, total: int, remedy: str = "", escalate: bool = True) -> None:
    """Log that rows or features were discarded, as a warning when half or more are dropped.

    Dropping 2 features of 3634 is routine and logged at info level. Dropping half or more
    is logged as a warning, since it can leave an object too small to use.

    ``escalate=False`` keeps the message at info level however much of the input is
    removed. It is for exclusions that are the documented default rather than a loss:
    reading a CellProfiler export drops whole-field ``Image_`` measurements by design, and
    a normal four-image export can have more of those than per-cell ones. A warning there
    would teach users to ignore warnings.
    """
    if dropped <= 0:
        return
    message = "dropped %d of %d %s" + (f"; {remedy}" if remedy else "")
    heavy = escalate and total and dropped >= max(total // 2, 1)
    log = get_logger().warning if heavy else get_logger().info
    log(message, dropped, total, what)
