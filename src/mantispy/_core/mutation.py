"""The in-place-or-copy contract every mutating function follows."""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from mantispy._core.provenance import record_params

F = TypeVar("F", bound=Callable[..., Any])

if TYPE_CHECKING:
    from anndata import AnnData


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
            # A `key_added` sends the result to a new layer, which a backed object takes in
            # memory, so only a call that leaves it unset is about to rewrite X.
            if backed and not copy and writes_layer and params.get("key_added") is None:
                raise ValueError(
                    f"{func.__name__} rewrites X, which a backed object holds read-only on disk. "
                    "Pass copy=True to get the result in memory, or call adata.to_memory() first."
                )
            # Copying a backed object needs a filename, so copy=True loads it into memory.
            target = (adata.to_memory() if backed else adata.copy()) if copy else adata
            if expects is not None:
                warn_resolution(target, expects)

            try:
                func(target, **params, **extra)
            except ValueError as error:
                # A function that drops rows or features cannot declare that in its signature,
                # so anndata is the one that refuses it, and its message names .to_memory()
                # but not this decorator's own way out.
                if backed and not copy and "backed mode" in str(error):
                    raise ValueError(
                        f"{func.__name__} drops rows or features, which a backed object cannot do in "
                        "place, because anndata cannot copy one without a filename. Pass copy=True to "
                        "get the result in memory, or call adata.to_memory() first."
                    ) from error
                raise

            written = params.get("key_added")
            name = f"{func.__name__}:{written}" if writes_layer and written else func.__name__
            record_params(target, name, {**params, **extra})
            return target if copy else None

        return wrapper  # type: ignore[return-value]

    return decorator
