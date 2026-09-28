import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mantispy._core.mutation import inplace_or_copy
from mantispy._core.provenance import record_params


def _adata():
    return ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))


def test_record_params_makes_values_storable():
    adata = _adata()
    record_params(adata, "f", {"array": np.arange(3), "scalar": np.float32(1.5), "tup": (1, 2), "fn": len})
    stored = adata.uns["mantispy"]["params"]["f"]
    assert stored["array"] == [0, 1, 2]
    assert stored["scalar"] == pytest.approx(1.5)
    assert stored["tup"] == [1, 2]
    assert isinstance(stored["fn"], str)


@inplace_or_copy()
def _double(adata, factor: float = 2.0, note: str = "hi", copy: bool = False):
    adata.X = adata.X + factor


def test_decorator_rejects_a_function_with_no_arguments():
    with pytest.raises(TypeError, match="AnnData as its first argument"):

        @inplace_or_copy()
        def _bad():
            pass


def test_decorator_requires_copy_to_be_declared():
    """copy must be in the signature so users and IDEs can see it."""
    with pytest.raises(TypeError, match="must declare `copy"):

        @inplace_or_copy()
        def _no_copy(adata, factor: float = 1.0):
            pass


def test_modifying_a_view_in_place_is_refused():
    """anndata materialises a view on write, so the parent would keep none of the result."""
    import numpy as np

    adata = ad.AnnData(
        X=np.zeros((4, 2), dtype=np.float32),
        obs=pd.DataFrame({"g": ["a", "a", "b", "b"]}, index=list("0123")),
    )
    with pytest.raises(ValueError, match="cannot modify a view in place"):
        _double(adata[adata.obs["g"] == "a"])
    assert _double(adata[adata.obs["g"] == "a"], copy=True) is not None
