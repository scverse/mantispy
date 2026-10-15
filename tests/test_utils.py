import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mantispy._core.mutation import inplace_or_copy
from mantispy._core.provenance import read_history, record_params


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


def test_a_decorated_op_appends_an_append_only_history_record():
    """§14.1/§14.2: every mutating call appends one record; a second call appends a second (no overwrite)."""
    adata = _adata()
    _double(adata, factor=3.0)
    _double(adata, factor=3.0)
    history = read_history(adata)
    assert [record["operation"] for record in history] == ["_double", "_double"]
    assert history[0]["params"]["factor"] == pytest.approx(3.0)
    assert history[0]["input"] == "X" and history[0]["output"] == "X"
    assert "mantispy_version" in history[0]
    # params (the latest-call store) keeps only one; history keeps both.
    assert list(adata.uns["mantispy"]["params"]) == ["_double"]


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
