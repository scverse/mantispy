import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mantispy._core._utils import inplace_or_copy, record_params, warn_resolution


def _adata():
    return ad.AnnData(X=np.zeros((2, 2), dtype=np.float32))


def test_record_params_nests_under_func_name():
    adata = _adata()
    record_params(adata, "normalize", {"method": "mad_robustize"})
    assert adata.uns["mantispy"]["params"]["normalize"]["method"] == "mad_robustize"


def test_record_params_makes_values_storable():
    adata = _adata()
    record_params(adata, "f", {"array": np.arange(3), "scalar": np.float32(1.5), "tup": (1, 2), "fn": len})
    stored = adata.uns["mantispy"]["params"]["f"]
    assert stored["array"] == [0, 1, 2]
    assert stored["scalar"] == pytest.approx(1.5)
    assert stored["tup"] == [1, 2]
    assert isinstance(stored["fn"], str)


def test_warn_resolution_warns_but_never_raises():
    adata = _adata()
    adata.uns["mantispy"] = {"resolution": "cell"}
    with pytest.warns(UserWarning, match="resolution"):
        warn_resolution(adata, "well")


def test_warn_resolution_is_silent_when_unset_or_matching():
    adata = _adata()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_resolution(adata, "well")
        adata.uns["mantispy"] = {"resolution": "well"}
        warn_resolution(adata, "well")


# --- the mutation contract -------------------------------------------------


@inplace_or_copy()
def _double(adata, factor: float = 2.0, note: str = "hi", copy: bool = False):
    adata.X = adata.X + factor


@inplace_or_copy(expects="well")
def _needs_wells(adata, copy: bool = False):
    adata.X = adata.X + 1


@inplace_or_copy()
def _with_kwargs(adata, base: int = 1, copy: bool = False, **extra):
    adata.uns["seen"] = dict(extra)


def test_copy_false_mutates_and_returns_none():
    adata = _adata()
    assert _double(adata) is None
    assert adata.X[0, 0] == 2.0


def test_copy_true_leaves_the_original_alone():
    adata = _adata()
    result = _double(adata, copy=True)
    assert result is not adata
    assert adata.X[0, 0] == 0.0
    assert result.X[0, 0] == 2.0


def test_params_come_from_the_real_signature_including_defaults():
    """Provenance is read from the signature, so it cannot drift from the arguments."""
    adata = _adata()
    _double(adata, factor=3.0)
    assert adata.uns["mantispy"]["params"]["_double"] == {"factor": 3.0, "note": "hi"}


def test_var_keyword_arguments_are_forwarded_and_recorded():
    adata = _adata()
    _with_kwargs(adata, extra_option=7)
    assert adata.uns["seen"] == {"extra_option": 7}
    assert adata.uns["mantispy"]["params"]["_with_kwargs"] == {"base": 1, "extra_option": 7}


def test_expects_triggers_the_resolution_warning():
    adata = _adata()
    adata.uns["mantispy"] = {"resolution": "cell"}
    with pytest.warns(UserWarning, match="well"):
        _needs_wells(adata)


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


def test_copy_is_visible_in_the_public_signature():
    import inspect

    assert "copy" in inspect.signature(_double).parameters


def test_copy_passed_positionally_is_honoured():
    """copy is an ordinary positional-or-keyword parameter, so it must work either way.
    Reading it from kwargs alone would mutate the caller's object and return None."""
    adata = _adata()
    result = _double(adata, 2.0, "hi", True)  # copy=True, positionally
    assert result is not None
    assert adata.X[0, 0] == 0.0  # source untouched
    assert result.X[0, 0] == 2.0


def test_modifying_a_view_in_place_is_refused():
    """anndata materialises a view on write, so the parent would keep none of the result."""
    import numpy as np
    import pandas as pd

    adata = ad.AnnData(
        X=np.zeros((4, 2), dtype=np.float32),
        obs=pd.DataFrame({"g": ["a", "a", "b", "b"]}, index=list("0123")),
    )
    with pytest.raises(ValueError, match="cannot modify a view in place"):
        _double(adata[adata.obs["g"] == "a"])
    assert _double(adata[adata.obs["g"] == "a"], copy=True) is not None


def test_every_control_mask_goes_through_reference_mask(cells):
    """An h5ad round trip can bring Metadata_Control back as a category of "True"/"False".
    Coercing that with .to_numpy(dtype=bool) reads every row as a control and gives
    plausible but wrong numbers."""
    import mantispy as mt

    round_tripped = cells.copy()
    round_tripped.obs["Metadata_Control"] = pd.Categorical(cells.obs["Metadata_Control"].astype(str))

    expected = cells.obs["Metadata_Control"].to_numpy(dtype=bool)
    np.testing.assert_array_equal(mt.get.controls(round_tripped), expected)

    mt.pp.well_qc(round_tripped, min_cells=0)
    assert round_tripped.uns["mantispy"]["well_qc"]["control_cv"].notna().any()


def test_two_layer_writing_calls_keep_separate_provenance():
    """key_added selects a different output matrix, so a single provenance key would leave
    the first call's parameters overwritten by the second's."""
    import mantispy as mt

    wells = mt.tl.aggregate(mt.ds.synthetic_plate(n_wells=8, n_cells=6, n_features=8, seed=0), min_cells=0)
    mt.pp.normalize(wells, by="Metadata_Plate", key_added="one")
    mt.pp.normalize(wells, by="Metadata_Plate", method="standardize", key_added="two")

    recorded = wells.uns["mantispy"]["params"]
    assert recorded["normalize:one"]["method"] == "mad_robustize"
    assert recorded["normalize:two"]["method"] == "standardize"

    # A key_added that names a column, not a layer, keeps the plain name.
    mt.pp.outliers(wells, method="mad")
    assert "outliers" in recorded
