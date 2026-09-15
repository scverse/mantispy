"""Backed objects: the same numbers as in memory, without materialising X.

A grouped operation on a backed object reads one group's rows at a time and gives the
same numbers as the in-memory path. Functions that write X in place refuse backed objects
with an error.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import mantispy as mt
from mantispy._core._reduce import MEAN, reduce_grouped


@pytest.fixture
def paths(tmp_path, cells):
    mt.io.write(cells, tmp_path / "cells.h5ad")
    return tmp_path / "cells.h5ad"


@pytest.fixture
def backed(paths):
    return mt.io.read(paths, backed="r")


def test_reading_backed_leaves_x_on_disk(backed, cells):
    assert backed.isbacked
    assert type(backed.X).__name__ == "Dataset"  # h5py, not a numpy array
    assert backed.shape == cells.shape
    assert mt.io.validate(backed).ok, mt.io.validate(backed).errors


def test_grouped_normalize_matches_the_in_memory_path(backed, cells):
    from_disk = mt.pp.normalize(backed, by="Metadata_Plate", reference="negcon", copy=True)
    in_memory = mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon", copy=True)
    np.testing.assert_allclose(np.asarray(from_disk.X), np.asarray(in_memory.X), rtol=1e-6)


def test_aggregate_and_feature_select_match(backed, cells):
    np.testing.assert_allclose(
        np.asarray(mt.tl.aggregate(backed, min_cells=0).X),
        np.asarray(mt.tl.aggregate(cells, min_cells=0).X),
        rtol=1e-6,
    )
    from_disk = mt.pp.feature_select(backed, copy=True)
    in_memory = mt.pp.feature_select(cells, copy=True)
    np.testing.assert_array_equal(from_disk.var["selected"], in_memory.var["selected"])


def test_per_group_reads_never_ask_for_the_whole_matrix(backed, monkeypatch):
    """A grouped reduction reads one group at a time and never the whole matrix."""
    from mantispy._core import _reduce

    original = _reduce.get_matrix
    asked = []

    def recording(adata, layer=None, rows=None):
        asked.append(adata.n_obs if rows is None else len(rows))
        return original(adata, layer, rows)

    monkeypatch.setattr(_reduce, "get_matrix", recording)
    _reduce.reduce_grouped(backed, "Metadata_Plate", _reduce.MEDIAN)

    assert asked, "nothing was read through the seam"
    assert max(asked) < backed.n_obs, f"a read of {max(asked)} rows is the whole matrix"


def test_streamed_and_single_pass_reductions_agree(backed, cells):
    """The backed path reduces group by group and the in-memory path in one kernel call;
    both must give the same numbers."""
    from mantispy._core._reduce import MAD, MEDIAN, STD, reduce_grouped

    for stat in (MEDIAN, MAD, STD):
        streamed, keys, counts = reduce_grouped(backed, "Metadata_Plate", stat)
        single, keys_memory, counts_memory = reduce_grouped(cells, "Metadata_Plate", stat)
        np.testing.assert_allclose(streamed, single, rtol=1e-10)
        np.testing.assert_array_equal(counts, counts_memory)
        assert list(keys) == list(keys_memory)


def test_writing_x_in_place_is_refused_with_the_way_out(backed):
    with pytest.raises(ValueError, match="copy=True"):
        mt.pp.normalize(backed, by="Metadata_Plate")


def test_a_backed_object_can_still_be_flagged_in_place(backed):
    """obs and var live in memory even when X does not, so QC works unchanged."""
    mt.pp.calculate_qc_metrics(backed)
    assert "qc_pass" in backed.obs


@pytest.mark.parametrize("container", ["dense", "sparse", "backed"])
def test_an_empty_group_has_no_statistic_whichever_path_reduces_it(container, tmp_path):
    """A group with no rows gets NaN on every path, since zero would read as a measurement.

    A fully masked reference group centred on 0.0 would let `pp.normalize` subtract nothing
    and report success.
    """
    obs = pd.DataFrame({"g": ["a", "a", "b", "b"]}, index=[str(i) for i in range(4)])
    values = np.arange(8, dtype=np.float32).reshape(4, 2)
    mask = np.array([True, True, False, False])  # group "b" contributes nothing

    if container == "sparse":
        adata = ad.AnnData(X=sparse.csr_matrix(values), obs=obs)
    elif container == "backed":
        path = tmp_path / "backed.h5ad"
        ad.AnnData(X=values.copy(), obs=obs).write_h5ad(path)
        adata = ad.read_h5ad(path, backed="r")
    else:
        adata = ad.AnnData(X=values.copy(), obs=obs)

    reduced, keys, counts = reduce_grouped(adata, "g", MEAN, mask=mask)
    assert list(keys) == ["a", "b"]
    assert counts.tolist() == [2, 0]
    np.testing.assert_allclose(reduced[0], [1.0, 2.0])
    assert np.isnan(reduced[1]).all(), "an empty group must be NaN, not zero"


def test_sparse_is_not_mistaken_for_an_on_disk_dataset():
    """It has .shape and .dtype like an h5py dataset, and is entirely in memory."""
    from mantispy._core._reduce import _reads_from_disk

    assert not _reads_from_disk(sparse.csr_matrix(np.eye(3)))
    assert not _reads_from_disk(np.eye(3))
