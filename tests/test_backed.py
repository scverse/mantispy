"""Backed objects: the same numbers as in memory, without materialising X.

A grouped operation on a backed object reads one group's rows at a time and gives the same numbers as the in-memory path.
Functions that write X in place refuse backed objects with an error.
"""

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import mantispy as mt
from mantispy._core._reduce import (
    MEAN,
    MEDIAN,
    _reads_from_disk,
    get_matrix,
    reduce_grouped,
    transform_grouped,
)


@pytest.fixture
def paths(tmp_path, cells):
    mt.io.write(cells, tmp_path / "cells.h5ad")
    return tmp_path / "cells.h5ad"


@pytest.fixture
def backed(paths):
    return mt.io.read(paths, backed="r")


@pytest.fixture
def reads(monkeypatch):
    """How every read through the matrix seam asked for its rows, in call order.

    An entry is the row index that was handed over, or ``None`` for a read of the whole matrix.
    Which rows were asked for matters as much as how many: h5py takes a fancy index only in increasing order, so get_matrix pays a full-size copy to restore any other one (#114).
    """
    from mantispy._core import _reduce

    original = _reduce.get_matrix
    asked: list[np.ndarray | None] = []

    def recording(adata, layer=None, rows=None):
        asked.append(None if rows is None else np.asarray(rows))
        return original(adata, layer, rows)

    monkeypatch.setattr(_reduce, "get_matrix", recording)
    return asked


grouped_paths = pytest.mark.parametrize(
    "call",
    [
        lambda adata: reduce_grouped(adata, "Metadata_Plate", MEDIAN),
        lambda adata: transform_grouped(adata, "Metadata_Plate", lambda _, block: block),
    ],
    ids=["reduce_grouped", "transform_grouped"],
)


@grouped_paths
def test_per_group_reads_never_ask_for_the_whole_matrix(backed, reads, call):
    """Both grouped paths read one group at a time.

    transform_grouped used to size its output with ``np.empty_like(get_matrix(adata, layer))``, a full read on top of the per-group ones (#67).
    """
    call(backed)

    assert reads, "nothing was read through the seam"
    sizes = [backed.n_obs if rows is None else rows.size for rows in reads]
    assert max(sizes) < backed.n_obs, f"a read of {max(sizes)} rows is the whole matrix"


@grouped_paths
def test_a_groups_rows_are_asked_for_in_increasing_order(backed, reads, call):
    """Which is why get_matrix can hand the index to h5py as it stands (#114).

    Both paths take their rows from one stable ordering, so a group's rows come out ascending.
    Should that ever stop being true, the sort in get_matrix is what keeps the read working, and this test is what says the copy behind it is no longer dead weight.
    """
    call(backed)

    for rows in reads:
        assert rows is not None, "a whole-matrix read has no group to be in order"
        assert (np.diff(rows) > 0).all(), f"rows {rows} are not increasing"


class _Dataset:
    """An h5py dataset as get_matrix sees one: a shape and a dtype, no ``toarray``, and h5py's own rules.

    Those rules are spelled out here element by element rather than with the expression get_matrix branches on, so that the double and the code it stands in for cannot be wrong in the same way.
    It records the index it was asked for and the block it gave back, so a test can say both how the rows were requested and whether what get_matrix returned is that block or a copy of it.
    """

    def __init__(self, values):
        self._values = values
        self.shape = values.shape
        self.dtype = values.dtype
        self.asked: list = []
        self.given: list = []

    def __getitem__(self, index):
        if not isinstance(index, slice):
            wanted = np.asarray(index).tolist()  # h5py takes a list as readily as an array
            if not all(later > earlier for earlier, later in zip(wanted, wanted[1:], strict=False)):
                raise TypeError("Indexing elements must be in increasing order")  # h5py's message
            if wanted and not 0 <= min(wanted) <= max(wanted) < self.shape[0]:
                raise OSError("Can't synchronously read data (selection + offset not within extent)")
        self.asked.append(index)
        # h5py reads into a buffer of its own, where a numpy slice would hand back a view.
        self.given.append(np.array(self._values[index]))
        return self.given[-1]


def _on_disk(values):
    """The least an object needs for get_matrix to treat its matrix as one on disk."""
    dataset = _Dataset(values)
    assert _reads_from_disk(dataset), "the double must take the on-disk branch, or these tests watch the wrong one"
    return SimpleNamespace(X=dataset, layers={}), dataset


@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32 file", "float64 file"])
def test_rows_already_in_order_are_read_without_a_second_copy(dtype):
    """Regression test for #114: the index was sorted for h5py and the block gathered back into the order asked for, but every caller asks in order, so the gather was a full-size copy of what had just been read.

    At JUMP well scale that copy is a 720 MB allocation made to be thrown away.
    """
    values = np.arange(40, dtype=dtype).reshape(10, 4)
    adata, dataset = _on_disk(values)

    block = get_matrix(adata, rows=np.array([1, 4, 7]))

    np.testing.assert_array_equal(block, values[[1, 4, 7]])
    assert len(dataset.asked) == 1, "the rows were read more than once"
    np.testing.assert_array_equal(dataset.asked[0], [1, 4, 7])
    if dtype is np.float32:
        # Only a float32 file makes the closing float32 cast in get_matrix the identity.
        assert block is dataset.given[0], "the block the dataset returned was copied again"


def test_rows_out_of_order_still_come_back_in_the_order_asked_for():
    """The sort is what lets h5py read them at all, so it stays for a caller that does not ask in order."""
    values = np.arange(40, dtype=np.float32).reshape(10, 4)
    adata, dataset = _on_disk(values)

    block = get_matrix(adata, rows=np.array([7, 1, 4]))

    np.testing.assert_array_equal(block, values[[7, 1, 4]])
    np.testing.assert_array_equal(dataset.asked[0], [1, 4, 7], err_msg="h5py is read in increasing order")


@pytest.fixture
def backed_sparse(tmp_path):
    """The same values as a sparse file on disk and as an in-memory object, in either storage order."""

    def build(fmt):
        obs = pd.DataFrame({"g": np.repeat(["a", "b", "c"], 4)}, index=[str(index) for index in range(12)])
        dense = np.zeros((12, 3), dtype=np.float32)
        dense[np.arange(12), np.arange(12) % 3] = np.arange(1, 13)
        maker = sparse.csr_matrix if fmt == "csr" else sparse.csc_matrix
        path = tmp_path / f"{fmt}.h5ad"
        ad.AnnData(X=maker(dense), obs=obs).write_h5ad(path)
        return ad.read_h5ad(path, backed="r"), ad.AnnData(X=dense.copy(), obs=obs.copy())

    return build


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_a_sparse_matrix_on_disk_can_be_read_whole(backed_sparse, fmt):
    """anndata's sparse datasets have no ``__array__``, so the float32 read that ends get_matrix saw a sequence of sparse rows and raised `setting an array element with a sequence`.

    Every caller that reads the matrix without naming rows took that path, `pp.calculate_qc_metrics` among them.
    """
    from_disk, in_memory = backed_sparse(fmt)
    expected = np.asarray(in_memory.X)

    np.testing.assert_array_equal(get_matrix(from_disk), expected)
    np.testing.assert_array_equal(get_matrix(from_disk, rows=np.arange(12)), expected)
    np.testing.assert_array_equal(get_matrix(from_disk, rows=np.array([1, 5, 9])), expected[[1, 5, 9]])


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_a_sparse_matrix_on_disk_reduces_to_what_it_does_in_memory(backed_sparse, fmt):
    from_disk, in_memory = backed_sparse(fmt)

    streamed, keys, counts = reduce_grouped(from_disk, "g", MEAN)
    single, keys_memory, counts_memory = reduce_grouped(in_memory, "g", MEAN)

    np.testing.assert_allclose(streamed, single, rtol=1e-10)
    np.testing.assert_array_equal(counts, counts_memory)
    assert list(keys) == list(keys_memory)


@pytest.mark.parametrize("container", ["backed", "dense"])
def test_a_mask_that_is_not_one_per_row_is_refused_on_both_paths(container, tmp_path):
    """Each group's rows are selected out of the mask, which reads only the entries that group owns, so a mask longer than the object went unnoticed on disk and raised in memory.

    A mask computed against a pre-filter superset is the way that happens.
    """
    obs = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 4 + ["c"] * 4}, index=[str(index) for index in range(12)])
    values = np.arange(24, dtype=np.float32).reshape(12, 2)

    if container == "backed":
        path = tmp_path / "masked.h5ad"
        ad.AnnData(X=values.copy(), obs=obs).write_h5ad(path)
        adata = ad.read_h5ad(path, backed="r")
    else:
        adata = ad.AnnData(X=values.copy(), obs=obs.copy())

    for wrong in (np.ones(15, dtype=bool), np.ones(10, dtype=bool)):
        with pytest.raises(ValueError, match="one boolean per row"):
            reduce_grouped(adata, "g", MEAN, mask=wrong)


@pytest.fixture
def many_groups(tmp_path):
    """The same rows backed and in memory, grouped far more finely than the other fixtures here.

    Every other backed fixture has two plates.
    The scan `reduce_grouped` used to run per group was two full-length passes over the codes each, so its cost is set by the group count and only shows above a few hundred (#113): a single plate's wells, and the grouping `tl.aggregate` takes on a cell-level object.
    """
    n_groups, per_group = 384, 3
    obs = pd.DataFrame(
        {"Metadata_Well": np.repeat([f"W{index:04d}" for index in range(n_groups)], per_group)},
        index=[str(index) for index in range(n_groups * per_group)],
    )
    values = np.random.default_rng(0).normal(size=(n_groups * per_group, 6)).astype(np.float32)
    path = tmp_path / "many.h5ad"
    ad.AnnData(X=values, obs=obs).write_h5ad(path)
    return ad.read_h5ad(path, backed="r"), ad.AnnData(X=values.copy(), obs=obs.copy())


@pytest.mark.parametrize("masked", [False, True], ids=["every row", "masked"])
def test_many_groups_reduce_to_what_the_single_kernel_call_gives(many_groups, masked):
    """Taking each group's rows from one ordering has to select exactly what the per-group scan did, including the rows a mask leaves out and a group it empties."""
    from_disk, in_memory = many_groups

    mask = None
    if masked:
        mask = np.ones(in_memory.n_obs, dtype=bool)
        mask[1::3] = False  # one row of every group
        mask[:3] = False  # and the whole of the first group, which then has no statistic

    streamed, keys, counts = reduce_grouped(from_disk, "Metadata_Well", MEAN, mask=mask)
    single, keys_memory, counts_memory = reduce_grouped(in_memory, "Metadata_Well", MEAN, mask=mask)

    np.testing.assert_allclose(streamed, single, rtol=1e-10)
    np.testing.assert_array_equal(counts, counts_memory)
    assert list(keys) == list(keys_memory)
    assert counts.tolist() == ([3] * 384 if not masked else [0] + [2] * 383)


def test_writing_x_in_place_is_refused_with_the_way_out(backed):
    with pytest.raises(ValueError, match="copy=True"):
        mt.pp.normalize(backed, by="Metadata_Plate")


def test_key_added_normalizes_a_backed_object_without_rewriting_x(backed):
    """io.read documents key_added= as one of the two ways out for a backed object, and the guard refused it for a call that writes a layer and never touches X."""
    mt.pp.normalize(backed, by="Metadata_Plate", reference="negcon", key_added="normalized")

    assert type(backed.X).__name__ == "Dataset", "X must still be the one on disk"
    np.testing.assert_allclose(
        np.asarray(backed.layers["normalized"]),
        np.asarray(mt.pp.normalize(backed, by="Metadata_Plate", reference="negcon", copy=True).X),
        rtol=1e-6,
    )


@pytest.mark.parametrize("name", ["filter_cells", "filter_features", "filter_images"])
def test_filtering_a_backed_object_names_the_way_out(backed, name):
    """These take no key_added, so the guard skipped them and anndata refused the subset with a message that names .to_memory() but not copy=True."""
    mt.pp.calculate_qc_metrics(backed)
    backed.obs["qc_image_pass"] = True

    with pytest.raises(ValueError, match="copy=True"):
        getattr(mt.pp, name)(backed)
