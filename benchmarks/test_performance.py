"""Performance and memory, on synthetic data at screen scale.

Not part of the test run: ``testpaths`` is ``tests``, so nothing here is collected unless
you ask for it by path.

    ./.venv/bin/pytest benchmarks -s

Every case reports wall time, peak allocation during the call, and the size of ``X`` it
worked on, so the overhead is reported rather than asserted. The few assertions are loose,
since timings on a laptop under load are noisy, and catch only order-of-magnitude
regressions.

The release plan asks for 1M cells by 4000 features, which is 16 GB of float32 before any
transform allocates its output. These run at 500 features, which keeps the shapes
representative and fits in memory.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import mantispy as mt
from mantispy._core.features import parse_feature_names
from mantispy._core.schema import stamp

SIZES = [(100_000, 500), (500_000, 500)]


def _plate(n_obs: int, n_vars: int) -> mt.AnnData:  # type: ignore[name-defined]
    """A screen-shaped object built directly, since synthetic_plate is not the thing timed."""
    import anndata as ad

    rng = np.random.default_rng(0)
    values = rng.normal(size=(n_obs, n_vars)).astype(np.float32)
    n_plates, n_wells = 20, 384

    # A well holds one perturbation, as a real plate does: aggregation has to carry the
    # perturbation through, and it only can when the column is constant within the group.
    plate = np.repeat(np.arange(n_plates), n_obs // n_plates + 1)[:n_obs]
    well = np.tile(np.arange(n_wells), n_obs // n_wells + 1)[:n_obs]
    rows, columns = well // 24, well % 24
    obs = pd.DataFrame(
        {
            "Metadata_Plate": [f"P{index:02d}" for index in plate],
            "Metadata_Well": [f"{chr(65 + row)}{column + 1:02d}" for row, column in zip(rows, columns, strict=True)],
            "Metadata_Perturbation": [f"pert{index:03d}" for index in well],
        },
        index=[str(index) for index in range(n_obs)],
    )
    obs["Metadata_Control"] = obs["Metadata_Perturbation"] == "pert000"
    names = [f"Cells_AreaShape_Feature{index}" for index in range(n_vars)]
    adata = ad.AnnData(X=values, obs=obs, var=parse_feature_names(names))
    stamp(adata, resolution="cell")
    return adata


def _measure(label: str, shape: tuple[int, int], call) -> float:
    """Time and peak memory, measured in separate passes.

    tracemalloc traces every allocation, which on code that makes many small ones costs
    more than the code itself (tl.aggregate takes 10.5 s traced and 0.4 s untraced). The
    first pass times the call without tracing; the second only measures memory.
    """
    start = time.perf_counter()
    call()
    elapsed = time.perf_counter() - start

    tracemalloc.start()
    call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    size = shape[0] * shape[1] * 4 / 1e6
    print(
        f"  {label:22s} {shape[0]:>7d} x {shape[1]:<4d}  {elapsed:6.2f}s  peak {peak / 1e6:7.0f} MB  (X is {size:.0f} MB)"
    )
    return elapsed


@pytest.mark.parametrize("shape", SIZES, ids=[f"{n // 1000}k" for n, _ in SIZES])
def test_preprocessing(shape):
    """Read-path and reduction costs at screen scale."""
    adata = _plate(*shape)
    print(f"\npreprocessing {shape}")

    baseline = _measure("numpy median per plate", shape, lambda: np.median(np.asarray(adata.X), axis=0))
    normalize = _measure("pp.normalize", shape, lambda: mt.pp.normalize(adata, by="Metadata_Plate"))
    _measure("pp.feature_select", shape, lambda: mt.pp.feature_select(adata))
    _measure("pp.downsample", shape, lambda: mt.pp.downsample(adata, n_per_group=50))
    _measure("tl.aggregate", shape, lambda: mt.tl.aggregate(adata, min_cells=0))

    # A grouped, NaN-aware, 20-plate normalization against one global median: the same
    # order of work, so anything past 50x is a regression rather than a cost.
    assert normalize < 50 * max(baseline, 1e-3)


@pytest.mark.parametrize("shape", SIZES, ids=[f"{n // 1000}k" for n, _ in SIZES])
def test_profile_level_tools(shape):
    """What the profile-level tools cost once the cells are aggregated."""
    adata = _plate(*shape)
    mt.pp.normalize(adata, by="Metadata_Plate")
    wells = mt.tl.aggregate(adata, min_cells=0)
    print(f"\nprofile tools, from {shape} to {wells.shape}")

    sc.pp.pca(wells, n_comps=30)
    _measure("pp.sphere", wells.shape, lambda: mt.pp.sphere(wells, reference=None, key_added="sphered"))
    _measure("tl.hit_calling", wells.shape, lambda: mt.tl.hit_calling(wells, use_rep="X_pca", n_permutations=1000))
    _measure("tl.edistance", wells.shape, lambda: mt.tl.edistance(wells, use_rep="X_pca", n_permutations=1000))
    _measure("tl.effect_size", wells.shape, lambda: mt.tl.effect_size(wells))

    map_available = pytest.importorskip("copairs", reason="mAP needs mantispy[map]") is not None
    if map_available:
        treated = wells[~wells.obs["Metadata_Control"].to_numpy()].copy()
        _measure("tl.map", treated.shape, lambda: mt.tl.map(treated, mode="activity", null_size=1000))


def test_backed_uses_less_memory_than_it_saves_time(tmp_path):
    """Backed mode trades speed for memory, and both sides are measured."""
    adata = _plate(100_000, 500)
    mt.io.write(adata, tmp_path / "big.h5ad")
    print("\nbacked against in memory, per-plate median")

    from mantispy._core._reduce import MEDIAN, reduce_grouped

    backed = mt.io.read(tmp_path / "big.h5ad", backed="r")
    on_disk = _measure("reduce_grouped backed", adata.shape, lambda: reduce_grouped(backed, "Metadata_Plate", MEDIAN))
    in_memory = _measure(
        "reduce_grouped in memory", adata.shape, lambda: reduce_grouped(adata, "Metadata_Plate", MEDIAN)
    )
    assert on_disk > in_memory  # slower is the expected cost of reading from disk
