"""Correctness of the column-blocked kernels: block iteration and float32 correlation.

Chunking is a re-association of the same math, so the result must equal the whole-matrix reference.
float32 correlation only diverges near the threshold, which these tests keep clear of.
"""

import numpy as np
import pandas as pd
import pytest

from mantispy._core._corr import correlated_pairs


def test_windowed_rejects_bad_window_and_stride():
    rng = np.random.default_rng(13)
    X = rng.standard_normal((50, 6))
    for bad_window in (0, -1):
        with pytest.raises(ValueError):
            correlated_pairs(X, 0.9, window=bad_window)
    for bad_stride in (0, -2):
        with pytest.raises(ValueError):
            correlated_pairs(X, 0.9, window=3, stride=bad_stride)


def test_feature_select_corr_window_drops_one_of_a_pair():
    from anndata import AnnData

    import mantispy as mt

    rng = np.random.default_rng(7)
    n_obs = 60
    X = rng.standard_normal((n_obs, 6)).astype(np.float32)
    X[:, 1] = X[:, 0] + rng.standard_normal(n_obs).astype(np.float32) * 0.01
    names = [f"Cells_Intensity_{i}" for i in range(6)]
    adata = AnnData(X=X, var=pd.DataFrame(index=names))

    mt.pp.feature_select(adata, operations=("correlation_threshold",), corr_window=3, corr_stride=1)
    selected = adata.var["selected"]
    assert selected.dtype == bool
    assert selected.shape == (6,)
    assert int(selected.sum()) == 5
    assert not (bool(selected.iloc[0]) and bool(selected.iloc[1]))


def test_the_extreme_value_kernel_caches_its_compilation(tmp_path):
    """A second process loads column_max_robust_z's machine code from the cache instead of compiling it again."""
    import os
    import subprocess
    import sys

    script = (
        "import warnings, numpy as np\n"
        "warnings.simplefilter('error')\n"
        "from mantispy._core._numba import _column_max_robust_z, column_max_robust_z\n"
        "column_max_robust_z(np.ones((4, 3)), np.arange(3), 1.4826)\n"
        "print(sum(_column_max_robust_z.stats.cache_hits.values()))\n"
    )
    env = {**os.environ, "NUMBA_CACHE_DIR": str(tmp_path)}
    runs = [subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True) for _ in range(2)]
    assert all(run.returncode == 0 for run in runs), [run.stderr for run in runs]
    assert [run.stdout.strip() for run in runs] == ["0", "1"]
