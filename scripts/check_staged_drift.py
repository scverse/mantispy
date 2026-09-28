"""Check the shipped staged variants against a rebuild from the current pipeline.

    python scripts/check_staged_drift.py

Exit 1 on drift. The registry sha256 only guards the downloaded bytes; this guards that the builders in
:mod:`mantispy.ds._build` still reproduce what was uploaded, so a pipeline change that no longer matches the
shipped artifacts is caught. It loops the :data:`mantispy.ds._build.STAGED` registry, so a newly staged dataset
is covered without editing this script. It is wired into CI as a blocking job.
"""

from __future__ import annotations

import sys

import numpy as np
from anndata import AnnData

from mantispy.ds._build import STAGED

if __name__ != "__main__":  # pragma: no cover
    raise ImportError("check_staged_drift is a script, run it with `python scripts/check_staged_drift.py`")


def _drift(shipped: AnnData, rebuilt: AnnData) -> str | None:
    """The first way ``rebuilt`` differs from ``shipped``, or ``None`` when they match."""
    if list(shipped.obs_names) != list(rebuilt.obs_names):
        return "obs drift"
    if list(shipped.var_names) != list(rebuilt.var_names):
        return "var drift"
    # Compare content, not the h5ad bytes: h5ad is not byte-reproducible across h5py/numpy versions, so a
    # raw-sha256 compare would false-positive. atol=1e-4 is a tunable balance between catching a real
    # pipeline change and tolerating float re-rounding on a rebuild.
    if not np.allclose(np.asarray(shipped.X), np.asarray(rebuilt.X), atol=1e-4, rtol=1e-3):
        return "value drift"
    return None


ok = True
for name, (builder, shipped_loader) in STAGED.items():
    shipped = shipped_loader()
    rebuilt = builder()
    for filename in shipped:
        reason = _drift(shipped[filename], rebuilt[filename])
        if reason is None:
            print(f"{filename}: matches the current pipeline")
            continue
        ok = False
        print(
            f"{filename} has drifted from the current pipeline ({reason}); regenerate with "
            f"scripts/build_staged_datasets.py, re-upload to s3://scverse-exampledata/mantispy/{name}/, "
            "and update the sha256 in registry.yaml."
        )

sys.exit(0 if ok else 1)
