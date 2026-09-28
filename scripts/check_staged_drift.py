"""Check the shipped staged rohban variants against a rebuild from the current pipeline.

    python scripts/check_staged_drift.py

Exit 1 on drift. The registry sha256 only guards the downloaded bytes; this guards that the code in
``scripts/build_staged_datasets.py`` still reproduces what was uploaded, so a pipeline change that no longer
matches the shipped artifacts is caught. It is wired into CI as a non-blocking job.
"""

from __future__ import annotations

import sys

import numpy as np
from anndata import AnnData
from build_staged_datasets import build_rohban_variants

import mantispy as mt

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


shipped = {
    "rohban_selected.h5ad": mt.ds.rohban(feature_selected=True),
    "rohban_gene.h5ad": mt.ds.rohban(aggregated=True),
    "rohban_gene_selected.h5ad": mt.ds.rohban(aggregated=True, feature_selected=True),
}
rebuilt = build_rohban_variants()

ok = True
for name in shipped:
    reason = _drift(shipped[name], rebuilt[name])
    if reason is None:
        print(f"{name}: matches the current pipeline")
        continue
    ok = False
    print(
        f"{name} has drifted from the current pipeline ({reason}); regenerate with "
        "scripts/build_staged_datasets.py, re-upload to s3://scverse-exampledata/mantispy/rohban/, "
        "and update the sha256 in registry.yaml."
    )

sys.exit(0 if ok else 1)
