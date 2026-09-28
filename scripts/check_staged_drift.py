"""Check the shipped staged variants against a rebuild from the current pipeline.

    python scripts/check_staged_drift.py                  # light datasets only (the per-PR job)
    python scripts/check_staged_drift.py --include-heavy   # also the heavy datasets (the cron job)

Exit 1 on drift. The registry sha256 only guards the downloaded bytes; this guards that the builders in
:mod:`mantispy.ds._build` still reproduce what was uploaded, so a pipeline change that no longer matches the
shipped artifacts is caught. It loops the :data:`mantispy.ds._build.STAGED` registry, so a newly staged dataset
is covered without editing this script. Heavy datasets (large raw inputs, too slow to rebuild on every pull
request) are skipped unless ``--include-heavy`` is passed; the per-PR job runs without it and the scheduled
job runs with it. It is wired into CI as a blocking job.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from anndata import AnnData

from mantispy.ds._build import STAGED


def _drift(shipped: AnnData, rebuilt: AnnData) -> str | None:
    """The first way ``rebuilt`` differs from ``shipped``, or ``None`` when they match."""
    if list(shipped.obs_names) != list(rebuilt.obs_names):
        return "obs drift"
    if list(shipped.var_names) != list(rebuilt.var_names):
        return "var drift"
    # Compare content, not the h5ad bytes: h5ad is not byte-reproducible across h5py/numpy versions, so a
    # raw-sha256 compare would false-positive. atol=1e-4 is a tunable balance between catching a real
    # pipeline change and tolerating float re-rounding on a rebuild. equal_nan matches a raw variant's NaN
    # cells to the same cells in the rebuild (the un-normalized base keeps them); a NaN against a finite value
    # is still caught.
    if not np.allclose(np.asarray(shipped.X), np.asarray(rebuilt.X), atol=1e-4, rtol=1e-3, equal_nan=True):
        return "value drift"
    return None


def main() -> int:
    """Rebuild the registered variants and compare against the shipped artifacts, returning the exit code."""
    parser = argparse.ArgumentParser(description="Check the shipped staged variants against a rebuild.")
    parser.add_argument(
        "--include-heavy",
        action="store_true",
        help="also rebuild the heavy datasets (large raw inputs); the per-PR job omits this, the cron job passes it",
    )
    args = parser.parse_args()

    ok = True
    for name, (builder, shipped_loader, heavy) in STAGED.items():
        if heavy and not args.include_heavy:
            print(f"{name}: skipped (heavy; run with --include-heavy)")
            continue
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
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
else:  # pragma: no cover
    raise ImportError("check_staged_drift is a script, run it with `python scripts/check_staged_drift.py`")
