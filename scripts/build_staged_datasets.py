"""Build the staged dataset variants that are rehosted on scverse-exampledata.

    python scripts/build_staged_datasets.py --print-sha256 --out build/
    python scripts/build_staged_datasets.py --only rohban

Thin CLI over the :data:`mantispy.ds._build.STAGED` registry, which is the single source of truth for the
variants the staged loaders fetch: ``scripts/check_staged_drift.py`` rebuilds through the same registry and
compares against the shipped artifacts, so the pipeline there and the uploaded bytes must not diverge. After a
change to a builder, rebuild with ``--print-sha256``, re-upload to
``s3://scverse-exampledata/mantispy/<name>/`` and update the sha256 in ``registry.yaml``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import mantispy as mt
from mantispy.ds._build import STAGED


def main() -> None:
    """Build the registered variants and, with ``--print-sha256``, write each h5ad and print its sha256."""
    parser = argparse.ArgumentParser(description="Build the staged dataset variants.")
    parser.add_argument("--out", type=Path, default=Path("build"), help="directory to write the h5ads into")
    parser.add_argument(
        "--print-sha256", action="store_true", help="write each h5ad and print its sha256 for the registry"
    )
    parser.add_argument("--only", metavar="NAME", help="build only this dataset (default: every registered dataset)")
    args = parser.parse_args()

    if args.only is not None and args.only not in STAGED:
        parser.error(f"unknown dataset {args.only!r}; registered: {', '.join(STAGED)}")

    if args.print_sha256:
        args.out.mkdir(parents=True, exist_ok=True)
    for name, (builder, _shipped) in STAGED.items():
        if args.only is not None and name != args.only:
            continue
        for filename, adata in builder().items():
            line = f"{filename}  {adata.n_obs} x {adata.n_vars}"
            if args.print_sha256:
                path = args.out / filename
                mt.io.write(adata, path)
                with path.open("rb") as file:
                    line += f"  {hashlib.file_digest(file, 'sha256').hexdigest()}"
            print(line)


if __name__ == "__main__":
    main()
