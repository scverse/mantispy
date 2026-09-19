"""Build a dataset from its pinned upstream files and write the h5ad that is rehosted.

    python scripts/build_dataset.py bbbc021 --out build/

Prints the file's size and sha256, which the registry pins, and a fingerprint of what it holds.
anndata or h5py releases can write the same object as different bytes, so a rebuild is checked against the
fingerprint, not the sha256.
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData

import mantispy as mt

#: The call that loads everything a dataset pins, where its defaults load less.
EVERYTHING: dict[str, dict] = {"jump_target2": {"plates": None}}


def fingerprint(adata: AnnData) -> str:
    """sha256 over ``X`` and the ``obs`` and ``var`` tables, however a file encodes them."""
    digest = hashlib.sha256(np.ascontiguousarray(adata.X, dtype=np.float32).tobytes())
    for frame in (adata.obs, adata.var):
        digest.update("\t".join(frame.columns).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    """Build the dataset named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="a dataset of mantispy.ds, such as bbbc021")
    parser.add_argument("--out", type=Path, default=Path("build"), help="directory to write <name>.h5ad into")
    args = parser.parse_args()

    adata = getattr(mt.ds, args.name)(**EVERYTHING.get(args.name, {}))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.name}.h5ad"
    mt.io.write(adata, path)
    print(f"{path}  {adata.n_obs} x {adata.n_vars}  {path.stat().st_size / 1e6:.1f} MB")
    print(f"sha256       {hashlib.sha256(path.read_bytes()).hexdigest()}")
    print(f"fingerprint  {fingerprint(mt.io.read(path))}")


if __name__ == "__main__":
    main()
