"""Build a dataset from its pinned upstream files and write the h5ad that is rehosted.

    python scripts/build_dataset.py bbbc021 --out build/

Prints the file's size and sha256, for the registry to pin, and a fingerprint of what it holds.
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
from mantispy.ds._datasets import _DATASETS


def fingerprint(adata: AnnData) -> str:
    """sha256 over ``X``, ``obs``, ``var`` and what ``uns`` says the object is, however a file encodes them."""
    store = adata.uns["mantispy"]
    # h5ad reads a list back as an array, so both are hashed as lists.
    identity = [np.asarray(store.get(key)).tolist() for key in ("resolution", "channels", "dataset")]
    digest = hashlib.sha256(repr(identity).encode())
    digest.update(np.ascontiguousarray(adata.X, dtype=np.float32))
    for frame in (adata.obs, adata.var):
        digest.update("\t".join(frame.columns).encode())
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy())
    return digest.hexdigest()


def main() -> None:
    """Build the dataset named on the command line."""
    parser = argparse.ArgumentParser(description="Build a dataset and write the h5ad that is rehosted.")
    # The datasets the registry records a shape for are the ones that are an AnnData.
    parser.add_argument("name", choices=sorted(name for name, entry in _DATASETS.items() if "shape" in entry.metadata))
    parser.add_argument("--out", type=Path, default=Path("build"), help="directory to write <name>.h5ad into")
    args = parser.parse_args()

    # jump_cells keeps its assembly in the cache, which would be written out again instead of rebuilt.
    for cached in Path(mt.settings.cache_dir).glob(f"{args.name}-*.h5ad"):
        cached.unlink()
    adata = getattr(mt.ds, args.name)(**({"plates": None} if args.name == "jump_target2" else {}))
    # The parameters it was read with name paths on this machine.
    adata.uns["mantispy"].pop("params", None)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.name}.h5ad"
    mt.io.write(adata, path)
    with path.open("rb") as file:
        sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    print(f"{path}  {adata.n_obs} x {adata.n_vars}  {path.stat().st_size / 1e6:.1f} MB")
    print(f"sha256       {sha256}")
    print(f"fingerprint  {fingerprint(adata)}")


if __name__ == "__main__":
    main()
