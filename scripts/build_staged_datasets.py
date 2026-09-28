"""Build the staged rohban variants that are rehosted on scverse-exampledata.

    python scripts/build_staged_datasets.py --print-sha256 --out build/

This is the single source of truth for the three variants ``mt.ds.rohban(aggregated=, feature_selected=)``
fetches: ``scripts/check_staged_drift.py`` rebuilds with :func:`build_rohban_variants` and compares against
the shipped artifacts, so the pipeline here and the uploaded bytes must not diverge. After a change here,
rebuild with ``--print-sha256``, re-upload to ``s3://scverse-exampledata/mantispy/rohban/`` and update the
sha256 in ``registry.yaml``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import mantispy as mt

if TYPE_CHECKING:
    from anndata import AnnData


def _zero_nonfinite(adata: AnnData) -> AnnData:
    """Replace NaN and +-inf in ``X`` with zero, in place, and return the object."""
    adata.X = np.nan_to_num(np.asarray(adata.X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return adata


def _stable(adata: AnnData) -> AnnData:
    """Rows sorted by ``obs_names`` and columns by ``var_names``, so a rebuild is order-stable."""
    rows = np.argsort(adata.obs_names.to_numpy(), kind="stable")
    cols = np.argsort(adata.var_names.to_numpy(), kind="stable")
    return adata[rows][:, cols].copy()


def _gene_consensus(block: AnnData, screened: np.ndarray) -> AnnData:
    """One modz consensus per gene over the screened wells of ``block``, NaN-zeroed."""
    gene = mt.tl.consensus(block[screened], by="Metadata_Gene", method="modz", correlation="spearman", min_replicates=2)
    return _zero_nonfinite(gene)


def build_rohban_variants(cache_dir: str | Path | None = None) -> dict[str, AnnData]:
    """Returns {'rohban_selected.h5ad': ad, 'rohban_gene.h5ad': ad, 'rohban_gene_selected.h5ad': ad}."""
    adata = mt.ds.rohban(cache_dir=cache_dir)
    adata.obs["is_untreated"] = (adata.obs["Metadata_Perturbation_Type"].astype(str) == "untreated").to_numpy()
    mt.pp.normalize(adata, method="mad_robustize", by="Metadata_Plate", reference="is_untreated")
    _zero_nonfinite(adata)

    # Well-level feature-selected block, pycytominer's default operations.
    mt.pp.feature_select(adata)
    selected = _zero_nonfinite(mt.pp.subset_features(adata))

    # Gene-level consensus over the screened wells: untreated and transfection controls dropped.
    screened = (~adata.obs["is_untreated"].to_numpy()) & (~adata.obs["Metadata_Control"].to_numpy())
    return {
        "rohban_selected.h5ad": _stable(selected),
        "rohban_gene.h5ad": _stable(_gene_consensus(adata, screened)),
        "rohban_gene_selected.h5ad": _stable(_gene_consensus(selected, screened)),
    }


def main() -> None:
    """Build the variants and, with ``--print-sha256``, write each h5ad and print its sha256."""
    parser = argparse.ArgumentParser(description="Build the staged rohban variants.")
    parser.add_argument("--out", type=Path, default=Path("build"), help="directory to write the h5ads into")
    parser.add_argument(
        "--print-sha256", action="store_true", help="write each h5ad and print its sha256 for the registry"
    )
    args = parser.parse_args()

    variants = build_rohban_variants()
    if args.print_sha256:
        args.out.mkdir(parents=True, exist_ok=True)
    for name, adata in variants.items():
        line = f"{name}  {adata.n_obs} x {adata.n_vars}"
        if args.print_sha256:
            path = args.out / name
            mt.io.write(adata, path)
            with path.open("rb") as file:
                line += f"  {hashlib.file_digest(file, 'sha256').hexdigest()}"
        print(line)


if __name__ == "__main__":
    main()
