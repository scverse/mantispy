"""Build the staged dataset variants that are rehosted on scverse-exampledata.

This is the single source of truth for the variants that the staged loaders fetch. Each entry in
:data:`STAGED` pairs a *builder* (rebuilds the variants from the raw pipeline) with a *shipped loader*
(returns the same filenames as loaded through the public ``mt.ds`` API). ``scripts/build_staged_datasets.py``
writes what the builders produce; ``scripts/check_staged_drift.py`` compares each shipped variant against a
rebuild, so the pipeline here and the uploaded bytes must not diverge. After a change to a builder, rebuild
with ``--print-sha256``, re-upload to ``s3://scverse-exampledata/mantispy/<name>/`` and update the sha256 in
``registry.yaml``.

Imports of the public pipeline are deferred into the functions: this module lives inside the package, so a
top-level ``import mantispy`` would cycle through :mod:`mantispy.ds`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from anndata import AnnData


def _zero_nonfinite(adata: AnnData) -> AnnData:
    """Replace NaN and +-inf in ``X`` with zero, in place, and return the object."""
    adata.X = np.nan_to_num(np.asarray(adata.X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return adata


def _stable(adata: AnnData) -> AnnData:
    """Rows sorted by ``obs_names`` and columns by ``var_names``, so a rebuild is order-stable."""
    rows = np.argsort(adata.obs_names.to_numpy(), kind="stable")
    cols = np.argsort(adata.var_names.to_numpy(), kind="stable")
    return adata[rows][:, cols].copy()


def _gene_consensus(block: AnnData, screened: np.ndarray) -> AnnData:
    """One modz consensus per gene over the screened wells of ``block``, NaN-zeroed."""
    from mantispy.tl._consensus import consensus

    gene = consensus(block[screened], by="Metadata_Gene", method="modz", correlation="spearman", min_replicates=2)
    return _zero_nonfinite(gene)


def build_rohban_variants(cache_dir: str | Path | None = None) -> dict[str, AnnData]:
    """Returns {'rohban_selected.h5ad': ad, 'rohban_gene.h5ad': ad, 'rohban_gene_selected.h5ad': ad}."""
    from mantispy.ds._datasets import rohban
    from mantispy.pp._normalize import normalize
    from mantispy.pp._select import feature_select, subset_features

    adata = rohban(cache_dir=cache_dir)
    adata.obs["is_untreated"] = (adata.obs["Metadata_Perturbation_Type"].astype(str) == "untreated").to_numpy()
    normalize(adata, method="mad_robustize", by="Metadata_Plate", reference="is_untreated")
    _zero_nonfinite(adata)

    # Well-level feature-selected block, pycytominer's default operations.
    feature_select(adata)
    selected = _zero_nonfinite(subset_features(adata))

    # Gene-level consensus over the screened wells: untreated and transfection controls dropped.
    screened = (~adata.obs["is_untreated"].to_numpy()) & (~adata.obs["Metadata_Control"].to_numpy())
    return {
        "rohban_selected.h5ad": _stable(selected),
        "rohban_gene.h5ad": _stable(_gene_consensus(adata, screened)),
        "rohban_gene_selected.h5ad": _stable(_gene_consensus(selected, screened)),
    }


def _shipped_rohban() -> dict[str, AnnData]:
    """The three rohban variants as fetched through the public ``mt.ds.rohban`` API."""
    from mantispy.ds._datasets import rohban

    return {
        "rohban_selected.h5ad": rohban(feature_selected=True),
        "rohban_gene.h5ad": rohban(aggregated=True),
        "rohban_gene_selected.h5ad": rohban(aggregated=True, feature_selected=True),
    }


# One entry per staged dataset: (builder rebuilding the variants from the raw pipeline, loader returning the
# shipped variants through the public ``mt.ds`` API, keyed by the same filenames). Later PRs stage a dataset
# by adding one entry here; the build and drift scripts loop this registry and need no per-dataset edit.
STAGED: dict[str, tuple[Callable[..., dict[str, AnnData]], Callable[[], dict[str, AnnData]]]] = {
    "rohban": (build_rohban_variants, _shipped_rohban),
}
