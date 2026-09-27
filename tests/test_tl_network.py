"""Enrichment of known interactions among the most-similar perturbation pairs."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantispy as mt


@pytest.fixture
def edges():
    """Four interacting pairs among the first eight genes."""
    return pd.DataFrame({"gene_a": ["G0", "G2", "G4", "G6"], "gene_b": ["G1", "G3", "G5", "G7"]})


def _screen(gene_order, edges, seed=0):
    """Twenty gene profiles where each edge pair shares an injected signal."""
    rng = np.random.default_rng(seed)
    genes = [f"G{i}" for i in range(20)]
    values = rng.normal(size=(20, 50))
    for a, b in edges.to_numpy():
        signal = rng.normal(size=50) * 3.0
        values[genes.index(a)] += signal
        values[genes.index(b)] += signal
    adata = ad.AnnData(
        X=values.astype(np.float32),
        obs=pd.DataFrame(
            {"Metadata_Gene": gene_order, "Metadata_Perturbation": genes}, index=[str(i) for i in range(20)]
        ),
        var=pd.DataFrame(index=[f"f{i}" for i in range(50)]),
    )
    adata.uns["mantispy"] = {"schema_version": "0.1", "resolution": "perturbation"}
    mt.tl.similarity(adata, metric="pearson")
    return adata


def test_planted_interactions_are_enriched(edges):
    adata = _screen([f"G{i}" for i in range(20)], edges)
    mt.tl.network_enrichment(adata, edges=edges, top_quantile=0.9)
    result = adata.uns["mantispy"]["network_enrichment"]
    assert result["odds_ratio"] > 1.0
    assert result["pvalue"] < 0.05
    # Every planted pair was made similar, so all four land among the top pairs.
    assert result["table"][0][0] == 4


def test_shuffled_genes_are_not_enriched(edges):
    # Same profiles, but the gene labels no longer match the injected similarity structure.
    genes = [f"G{i}" for i in range(20)]
    shuffled = list(np.random.default_rng(1).permutation(genes))
    adata = _screen(shuffled, edges)
    mt.tl.network_enrichment(adata, edges=edges, top_quantile=0.9)
    assert adata.uns["mantispy"]["network_enrichment"]["pvalue"] > 0.05


def test_similarity_must_be_computed_first(edges):
    adata = ad.AnnData(
        X=np.zeros((4, 3), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Gene": ["G0", "G1", "G2", "G3"]}, index=list("abcd")),
        var=pd.DataFrame(index=["x", "y", "z"]),
    )
    adata.uns["mantispy"] = {"schema_version": "0.1", "resolution": "perturbation"}
    with pytest.raises(KeyError, match="similarity"):
        mt.tl.network_enrichment(adata, edges=edges)
