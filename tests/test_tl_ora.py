"""Over-representation of gene sets among a group's genes."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantispy as mt


@pytest.fixture
def net():
    """Three disjoint sets of ten genes each."""
    return pd.DataFrame(
        {
            "source": sum([[s] * 10 for s in ["A", "B", "C"]], []),
            "target": [f"g{i}" for i in range(30)],
        }
    )


@pytest.fixture
def screen():
    """Thirty genes in three clusters, each cluster the genes of one set."""
    genes = [f"g{i}" for i in range(30)]
    clusters = ["1"] * 10 + ["2"] * 10 + ["3"] * 10
    adata = ad.AnnData(
        X=np.zeros((30, 3), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Gene": genes, "cluster": clusters}, index=[str(i) for i in range(30)]),
        var=pd.DataFrame(index=["a", "b", "c"]),
    )
    adata.uns["mantispy"] = {"schema_version": "0.1", "resolution": "perturbation"}
    return adata


def test_the_seeded_set_is_the_top_hit(screen, net):
    mt.tl.ora(screen, groupby="cluster", net=net, tmin=1)
    table = screen.uns["mantispy"]["ora"]
    assert {"group", "source", "n", "odds_ratio", "pvalue", "qvalue"} == set(table.columns)

    top = table[table["group"] == "1"].sort_values("pvalue").iloc[0]
    assert top["source"] == "A"  # cluster 1 is exactly set A's genes
    assert top["n"] == 10
    assert top["odds_ratio"] > 0
    assert top["pvalue"] < 0.05


def test_a_set_the_group_avoids_is_depleted_not_over_represented(screen, net):
    mt.tl.ora(screen, groupby="cluster", net=net, tmin=1)
    table = screen.uns["mantispy"]["ora"]
    other = table[(table["group"] == "1") & (table["source"] == "B")].iloc[0]
    # No genes overlap and the odds ratio is negative: the set is depleted, the opposite of over-represented.
    # The Fisher test is two-tailed (as decoupler's is), so depletion is significant too; the sign, not the p, marks direction.
    assert other["n"] == 0
    assert other["odds_ratio"] < 0


def test_ora_handles_a_control_with_no_gene(net):
    # A control well carries no gene; ora must skip it, not crash on the shorter, dropna'd gene Series.
    genes = [f"g{i}" for i in range(30)] + [None]
    clusters = ["1"] * 10 + ["2"] * 10 + ["3"] * 10 + ["ctrl"]
    adata = ad.AnnData(
        X=np.zeros((31, 3), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Gene": genes, "cluster": clusters}, index=[str(i) for i in range(31)]),
        var=pd.DataFrame(index=["a", "b", "c"]),
    )
    adata.uns["mantispy"] = {"schema_version": "0.1", "resolution": "perturbation"}
    mt.tl.ora(adata, groupby="cluster", net=net, tmin=1)
    table = adata.uns["mantispy"]["ora"]
    assert "ctrl" not in set(table["group"])  # no gene, so no tests
    assert table[table["group"] == "1"].sort_values("pvalue").iloc[0]["source"] == "A"


def test_sets_below_tmin_are_skipped_without_crashing(screen):
    # Sets with fewer than tmin measured genes must be dropped, not sent to a test that would assert.
    small_net = pd.DataFrame({"source": ["A", "A", "B", "B"], "target": ["g0", "g1", "g2", "g3"]})
    mt.tl.ora(screen, groupby="cluster", net=small_net)  # default tmin=5, every set has 2 genes
    assert len(screen.uns["mantispy"]["ora"]) == 0


def test_net_is_required(screen):
    with pytest.raises(ValueError, match="net is required"):
        mt.tl.ora(screen, groupby="cluster")


def test_a_missing_gene_column_is_reported(screen, net):
    with pytest.raises(KeyError, match="Metadata_gene_name"):
        mt.tl.ora(screen, groupby="cluster", net=net, gene_key="Metadata_gene_name", tmin=1)
