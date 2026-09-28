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
    assert top["source"] == "A"
    assert top["n"] == 10
    assert top["odds_ratio"] > 0
    assert top["pvalue"] < 0.05


def test_a_set_the_group_avoids_is_depleted_not_over_represented(screen, net):
    mt.tl.ora(screen, groupby="cluster", net=net, tmin=1)
    table = screen.uns["mantispy"]["ora"]
    other = table[(table["group"] == "1") & (table["source"] == "B")].iloc[0]
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
    assert "ctrl" not in set(table["group"])
    assert table[table["group"] == "1"].sort_values("pvalue").iloc[0]["source"] == "A"


def test_sets_below_tmin_are_skipped_without_crashing(screen):
    # Sets with fewer than tmin measured genes must be dropped, not sent to a test that would assert.
    small_net = pd.DataFrame({"source": ["A", "A", "B", "B"], "target": ["g0", "g1", "g2", "g3"]})
    mt.tl.ora(screen, groupby="cluster", net=small_net)  # default tmin=5, every set has 2 genes
    assert len(screen.uns["mantispy"]["ora"]) == 0


def test_padj_by_group_is_less_conservative_than_pooling_every_group():
    # One strongly enriched group among many null ones, so pooling all their tests into one BH correction over-penalizes it.
    genes = [f"g{i}" for i in range(200)]
    set_genes = genes[:20]
    net = pd.DataFrame({"source": ["S"] * 20, "target": set_genes})

    gene_col = set_genes[:10]
    clusters = ["hit"] * 10
    nonset = genes[20:]
    for g in range(45):
        gene_col += nonset[g * 4 : g * 4 + 4]
        clusters += [f"n{g}"] * 4

    adata = ad.AnnData(
        X=np.zeros((len(gene_col), 3), dtype=np.float32),
        obs=pd.DataFrame(
            {"Metadata_Gene": gene_col, "cluster": clusters}, index=[str(i) for i in range(len(gene_col))]
        ),
        var=pd.DataFrame(index=["a", "b", "c"]),
    )
    adata.uns["mantispy"] = {"schema_version": "0.1", "resolution": "perturbation"}

    pooled, per_group = adata.copy(), adata.copy()
    mt.tl.ora(pooled, groupby="cluster", net=net, tmin=1, padj_by="all")
    mt.tl.ora(per_group, groupby="cluster", net=net, tmin=1, padj_by="group")
    t_all = pooled.uns["mantispy"]["ora"]
    t_grp = per_group.uns["mantispy"]["ora"]

    merged = t_all.merge(t_grp, on=["group", "source"], suffixes=("_all", "_grp"))
    assert np.allclose(merged["pvalue_all"], merged["pvalue_grp"])

    q_all = t_all.loc[t_all["group"] == "hit", "qvalue"].min()
    q_grp = t_grp.loc[t_grp["group"] == "hit", "qvalue"].min()
    assert q_grp < q_all

    with pytest.raises(ValueError, match="padj_by must be 'all' or 'group'"):
        mt.tl.ora(adata.copy(), groupby="cluster", net=net, tmin=1, padj_by="within")


def test_net_is_required(screen):
    with pytest.raises(ValueError, match="net is required"):
        mt.tl.ora(screen, groupby="cluster")


def test_a_missing_gene_column_is_reported(screen, net):
    with pytest.raises(KeyError, match="Metadata_gene_name"):
        mt.tl.ora(screen, groupby="cluster", net=net, gene_key="Metadata_gene_name", tmin=1)
