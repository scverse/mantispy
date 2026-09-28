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


def test_a_set_the_group_avoids_is_not_a_tested_hypothesis(screen, net):
    # By default (min_overlap=1) a set with none of the group's genes cannot be over-represented, so it is
    # not a tested hypothesis: it is dropped from the group's rows rather than sitting there depleted.
    mt.tl.ora(screen, groupby="cluster", net=net, tmin=1)
    table = screen.uns["mantispy"]["ora"]
    group_one = table[table["group"] == "1"]
    assert "B" not in set(group_one["source"])  # cluster 1 shares no gene with set B
    assert (group_one["n"] >= 1).all()  # every set left in a group's rows overlaps that group


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


def test_padj_by_group_is_less_conservative_than_pooling_every_group():
    # One strongly enriched "hit" group among many null groups. Pooling every group x set test into a single
    # Benjamini-Hochberg correction over-penalizes the hit; correcting within each group does not.
    genes = [f"g{i}" for i in range(200)]
    set_genes = genes[:20]
    net = pd.DataFrame({"source": ["S"] * 20, "target": set_genes})

    gene_col = set_genes[:10]  # hit group: ten genes, all from set S
    clusters = ["hit"] * 10
    nonset = genes[20:]
    spare_set_genes = set_genes[10:]  # ten set-S genes the hit group does not use
    for g in range(45):  # 45 null groups, each with one set-S gene so they stay tested (a>=1) but null
        gene_col += [spare_set_genes[g % 10], *nonset[g * 3 : g * 3 + 3]]
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

    # 1. The correction scope does not change the test itself, so the p-values are identical.
    merged = t_all.merge(t_grp, on=["group", "source"], suffixes=("_all", "_grp"))
    assert np.allclose(merged["pvalue_all"], merged["pvalue_grp"])

    # 2. The hit group's best q is strictly smaller when the other groups do not weigh on its correction.
    q_all = t_all.loc[t_all["group"] == "hit", "qvalue"].min()
    q_grp = t_grp.loc[t_grp["group"] == "hit", "qvalue"].min()
    assert q_grp < q_all

    # 3. An unknown scope is rejected.
    with pytest.raises(ValueError, match="padj_by must be 'all' or 'group'"):
        mt.tl.ora(adata.copy(), groupby="cluster", net=net, tmin=1, padj_by="within")


def test_min_overlap_scopes_the_tested_and_corrected_sets():
    # min_overlap bounds which sets are tested for a group, and the q-values are corrected only over the
    # retained (overlapping) sets, so untested zero-overlap sets do not inflate the correction denominator.
    genes = [f"g{i}" for i in range(30)]
    net = pd.DataFrame({"source": sum([[s] * 10 for s in ["X", "Y", "Z"]], []), "target": genes})

    grp_genes = ["g0", "g1", "g2", "g10"]  # three in set X, one in set Y, none in set Z
    rest_genes = [g for g in genes if g not in grp_genes]  # fill the universe so every set has measured genes
    gene_col = grp_genes + rest_genes
    clusters = ["grp"] * len(grp_genes) + ["rest"] * len(rest_genes)
    adata = ad.AnnData(
        X=np.zeros((len(gene_col), 3), dtype=np.float32),
        obs=pd.DataFrame(
            {"Metadata_Gene": gene_col, "cluster": clusters}, index=[str(i) for i in range(len(gene_col))]
        ),
        var=pd.DataFrame(index=["a", "b", "c"]),
    )
    adata.uns["mantispy"] = {"schema_version": "0.1", "resolution": "perturbation"}

    # 1. min_overlap=1 keeps only the sets grp's genes hit (X and Y), not the zero-overlap set Z.
    one = adata.copy()
    mt.tl.ora(one, groupby="cluster", net=net, tmin=1, padj_by="group", min_overlap=1)
    grp = one.uns["mantispy"]["ora"]
    grp = grp[grp["group"] == "grp"]
    assert set(grp["source"]) == {"X", "Y"}  # set Z shares no gene with grp and is dropped
    assert (grp["n"] >= 1).all()

    from scipy.stats import fisher_exact

    from mantispy._core._stats import benjamini_hochberg

    p = dict(zip(grp["source"], grp["pvalue"], strict=True))
    q = dict(zip(grp["source"], grp["qvalue"], strict=True))
    # Set Z's two-tailed Fisher p-value, had the zero-overlap set stayed in the table: a=0, k=4, set size 10,
    # universe 30, so b=4, c=10, d=16.
    p_z = float(fisher_exact([[0, 4], [10, 16]], alternative="two-sided")[1])
    q_reduced = benjamini_hochberg(np.array([p["X"], p["Y"]]))
    q_all = benjamini_hochberg(np.array([p["X"], p["Y"], p_z]))
    # The run corrected over the two overlapping sets, and X's q is strictly smaller than a manual all-sets
    # correction that let the untested set Z enlarge the family.
    assert np.isclose(q["X"], q_reduced[0])
    assert q["X"] < q_all[0]

    # 2. min_overlap=2 keeps only sets with at least two of grp's genes (X has three, Y has one).
    two = adata.copy()
    mt.tl.ora(two, groupby="cluster", net=net, tmin=1, padj_by="group", min_overlap=2)
    grp2 = two.uns["mantispy"]["ora"]
    grp2 = grp2[grp2["group"] == "grp"]
    assert set(grp2["source"]) == {"X"}

    # 3. A min_overlap below 1, or not an integer, is rejected.
    with pytest.raises(ValueError, match="min_overlap must be an int >= 1"):
        mt.tl.ora(adata.copy(), groupby="cluster", net=net, tmin=1, min_overlap=0)
    with pytest.raises(ValueError, match="min_overlap must be an int >= 1"):
        mt.tl.ora(adata.copy(), groupby="cluster", net=net, tmin=1, min_overlap=1.5)


def test_net_is_required(screen):
    with pytest.raises(ValueError, match="net is required"):
        mt.tl.ora(screen, groupby="cluster")


def test_a_missing_gene_column_is_reported(screen, net):
    with pytest.raises(KeyError, match="Metadata_gene_name"):
        mt.tl.ora(screen, groupby="cluster", net=net, gene_key="Metadata_gene_name", tmin=1)
