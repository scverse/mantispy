"""Prior-knowledge gene sets on a genetic screen."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.stats as st

import mantispy as mt


@pytest.fixture
def net():
    """A small offline network, so nothing here needs a download."""
    return pd.DataFrame(
        {
            "source": ["cycle"] * 6 + ["other"] * 6,
            "target": [f"G{i}" for i in range(6)] + [f"G{i}" for i in range(6, 12)],
            "weight": 1.0,
        }
    )


@pytest.fixture
def screen():
    """A genetic screen where the 'cycle' genes share a phenotype."""
    rng = np.random.default_rng(0)
    genes = [f"G{i}" for i in range(12)]
    values = rng.standard_normal((12, 10))
    values[:6, :4] += 4.0
    adata = ad.AnnData(
        X=values.astype(np.float32),
        obs=pd.DataFrame({"Metadata_Gene": genes}, index=[str(i) for i in range(12)]),
        var=pd.DataFrame(index=[f"f{i}" for i in range(10)]),
    )
    adata.uns["mantispy"] = {"schema_version": "0.1", "resolution": "perturbation"}
    return adata


def test_pathway_coherence_ranks_the_coherent_set_first(screen, net):
    mt.tl.pathway_coherence(screen, net, min_genes=3, n_permutations=200)
    table = screen.uns["mantispy"]["pathway_coherence"]
    assert {"set", "n_genes", "coherence", "pvalue", "qvalue"} <= set(table.columns)
    assert table["set"].iloc[0] == "cycle"  # sorted by coherence, never by p-value
    assert table["coherence"].iloc[0] > table["coherence"].iloc[-1]


def test_a_set_with_too_few_genes_present_is_skipped(screen, net):
    mt.tl.pathway_coherence(screen, net, min_genes=7, n_permutations=50)
    assert len(screen.uns["mantispy"]["pathway_coherence"]) == 0


def test_enrich_hits_finds_the_right_set(screen, net):
    screen.obs["hits_qvalue"] = [0.001] * 6 + [0.9] * 6
    mt.tl.enrich_hits(screen, net, threshold=0.05)
    table = screen.uns["mantispy"]["enrich_hits"]
    assert table["set"].iloc[0] == "cycle"
    assert table["odds_ratio"].iloc[0] > 0


def test_enrich_hits_says_what_to_run_first(screen, net):
    with pytest.raises(KeyError, match="mt.tl.hit_calling"):
        mt.tl.enrich_hits(screen, net)


def test_gene_sets_reads_a_gmt_file(tmp_path):
    gmt = tmp_path / "sets.gmt"
    gmt.write_text("setA\tdescription\tG0\tG1\tG2\nsetB\tdescription\tG5\tG6\n")
    net = mt.tl.gene_sets(source=str(gmt))
    assert list(net.columns) == ["source", "target", "weight"]
    assert set(net["source"]) == {"setA", "setB"}


def test_gene_sets_rejects_an_unknown_source():
    with pytest.raises(ValueError, match="source must be one of"):
        mt.tl.gene_sets(source="nonsense")


def test_the_coherence_plot_draws_and_refuses_an_empty_table(screen, net):
    import matplotlib

    matplotlib.use("Agg")
    mt.tl.pathway_coherence(screen, net, min_genes=3, n_permutations=50)
    assert isinstance(mt.pl.pathway_coherence(screen), matplotlib.axes.Axes)

    screen.uns["mantispy"]["pathway_coherence"] = screen.uns["mantispy"]["pathway_coherence"].iloc[:0]
    with pytest.raises(ValueError, match="is empty"):
        mt.pl.pathway_coherence(screen)


def test_enrich_hits_uses_the_screened_genes_as_the_background():
    """A 0/1 membership row makes decoupler drop the non-hit genes as empty and default
    n_bg to 20000, which inflates enrichment by orders of magnitude and removes depletion
    from the table."""
    import anndata as ad

    from mantispy._core.schema import stamp

    genes = [f"G{i:03d}" for i in range(200)]
    adata = ad.AnnData(
        np.zeros((len(genes), 2), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "Metadata_Plate": "P1",
                "Metadata_Well": [f"A{i:04d}" for i in range(len(genes))],
                "Metadata_Gene": genes,
                # The first 40 genes are hits; the rest are the background this test is about.
                "hits_qvalue": np.where(np.arange(len(genes)) < 40, 0.001, 0.5),
            },
            index=[str(i) for i in range(len(genes))],
        ),
        var=pd.DataFrame(index=["Cells_AreaShape_Area", "Cells_AreaShape_Perimeter"]),
    )
    stamp(adata, resolution="perturbation")

    net = pd.DataFrame(
        {
            "source": ["S1"] * 20 + ["S2"] * 20,
            "target": genes[30:50] + genes[100:120],
            "weight": 1.0,
        }
    )
    mt.tl.enrich_hits(adata, net=net, gene_key="Metadata_Gene", threshold=0.05)
    table = adata.uns["mantispy"]["enrich_hits"].set_index("set")

    assert {"S1", "S2"} <= set(table.index), "a set with no hits is still a result"
    # S1: 10 of its 20 genes are hits, against 40 hits in 200 screened genes.
    expected = st.fisher_exact([[10, 10], [30, 150]])[1]
    assert abs(np.log10(float(table.loc["S1", "qvalue"])) - np.log10(expected)) < 2


def _screen_of_200_genes(hits_qvalue):
    """The same 200-gene fixture as the background test above, with a caller-chosen
    per-gene q-value so the two boundary tests below can put every gene on one side of the
    hit/non-hit line."""
    import anndata as ad

    from mantispy._core.schema import stamp

    genes = [f"G{i:03d}" for i in range(200)]
    adata = ad.AnnData(
        np.zeros((len(genes), 2), dtype=np.float32),
        obs=pd.DataFrame(
            {
                "Metadata_Plate": "P1",
                "Metadata_Well": [f"A{i:04d}" for i in range(len(genes))],
                "Metadata_Gene": genes,
                "hits_qvalue": hits_qvalue,
            },
            index=[str(i) for i in range(len(genes))],
        ),
        var=pd.DataFrame(index=["Cells_AreaShape_Area", "Cells_AreaShape_Perimeter"]),
    )
    stamp(adata, resolution="perturbation")
    return adata, genes


def test_enrich_hits_refuses_a_screen_with_no_hits():
    """A clean screen calls nothing. Without a guard, decoupler's ORA gets n_up = n_bg and
    raises a bare ValueError('cannot compute fingerprint of empty set') that names neither
    the threshold nor the remedy."""
    adata, genes = _screen_of_200_genes(0.5)  # every gene misses threshold=0.05
    net = pd.DataFrame({"source": ["S1"] * 20, "target": genes[30:50], "weight": 1.0})

    with pytest.raises(ValueError, match="threshold"):
        mt.tl.enrich_hits(adata, net=net, gene_key="Metadata_Gene", threshold=0.05)


def test_enrich_hits_refuses_a_screen_where_everything_is_a_hit():
    """A permissive threshold calls everything. Without a guard, decoupler's ORA gets
    n_up = 0 and raises a bare AssertionError('n_up must be numeric and > 0'). rohban at
    190 hits of 193 genes (see the Notes of enrich_hits) still works; a screen where every
    gene is a hit raises a clear error."""
    adata, genes = _screen_of_200_genes(0.001)  # every gene beats threshold=0.05
    net = pd.DataFrame({"source": ["S1"] * 20, "target": genes[30:50], "weight": 1.0})

    with pytest.raises(ValueError, match="threshold"):
        mt.tl.enrich_hits(adata, net=net, gene_key="Metadata_Gene", threshold=0.05)
