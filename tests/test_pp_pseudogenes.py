"""pp.pseudogenes: grouping control guides into pseudo-genes matched to the real genes in guides and cells."""

import logging

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy._core.schema import stamp


def _screen(n_control_guides=10, n_cells=8, n_control_cells=None, n_features=3, seed=0):
    """Two real genes of two guides each, one shifted, and ``n_control_guides`` non-targeting guides."""
    n_control_cells = n_cells if n_control_cells is None else n_control_cells
    rows = [("GENE_A", "a1", n_cells), ("GENE_A", "a2", n_cells), ("GENE_B", "b1", n_cells), ("GENE_B", "b2", n_cells)]
    rows += [("nontargeting", f"nt{i}", n_control_cells) for i in range(n_control_guides)]
    obs = pd.DataFrame(
        [(gene, guide) for gene, guide, n in rows for _ in range(n)], columns=["Metadata_Gene", "Metadata_sgRNA"]
    )
    obs["Metadata_Control_Type"] = np.where(obs["Metadata_Gene"] == "nontargeting", "negcon", "treatment")
    obs.index = obs.index.astype(str)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(len(obs), n_features))
    x[(obs["Metadata_Gene"] == "GENE_A").to_numpy()] += 1.0
    adata = ad.AnnData(x, obs=obs)
    adata.var_names = [f"Cells_Intensity_F{i}" for i in range(n_features)]
    stamp(adata, resolution="object")
    return adata


def _groups(adata, key="pseudogene"):
    """Each pseudo-gene's set of guides, a guide being its gene and its label."""
    obs = adata.obs[adata.obs[key].notna()]
    return {
        str(name): frozenset(zip(rows["Metadata_Gene"].astype(str), rows["Metadata_sgRNA"].astype(str), strict=True))
        for name, rows in obs.groupby(key, observed=True)
    }


def _kept(adata, key="pseudogene"):
    """Cells each pseudo-gene guide keeps."""
    return adata.obs[adata.obs[key].notna()].groupby(["Metadata_Gene", "Metadata_sgRNA"], observed=True).size()


def test_controls_are_cut_into_groups_and_the_leftover_joins_none():
    adata = _screen(n_control_guides=10)
    before = adata.obs["Metadata_Gene"].copy()
    mt.pp.pseudogenes(adata, size=4)
    groups = _groups(adata)
    assert sorted(groups) == ["pseudogene_01", "pseudogene_02"]
    assert all(len(guides) == 4 for guides in groups.values())
    obs = adata.obs
    assert obs.loc[obs["Metadata_Control_Type"] != "negcon", "pseudogene"].isna().all()
    pd.testing.assert_series_equal(obs["Metadata_Gene"], before)
    assert adata.uns["mantispy"]["pseudogenes"]["pseudogene"] == {
        "groupby": "Metadata_Gene",
        "replicate": "Metadata_sgRNA",
    }


def test_groups_follow_the_seed_and_not_the_row_order_or_guide_column_dtype():
    adata = _screen(n_control_guides=21)
    mt.pp.pseudogenes(adata, seed=1, match_cells=False)
    first = _groups(adata)
    mt.pp.pseudogenes(adata, seed=1, match_cells=False)
    assert _groups(adata) == first
    mt.pp.pseudogenes(adata, seed=2, match_cells=False)
    assert _groups(adata) != first
    shuffled = adata[np.random.default_rng(0).permutation(adata.n_obs)].copy()
    shuffled.obs["Metadata_sgRNA"] = shuffled.obs["Metadata_sgRNA"].astype("category")
    mt.pp.pseudogenes(shuffled, seed=1, match_cells=False)
    assert _groups(shuffled) == first


def test_integer_guide_ids_group_like_their_text():
    adata = _screen(n_control_guides=21)
    adata.obs["Metadata_sgRNA"] = pd.factorize(adata.obs["Metadata_sgRNA"])[0]
    as_text = adata.copy()
    as_text.obs["Metadata_sgRNA"] = as_text.obs["Metadata_sgRNA"].astype(str)
    mt.pp.pseudogenes(adata)
    mt.pp.pseudogenes(as_text)
    pd.testing.assert_series_equal(adata.obs["pseudogene"], as_text.obs["pseudogene"])


def test_control_genes_that_share_a_guide_label_keep_two_guides():
    adata = _screen(n_control_guides=4)
    twin = adata[adata.obs["Metadata_Gene"] == "nontargeting"].copy()
    twin.obs["Metadata_Gene"] = "safe_harbor"
    twin.obs_names = [f"twin{i}" for i in range(twin.n_obs)]
    both = ad.concat([adata, twin])
    stamp(both, resolution="object")
    mt.pp.pseudogenes(both, size=2, match_cells=False)
    # Four labels on two control genes are eight guides, so four pseudo-genes of two.
    groups = _groups(both)
    assert len(groups) == 4 and all(len(guides) == 2 for guides in groups.values())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"size": 0}, "size must be"), ({"size": 11}, "fewer than size"), ({"size": 10}, "all 10 control guides")],
)
def test_bad_calls_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        mt.pp.pseudogenes(_screen(), **kwargs)


def test_matching_needs_a_real_guide():
    adata = _screen()
    adata = adata[adata.obs["Metadata_Control_Type"] == "negcon"].copy()
    with pytest.raises(ValueError, match="no guide lies outside the reference"):
        mt.pp.pseudogenes(adata, size=2)
    mt.pp.pseudogenes(adata, size=2, match_cells=False)
    assert len(_groups(adata)) == 5


def test_a_missing_replicate_column_names_the_argument():
    with pytest.raises(KeyError, match="replicate='Metadata_Guide'"):
        mt.pp.pseudogenes(_screen(), replicate="Metadata_Guide")


def test_the_default_size_is_the_median_guides_per_gene_rounded_half_up(caplog):
    adata = _screen(n_control_guides=10)
    with caplog.at_level(logging.INFO, logger="scverse.mantispy"):
        mt.pp.pseudogenes(adata)
    assert "size=2" in caplog.text
    assert len(_groups(adata)) == 5
    # A third guide of GENE_B makes the median two and a half, which rounds to three.
    adata.obs["Metadata_sgRNA"] = adata.obs["Metadata_sgRNA"].astype(str)
    adata.obs.loc[adata.obs.index[:4], ["Metadata_Gene", "Metadata_sgRNA"]] = ["GENE_B", "b3"]
    mt.pp.pseudogenes(adata)
    assert len(_groups(adata)) == 3 and all(len(guides) == 3 for guides in _groups(adata).values())


def test_missing_genes_and_guides_join_no_pseudogene():
    adata = _screen(n_control_guides=10)
    obs = adata.obs
    obs["Metadata_Gene"] = obs["Metadata_Gene"].astype(object)
    obs["Metadata_sgRNA"] = obs["Metadata_sgRNA"].astype(object)
    no_gene = (obs["Metadata_sgRNA"] == "nt1").to_numpy()
    obs.loc[no_gene, "Metadata_Gene"] = np.nan
    no_guide = (obs["Metadata_sgRNA"] == "nt0").to_numpy()
    obs.loc[no_guide, "Metadata_sgRNA"] = np.nan
    mt.pp.pseudogenes(adata, size=3, match_cells=False)
    assert adata.obs.loc[no_gene | no_guide, "pseudogene"].isna().all()
    assert len(_groups(adata)) == 2 and all(len(guides) == 3 for guides in _groups(adata).values())


def test_each_pseudogene_guide_keeps_a_subset_the_size_of_a_real_guide():
    adata = _screen(n_control_guides=12, n_control_cells=40)
    adata = adata[~((adata.obs["Metadata_sgRNA"] == "b2").to_numpy() & (np.arange(adata.n_obs) % 2 == 0))].copy()
    mt.pp.pseudogenes(adata, size=2, seed=3)
    kept = _kept(adata)
    assert len(kept) == 12 and set(kept) <= {4, 8} and len(set(kept)) == 2
    assert (adata.obs.loc[adata.obs["pseudogene"].notna(), "Metadata_Control_Type"] == "negcon").all()
    mt.pp.pseudogenes(adata, size=2, seed=3, match_cells=False)
    assert set(_kept(adata)) == {40}


def test_controls_smaller_than_the_real_guides_warn_and_keep_every_cell(caplog):
    adata = _screen(n_control_guides=12, n_cells=20, n_control_cells=5)
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        mt.pp.pseudogenes(adata, size=2)
    assert "12 of 12 pseudo-gene guide(s) have fewer control cells" in caplog.text
    assert "180 cell(s) short" in caplog.text
    assert set(_kept(adata)) == {5}


def test_a_backed_object_takes_the_column(tmp_path):
    _screen(n_control_guides=10).write_h5ad(tmp_path / "cells.h5ad")
    backed = ad.read_h5ad(tmp_path / "cells.h5ad", backed="r")
    mt.pp.pseudogenes(backed, size=4)
    assert len(_groups(backed)) == 2
