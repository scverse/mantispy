"""pp.pseudogenes: control guides relabelled as pseudo-genes, and the reference column a hit caller reads."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy._core.schema import stamp


def _screen(n_control_guides=21, n_cells=3, n_features=2, seed=0):
    """Two real genes of two guides each, and ``n_control_guides`` non-targeting guides, all null."""
    rows = [("GENE_A", "a1"), ("GENE_A", "a2"), ("GENE_B", "b1"), ("GENE_B", "b2")]
    rows += [("nontargeting", f"nt{i}") for i in range(n_control_guides)]
    obs = pd.DataFrame([row for row in rows for _ in range(n_cells)], columns=["Metadata_Gene", "Metadata_sgRNA"])
    obs["Metadata_Control_Type"] = np.where(obs["Metadata_Gene"] == "nontargeting", "negcon", "treatment")
    obs.index = obs.index.astype(str)
    x = np.random.default_rng(seed).normal(size=(len(obs), n_features))
    adata = ad.AnnData(x, obs=obs)
    adata.var_names = [f"Cells_Intensity_F{i}" for i in range(n_features)]
    stamp(adata, resolution="object")
    return adata


def _groups(adata, key="Metadata_Gene_pseudo"):
    """Each pseudo-gene's set of guides."""
    obs = adata.obs[adata.obs[key].str.startswith("pseudogene_")]
    return {name: frozenset(guides["Metadata_sgRNA"].astype(str)) for name, guides in obs.groupby(key)}


def test_controls_are_cut_into_groups_and_the_rest_stay_controls():
    adata = _screen(n_control_guides=21)
    mt.pp.pseudogenes(adata, size=4)
    groups = _groups(adata)
    # Half of 21 guides fills two groups of four; the other 13 stay controls.
    assert sorted(groups) == ["pseudogene_01", "pseudogene_02"]
    assert all(len(guides) == 4 for guides in groups.values())
    obs = adata.obs
    real = obs["Metadata_Control_Type"] != "negcon"
    assert (obs.loc[real, "Metadata_Gene_pseudo"] == obs.loc[real, "Metadata_Gene"]).all()
    left = obs["Metadata_Gene_pseudo"] == "nontargeting"
    assert obs.loc[left, "Metadata_sgRNA"].nunique() == 13
    # The reference column holds the leftover controls only, unless the pseudo-genes are kept in it.
    assert (obs["Metadata_Gene_pseudo_reference"] == left).all()
    mt.pp.pseudogenes(adata, size=4, keep_in_reference=True)
    assert (obs["Metadata_Gene_pseudo_reference"] == ~real).all()


def test_n_groups_sets_the_count_and_a_leftover_short_of_size_stays_a_control():
    adata = _screen(n_control_guides=10)
    mt.pp.pseudogenes(adata, size=3, n_groups=3, key_added="pg")
    assert sorted(_groups(adata, "pg")) == ["pseudogene_01", "pseudogene_02", "pseudogene_03"]
    assert adata.obs.loc[adata.obs["pg"] == "nontargeting", "Metadata_sgRNA"].nunique() == 1
    assert adata.obs["pg_reference"].sum() == 3


def test_groups_follow_the_seed_and_not_the_guide_column_dtype():
    adata = _screen()
    mt.pp.pseudogenes(adata, seed=1)
    first = _groups(adata)
    mt.pp.pseudogenes(adata, seed=1)
    assert _groups(adata) == first
    mt.pp.pseudogenes(adata, seed=2)
    assert _groups(adata) != first
    shuffled = adata[np.random.default_rng(0).permutation(adata.n_obs)].copy()
    shuffled.obs["Metadata_sgRNA"] = shuffled.obs["Metadata_sgRNA"].astype("category")
    mt.pp.pseudogenes(shuffled, seed=1)
    assert _groups(shuffled) == first


def test_integer_guide_ids_group_like_their_text():
    adata = _screen()
    adata.obs["Metadata_sgRNA"] = pd.factorize(adata.obs["Metadata_sgRNA"])[0]
    as_text = adata.copy()
    as_text.obs["Metadata_sgRNA"] = as_text.obs["Metadata_sgRNA"].astype(str)
    mt.pp.pseudogenes(adata)
    mt.pp.pseudogenes(as_text)
    assert (adata.obs["Metadata_Gene_pseudo"] == as_text.obs["Metadata_Gene_pseudo"]).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"size": 0}, "size must be"), ({"n_groups": 0}, "n_groups must be"), ({"size": 11}, "cannot fill")],
)
def test_bad_calls_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        mt.pp.pseudogenes(_screen(), **kwargs)


def test_bootstrap_calls_pseudogenes_of_a_null_screen_at_about_alpha():
    """End to end on a screen of control guides only: the share of pseudo-gene calls is near the nominal rate."""
    rng = np.random.default_rng(0)
    n_guides, n_features = 80, 40
    cells = rng.integers(30, 90, size=n_guides)
    offsets = rng.normal(scale=0.3, size=(n_guides, n_features))
    x = np.concatenate([rng.normal(size=(n, n_features)) + offset for n, offset in zip(cells, offsets, strict=True)])
    obs = pd.DataFrame({"Metadata_sgRNA": np.repeat([f"nt{i}" for i in range(n_guides)], cells)})
    obs["Metadata_Gene"] = "nontargeting"
    obs["Metadata_Control_Type"] = "negcon"
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(x, obs=obs)
    adata.var_names = [f"Cells_Intensity_F{i}" for i in range(n_features)]
    stamp(adata, resolution="object")

    mt.pp.pseudogenes(adata, size=4)
    mt.tl.bootstrap_features(adata, groupby="Metadata_Gene_pseudo", reference="Metadata_Gene_pseudo_reference")
    table = adata.uns["mantispy"]["bootstrap"]
    assert table["group"].str.startswith("pseudogene_").all()
    assert table["group"].nunique() == 10
    assert 0.02 < (table["pvalue"] < 0.05).mean() < 0.085
