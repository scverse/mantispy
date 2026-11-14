"""pp.pseudogenes, and tl.bootstrap_features testing each pseudo-gene against every control guide but its own."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy._core.schema import set_control_columns, stamp


def _screen(n_control_guides=10, n_cells=8, n_features=3, seed=0):
    """Two real genes of two guides each, one shifted, and ``n_control_guides`` non-targeting guides."""
    rows = [("GENE_A", "a1"), ("GENE_A", "a2"), ("GENE_B", "b1"), ("GENE_B", "b2")]
    rows += [("nontargeting", f"nt{i}") for i in range(n_control_guides)]
    obs = pd.DataFrame([row for row in rows for _ in range(n_cells)], columns=["Metadata_Gene", "Metadata_sgRNA"])
    obs["Metadata_Control_Type"] = np.where(obs["Metadata_Gene"] == "nontargeting", "negcon", "treatment")
    obs.index = obs.index.astype(str)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(len(obs), n_features)) + 0.3 * rng.normal(size=(len(rows), n_features)).repeat(n_cells, 0)
    x[(obs["Metadata_Gene"] == "GENE_A").to_numpy()] += 1.0
    adata = ad.AnnData(x, obs=obs)
    adata.var_names = [f"Cells_Intensity_F{i}" for i in range(n_features)]
    stamp(adata, resolution="object")
    return adata


def _groups(adata, key="Metadata_Gene_pseudo"):
    """Each pseudo-gene's set of guides."""
    obs = adata.obs[adata.obs[key].str.startswith("pseudogene_")]
    return {name: frozenset(guides["Metadata_sgRNA"].astype(str)) for name, guides in obs.groupby(key)}


def test_controls_are_cut_into_groups_and_the_leftover_stays_a_control():
    adata = _screen(n_control_guides=10)
    mt.pp.pseudogenes(adata, size=4)
    groups = _groups(adata)
    assert sorted(groups) == ["pseudogene_01", "pseudogene_02"]
    assert all(len(guides) == 4 for guides in groups.values())
    obs = adata.obs
    real = obs["Metadata_Control_Type"] != "negcon"
    assert (obs.loc[real, "Metadata_Gene_pseudo"] == obs.loc[real, "Metadata_Gene"]).all()
    assert obs.loc[obs["Metadata_Gene_pseudo"] == "nontargeting", "Metadata_sgRNA"].nunique() == 2


def test_groups_follow_the_seed_and_not_the_guide_column_dtype():
    adata = _screen(n_control_guides=21)
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
    adata = _screen(n_control_guides=21)
    adata.obs["Metadata_sgRNA"] = pd.factorize(adata.obs["Metadata_sgRNA"])[0]
    as_text = adata.copy()
    as_text.obs["Metadata_sgRNA"] = as_text.obs["Metadata_sgRNA"].astype(str)
    mt.pp.pseudogenes(adata)
    mt.pp.pseudogenes(as_text)
    assert (adata.obs["Metadata_Gene_pseudo"] == as_text.obs["Metadata_Gene_pseudo"]).all()


@pytest.mark.parametrize(("kwargs", "match"), [({"size": 0}, "size must be"), ({"size": 11}, "fewer than size")])
def test_bad_calls_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        mt.pp.pseudogenes(_screen(), **kwargs)


def _pvalues(adata, key="bootstrap"):
    return adata.uns["mantispy"][key].set_index(["group", "feature"])["pvalue"]


def test_a_pseudogene_is_tested_against_every_control_guide_but_its_own():
    adata = _screen()
    mt.pp.pseudogenes(adata, size=4)
    mt.tl.bootstrap_features(adata, groupby="Metadata_Gene_pseudo")
    ours = _pvalues(adata)
    for name, guides in _groups(adata).items():
        # The same test by hand: its guides are taken out of the controls and it is tested as a gene.
        alone = adata.copy()
        own = alone.obs["Metadata_sgRNA"].isin(guides).to_numpy()
        alone.obs["Metadata_Gene"] = np.where(own, name, alone.obs["Metadata_Gene"])
        alone.obs["is_ref"] = (alone.obs["Metadata_Control_Type"] == "negcon").to_numpy() & ~own
        alone = alone[own | alone.obs["is_ref"].to_numpy()].copy()
        mt.tl.bootstrap_features(alone, reference="is_ref")
        np.testing.assert_allclose(ours[name].to_numpy(), _pvalues(alone)[name].to_numpy(), rtol=1e-12, atol=0)


def test_real_genes_are_unchanged_by_pseudogenes():
    plain = _screen()
    mt.tl.bootstrap_features(plain)
    relabelled = plain.copy()
    mt.pp.pseudogenes(relabelled, size=4)
    mt.tl.bootstrap_features(relabelled, groupby="Metadata_Gene_pseudo")
    for key in ("bootstrap", "bootstrap_guides"):
        table = relabelled.uns["mantispy"][key]
        real = table[~table["group"].str.startswith("pseudogene_")].reset_index(drop=True)
        pd.testing.assert_frame_equal(real.drop(columns="qvalue"), plain.uns["mantispy"][key].drop(columns="qvalue"))


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
    mt.tl.bootstrap_features(adata, groupby="Metadata_Gene_pseudo")
    table = adata.uns["mantispy"]["bootstrap"]
    assert table["group"].nunique() == 20
    assert 0.02 < (table["pvalue"] < 0.05).mean() < 0.085


def test_a_pseudogene_label_without_the_record_is_an_ordinary_control_label():
    adata = _screen()
    mt.tl.bootstrap_features(adata)
    expected = _pvalues(adata)
    adata.obs["Metadata_Gene"] = np.where(
        adata.obs["Metadata_Gene"] == "nontargeting", "pseudogene_01", adata.obs["Metadata_Gene"]
    )
    mt.tl.bootstrap_features(adata)
    assert "pseudogene_01" not in _pvalues(adata).index.get_level_values("group")
    pd.testing.assert_series_equal(_pvalues(adata), expected)


def test_the_leave_own_out_test_survives_a_round_trip(tmp_path):
    adata = mt.ds.synthetic_plate(n_plates=1, n_wells=24, n_cells=10, n_features=3, seed=0)
    guide = np.arange(adata.n_obs) % 14
    adata.obs["Metadata_sgRNA"] = [f"g{i}" for i in guide]
    adata.obs["Metadata_Gene"] = np.where(guide < 10, "nontargeting", np.where(guide < 12, "GENE_A", "GENE_B"))
    set_control_columns(adata.obs, guide < 10)
    mt.pp.pseudogenes(adata, size=4)
    path = tmp_path / "screen.h5ad"
    mt.io.write(adata, path)
    read = mt.io.read(path)
    assert mt.io.validate(read).ok
    mt.tl.bootstrap_features(adata, groupby="Metadata_Gene_pseudo")
    mt.tl.bootstrap_features(read, groupby="Metadata_Gene_pseudo")
    pd.testing.assert_series_equal(_pvalues(read), _pvalues(adata))
    assert "pseudogene_01" in _pvalues(read).index.get_level_values("group")
