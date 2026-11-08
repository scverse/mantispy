"""Cluster composition, cell cycle, subpopulation hits and local density."""

import logging
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from test_tl_hits import _well_effect_cells

import mantispy as mt
from mantispy._core.schema import stamp


def _one_cluster(seed):
    """A well-random-effect cell screen (issue #68 fixture) collapsed into a single shared cluster.

    Every group is then compared against the controls within one cell state, the pseudoreplication setting of #137.
    """
    adata = _well_effect_cells(seed=seed)
    adata.obs["cluster"] = "c0"
    return adata


def _subpopulation_fpr(seeds, *, drop_well):
    """Fraction of non-reference (cluster, group) pairs called at raw p < 0.05 across ``seeds``.

    Dropping the well column forces the analytic KS fallback; keeping it runs the default well-block permutation.
    """
    called = total = 0
    for seed in seeds:
        adata = _one_cluster(seed)
        if drop_well:
            del adata.obs["Metadata_Well"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mt.tl.subpopulation_hits(adata, cluster_key="cluster", seed=seed, n_permutations=300)
        table = adata.uns["mantispy"]["subpopulation_hits"]
        treated = table[table["group"] != "DMSO"]
        called += int((treated["pvalue"] < 0.05).sum())
        total += int(len(treated))
    return called / total


def test_subpopulation_hits_well_block_fixes_cell_resolution_pseudoreplication():
    """Regression for #137: the analytic KS p-value is built from cell counts, so cells within a well inflate it.

    A well-block permutation matches the null's unit to the design and brings the pure-null false positive rate back to nominal.
    """
    block_rate = _subpopulation_fpr(range(8), drop_well=False)
    analytic_rate = _subpopulation_fpr(range(8), drop_well=True)
    assert block_rate <= 0.08, f"the well-block null called {block_rate:.1%} of pure-null pseudo-treatments"
    assert analytic_rate >= 0.2, f"the analytic path called {analytic_rate:.1%}, so the bug is not reproduced"


def test_subpopulation_hits_defaults_to_the_well_block_at_cell_resolution():
    adata = _one_cluster(seed=0)
    default = mt.tl.subpopulation_hits(adata, cluster_key="cluster", n_permutations=200, seed=0, copy=True)
    explicit = mt.tl.subpopulation_hits(
        adata, cluster_key="cluster", block="Metadata_Well", n_permutations=200, seed=0, copy=True
    )
    pd.testing.assert_frame_equal(
        default.uns["mantispy"]["subpopulation_hits"], explicit.uns["mantispy"]["subpopulation_hits"]
    )


def test_subpopulation_hits_warns_without_a_well_column_at_cell_resolution():
    adata = _one_cluster(seed=0)
    del adata.obs["Metadata_Well"]
    with pytest.warns(UserWarning, match="not independent replicates"):
        mt.tl.subpopulation_hits(adata, cluster_key="cluster", n_permutations=100, seed=0)


@pytest.fixture(scope="module")
def clustered():
    cells = mt.ds.synthetic_plate(n_wells=48, n_cells=40, n_features=20, n_perturbations=3, effect_size=4.0, seed=0)
    mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon")
    sc.pp.pca(cells, n_comps=10)
    sc.pp.neighbors(cells)
    sc.tl.leiden(cells, key_added="leiden", flavor="igraph", n_iterations=2)
    return cells


def test_cell_cycle_refuses_normalized_values(clustered):
    """After normalize the values are z-scores, and half of them have no logarithm."""
    with pytest.raises(ValueError, match="raw intensities"):
        mt.tl.cell_cycle_phase(clustered, dna_feature=clustered.var_names[0])


_DNA = "Nuclei_Intensity_IntegratedIntensity_DNA"


def _cycling_wells(shares, scales, n_cells=2000, seed=0):
    """Cells with known 2N, S and 4N shares per well, each well stained at its own intensity."""
    rng = np.random.default_rng(seed)
    values, wells, states = [], [], []
    for well, (share, scale) in enumerate(zip(shares, scales, strict=True)):
        counts = np.round(np.asarray(share) * n_cells).astype(int)
        fold = np.concatenate(
            [
                2 ** rng.normal(0, 0.05, counts[0]),
                rng.uniform(1.35, 1.65, counts[1]),
                2 ** rng.normal(1, 0.05, counts[2]),
            ]
        )
        values.append(scale * fold)
        wells += [f"W{well}"] * counts.sum()
        states += [state for state, count in zip(("2N", "S", "4N"), counts, strict=True) for _ in range(count)]
    values = np.concatenate(values)
    obs = pd.DataFrame(
        {"Metadata_Plate": "P1", "Metadata_Well": wells, "state": states},
        index=[str(row) for row in range(values.size)],
    )
    return ad.AnnData(X=values[:, None].astype(np.float32), obs=obs, var=pd.DataFrame(index=[_DNA]))


def test_dna_content_and_state_recover_known_shares_across_staining_scales():
    """The G1 peak lands on 0 and G2 on 1 in every well, including one arrested mostly in G2."""
    shares = [(0.6, 0.2, 0.2), (0.4, 0.3, 0.3), (0.25, 0.15, 0.6)]
    scales = [300.0, 1000.0, 5000.0]
    cells = _cycling_wells(shares, scales)
    mt.tl.dna_content(cells, dna_feature=_DNA)
    mt.tl.dna_state(cells)

    obs = cells.obs
    for well, share in enumerate(shares):
        in_well = obs["Metadata_Well"] == f"W{well}"
        content = obs.loc[in_well, "Metadata_DNAContent"]
        assert abs(content[obs["state"] == "2N"].median()) < 0.05
        assert abs(content[obs["state"] == "4N"].median() - 1) < 0.05
        called = obs.loc[in_well, "Metadata_DNAState"].value_counts(normalize=True)
        np.testing.assert_allclose(called[["2N", "S", "4N"]].to_numpy(), share, atol=0.02)

    peaks = cells.uns["mantispy"]["dna_content"]
    assert list(peaks["Metadata_Well"]) == ["W0", "W1", "W2"]
    np.testing.assert_allclose(peaks["g1_peak"], np.log2(scales), atol=0.05)
    assert cells.uns["mantispy"]["dna_state"]["key"] == "Metadata_DNAContent"
    np.testing.assert_array_equal(cells.uns["mantispy"]["dna_state"]["cutoffs"], [1.25, 1.75])


def test_dna_state_records_custom_cutoffs_and_refuses_bad_ones():
    cells = _cycling_wells([(0.5, 0.2, 0.3)], [1000.0])
    mt.tl.dna_content(cells, dna_feature=_DNA)
    mt.tl.dna_state(cells, cutoffs=(1.2, 1.8))
    np.testing.assert_array_equal(cells.uns["mantispy"]["dna_state"]["cutoffs"], [1.2, 1.8])
    with pytest.raises(ValueError, match="increasing"):
        mt.tl.dna_state(cells, cutoffs=(1.75, 1.25))
    with pytest.raises(KeyError, match="dna_content"):
        mt.tl.dna_state(cells, key="Metadata_Missing")


def test_dna_content_leaves_unusable_intensities_and_small_wells_unassigned(caplog):
    """A missing, zero or negative intensity has no log2, and a well of a few cells has no G1 peak to measure from."""
    cells = _cycling_wells([(0.5, 0.2, 0.3), (0.5, 0.2, 0.3)], [1000.0, 1000.0])
    cells.X[:3, 0] = [np.nan, 0.0, -5.0]
    cells = cells[np.r_[0 : cells.n_obs - 1990]].copy()  # the second well keeps 10 cells
    with caplog.at_level(logging.WARNING, logger="mantispy"):
        mt.tl.dna_content(cells, dna_feature=_DNA)
    mt.tl.dna_state(cells)

    content, state = cells.obs["Metadata_DNAContent"], cells.obs["Metadata_DNAState"]
    small = (cells.obs["Metadata_Well"] == "W1").to_numpy()
    assert content.iloc[:3].isna().all() and state.iloc[:3].isna().all()
    assert content[small].isna().all() and state[small].isna().all()
    assert content[~small].iloc[3:].notna().all() and state[~small].iloc[3:].notna().all()
    assert np.isnan(cells.uns["mantispy"]["dna_content"]["g1_peak"].iloc[1])
    assert any("too few to place a G1 peak" in record.getMessage() for record in caplog.records)


def test_local_density_is_computed_within_a_field(clustered):
    clustered = clustered.copy()  # module-scoped fixture; this test writes obs coordinates and density
    rng = np.random.default_rng(0)
    clustered.obs["Metadata_Center_X"] = rng.uniform(0, 1000, clustered.n_obs)
    clustered.obs["Metadata_Center_Y"] = rng.uniform(0, 1000, clustered.n_obs)
    mt.tl.neighbors_local_density(clustered, k=5)
    density = clustered.obs["Metadata_LocalDensity"].to_numpy()
    assert np.isfinite(density).all() and (density > 0).all()

    # A cell alone in its field has no neighbours and gets NaN.
    clustered.obs["Metadata_ImageID"] = [str(i) for i in range(clustered.n_obs)]
    mt.tl.neighbors_local_density(clustered, k=5)
    assert clustered.obs["Metadata_LocalDensity"].isna().all()


def test_round_trip(clustered, tmp_path):
    composition = mt.tl.cluster_composition(clustered)
    mt.io.write(composition, tmp_path / "composition.h5ad")
    loaded = mt.io.read(tmp_path / "composition.h5ad")
    assert loaded.n_vars == composition.n_vars
    assert len(loaded.uns["mantispy"]["composition_test"]) == composition.n_obs
    # empty_annotation supplies the empty columns as categoricals so that the h5ad writer keeps them (#103).
    assert list(loaded.var.columns) == list(composition.var.columns)
    assert set(loaded.var["feature_group"]) == {"Composition"}
    for column in ("channel", "radial_bin", "params"):
        assert loaded.var[column].isna().all(), column
        assert isinstance(loaded.var[column].dtype, pd.CategoricalDtype), column


def _null_wells(layout, n_controls):
    """Every well drawn from one composition, no real hits: the first ``n_controls`` are the controls.

    The rest carry a perturbation label but the same null composition, so any of them called at p < 0.05 is a false positive.
    This is the issue #89 reproduction as a well-level object.
    """
    rows = [
        {
            "Metadata_Plate": "P1",
            "Metadata_Well": w,
            "Metadata_Perturbation": "DMSO" if i < n_controls else "pert",
            "Metadata_Control": i < n_controls,
            "Metadata_Control_Type": "negcon" if i < n_controls else "treatment",
            "leiden": str(c),
        }
        for i, (w, clusters) in enumerate(layout.items())
        for c, n in clusters.items()
        for _ in range(n)
    ]
    obs = pd.DataFrame(rows, index=[str(i) for i in range(len(rows))])
    adata = ad.AnnData(
        X=np.zeros((len(rows), 4), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"Cells_AreaShape_f{i}" for i in range(4)]),
    )
    stamp(adata, resolution="object")
    return adata


def _null_plate_fpr(n_controls, trials=200):
    """Fraction of pseudo-treatment wells called at p < 0.05 across ``trials`` pure-null plates."""
    called = total = 0
    for trial in range(trials):
        rng = np.random.default_rng(trial)
        shares = rng.dirichlet(np.full(4, 40.0), size=n_controls * 2)
        layout = {
            f"W{i:03d}": dict(enumerate(np.bincount(rng.choice(4, size=300, p=s), minlength=4)))
            for i, s in enumerate(shares)
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            comp = mt.tl.cluster_composition(_null_wells(layout, n_controls))
        p = comp.uns["mantispy"]["composition_test"]["pvalue"].to_numpy()
        treated = ~comp.obs["Metadata_Control"].to_numpy(dtype=bool)
        called += int((p[treated] < 0.05).sum())
        total += int(treated.sum())
    return called / total


@pytest.mark.parametrize("n_controls", [8, 16, 32, 64])
def test_a_null_plate_is_called_at_most_at_the_nominal_rate(n_controls):
    """Issue #89: with every well drawn from one composition and no real hits, the pure-null false positive rate ran above the nominal 0.05 and worse with fewer controls (0.138, 0.092, 0.070, 0.059 at 8, 16, 32, 64 controls).

    Two causes: the dispersion was estimated from control wells each scored against a pool that included itself, which shrank their statistics and biased it low; and statistic / dispersion was referred to chi-square, treating the estimated dispersion as known.
    Leave-one-out calibration and an F reference each address one, and together bring the false positive rate to nominal for every control count.
    """
    assert _null_plate_fpr(n_controls) <= 0.06


def _clustered_wells(layout: dict[str, dict[int, int]], n_clusters: int):
    """Cells labeled by well and cluster; wells whose name starts with A are the controls."""
    rows = [
        {
            "Metadata_Plate": "P1",
            "Metadata_Well": well,
            "Metadata_Perturbation": "DMSO" if well.startswith("A") else "pert",
            "Metadata_Control": well.startswith("A"),
            "Metadata_Control_Type": "negcon" if well.startswith("A") else "treatment",
            "leiden": str(cluster),
        }
        for well, clusters in layout.items()
        for cluster, count in clusters.items()
        for _ in range(count)
    ]
    obs = pd.DataFrame(rows, index=[str(index) for index in range(len(rows))])
    adata = ad.AnnData(
        X=np.random.default_rng(0).standard_normal((len(rows), 4)).astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=[f"Cells_AreaShape_f{index}" for index in range(4)]),
    )
    assert obs["leiden"].nunique() == n_clusters
    stamp(adata, resolution="object")
    return adata


def test_a_cluster_the_controls_never_reached_does_not_fabricate_a_hit():
    """An expected count floored at 1e-9 turned five cells in a treatment-only cluster into chi-square 7.5e10 and p exactly 0."""
    layout = {
        "A01": {0: 50, 1: 50},
        "A02": {0: 60, 1: 40},
        "A03": {0: 55, 1: 45},
        "B01": {0: 50, 1: 45, 2: 5},  # five cells where no control cell was seen
        "B02": {0: 52, 1: 48},
    }
    composition = mt.tl.cluster_composition(_clustered_wells(layout, n_clusters=3))
    test = composition.uns["mantispy"]["composition_test"].set_index("group")

    assert float(test.loc["P1/B01", "statistic"]) < 1e3
    assert float(test.loc["P1/B01", "pvalue"]) > 0.0
    assert float(test.loc["P1/B01", "qvalue"]) > 0.0
    row = composition.obs_names[composition.obs["Metadata_Well"].astype(str).to_numpy() == "B01"][0]
    assert float(composition[row, "2"].X[0, 0]) == pytest.approx(0.05)


@pytest.mark.filterwarnings("ignore:the controls occupy")
def test_a_well_with_no_assigned_cell_is_left_out(clustered):
    """Zero in every cluster said the well was measured and found empty everywhere; NaN said it was unknown, and tl.map refuses an object with missing values although the Returns clause promises tl.map accepts it.

    A well with nothing to measure is dropped, like any other empty group.
    """
    clustered = clustered.copy()  # module-scoped fixture; this test rewrites obs["leiden"]
    clusters = clustered.obs["leiden"].astype(str)
    wells = clustered.obs["Metadata_Well"].to_numpy()
    blanked = wells == wells[0]
    clustered.obs["leiden"] = pd.Categorical(np.where(blanked, None, clusters))

    composition = mt.tl.cluster_composition(clustered)

    assert wells[0] not in set(composition.obs["Metadata_Well"])
    assert not np.isnan(np.asarray(composition.X)).any(), "the result stays mappable"
    pytest.importorskip("copairs")  # tl.map below; copairs declares requires-python <3.13
    mt.tl.map(composition, mode="activity", null_size=50)


def test_the_drop_is_reported_only_when_something_is_dropped(clustered, caplog):
    """report_drop ran before the no-cluster refusal, so an unclustered object was told its cells were 'left out of the fractions' immediately before being told there are no fractions."""
    clustered = clustered.copy()  # module-scoped fixture; this test rewrites obs["leiden"]
    clustered.obs["leiden"] = pd.Categorical([None] * clustered.n_obs)
    with caplog.at_level(logging.INFO, logger="mantispy"), pytest.raises(ValueError, match="no cell"):
        mt.tl.cluster_composition(clustered)
    assert not [record for record in caplog.records if "left out of the fractions" in record.getMessage()]
