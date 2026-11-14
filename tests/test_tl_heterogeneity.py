"""Cluster composition, cell cycle, subpopulation hits and local density."""

import logging
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from test_tl_hits import _pooled_cells, _well_effect_cells

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
        adata, cluster_key="cluster", replicate="Metadata_Well", n_permutations=200, seed=0, copy=True
    )
    pd.testing.assert_frame_equal(
        default.uns["mantispy"]["subpopulation_hits"], explicit.uns["mantispy"]["subpopulation_hits"]
    )


def test_subpopulation_hits_warns_without_a_well_column_at_cell_resolution():
    adata = _one_cluster(seed=0)
    del adata.obs["Metadata_Well"]
    with pytest.warns(UserWarning, match="independent replicates"):
        mt.tl.subpopulation_hits(adata, cluster_key="cluster", n_permutations=100, seed=0)


def test_subpopulation_hits_needs_an_explicit_replicate_on_a_pooled_design():
    adata = _pooled_cells()
    adata.obs["cluster"] = "c0"
    with pytest.raises(ValueError, match="pooled screen.*Metadata_sgRNA"):
        mt.tl.subpopulation_hits(adata, cluster_key="cluster", n_permutations=50, seed=0)


def test_subpopulation_hits_finds_a_pooled_effect_with_a_guide_replicate():
    adata = _pooled_cells()
    adata.obs["cluster"] = "c0"
    mt.tl.subpopulation_hits(adata, cluster_key="cluster", replicate="Metadata_sgRNA", n_permutations=1000, seed=0)
    table = adata.uns["mantispy"]["subpopulation_hits"].set_index("group")
    assert table.loc["G", "pvalue"] < 0.05


@pytest.fixture(scope="module")
def clustered():
    cells = mt.ds.synthetic_plate(n_wells=48, n_cells=40, n_features=20, n_perturbations=3, effect_size=4.0, seed=0)
    mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon")
    sc.pp.pca(cells, n_comps=10)
    sc.pp.neighbors(cells)
    sc.tl.leiden(cells, key_added="leiden", flavor="igraph", n_iterations=2)
    return cells


def test_dna_content_refuses_normalized_values(clustered):
    """After normalize the values are z-scores, and half of them have no logarithm."""
    with pytest.raises(ValueError, match="raw intensities"):
        mt.tl.dna_content(clustered, dna_feature=clustered.var_names[0])


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


def test_dna_content_measures_an_arrested_well_against_the_controls_of_its_plate():
    """A well arrested in 4N has no 2N peak of its own; the controls on its plate supply the G1 peak instead."""
    cells = _cycling_wells([(0.6, 0.2, 0.2), (0.0, 0.0, 1.0)], [1000.0, 1000.0])
    cells.obs["Metadata_Control_Type"] = np.where(cells.obs["Metadata_Well"] == "W0", "negcon", "trt")
    arrested = (cells.obs["Metadata_Well"] == "W1").to_numpy()
    mt.tl.dna_content(cells, dna_feature=_DNA)
    assert abs(cells.obs.loc[arrested, "Metadata_DNAContent"].median()) < 0.05  # taken for G1 within its own well
    mt.tl.dna_content(cells, dna_feature=_DNA, by="Metadata_Plate", reference="negcon")
    assert abs(cells.obs.loc[arrested, "Metadata_DNAContent"].median() - 1) < 0.05
    assert cells.uns["mantispy"]["dna_content"]["n_cells"].iloc[0] == (~arrested).sum()


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
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
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


def test_local_density_is_higher_where_cells_are_crowded(clustered):
    # A density, not a distance: the tight cluster must score higher than the sparse one (#271).
    clustered = clustered.copy()
    n = clustered.n_obs
    rng = np.random.default_rng(1)
    tight = np.arange(n) < n // 2
    spread = np.where(tight, 10.0, 200.0)
    centre = np.where(tight, 100.0, 2000.0)
    clustered.obs["Metadata_Center_X"] = centre + rng.normal(0, 1, n) * spread
    clustered.obs["Metadata_Center_Y"] = centre + rng.normal(0, 1, n) * spread
    clustered.obs["Metadata_ImageID"] = "field"
    mt.tl.neighbors_local_density(clustered, k=5)
    density = clustered.obs["Metadata_LocalDensity"].to_numpy()
    assert np.median(density[tight]) > 10 * np.median(density[~tight])


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
    mt.tl.map(composition, mode="activity", null_size=50)


def test_the_drop_is_reported_only_when_something_is_dropped(clustered, caplog):
    """report_drop ran before the no-cluster refusal, so an unclustered object was told its cells were 'left out of the fractions' immediately before being told there are no fractions."""
    clustered = clustered.copy()  # module-scoped fixture; this test rewrites obs["leiden"]
    clustered.obs["leiden"] = pd.Categorical([None] * clustered.n_obs)
    with caplog.at_level(logging.INFO, logger="scverse.mantispy"), pytest.raises(ValueError, match="no cell"):
        mt.tl.cluster_composition(clustered)
    assert not [record for record in caplog.records if "left out of the fractions" in record.getMessage()]


def _phase_object(n_wells, ko_g1, guides=4, ntc_guides=4, cells_per_guide=60, guide_sd=0.03, seed=0):
    """NTC and KO guides in every well; NTC cells are 40% G1, KO cells ``ko_g1``.

    Each guide carries its own G1 offset into every well, as an off-target or clonal effect would.
    """
    rng = np.random.default_rng(seed)
    offsets = {
        (gene, guide): rng.normal(0.0, guide_sd)
        for gene, n in (("NTC", ntc_guides), ("KO", guides))
        for guide in range(n)
    }
    rows = []
    for well in range(n_wells):
        for (gene, guide), offset in offsets.items():
            share = np.clip((0.4 if gene == "NTC" else ko_g1) + offset, 0.0, 1.0)
            phases = rng.choice(["G1", "S", "G2M"], size=cells_per_guide, p=[share, (1 - share) / 2, (1 - share) / 2])
            rows += [(f"A{well + 1:02d}", gene, f"{gene}_g{guide}", phase) for phase in phases]
    frame = pd.DataFrame(rows, columns=["Metadata_Well", "Metadata_Perturbation", "Metadata_sgRNA", "phase"])
    frame["is_ntc"] = frame["Metadata_Perturbation"].eq("NTC")
    frame["Metadata_ImageID"], frame["Metadata_ObjectType"] = "img", "Cells"
    frame["Metadata_ObjectNumber"] = np.arange(len(frame))
    frame.index = frame.index.astype(str)
    return _object_from_obs(frame)


def _object_from_obs(frame):
    adata = ad.AnnData(X=np.zeros((len(frame), 1), dtype=np.float32), obs=frame)
    stamp(adata, resolution="object")
    return adata


def _composition(adata, **kwargs):
    mt.tl.composition_shift(adata, **{"key": "phase", "reference": "is_ntc", "n_permutations": 199, **kwargs})
    return adata.uns["mantispy"]["composition_shift"].set_index("group")


def test_composition_shift_finds_a_g1_gain_with_three_wells_and_the_guide_as_unit():
    moved = _composition(_phase_object(3, ko_g1=0.7))
    assert moved.loc["KO", "G1"] == pytest.approx(0.3, abs=0.05)
    assert moved.loc["KO", "distance"] == pytest.approx(0.3, abs=0.05)
    assert moved.loc["KO", "pvalue"] < 0.05
    assert "NTC" not in moved.index


def test_composition_shift_null_is_calibrated_when_guide_effects_repeat_across_wells():
    """Relabelling guides separately in each well would call about a third of these; one relabelling must not."""
    calls = [
        _composition(_phase_object(3, ko_g1=0.4, ntc_guides=8, guide_sd=0.08, seed=seed), n_permutations=99).loc[
            "KO", "pvalue"
        ]
        < 0.05
        for seed in range(60)
    ]
    assert np.mean(calls) <= 0.12


def test_composition_shift_reports_the_floor_of_a_small_design():
    """One guide per side has two relabellings, so no p-value can go below 1/2."""
    table = _composition(_phase_object(3, ko_g1=0.9, guides=1, ntc_guides=1))
    assert table.loc["KO", "min_pvalue"] == pytest.approx(0.5)
    assert table.loc["KO", "pvalue"] >= 0.5


def test_composition_shift_scores_each_guide_against_the_reference_guides():
    adata = _phase_object(3, ko_g1=0.7)
    table = _composition(adata, groupby="Metadata_sgRNA", replicate="Metadata_sgRNA")
    assert sorted(table.index) == [f"KO_g{guide}" for guide in range(4)]
    assert (table["G1"] > 0.2).all()


def test_composition_shift_keeps_the_same_well_on_two_plates_apart():
    """A well holding no reference cells on one plate must not borrow the reference of that well on another."""
    first, second = _phase_object(2, ko_g1=0.4), _phase_object(2, ko_g1=0.9, seed=1)
    first.obs["Metadata_Plate"], second.obs["Metadata_Plate"] = "P1", "P2"
    second = second[~second.obs["is_ntc"].to_numpy()].copy()  # plate 2 holds only knockout cells
    second.obs_names = [f"p2_{name}" for name in second.obs_names]
    table = _composition(ad.concat([first, second], merge="same"))
    assert table.loc["KO", "n_blocks"] == 2
    assert table.loc["KO", "G1"] == pytest.approx(0.0, abs=0.05)  # plate 2's G1-rich knockout cells have no pair


def test_composition_shift_warns_when_cells_are_the_unit():
    with pytest.warns(UserWarning, match="independent replicates"):
        _composition(_phase_object(2, ko_g1=0.4), replicate=None, n_permutations=9)


_AREA, _SOLIDITY = "Nuclei_AreaShape_Area", "Nuclei_AreaShape_Solidity"
_G4 = 0.9  # where a measured 4N peak sits, short of the 1 a perfect stain gives


def _ages(n_cells, rng):
    return -np.log2(1 - rng.random(n_cells) / 2)


def _cycle_cells(n_cells=4000, kind="cycle", seed=0):
    """Cells of an exponentially growing population: G1, then S, then G2M, with S filling in DNA linearly.

    The 4N peak sits at ``_G4`` doublings rather than 1, and through S the amount of DNA, not its logarithm, rises
    linearly, so a model that assumes either a 4N peak at 1 or a linear rise on the log scale misplaces cells.
    ``kind`` replaces the DNA content with a population that does not cycle: a narrow line from 2N, a single 2N blob,
    or 2N and 4N clusters.
    """
    rng = np.random.default_rng(seed)
    age = _ages(n_cells, rng)
    progress = np.clip((age - 0.45) / 0.3, 0, 1)
    content = np.log2(1 + (2**_G4 - 1) * progress)
    if kind == "line":
        content = rng.uniform(0, 0.3, n_cells)
    elif kind == "blob":
        content = np.zeros(n_cells)
    elif kind == "two":
        content = (rng.random(n_cells) < 0.4) * _G4
    content = content + rng.normal(0, 0.03, n_cells)
    integrated = 1000 * 2**content
    phase = np.where(age < 0.45, "G1", np.where(age < 0.75, "S", "G2M"))
    obs = pd.DataFrame(
        {
            "Metadata_Plate": "P1",
            "Metadata_Well": [f"W{row % 4}" for row in range(n_cells)],
            "Metadata_Control_Type": "negcon",
            "Metadata_Perturbation": "NTC",
            "age": age,
            "true_phase": phase,
        },
        index=[str(row) for row in range(n_cells)],
    )
    area = 1000 * 2 ** (0.7 * content + rng.normal(0, 0.05, n_cells))
    solidity = np.clip(rng.normal(0.975, 0.004, n_cells), 0, 1)
    X = np.column_stack([integrated, area, solidity]).astype(np.float32)
    cells = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[_DNA, _AREA, _SOLIDITY]))
    mt.tl.dna_content(cells, dna_feature=_DNA)
    return cells


def _clock(cells, **kwargs):
    mt.tl.fit_cell_cycle(cells, qc_features=(_AREA, _SOLIDITY), **kwargs)
    mt.tl.project_cell_cycle(cells)
    return cells


def test_cell_cycle_recovers_the_phases_and_places_cells_with_calibrated_intervals():
    """The model finds the 4N peak where it is, the true phase shares, and S positions to within a few hundredths."""
    cells = _clock(_cycle_cells(n_cells=6000))
    model = cells.uns["mantispy"]["cell_cycle"]
    assert abs(model["peaks"][1] - _G4) < 0.03
    truth = 2 * (1 - 2.0 ** -np.array([0.45, 0.75]))  # the share of a growing population younger than each boundary
    np.testing.assert_allclose(np.cumsum(model["fractions"])[:2], truth, atol=0.02)

    obs, quantiles = cells.obs, cells.obsm["cell_cycle_quantiles"].to_numpy()
    age, position = obs["age"].to_numpy(), obs["cell_cycle_position"].to_numpy()
    in_s = (obs["true_phase"] == "S").to_numpy() & (np.abs(age - 0.6) < 0.12)
    assert np.median(position[in_s] - age[in_s]) == pytest.approx(0, abs=0.006)
    placed = np.isfinite(quantiles).all(axis=1)
    inside90 = (age >= quantiles[:, 0]) & (age <= quantiles[:, 4])
    inside50 = (age >= quantiles[:, 1]) & (age <= quantiles[:, 3])
    assert inside90[placed].mean() > 0.85
    assert 0.35 < inside50[placed].mean() < 0.7

    probabilities = cells.obsm["cell_cycle_probabilities"]
    assert list(probabilities.columns) == ["G1", "S", "G2M", "sub-2N", "above-4N"]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    truth_probability = probabilities.to_numpy()[
        np.arange(cells.n_obs), probabilities.columns.get_indexer(obs["true_phase"])
    ]
    assert truth_probability.mean() > 0.85
    called = obs["cell_cycle_phase"].astype(str)
    assert (called == obs["true_phase"]).mean() > 0.9
    assert (called == probabilities.idxmax(axis=1)).all()


def test_cell_cycle_flags_merged_nuclei_and_debris_against_the_reference(caplog):
    cells = _cycle_cells(n_cells=3000)
    merged, debris = np.arange(10), np.arange(10, 20)
    cells.X[merged, 2] = 0.85
    cells.X[debris, 1] = 40
    _clock(cells)
    flags = cells.obs["cell_cycle_qc"].astype(str).to_numpy()
    assert (flags[merged] == "irregular").all()
    assert (flags[debris] == "small").all()
    assert (flags[20:] == "ok").mean() > 0.99
    assert set(cells.uns["mantispy"]["cell_cycle"]["qc"]) == {"features", "cutoffs"}
    mt.tl.project_cell_cycle(cells, checks=["small"])
    flags = cells.obs["cell_cycle_qc"].astype(str).to_numpy()
    assert (flags[merged] == "ok").all() and (flags[debris] == "small").all()
    with pytest.raises(ValueError, match="checks"):
        mt.tl.project_cell_cycle(cells, checks=["round"])


def test_cell_cycle_projects_with_the_layer_it_was_fitted_on():
    cells = _cycle_cells(n_cells=2000)
    cells.layers["raw"] = cells.X.copy()
    cells.X = cells.X / cells.X.max(axis=0)  # a normalized X, on another scale than the raw cutoffs
    mt.tl.fit_cell_cycle(cells, qc_features=(_AREA, _SOLIDITY), layer="raw")
    mt.tl.project_cell_cycle(cells)
    assert (cells.obs["cell_cycle_qc"] == "ok").mean() > 0.99


def test_cell_cycle_shape_check_survives_a_coarsely_rounded_feature():
    """Solidity at two decimals leaves most nuclei on one value, so the deviation is zero; merged nuclei still fail."""
    cells = _cycle_cells(n_cells=3000)
    cells.X[:, 2] = np.round(cells.X[:, 2], 2)
    cells.X[:10, 2] = 0.85
    _clock(cells)
    assert (cells.obs["cell_cycle_qc"].iloc[:10] == "irregular").all()


def test_draw_positions_spreads_the_outer_tenth_over_the_tails():
    from mantispy.pl._heterogeneity import _draw_positions

    quantiles = np.tile([0.3, 0.4, 0.5, 0.6, 0.7], (20000, 1))
    drawn = _draw_positions(quantiles, np.array([0.05, 0.25, 0.5, 0.75, 0.95]), seed=1)
    low = drawn[drawn < 0.3]
    assert 0.04 < low.size / drawn.size < 0.06
    assert np.unique(low.round(4)).size > 100
    assert drawn.min() >= 0.25 and drawn.max() <= 0.75


def _stained(shift, well="W3", **kwargs):
    """Cycling cells whose ``well`` reads its DNA stain ``shift`` doublings off, set against the plate's controls."""
    cells = _cycle_cells(**kwargs)
    off = (cells.obs["Metadata_Well"] == well).to_numpy()
    cells.X[off, 0] *= 2**shift
    mt.tl.dna_content(cells, dna_feature=_DNA, by="Metadata_Plate", reference="negcon")
    return cells


def test_cell_cycle_warns_about_a_well_whose_peaks_both_sit_off_the_reference(caplog):
    cells = _stained(-0.3)
    content = cells.obs["Metadata_DNAContent"].copy()
    with caplog.at_level(logging.INFO, logger="scverse.mantispy"):
        _clock(cells)
    wells = cells.uns["mantispy"]["cell_cycle_wells"].set_index("Metadata_Well")
    assert -0.4 < wells.loc["W3", "shift"] < -0.2
    assert (wells.drop(index="W3")["shift"].abs() < cells.uns["mantispy"]["cell_cycle"]["sd"]).all()
    assert (wells["n_cells"] == 1000).all()
    warned = [
        record.deep
        for record in caplog.records
        if record.levelno == logging.WARNING and "peaks off the reference" in record.getMessage()
    ]
    assert len(warned) == 1 and "W3" in warned[0] and "W0" not in warned[0]
    pd.testing.assert_series_equal(cells.obs["Metadata_DNAContent"], content)


@pytest.mark.parametrize("shift", [-0.5, 0.5])
def test_cell_cycle_finds_the_shift_of_a_well_whose_stain_reads_far_off(shift):
    cells = _clock(_stained(shift))
    found = cells.uns["mantispy"]["cell_cycle_wells"].set_index("Metadata_Well").loc["W3", "shift"]
    assert found == pytest.approx(shift, abs=0.05)


def test_cell_cycle_leaves_a_well_unchecked_without_both_peaks(caplog):
    """An arrested well has one peak, so a shifted stain and a shifted population look alike; it is not judged."""
    cells = _stained(0.0)
    arrested = ((cells.obs["Metadata_Well"] == "W3") & (cells.obs["true_phase"] != "G2M")).to_numpy()
    cells = cells[~arrested].copy()
    in_w2 = (cells.obs["Metadata_Well"] == "W2").to_numpy()
    cells = cells[~(in_w2 & (np.cumsum(in_w2) > 30))].copy()
    with caplog.at_level(logging.DEBUG, logger="scverse.mantispy"):
        _clock(cells)
    shift = cells.uns["mantispy"]["cell_cycle_wells"].set_index("Metadata_Well")["shift"]
    assert np.isnan(shift["W3"]) and np.isnan(shift["W2"])
    assert shift[["W0", "W1"]].abs().max() < cells.uns["mantispy"]["cell_cycle"]["sd"]
    assert not any("peaks off the reference" in record.getMessage() for record in caplog.records)
    (unchecked,) = [record.deep for record in caplog.records if "could not be checked" in record.getMessage()]
    assert "too few cells" in unchecked and "no 2N peak" in unchecked
    mt.tl.project_cell_cycle(cells, by=None)
    assert cells.uns["mantispy"]["cell_cycle_wells"]["shift"].isna().all()
    with pytest.raises(KeyError, match="Metadata_Missing"):
        mt.tl.project_cell_cycle(cells, by="Metadata_Missing")
    cells.obs["Metadata_Well"] = cells.obs["Metadata_Well"].astype(object)
    cells.obs.iloc[0, cells.obs.columns.get_loc("Metadata_Well")] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        mt.tl.project_cell_cycle(cells)


@pytest.mark.parametrize("kind", ["line", "blob", "two"])
def test_cell_cycle_gate_fails_without_a_cycling_population(kind, caplog):
    """2N and 4N clusters without S cells between them are not a cycle either."""
    cells = _cycle_cells(kind=kind)
    with caplog.at_level(logging.INFO, logger="scverse.mantispy"):
        _clock(cells)
    model = cells.uns["mantispy"]["cell_cycle"]
    assert not model["passed"] and "edges" not in model
    assert {"is_peak", "peak_ratio", "valley_ratio", "s_floor", "s_share"} <= set(model["check"])
    assert cells.obs["cell_cycle_position"].isna().all()
    assert cells.obs["cell_cycle_phase"].isna().all()
    assert cells.obsm["cell_cycle_probabilities"].isna().all().all()
    assert any("failed its check" in record.getMessage() for record in caplog.records)


def test_cell_cycle_needs_dna_content_and_a_reference():
    cells = _cycle_cells(n_cells=500)
    with pytest.raises(KeyError, match="dna_content"):
        mt.tl.fit_cell_cycle(cells, key="Metadata_Missing")
    cells.obs["one"] = np.arange(cells.n_obs) == 0
    with pytest.raises(ValueError, match="at least"):
        mt.tl.fit_cell_cycle(cells, reference="one")
    cells.obs["none"] = False
    with pytest.raises(ValueError, match="at least"):
        mt.tl.fit_cell_cycle(cells, reference="none")
    constant = cells.copy()
    constant.obs["Metadata_DNAContent"] = 0.0
    mt.tl.fit_cell_cycle(constant)
    assert not constant.uns["mantispy"]["cell_cycle"]["passed"]
    with pytest.raises(KeyError, match="lacks"):
        mt.tl.fit_cell_cycle(cells, qc_features=(_AREA, "Nuclei_AreaShape_Missing"))
    with pytest.raises(KeyError, match="fit_cell_cycle"):
        mt.tl.project_cell_cycle(cells)


def test_cell_cycle_leaves_cells_without_dna_content_unplaced():
    cells = _cycle_cells()
    cells.obs.loc[cells.obs_names[:5], "Metadata_DNAContent"] = np.nan
    _clock(cells)
    assert cells.obs["cell_cycle_position"].iloc[:5].isna().all()
    assert cells.obs["cell_cycle_phase"].iloc[:5].isna().all()
    assert cells.obs["cell_cycle_position"].iloc[5:].notna().mean() > 0.95


def test_cell_cycle_projects_perturbed_cells_through_the_reference_model():
    """Cells arrested in G2M are placed on the reference's cycle and do not change the model fitted on it."""
    control = _cycle_cells(seed=1)
    arrested = _cycle_cells(seed=2)
    arrested = arrested[(arrested.obs["true_phase"] == "G2M").to_numpy()].copy()
    arrested.obs["Metadata_Control_Type"], arrested.obs["Metadata_Perturbation"] = "trt", "KO"
    arrested.obs_names = [f"ko{name}" for name in arrested.obs_names]
    cells = ad.concat([control, arrested], merge="same")
    _clock(cells)
    alone = _clock(control.copy())
    for name in ("sd", "off_cycle", "fractions", "peaks"):
        np.testing.assert_allclose(cells.uns["mantispy"]["cell_cycle"][name], alone.uns["mantispy"]["cell_cycle"][name])
    ko = (cells.obs["Metadata_Perturbation"] == "KO").to_numpy()
    assert (cells.obs.loc[ko, "cell_cycle_phase"] == "G2M").mean() > 0.9
    assert cells.obs.loc[ko, "cell_cycle_position"].median() > cells.obs.loc[~ko, "cell_cycle_position"].quantile(0.75)


def test_cell_cycle_model_round_trips_through_h5ad_and_projects_another_object(tmp_path):
    cells = _clock(_cycle_cells(n_cells=1000))
    path = tmp_path / "cycle.h5ad"
    cells.write_h5ad(path)
    back = ad.read_h5ad(path)
    mt.tl.project_cell_cycle(back, key_added="again")
    np.testing.assert_allclose(back.obs["again_position"], cells.obs["cell_cycle_position"])
    other = _cycle_cells(n_cells=1000, seed=3)
    model = back.uns["mantispy"]["cell_cycle"]
    mt.tl.project_cell_cycle(other, model=model, key_added="placed")
    assert other.obs["placed_qc"].notna().all()
    assert other.obs["placed_phase"].notna().all()
    np.testing.assert_allclose(other.uns["mantispy"]["placed"]["edges"], model["edges"])  # what the plots read
    no_wells = other.copy()
    del no_wells.obs["Metadata_Well"]
    mt.tl.project_cell_cycle(no_wells, model=model)  # the default checks whichever well columns exist
    assert list(no_wells.uns["mantispy"]["cell_cycle_wells"].columns[:1]) == ["Metadata_Plate"]
    with pytest.raises(KeyError, match="lacks"):
        mt.tl.project_cell_cycle(other[:, [_DNA]].copy(), model=model)


def test_cell_cycle_calls_cells_below_2n_and_above_4n_off_the_cycle_and_registers_its_keys():
    cells = _cycle_cells(n_cells=3000)
    cells.obs.loc[cells.obs_names[:20], "Metadata_DNAContent"] = -1.5
    cells.obs.loc[cells.obs_names[20:40], "Metadata_DNAContent"] = 2.2
    _clock(cells)
    phase = cells.obs["cell_cycle_phase"].astype(str).to_numpy()
    assert (phase[:20] == "sub-2N").all() and (phase[20:40] == "above-4N").all()
    assert cells.obs["cell_cycle_position"].iloc[:40].isna().all()

    cells.obs["Metadata_ImageID"], cells.obs["Metadata_ObjectType"] = "img", "Nuclei"
    cells.obs["Metadata_ObjectNumber"] = np.arange(cells.n_obs)
    stamp(cells, resolution="object")
    assert not [warning for warning in mt.io.validate(cells).warnings if "cell_cycle" in warning]


def test_cell_cycle_drops_columns_an_earlier_projection_left():
    cells = _clock(_cycle_cells(n_cells=2000))
    cells.obs["cell_cycle_G2M_order"] = 0.5
    mt.tl.fit_cell_cycle(cells)
    mt.tl.project_cell_cycle(cells)
    assert "cell_cycle_qc" not in cells.obs and "cell_cycle_G2M_order" not in cells.obs


def test_dna_content_finds_the_nuclear_dna_feature_on_its_own():
    rng = np.random.default_rng(0)
    names = [
        "Cells_Intensity_IntegratedIntensity_DAPI",
        "Nuclei_Intensity_IntegratedIntensityEdge_DAPI",
        "Nuclei_Intensity_IntegratedIntensity_Hoechst",
    ]
    values = 1000 * 2 ** rng.normal(0, 0.1, (300, 3))
    values[:, 2] *= 7
    cells = ad.AnnData(
        X=values,
        obs=pd.DataFrame({"Metadata_Plate": "P1"}, index=[str(row) for row in range(300)]),
        var=pd.DataFrame({"channel": ["DAPI", "DAPI", "Hoechst"]}, index=names),
    )
    mt.tl.dna_content(cells, by="Metadata_Plate")
    peak = cells.uns["mantispy"]["dna_content"]["g1_peak"].iloc[0]
    assert peak == pytest.approx(np.log2(7000), abs=0.1)
    for names in (["Intensity_IntegratedIntensity__DNA"], ["Cells_Intensity_IntegratedIntensity_DNA"]):
        single = ad.AnnData(X=values[:, 2:], obs=cells.obs[["Metadata_Plate"]], var=pd.DataFrame(index=names))
        mt.tl.dna_content(single, by="Metadata_Plate")  # no object prefix, or the whole cell only
        assert single.uns["mantispy"]["dna_content"]["g1_peak"].iloc[0] == pytest.approx(peak)


def test_dna_content_names_reference_cells_when_a_group_has_too_few(caplog):
    cells = _cycle_cells(n_cells=400)
    cells.obs["Metadata_Control_Type"] = np.where(cells.obs["Metadata_Well"] == "W0", "negcon", "trt")
    with caplog.at_level(logging.INFO, logger="scverse.mantispy"):
        mt.tl.dna_content(cells, dna_feature=_DNA, reference="negcon")
    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING and "too few" in r.getMessage()]
    assert "'negcon' reference cells" in record.getMessage() and "by='Metadata_Plate'" in record.deep


def _progress_cells(n_cells=3000, seed=0):
    """Cycling cells plus a tubulin feature that rises with age through the whole cycle, G2M included."""
    cells = _cycle_cells(n_cells=n_cells, seed=seed)
    tubulin = cells.obs["age"].to_numpy() + np.random.default_rng(seed + 10).normal(0, 0.03, cells.n_obs)
    X = np.column_stack([cells.X, 100 * (1 + tubulin)]).astype(np.float32)
    return ad.AnnData(X=X, obs=cells.obs.copy(), var=pd.DataFrame(index=[*cells.var_names, _TUBULIN]))


_TUBULIN = "Cells_Intensity_MeanIntensity_Tubulin"


@pytest.mark.filterwarnings("ignore:phase_order is experimental")
def test_phase_order_follows_progress_within_g2m_and_leaves_the_rest():
    cells = _clock(_progress_cells())
    position = cells.obs["cell_cycle_position"].copy()
    mt.tl.phase_order(cells)
    entry = cells.uns["mantispy"]["cell_cycle_G2M_order"]
    assert entry["phase"] == "G2M" and entry["features"] == [_TUBULIN]  # no DNA stain and no nuclear shape
    pd.testing.assert_series_equal(cells.obs["cell_cycle_position"], position)

    order = cells.obs["cell_cycle_G2M_order"].to_numpy()
    g2m = (cells.obs["cell_cycle_phase"] == "G2M").to_numpy()
    assert np.isfinite(order[g2m]).all() and np.isnan(order[~g2m]).all()
    assert 0 < order[g2m].min() and order[g2m].max() < 1
    age = cells.obs["age"].to_numpy()
    assert pd.Series(order[g2m]).corr(pd.Series(age[g2m]), method="spearman") > 0.5

    mt.tl.phase_order(cells, phase="S", features=[_AREA, _TUBULIN], key_added="by_hand")
    assert cells.uns["mantispy"]["by_hand"]["features"] == [_AREA, _TUBULIN]
    with pytest.raises(ValueError, match="phase must be one of"):
        mt.tl.phase_order(cells, phase="M")


@pytest.mark.filterwarnings("ignore:phase_order is experimental")
def test_phase_order_of_a_cell_depends_on_the_reference_not_on_the_cells_beside_it(tmp_path):
    """A late arrest placed alone keeps its late order, since every cell is placed against the reference cells."""
    cells = _clock(_progress_cells())
    mt.tl.phase_order(cells)
    path = tmp_path / "order.h5ad"
    cells.write_h5ad(path)
    anchor = ad.read_h5ad(path).uns["mantispy"]["cell_cycle_G2M_order"]

    g2m = (cells.obs["cell_cycle_phase"] == "G2M").to_numpy()
    late = g2m & (cells.obs["age"].to_numpy() > np.quantile(cells.obs["age"].to_numpy()[g2m], 0.75))
    alone = cells[late].copy()
    mt.tl.phase_order(alone, model=cells.uns["mantispy"]["cell_cycle"], anchor=anchor)
    np.testing.assert_allclose(alone.obs["cell_cycle_G2M_order"], cells.obs.loc[late, "cell_cycle_G2M_order"])
    assert alone.obs["cell_cycle_G2M_order"].median() > 0.6

    no_g2m = cells[(cells.obs["true_phase"] == "G1").to_numpy()].copy()
    mt.tl.phase_order(no_g2m, model=cells.uns["mantispy"]["cell_cycle"], anchor=anchor)
    assert no_g2m.obs["cell_cycle_G2M_order"].isna().all()
    with pytest.raises(ValueError, match="anchor orders G2M"):
        mt.tl.phase_order(no_g2m, phase="G1", model=cells.uns["mantispy"]["cell_cycle"], anchor=anchor)
    with pytest.raises(ValueError, match="at least 50"):
        mt.tl.phase_order(cells[:120].copy(), model=cells.uns["mantispy"]["cell_cycle"])
