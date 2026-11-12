"""Plots for cell-state composition, cell cycle and subpopulation effects."""

import matplotlib
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import mantispy as mt


@pytest.fixture(scope="module")
def clustered():
    cells = mt.ds.synthetic_plate(n_wells=48, n_cells=30, n_features=20, n_perturbations=3, effect_size=4.0, seed=0)
    mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon", keep_raw=True)
    sc.pp.pca(cells, n_comps=10)
    sc.pp.neighbors(cells)
    sc.tl.leiden(cells, key_added="leiden", flavor="igraph", n_iterations=2)
    cells.obs["state"] = np.where(np.random.default_rng(0).random(cells.n_obs) < 0.5, "a", "b")
    mt.tl.subpopulation_hits(cells, cluster_key="state")
    return cells


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_composition_and_subpopulation_plots_draw(clustered):
    composition = mt.tl.cluster_composition(clustered)
    _, composition_ax = plt.subplots()
    assert isinstance(mt.pl.cluster_composition(composition, ax=composition_ax), matplotlib.axes.Axes)
    _, hits_ax = plt.subplots()
    assert isinstance(mt.pl.subpopulation_hits(clustered, ax=hits_ax), matplotlib.axes.Axes)


def test_feature_distributions_gains_a_ridge_kind(clustered):
    # feature_distributions has no ax parameter and always owns its figure, so inspect the drawn panels.
    assert mt.pl.feature_distributions(clustered, features=list(clustered.var_names[:2]), kind="ridge") is None
    panels = plt.gcf().axes
    assert panels[0].collections  # filled density curves; a histogram would draw patches instead
    assert len(panels) >= 2
    with pytest.raises(ValueError, match="'ecdf', 'hist' or 'ridge'"):
        mt.pl.feature_distributions(clustered, features=[clustered.var_names[0]], kind="violin")


def test_an_empty_subpopulation_table_is_refused_rather_than_drawn(clustered):
    empty = clustered.copy()
    mt.tl.subpopulation_hits(empty, cluster_key="leiden")
    empty.uns["mantispy"]["subpopulation_hits"] = empty.uns["mantispy"]["subpopulation_hits"].iloc[:0]
    with pytest.raises(ValueError, match="is empty"):
        mt.pl.subpopulation_hits(empty)


def test_density_plot_draws_and_says_what_to_run_first(clustered):
    rng = np.random.default_rng(0)
    clustered.obs["Metadata_Center_X"] = rng.uniform(0, 1000, clustered.n_obs)
    clustered.obs["Metadata_Center_Y"] = rng.uniform(0, 1000, clustered.n_obs)

    with pytest.raises(KeyError, match="mt.tl.neighbors_local_density"):
        mt.pl.density(clustered, feature=clustered.var_names[0])

    mt.tl.neighbors_local_density(clustered, k=5)
    _, ax = plt.subplots()
    assert isinstance(mt.pl.density(clustered, feature=clustered.var_names[0], ax=ax), matplotlib.axes.Axes)


@pytest.fixture(scope="module")
def placed():
    from test_tl_heterogeneity import _clock, _cycle_cells

    cells = _cycle_cells(n_cells=2000)
    cells.obs["Metadata_Perturbation"] = np.where(np.arange(cells.n_obs) % 2, "NTC", "KO")
    return _clock(cells)


def test_cell_cycle_plot_draws_a_feature_on_a_log_scale_with_the_projected_phases_by_default(placed):
    # cell_cycle has no ax parameter and always owns its figure, so it draws and returns None.
    assert (
        mt.pl.cell_cycle(placed, dna_feature="Nuclei_Intensity_IntegratedIntensity_DNA", by="Metadata_Perturbation")
        is None
    )
    panels = plt.gcf().axes
    assert len(panels) == 2 and panels[0].get_xlabel() == "log DNA intensity"


def test_cell_cycle_plot_colours_the_projected_phases(placed):
    assert mt.pl.cell_cycle(placed, dna_feature="Metadata_DNAContent", key="cell_cycle_phase", by=None) is None
    (panel,) = plt.gcf().axes
    assert {"G1", "S", "G2M"} <= {text.get_text() for text in panel.get_legend().get_texts()}
    assert panel.get_xlabel() == "Metadata_DNAContent"
    mt.pl.cell_cycle(placed, dna_feature="Metadata_DNAContent", key="cell_cycle_phase", by="Metadata_Well", ncols=3)
    axes = plt.gcf().axes
    visible = [axis for axis in axes if axis.get_visible()]
    assert len(axes) == 6 and len(visible) == 4
    assert all(any(label.get_visible() for label in axis.get_xticklabels()) for axis in visible)
    placed.obs["Nuclei_Intensity_IntegratedIntensity_DNA"] = 1.0
    with pytest.raises(ValueError, match="both an obs column and a feature"):
        mt.pl.cell_cycle(placed, dna_feature="Nuclei_Intensity_IntegratedIntensity_DNA", key="cell_cycle_phase")
    del placed.obs["Nuclei_Intensity_IntegratedIntensity_DNA"]


def test_cell_cycle_occupancy_draws_one_curve_per_group_and_marks_s(placed):
    assert mt.pl.cell_cycle_occupancy(placed, by="Metadata_Perturbation", highlight="NTC") is None
    (panel,) = plt.gcf().axes
    assert sorted(text.get_text() for text in panel.get_legend().get_texts()) == ["KO", "NTC"]
    assert len(panel.get_lines()) == 2
    _, ax = plt.subplots()
    assert mt.pl.cell_cycle_occupancy(placed, by=None, ax=ax) is ax
    with pytest.raises(KeyError, match="project_cell_cycle"):
        mt.pl.cell_cycle_occupancy(placed, key="missing")


def test_cell_cycle_clock_returns_axes_and_colours_by_phase(placed):
    _, ax = plt.subplots()
    assert mt.pl.cell_cycle_clock(placed, ax=ax) is ax
    assert {"G1", "S", "G2M"} <= {text.get_text() for text in ax.get_legend().get_texts()}
    assert ax.get_aspect() == 1.0  # a ring, drawn with equal aspect
    _, cyclic = plt.subplots()
    mt.pl.cell_cycle_clock(placed, color="position", ax=cyclic)
    x, y = cyclic.collections[-1].get_offsets().T
    turn = (np.arctan2(x, y) % (2 * np.pi)) / (2 * np.pi)
    assert abs(np.corrcoef(np.hypot(x, y), turn)[0, 1]) < 0.1  # the ring jitter does not follow the position


def test_cell_cycle_clock_colours_continuous_and_the_cyclic_position(placed):
    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(placed, color="Metadata_DNAContent", ax=ax)
    assert ax.get_legend() is None
    assert ax.collections[-1].get_array() is not None  # a colorbar mappable, not a categorical legend
    _, cyclic = plt.subplots()
    mt.pl.cell_cycle_clock(placed, color="position", ax=cyclic)
    assert cyclic.collections[-1].get_cmap().name == "twilight"


def test_cell_cycle_clock_drops_cells_without_a_position(placed):
    cells = placed.copy()
    cells.obsm["cell_cycle_quantiles"].iloc[:100] = np.nan
    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(cells, color="Metadata_DNAContent", ax=ax)
    drawn = ax.collections[-1].get_offsets().shape[0]
    assert drawn == int(np.isfinite(cells.obsm["cell_cycle_quantiles"].to_numpy()).all(axis=1).sum())


def test_cell_cycle_clock_groupby_and_radius(placed):
    cells = placed.copy()
    cells.obs["score"] = np.where(cells.obs["Metadata_Perturbation"] == "KO", 10.0, 100.0)
    assert mt.pl.cell_cycle_clock(cells, color="score", groupby="Metadata_Perturbation") is None
    clocks = [axis for axis in plt.gcf().axes if axis.get_title() in ("KO", "NTC")]
    assert len(clocks) == 2
    assert {clock.collections[-1].get_clim() for clock in clocks} == {(10.0, 100.0)}  # one scale over both panels
    g1_only = cells.obs["cell_cycle_phase"].eq("G1").to_numpy() | (cells.obs["Metadata_Perturbation"] == "NTC")
    mt.pl.cell_cycle_clock(cells[g1_only].copy(), groupby="Metadata_Perturbation")
    (legend,) = plt.gcf().legends  # one legend beside the grid, not inside a panel
    assert {"G1", "S", "G2M"} <= {text.get_text() for text in legend.get_texts()}  # though the first panel is G1 only
    with pytest.raises(ValueError, match="single clock"):
        mt.pl.cell_cycle_clock(cells, groupby="Metadata_Perturbation", ax=plt.subplots()[1])
    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(placed, color="Metadata_DNAContent", radius="Metadata_DNAContent", ax=ax)
    assert ax.get_title() == "radius: Metadata_DNAContent"


def test_cell_cycle_clock_highlights_only_the_named_groups(placed):
    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(placed, color="Metadata_Perturbation", groups="KO", ax=ax)
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["KO"]
    greys = [c for c in ax.collections if np.allclose(c.get_facecolor()[0][:3], to_rgba("lightgray")[:3])]
    assert len(greys) == 1  # the background ring of all other cells, in na_color
    with pytest.raises(ValueError, match="not categories"):
        mt.pl.cell_cycle_clock(placed, color="Metadata_Perturbation", groups=["missing"], ax=ax)


def test_cell_cycle_clock_colours_categories_apart_and_takes_scatter_options(placed):
    cells = placed.copy()
    guides = np.array([f"c{index:02d}" for index in range(12)])[np.arange(cells.n_obs) % 12]
    cells.obs["guide"] = pd.Categorical(np.where(np.arange(cells.n_obs) % 50 == 0, None, guides))
    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(cells, color="guide", groups=["c00", "c10"], ax=ax, alpha=0.3, marker="x")
    handles = ax.get_legend().legend_handles
    assert [handle.get_label() for handle in handles] == ["c00", "c10"]
    colours = {tuple(handle.get_color()) for handle in handles}
    phase_colours = {tuple(to_rgba(colour)[:3]) for colour in ("tab:blue", "tab:red", "tab:grey", "tab:purple")}
    assert len(colours) == 2 and not {colour[:3] for colour in colours} & phase_colours
    assert ax.collections[-1].get_alpha() == 0.3
    with pytest.raises(ValueError, match="not categories"):
        mt.pl.cell_cycle_clock(cells, color="guide", groups=["nan"], ax=ax)
    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(cells, color="Metadata_DNAContent", ax=ax, vmin=0, vmax=1)
    assert ax.collections[-1].get_clim() == (0, 1)


def test_cell_cycle_clock_raises_on_missing_key_and_bad_colour(placed):
    with pytest.raises(KeyError, match="project_cell_cycle"):
        mt.pl.cell_cycle_clock(placed, key="missing")
    with pytest.raises(KeyError, match="neither an obs column nor a feature"):
        mt.pl.cell_cycle_clock(placed, color="nope")
    with pytest.raises(ValueError, match="needs a categorical"):
        mt.pl.cell_cycle_clock(placed, color="Metadata_DNAContent", groups=["x"])
    placed.obs["Nuclei_Intensity_IntegratedIntensity_DNA"] = 1.0
    with pytest.raises(ValueError, match="both an obs column and a feature"):
        mt.pl.cell_cycle_clock(placed, color="Nuclei_Intensity_IntegratedIntensity_DNA")
    del placed.obs["Nuclei_Intensity_IntegratedIntensity_DNA"]


def test_cell_cycle_clock_places_g2m_cells_by_the_refined_order(placed):
    cells = placed.copy()
    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(cells, color="position", ax=ax)
    assert "G2M\npopulation prior" in [text.get_text() for text in ax.texts]

    g2m = (cells.obs["cell_cycle_phase"] == "G2M").to_numpy()
    cells.obs["cell_cycle_G2M_order"] = 0.0  # a stale order on every cell; only the cells called G2M take it
    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(cells, color="position", ax=ax)
    texts = [text.get_text() for text in ax.texts]
    assert "G2M\nexperimental order" in texts and "G1\npopulation prior" in texts
    x, y = ax.collections[-1].get_offsets().T
    turn = (np.arctan2(x, y) % (2 * np.pi)) / (2 * np.pi)
    start = cells.uns["mantispy"]["cell_cycle"]["edges"][2]
    assert np.isclose(turn, start, atol=1e-6).sum() == g2m.sum()  # every G2M cell at order 0, the start of G2M


def test_cell_cycle_clock_takes_scanpys_size_alpha_palette_and_scale(placed):
    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(placed, ax=ax)
    assert np.allclose(ax.collections[-1].get_sizes(), 120_000 / placed.n_obs)  # scanpy's default
    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(placed, palette={"G1": "black"}, alpha=0.2, ax=ax)
    handles = {handle.get_label(): handle.get_color() for handle in ax.get_legend().legend_handles}
    assert handles["G1"] == "black" and handles["S"] == "tab:grey" and ax.collections[-1].get_alpha() == 0.2
    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(placed, color="Metadata_Perturbation", palette=["tab:orange", "tab:green"], ax=ax)
    assert [handle.get_color() for handle in ax.get_legend().legend_handles] == ["tab:orange", "tab:green"]
    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(placed, color="Metadata_DNAContent", vmin="p50", vmax=1.5, ax=ax)
    assert ax.collections[-1].get_clim() == (np.median(placed.obs["Metadata_DNAContent"][_drawn(placed)]), 1.5)
    with pytest.raises(ValueError, match="percentile"):
        mt.pl.cell_cycle_clock(placed, color="Metadata_DNAContent", vmin="half")


def _drawn(cells):
    return np.isfinite(cells.obsm["cell_cycle_quantiles"].to_numpy()).all(axis=1)


def test_cell_cycle_clock_width_spreads_the_band_and_na_color_keeps_cells_without_a_value(placed):
    radii = {}
    for width in (0.1, 0.5):
        _, ax = plt.subplots()
        mt.pl.cell_cycle_clock(placed, width=width, ax=ax)
        x, y = ax.collections[-1].get_offsets().T
        radii[width] = np.hypot(x, y)
    assert np.ptp(radii[0.1]) <= 0.1 + 1e-9 < np.ptp(radii[0.5]) <= 0.5 + 1e-9
    cells = placed.copy()
    cells.obs["score"] = np.where(np.arange(cells.n_obs) % 4 == 0, np.nan, 1.0)
    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(cells, color="score", na_color="red", ax=ax)
    missing = [c for c in ax.collections if np.allclose(c.get_facecolor()[0][:3], to_rgba("red")[:3])]
    assert len(missing) == 1
    assert missing[0].get_offsets().shape[0] == int((_drawn(cells) & cells.obs["score"].isna()).sum())


def test_cell_cycle_clock_legend_options_and_frame(placed):
    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(placed, legend_loc="on data", legend_fontsize=9, ax=ax)
    assert ax.get_legend() is None
    labels = {text.get_text(): text.get_fontsize() for text in ax.texts}
    assert labels["G1"] == 9 and labels["G2M"] == 9
    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(placed, legend_loc=None, frameon=True, ax=ax)
    assert ax.get_legend() is None and ax.axison
    with pytest.raises(ValueError, match="legend_loc"):
        mt.pl.cell_cycle_clock(placed, legend_loc="upper left")
    _, ax = plt.subplots()
    mt.pl.cell_cycle_occupancy(placed, palette=["tab:orange", "tab:green"], legend_loc=None, ax=ax)
    assert ax.get_legend() is None
    assert {line.get_edgecolor()[:3] for line in ax.patches} == {to_rgba("tab:orange")[:3], to_rgba("tab:green")[:3]}


def test_cell_cycle_clock_draws_densities_as_rings_or_rows_in_the_order_of_groups(placed):
    _, ax = plt.subplots()
    assert (
        mt.pl.cell_cycle_clock(placed, kind="rings", color="Metadata_Perturbation", groups=["NTC", "KO"], ax=ax) is ax
    )
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["NTC", "KO"]
    from matplotlib.patches import Polygon

    rings = [patch for patch in ax.patches if isinstance(patch, Polygon)]
    reach = [np.hypot(*ring.get_xy().T).min() for ring in rings]
    colours = [handle.get_facecolor() for handle in ax.get_legend().legend_handles]
    assert len(rings) == 2 and reach[0] > reach[1]  # the first group is the outer ring
    assert [ring.get_facecolor() for ring in rings] == colours

    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(placed, kind="ridge", color="Metadata_Perturbation", groups=["KO", "NTC"], ax=ax)
    rows = sorted(zip(ax.get_yticks(), [label.get_text() for label in ax.get_yticklabels()], strict=True))
    assert [name for _, name in rows] == ["NTC", "KO"]  # the first group on top

    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(placed, kind="ridge", ax=ax)
    fills = {tuple(to_rgba(c.get_facecolor()[0])[:3]) for c in ax.collections if len(c.get_facecolor())}
    assert {to_rgba(PHASE)[:3] for PHASE in ("tab:blue", "tab:grey", "tab:red")} <= fills  # one curve, filled by phase

    with pytest.raises(ValueError, match="categorical"):
        mt.pl.cell_cycle_clock(placed, kind="rings", color="Metadata_DNAContent")
    with pytest.raises(ValueError, match="kind"):
        mt.pl.cell_cycle_clock(placed, kind="violin")


def test_circular_density_wraps_across_division_and_reads_one_for_even_cells():
    from mantispy.pl._heterogeneity import _circular_density

    even = _circular_density(np.random.default_rng(0).random(200_000), 0.02)
    assert np.allclose(even, 1, atol=0.05)
    late = _circular_density(np.full(100, 0.995), 0.02)
    assert np.isclose(late[1], late[-5])  # as high just after division as just before


def test_cell_cycle_clock_compares_densities_to_a_reference(placed):
    _, ax = plt.subplots()
    mt.pl.cell_cycle_clock(placed, kind="ridge", color="Metadata_Perturbation", groups=["KO"], reference="NTC", ax=ax)
    rows = sorted(zip(ax.get_yticks(), [label.get_text() for label in ax.get_yticklabels()], strict=True))
    assert [name for _, name in rows] == ["KO", "NTC"]  # the reference joins, drawn first
    dashed = [line for line in ax.get_lines() if line.get_linestyle() == "--"]
    assert np.ptp(dashed[0].get_ydata()) > 0  # the reference's density, not the flat even line

    mt.pl.cell_cycle_clock(placed, kind="rings", groupby="Metadata_Perturbation", reference="NTC")
    assert [axis.get_title() for axis in plt.gcf().axes] == ["NTC", "KO"]  # the reference's panel first
    with pytest.raises(ValueError, match="not a category"):
        mt.pl.cell_cycle_clock(placed, kind="rings", color="Metadata_Perturbation", reference="missing")


def test_cell_cycle_clock_warns_that_many_rings_are_hard_to_read(placed):
    cells = placed.copy()
    cells.obs["guide"] = np.array([f"g{index:02d}" for index in range(10)])[np.arange(cells.n_obs) % 10]
    with pytest.warns(UserWarning, match="kind='ridge'"):
        mt.pl.cell_cycle_clock(cells, kind="rings", color="guide", ax=plt.subplots()[1])


def test_cell_cycle_clock_density_honours_groups_when_color_is_none(placed):
    # With color=None the single density is drawn over the groups' cells, not every placed cell (finding 2).
    def curve(groups):
        _, ax = plt.subplots()
        mt.pl.cell_cycle_clock(placed, kind="ridge", groups=groups, ax=ax)
        return np.concatenate([line.get_ydata() for line in ax.get_lines() if line.get_linestyle() != "--"])

    assert not np.array_equal(curve(["G1"]), curve(None))


def test_cell_cycle_clock_reference_does_not_reorder_colour_panels(placed):
    # A colour reference is a ring, not a panel, so a groupby key named like it must not be promoted first (finding 3).
    cells = placed.copy()
    cells.obs["batch"] = np.where((np.arange(cells.n_obs) // 3) % 2 == 0, "AAA", "NTC")
    mt.pl.cell_cycle_clock(cells, kind="rings", color="Metadata_Perturbation", groupby="batch", reference="NTC")
    assert [axis.get_title() for axis in plt.gcf().axes if axis.get_title()] == ["AAA", "NTC"]


def test_cell_cycle_clock_rejects_a_nonpositive_bandwidth(placed):
    with pytest.raises(ValueError, match="bandwidth must be > 0"):
        mt.pl.cell_cycle_clock(placed, kind="rings", color="Metadata_Perturbation", bandwidth=0)


def test_palette_counts_falsy_entries_so_a_finite_scheme_is_not_overrun():
    # A category present in the palette with a falsy value still consumes a default colour (finding 4).
    from mantispy.pl._heterogeneity import _palette

    shown = [f"c{index:02d}" for index in range(21)]
    result = _palette(shown, {"c00": None})
    assert len(result) == 21 and all(result.values())
