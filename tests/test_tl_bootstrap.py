"""tl.bootstrap_features: exactness, calibration, ties, missing values, control scopes and the shared layout."""

import itertools
import warnings

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy._core.schema import stamp

matplotlib.use("Agg")


def _median(values, median="midpoint"):
    """The median of ``values``: the lower of the two middle ones, or their mean, for an even count."""
    if median == "midpoint":
        return float(np.median(values))
    return float(np.sort(values)[(len(values) - 1) // 2])


def _combined(up, down, n_cells, method):
    """A gene's two-sided p-value from its guides' one-sided ones, by scipy's combination of ``method``."""
    from scipy import stats

    def one_way(p, q):
        if method == "stouffer":  # each z from the smaller tail, clipped, so a p-value near one keeps its digits
            z = np.where(p <= q, stats.norm.isf(np.maximum(p, 1e-12)), -stats.norm.isf(np.maximum(q, 1e-12)))
            w = np.sqrt(n_cells)
            return stats.norm.sf(np.sum(w * z) / np.sqrt(np.sum(w**2)))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        if method == "order":
            r = len(p) // 2 + 1
            return stats.beta.cdf(np.sort(p)[r - 1], r, len(p) - r + 1)
        return stats.combine_pvalues(p, method).pvalue

    up, down = np.array(up), np.array(down)
    return min(1.0, 2 * min(one_way(up, down), one_way(down, up)))


def _brute_pvalues(adata, median="midpoint", method="median"):
    """Exact p-values by full enumeration of every resample; only for tiny screens.

    Per feature, only the cells that measure it count: the guide null draws one control guide
    (probability proportional to its measured cells), then draws as many of those as the guide has
    measured, with replacement, and takes their median (``median`` of an even count). Under ``"median"`` the gene
    null is the same median over its measured guides of independent such draws, and otherwise the guides' one-sided
    p-values are combined by scipy. Two-sided, a tie counts half, no floor.
    """
    x = np.asarray(adata.X, dtype=float)
    obs = adata.obs
    is_ctrl = (obs["Metadata_Control_Type"] == "negcon").to_numpy()
    sgrna = obs["Metadata_sgRNA"].to_numpy()
    gene = obs["Metadata_Gene"].to_numpy()
    ctrl_guides = [x[(sgrna == c) & is_ctrl] for c in sorted(set(sgrna[is_ctrl]))]
    n_feat = x.shape[1]

    def construct_pmf(m, feat):
        """{value: prob} of the null of a guide measuring m cells, for one feature, and P(nan)."""
        cols = [pool[~np.isnan(pool[:, feat]), feat] for pool in ctrl_guides]
        total = sum(col.size for col in cols)
        if total == 0:
            return {}, 1.0
        pmf = {}
        for col in cols:
            n = col.size
            for idx in itertools.product(range(n), repeat=m):
                value = _median(col[list(idx)], median)
                pmf[value] = pmf.get(value, 0.0) + n / total / n**m
        return pmf, 0.0

    def one_sided(pmf, nan, o):
        if nan >= 1.0 - 1e-12:
            return np.nan
        gt = sum(p for v, p in pmf.items() if v > o)
        eq = sum(p for v, p in pmf.items() if v == o)
        return (gt + 0.5 * eq) / (1.0 - nan)

    def two_sided(pmf, nan, o):
        one = one_sided(pmf, nan, o)
        return 2.0 * min(one, 1.0 - one)

    guide_p, gene_p = {}, {}
    for g in sorted(set(gene[~is_ctrl])):
        constructs = sorted(set(sgrna[(gene == g)]))
        cells = {c: x[sgrna == c] for c in constructs}
        for feat in range(n_feat):
            measured = {c: cells[c][~np.isnan(cells[c][:, feat]), feat] for c in constructs}
            kept = [c for c in constructs if measured[c].size]
            cpmfs = [construct_pmf(measured[c].size, feat) for c in kept]
            for c in constructs:
                guide_p[(g, c, feat)] = np.nan
            for c, (pmf, nan) in zip(kept, cpmfs, strict=True):
                guide_p[(g, c, feat)] = two_sided(pmf, nan, _median(measured[c], median))
            if not kept:
                gene_p[(g, feat)] = np.nan
                continue
            if method != "median":  # a guide has no null only where no control cell measures the feature
                ups = np.array(
                    [
                        one_sided(pmf, nan, _median(measured[c], median))
                        for c, (pmf, nan) in zip(kept, cpmfs, strict=True)
                    ]
                )
                sizes = [measured[c].size for c in kept]
                gene_p[(g, feat)] = np.nan if np.isnan(ups).any() else _combined(ups, 1 - ups, sizes, method)
                continue
            # gene null: enumerate the joint over its measured constructs
            gpmf, gnan = {}, 0.0
            items = [list(pmf.items()) + ([(np.nan, nan)] if nan > 0 else []) for pmf, nan in cpmfs]
            for combo in itertools.product(*items):
                prob = 1.0
                vals = []
                for v, p in combo:
                    prob *= p
                    vals.append(v)
                good = [v for v in vals if not np.isnan(v)]
                if not good:
                    gnan += prob
                else:
                    value = _median(good, median)
                    gpmf[value] = gpmf.get(value, 0.0) + prob
            og = _median([_median(measured[c], median) for c in kept], median)
            gene_p[(g, feat)] = two_sided(gpmf, gnan, og)
    return gene_p, guide_p


def _assert_matches_brute_force(adata, rtol, median="midpoint", method="median"):
    gene_p, guide_p = _brute_pvalues(adata, median, method)
    features = list(adata.var_names)
    table = adata.uns["mantispy"]["bootstrap"]
    for row in table.itertuples():
        # a gene's p-value is within rtol of itself, up to rounding; its guides' are exact
        expected = gene_p[(row.group, features.index(row.feature))]
        np.testing.assert_allclose(row.pvalue, expected, rtol=rtol, atol=1e-9)
    guides = adata.uns["mantispy"]["bootstrap_guides"]
    for row in guides.itertuples():
        expected = guide_p[(row.group, row.replicate, features.index(row.feature))]
        np.testing.assert_allclose(row.pvalue, expected, atol=1e-9)


@pytest.mark.parametrize(
    ("method", "rtol"), [("stouffer", 0.0), ("order", 0.0), ("fisher", 0.0), ("median", 0.0), ("median", 1e-5)]
)
@pytest.mark.parametrize("median", ["lower", "midpoint"])
def test_exact_pvalues_match_brute_force_enumeration(method, median, rtol):
    """On a tiny screen the analytic p-values equal a full enumeration of every resample.

    Under the combination rules the gene's p-value is scipy's combination of the enumerated guides' one-sided ones.
    """
    rng = np.random.default_rng(0)
    # integer values so even-count medians land on exact ties, exercising the half-count rule
    groups = []
    for g in range(3):
        for s in range(1 + g % 2 + 1):
            groups.append((f"G{g}", f"g{g}_{s}", int(rng.integers(2, 5)), 0.0, "a"))
    for s in range(3):
        groups.append(("nontargeting", f"nt{s}", int(rng.integers(3, 5)), 0.0, "a"))
    adata = _cells(groups, rng, n_features=3)
    adata.X = rng.integers(0, 4, size=adata.X.shape).astype(float)
    mt.tl.bootstrap_features(adata, method=method, median=median, rtol=rtol)
    _assert_matches_brute_force(adata, rtol, median, method)


@pytest.mark.parametrize(("method", "rtol"), [("stouffer", 0.0), ("median", 0.0), ("median", 1e-5)])
@pytest.mark.parametrize("median", ["lower", "midpoint"])
@pytest.mark.parametrize(("dtype", "missing"), [(np.float32, 0.0), (np.float32, 0.2), (np.float64, 0.2)])
def test_missing_cells_match_brute_force_enumeration(dtype, missing, median, rtol, method):
    """Missing cells in control and tested guides: the null draws the measured count and skips nan draws.

    A guide measuring nothing leaves its gene, and a null median that is missing counts neither way.
    Values on a grid of halves make even-count midpoints that float32 would round differently.
    """
    rng = np.random.default_rng(1)
    groups = [("G0", "g0_0", 3, 0.0, "a"), ("G0", "g0_1", 4, 0.0, "a"), ("G1", "g1_0", 2, 0.0, "a")]
    groups += [("G1", "g1_1", 3, 0.0, "a"), ("G1", "g1_2", 2, 0.0, "a")]
    groups += [("nontargeting", f"nt{s}", 4, 0.0, "a") for s in range(3)]
    adata = _cells(groups, rng, n_features=3)
    x = rng.integers(0, 7, size=adata.X.shape) * 0.55
    if missing:
        x[rng.random(x.shape) < missing] = np.nan
        x[(adata.obs["Metadata_sgRNA"] == "g1_2").to_numpy(), 1] = np.nan
    adata.X = x.astype(dtype)
    mt.tl.bootstrap_features(adata, method=method, median=median, rtol=rtol)
    _assert_matches_brute_force(adata, rtol, median, method)


@pytest.mark.parametrize("method", ["stouffer", "median"])
@pytest.mark.parametrize("seed", range(4))
def test_lower_guide_medians_match_brute_force_enumeration(seed, method):
    """Guides' lower medians give the p-values of a full enumeration, for odd and even counts of cells and of guides.

    Genes of one to four guides of one to four cells, on a grid of halves for ties at the observed medians, with
    missing cells, a guide that measures nothing in one feature, and a feature no control cell measures.
    """
    rng = np.random.default_rng(seed)
    groups = [(f"G{g}", f"g{g}_{s}", int(rng.integers(1, 5)), 0.0, "a") for g in range(4) for s in range(g + 1)]
    groups += [("nontargeting", f"nt{s}", int(rng.integers(1, 5)), 0.0, "a") for s in range(3)]
    adata = _cells(groups, rng, n_features=4)
    x = rng.integers(0, 5, size=adata.X.shape) * 0.5
    x[rng.random(x.shape) < 0.15] = np.nan
    x[(adata.obs["Metadata_sgRNA"] == "g3_0").to_numpy(), 1] = np.nan
    x[(adata.obs["Metadata_Gene"] == "nontargeting").to_numpy(), 3] = np.nan
    adata.X = x
    mt.tl.bootstrap_features(adata, method=method, rtol=0.0)
    _assert_matches_brute_force(adata, 0.0, "lower", method)


@pytest.mark.parametrize("rtol", [0.0, 1e-5])
def test_nulls_of_a_growing_store_match_brute_force_enumeration(rtol):
    """Each feature's guide nulls are counted and then built in place, in a store that grows with them.

    Integer values in the first feature put null atoms at the observed medians, so the lower side of the
    boundary is swept on its own; the continuous second feature has more atoms and grows the store; guides of
    one size share a null.
    """
    rng = np.random.default_rng(19)
    groups = [("G0", "g0_0", 2, 0.0, "a"), ("G0", "g0_1", 2, 0.0, "a"), ("G1", "g1_0", 3, 0.0, "a")]
    groups += [("G1", "g1_1", 2, 0.0, "a"), ("G2", "g2_0", 1, 0.0, "a"), ("G2", "g2_1", 4, 0.0, "a")]
    groups += [("nontargeting", f"nt{s}", 4, 0.0, "a") for s in range(3)]
    adata = _cells(groups, rng, n_features=2)
    adata.X[:, 0] = rng.integers(0, 3, size=adata.n_obs)
    mt.tl.bootstrap_features(adata, method="median", median="midpoint", rtol=rtol)
    _assert_matches_brute_force(adata, rtol)


def test_gene_pvalues_stay_within_rtol_of_the_exact_ones():
    """The far tails a gene's p-value leaves out move it by at most rtol of itself, from the exact rtol=0 one.

    Control guides of tens of cells give guide nulls of thousands of atoms, so the tails left out hold many of them;
    shifted genes put p-values deep in the tails, where the bound is relative; the features hold ties at the observed
    median (rounded and integer values) and missing cells.
    """
    rng = np.random.default_rng(23)
    groups = [("nontargeting", f"nt{i}", int(rng.integers(30, 60)), 0.0, "a") for i in range(6)]
    for g in range(16):
        shift = [0.0, 0.0, 0.5, 1.5][g % 4]
        groups += [(f"G{g}", f"g{g}_{s}", int(rng.integers(3, 30)), shift, "a") for s in range(2 + 2 * (g % 2))]
    adata = _cells(groups, rng, n_features=3)
    adata.X[:, 1] = np.round(adata.X[:, 1], 1)
    adata.X[:, 2] = np.round(3 * adata.X[:, 2])
    adata.X[rng.random(adata.X.shape) < 0.1] = np.nan
    copy = mt.tl.bootstrap_features(adata, method="median", median="midpoint", rtol=0.0, copy=True)
    exact = copy.uns["mantispy"]["bootstrap"]["pvalue"].to_numpy()
    mt.tl.bootstrap_features(adata, method="median", median="midpoint")
    bounded = adata.uns["mantispy"]["bootstrap"]["pvalue"].to_numpy()
    np.testing.assert_array_equal(np.isnan(exact), np.isnan(bounded))
    assert np.nanmin(exact) < 1e-8
    assert (bounded != exact).any()  # tails were left out
    assert np.nanmax(np.abs(bounded - exact) - 1e-5 * exact) <= 1e-15
    assert adata.uns["mantispy"]["params"]["bootstrap_features"]["rtol"] == 1e-5


def test_a_gene_whose_pvalue_rests_on_the_far_tails_of_its_guide_nulls_stays_within_rtol():
    """Two guides of 30 cells at the top of a control guide of ten values: their nulls' far tails decide the p-value.

    Thirty draws put a median at 9 with a probability under the pruned tail mass, so the guide nulls are built again
    with every tail that matters. The exact p-values are from a rational-arithmetic enumeration of the gene null.
    """
    rng = np.random.default_rng(0)
    groups = [("nontargeting", "nt0", 10, 0.0, "a"), ("G", "g0", 30, 0.0, "a"), ("G", "g1", 30, 0.0, "a")]
    adata = _cells(groups, rng, n_features=2)
    adata.X[:10] = np.arange(10.0)[:, None]
    adata.X[10:] = 9.0
    adata.X[40:, 1] = 8.0  # the gene's median is 8.5, between the two guides
    mt.tl.bootstrap_features(adata, method="median", median="midpoint")
    pvalue = adata.uns["mantispy"]["bootstrap"]["pvalue"].to_numpy()
    exact = np.array([1.3377093339925794e-17, 3.841393581090315e-13])
    assert np.all(np.abs(pvalue - exact) <= 1e-5 * exact + 1e-15)


def _cells(groups, rng, n_features=4, guide_sd=0.1):
    """Cells from ``(gene, guide, n_cells, shift, condition)`` tuples, each guide with its own small offset."""
    blocks, rows = [], []
    for gene, guide, n, shift, condition in groups:
        offset = rng.normal(scale=guide_sd, size=n_features)
        blocks.append(rng.normal(size=(n, n_features)) + shift + offset)
        rows += [(gene, guide, condition)] * n
    obs = pd.DataFrame(rows, columns=["Metadata_Gene", "Metadata_sgRNA", "Metadata_Treatment"])
    obs["Metadata_Control_Type"] = np.where(obs["Metadata_Gene"] == "nontargeting", "negcon", "treatment")
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(np.concatenate(blocks), obs=obs)
    adata.var_names = [f"Cells_Intensity_F{i}" for i in range(n_features)]
    stamp(adata, resolution="object")
    return adata


def _table(adata, key="bootstrap"):
    return adata.uns["mantispy"][key].pivot(index="group", columns="feature", values="pvalue")


@pytest.mark.parametrize("method", ["stouffer", "order", "fisher", "median"])
def test_nontargeting_pseudogenes_are_called_at_the_nominal_rate(method):
    """Non-targeting guides grouped into pseudo-genes and tested against the other non-targeting guides."""
    rng = np.random.default_rng(0)
    groups = [("nontargeting", f"nt{i}", int(rng.integers(30, 90)), 0.0, "a") for i in range(30)]
    groups += [(f"pseudo{i // 4}", f"ps{i}", int(rng.integers(30, 90)), 0.0, "a") for i in range(40)]
    # The null also carries the control guide's own sampling error, so it runs conservative when guides barely differ.
    adata = _cells(groups, rng, n_features=40, guide_sd=0.3)
    mt.tl.bootstrap_features(adata, method=method)
    for key in ("bootstrap", "bootstrap_guides"):
        pvalues = adata.uns["mantispy"][key]["pvalue"].to_numpy()
        assert 0.025 < (pvalues < 0.05).mean() < 0.085


def test_missing_cells_are_called_at_the_nominal_rate():
    """A pure-null screen with half of every cell's features missing, controls included, keeps the nominal rate.

    Drawing a guide's measured count from all of a control guide's cells, missing ones included, would count
    the missing values twice and run conservative.
    """
    rng = np.random.default_rng(0)
    groups = [("nontargeting", f"nt{i}", 150, 0.0, "a") for i in range(8)]
    groups += [(f"pseudo{i // 3}", f"ps{i}", int(rng.integers(30, 71)), 0.0, "a") for i in range(120)]
    adata = _cells(groups, rng, n_features=20, guide_sd=0.3)
    adata.X[rng.random(adata.X.shape) < 0.5] = np.nan
    mt.tl.bootstrap_features(adata)
    for key in ("bootstrap", "bootstrap_guides"):
        pvalues = adata.uns["mantispy"][key]["pvalue"].to_numpy()
        assert 0.035 < (pvalues < 0.05).mean() < 0.1


def test_a_feature_is_tested_on_the_cells_that_measure_it():
    """Missing cells, in control and tested guides alike, give the p-values of the same test without those cells."""
    rng = np.random.default_rng(15)
    groups = [("nontargeting", f"nt{i}", int(rng.integers(20, 40)), 0.0, "a") for i in range(5)]
    groups += [(f"G{g}", f"g{g}_{s}", int(rng.integers(5, 15)), 0.0, "a") for g in range(3) for s in range(2 + g % 2)]
    adata = _cells(groups, rng, n_features=3)
    adata.X[rng.random(adata.X.shape) < 0.3] = np.nan
    mt.tl.bootstrap_features(adata)
    for key in ("bootstrap", "bootstrap_guides"):
        table = adata.uns["mantispy"][key]
        for j, feature in enumerate(adata.var_names):
            measured = adata[~np.isnan(adata.X[:, j]), [feature]].copy()
            mt.tl.bootstrap_features(measured)
            np.testing.assert_allclose(
                table.loc[table["feature"] == feature, "pvalue"].to_numpy(),
                measured.uns["mantispy"][key]["pvalue"].to_numpy(),
                atol=1e-12,
            )


def test_a_guide_missing_a_feature_leaves_its_gene_for_that_feature():
    """Pseudo-genes of two guides, one of which measures nothing in any feature: still called at the nominal rate."""
    rng = np.random.default_rng(0)
    groups = [("nontargeting", f"nt{i}", int(rng.integers(30, 90)), 0.0, "a") for i in range(30)]
    groups += [(f"pseudo{i // 2}", f"ps{i}", int(rng.integers(30, 90)), 0.0, "a") for i in range(80)]
    adata = _cells(groups, rng, n_features=40, guide_sd=0.3)
    unmeasured = adata.obs["Metadata_sgRNA"].isin([f"ps{i}" for i in range(0, 80, 2)]).to_numpy()
    adata.X[unmeasured] = np.nan
    mt.tl.bootstrap_features(adata, method="median", median="midpoint")
    gene = adata.uns["mantispy"]["bootstrap"]
    assert 0.02 < (gene["pvalue"] < 0.05).mean() < 0.085
    guides = adata.uns["mantispy"]["bootstrap_guides"].set_index(["replicate", "feature"])["pvalue"]
    measured = guides.drop([f"ps{i}" for i in range(0, 80, 2)], level="replicate")
    np.testing.assert_allclose(np.sort(gene["pvalue"]), np.sort(measured), atol=1e-12)


def _nanmedian(a, axis, median):
    """``np.nanmedian``, or the lower median of the values that are not ``NaN``, along ``axis``."""
    if median == "midpoint":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a slice of only NaN
            return np.nanmedian(a, axis=axis)
    a = np.sort(np.moveaxis(np.asarray(a, dtype=float), axis, -1), axis=-1)  # NaN sorts last
    n = (~np.isnan(a)).sum(axis=-1)
    picked = np.take_along_axis(a, np.maximum(n - 1, 0)[..., None] // 2, axis=-1)[..., 0]
    return np.where(n > 0, picked, np.nan)


def _simulated_gene_pvalue(adata, gene, feature, n_sims, rng, median="midpoint"):
    """Two-sided p of one gene and feature from a simulation of the documented null, on the cells that measure it."""
    x = np.asarray(adata.X[:, feature], dtype=float)
    obs = adata.obs
    sgrna = obs["Metadata_sgRNA"].to_numpy()
    is_ctrl = (obs["Metadata_Control_Type"] == "negcon").to_numpy()
    controls = [x[(sgrna == c) & is_ctrl & ~np.isnan(x)] for c in sorted(set(sgrna[is_ctrl]))]
    sizes = np.array([c.size for c in controls], dtype=float)
    guides = [x[sgrna == c] for c in sorted(set(sgrna[(obs["Metadata_Gene"] == gene).to_numpy()]))]
    padded = np.full((len(controls), int(sizes.max())), np.nan)
    for i, cells in enumerate(controls):
        padded[i, : cells.size] = cells
    nulls = []
    for cells in guides:
        m = int((~np.isnan(cells)).sum())
        pick = rng.choice(len(controls), size=n_sims, p=sizes / sizes.sum())
        draws = padded[pick[:, None], rng.integers(0, sizes[pick].astype(int)[:, None], size=(n_sims, m))]
        nulls.append(_nanmedian(draws, 1, median))
    null = _nanmedian(np.array(nulls), 0, median)
    observed = _nanmedian(np.array([_nanmedian(cells, 0, median) for cells in guides]), 0, median)
    one = ((null > observed).mean() + 0.5 * (null == observed).mean()) / (~np.isnan(null)).mean()
    return 2 * min(one, 1 - one)


def test_sporadic_missing_control_cells_match_a_simulation():
    """A missing cell in every control guide gives each guide null one band per measured draw count."""
    rng = np.random.default_rng(10)
    groups = [("nontargeting", f"nt{i}", 50, 0.0, "a") for i in range(6)]
    groups += [("TWO", "two0", 40, 0.0, "a"), ("TWO", "two1", 30, 0.0, "a"), ("ONE", "one0", 40, 0.0, "a")]
    adata = _cells(groups, rng, n_features=2)
    for i in range(6):
        adata.X[np.flatnonzero((adata.obs["Metadata_sgRNA"] == f"nt{i}").to_numpy())[0], 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mt.tl.bootstrap_features(adata, method="median", median="midpoint")
    gene = _table(adata)
    guide = adata.uns["mantispy"]["bootstrap_guides"].pivot(index="replicate", columns="feature", values="pvalue")
    np.testing.assert_allclose(gene.loc["ONE"], guide.loc["one0"], atol=1e-12)
    simulated = _simulated_gene_pvalue(adata, "TWO", 0, 20_000, rng)
    assert gene.loc["TWO", "Cells_Intensity_F0"] == pytest.approx(simulated, abs=0.03)


def test_finely_tied_control_values_match_a_simulation():
    """Ties make the sizes of a median's atoms rise and fall, so stopping at the first small one dropped central mass."""
    rng = np.random.default_rng(17)
    groups = [("nontargeting", f"nt{i}", 1000, 0.0, "a") for i in range(2)]
    groups += [("G", "g0", 50, 0.1, "a"), ("G", "g1", 50, 0.1, "a")]
    adata = _cells(groups, rng, n_features=4, guide_sd=0.0)
    adata.X = np.round(adata.X, 3)
    mt.tl.bootstrap_features(adata, method="median", median="midpoint")
    exact = _table(adata).loc["G", adata.var_names].to_numpy()
    simulated = [_simulated_gene_pvalue(adata, "G", f, 100_000, rng) for f in range(4)]
    np.testing.assert_allclose(exact, simulated, atol=0.01)


@pytest.mark.parametrize("median", ["lower", "midpoint"])
@pytest.mark.parametrize("n_guides", [64, 65])
def test_a_gene_of_many_guides_matches_a_simulation(n_guides, median):
    """The gene null sums over its guides by a dynamic program, so a gene of many guides costs no more than a few."""
    rng = np.random.default_rng(12)
    groups = [("nontargeting", f"nt{i}", 20, 0.0, "a") for i in range(8)]
    groups += [("BIG", f"big{i}", 5, np.array([0.0, 0.08]), "a") for i in range(n_guides)]
    adata = _cells(groups, rng, n_features=2)
    adata.X[rng.random(adata.X.shape) < 0.05] = np.nan
    mt.tl.bootstrap_features(adata, method="median", median=median)
    exact = _table(adata).loc["BIG", adata.var_names].to_numpy()
    simulated = [_simulated_gene_pvalue(adata, "BIG", f, 4000, rng, median) for f in range(2)]
    np.testing.assert_allclose(exact, simulated, atol=0.06)


def test_a_one_guide_gene_against_large_control_guides_has_its_guide_pvalue():
    """Every median atom of a two-cell guide's null is tiny when the control guides are large; the count stays exact."""
    rng = np.random.default_rng(11)
    groups = [("nontargeting", f"nt{i}", 15_000, 0.0, "a") for i in range(2)] + [("G", "g0", 2, 0.0, "a")]
    adata = _cells(groups, rng, n_features=1)
    mt.tl.bootstrap_features(adata, method="median", median="midpoint")
    gene = adata.uns["mantispy"]["bootstrap"]["pvalue"].to_numpy()
    guide = adata.uns["mantispy"]["bootstrap_guides"]["pvalue"].to_numpy()
    assert 0.0 < gene[0] == pytest.approx(guide[0], abs=1e-12)


def test_a_gene_whose_guide_nulls_are_too_fine_to_hold_is_left_nan(caplog):
    """Two-cell guides against control guides of 15,000 cells: their nulls have more distinct medians than fit.

    The boundary of an even guide count needs those medians, so under the midpoint median the gene is left missing
    with a warning rather than given the p-value of a tie; its guides keep their exact p-values. Their lower
    medians have no more distinct values than the control guides, which fit, so the gene is tested.
    """
    import logging

    rng = np.random.default_rng(3)
    groups = [("nontargeting", f"nt{i}", 15_000, 0.0, "a") for i in range(2)]
    groups += [("G", "g0", 2, 0.5, "a"), ("G", "g1", 2, 0.5, "a")]
    adata = _cells(groups, rng, n_features=1)
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        mt.tl.bootstrap_features(adata, method="median", median="midpoint")
    assert "1 gene-feature p-value(s) are left NaN" in caplog.text
    assert np.isnan(adata.uns["mantispy"]["bootstrap"]["pvalue"]).all()
    assert np.isfinite(adata.uns["mantispy"]["bootstrap_guides"]["pvalue"]).all()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        mt.tl.bootstrap_features(adata, method="median")
    assert "left NaN" not in caplog.text
    assert np.isfinite(adata.uns["mantispy"]["bootstrap"]["pvalue"]).all()


def test_no_measured_guide_and_no_feature_leave_empty_results():
    """No tested cell measuring any feature leaves every p-value missing; no feature at all leaves empty tables."""
    rng = np.random.default_rng(16)
    groups = [("nontargeting", f"nt{i}", 20, 0.0, "a") for i in range(3)] + [("G", "g0", 10, 0.0, "a")]
    adata = _cells(groups, rng, n_features=2)
    adata.X[(adata.obs["Metadata_Gene"] == "G").to_numpy()] = np.nan
    mt.tl.bootstrap_features(adata)
    for key in ("bootstrap", "bootstrap_guides"):
        assert len(adata.uns["mantispy"][key]) == 2
        assert np.isnan(adata.uns["mantispy"][key]["pvalue"]).all()
    empty = adata[:, []].copy()
    mt.tl.bootstrap_features(empty)
    for key in ("bootstrap", "bootstrap_guides"):
        assert empty.uns["mantispy"][key].empty


def test_a_shifted_gene_is_called_with_the_sign_of_its_shift():
    rng = np.random.default_rng(1)
    shift = np.array([1.0, -1.0, 0.0, 0.0])
    groups = [("nontargeting", f"nt{i}", 50, 0.0, "a") for i in range(8)]
    groups += [("HIT", f"hit{i}", 40, shift, "a") for i in range(3)]
    adata = _cells(groups, rng)
    mt.tl.bootstrap_features(adata)
    table = adata.uns["mantispy"]["bootstrap"].set_index("feature")
    assert (table["qvalue"].iloc[:2] < 0.01).all()
    assert np.sign(table["effect"].iloc[:2]).tolist() == [1.0, -1.0]
    guides = adata.uns["mantispy"]["bootstrap_guides"]
    assert set(guides["replicate"]) == {"hit0", "hit1", "hit2"} and set(guides["n_cells"]) == {40}
    assert not any("not a registered result key" in w for w in mt.io.validate(adata).warnings)
    mt.pl.feature_volcano(adata, "HIT", key="bootstrap")
    mt.pl.effect_sizes(adata, "HIT", key="bootstrap")


def test_a_gene_whose_guides_disagree_has_the_effect_sign_of_its_call():
    """One large guide moves up and two small ones down: under every rule the effect points the way the gene is called."""
    rng = np.random.default_rng(0)
    groups = [("nontargeting", f"nt{i}", 300, 0.0, "a") for i in range(8)]
    groups += [("G", "big", 3000, 0.3, "a"), ("G", "s1", 60, -0.6, "a"), ("G", "s2", 60, -0.6, "a")]
    adata = _cells(groups, rng, n_features=1, guide_sd=0.0)
    # Stouffer and Fisher follow the large guide; the order rule needs two of three guides, so it follows the small ones
    for method, sign in (("stouffer", 1.0), ("fisher", 1.0), ("order", -1.0)):
        mt.tl.bootstrap_features(adata, method=method)
        guides = adata.uns["mantispy"]["bootstrap_guides"].set_index("replicate")
        assert np.sign(guides.loc[["big", "s1", "s2"], "effect"]).tolist() == [1.0, -1.0, -1.0]
        gene = adata.uns["mantispy"]["bootstrap"].iloc[0]
        assert gene["pvalue"] < 1e-4 and np.sign(gene["effect"]) == sign
        agreeing = guides[np.sign(guides["effect"]) == sign]
        assert gene["effect"] == pytest.approx(np.average(agreeing["effect"], weights=np.sqrt(agreeing["n_cells"])))


def test_the_order_rule_needs_both_guides_of_a_two_guide_gene():
    """A strict majority of two guides is both, so one shifted guide does not carry its gene under "order"."""
    rng = np.random.default_rng(3)
    groups = [("nontargeting", f"nt{i}", 50, 0.0, "a") for i in range(8)]
    groups += [("ONE", "one0", 40, 2.0, "a"), ("ONE", "one1", 40, 0.0, "a")]
    groups += [("BOTH", "both0", 40, 2.0, "a"), ("BOTH", "both1", 40, 2.0, "a")]
    adata = _cells(groups, rng, n_features=1, guide_sd=0.0)
    mt.tl.bootstrap_features(adata, method="order")
    p = _table(adata)["Cells_Intensity_F0"]
    assert p["BOTH"] < 1e-6
    assert p["ONE"] > 0.05


def _spread_null_screen():
    """A pure-null screen whose guides differ by more than the error of their medians: 250 pseudo-genes of 4 guides."""
    rng = np.random.default_rng(5)
    n_cells, n_features = 400, 20
    names = [("nontargeting", f"nt{i}") for i in range(200)] + [(f"ps{i // 4}", f"ps{i}") for i in range(1000)]
    blocks = [rng.normal(size=(n_cells, n_features)) + rng.normal(scale=0.3, size=n_features) for _ in names]
    obs = pd.DataFrame(np.repeat(names, n_cells, axis=0), columns=["Metadata_Gene", "Metadata_sgRNA"])
    obs["Metadata_Control_Type"] = np.where(obs["Metadata_Gene"] == "nontargeting", "negcon", "treatment")
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(np.concatenate(blocks).astype(np.float32), obs=obs)
    adata.var_names = [f"Cells_Intensity_F{i}" for i in range(n_features)]
    stamp(adata, resolution="object")
    return adata


@pytest.mark.parametrize(
    "method",
    [
        pytest.param(
            "stouffer",
            marks=pytest.mark.xfail(
                reason="Stouffer adds the guides' spread around the controls, which each guide's null already "
                "carries, so with guides that differ by more than the error of their medians it calls null genes",
                strict=True,
            ),
        ),
        "median",
    ],
)
def test_guide_spread_does_not_call_null_genes(method):
    """Pseudo-genes of control guides with spread between guides: no gene-feature pair passes q < 0.05."""
    adata = _spread_null_screen()
    mt.tl.bootstrap_features(adata, method=method)
    assert (adata.uns["mantispy"]["bootstrap"]["qvalue"] < 0.05).sum() == 0


def test_a_median_that_cannot_move_is_not_called():
    """A feature that is zero in most cells has a median of zero in the null and the guide alike: a tie, not a hit."""
    rng = np.random.default_rng(2)
    groups = [("nontargeting", f"nt{i}", 50, 0.0, "a") for i in range(6)] + [("G", "g0", 40, 0.0, "a")]
    adata = _cells(groups, rng)
    adata.X[:, 0] = np.where(rng.random(adata.n_obs) < 0.9, 0.0, 1.0)
    mt.tl.bootstrap_features(adata)
    assert _table(adata).loc["G", "Cells_Intensity_F0"] > 0.5


def test_missing_values_are_skipped_and_an_unmeasured_feature_is_nan():
    rng = np.random.default_rng(3)
    groups = [("nontargeting", f"nt{i}", 50, 0.0, "a") for i in range(6)] + [("G", "g0", 40, 0.0, "a")]
    adata = _cells(groups, rng)
    is_gene = (adata.obs["Metadata_Gene"] == "G").to_numpy()
    adata.X[is_gene, 0] = np.nan
    adata.X[rng.random(adata.n_obs) < 0.2, 1] = np.nan
    adata.X[~is_gene, 2] = np.nan  # no control cell measures it, so there is no null
    mt.tl.bootstrap_features(adata)
    row = _table(adata).loc["G"]
    assert np.isnan(row[["Cells_Intensity_F0", "Cells_Intensity_F2"]]).all()
    assert np.isfinite(row.drop(["Cells_Intensity_F0", "Cells_Intensity_F2"])).all()
    guides = adata.uns["mantispy"]["bootstrap_guides"].set_index("feature")["pvalue"]
    assert np.isnan(guides["Cells_Intensity_F2"])


def test_control_guides_of_two_control_genes_stay_apart_when_they_share_a_label():
    """A guide is its gene and its label, so two control genes that both number their guides g0, g1 keep four guides."""
    rng = np.random.default_rng(14)
    groups = [
        (control, f"g{i}", 30, 1.0 * (control == "intergenic"), "a")
        for control in ("nontargeting", "intergenic")
        for i in range(2)
    ]
    groups += [("G", "x0", 20, 0.0, "a"), ("G", "x1", 25, 0.0, "a")]
    shared = _cells(groups, rng, n_features=2)
    shared.obs["Metadata_Control_Type"] = np.where(shared.obs["Metadata_Gene"] == "G", "treatment", "negcon")
    distinct = shared.copy()
    distinct.obs["Metadata_sgRNA"] = distinct.obs["Metadata_Gene"].str[:2] + distinct.obs["Metadata_sgRNA"]
    for adata in (shared, distinct):
        mt.tl.bootstrap_features(adata, method="median", median="midpoint")
    pd.testing.assert_frame_equal(_table(shared), _table(distinct))


def test_a_gene_keeps_its_pvalues_when_other_genes_are_dropped():
    """A gene's p-values use only its guides and the controls, so dropping other genes cannot change them."""
    rng = np.random.default_rng(4)
    groups = [("nontargeting", f"nt{i}", 50, 0.0, "a") for i in range(6)]
    groups += [(f"G{g}", f"g{g}_{s}", 30, 0.2 * g, "a") for g in range(4) for s in range(2)]
    adata = _cells(groups, rng)
    mt.tl.bootstrap_features(adata, method="median", median="midpoint")
    subset = adata[adata.obs["Metadata_Gene"].isin(["nontargeting", "G2"])].copy()
    mt.tl.bootstrap_features(subset, method="median", median="midpoint")
    pd.testing.assert_series_equal(_table(adata).loc["G2"], _table(subset).loc["G2"])


@pytest.mark.parametrize(("n", "p"), [(40, 0.3), (1500, 0.5), (20000, 0.37), (5000, 0.999)])
def test_binomial_cdf_matches_scipy(n, p):
    from scipy.stats import binom

    from mantispy.tl._bootstrap import _binom_cdf

    for k in sorted({0, n // 4, int(n * p) - 1, int(n * p), n // 2, n - 1}):
        assert _binom_cdf(k, n, p) == pytest.approx(binom.cdf(k, n, p), abs=1e-12, rel=1e-9)


def test_nulls_no_earlier_feature_built_are_counted_by_every_thread():
    """A feature's first-built nulls carry no cost from the last one; they are dealt by their bound, not to one slot."""
    from mantispy.tl._bootstrap import _bound_unknown_costs, _deal

    n_nulls, n_slots = 64, 8
    cost = np.zeros(n_nulls)  # the costs the previous feature left: it built none of these
    used = np.ones(n_nulls, dtype=bool)
    _bound_unknown_costs(cost, used, np.arange(1, n_nulls + 1, dtype=np.int64))
    _, start = _deal(cost, n_slots)
    assert np.diff(start).max() < n_nulls // 2


@pytest.mark.parametrize("decimals", [None, 3])
@pytest.mark.parametrize("m", [2, 7, 50, 1500])
def test_a_guide_null_keeps_its_mass(decimals, m):
    """A guide null's atoms carry all of its mass but its far tails, on continuous and on finely tied values alike."""
    from mantispy.tl._bootstrap import _ATOM_EPS, _SORT_SCRATCH, _build_construct, _null_caps

    rng = np.random.default_rng(18)
    n, n_subs = 1000, 2
    values = rng.normal(size=(n_subs, n))
    if decimals is not None:
        values = np.round(values, decimals)
    row = np.sort(values, axis=1).ravel()
    start = np.arange(n_subs, dtype=np.int64) * n
    cap = _null_caps(start, start + n, np.array([0]), np.array([n_subs]), np.array([0]), np.array([m]))[0]
    val, prob = np.empty(cap), np.empty(cap)
    scratch = [np.empty(n) for _ in range(3)]
    sort_scratch = (np.empty(cap), np.empty(cap), np.empty(_SORT_SCRATCH, np.int64))
    count = _build_construct(
        row, np.full(n_subs, n), start, 0, n_subs, m, _ATOM_EPS, val, prob, *sort_scratch, *scratch
    )
    assert count > 0
    assert np.all(np.diff(val[:count]) > 0)
    assert 1.0 - 4 * _ATOM_EPS < prob[:count].sum() <= 1.0 + 1e-12


def test_guides_of_thousands_of_cells_match_a_simulation():
    """Binomial and order-statistic terms that underflow for thousands of draws are formed in log space."""
    rng = np.random.default_rng(8)
    groups = [("nontargeting", f"nt{i}", 50, 0.0, "a") for i in range(6)]
    groups += [("BIG", f"big{i}", 2400, 0.0, "a") for i in range(2)]
    adata = _cells(groups, rng, n_features=3)
    mt.tl.bootstrap_features(adata, method="median", median="midpoint")

    x = np.asarray(adata.X)
    obs = adata.obs
    controls = [x[(obs["Metadata_sgRNA"] == f"nt{i}").to_numpy()] for i in range(6)]
    big = [x[(obs["Metadata_sgRNA"] == f"big{i}").to_numpy()] for i in range(2)]
    n_sims = 4000
    # equal-sized control guides, so each is drawn with equal probability
    nulls = [
        [np.median(controls[rng.integers(6)][rng.integers(0, 50, size=cells.shape[0])], axis=0) for _ in range(n_sims)]
        for cells in big
    ]
    gene_null = np.median(np.array(nulls), axis=0)
    gene_obs = np.median([np.median(cells, axis=0) for cells in big], axis=0)
    one = (gene_null > gene_obs).mean(axis=0)
    simulated = 2 * np.minimum(one, 1 - one)
    exact = _table(adata).loc["BIG", adata.var_names].to_numpy()
    np.testing.assert_allclose(exact, simulated, atol=0.06)


def test_a_guide_with_no_null_mass_beyond_the_observed_median():
    """A gene far out in the tail: its large guides' nulls keep no atom past the observed median.

    Their exact probability of lying there is far below the pruned atoms', so the boundary of a
    resample that puts them there has no shape to draw from; it must be left out, not divided by zero.
    """
    rng = np.random.default_rng(9)
    groups = [("nontargeting", f"nt{i}", 60, 0.0, "a") for i in range(6)]
    groups += [("FAR", f"far{i}", n, -1.5, "a") for i, n in enumerate((3, 3, 200, 200))]
    adata = _cells(groups, rng, n_features=2, guide_sd=0.0)
    mt.tl.bootstrap_features(adata, method="median", median="midpoint")
    p = _table(adata).loc["FAR"].to_numpy()
    assert np.all((p >= 0.0) & (p < 1e-3))


def _two_conditions(rng):
    """Controls shifted in condition "b"; gene SAME is like the controls of its condition; FLIP moves only in "b"."""
    shift = np.array([1.5, 0.0, 0.0, 0.0])
    groups = []
    for condition, offset in (("a", 0.0), ("b", shift)):
        groups += [("nontargeting", f"nt{i}", 50, offset, condition) for i in range(6)]
        groups += [("SAME", f"same{i}", 40, offset, condition) for i in range(2)]
    groups += [("FLIP", f"flip{i}", 40, 0.0, "a") for i in range(2)]
    groups += [("FLIP", f"flip{i}", 40, 0.0, "b") for i in range(2)]
    return _cells(groups, rng)


@pytest.mark.parametrize("method", ["stouffer", "order", "fisher", "median"])
def test_control_scopes_pick_the_null(method):
    rng = np.random.default_rng(5)
    adata = _two_conditions(rng)
    by = ["Metadata_Gene", "Metadata_Treatment"]
    kwargs = {"groupby": by, "condition": "Metadata_Treatment", "method": method}
    f0 = "Cells_Intensity_F0"

    mt.tl.bootstrap_features(adata, control_scope="within_group", key_added="within", **kwargs)
    within = _table(adata, "within")[f0]
    assert within["('SAME', 'b')"] > 0.05, "against its own condition's controls, SAME is not a hit"
    assert within["('FLIP', 'b')"] < 0.01

    mt.tl.bootstrap_features(
        adata, control_scope="reference_group", reference_condition="a", key_added="reference", **kwargs
    )
    reference = _table(adata, "reference")[f0]
    assert reference["('SAME', 'b')"] < 0.01, "against condition a's controls, condition b moves"
    assert reference["('FLIP', 'b')"] > 0.05, "FLIP in b undoes the shift of b"

    mt.tl.bootstrap_features(
        adata, control_scope="within_perturbation", reference_condition="a", key_added="own", **kwargs
    )
    own = _table(adata, "own")[f0]
    assert set(own.index) == {"('SAME', 'b')", "('FLIP', 'b')", "('nontargeting', 'b')"}
    assert own["('nontargeting', 'b')"] < 0.01 and own["('SAME', 'b')"] < 0.01
    assert own["('FLIP', 'b')"] > 0.05, "FLIP in b sits where FLIP in a does"


def test_within_perturbation_tests_a_control_gene_of_many_guides():
    """Each gene is its own control, so the non-targeting group, with all its guides, is tested too."""
    rng = np.random.default_rng(13)
    groups = [("nontargeting", f"nt{i}", 5, 0.0, condition) for condition in "ab" for i in range(63)]
    groups += [("G", f"g{i}", 5, 0.0, condition) for condition in "ab" for i in range(2)]
    adata = _cells(groups, rng, n_features=2)
    mt.tl.bootstrap_features(
        adata,
        groupby=["Metadata_Gene", "Metadata_Treatment"],
        control_scope="within_perturbation",
        condition="Metadata_Treatment",
        reference_condition="a",
        method="median",
        median="midpoint",
    )
    p = _table(adata).loc["('nontargeting', 'b')"].to_numpy()
    assert np.all((p > 0.0) & (p <= 1.0))


def test_a_group_without_a_reference_arm_is_left_out(caplog):
    import logging

    rng = np.random.default_rng(6)
    adata = _two_conditions(rng)
    adata = adata[~((adata.obs["Metadata_Gene"] == "SAME") & (adata.obs["Metadata_Treatment"] == "a"))].copy()
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        mt.tl.bootstrap_features(
            adata,
            groupby=["Metadata_Gene", "Metadata_Treatment"],
            control_scope="within_perturbation",
            condition="Metadata_Treatment",
            reference_condition="a",
        )
    assert "1 group(s) have no 'a' arm" in caplog.text
    assert "('SAME', 'b')" not in set(adata.uns["mantispy"]["bootstrap"]["group"])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"control_scope": "nearby"}, "control_scope must be one of"),
        ({"median": "linear"}, "median must be one of"),
        ({"method": "mean"}, "method must be one of"),
        ({"rtol": -1e-3}, "rtol must be"),
        ({"rtol": np.nan}, "rtol must be"),
        ({"rtol": np.inf}, "rtol must be"),
        ({"control_scope": "within_group"}, "needs condition="),
        ({"control_scope": "reference_group", "condition": "Metadata_Treatment"}, "needs reference_condition="),
        (
            {"control_scope": "within_perturbation", "condition": "Metadata_Treatment", "reference_condition": "a"},
            "among the groupby columns",
        ),
        (
            {"control_scope": "reference_group", "condition": "Metadata_Treatment", "reference_condition": "z"},
            "no control cell has",
        ),
    ],
)
def test_bad_calls_raise(kwargs, match):
    adata = _two_conditions(np.random.default_rng(7))
    with pytest.raises(ValueError, match=match):
        mt.tl.bootstrap_features(adata, **kwargs)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_an_infinite_value_raises(value):
    """A median over both infinities is undefined, so an infinite value is refused rather than left to hang the sweep."""
    adata = _two_conditions(np.random.default_rng(7))
    adata.X[0, 0] = value
    with pytest.raises(ValueError, match="infinite values"):
        mt.tl.bootstrap_features(adata)
