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


def _brute_pvalues(adata):
    """Exact p-values by full enumeration of every resample; only for tiny screens.

    Per feature, only the cells that measure it count: the guide null draws one control guide
    (probability proportional to its measured cells), then draws as many of those as the guide has
    measured, with replacement, and takes their median; the gene null is the median over its measured
    guides of independent such draws. Two-sided, a tie counts half, no floor.
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
                median = float(np.median(col[list(idx)]))
                pmf[median] = pmf.get(median, 0.0) + n / total / n**m
        return pmf, 0.0

    def two_sided(pmf, nan, o):
        if nan >= 1.0 - 1e-12:
            return np.nan
        gt = sum(p for v, p in pmf.items() if v > o)
        eq = sum(p for v, p in pmf.items() if v == o)
        one = (gt + 0.5 * eq) / (1.0 - nan)
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
                guide_p[(g, c, feat)] = two_sided(pmf, nan, float(np.median(measured[c])))
            if not kept:
                gene_p[(g, feat)] = np.nan
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
                    gpmf[float(np.median(good))] = gpmf.get(float(np.median(good)), 0.0) + prob
            og = float(np.median([np.median(measured[c]) for c in kept]))
            gene_p[(g, feat)] = two_sided(gpmf, gnan, og)
    return gene_p, guide_p


def _assert_matches_brute_force(adata):
    gene_p, guide_p = _brute_pvalues(adata)
    features = list(adata.var_names)
    table = adata.uns["mantispy"]["bootstrap"]
    for row in table.itertuples():
        np.testing.assert_allclose(row.pvalue, gene_p[(row.group, features.index(row.feature))], atol=1e-9)
    guides = adata.uns["mantispy"]["bootstrap_guides"]
    for row in guides.itertuples():
        expected = guide_p[(row.group, row.replicate, features.index(row.feature))]
        np.testing.assert_allclose(row.pvalue, expected, atol=1e-9)


def test_exact_pvalues_match_brute_force_enumeration():
    """On a tiny screen the analytic p-values equal a full enumeration of every resample."""
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
    mt.tl.bootstrap_features(adata)
    _assert_matches_brute_force(adata)


@pytest.mark.parametrize(("dtype", "missing"), [(np.float32, 0.0), (np.float32, 0.2), (np.float64, 0.2)])
def test_missing_cells_match_brute_force_enumeration(dtype, missing):
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
    mt.tl.bootstrap_features(adata)
    _assert_matches_brute_force(adata)


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


def test_nontargeting_pseudogenes_are_called_at_the_nominal_rate():
    """Non-targeting guides grouped into pseudo-genes and tested against the other non-targeting guides."""
    rng = np.random.default_rng(0)
    groups = [("nontargeting", f"nt{i}", int(rng.integers(30, 90)), 0.0, "a") for i in range(30)]
    groups += [(f"pseudo{i // 4}", f"ps{i}", int(rng.integers(30, 90)), 0.0, "a") for i in range(40)]
    # The null also carries the control guide's own sampling error, so it runs conservative when guides barely differ.
    adata = _cells(groups, rng, n_features=40, guide_sd=0.3)
    mt.tl.bootstrap_features(adata)
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
    mt.tl.bootstrap_features(adata)
    gene = adata.uns["mantispy"]["bootstrap"]
    assert 0.02 < (gene["pvalue"] < 0.05).mean() < 0.085
    guides = adata.uns["mantispy"]["bootstrap_guides"].set_index(["replicate", "feature"])["pvalue"]
    measured = guides.drop([f"ps{i}" for i in range(0, 80, 2)], level="replicate")
    np.testing.assert_allclose(np.sort(gene["pvalue"]), np.sort(measured), atol=1e-12)


def _simulated_gene_pvalue(adata, gene, feature, n_sims, rng):
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a guide measuring nothing
            nulls.append(np.nanmedian(draws, axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        null = np.nanmedian(np.array(nulls), axis=0)
    observed = np.median([np.nanmedian(cells) for cells in guides])
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
        mt.tl.bootstrap_features(adata)
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
    mt.tl.bootstrap_features(adata)
    exact = _table(adata).loc["G", adata.var_names].to_numpy()
    simulated = [_simulated_gene_pvalue(adata, "G", f, 100_000, rng) for f in range(4)]
    np.testing.assert_allclose(exact, simulated, atol=0.01)


@pytest.mark.parametrize("n_guides", [64, 65])
def test_a_gene_of_many_guides_matches_a_simulation(n_guides):
    """The gene null sums over its guides by a dynamic program, so a gene of many guides costs no more than a few."""
    rng = np.random.default_rng(12)
    groups = [("nontargeting", f"nt{i}", 20, 0.0, "a") for i in range(8)]
    groups += [("BIG", f"big{i}", 5, np.array([0.0, 0.08]), "a") for i in range(n_guides)]
    adata = _cells(groups, rng, n_features=2)
    adata.X[rng.random(adata.X.shape) < 0.05] = np.nan
    mt.tl.bootstrap_features(adata)
    exact = _table(adata).loc["BIG", adata.var_names].to_numpy()
    simulated = [_simulated_gene_pvalue(adata, "BIG", f, 4000, rng) for f in range(2)]
    np.testing.assert_allclose(exact, simulated, atol=0.06)


def test_a_one_guide_gene_against_large_control_guides_has_its_guide_pvalue():
    """Every median atom of a two-cell guide's null is tiny when the control guides are large; the count stays exact."""
    rng = np.random.default_rng(11)
    groups = [("nontargeting", f"nt{i}", 15_000, 0.0, "a") for i in range(2)] + [("G", "g0", 2, 0.0, "a")]
    adata = _cells(groups, rng, n_features=1)
    mt.tl.bootstrap_features(adata)
    gene = adata.uns["mantispy"]["bootstrap"]["pvalue"].to_numpy()
    guide = adata.uns["mantispy"]["bootstrap_guides"]["pvalue"].to_numpy()
    assert 0.0 < gene[0] == pytest.approx(guide[0], abs=1e-12)


def test_a_gene_whose_guide_nulls_are_too_fine_to_hold_is_left_nan(caplog):
    """Two-cell guides against control guides of 15,000 cells: their nulls have more distinct medians than fit.

    The boundary of an even guide count needs those medians, so the gene is left missing with a warning rather
    than given the p-value of a tie; its guides keep their exact p-values.
    """
    import logging

    rng = np.random.default_rng(3)
    groups = [("nontargeting", f"nt{i}", 15_000, 0.0, "a") for i in range(2)]
    groups += [("G", "g0", 2, 0.5, "a"), ("G", "g1", 2, 0.5, "a")]
    adata = _cells(groups, rng, n_features=1)
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        mt.tl.bootstrap_features(adata)
    assert "1 gene-feature p-value(s) are left NaN" in caplog.text
    assert np.isnan(adata.uns["mantispy"]["bootstrap"]["pvalue"]).all()
    assert np.isfinite(adata.uns["mantispy"]["bootstrap_guides"]["pvalue"]).all()


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
        mt.tl.bootstrap_features(adata)
    pd.testing.assert_frame_equal(_table(shared), _table(distinct))


def test_a_gene_keeps_its_pvalues_when_other_genes_are_dropped():
    """A gene's p-values use only its guides and the controls, so dropping other genes cannot change them."""
    rng = np.random.default_rng(4)
    groups = [("nontargeting", f"nt{i}", 50, 0.0, "a") for i in range(6)]
    groups += [(f"G{g}", f"g{g}_{s}", 30, 0.2 * g, "a") for g in range(4) for s in range(2)]
    adata = _cells(groups, rng)
    mt.tl.bootstrap_features(adata)
    subset = adata[adata.obs["Metadata_Gene"].isin(["nontargeting", "G2"])].copy()
    mt.tl.bootstrap_features(subset)
    pd.testing.assert_series_equal(_table(adata).loc["G2"], _table(subset).loc["G2"])


@pytest.mark.parametrize(("n", "p"), [(40, 0.3), (1500, 0.5), (20000, 0.37), (5000, 0.999)])
def test_binomial_cdf_matches_scipy(n, p):
    from scipy.stats import binom

    from mantispy.tl._bootstrap import _binom_cdf

    for k in sorted({0, n // 4, int(n * p) - 1, int(n * p), n // 2, n - 1}):
        assert _binom_cdf(k, n, p) == pytest.approx(binom.cdf(k, n, p), abs=1e-12, rel=1e-9)


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
    subs = np.arange(n_subs, dtype=np.int64)
    cap = _null_caps(start, start + n, subs, np.array([0]), np.array([n_subs]), np.array([0]), np.array([m]))[0]
    val, prob = np.empty(cap), np.empty(cap)
    scratch = [np.empty(n) for _ in range(3)]
    sort_scratch = (np.empty(cap), np.empty(cap), np.empty(_SORT_SCRATCH, np.int64))
    count = _build_construct(
        row, np.full(n_subs, n), start, subs, 0, n_subs, m, _ATOM_EPS, val, prob, *sort_scratch, *scratch
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
    mt.tl.bootstrap_features(adata)

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
    mt.tl.bootstrap_features(adata)
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


def test_control_scopes_pick_the_null():
    rng = np.random.default_rng(5)
    adata = _two_conditions(rng)
    by = ["Metadata_Gene", "Metadata_Treatment"]
    kwargs = {"groupby": by, "condition": "Metadata_Treatment"}
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


def _pseudo_screen(rng, n_cells=20, n_control=10, n_control_cells=30, n_features=3, conditions="a", guide_sd=0.1):
    """Two real genes of two guides, ``n_control`` non-targeting guides, each guide in every condition."""
    groups = [("GENE_A", f"a{i}", n_cells + i, 1.0, c) for i in range(2) for c in conditions]
    groups += [("GENE_B", f"b{i}", n_cells - i, 0.0, c) for i in range(2) for c in conditions]
    groups += [("nontargeting", f"nt{i}", n_control_cells, 0.0, c) for i in range(n_control) for c in conditions]
    return _cells(groups, rng, n_features=n_features, guide_sd=guide_sd)


def _by_hand(adata, name, **kwargs):
    """A pseudo-gene's test run as a plain one: its kept cells as a gene, against the controls without its guides."""
    obs = adata.obs
    control = (obs["Metadata_Control_Type"] == "negcon").to_numpy()
    kept = (obs["pseudogene"] == name).to_numpy()
    own = control & obs["Metadata_sgRNA"].isin(obs.loc[kept, "Metadata_sgRNA"]).to_numpy()
    keep = kept | (control & ~own)
    alone = adata[keep].copy()
    alone.obs["Metadata_Gene"] = np.where(kept[keep], name, alone.obs["Metadata_Gene"].astype(str))
    alone.obs["is_ref"] = ~kept[keep]
    mt.tl.bootstrap_features(alone, reference="is_ref", **kwargs)
    return alone.uns["mantispy"]["bootstrap"].set_index(["group", "feature"])


@pytest.mark.parametrize("missing", [0.0, 0.2])
def test_a_pseudogene_is_tested_against_every_control_guide_but_its_own(missing):
    """Exactly the plain test of its kept cells against the other control guides, missing cells included."""
    rng = np.random.default_rng(21)
    adata = _pseudo_screen(rng)
    if missing:
        adata.X[rng.random(adata.X.shape) < missing] = np.nan
    mt.pp.pseudogenes(adata, size=2)
    mt.tl.bootstrap_features(adata, pseudogenes="pseudogene")
    ours = adata.uns["mantispy"]["bootstrap"].set_index(["group", "feature"])
    names = list(adata.obs["pseudogene"].cat.categories)
    assert len(names) == 5 and set(ours.loc[ours["is_pseudogene"]].index.get_level_values("group")) == set(names)
    for name in names:
        alone = _by_hand(adata, name)
        np.testing.assert_allclose(ours.loc[name, "pvalue"], alone.loc[name, "pvalue"], rtol=1e-12, atol=0)
        np.testing.assert_array_equal(ours.loc[name, "effect"], alone.loc[name, "effect"])


def test_a_pseudogene_split_by_another_column_leaves_all_its_guides_out():
    """With a second groupby column each split of a pseudo-gene draws from the controls without any of its guides."""
    rng = np.random.default_rng(22)
    adata = _pseudo_screen(rng, n_cells=10, n_control_cells=12, conditions="ab")
    mt.pp.pseudogenes(adata, size=2)
    by = ["Metadata_Gene", "Metadata_Treatment"]
    mt.tl.bootstrap_features(adata, groupby=by, pseudogenes="pseudogene")
    ours = adata.uns["mantispy"]["bootstrap"].set_index(["group", "feature"])
    for name in adata.obs["pseudogene"].cat.categories:
        alone = _by_hand(adata, name, groupby=by)
        for split in alone.index.get_level_values("group").unique():
            if name in split:
                np.testing.assert_allclose(ours.loc[split, "pvalue"], alone.loc[split, "pvalue"], rtol=1e-12, atol=0)


def test_real_genes_are_unchanged_by_pseudogenes():
    plain = _pseudo_screen(np.random.default_rng(23))
    mt.tl.bootstrap_features(plain)
    with_pseudo = plain.copy()
    mt.pp.pseudogenes(with_pseudo, size=2)
    mt.tl.bootstrap_features(with_pseudo, pseudogenes="pseudogene")
    for key in ("bootstrap", "bootstrap_guides"):
        table = with_pseudo.uns["mantispy"][key]
        assert (table["is_pseudogene"] == table["group"].str.startswith("pseudogene_")).all()
        real = table[~table["is_pseudogene"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(real, plain.uns["mantispy"][key])
        assert not plain.uns["mantispy"][key]["is_pseudogene"].any()
    assert "bootstrap_summary" not in plain.uns["mantispy"]


def test_a_gene_whose_guide_is_a_reference_guide_keeps_that_guide_in_its_null():
    """Without pseudo-genes no group's guides leave its null, even when its label also sits on reference cells."""
    adata = _pseudo_screen(np.random.default_rng(24), n_cells=6, n_control=4, n_control_cells=8)
    adata.obs["is_ref"] = (adata.obs["Metadata_Control_Type"] == "negcon").to_numpy() | (
        adata.obs["Metadata_sgRNA"] == "a0"
    ).to_numpy()
    mt.tl.bootstrap_features(adata, reference="is_ref")
    renamed = adata.copy()
    renamed.obs["Metadata_Gene"] = np.where(
        renamed.obs["Metadata_sgRNA"] == "a0", "a0_control", renamed.obs["Metadata_Gene"].astype(str)
    )
    mt.tl.bootstrap_features(renamed, reference="is_ref")
    np.testing.assert_allclose(_table(adata).loc["GENE_A"], _table(renamed).loc["GENE_A"], rtol=1e-12, atol=0)


def test_a_pseudogene_with_no_other_control_guide_in_its_scope_is_skipped(caplog):
    import logging

    rng = np.random.default_rng(25)
    adata = _pseudo_screen(rng, n_cells=6, n_control=12, n_control_cells=8)
    mt.pp.pseudogenes(adata, size=4)
    first = (
        adata.obs["Metadata_sgRNA"]
        .isin(adata.obs.loc[adata.obs["pseudogene"] == "pseudogene_01", "Metadata_sgRNA"])
        .to_numpy()
    )
    adata.obs["Metadata_Treatment"] = np.where(first, "a", "b")
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        mt.tl.bootstrap_features(
            adata, control_scope="within_group", condition="Metadata_Treatment", pseudogenes="pseudogene"
        )
    assert "1 pseudo-gene group(s) hold every control guide of their scope" in caplog.text
    groups = set(adata.uns["mantispy"]["bootstrap"]["group"])
    assert "pseudogene_01" not in groups and {"pseudogene_02", "pseudogene_03", "GENE_A"} <= groups


def test_a_null_too_large_for_its_buffer_in_one_feature_leaves_the_next_features_exact(monkeypatch):
    """A base null that overflows in one feature still gives its derived nulls their atoms in the next."""
    from mantispy.tl import _bootstrap

    rng = np.random.default_rng(26)
    adata = _pseudo_screen(rng, n_cells=10, n_control=20, n_control_cells=10, n_features=2)
    adata.X[:, 1] = rng.integers(0, 3, size=adata.n_obs)
    mt.pp.pseudogenes(adata, size=2)
    caps = _bootstrap._null_caps
    monkeypatch.setattr(_bootstrap, "_null_caps", lambda *args: np.minimum(caps(*args), 400))
    mt.tl.bootstrap_features(adata, pseudogenes="pseudogene")
    both = adata.uns["mantispy"]["bootstrap"]
    second = adata[:, [1]].copy()
    mt.tl.bootstrap_features(second, pseudogenes="pseudogene")
    feature = adata.var_names[1]
    np.testing.assert_allclose(
        both.loc[both["feature"] == feature, "pvalue"],
        second.uns["mantispy"]["bootstrap"]["pvalue"],
        rtol=1e-12,
        atol=0,
    )


def test_the_summary_gives_the_rates_at_the_real_genes_cutoff():
    rng = np.random.default_rng(27)
    adata = _pseudo_screen(rng, n_cells=15, n_control=24, n_control_cells=20, n_features=8)
    adata.X[(adata.obs["Metadata_Control_Type"] == "negcon").to_numpy(), 7] = np.nan  # no pseudo-gene measures it
    mt.pp.pseudogenes(adata, size=2)
    mt.tl.bootstrap_features(adata, pseudogenes="pseudogene")
    store = adata.uns["mantispy"]
    table, summary = store["bootstrap"], store["bootstrap_summary"]
    real, pseudo = table[~table["is_pseudogene"]], table[table["is_pseudogene"]].dropna(subset=["pvalue"])
    assert table.loc[table["is_pseudogene"], "pvalue"].isna().any()
    cutoff = real.loc[real["qvalue"] < 0.05, "pvalue"].max()
    called = pseudo["pvalue"] <= cutoff
    per_gene = called.groupby(pseudo["group"]).any()
    assert summary["cutoff"] == cutoff and summary["alpha"] == 0.05
    rates = summary["rates"]
    assert rates.loc["per_test", ["n", "n_called"]].tolist() == [len(called), int(called.sum())]
    assert rates.loc["per_gene", ["n", "n_called"]].tolist() == [12, int(per_gene.sum())]
    assert rates.loc["per_test", "rate"] == pytest.approx(called.mean())
    assert rates.loc["per_test", "ci_low"] <= called.mean() <= rates.loc["per_test", "ci_high"]
    cells = summary["cells_per_guide"]
    guides = store["bootstrap_guides"].drop_duplicates(["group", "replicate"])
    assert cells.loc["real", "median"] == guides.loc[~guides["is_pseudogene"], "n_cells"].median()
    assert cells.loc["pseudogene", "n_guides"] == 24


def test_the_pseudogene_test_survives_a_round_trip(tmp_path):
    adata = _pseudo_screen(np.random.default_rng(28), n_cells=6, n_control=8, n_control_cells=8)
    mt.pp.pseudogenes(adata, size=2)
    path = tmp_path / "screen.h5ad"
    adata.write_h5ad(path)
    read = ad.read_h5ad(path)
    mt.tl.bootstrap_features(adata, pseudogenes="pseudogene")
    mt.tl.bootstrap_features(read, pseudogenes="pseudogene")
    pd.testing.assert_frame_equal(read.uns["mantispy"]["bootstrap"], adata.uns["mantispy"]["bootstrap"])
    read.write_h5ad(path)
    again = ad.read_h5ad(path)
    assert not [warning for warning in mt.io.validate(again).warnings if "registered" in warning]
    summary = again.uns["mantispy"]["bootstrap_summary"]
    pd.testing.assert_frame_equal(summary["rates"], read.uns["mantispy"]["bootstrap_summary"]["rates"])


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda adata: adata.uns["mantispy"].pop("pseudogenes"), "no record of mt.pp.pseudogenes"),
        (lambda adata: adata.obs.insert(0, "gene", adata.obs.pop("Metadata_Gene")), "groupby must include"),
        (lambda adata: adata.obs.__setitem__("Metadata_Control_Type", "treatment"), "outside reference="),
    ],
)
def test_a_pseudogene_column_without_its_record_or_reference_is_refused(change, match):
    adata = _pseudo_screen(np.random.default_rng(29), n_cells=6, n_control=8, n_control_cells=8)
    mt.pp.pseudogenes(adata, size=2)
    change(adata)
    groupby = "gene" if "gene" in adata.obs else "Metadata_Gene"
    with pytest.raises(ValueError, match=match):
        mt.tl.bootstrap_features(adata, groupby=groupby, reference="negcon", pseudogenes="pseudogene")


def test_within_perturbation_refuses_pseudogenes():
    with pytest.raises(ValueError, match="no control guides to test pseudo-genes against"):
        mt.tl.bootstrap_features(
            _two_conditions(np.random.default_rng(30)),
            groupby=["Metadata_Gene", "Metadata_Treatment"],
            control_scope="within_perturbation",
            condition="Metadata_Treatment",
            reference_condition="a",
            pseudogenes="pseudogene",
        )


def test_matched_pseudogenes_are_called_as_often_as_null_genes():
    """A screen with no phenotype, whose control guides carry six times the cells of a real guide.

    Matched in cells per guide, the pseudo-genes are called as often as the real genes; with every cell of a control
    guide, their medians are more precise than the null assumes and they are called far less often.
    """
    rng = np.random.default_rng(200)
    groups = [(f"G{g}", f"g{g}_{i}", 10, 0.0, "a") for g in range(40) for i in range(4)]
    groups += [("nontargeting", f"nt{i}", 60, 0.0, "a") for i in range(80)]
    adata = _cells(groups, rng, n_features=40, guide_sd=0.0)
    rates = {}
    for match in (True, False):
        mt.pp.pseudogenes(adata, size=4, match_cells=match)
        mt.tl.bootstrap_features(adata, pseudogenes="pseudogene")
        table = adata.uns["mantispy"]["bootstrap"]
        called = table["pvalue"] < 0.1
        rates[match] = called[~table["is_pseudogene"]].mean(), called[table["is_pseudogene"]].mean()
    real, matched = rates[True]
    unmatched = rates[False][1]
    assert abs(matched - real) < 0.025, rates
    assert unmatched < real / 2, rates


def test_empirical_fdr_takes_the_pseudogenes_as_its_null():
    adata = _pseudo_screen(np.random.default_rng(32), n_cells=10, n_control=20, n_control_cells=12)
    mt.pp.pseudogenes(adata, size=2)
    mt.tl.bootstrap_features(adata, pseudogenes="pseudogene")
    table = adata.uns["mantispy"]["bootstrap"]
    score = -np.log10(table["pvalue"].clip(lower=1e-300)).groupby(table["group"]).sum()
    genes = ad.AnnData(obs=pd.DataFrame({"gene": score.index, "score": score.to_numpy()}, index=score.index))
    labels = list(adata.obs["pseudogene"].cat.categories)
    mt.tl.empirical_fdr(genes, control_genes=labels, score="score", group="gene")
    result = genes.uns["mantispy"]["empirical_fdr"].set_index("group")
    assert result.loc[labels, "is_control"].all() and not result.loc[["GENE_A", "GENE_B"], "is_control"].any()
    assert result.loc["GENE_A", "pvalue"] == 1 / (len(labels) + 1)
