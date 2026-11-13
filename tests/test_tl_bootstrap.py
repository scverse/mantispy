"""tl.bootstrap_features: exactness, calibration, ties, missing values, control scopes and the shared layout."""

import itertools

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

    The guide null draws one control guide (probability proportional to its cell count), then draws
    as many of its cells as the guide has, with replacement, and takes the median; the gene null is
    the median over its guides of independent such draws. Two-sided, a tie counts half, no floor.
    """
    x = np.asarray(adata.X, dtype=float)
    obs = adata.obs
    is_ctrl = (obs["Metadata_Control_Type"] == "negcon").to_numpy()
    sgrna = obs["Metadata_sgRNA"].to_numpy()
    gene = obs["Metadata_Gene"].to_numpy()
    ctrl_guides = [x[(sgrna == c) & is_ctrl] for c in sorted(set(sgrna[is_ctrl]))]
    sizes = np.array([c.shape[0] for c in ctrl_guides], dtype=float)
    weights = sizes / sizes.sum()
    n_feat = x.shape[1]

    def construct_pmf(cells, feat):
        """{value: prob} of the guide null for one feature, and P(nan)."""
        m = cells.shape[0]
        pmf, nan = {}, 0.0
        for w, pool in zip(weights, ctrl_guides, strict=True):
            col = pool[:, feat]
            n = col.size
            for idx in itertools.product(range(n), repeat=m):
                draw = col[list(idx)]
                good = draw[~np.isnan(draw)]
                p = w / n**m
                if good.size == 0:
                    nan += p
                else:
                    pmf[float(np.median(good))] = pmf.get(float(np.median(good)), 0.0) + p
        return pmf, nan

    def two_sided(pmf, nan, o):
        gt = sum(p for v, p in pmf.items() if v > o)
        eq = sum(p for v, p in pmf.items() if v == o)
        one = gt + 0.5 * eq
        return 2.0 * min(one, 1.0 - one)

    guide_p, gene_p = {}, {}
    for g in sorted(set(gene[~is_ctrl])):
        constructs = sorted(set(sgrna[(gene == g)]))
        cells = {c: x[sgrna == c] for c in constructs}
        for feat in range(n_feat):
            cpmfs = [construct_pmf(cells[c], feat) for c in constructs]
            for c, (pmf, nan) in zip(constructs, cpmfs, strict=True):
                o = float(np.median(cells[c][:, feat]))
                guide_p[(g, c, feat)] = two_sided(pmf, nan, o)
            # gene null: enumerate the joint over its constructs
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
            og = float(np.median([np.median(cells[c][:, feat]) for c in constructs]))
            gene_p[(g, feat)] = two_sided(gpmf, gnan, og)
    return gene_p, guide_p


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
    gene_p, guide_p = _brute_pvalues(adata)
    table = adata.uns["mantispy"]["bootstrap"]
    features = list(adata.var_names)
    for _, row in table.iterrows():
        feat = features.index(row["feature"])
        assert abs(row["pvalue"] - gene_p[(row["group"], feat)]) < 1e-9
    guides = adata.uns["mantispy"]["bootstrap_guides"]
    for _, row in guides.iterrows():
        feat = features.index(row["feature"])
        assert abs(row["pvalue"] - guide_p[(row["group"], row["replicate"], feat)]) < 1e-9


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
    mt.tl.bootstrap_features(adata)
    row = _table(adata).loc["G"]
    assert np.isnan(row["Cells_Intensity_F0"])
    assert np.isfinite(row.drop("Cells_Intensity_F0")).all()


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
