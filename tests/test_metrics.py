"""Integration metrics: the definitions follow scib, which nothing here imports, so these tests pin the behaviour rather than an equivalence."""

import inspect

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import mantispy as mt
from mantispy.ds import synthetic_plate


@pytest.fixture
def corrected():
    cells = synthetic_plate(
        n_plates=4,
        n_wells=96,
        n_cells=6,
        n_features=20,
        n_batches=2,
        batch_effect=4.0,
        n_perturbations=4,
        effect_size=3.0,
        seed=0,
    )
    wells = mt.tl.aggregate(cells, min_cells=0)
    sc.pp.pca(wells, n_comps=10)
    return wells


def _value(frame, metric):
    return float(frame.loc[frame["metric"] == metric, "value"].iloc[0])


@pytest.mark.parametrize(
    "call",
    [
        lambda a: mt.metrics.silhouette_label(a, label_key="Metadata_Perturbation"),
        lambda a: mt.metrics.silhouette_batch(a, label_key="Metadata_Perturbation", batch_key="Metadata_Batch"),
        lambda a: mt.metrics.lisi(a, key="Metadata_Batch"),
        lambda a: mt.metrics.pc_regression(a, key="Metadata_Batch"),
    ],
    ids=["silhouette_label", "silhouette_batch", "lisi", "pc_regression"],
)
def test_every_metric_returns_a_tidy_row(corrected, call):
    frame = call(corrected)
    assert {"metric", "representation", "key", "value"} <= set(frame.columns)
    assert len(frame) == 1 and np.isfinite(frame["value"]).all()


def test_pc_regression_detects_the_injected_batch_effect(corrected):
    with_effect = _value(mt.metrics.pc_regression(corrected, key="Metadata_Batch"), "pc_regression")
    shuffled = corrected.copy()
    shuffled.obs["Metadata_Batch"] = np.random.default_rng(0).permutation(shuffled.obs["Metadata_Batch"].to_numpy())
    without = _value(mt.metrics.pc_regression(shuffled, key="Metadata_Batch"), "pc_regression")
    assert with_effect > without


def test_correction_moves_the_batch_metric_the_right_way(corrected):
    """Centring each batch is the simplest correction, and the batch metric must register it."""
    before = _value(mt.metrics.pc_regression(corrected, key="Metadata_Batch", use_rep="X_pca"), "pc_regression")

    centred = corrected.copy()
    mt.pp.normalize(centred, method="standardize", by="Metadata_Batch")
    sc.pp.pca(centred, n_comps=10)
    corrected.obsm["X_centred"] = centred.obsm["X_pca"]

    after = _value(mt.metrics.pc_regression(corrected, key="Metadata_Batch", use_rep="X_centred"), "pc_regression")
    assert after < before


def test_evaluate_correction_stacks_and_names_both_lisis(corrected):
    """Two rows both called 'lisi' would collide when the table is pivoted."""
    frame = mt.metrics.evaluate_correction(corrected, reps=("X_pca",))
    assert {"ilisi", "clisi"} <= set(frame["metric"])
    assert frame["metric"].is_unique
    assert set(frame["better"]) <= {"higher", "lower"}


def test_evaluate_correction_compares_representations_and_names_the_map_row_honestly(corrected):
    """Reading one uns table once per representation gave every representation the same mAP, 0.7757 under both X_pca and X_other, inside the function whose purpose is comparing them."""
    pytest.importorskip("copairs")  # copairs declares requires-python <3.13
    corrected.obsm["X_other"] = np.asarray(corrected.obsm["X_pca"])[:, :5]
    mt.tl.map(corrected, mode="activity", null_size=200)  # scores X, neither representation

    frame = mt.metrics.evaluate_correction(corrected, reps=("X_pca", "X_other"), map_key="map")
    assert {"X_pca", "X_other"} <= set(frame["representation"])

    rows = frame[frame["metric"] == "mean_average_precision"]
    assert len(rows) == 1
    assert rows["representation"].iloc[0] == "X"


@pytest.fixture
def small_plate():
    """A two-batch plate of 48 wells, fewer than the 91 rows the default perplexity of 30 needs."""
    cells = synthetic_plate(
        n_plates=2,
        n_wells=24,
        n_cells=15,
        n_features=20,
        n_batches=2,
        batch_effect=3.0,
        n_perturbations=3,
        effect_size=3.0,
        seed=0,
    )
    wells = mt.tl.aggregate(cells, min_cells=0)
    sc.pp.pca(wells, n_comps=10)
    return wells


def test_lisi_warns_and_reports_nan_when_the_neighborhoods_cannot_support_the_perplexity(small_plate):
    """Clamping the neighborhood to n_obs let LISI collapse to the count of distinct labels: perplexity 100 and 1000 both returned 1.9991 here, the number of batches.

    Raising instead took every other metric down with it, so the undefined case warns and reports NaN, as the silhouettes do.
    """
    assert small_plate.n_obs == 48
    with pytest.warns(UserWarning, match="perplexity"):
        frame = mt.metrics.lisi(small_plate, key="Metadata_Batch", perplexity=100)
    assert np.isnan(frame["value"].iloc[0])
    assert frame["metric"].iloc[0] == "ilisi"  # named even when undefined, or the row cannot stack

    # A 48-well plate is an ordinary input, and the default perplexity of 30 needs 91 rows.
    with pytest.warns(UserWarning, match="48-well plate"):
        assert np.isnan(mt.metrics.lisi(small_plate, key="Metadata_Plate")["value"].iloc[0])

    # 15 fits: 3 * 15 neighbors plus the row itself is 46 of the 48 rows.
    value = _value(mt.metrics.lisi(small_plate, key="Metadata_Batch", perplexity=15), "ilisi")
    assert 1.0 <= value < small_plate.obs["Metadata_Batch"].nunique()


def test_evaluate_correction_reports_nan_for_a_metric_it_cannot_compute(small_plate):
    """One undefined metric used to abort the whole call: the default perplexity of 30 needs 91 rows, so on a 48-well plate LISI raised and the caller got no table at all rather than one suspect row."""
    with pytest.warns(UserWarning, match="perplexity"):
        frame = mt.metrics.evaluate_correction(small_plate)

    undefined = frame[frame["metric"].isin(("ilisi", "clisi"))]
    assert len(undefined) == 2 and undefined["value"].isna().all()

    measured = frame[~frame["metric"].isin(("ilisi", "clisi"))]
    assert len(measured) == 3 and np.isfinite(measured["value"]).all()
    assert set(frame["better"]) <= {"higher", "lower"}

    frame = mt.metrics.evaluate_correction(small_plate, perplexity=10)
    assert {"ilisi", "clisi"} <= set(frame["metric"])
    assert np.isfinite(frame["value"]).all()


def test_evaluate_correction_takes_every_option_by_keyword():
    """``perplexity`` was added ahead of the pre-existing ``map_key``, so a fifth positional argument silently became a perplexity; keyword-only parameters make that unrepresentable."""
    parameters = inspect.signature(mt.metrics.evaluate_correction).parameters
    assert [name for name, p in parameters.items() if p.kind is p.POSITIONAL_OR_KEYWORD] == ["adata"]
    assert {name for name, p in parameters.items() if p.kind is p.KEYWORD_ONLY} == {
        "reps",
        "label_key",
        "batch_key",
        "covariates",
        "map_key",
        "perplexity",
    }

    # The call fails on arity before the body runs, so no object is needed to provoke it.
    with pytest.raises(TypeError, match="positional"):
        mt.metrics.evaluate_correction(None, ("X_pca",), "Metadata_Perturbation", "Metadata_Batch", "map")


def test_evaluate_correction_survives_an_object_where_most_metrics_are_undefined():
    """A consensus object holds one row per perturbation, where the label silhouette, batch mixing and both LISIs are undefined at once, and the table still has to come back with whatever can be measured."""
    import anndata as ad

    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "Metadata_Perturbation": ["a", "b", "c", "d"],
            "Metadata_Batch": ["B1", "B2", "B1", "B2"],
        },
        index=[str(index) for index in range(4)],
    )
    adata = ad.AnnData(X=rng.normal(size=(4, 5)).astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = rng.normal(size=(4, 3))

    with pytest.warns(UserWarning):
        frame = mt.metrics.evaluate_correction(adata)

    values = frame.set_index("metric")["value"]
    assert values[["silhouette_label", "silhouette_batch", "ilisi", "clisi"]].isna().all()
    assert np.isfinite(values["pc_regression"])


def test_the_label_silhouette_says_why_it_cannot_score_one_row_per_label(small_plate):
    """A consensus object holds one row per perturbation, where the label silhouette is undefined, and sklearn's own error names neither the column nor the cause."""
    consensus = mt.tl.consensus(small_plate)
    assert consensus.n_obs == consensus.obs["Metadata_Perturbation"].nunique()
    sc.pp.pca(consensus, n_comps=3)

    with pytest.warns(UserWarning, match="row per label"):
        frame = mt.metrics.silhouette_label(consensus, label_key="Metadata_Perturbation")
    assert np.isnan(frame["value"].iloc[0])


def test_batch_variance_explained_covers_every_key(corrected):
    frame = mt.metrics.batch_variance_explained(corrected, keys=["Metadata_Batch", "Metadata_Plate"])
    assert set(frame["key"]) == {"Metadata_Batch", "Metadata_Plate"}


def test_missing_representation_says_how_to_make_one(corrected):
    with pytest.raises(KeyError, match="sc.pp.pca"):
        mt.metrics.silhouette_label(corrected, label_key="Metadata_Perturbation", use_rep="X_nope")


def test_silhouette_batch_skips_labels_where_mixing_is_undefined():
    """One well per plate on three plates is three points in three groups, which sklearn
    refuses. That label is skipped and the others are still scored."""
    import anndata as ad

    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "Metadata_Compound": ["a", "a", "a", "b", "b", "b", "b"],
            "Metadata_Plate": ["P1", "P2", "P3", "P1", "P1", "P2", "P2"],
        },
        index=[str(index) for index in range(7)],
    )
    adata = ad.AnnData(X=rng.normal(size=(7, 4)).astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = rng.normal(size=(7, 3))

    result = mt.metrics.silhouette_batch(adata, label_key="Metadata_Compound", batch_key="Metadata_Plate")
    assert np.isfinite(result["value"].iloc[0])  # 'b' is usable; 'a' is skipped


def _labelled_blobs(labels=False, batches=False):
    """Two labels and two batches in one embedding; each flag splits that grouping into distant blobs instead of leaving it in a single one."""
    import anndata as ad

    rng = np.random.default_rng(0)
    n_per_label = 12
    n_obs = 2 * n_per_label
    label_of = np.repeat(["alpha", "beta"], n_per_label)
    batch_of = np.tile(np.repeat(["b1", "b2"], n_per_label // 2), 2)

    coords = rng.normal(scale=0.05, size=(n_obs, 2))
    if labels:
        coords[label_of == "beta", 0] += 10.0
    if batches:
        coords[batch_of == "b2", 1] += 10.0

    obs = pd.DataFrame(
        {"Metadata_Perturbation": label_of, "Metadata_Batch": batch_of},
        index=[str(index) for index in range(n_obs)],
    )
    adata = ad.AnnData(X=coords.astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = coords
    return adata


def test_the_label_silhouette_scores_separated_labels_near_one():
    """Sign, not magnitude: evaluate_correction reports this as 'higher is better', so an inverted silhouette would score maximally separated labels near zero."""
    key = "Metadata_Perturbation"
    apart = _value(mt.metrics.silhouette_label(_labelled_blobs(labels=True), label_key=key), "silhouette_label")
    together = _value(mt.metrics.silhouette_label(_labelled_blobs(), label_key=key), "silhouette_label")

    assert apart == pytest.approx(1.0, abs=0.01)
    assert apart > together


def test_the_batch_silhouette_puts_mixed_batches_above_split_ones():
    """Sign, not magnitude: it reports 1 - mean|silhouette|, so batches sharing one blob must land above the midpoint of its [0, 1] range and batches in their own blobs below it.

    The midpoint is the one bound that means something here, and a direction flip crosses it in both directions rather than merely shifting the value.
    """
    keys = {"label_key": "Metadata_Perturbation", "batch_key": "Metadata_Batch"}
    mixed = _value(mt.metrics.silhouette_batch(_labelled_blobs(labels=True), **keys), "silhouette_batch")
    split = _value(mt.metrics.silhouette_batch(_labelled_blobs(labels=True, batches=True), **keys), "silhouette_batch")

    assert mixed > 0.5 > split


def test_diagnose_testing_measures_the_hit_callers_on_this_screen(pure_noise_screen):
    """Both hit callers are only approximately calibrated, to a degree that depends on the
    control count, so diagnose_testing measures them on the screen at hand."""
    adata = pure_noise_screen(n_control=200, n_groups=6, per_group=8, n_features=8)
    report = mt.metrics.diagnose_testing(adata, n_draws=3, n_permutations=200)

    checks = set(report["check"])
    assert {"hit_calling null rate", "edistance null rate"} <= checks

    # Smoke test: on pure noise the battery must pass both hit callers. Both call 0 of 3
    # here, which passes under a binomial and a fixed-rate cutoff alike; the test below
    # checks the cutoff.
    verdicts = report.set_index("check")
    for name in ("hit_calling null rate", "edistance null rate"):
        assert verdicts.loc[name, "verdict"] == "pass", report


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "pass"),
        (1, "pass"),  # rate 0.125, which a fixed 0.10 threshold would fail
        (2, "pass"),  # rate 0.250; 2 is the binomial cutoff at eight draws
        (3, "warn"),
        (4, "FAIL"),
    ],
)
def test_the_null_rate_verdict_is_a_binomial_tail(monkeypatch, pure_noise_screen, count, expected):
    """The null-rate verdict uses a binomial tail, checked at counts where a fixed rate disagrees.

    A calibrated test calls a pseudo-treatment with probability ``alpha``, so the count over
    ``n_draws`` is Binomial(n_draws, alpha) and the cutoff is that distribution's upper tail
    (2 at eight draws). A fixed 0.10 rate fails counts of 1 and 2, which calibrated screens
    produce often: that rule fails 34% of them at eight draws and 87% at forty.

    The count is stubbed rather than simulated so the assertion does not depend on a random
    draw.
    """
    from mantispy.metrics import _diagnose

    monkeypatch.setattr(
        _diagnose,
        "_empirical_hit_rate",
        lambda *args, **kwargs: {"hit_calling": count, "edistance": count},
    )
    adata = pure_noise_screen(n_control=40, n_groups=3, per_group=6, n_features=6)
    report = mt.metrics.diagnose_testing(adata, n_draws=8, n_permutations=50)

    verdicts = report.set_index("check")
    for name in ("hit_calling null rate", "edistance null rate"):
        assert verdicts.loc[name, "verdict"] == expected, report


def _gene_map(n_sets=4, per_set=3, n_background=24, n_features=16, noise=0.1, seed=0):
    """One profile per gene, where the genes of a set share a direction and the rest are noise."""
    import anndata as ad

    generator = np.random.default_rng(seed)
    names, rows, edges = [], [], []
    for index in range(n_sets):
        direction = generator.normal(size=n_features)
        for member in range(per_set):
            names.append(f"SET{index}G{member}")
            rows.append(direction + noise * generator.normal(size=n_features))
            edges.append({"source": f"complex{index}", "target": names[-1]})
    for index in range(n_background):
        names.append(f"BG{index}")
        rows.append(generator.normal(size=n_features))

    obs = pd.DataFrame({"Metadata_Perturbation": names}, index=pd.Index(names, name="gene"))
    return ad.AnnData(np.asarray(rows, dtype=np.float32), obs=obs), pd.DataFrame(edges)


def _recall_by_hand(adata, net, percentile=5.0):
    """The same quantity from the definition, pair by pair, as an independent check.

    This is `EFAAR_benchmarking` at 2935f21: the comparison distribution is the strict upper
    triangle, a query value counts when its rank fraction among the null is at or below the
    lower threshold or at or above the upper one, and the ranks come from searchsorted rather
    than from an interpolated quantile.
    """
    values = np.asarray(adata.X, dtype=np.float64)
    unit = values / np.linalg.norm(values, axis=1, keepdims=True)
    genes = list(adata.obs["Metadata_Perturbation"])
    position = {gene: index for index, gene in enumerate(genes)}

    background = np.sort([unit[i] @ unit[j] for i in range(len(genes)) for j in range(i + 1, len(genes))])

    related = set()
    for _, block in net.groupby("source"):
        members = sorted({gene for gene in block["target"] if gene in position})
        related |= {
            (position[members[a]], position[members[b]])
            for a in range(len(members))
            for b in range(a + 1, len(members))
        }
    scores = np.array([unit[i] @ unit[j] for i, j in sorted(related)])
    below = np.searchsorted(background, scores, side="right") / len(background)
    above = np.searchsorted(background, scores, side="left") / len(background)
    return float(np.mean((below <= percentile / 100) | (above >= 1 - percentile / 100)))


def test_known_relationships_separates_a_structured_map_from_a_shuffled_annotation():
    """Recall is near 1 when the annotated genes share a direction, and near the 2 x percentile
    baseline when the same number of pairs is drawn at random."""
    adata, net = _gene_map()
    recall = _value(mt.metrics.known_relationships(adata, net), "known_relationships")

    shuffled = net.copy()
    shuffled["target"] = np.random.default_rng(1).permutation(adata.obs["Metadata_Perturbation"].to_numpy())[: len(net)]
    baseline = _value(mt.metrics.known_relationships(adata, shuffled), "known_relationships")

    assert recall > 0.9
    assert baseline < 0.4, baseline


def test_known_relationships_measures_chance_for_a_perturbation_in_many_sets():
    """A compound annotated to many targets draws many of the pairs. This one points away from every other
    profile, so its pairs sit in a tail whatever the annotation says: the recall is far above 2 x percentile,
    and only a shuffle that keeps how many sets each perturbation belongs to shows that it is chance."""
    import anndata as ad

    values = 3.0 + np.random.default_rng(0).normal(size=(60, 16))
    values[0] *= -1
    names = [f"G{index}" for index in range(60)]
    adata = ad.AnnData(values.astype(np.float32), obs=pd.DataFrame({"Metadata_Perturbation": names}, index=names))
    hub = [(f"hub{k}", member) for k in range(1, 20) for member in ("G0", f"G{k}")]
    rest = [(f"pair{k}", f"G{member}") for k in range(20) for member in (20 + 2 * k, 21 + 2 * k)]
    net = pd.DataFrame(hub + rest, columns=["source", "target"])

    result = mt.metrics.known_relationships(adata, net, n_permutations=200, seed=0).iloc[0]

    assert result["value"] > 0.3
    assert result["null"] > 0.3
    assert result["p_value"] > 0.05


def test_known_relationships_counts_the_lower_tail():
    """Two perturbations with opposite effects are related, so the recall is two-sided.

    A one-sided implementation scores this pair 0: its cosine is -1, the least similar pair
    in the map.
    """
    import anndata as ad

    generator = np.random.default_rng(2)
    direction = generator.normal(size=16)
    values = np.vstack([generator.normal(size=(30, 16)), direction, -direction]).astype(np.float32)
    names = [f"BG{index}" for index in range(30)] + ["OPP0", "OPP1"]
    adata = ad.AnnData(values, obs=pd.DataFrame({"Metadata_Perturbation": names}, index=pd.Index(names, name="gene")))

    net = pd.DataFrame([{"source": "opposing", "target": "OPP0"}, {"source": "opposing", "target": "OPP1"}])
    assert _value(mt.metrics.known_relationships(adata, net), "known_relationships") == 1.0


def test_known_relationships_matches_the_definition():
    adata, net = _gene_map(seed=3)
    measured = _value(mt.metrics.known_relationships(adata, net), "known_relationships")
    assert measured == pytest.approx(_recall_by_hand(adata, net), abs=1e-12)


def test_a_set_expands_into_every_pair_within_it():
    """Sets are expanded by arithmetic over the whole annotation rather than one group at a time, so the
    expansion has to hold for a set larger than a pair, for a member listed twice, and for one not profiled."""
    from mantispy.metrics._relationships import _pairs_from_sets

    net = pd.DataFrame(
        {
            "source": ["big", "big", "big", "big", "pair", "pair", "lonely", "outside"],
            "target": ["a", "b", "c", "b", "a", "d", "a", "gone"],
        }
    )
    pairs = _pairs_from_sets(net, {"a": 0, "b": 1, "c": 2, "d": 3})

    assert {tuple(row) for row in pairs} == {(0, 1), (0, 2), (1, 2), (0, 3)}
    assert (pairs[:, 0] < pairs[:, 1]).all()


def test_known_relationships_refuses_input_it_cannot_score():
    adata, net = _gene_map(seed=4)
    with pytest.raises(ValueError, match="aggregate first"):
        mt.metrics.known_relationships(adata[[0, 0, 1]].copy(), net)
    with pytest.raises(ValueError, match="relates no two"):
        mt.metrics.known_relationships(adata, net.assign(target="ABSENT" + net["target"]))
    with pytest.raises(KeyError, match="Metadata_Missing"):
        mt.metrics.known_relationships(adata, net, label_key="Metadata_Missing")
    with pytest.raises(ValueError, match=r"must be in \(0, 50\)"):
        mt.metrics.known_relationships(adata, net, percentile=150)


def test_known_relationships_names_each_annotation_source():
    """Every source is scored on its own, so stacking two unnamed rows gave a table that
    pl.metrics could not pivot: 'Index contains duplicate entries'."""
    adata, net = _gene_map(seed=7)
    rows = pd.concat(
        [
            mt.metrics.known_relationships(adata, net, name="corum"),
            mt.metrics.known_relationships(adata, net, name="reactome"),
        ]
    )

    assert list(rows["metric"]) == ["known_relationships:corum", "known_relationships:reactome"]
    mt.pl.metrics(rows)


def test_known_relationships_caps_what_one_set_expands_into(monkeypatch):
    """A set of n members is n(n-1)/2 pairs, so one set naming every gene in a genome would both
    dominate the recall and exhaust memory. The message has to name the way out."""
    adata, _ = _gene_map(seed=6)
    genes = adata.obs["Metadata_Perturbation"].to_numpy()
    everything = pd.DataFrame({"source": "all", "target": genes})

    monkeypatch.setattr(mt.metrics._relationships, "MAX_PAIRS", 5)
    with pytest.raises(ValueError, match="drop the largest ones"):
        mt.metrics.known_relationships(adata, everything)


def test_known_relationships_scores_a_map_that_says_nothing_at_zero():
    """Every profile identical makes every pair tie at a cosine of 1, so no pair ranks in a tail.
    Reading the tails off interpolated quantiles instead would call all of them extreme."""
    import anndata as ad

    genes = [f"GENE{index}" for index in range(8)]
    adata = ad.AnnData(
        np.ones((8, 4), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Perturbation": genes}, index=genes),
    )
    net = pd.DataFrame({"source": "complex", "target": genes[:3]})

    assert _value(mt.metrics.known_relationships(adata, net), "known_relationships") == 0.0


@pytest.mark.parametrize(("percentile", "expected"), [(10.0, 0.0), (20.0, 1.0)])
def test_known_relationships_ranks_ties_as_the_reference_does(percentile, expected):
    """The pinned vector from `EFAAR_benchmarking` at 2935f21, tests/test_benchmarking.py.

    The two query values are the smallest and largest of a five-value null, so an implementation
    that read interpolated quantiles would call both of them extreme at any threshold. Ranking
    them by position instead puts each at a fifth of the distribution, which the 10% tails do
    not reach and the 20% tails do.
    """
    from mantispy.metrics._relationships import _recall

    assert _recall(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([1.0, 5.0]), percentile / 100) == expected


def test_known_relationships_takes_a_pair_list_once_it_is_reshaped():
    """The reference relationship sets ship one pair per row. There is one annotation shape, so
    the conversion is the caller's, and it has to give the same answer as the sets do."""
    adata, _ = _gene_map(seed=5)
    genes = adata.obs["Metadata_Perturbation"].to_numpy()
    # Written both ways round, so the direction a pair appears in cannot change the answer.
    pairs = pd.DataFrame(
        [
            {"entity1": genes[3 * index + a], "entity2": genes[3 * index + b]}
            for index in range(4)
            for a, b in ((0, 1), (1, 0))
        ]
    )
    reshaped = pairs.assign(source=pairs.index.astype(str)).melt(id_vars="source", value_name="target")[
        ["source", "target"]
    ]
    sets = pd.DataFrame(
        [{"source": f"complex{index}", "target": genes[3 * index + member]} for index in range(4) for member in (0, 1)]
    )

    assert _value(mt.metrics.known_relationships(adata, reshaped), "known_relationships") == _value(
        mt.metrics.known_relationships(adata, sets), "known_relationships"
    )

    with pytest.raises(ValueError, match="net needs"):
        mt.metrics.known_relationships(adata, pairs)


def test_evaluate_correction_reports_a_covariate_nothing_else_would_catch(corrected):
    """A representation can be dominated by something that is neither the batch nor the label.
    On the learned embeddings of `ds.jump_lite` the cell count explains several times more of the
    variance than the source does, and no other row of this table would say so."""
    generator = np.random.default_rng(0)
    embedding = np.asarray(corrected.obsm["X_pca"]).copy()
    # A covariate written straight into the first component, and one that is pure noise.
    corrected.obs["Metadata_CellCount"] = embedding[:, 0] * 10 + generator.normal(scale=0.01, size=corrected.n_obs)
    corrected.obs["Metadata_Unrelated"] = generator.normal(size=corrected.n_obs)

    frame = mt.metrics.evaluate_correction(
        corrected, reps=("X_pca",), covariates=("Metadata_CellCount", "Metadata_Unrelated"), perplexity=10
    )

    assert frame["metric"].is_unique  # a second row called "pc_regression" would collide
    dominant = _value(frame, "pc_regression:Metadata_CellCount")
    unrelated = _value(frame, "pc_regression:Metadata_Unrelated")
    batch = _value(frame, "pc_regression")
    assert dominant > batch and dominant > 0.5
    assert unrelated < 0.2

    # Whether a small share is better depends on what the covariate is, so no direction is claimed.
    covariate_rows = frame[frame["metric"].str.startswith("pc_regression:")]
    assert covariate_rows["better"].isna().all()
    assert not frame[~frame["metric"].str.startswith("pc_regression:")]["better"].isna().any()
