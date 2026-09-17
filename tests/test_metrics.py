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


def _labelled_blobs(separated, mixed_batches):
    """Two labels and two batches in one embedding; each flag either splits that grouping into distant blobs or leaves it in a single one."""
    import anndata as ad

    rng = np.random.default_rng(0)
    n_per_label = 12
    labels = np.repeat(["alpha", "beta"], n_per_label)
    batches = np.tile(np.repeat(["b1", "b2"], n_per_label // 2), 2)

    coords = rng.normal(scale=0.05, size=(2 * n_per_label, 2))
    if separated:
        coords[labels == "beta", 0] += 10.0
    if not mixed_batches:
        coords[batches == "b2", 1] += 10.0

    obs = pd.DataFrame(
        {"Metadata_Perturbation": labels, "Metadata_Batch": batches},
        index=[str(index) for index in range(2 * n_per_label)],
    )
    adata = ad.AnnData(X=coords.astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = coords
    return adata


def test_the_label_silhouette_scores_separated_labels_near_one():
    """Sign, not magnitude: evaluate_correction reports this as 'higher is better', so an inverted silhouette would score maximally separated labels near zero."""
    key = "Metadata_Perturbation"
    apart = _value(mt.metrics.silhouette_label(_labelled_blobs(True, True), label_key=key), "silhouette_label")
    together = _value(mt.metrics.silhouette_label(_labelled_blobs(False, True), label_key=key), "silhouette_label")

    assert apart == pytest.approx(1.0, abs=0.01)
    assert apart > together


def test_the_batch_silhouette_scores_mixed_batches_near_one():
    """Sign, not magnitude: it reports 1 - mean|silhouette|, so batches sharing one blob must score high and batches in their own blobs near zero."""
    keys = {"label_key": "Metadata_Perturbation", "batch_key": "Metadata_Batch"}
    mixed = _value(mt.metrics.silhouette_batch(_labelled_blobs(True, True), **keys), "silhouette_batch")
    split = _value(mt.metrics.silhouette_batch(_labelled_blobs(True, False), **keys), "silhouette_batch")

    assert mixed > 0.7
    assert split < 0.05


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
