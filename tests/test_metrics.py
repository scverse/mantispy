"""Integration metrics, including equivalence with scib where the definition is shared."""

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


def test_lisi_refuses_a_perplexity_its_neighborhoods_cannot_support(small_plate):
    """Clamping the neighborhood to n_obs let LISI collapse to the count of distinct labels: perplexity 100 and 1000 both returned 1.9991 here, the number of batches."""
    assert small_plate.n_obs == 48
    with pytest.raises(ValueError, match="perplexity"):
        mt.metrics.lisi(small_plate, key="Metadata_Batch", perplexity=100)

    # 15 fits: 3 * 15 neighbors plus the row itself is 46 of the 48 rows.
    value = _value(mt.metrics.lisi(small_plate, key="Metadata_Batch", perplexity=15), "ilisi")
    assert 1.0 <= value < small_plate.obs["Metadata_Batch"].nunique()


def test_evaluate_correction_says_what_perplexity_a_small_object_can_take(small_plate):
    """Its default perplexity of 30 needs 91 rows, so on 48 it has to say so instead of reporting a LISI that saturated at the number of batches."""
    with pytest.raises(ValueError, match="perplexity"):
        mt.metrics.evaluate_correction(small_plate)

    frame = mt.metrics.evaluate_correction(small_plate, perplexity=10)
    assert {"ilisi", "clisi"} <= set(frame["metric"])
    assert np.isfinite(frame["value"]).all()


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
