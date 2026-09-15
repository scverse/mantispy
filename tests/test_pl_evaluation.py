"""The 0.3 evaluation plots."""

import matplotlib
import numpy as np
import pytest
import scanpy as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mantispy as mt
from mantispy.ds import synthetic_plate


@pytest.fixture
def evaluated():
    pytest.importorskip("copairs")  # tl.map below; copairs declares requires-python <3.13
    cells = synthetic_plate(
        n_plates=2,
        n_wells=96,
        n_cells=8,
        n_features=20,
        n_batches=2,
        batch_effect=2.0,
        n_perturbations=5,
        effect_size=4.0,
        seed=0,
    )
    mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon")
    wells = mt.tl.aggregate(cells, min_cells=0)
    sc.pp.pca(wells, n_comps=10)
    mt.tl.map(wells, mode="activity", null_size=200)
    mt.tl.percent_replicating(wells, null_size=200)
    mt.tl.similarity(wells)
    return wells


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.mark.parametrize(
    "draw",
    [
        lambda a: mt.pl.map(a),
        lambda a: mt.pl.replicate_correlation(a),
        lambda a: mt.pl.batch_variance(a, keys=["Metadata_Batch", "Metadata_Plate"]),
        lambda a: mt.pl.similarity(a),
        lambda a: mt.pl.metrics(mt.metrics.evaluate_correction(a, reps=("X_pca",))),
    ],
    ids=["map", "replicate_correlation", "batch_variance", "similarity", "metrics"],
)
def test_every_plot_draws_and_does_not_mutate(evaluated, draw):
    before = (list(evaluated.obs.columns), evaluated.X.copy())
    result = np.asarray(draw(evaluated))
    assert all(isinstance(axis, matplotlib.axes.Axes) for axis in result.ravel())
    assert list(evaluated.obs.columns) == before[0]
    np.testing.assert_array_equal(evaluated.X, before[1])


def test_map_plot_draws_the_threshold_actually_used(evaluated):
    ax = mt.pl.map(evaluated)
    lines = [line.get_ydata()[0] for line in ax.get_lines()]
    assert any(abs(y - -np.log10(0.05)) < 1e-9 for y in lines)


def test_metrics_plot_marks_which_direction_is_better(evaluated):
    table = mt.metrics.evaluate_correction(evaluated, reps=("X_pca",))
    ax = mt.pl.metrics(table)
    assert any("better" in label.get_text() for label in ax.get_xticklabels())


def test_similarity_subsamples_a_large_object(evaluated):
    ax = mt.pl.similarity(evaluated, max_obs=50)
    assert ax.images[0].get_array().shape == (50, 50)


def test_plots_say_what_to_run_first():
    fresh = mt.tl.aggregate(synthetic_plate(n_wells=8, n_cells=4, n_features=8, seed=0), min_cells=0)
    with pytest.raises(KeyError, match="mt.tl.map"):
        mt.pl.map(fresh)
    with pytest.raises(KeyError, match="mt.tl.similarity"):
        mt.pl.similarity(fresh)


def test_similarity_keeps_the_group_ordering_when_it_subsamples():
    """Subsampling must keep rows in group order. Sorting sampled row indices numerically
    would scatter the groups and hide the blocks the plot shows."""
    profiles = mt.tl.aggregate(
        mt.ds.synthetic_plate(n_plates=2, n_wells=96, n_cells=4, n_features=10, n_perturbations=3, seed=0),
        min_cells=0,
    )
    mt.tl.similarity(profiles, metric="cosine")
    ax = mt.pl.similarity(profiles, max_obs=50)
    assert isinstance(ax, matplotlib.axes.Axes)

    # Reproduce the plot's ordering and check that it is grouped.
    order = np.argsort(profiles.obs["Metadata_Perturbation"].astype(str).to_numpy(), kind="stable")
    picked = np.sort(np.random.default_rng(0).choice(order.size, size=50, replace=False))
    labels = profiles.obs["Metadata_Perturbation"].astype(str).to_numpy()[order[picked]]
    assert list(labels) == sorted(labels)
