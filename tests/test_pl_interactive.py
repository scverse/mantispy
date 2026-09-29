"""The optional plotly twin: the runtime gate, and that a fired twin closes the static figure.

The gate is forced down each branch by monkeypatching, so these run without a real notebook.
The False branch runs everywhere; the True branch needs plotly and IPython.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mantispy as mt
from mantispy.pl import _common


def _fake_shell(name):
    """An object whose class name is ``name``, as ``get_ipython`` returns one."""
    return type(name, (), {})()


class _Capture:
    """Stands in for a plotly figure's ``show`` and keeps the figures it was handed."""

    def __init__(self):
        self.figures = []

    def __call__(self, figure):
        self.figures.append(figure)


@pytest.fixture(autouse=True)
def _clear_gate_cache():
    try:
        _common.interactive_available.cache_clear()
    except AttributeError:
        pass
    yield
    try:
        _common.interactive_available.cache_clear()
    except AttributeError:
        pass


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_gate_is_false_without_plotly(monkeypatch):
    monkeypatch.setattr(_common, "find_spec", lambda name: None)
    assert _common.interactive_available() is False


def test_gate_is_false_outside_a_notebook(monkeypatch):
    import IPython

    monkeypatch.setattr(_common, "find_spec", lambda name: object())
    monkeypatch.setattr(IPython, "get_ipython", lambda: _fake_shell("TerminalInteractiveShell"))
    monkeypatch.delenv("MANTISPY_PLOTTING_BACKEND", raising=False)
    assert _common.interactive_available() is False


def test_gate_is_false_when_the_backend_is_forced_static(monkeypatch):
    import IPython

    monkeypatch.setattr(_common, "find_spec", lambda name: object())
    monkeypatch.setattr(IPython, "get_ipython", lambda: _fake_shell("ZMQInteractiveShell"))
    monkeypatch.setenv("MANTISPY_PLOTTING_BACKEND", "static")
    assert _common.interactive_available() is False


def test_gate_is_true_in_a_notebook_with_plotly(monkeypatch):
    pytest.importorskip("plotly")
    import IPython

    monkeypatch.setattr(_common, "find_spec", lambda name: object())
    monkeypatch.setattr(IPython, "get_ipython", lambda: _fake_shell("ZMQInteractiveShell"))
    monkeypatch.delenv("MANTISPY_PLOTTING_BACKEND", raising=False)
    assert _common.interactive_available() is True


_FRAME = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0], "g": ["x", "y", "x"], "lab": ["p", "q", "r"]})

_KINDS = {
    "scatter": {"data": _FRAME, "x": "a", "y": "b", "color": "g", "hover": ["lab"]},
    "line": {"data": _FRAME, "x": "a", "y": "b", "color": "g"},
    "barh": {"data": _FRAME, "x": "a", "y": "lab", "color": "g"},
    "histogram": {"data": _FRAME, "x": "a"},
    "box": {"data": _FRAME, "x": "g", "y": "a"},
    "heatmap": {
        "matrix": np.arange(6).reshape(2, 3),
        "rows": ["r0", "r1"],
        "columns": ["c0", "c1", "c2"],
        "value_label": "v",
    },
}


@pytest.fixture
def displayed(monkeypatch):
    """Force the gate True and capture the figures a twin would show, without a real frontend."""
    pytest.importorskip("plotly")
    import plotly.graph_objects as go

    capture = _Capture()
    monkeypatch.setattr(_common, "interactive_available", lambda: True)
    monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: capture(self))
    return capture, go.Figure


@pytest.mark.parametrize("kind", list(_KINDS))
def test_each_kind_builds_and_displays_one_figure(displayed, kind):
    capture, figure_type = displayed
    assert _common.maybe_interactive(kind, **_KINDS[kind]) is True
    assert len(capture.figures) == 1
    assert isinstance(capture.figures[0], figure_type)


def test_an_unknown_kind_is_refused(displayed):
    with pytest.raises(ValueError, match="unknown interactive kind"):
        _common.maybe_interactive("violin", data=_FRAME, x="a", y="b")


@pytest.fixture(scope="module")
def scored():
    cells = mt.ds.synthetic_plate(
        n_plates=2, n_wells=96, n_cells=10, n_features=25, n_perturbations=4, effect_size=4.0, seed=0
    )
    mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon")
    wells = mt.tl.aggregate(cells, min_cells=0)
    wells.obs["Metadata_MOA"] = wells.obs["Metadata_Perturbation"].astype(str).to_numpy()
    mt.tl.hit_calling(wells, n_permutations=100)
    mt.tl.effect_size(wells)
    mt.tl.nn_moa_classify(wells, scheme="nn")
    mt.tl.replicate_saturation(wells, n_draws=2, max_replicates=3)
    return wells


_REPRESENTATIVES = {
    "scatter": lambda a: mt.pl.hits(a),
    "barh": lambda a: mt.pl.effect_sizes(a, group="pert00"),
    "heatmap": lambda a: mt.pl.moa_confusion(a),
    "line": lambda a: mt.pl.replicate_saturation(a),
    "box": lambda a: mt.pl.cell_counts(a),
}


@pytest.mark.parametrize("kind", list(_REPRESENTATIVES))
def test_a_fired_twin_closes_the_owned_static_figure(displayed, scored, kind):
    capture, figure_type = displayed
    open_before = set(plt.get_fignums())
    # These calls own their figure, so nothing is returned and the fired twin closes the static one.
    assert _REPRESENTATIVES[kind](scored) is None
    assert len(capture.figures) == 1
    assert isinstance(capture.figures[0], figure_type)
    assert set(plt.get_fignums()) == open_before, "the owned static figure must be closed so it is not shown twice"


@pytest.mark.parametrize("kind", list(_REPRESENTATIVES))
def test_the_static_plot_is_untouched_without_a_frontend(monkeypatch, scored, kind):
    monkeypatch.setattr(_common, "interactive_available", lambda: False)
    open_before = set(plt.get_fignums())
    # The plot owns its figure and returns None; without a twin the static figure stays open.
    assert _REPRESENTATIVES[kind](scored) is None
    assert len(set(plt.get_fignums()) - open_before) == 1, "the owned static figure stays open when no twin fires"
