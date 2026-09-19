"""Chatterjee equivalence against scmorph, which is where this coefficient came from.

scmorph uses the coefficient to filter features that are redundant with each other, while
``pp.feature_select_chatterjee`` scores each feature against the perturbation. The
statistic underneath is the same on input without ties, and these tests assert that.
scmorph breaks ties in y at random, where mantispy handles them as Chatterjee does.
"""

import numpy as np
import pytest

from mantispy.pp._chatterjee import chatterjee_xi

scmorph = pytest.importorskip("scmorph")
from scmorph.pp.correlation import xim  # noqa: E402


@pytest.fixture
def paired():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    return {
        "monotonic": (x, 2 * x + rng.normal(0, 0.1, 400)),
        "v_shaped": (x, np.abs(x) + rng.normal(0, 0.1, 400)),
        "independent": (x, rng.normal(size=400)),
    }


@pytest.mark.parametrize("m", [1, 5])
def test_our_coefficient_is_scmorphs(paired, m):
    """The m-nearest-neighbour form of :cite:t:`Lin_2022`."""
    for name, (x, y) in paired.items():
        ours = float(chatterjee_xi(x, y, m=m)[0])
        theirs = float(xim(x, y, M=m)[0, 1])
        assert ours == pytest.approx(theirs, abs=1e-9), f"{name} at m={m}: {ours} vs {theirs}"


def test_more_neighbours_lower_the_noise_floor(paired):
    """Larger m keeps the limit under dependence and lowers the noise under independence."""
    x, y = paired["independent"]
    assert abs(float(chatterjee_xi(x, y, m=5)[0])) < abs(float(chatterjee_xi(x, y, m=1)[0]))

    x, y = paired["monotonic"]
    assert float(chatterjee_xi(x, y, m=5)[0]) > 0.9


def test_it_scores_every_column_in_one_pass(paired):
    """scmorph computes a pairwise matrix; mantispy only needs one column against the rest."""
    x, _ = paired["monotonic"]
    columns = np.column_stack([y for _, y in paired.values()])
    together = chatterjee_xi(x, columns, m=5)
    apart = [float(chatterjee_xi(x, columns[:, index], m=5)[0]) for index in range(columns.shape[1])]
    np.testing.assert_allclose(together, apart, rtol=1e-12)
