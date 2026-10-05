"""Equivalence of ``tl.dose_response``'s continuous hit call against tcplfit2.

``hitcall`` ports tcplfit2's ``hitcontinner``/``toplikelihood`` (``_dose._hitcall``). The tests in
``test_tl_dose.py`` pin orderings and ranges only, which catches a structural mistake but not a wrong
constant. Here the call is asserted against the number tcplfit2 0.1.9 reports for the same input.

The reference values were produced once with ``tcplfit2::concRespCore`` (``fitmodels = c("cnst", "hill")``,
``conthits = TRUE``) on the baseline-corrected responses below, and are stored inline so the test adds no
dependency on R or tcplfit2. The script that made them lives outside the repo.

``hitcall`` is ``P1 * P2 * P3``. P2 (the cutoff tail) and P3 (the top-likelihood profile) are computed from
the winning fit and port directly. P1, the Akaike weight against the constant model, is taken over mantispy's
two models (logistic, line) rather than tcplfit2's ten, so the scenarios are chosen where P1 saturates and the
assertion rests on the ported P2/P3 arithmetic: a decisive curve (``clear``), a scattered curve whose top sits
near the cutoff so the call lands mid-range (``mid_noisy``), and a flat response the constant wins (``miss``).
``atol`` leaves room for mantispy's four-parameter logistic fitting a slightly different top than tcplfit2's
three-parameter Hill, while staying well inside the shift a wrong constant would cause.
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantispy as mt

_CONC = np.array([0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])

# name -> (concentrations, baseline-corrected responses, cutoff, tcplfit2 0.1.9 hitcall)
_SCENARIOS = {
    "clear": (
        _CONC,
        np.array([0.02, 0.05, 0.18, 0.52, 0.86, 0.98, 1.01]),
        0.25,
        1.000000,
    ),
    "mid_noisy": (
        np.repeat(_CONC, 4),
        np.array(
            [
                0.12, -0.10, 0.05, -0.02, 0.19, -0.03, 0.14, 0.02, 0.33, 0.12, 0.28, 0.16,
                0.61, 0.40, 0.55, 0.44, 0.88, 0.66, 0.82, 0.70, 1.04, 0.84, 0.99, 0.86,
                1.10, 0.90, 1.05, 0.92,
            ]
        ),
        0.95,
        0.711225,
    ),
    "miss": (
        np.repeat(_CONC, 3),
        np.array(
            [
                0.04, -0.06, 0.02, -0.03, 0.05, -0.01, 0.06, -0.02, 0.03, -0.04, 0.02, 0.05,
                0.01, -0.05, 0.03, 0.04, 0.00, -0.03, -0.02, 0.05, 0.01,
            ]
        ),
        0.25,
        0.000000,
    ),
}


@pytest.mark.parametrize("name", list(_SCENARIOS))
def test_hitcall_matches_tcplfit2(name):
    conc, resp, cutoff, expected = _SCENARIOS[name]
    obs = pd.DataFrame(
        {
            "Metadata_Compound": [name] * conc.size,
            "Metadata_Concentration": conc,
            "hits_row_distance": resp,
        },
        index=[str(i) for i in range(conc.size)],
    )
    adata = ad.AnnData(np.zeros((conc.size, 1)), obs=obs)

    # Responses are already baseline-corrected (tcplfit2's bmed=0), so no controls and an explicit cutoff.
    mt.tl.dose_response(adata, reference=None, cutoff=cutoff, min_doses=4)
    table = adata.uns["mantispy"]["dose_response"].set_index("compound")

    assert table.loc[name, "hitcall"] == pytest.approx(expected, abs=0.02)
