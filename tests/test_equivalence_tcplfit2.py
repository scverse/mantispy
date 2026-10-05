"""Equivalence of ``tl.dose_response``'s continuous hit call against tcplfit2.

``hitcall`` ports tcplfit2's ``hitcontinner``/``toplikelihood`` (``_dose._hitcall``). The tests in
``test_tl_dose.py`` pin orderings and ranges only, which would miss a wrong constant that shifts
every call in the same direction. Here the call is asserted against the number tcplfit2 0.1.9 reports
for the same input.

``hitcall`` is ``P1 * P2 * P3``. P1 is the Akaike weight against the constant model, taken over
mantispy's two models (logistic, line) rather than tcplfit2's ten, so it is not tcplfit2's P1. The
scenarios are therefore decisive curves where P1 saturates to one, which leaves the ported P2 (the
cutoff tail) and P3 (the top-likelihood profile) as what the match rests on: two independent strong
curves whose cutoff is below the fitted top so the call is mid-range (``strong_high`` 0.80,
``mid_noisy`` 0.71), a curve that clears the cutoff throughout (``clear`` 1.0), and a flat response
where the constant wins (``miss`` 0.0). A call below about 0.6 is not asserted: there the
four-parameter logistic and tcplfit2's three-parameter Hill settle on different tops and the two
implementations part by more than the tolerance. ``atol`` leaves room for that fit difference (the
mid-range calls match to under 0.01) while staying well inside the shift a wrong constant causes.

Reference values were produced once and stored inline, so the test adds no dependency on R or
tcplfit2. For each scenario, with the responses already baseline-corrected (tcplfit2's ``bmed`` 0):
``concRespCore(list(conc=conc, resp=resp, bmed=0, cutoff=cutoff, onesd=0.1),
fitmodels=c("cnst", "hill"), conthits=TRUE)$hitcall`` (``onesd`` does not enter the hit call).
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
    "strong_high": (
        np.repeat(_CONC, 4),
        np.array(
            [
                0.20,
                -0.15,
                0.08,
                -0.05,
                0.30,
                -0.05,
                0.22,
                0.03,
                0.55,
                0.20,
                0.45,
                0.28,
                0.95,
                0.62,
                0.86,
                0.70,
                1.35,
                1.05,
                1.26,
                1.10,
                1.60,
                1.30,
                1.52,
                1.34,
                1.68,
                1.40,
                1.60,
                1.44,
            ]
        ),
        1.45,
        0.800145,
    ),
    "mid_noisy": (
        np.repeat(_CONC, 4),
        np.array(
            [
                0.12,
                -0.10,
                0.05,
                -0.02,
                0.19,
                -0.03,
                0.14,
                0.02,
                0.33,
                0.12,
                0.28,
                0.16,
                0.61,
                0.40,
                0.55,
                0.44,
                0.88,
                0.66,
                0.82,
                0.70,
                1.04,
                0.84,
                0.99,
                0.86,
                1.10,
                0.90,
                1.05,
                0.92,
            ]
        ),
        0.95,
        0.711225,
    ),
    "miss": (
        np.repeat(_CONC, 3),
        np.array(
            [
                0.04,
                -0.06,
                0.02,
                -0.03,
                0.05,
                -0.01,
                0.06,
                -0.02,
                0.03,
                -0.04,
                0.02,
                0.05,
                0.01,
                -0.05,
                0.03,
                0.04,
                0.00,
                -0.03,
                -0.02,
                0.05,
                0.01,
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
    mt.tl.dose_response(adata, reference=None, cutoff=cutoff)
    table = adata.uns["mantispy"]["dose_response"].set_index("compound")

    assert table.loc[name, "hitcall"] == pytest.approx(expected, abs=0.02)
