"""Differential features at well level: calibration, blocking, and the layouts that cannot work."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy._core.schema import stamp


def test_cells_are_refused_with_the_fix_named(well_profiles):
    """Testing per cell is the mistake this function exists to prevent."""
    cells = mt.ds.synthetic_plate(n_wells=24, n_cells=20, n_features=20, seed=0)
    with pytest.raises(ValueError, match="mt.tl.aggregate"):
        mt.tl.differential_features(cells)


def test_a_group_confounded_with_its_plate_is_skipped_not_scored(well_profiles):
    """Treatment and plate become the same variable, and no test can separate them."""
    rng = np.random.default_rng(3)
    n_features = 200
    plate_shift = rng.normal(scale=1.5, size=(2, n_features))
    values = np.concatenate(
        [rng.normal(size=(8, n_features)) + plate_shift[0], rng.normal(size=(8, n_features)) + plate_shift[1]]
    )
    treated = np.arange(16) < 8
    obs = pd.DataFrame(
        {
            "Metadata_Plate": np.where(treated, "P0", "P1"),
            "Metadata_Well": [f"A{i:02d}" for i in range(16)],
            "Metadata_Perturbation": np.where(treated, "compound", "DMSO"),
            "Metadata_Control": ~treated,
        },
        index=[str(i) for i in range(16)],
    )
    adata = ad.AnnData(X=values.astype(np.float32), obs=obs)
    adata.var_names = [f"Cells_AreaShape_F{i}" for i in range(n_features)]
    stamp(adata, resolution="well")

    with pytest.raises(ValueError, match="no group had enough replicates"):
        mt.tl.differential_features(adata)

    mt.tl.differential_features(adata, block=None, key_added="unblocked")
    table = adata.uns["mantispy"]["unblocked"]
    assert (table["qvalue"] < 0.05).mean() > 0.1, "the unblocked test reports the plate as biology"


def test_the_diagnostic_catches_what_the_docstrings_claim(well_profiles):
    """The battery has to reach the right verdict on layouts built to fail each check."""
    healthy = well_profiles(n_plates=4, per_plate=16, n_features=100, seed=7)
    report = mt.metrics.diagnose_testing(healthy, n_draws=3).set_index("check")["verdict"]
    assert report["null p < 0.05"] == "pass"
    assert report["null discoveries"] == "pass"

    # Two wells per treatment: a rank test cannot reach the threshold that many tests demand, whatever the effect size.
    rng = np.random.default_rng(8)
    n_features, n_groups = 200, 12
    labels = ["DMSO"] * 24 + [f"pert{i:02d}" for i in range(n_groups) for _ in range(2)]
    obs = pd.DataFrame(
        {
            "Metadata_Plate": [f"P{i % 4}" for i in range(len(labels))],
            "Metadata_Well": [f"A{i % 24 + 1:02d}" for i in range(len(labels))],
            "Metadata_Perturbation": labels,
            "Metadata_Control": [label == "DMSO" for label in labels],
        },
        index=[str(i) for i in range(len(labels))],
    )
    thin = ad.AnnData(X=rng.normal(size=(len(labels), n_features)).astype(np.float32), obs=obs)
    thin.var_names = [f"Cells_AreaShape_F{i}" for i in range(n_features)]
    stamp(thin, resolution="well")

    thin_report = mt.metrics.diagnose_testing(thin, n_draws=3).set_index("check")["verdict"]
    assert thin_report["rank test resolution"] == "FAIL", "two wells cannot reach any FDR threshold"


def test_the_diagnostic_names_a_confounded_layout(well_profiles):
    """Treatments that share no plate with the reference fail, since no test can separate treatment from plate."""
    adata = well_profiles(n_plates=4, per_plate=8, n_features=60, seed=9)
    adata.obs["Metadata_Plate"] = np.where(adata.obs["Metadata_Control"].to_numpy(), "P2", "P0")
    report = mt.metrics.diagnose_testing(adata, n_draws=2).set_index("check")["verdict"]
    assert report["treatments sharing a Metadata_Plate with the reference"] == "FAIL"


def _null_rate_counts(report):
    """The integer count out of each ``X of N`` null-rate row, keyed by check name."""
    return {
        row["check"]: int(str(row["value"]).split()[0])
        for _, row in report.iterrows()
        if row["check"].endswith("null rate")
    }


def test_the_diagnostic_exposes_cell_within_well_pseudoreplication():
    """At cell resolution the naive cell-shuffle null inflates where the well-block null does not.

    A pure-null plate with row and column gradients and a cell-count confounder gives each well a shift
    its cells share. A null that shuffles cells ignores that shared shift and calls pure noise, while a
    null that draws whole wells does not.
    """
    plate = mt.ds.synthetic_plate(
        n_plates=2,
        n_wells=48,
        n_cells=40,
        n_features=10,
        n_perturbations=11,
        effect_size=0.0,
        row_gradient=3.0,
        col_gradient=3.0,
        confounder_effect=3.0,
        seed=0,
    )
    report = mt.metrics.diagnose_testing(plate, n_draws=8, n_permutations=30)
    assert list(report.columns) == ["check", "value", "expected", "verdict", "note"]

    verdict = report.set_index("check")["verdict"]
    counts = _null_rate_counts(report)
    for caller in ("hit_calling", "edistance"):
        well_block = counts[f"{caller} well-block null rate"]
        cell_shuffle = counts[f"{caller} cell-shuffle null rate"]
        assert verdict[f"{caller} well-block null rate"] != "FAIL", f"{caller} well-block null is not calibrated"
        assert cell_shuffle > well_block, f"{caller} cell-shuffle {cell_shuffle} not above well-block {well_block}"


def test_the_well_block_null_is_never_a_silent_cell_fallback():
    """Below eight control wells the well-block row must stay a well-block null, or the call must raise.

    hit_calling keeps its well block only with four or more reference wells and otherwise drops to a
    per-cell KS test, so with too few references left the row billed the rate to trust would report the
    anti-conservative cell rate. The pseudo-treatment is capped to keep four references; when even that is
    impossible the call raises. The existing pseudoreplication test sits at eight wells; this covers below.
    """
    from mantispy._core._stats import _default_well_block

    plate = mt.ds.synthetic_plate(
        n_plates=2,
        n_wells=25,
        n_cells=30,
        n_features=10,
        n_perturbations=11,
        effect_size=0.0,
        row_gradient=3.0,
        col_gradient=3.0,
        confounder_effect=3.0,
        seed=0,
    )
    controls = plate[plate.obs["Metadata_Control"].to_numpy()].copy()
    assert int(np.unique(_default_well_block(controls, block=None)).size) == 6, "the regime is six control wells"

    report = mt.metrics.diagnose_testing(plate, n_draws=6, n_permutations=30)
    verdict = report.set_index("check")["verdict"]
    counts = _null_rate_counts(report)
    for caller in ("hit_calling", "edistance"):
        # A silent cell fallback would report the anti-conservative cell rate here and fail the well-block row.
        assert verdict[f"{caller} well-block null rate"] != "FAIL", (
            f"{caller} well-block null looks like a cell fallback"
        )
        assert counts[f"{caller} cell-shuffle null rate"] >= counts[f"{caller} well-block null rate"]

    # Five control wells cannot leave four references and a two-well pseudo-treatment, so the call must raise.
    too_few = mt.ds.synthetic_plate(
        n_plates=1, n_wells=49, n_cells=8, n_features=10, n_perturbations=11, effect_size=0.0, seed=0
    )
    too_few_controls = too_few[too_few.obs["Metadata_Control"].to_numpy()].copy()
    assert int(np.unique(_default_well_block(too_few_controls, block=None)).size) == 5, (
        "the regime is five control wells"
    )
    with pytest.raises(ValueError, match="at least six reference wells"):
        mt.metrics.diagnose_testing(too_few, n_draws=4, n_permutations=20)


def test_an_unstamped_object_is_told_to_stamp_or_aggregate():
    """An unstamped object has complete wells, so the error must be about the stamp, not the wells."""
    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "Metadata_Plate": ["P0"] * 8,
            "Metadata_Well": [f"A{index + 1:02d}" for index in range(8)],
            "Metadata_Perturbation": ["DMSO"] * 4 + ["pert"] * 4,
            "Metadata_Control": [True] * 4 + [False] * 4,
        },
        index=[str(index) for index in range(8)],
    )
    adata = ad.AnnData(X=rng.normal(size=(8, 6)).astype(np.float32), obs=obs)
    adata.var_names = [f"Cells_AreaShape_F{index}" for index in range(6)]
    # Deliberately not stamped, though the wells are complete: a missing-well error here would mislead.
    with pytest.raises(ValueError, match="not stamped"):
        mt.metrics.diagnose_testing(adata)


def test_the_cell_shuffle_verdict_tracks_the_two_rates():
    """The verdict must fail on pseudoreplication and pass on an exchangeable screen, not always warn."""
    gradient = mt.ds.synthetic_plate(
        n_plates=2,
        n_wells=48,
        n_cells=40,
        n_features=10,
        n_perturbations=11,
        effect_size=0.0,
        row_gradient=3.0,
        col_gradient=3.0,
        confounder_effect=3.0,
        seed=0,
    )
    inflated = mt.metrics.diagnose_testing(gradient, n_draws=8, n_permutations=30).set_index("check")["verdict"]
    for caller in ("hit_calling", "edistance"):
        assert inflated[f"{caller} cell-shuffle null rate"] in {"warn", "FAIL"}, "pseudoreplication must not pass"

    # No within-well structure, so the two nulls agree and the cell-shuffle row must not spuriously warn.
    flat = mt.ds.synthetic_plate(
        n_plates=2, n_wells=48, n_cells=20, n_features=10, n_perturbations=11, effect_size=0.0, seed=1
    )
    agreeing = mt.metrics.diagnose_testing(flat, n_draws=8, n_permutations=30).set_index("check")["verdict"]
    for caller in ("hit_calling", "edistance"):
        assert agreeing[f"{caller} cell-shuffle null rate"] == "pass", "an exchangeable screen must not warn"


@pytest.mark.network
@pytest.mark.slow
def test_the_diagnostic_holds_on_a_real_pooled_screen():
    """The inflation direction holds on cp_posh's real well edge effects, not only on synthetic gradients.

    cp_posh is well-normalized single cells, so no preprocessing is needed; the controls are the
    non-targeting and intergenic guides. Both are bounded to a handful of wells and cells for speed.
    """
    adata = mt.ds.cp_posh()
    rng = np.random.default_rng(0)
    wells = adata.obs["Metadata_Well"].astype(str).to_numpy()
    control_wells = np.unique(wells[adata.obs["Metadata_Control"].to_numpy()])
    keep = np.isin(wells, rng.choice(control_wells, size=min(16, control_wells.size), replace=False))
    subset = adata[keep].copy()
    if subset.n_obs > 6000:
        subset = subset[np.sort(rng.choice(subset.n_obs, size=6000, replace=False))].copy()

    report = mt.metrics.diagnose_testing(subset, n_draws=6, n_permutations=30)
    verdict = report.set_index("check")["verdict"]
    counts = _null_rate_counts(report)
    for caller in ("hit_calling", "edistance"):
        assert verdict[f"{caller} well-block null rate"] != "FAIL"
        assert counts[f"{caller} cell-shuffle null rate"] >= counts[f"{caller} well-block null rate"]
