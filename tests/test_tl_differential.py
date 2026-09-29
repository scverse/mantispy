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
