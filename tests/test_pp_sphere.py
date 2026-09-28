"""Sphering on reference sets that are too small to define a covariance."""

import numpy as np
import pytest

import mantispy as mt


def test_a_plate_with_one_control_well_raises_rather_than_zeroing_it(wells):
    """With a single reference row the rank guard `rank == n_obs - 1` reads `0 == 0`, the padded singular values are all zero, and the plate's X would come back all zero."""
    control = np.zeros(wells.n_obs, dtype=bool)
    plate = wells.obs["Metadata_Plate"].to_numpy()
    control[np.flatnonzero(plate == plate[0])[:4]] = True
    control[np.flatnonzero(plate != plate[0])[0]] = True  # a single control well on the other plate
    wells.obs["Metadata_Control"] = control

    with pytest.raises(ValueError, match="at least 2 reference rows"):
        mt.pp.sphere(wells, method="ZCA", by="Metadata_Plate")


def _batched(n_batches=3, per_batch=40, n_features=6, effect=3.0, seed=0):
    """Batches whose controls share a mean but not a covariance, which is what CORAL aligns.

    A shift alone would be removed by the centring that precedes CORAL, so each batch mixes its features differently and the perturbation pushes along one fixed direction.
    """
    import anndata as ad
    import pandas as pd

    from mantispy._core.schema import stamp

    generator = np.random.default_rng(seed)
    direction = generator.normal(size=n_features)
    blocks, frames = [], []
    for batch in range(n_batches):
        mixing = generator.normal(size=(n_features, n_features))
        values = generator.normal(size=(per_batch, n_features)) @ mixing
        treated = np.arange(per_batch) % 4 == 0
        values[treated] += effect * direction
        blocks.append(values)
        frames.append(
            pd.DataFrame(
                {
                    "Metadata_Batch": f"B{batch}",
                    "Metadata_Plate": f"P{batch}",
                    "Metadata_Well": [f"A{index + 1:02d}" for index in range(per_batch)],
                    "Metadata_Perturbation": np.where(treated, "compound", "DMSO"),
                    "Metadata_Control": ~treated,
                }
            )
        )

    obs = pd.concat(frames, ignore_index=True)
    obs.index = [str(index) for index in range(len(obs))]
    adata = ad.AnnData(X=np.vstack(blocks).astype(np.float32), obs=obs)
    adata.var_names = [f"Cells_AreaShape_F{index}" for index in range(n_features)]
    stamp(adata, resolution="well")
    return adata


def test_tvn_keeps_one_component_per_control_when_the_controls_are_few():
    """The rotation is fitted on the controls, so a control-poor screen comes back narrower than it went in.

    That is why this writes obsm: var would no longer describe the columns.
    Nine controls per batch against eighteen components is also the case the warning is for: the directions a batch's controls do not span have spread that is tiny rather than zero, so the check inside _centre_scale never fires and they are divided by it anyway.
    """
    adata = _batched(n_batches=2, per_batch=12, n_features=20)
    with pytest.warns(UserWarning, match="no more reference rows than the 18 component"):
        mt.pp.tvn(adata, use_rep=None)
    assert adata.obsm["X_tvn"].shape == (adata.n_obs, int(adata.obs["Metadata_Control"].sum()))
    assert np.isfinite(adata.obsm["X_tvn"]).all()


def test_tvn_reads_the_default_representation_and_can_leave_the_original_alone():
    """Every other test passes use_rep=None, which is not the documented default: obsm['X_pca'] is, and a copy has to come back stamped with the result rather than writing through."""
    import scanpy as sc

    adata = _batched()
    sc.pp.pca(adata, n_comps=5)

    aligned = mt.pp.tvn(adata, key_added="X_aligned", copy=True)
    assert aligned.obsm["X_aligned"].shape == (adata.n_obs, 5)
    assert "X_aligned" not in adata.obsm

    with pytest.raises(KeyError, match="Metadata_Nothing"):
        mt.pp.tvn(adata, batch_key="Metadata_Nothing")


def test_tvn_refuses_a_batch_it_cannot_estimate_a_covariance_for():
    adata = _batched(n_batches=2, per_batch=8)
    control = adata.obs["Metadata_Control"].to_numpy(dtype=bool).copy()
    control[np.flatnonzero(adata.obs["Metadata_Batch"].to_numpy() == "B1")[1:]] = False
    adata.obs["Metadata_Control"] = control

    with pytest.raises(ValueError, match="at least 2"):
        mt.pp.tvn(adata, use_rep=None)
    adata.obs["Metadata_Nothing"] = np.zeros(adata.n_obs, dtype=bool)
    with pytest.raises(ValueError, match="no reference rows"):
        mt.pp.tvn(adata, use_rep=None, reference="Metadata_Nothing")


def test_tvn_says_when_a_batch_cannot_scale_a_dimension():
    """A batch whose controls are constant in one dimension scales it by 1 and says so.

    The per-batch pass is where this bites: it is fitted on that batch's controls alone.
    """
    adata = _batched(n_batches=2, per_batch=20)
    values = np.asarray(adata.X).copy()
    control = adata.obs["Metadata_Control"].to_numpy(dtype=bool)
    batch = adata.obs["Metadata_Batch"].to_numpy() == "B1"
    # Whole wells are made identical because one constant feature would not survive the PCA as a constant component.
    values[batch & control] = values[np.flatnonzero(batch & control)[0]]
    adata.X = values

    with pytest.warns(UserWarning, match=r"no spread among the .* of batch 'B1'"):
        mt.pp.tvn(adata, use_rep=None)
    assert np.isfinite(adata.obsm["X_tvn"]).all()
    # The rotation leaves those components a spread of about 1e-16, not zero, and dividing by it gives finite 1e16s.
    assert np.abs(adata.obsm["X_tvn"]).max() < 1e3
