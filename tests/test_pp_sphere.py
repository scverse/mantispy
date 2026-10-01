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


def _plantable(seed=0, n_groups=8, per_group=6, n_control=48):
    """A screen where replicate structure lives in a few directions buried under nuisance and noise.

    The signal separating the perturbation groups is a small, moderate-variance subspace.
    It is drowned in the raw profiles by several high-variance nuisance directions that no group owns, and whitening with a tiny epsilon brings a stack of near-degenerate pure-noise directions up to unit weight, so both the fixed default and the grid edges score worse than an interior candidate that suppresses the degenerate noise while equalising the nuisance.
    """
    import anndata as ad
    import pandas as pd

    from mantispy._core.schema import stamp

    rng = np.random.default_rng(seed)
    n_signal, n_nuisance, n_degenerate = 2, 3, 8
    n_features = n_signal + n_nuisance + n_degenerate
    n = n_control + n_groups * per_group

    values = np.zeros((n, n_features))
    # Signal: controls sit at the origin with unit spread; each group sits at its own well-separated centre.
    centres = rng.normal(size=(n_groups, n_signal)) * 3.0
    values[:n_control, :n_signal] = rng.normal(size=(n_control, n_signal))
    labels = ["DMSO"] * n_control
    row = n_control
    for group in range(n_groups):
        values[row : row + per_group, :n_signal] = centres[group] + rng.normal(scale=0.3, size=(per_group, n_signal))
        labels += [f"p{group:02d}"] * per_group
        row += per_group
    # Nuisance: high variance shared by every well, so it dominates the raw cosine geometry.
    values[:, n_signal : n_signal + n_nuisance] = rng.normal(scale=15.0, size=(n, n_nuisance))
    # Degenerate: tiny variance pure noise; harmless until whitening with a tiny epsilon normalises it up.
    values[:, n_signal + n_nuisance :] = rng.normal(scale=0.1, size=(n, n_degenerate))

    obs = pd.DataFrame(
        {
            "Metadata_Plate": "P1",
            "Metadata_Well": [f"A{index + 1:04d}" for index in range(n)],
            "Metadata_Perturbation": labels,
            "Metadata_Control": [label == "DMSO" for label in labels],
        },
        index=[str(index) for index in range(n)],
    )
    adata = ad.AnnData(X=values.astype(np.float32), obs=obs)
    adata.var_names = [f"Cells_AreaShape_F{index}" for index in range(n_features)]
    stamp(adata, resolution="well")
    return adata


def _combined_map(adata):
    """The combined activity/replicability mAP a float epsilon produces, scored the public way."""
    import mantispy as mt

    activity = mt.tl.map(adata, mode="activity", null_size=100, seed=0, copy=True)
    replicability = mt.tl.map(adata, mode="replicability", null_size=100, seed=0, copy=True)
    return (
        activity.uns["mantispy"]["map"]["mean_average_precision"].mean()
        + replicability.uns["mantispy"]["map"]["mean_average_precision"].mean()
    ) / 2


def test_auto_grid_is_the_recipe_grid_and_seed_deterministic():
    """The sweep uses the recipe's fixed log-uniform grid, reproduced from the same seed."""
    pytest.importorskip("copairs")
    adata = _plantable()
    mt.pp.sphere(adata, method="ZCA", epsilon="auto")

    grid = np.asarray(adata.uns["mantispy"]["sphere_epsilon"]["grid"])
    expected = 10.0 ** np.random.default_rng((6, 12, 2022)).uniform(-5.0, 3.0, 25)
    assert grid.shape == (25,)
    np.testing.assert_allclose(grid, expected)
    assert grid[0] == expected[0]
    assert grid[-1] == expected[-1]


def test_auto_records_chosen_value_grid_and_scores():
    """uns holds the chosen float alongside every candidate and its score."""
    pytest.importorskip("copairs")
    adata = _plantable()
    mt.pp.sphere(adata, method="ZCA", epsilon="auto")

    info = adata.uns["mantispy"]["sphere_epsilon"]
    grid = np.asarray(info["grid"])
    scores = np.asarray(info["scores"])
    assert grid.shape == (25,)
    assert scores.shape == (25,)
    assert isinstance(info["epsilon"], float)
    assert info["epsilon"] == pytest.approx(grid[int(np.argmax(scores))])
    assert np.isfinite(adata.X).all()


def test_auto_beats_the_fixed_default_and_picks_an_interior_candidate():
    """The planted screen has an interior optimum, so the sweep beats both grid edges and the below-grid default."""
    pytest.importorskip("copairs")
    adata = _plantable()
    auto = mt.pp.sphere(adata, method="ZCA", epsilon="auto", copy=True)

    info = auto.uns["mantispy"]["sphere_epsilon"]
    scores = np.asarray(info["scores"])
    grid = np.asarray(info["grid"])
    best = int(np.argmax(scores))

    # The grid is in sample order, not sorted, so "edge" means the extreme epsilon value, not the end index.
    assert grid.min() < grid[best] < grid.max()
    assert scores[best] > scores[int(np.argmin(grid))]
    assert scores[best] > scores[int(np.argmax(grid))]

    fixed = mt.pp.sphere(adata, method="ZCA", epsilon=1e-6, copy=True)
    assert scores[best] > _combined_map(fixed)


def test_auto_without_the_columns_or_reference_it_needs_names_the_gap():
    """Asking for auto without the replicate grouping or a negcon reference raises, naming each gap."""
    pytest.importorskip("copairs")
    adata = _plantable()

    missing_label = adata.copy()
    del missing_label.obs["Metadata_Perturbation"]
    with pytest.raises(ValueError, match="Metadata_Perturbation"):
        mt.pp.sphere(missing_label, method="ZCA", epsilon="auto")

    missing_reference = adata.copy()
    del missing_reference.obs["Metadata_Control"]
    with pytest.raises(ValueError, match="Metadata_Control"):
        mt.pp.sphere(missing_reference, method="ZCA", epsilon="auto")


def test_auto_without_replicate_pairs_raises():
    """A screen where every perturbation is a singleton cannot be scored, so the guard fires."""
    pytest.importorskip("copairs")
    adata = _plantable()
    treated = ~adata.obs["Metadata_Control"].to_numpy(dtype=bool)
    labels = adata.obs["Metadata_Perturbation"].to_numpy().copy()
    labels[treated] = [f"singleton{index}" for index in range(int(treated.sum()))]
    adata.obs["Metadata_Perturbation"] = labels

    with pytest.raises(ValueError, match="needs replicate pairs to score"):
        mt.pp.sphere(adata, method="ZCA", epsilon="auto")


def test_auto_with_no_treated_wells_raises():
    """An all-controls object leaves nothing to score, and an empty count must not read nan < 2 as False."""
    pytest.importorskip("copairs")
    adata = _plantable()
    adata.obs["Metadata_Control"] = np.ones(adata.n_obs, dtype=bool)

    with pytest.raises(ValueError, match="needs replicate pairs to score"):
        mt.pp.sphere(adata, method="ZCA", epsilon="auto")


def test_a_string_epsilon_other_than_auto_is_rejected():
    adata = _plantable()
    with pytest.raises(ValueError, match="epsilon must be a float or 'auto'"):
        mt.pp.sphere(adata, method="ZCA", epsilon="small")


def test_a_float_epsilon_is_unchanged_and_records_no_sweep():
    """The float path still whitens deterministically and leaves no sweep behind it."""
    reference = mt.pp.sphere(_plantable(), method="ZCA", epsilon=1e-2, copy=True)
    again = mt.pp.sphere(_plantable(), method="ZCA", epsilon=1e-2, copy=True)
    np.testing.assert_allclose(np.asarray(reference.X), np.asarray(again.X))
    assert np.isfinite(reference.X).all()
    assert "sphere_epsilon" not in reference.uns.get("mantispy", {})


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
