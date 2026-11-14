"""mAP equivalence against copairs, the reference implementation mantispy's kernels replace."""

import numpy as np
import pandas as pd
import pytest

from mantispy._core import _average_precision as ap_kernel
from mantispy._core import _map_null as map_null
from mantispy._core import _map_permutation as map_permutation

copairs_map = pytest.importorskip("copairs.map")
copairs_multilabel = pytest.importorskip("copairs.map.multilabel")

#: The pair definitions of tl.map's presets, on the columns of :func:`_screen`.
MODES = {
    "activity": (["Metadata_Perturbation", "ref"], [], ["Metadata_Plate"], ["Metadata_Perturbation", "ref"]),
    "replicability": (["Metadata_Perturbation"], [], ["Metadata_Plate"], ["Metadata_Perturbation"]),
    "cross_plate": (["Metadata_Perturbation"], ["Metadata_Plate"], [], ["Metadata_Perturbation"]),
}
DISTANCES = ["cosine", "abs_cosine", "correlation", "euclidean", "manhattan", "chebyshev", "cityblock", "braycurtis"]


def _screen(seed: int = 0, ties: bool = False, n_features: int = 150) -> tuple[pd.DataFrame, np.ndarray]:
    """Three plates of twelve perturbations in triplicate beside eight controls; ``ties`` duplicates and rounds profiles."""
    rng = np.random.default_rng(seed)
    perturbations = [f"g{g}" for g in range(12) for _ in range(3)] + ["ctrl"] * 8
    meta = pd.DataFrame(
        {
            "Metadata_Plate": np.repeat([f"P{p}" for p in range(3)], len(perturbations)),
            "Metadata_Perturbation": perturbations * 3,
        }
    )
    control = meta["Metadata_Perturbation"].eq("ctrl").to_numpy()
    meta["ref"] = np.where(control, np.arange(len(meta)), -1)
    effects = {name: rng.standard_normal(n_features) for name in meta["Metadata_Perturbation"].unique()}
    x = rng.standard_normal((len(meta), n_features)) + 0.4 * np.stack(
        [effects[p] for p in meta["Metadata_Perturbation"]]
    )
    x = x.astype(np.float32)
    if ties:  # exact duplicates and coarse values tie similarities across positives and negatives
        x[1::5] = x[0::5][: len(x[1::5])]
        x = np.round(x)
    return meta, x


def _ours(meta, x, spec, distance="cosine", solver="exact", multilabel_col=None):
    lay = ap_kernel.layout(meta, *spec, multilabel_col=multilabel_col)
    return lay, ap_kernel.average_precision(lay, x, distance, solver)


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("distance", DISTANCES)
def test_average_precision_matches_copairs(mode, distance, ties):
    """Every similarity is copairs' float32 value, so ranks, ties included, and pair counts are identical."""
    meta, x = _screen(ties=ties)
    expected = copairs_map.average_precision(meta, x, *MODES[mode], distance=distance, progress_bar=False)
    _, result = _ours(meta, x, MODES[mode], distance)

    np.testing.assert_array_equal(result.n_pos_pairs, expected["n_pos_pairs"])
    np.testing.assert_array_equal(result.n_total_pairs, expected["n_total_pairs"])
    np.testing.assert_allclose(result.average_precision, expected["average_precision"], rtol=0, atol=1e-12)
    normalized = ap_kernel.normalized_average_precision(
        result.average_precision, result.n_pos_pairs, result.n_total_pairs
    )
    np.testing.assert_allclose(normalized, expected["normalized_average_precision"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("distance", ap_kernel.BLAS_DISTANCES)
def test_blas_similarities_match_copairs_without_near_ties(distance):
    """BLAS rounds differently, which moves a rank only where two similarities lie within float32 rounding."""
    meta, x = _screen(seed=1)
    expected = copairs_map.average_precision(meta, x, *MODES["replicability"], distance=distance, progress_bar=False)
    _, result = _ours(meta, x, MODES["replicability"], distance, solver="blas")
    np.testing.assert_array_equal(result.n_total_pairs, expected["n_total_pairs"])
    np.testing.assert_allclose(result.average_precision, expected["average_precision"], atol=1e-12)


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize(
    "spec",
    [
        (["moa"], ["Metadata_Perturbation"], [], ["moa"]),  # tl.map's consistency preset
        (["moa"], [], ["Metadata_Plate"], ["moa"]),
        (["moa", "Metadata_Plate"], [], [], ["moa"]),
    ],
)
def test_multilabel_average_precision_matches_copairs(spec, ties):
    """A positive shares a label, a negative shares none; a profile is scored once per label, in copairs' row order."""
    meta, x = _screen(ties=ties)
    labels = {f"g{g}": [f"m{g % 4}"] + ([f"m{(g + 1) % 4}"] if g % 3 == 0 else []) for g in range(12)}
    meta["moa"] = meta["Metadata_Perturbation"].map(labels | {"ctrl": []})
    meta["row"] = np.arange(len(meta))
    expected = copairs_multilabel.average_precision(meta, x, *spec, multilabel_col="moa", progress_bar=False)
    lay, result = _ours(meta, x, spec, multilabel_col="moa")

    np.testing.assert_array_equal(result.row_profile, expected["row"])
    np.testing.assert_array_equal(lay.label_values[result.row_label], expected["moa"])
    np.testing.assert_array_equal(result.n_pos_pairs, expected["n_pos_pairs"])
    np.testing.assert_array_equal(result.n_total_pairs, expected["n_total_pairs"])
    np.testing.assert_allclose(result.average_precision, expected["average_precision"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("n_features", [5, 8, 128, 129, 300, 1483])
def test_exact_similarity_is_copairs_float32_value(n_features):
    """The pairwise-summation tree, evaluated without recursion, reproduces numpy's float32 sums bit for bit."""
    from copairs import compute

    rng = np.random.default_rng(n_features)
    a, b = (rng.standard_normal((200, n_features)).astype(np.float32) for _ in range(2))
    expected = compute.pairwise_cosine(a, b)
    rows, scale = ap_kernel.prepare_rows(np.concatenate([a, b]), "cosine")
    leaf_lo, leaf_n, ops = ap_kernel.summation_plan(n_features)
    stack = np.empty(64, np.float32)
    ours = np.array(
        [ap_kernel._similarity(rows, scale, i, 200 + i, 0, leaf_lo, leaf_n, ops, stack) for i in range(200)],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(ours.view(np.uint32), expected.view(np.uint32))


def test_unsupported_distances_are_refused():
    meta, x = _screen()
    lay = ap_kernel.layout(meta, *MODES["replicability"])
    with pytest.raises(ValueError, match="estimates its scale"):
        ap_kernel.average_precision(lay, x, "mahalanobis")
    with pytest.raises(ValueError, match="'blas' supports"):
        ap_kernel.average_precision(lay, x, "manhattan", solver="blas")
    with pytest.raises(ValueError, match="distance must be one of"):
        ap_kernel.average_precision(lay, x, "nonsense")


def _copairs_groups(meta, x, spec, null, seed=7, null_size=2000, multilabel_col=None):
    if multilabel_col is None:
        ap = copairs_map.average_precision(meta, x, *spec, progress_bar=False)
    else:
        ap = copairs_multilabel.average_precision(meta, x, *spec, multilabel_col=multilabel_col, progress_bar=False)
    ap = ap[ap["ref"] < 0] if "ref" in spec[0] else ap
    sameby = [c for c in spec[0] if c != "ref"]
    return copairs_map.mean_average_precision(
        ap, sameby, null_size=null_size, threshold=0.05, seed=seed, progress_bar=False, null=null
    ), sameby


def _our_groups(meta, x, spec, multilabel_col=None):
    lay, result = _ours(meta, x, spec, multilabel_col=multilabel_col)
    ap = ap_kernel.ap_frame(meta, lay, result, multilabel_col)
    ap = ap[ap["ref"] < 0] if "ref" in spec[0] else ap
    sameby = [c for c in spec[0] if c != "ref"]
    table, groups, tested = map_null.group_scores(ap, sameby)
    return table, groups, tested, result


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("mode", list(MODES))
def test_copied_null_matches_copairs(mode, ties):
    """The exact sampler draws copairs' numpy stream, so every p-value is copairs'; Floyd's sampler agrees within Monte Carlo error."""
    meta, x = _screen(ties=ties)
    expected, sameby = _copairs_groups(meta, x, MODES[mode], "copied")
    table, groups, valid, _ = _our_groups(meta, x, MODES[mode])
    order = np.concatenate(groups)
    ptr = np.concatenate([[0], np.cumsum([len(g) for g in groups])])
    n_pos, n_total = valid["n_pos_pairs"].to_numpy()[order], valid["n_total_pairs"].to_numpy()[order]
    scores = table["mean_average_precision"].to_numpy()
    np.testing.assert_allclose(scores, expected["mean_average_precision"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        table["mean_normalized_average_precision"], expected["mean_normalized_average_precision"], atol=1e-12
    )

    exact = map_null.copied_pvalues(scores, ptr, n_pos, n_total, 2000, 7, "exact", n_threads=2)
    np.testing.assert_array_equal(exact, expected["p_value"])
    floyd = map_null.copied_pvalues(scores, ptr, n_pos, n_total, 2000, 7, "floyd")
    p = expected["p_value"].to_numpy()
    assert np.all(np.abs(floyd - p) <= 5 * np.sqrt(p * (1 - p) / 2000) + 3 / 2001)


@pytest.mark.parametrize(("num_pos", "total"), [(1, 5), (3, 300), (200, 300), (5, 65535), (3, 70000)])
def test_copied_null_draws_are_copairs_draws(num_pos, total):
    """Both index widths copairs uses (uint16 below 2**16 profiles, uint32 above) and sums longer than one block."""
    from copairs.compute import random_ap

    ours = map_null._null_exact(np.array([[num_pos, total]]), 64, 11, 1)[0]
    expected = random_ap(64, num_pos, total, map_null._conf_seed(11, num_pos, total))
    np.testing.assert_array_equal(ours.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("multilabel", [False, True])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("mode", list(MODES))
def test_joint_null_matches_copairs(mode, ties, multilabel):
    """copairs' Philox sampler, compiled: the same draws for the same seed, so every p-value is copairs'."""
    meta, x = _screen(ties=ties)
    spec, column = MODES[mode], None
    if multilabel:
        meta["moa"] = meta["Metadata_Perturbation"].map(lambda p: [] if p == "ctrl" else [f"m{int(p[1:]) % 4}"])
        spec, column = (["moa"], ["Metadata_Perturbation"], [], ["moa"]), "moa"
    expected, _ = _copairs_groups(meta, x, spec, "joint", multilabel_col=column)
    table, groups, tested, result = _our_groups(meta, x, spec, multilabel_col=column)
    n_pos = tested["n_pos_pairs"].to_numpy()
    edges = map_null.group_edges(groups, tested.index.to_numpy(), result)
    p = map_null.joint_pvalues(
        table["mean_average_precision"].to_numpy(),
        groups,
        n_pos,
        tested["n_total_pairs"].to_numpy() - n_pos,
        edges,
        2000,
        7,
    )
    np.testing.assert_array_equal(p, expected["p_value"])


def _permutation(meta, x, spec, null_size, seed, multilabel_col=None):
    table, groups, tested, result = _our_groups(meta, x, spec, multilabel_col=multilabel_col)
    lay = ap_kernel.layout(meta, *spec, multilabel_col=multilabel_col)
    p = map_permutation.permutation_pvalues(
        lay,
        result,
        x,
        "cosine",
        table["mean_average_precision"].to_numpy(),
        groups,
        tested.index.to_numpy(),
        null_size,
        seed,
    )
    return table.assign(p_value=p)


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("mode", [*MODES, "multilabel"])
def test_permutation_null_agrees_with_copairs_in_distribution(mode, ties):
    """The same null, relabelling within blocks, from other draws: p-values agree within their Monte Carlo error."""
    meta, x = _screen(seed=3, ties=ties, n_features=40)
    x = x * 0.2 + np.random.default_rng(4).standard_normal(x.shape).astype(
        np.float32
    )  # weak effects spread the p-values
    spec, column = MODES.get(mode), None
    if mode == "multilabel":
        labels = {f"g{g}": [f"m{g % 4}"] + ([f"m{(g + 1) % 4}"] if g % 3 == 0 else []) for g in range(12)}
        meta["moa"] = meta["Metadata_Perturbation"].map(labels | {"ctrl": []})
        spec, column = (["moa"], ["Metadata_Perturbation"], [], ["moa"]), "moa"
    expected, sameby = _copairs_groups(meta, x, spec, "permutation", null_size=4000, multilabel_col=column)
    ours = _permutation(meta, x, spec, 4000, 0, multilabel_col=column)
    joined = expected.merge(ours, on=sameby, suffixes=("_copairs", ""))
    p = joined["p_value_copairs"].to_numpy()
    error = np.sqrt(2 * p * (1 - p) / 4000)
    assert np.all(np.abs(joined["p_value"] - p) <= 4 * error + 3 / 4001)


def test_permutation_null_is_uniform_on_null_data():
    """Labels assigned at random to identically distributed profiles: p-values are uniform, so the type-I rate holds."""
    from scipy.stats import binomtest, kstest

    rng = np.random.default_rng(5)
    n_groups, reps, n_plates = 150, 3, 4
    meta = pd.DataFrame(
        {
            "Metadata_Plate": np.repeat([f"P{p}" for p in range(n_plates)], n_groups * reps + 40),
            "Metadata_Perturbation": np.tile(
                np.concatenate([np.repeat([f"g{g}" for g in range(n_groups)], reps), ["ctrl"] * 40]), n_plates
            ),
        }
    )
    meta["Metadata_Perturbation"] = meta["Metadata_Perturbation"] + "_" + meta["Metadata_Plate"]
    meta.loc[meta["Metadata_Perturbation"].str.startswith("ctrl"), "Metadata_Perturbation"] = "ctrl"
    x = rng.standard_normal((len(meta), 30)).astype(np.float32)
    spec = MODES["replicability"]
    p = _permutation(meta, x, spec, 999, 0)
    p = p.loc[p["Metadata_Perturbation"] != "ctrl", "p_value"].to_numpy()
    assert kstest(p, "uniform").pvalue > 1e-3
    for alpha in (0.05, 0.01):
        assert binomtest(int((p <= alpha).sum()), len(p), alpha).pvalue > 1e-3


def test_permutation_null_refuses_positives_outside_the_group():
    """Relabelling a group cannot move a positive it does not hold."""
    meta, x = _screen()
    lay, result = _ours(meta, x, MODES["replicability"])
    ap = ap_kernel.ap_frame(meta, lay, result)
    ap = ap.drop(index=0)  # a replicate whose partners still count it as a positive
    table, groups, tested = map_null.group_scores(ap, ["Metadata_Perturbation"])
    with pytest.raises(ValueError, match="positives to be its other members"):
        map_permutation.permutation_pvalues(
            lay, result, x, "cosine", table["mean_average_precision"].to_numpy(), groups, tested.index.to_numpy(), 10, 0
        )


def _copairs_map(adata, mode, null, null_size=1000, seed=0):
    """tl.map's table as copairs computes it, on the same metadata and features."""
    from mantispy._core.masks import reference_mask
    from mantispy.tl._map import REFERENCE_COLUMN, _resolve_mode

    settings = _resolve_mode(mode, label="Metadata_Perturbation", batch="Metadata_Plate")
    meta = adata.obs[["Metadata_Perturbation", "Metadata_Plate"]].reset_index(drop=True).astype(str)
    x = np.asarray(adata.X, dtype=np.float32)
    control = reference_mask(adata, "negcon")
    if mode == "activity":
        meta[REFERENCE_COLUMN] = np.where(control, np.arange(adata.n_obs), -1)
    else:
        meta, x = meta[~control].reset_index(drop=True), x[~control]
    ap = copairs_map.average_precision(meta, x, **settings, progress_bar=False)
    if mode == "activity":
        ap = ap[~control]
    return copairs_map.mean_average_precision(
        ap, ["Metadata_Perturbation"], null_size=null_size, threshold=0.05, seed=seed, progress_bar=False, null=null
    )


@pytest.mark.parametrize("null", ["copied", "joint"])
@pytest.mark.parametrize("mode", ["activity", "replicability"])
def test_tl_map_reproduces_copairs(mode, null):
    """Out of the box, tl.map's scores, p-values and q-values are copairs'."""
    import mantispy as mt
    from mantispy.ds import synthetic_plate

    cells = synthetic_plate(
        n_plates=2, n_wells=96, n_cells=10, n_features=20, n_perturbations=8, effect_size=0.5, seed=0
    )
    mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon")
    wells = mt.tl.aggregate(cells, min_cells=0)
    ours = mt.tl.map(wells, mode=mode, null=null, null_size=1000, copy=True).uns["mantispy"]["map"]
    expected = _copairs_map(wells, mode, null)
    joined = ours.assign(Metadata_Perturbation=ours["Metadata_Perturbation"].astype(str)).merge(
        expected, on="Metadata_Perturbation"
    )
    assert len(joined) == len(ours) > 0
    np.testing.assert_allclose(joined["score"], joined["mean_average_precision"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(joined["pvalue"], joined["p_value"])
    np.testing.assert_allclose(joined["qvalue"], joined["corrected_p_value"], rtol=1e-12)


@pytest.mark.parametrize("mode", [*MODES, "multilabel"])
def test_permutation_null_does_not_depend_on_where_its_keys_come_from(mode, monkeypatch):
    """A shared table, per-group tables or keys scored during the draws hold the same values, so the draws agree exactly."""
    meta, x = _screen(seed=6, ties=True, n_features=20)
    spec, column = MODES.get(mode), None
    if mode == "multilabel":
        meta["moa"] = meta["Metadata_Perturbation"].map(lambda p: [] if p == "ctrl" else [f"m{int(p[1:]) % 3}"])
        spec, column = (["moa"], ["Metadata_Perturbation"], [], ["moa"]), "moa"
    shared = _permutation(meta, x, spec, 500, 0, multilabel_col=column)["p_value"]
    monkeypatch.setattr(map_permutation, "_TABLE_BUDGET", 4 * len(meta) ** 2 // 3)  # too small for the shared one
    per_group = _permutation(meta, x, spec, 500, 0, multilabel_col=column)["p_value"]
    monkeypatch.setattr(map_permutation, "_TABLE_BUDGET", 0)
    scored = _permutation(meta, x, spec, 500, 0, multilabel_col=column)["p_value"]
    np.testing.assert_array_equal(shared, per_group)
    np.testing.assert_array_equal(shared, scored)
