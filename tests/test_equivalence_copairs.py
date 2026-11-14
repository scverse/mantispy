"""mAP equivalence against copairs, the reference implementation mantispy's kernels replace."""

import numpy as np
import pandas as pd
import pytest

from mantispy._core import _average_precision as ap_kernel
from mantispy._core import _map_null as map_null

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
