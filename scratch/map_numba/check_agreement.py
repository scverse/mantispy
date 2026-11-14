"""Agreement of the numba prototype with copairs on small synthetic layouts (cheap; runs anywhere)."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import map_nb  # noqa: E402
from copairs import compute  # noqa: E402
from copairs import map as cm  # noqa: E402


def layout(rng, n_plates=4, n_perts=30, reps=3, n_ctrl=16, n_feat=300, ties=False):
    rows = []
    for p in range(n_plates):
        for g in range(n_perts):
            for r in range(reps):
                rows.append((f"P{p}", f"g{g}", -1))
        for c in range(n_ctrl):
            rows.append((f"P{p}", "ctrl", 0))
    meta = pd.DataFrame(rows, columns=["Metadata_Plate", "Metadata_Perturbation", "ref"])
    is_ctrl = meta["Metadata_Perturbation"].eq("ctrl").to_numpy()
    meta["ref"] = np.where(is_ctrl, np.arange(len(meta)), -1)
    x = rng.standard_normal((len(meta), n_feat)).astype(np.float32)
    effect = {g: rng.standard_normal(n_feat).astype(np.float32) for g in meta["Metadata_Perturbation"].unique()}
    x += 0.3 * np.stack([effect[g] for g in meta["Metadata_Perturbation"]])
    if ties:  # duplicate profiles create exact similarity ties across pos/neg
        x[1::7] = x[0::7][: len(x[1::7])]
        x = np.round(x, 1)
    return meta, x, is_ctrl


MODES = {
    "activity": (["Metadata_Perturbation", "ref"], [], ["Metadata_Plate"], ["Metadata_Perturbation", "ref"]),
    "replicability": (["Metadata_Perturbation"], [], ["Metadata_Plate"], ["Metadata_Perturbation"]),
    "cross_plate": (["Metadata_Perturbation"], ["Metadata_Plate"], [], ["Metadata_Perturbation"]),
}


def main():
    rng = np.random.default_rng(0)
    # 1. bit-identical similarities
    for f in (5, 8, 100, 128, 129, 300, 1483, 2000):
        a = rng.standard_normal((500, f)).astype(np.float32)
        b = rng.standard_normal((500, f)).astype(np.float32)
        ref = compute.pairwise_cosine(a, b)
        an, bn = map_nb.normalize_rows(a), map_nb.normalize_rows(b)
        mine = np.array([map_nb._dot_rec(an[i], bn[i], 0, f) for i in range(500)], np.float32)
        print(f"sim F={f}: bit-identical {np.array_equal(ref.view(np.uint32), mine.view(np.uint32))}")

    for ties in (False, True):
        meta, x, is_ctrl = layout(rng, ties=ties)
        for mode, (ps, pd_, ns, nd) in MODES.items():
            ref = cm.average_precision(meta, x, ps, pd_, ns, nd, progress_bar=False)
            mine = map_nb.average_precision(meta, x, ps, pd_, ns, nd)
            same_counts = np.array_equal(ref["n_pos_pairs"], mine["n_pos_pairs"]) and np.array_equal(
                ref["n_total_pairs"], mine["n_total_pairs"]
            )
            d = np.nanmax(np.abs(ref["average_precision"].to_numpy() - mine["average_precision"].to_numpy()))
            nan_same = np.array_equal(ref["average_precision"].isna(), mine["average_precision"].isna())
            # p-values: copied null, identical draws
            sameby = [c for c in ps if c != "ref"]
            r = ref[~is_ctrl] if mode == "activity" else ref
            m = mine[~is_ctrl] if mode == "activity" else mine
            with tempfile.TemporaryDirectory() as cache:
                rmap = cm.mean_average_precision(r, sameby, null_size=2000, threshold=0.05, seed=0,
                                                 progress_bar=False, cache_dir=cache, null="copied")  # fmt: skip
            mmap = map_nb.mean_average_precision(m, sameby, null_size=2000, seed=0, null_kind="numpy")
            fmap = map_nb.mean_average_precision(m, sameby, null_size=2000, seed=0, null_kind="floyd")
            j = rmap.merge(mmap, on=sameby, suffixes=("_c", "_n")).merge(fmap, on=sameby)
            print(
                f"ties={ties} {mode:13s} counts equal {same_counts}, NaN pattern equal {nan_same}, "
                f"max|dAP| {d:.2e}, max|dmAP| {np.abs(j.mean_average_precision_c - j.mean_average_precision_n).max():.2e}, "
                f"p identical {np.array_equal(j.p_value_c.to_numpy(), j.p_value_n.to_numpy())} ({len(j)} groups), "
                f"floyd max|dp| {np.abs(j.p_value_c - j.p_value).max():.3f}"
            )


if __name__ == "__main__":
    main()
