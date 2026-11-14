"""Time copairs (via mantispy's _score_map) vs the numba prototype on one dataset/mode, in its own process.

    python bench.py DATASET MODE IMPL OUTDIR [--null-size N] [--threads T]

IMPL: copairs | numba (copairs' exact numpy null stream) | floyd (numba null sampler) | gemm (BLAS sims + floyd).
Writes OUTDIR/{dataset}_{mode}_{impl}.parquet (group table) and .json (timings, peak RSS).
"""

import argparse
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

FILES = {
    "target2": "/lustre/groups/ml01/workspace/tim.treis/mantispy-cache/mantispy/jump_target2.h5ad",
    "crispr": "/lustre/groups/ml01/workspace/tim.treis/mantispy-cache/mantispy/jump_crispr.h5ad",
    "acon": "/lustre/groups/ml01/workspace/tim.treis/acon-build/build/hosted/aconcagua_guide_well.h5ad",
}


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2  # linux: KiB


class Timer:
    def __init__(self):
        self.t = {}

    def wrap(self, name, fn):
        def inner(*a, **k):
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t0

        return inner


def prepare(dataset, mode):
    import anndata as ad

    from mantispy._core.masks import reference_mask
    from mantispy._core.schema import resolve_perturbation_column
    from mantispy.tl._map import REFERENCE_COLUMN, _resolve_mode

    adata = ad.read_h5ad(FILES[dataset])
    label = resolve_perturbation_column(adata, None)
    x = np.asarray(adata.X, dtype=np.float32)
    keep = ~np.isnan(x).any(axis=0)
    x = np.ascontiguousarray(x[:, keep])
    settings = _resolve_mode(mode, label=label, batch="Metadata_Plate")
    obs = adata.obs.reset_index(drop=True)
    cols = sorted({c for v in settings.values() for c in v} - {REFERENCE_COLUMN} | {label})
    meta = obs[cols].copy()
    for c in cols:
        if isinstance(meta[c].dtype, pd.CategoricalDtype):
            meta[c] = meta[c].astype(str)
    is_control = reference_mask(adata, "negcon")
    drop = None
    if mode == "activity":
        meta[REFERENCE_COLUMN] = np.where(is_control, np.arange(len(meta)), -1)
        drop = np.flatnonzero(is_control)
    else:
        meta, x = meta[~is_control].reset_index(drop=True), x[~is_control]
    return meta, x, settings, drop, label, is_control


def run_numba(meta, x, settings, drop, null_size, seed, null_kind, timer, n_jobs, sims="exact"):
    import map_nb
    from mantispy.tl._map import REFERENCE_COLUMN

    t0 = time.perf_counter()
    xn = map_nb.normalize_rows(x)
    timer.t["normalize"] = time.perf_counter() - t0
    ap = timer.wrap("ap_kernel", map_nb.average_precision)(meta, x, **settings, xn=xn, sims=sims)
    if drop is not None:
        ap = ap[~ap.index.isin(drop)]
    stranded = (ap["n_pos_pairs"] > 0) & (ap["n_total_pairs"] == ap["n_pos_pairs"])
    ap = ap[~stranded]
    groups = [c for c in settings["pos_sameby"] if c != REFERENCE_COLUMN]
    return timer.wrap("map_null", map_nb.mean_average_precision)(
        ap, groups, null_size=null_size, seed=seed, null_kind=null_kind, n_jobs=n_jobs
    )


def run_copairs(meta, x, settings, drop, null_size, seed, timer, n_jobs):
    import copairs.compute as compute
    import copairs.map  # noqa: F401 - registers the submodules, whose names the package's functions shadow

    cap = sys.modules["copairs.map.average_precision"]
    cmm = sys.modules["copairs.map.map"]

    from mantispy.tl._map import _score_map

    cap.find_pairs = timer.wrap("find_pairs (duckdb)", cap.find_pairs)
    cap.build_rank_lists = timer.wrap("build_rank_lists (lexsort)", cap.build_rank_lists)
    compute.ap_contiguous = timer.wrap("ap_contiguous", compute.ap_contiguous)
    original = compute.get_similarity_fn
    compute.get_similarity_fn = lambda *a, **k: timer.wrap("similarity", original(*a, **k))
    compute.get_null_dists = timer.wrap("null get_null_dists", compute.get_null_dists)
    cmm.compute = compute
    cap.compute = compute
    cmm.mean_average_precision = timer.wrap("mean_average_precision (total)", cmm.mean_average_precision)
    import copairs.map as cm

    cm.mean_average_precision = cmm.mean_average_precision
    cm.average_precision = timer.wrap("average_precision (total)", cm.average_precision)
    table, _ = _score_map(
        meta, x, settings, null_size=null_size, threshold=0.05, seed=seed, null="copied",
        drop_positions=drop, n_jobs=n_jobs, warn=False,
    )  # fmt: skip
    return table.rename(columns={"score": "mean_average_precision", "pvalue": "p_value"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("mode")
    p.add_argument("impl")
    p.add_argument("outdir")
    p.add_argument("--null-size", type=int, default=10_000)
    p.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = p.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.dataset}_{a.mode}_{a.impl}"

    t0 = time.perf_counter()
    meta, x, settings, drop, label, is_control = prepare(a.dataset, a.mode)
    t_load = time.perf_counter() - t0
    rss_load = rss_gb()
    timer = Timer()
    if a.impl != "copairs":  # compile outside the timing on a tiny slice
        import map_nb

        sub = meta.iloc[:200].reset_index(drop=True)
        try:
            map_nb.average_precision(sub, x[:200], **settings)
            map_nb.average_precision(sub, x[:200], **settings, sims="gemm")
        except Exception:
            pass
        map_nb.null_floyd(np.array([[1, 3]]), 10, 0)
        map_nb._group_pvalues(np.zeros((1, 2), np.float32), np.zeros(1, np.int64), np.array([0, 1]), np.zeros(1))
    t1 = time.perf_counter()
    if a.impl == "copairs":
        table = run_copairs(meta, x, settings, drop, a.null_size, 0, timer, a.threads)
    else:
        table = run_numba(meta, x, settings, drop, a.null_size, 0, "numpy" if a.impl == "numba" else "floyd",
                          timer, a.threads, sims="gemm" if a.impl == "gemm" else "exact")  # fmt: skip
    wall = time.perf_counter() - t1
    if a.mode == "activity":
        ctrl = set(meta.loc[is_control, label])
        table = table[~table[label].isin(ctrl)]
    keep = [c for c in table.columns if c in (label, "mean_average_precision", "p_value")]
    table[keep].reset_index(drop=True).to_parquet(out / f"{tag}.parquet")
    rec = dict(
        dataset=a.dataset, mode=a.mode, impl=a.impl, n_rows=len(meta), n_feat=x.shape[1], threads=a.threads,
        null_size=a.null_size, load_s=round(t_load, 2), wall_s=round(wall, 2), rss_after_load_gb=round(rss_load, 2),
        peak_rss_gb=round(rss_gb(), 2), steps={k: round(v, 2) for k, v in timer.t.items()}, n_groups=len(table),
        cpu_count=os.cpu_count(),
    )  # fmt: skip
    (out / f"{tag}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
