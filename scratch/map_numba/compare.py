"""Summarise bench results: timings, peak RSS, and agreement of numba/floyd with copairs."""

import json
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import false_discovery_control

res = Path(sys.argv[1])
rows = [json.loads(p.read_text()) for p in sorted(res.glob("*.json"))]
print(pd.DataFrame(rows)[["dataset", "mode", "impl", "n_rows", "n_feat", "n_groups", "wall_s", "rss_after_load_gb",
                          "peak_rss_gb"]].to_string(index=False))  # fmt: skip
for r in rows:
    print(r["dataset"], r["mode"], r["impl"], r["steps"])

for p in sorted(res.glob("*_copairs.parquet")):
    ref = pd.read_parquet(p)
    key = ref.columns[0]
    for impl in ("numba", "floyd", "gemm"):
        q = p.with_name(p.name.replace("_copairs", f"_{impl}"))
        if not q.exists():
            continue
        j = ref.merge(pd.read_parquet(q), on=key, suffixes=("_c", "_n"))
        dmap = (j.mean_average_precision_c - j.mean_average_precision_n).abs().max()
        same_p = (j.p_value_c == j.p_value_n).mean()
        dp = (j.p_value_c - j.p_value_n).abs().max()
        calls_c = false_discovery_control(j.p_value_c) < 0.05
        calls_n = false_discovery_control(j.p_value_n) < 0.05
        print(f"{p.stem} vs {impl}: groups {len(j)}/{len(ref)}, max|dmAP| {dmap:.1e}, p identical {same_p:.4f}, "
              f"max|dp| {dp:.4f}, BH calls {calls_c.sum()} vs {calls_n.sum()} (disagree {(calls_c != calls_n).sum()})")  # fmt: skip
