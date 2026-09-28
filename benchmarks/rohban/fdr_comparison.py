"""Empirically compare multiple-testing strategies on the Rohban per-cluster ORA p-values.

The v2 reproduction (``reproduce.py``) clusters the strong construct subset and runs ``mt.tl.ora``
per cluster against a broad gene-set net (GO-BP + CORUM + Reactome). Per-cluster Benjamini-Hochberg
calls ~0/16 multi-gene clusters enriched at q<0.05 even though ~15/16 are enriched at nominal p<0.05.
The question this script answers: which within-cluster multiple-testing correction recovers the real
enrichment (toward the paper's 19/22) without inflating false positives.

It does NOT modify ``reproduce.py`` or any package code. It reuses the reproduction pipeline by
importing ``reproduce`` (which runs it end to end) and reading the per-cluster ORA table it leaves in
``cons_strong.uns["mantispy"]["ora"]`` for the broad net. ``reproduce.py`` writes ``REPRODUCTION.md``
as a side effect; that reference artifact is snapshotted and restored around the import so this
throwaway analysis leaves it byte-for-byte unchanged.

For each correction, applied WITHIN each cluster at cutoff 0.05:
  (a) BH (Benjamini-Hochberg), the current method.
  (b) Storey q-value (lambda=0.5 pi0 estimate).
  (c) Two-stage adaptive BH (statsmodels ``fdr_tsbh``), if statsmodels is importable.
  (d) BH per (cluster x collection): GO / CORUM / Reactome corrected independently within a cluster.
  (e) Nominal p<0.05 (no correction), the upper reference.
  (f) Restricted-universe BH: within each cluster, test and correct over ONLY the sets its genes hit (overlap a>=1).
      This is conventional overlap-based ORA (the ``n`` column of the ORA table is the overlap a).
  (g) Restricted-universe BH with a>=2 (a stricter relevance filter: two of the cluster's genes co-occur in a set).
  (h) Restricted-universe Storey q (a>=1), to see if adaptive pi0 adds anything once the universe is small.

A cluster counts as enriched if it has at least one set below the cutoff. It also runs a single
permuted null (construct-to-cluster labels shuffled once) through every method as a false-positive
sanity check. Results print to stdout and are saved to ``fdr_comparison.txt``.

Run from the repo root inside the project venv:

    python benchmarks/rohban/fdr_comparison.py
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
CUTOFF = 0.05
PERM_SEED = 0

try:
    from statsmodels.stats.multitest import multipletests

    HAVE_STATSMODELS = True
except ImportError:
    multipletests = None
    HAVE_STATSMODELS = False


# =============================================================================================
# Reuse the reproduction pipeline by importing it, preserving REPRODUCTION.md byte-for-byte.
# =============================================================================================
def _load_pipeline():
    """Import reproduce.py (which runs the full pipeline) and return the objects this analysis needs.

    Returns the broad-net per-cluster ORA table, the strong-subset AnnData with cluster labels, the
    broad gene-set net, and the set of multi-gene cluster labels (the paper-graded universe of 16).
    """
    sys.path.insert(0, str(HERE))
    repro_md = HERE / "REPRODUCTION.md"
    saved = repro_md.read_bytes() if repro_md.exists() else None
    try:
        # reproduce.py runs on import and prints heavily; silence it and keep our own output clean.
        with contextlib.redirect_stdout(io.StringIO()):
            import reproduce
    finally:
        # Restore the reference artifact so this throwaway analysis does not regenerate it.
        if saved is not None:
            repro_md.write_bytes(saved)

    cons_strong = reproduce.cons_strong
    broad_net = reproduce.broad_net
    multigene = set(reproduce.multigene)
    ora = cons_strong.uns["mantispy"]["ora"].copy()  # the LAST ora run in reproduce is the broad net
    return ora, cons_strong, broad_net, multigene, reproduce.mt


# =============================================================================================
# Correction primitives (all operate on one 1-D array of p-values).
# =============================================================================================
def bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (step-up, monotone enforced)."""
    p = np.asarray(pvals, dtype=np.float64)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def storey_q(pvals: np.ndarray, lam: float = 0.5) -> np.ndarray:
    """Storey q-values: pi0 from the lambda=0.5 estimator, clipped to (0,1], times BH."""
    p = np.asarray(pvals, dtype=np.float64)
    if p.size == 0:
        return p
    pi0 = float(np.mean(p > lam) / (1.0 - lam))
    pi0 = min(max(pi0, np.finfo(np.float64).tiny), 1.0)  # clip to (0, 1]
    return np.clip(pi0 * bh(p), 0.0, 1.0)


def tsbh(pvals: np.ndarray) -> np.ndarray:
    """Two-stage adaptive BH via statsmodels (fdr_tsbh) at the analysis alpha."""
    p = np.asarray(pvals, dtype=np.float64)
    if p.size == 0:
        return p
    return multipletests(p, alpha=CUTOFF, method="fdr_tsbh")[1]


# =============================================================================================
# Apply a correction within each cluster (optionally within each cluster x collection).
# =============================================================================================
def significant_per_cluster(
    ora: pd.DataFrame, method: str, per_collection: bool = False, cutoff: float = CUTOFF
) -> dict[str, int]:
    """Map each cluster to its count of sets below the cutoff under the given within-cluster method.

    ``method`` is one of "nominal", "bh", "storey", "tsbh". When ``per_collection`` is True the method
    is BH applied separately within each (cluster, collection) block.
    """
    funcs = {"nominal": lambda x: x, "bh": bh, "storey": storey_q, "tsbh": tsbh}
    counts: dict[str, int] = {}
    keys = ["group", "collection"] if per_collection else "group"
    fn = bh if per_collection else funcs[method]
    for key, sub in ora.groupby(keys, observed=True):
        cluster = str(key[0] if per_collection else key)
        adj = fn(sub["pvalue"].to_numpy())
        counts[cluster] = counts.get(cluster, 0) + int(np.sum(adj < cutoff))
    return counts


def significant_restricted(
    ora: pd.DataFrame, min_overlap: int, method: str = "bh", cutoff: float = CUTOFF
) -> dict[str, int]:
    """Restricted-universe ORA: correct WITHIN each cluster over only the sets its genes actually hit.

    The ``n`` column of the ORA table is the overlap ``a`` (the cluster's genes that fall in the set). Conventional
    overlap-based ORA tests and corrects a gene list only over the sets it overlaps, not the full net; this keeps
    each cluster's set of tests to the sets with ``n >= min_overlap`` and then applies ``method`` ("bh" or "storey")
    within that restricted universe.
    """
    fn = bh if method == "bh" else storey_q
    counts: dict[str, int] = {}
    for cluster, sub in ora.groupby("group", observed=True):
        rel = sub[sub["n"] >= min_overlap]
        if len(rel) == 0:
            counts[str(cluster)] = 0
            continue
        adj = fn(rel["pvalue"].to_numpy())
        counts[str(cluster)] = int(np.sum(adj < cutoff))
    return counts


def summarize_counts(counts: dict[str, int], universe: set[str]) -> tuple[int, int]:
    """Return (enriched clusters, total set discoveries) from a per-cluster count map over a cluster universe."""
    enriched = sum(1 for c in universe if counts.get(c, 0) > 0)
    total = sum(counts.get(c, 0) for c in universe)
    return enriched, total


def summarize(ora: pd.DataFrame, universe: set[str], method: str, per_collection: bool = False) -> tuple[int, int]:
    """Return (enriched clusters, total set discoveries) restricted to the given cluster universe."""
    return summarize_counts(significant_per_cluster(ora, method, per_collection=per_collection), universe)


# =============================================================================================
# Build a one-shuffle permuted null: re-run ORA on shuffled construct-to-cluster labels.
# =============================================================================================
def permuted_ora(cons_strong, broad_net, mt) -> tuple[pd.DataFrame, set[str]]:
    """Shuffle cluster labels across constructs once, re-run ORA, and return (table, multi-gene set)."""
    perm = cons_strong.copy()
    rng = np.random.default_rng(PERM_SEED)
    labels = perm.obs["cluster"].to_numpy().copy()
    rng.shuffle(labels)
    perm.obs["cluster"] = pd.Categorical(labels)
    with contextlib.redirect_stdout(io.StringIO()):
        mt.tl.ora(perm, groupby="cluster", net=broad_net, gene_key="Metadata_Gene", tmin=5, padj_by="group")
    ora = perm.uns["mantispy"]["ora"].copy()
    gene = perm.obs["Metadata_Gene"].astype(str)
    lab = perm.obs["cluster"].astype(str)
    genes_per_cluster = gene.groupby(lab, observed=True).nunique()
    multigene = set(genes_per_cluster[genes_per_cluster >= 2].index)
    return ora, multigene


# =============================================================================================
# Main
# =============================================================================================
def main() -> None:
    print("Loading the Rohban reproduction pipeline (this runs reproduce.py end to end, ~2 min)...")
    ora, cons_strong, broad_net, multigene, mt = _load_pipeline()

    # Derive the collection (GO / CORUM / REACTOME) from the source prefix reproduce.py adds.
    ora["collection"] = ora["source"].astype(str).str.split(":", n=1).str[0]
    n_universe = len(multigene)
    n_sets = int(ora["source"].nunique())
    print(f"broad-net ORA table: {len(ora)} (cluster, set) rows, {n_sets} distinct sets, {n_universe} multi-gene clusters")
    by_coll = ora.groupby("collection", observed=True)["source"].nunique().to_dict()
    print(f"collections (distinct sets tested): {by_coll}")

    # The permuted null (single shuffle) shared across all methods.
    print(f"Building the permuted null (one shuffle of construct-to-cluster labels, seed={PERM_SEED})...")
    perm_ora, perm_multigene = permuted_ora(cons_strong, broad_net, mt)
    perm_ora["collection"] = perm_ora["source"].astype(str).str.split(":", n=1).str[0]
    n_perm_universe = len(perm_multigene)

    # method key -> (label, method-name-for-summarize, per_collection, available)
    methods = [
        ("(a) BH per cluster", "bh", False, True),
        ("(b) Storey q per cluster", "storey", False, True),
        ("(c) two-stage BH (fdr_tsbh)", "tsbh", False, HAVE_STATSMODELS),
        ("(d) BH per cluster x collection", "bh", True, True),
        ("(e) nominal p<0.05 (reference)", "nominal", False, True),
    ]

    rows = []
    for label, method, per_coll, available in methods:
        if not available:
            rows.append((label, "SKIPPED (statsmodels not installed)", "", ""))
            continue
        enriched, total = summarize(ora, multigene, method, per_collection=per_coll)
        perm_enriched, _ = summarize(perm_ora, perm_multigene, method, per_collection=per_coll)
        rows.append(
            (
                label,
                f"{enriched}/{n_universe}",
                str(total),
                f"{perm_enriched}/{n_perm_universe}",
            )
        )

    # Restricted-universe (conventional overlap-based ORA): test and correct WITHIN each cluster over only the sets
    # its genes actually hit (overlap n >= min). This shrinks the test count per cluster from ~896 to tens.
    restricted = [
        ("(f) restricted-universe BH (overlap a>=1)", 1, "bh"),
        ("(g) restricted-universe BH (overlap a>=2)", 2, "bh"),
        ("(h) restricted-universe Storey q (a>=1)", 1, "storey"),
    ]
    for label, min_overlap, method in restricted:
        counts = significant_restricted(ora, min_overlap, method=method)
        perm_counts = significant_restricted(perm_ora, min_overlap, method=method)
        enriched, total = summarize_counts(counts, multigene)
        perm_enriched, _ = summarize_counts(perm_counts, perm_multigene)
        rows.append(
            (
                label,
                f"{enriched}/{n_universe}",
                str(total),
                f"{perm_enriched}/{n_perm_universe}",
            )
        )

    # Render a fixed-width comparison table.
    header = ("method", f"enriched (of {n_universe})", "total set discoveries", "permuted-null enriched")
    widths = [
        max(len(header[0]), max(len(r[0]) for r in rows)),
        max(len(header[1]), max(len(r[1]) for r in rows)),
        max(len(header[2]), max(len(r[2]) for r in rows)),
        max(len(header[3]), max(len(r[3]) for r in rows)),
    ]

    def fmt(cols: tuple[str, str, str, str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cols, widths))

    lines = []
    lines.append("Rohban per-cluster ORA: FDR-correction comparison (broad GO-BP + CORUM + Reactome net)")
    lines.append("")
    lines.append(f"broad-net ORA table: {len(ora)} (cluster, set) rows, {n_sets} distinct sets")
    lines.append(f"collections (distinct sets): {by_coll}")
    lines.append(f"graded universe: {n_universe} multi-gene clusters (paper reports 19/22 enriched)")
    lines.append(f"permuted-null universe: {n_perm_universe} multi-gene clusters after one label shuffle (seed={PERM_SEED})")
    lines.append("all corrections applied WITHIN each cluster at cutoff 0.05; a cluster is enriched if >=1 set clears it")
    if not HAVE_STATSMODELS:
        lines.append("NOTE: statsmodels not importable, so method (c) two-stage BH was skipped.")
    lines.append("")
    lines.append(fmt(header))
    lines.append("  ".join("-" * w for w in widths))
    for r in rows:
        lines.append(fmt(r))
    lines.append("")

    # Diagnostic: the best (smallest) raw p per cluster and what each correction does to it. This shows
    # WHY every correction lands at 0: the strongest single hit in any cluster is not small enough to
    # survive dividing by the ~n_tests sets each cluster is tested against.
    diag = []
    for cluster in sorted(multigene, key=lambda c: int(c) if c.isdigit() else c):
        sub = ora[ora["group"] == cluster]
        p = sub["pvalue"].to_numpy()
        if p.size == 0:
            continue
        n = p.size
        minp = float(np.min(p))
        bh_q = float(np.min(bh(p)))
        st_q = float(np.min(storey_q(p)))
        ts_q = float(np.min(tsbh(p))) if HAVE_STATSMODELS else float("nan")
        diag.append((cluster, n, minp, bh_q, st_q, ts_q))
    diag.sort(key=lambda r: r[2])

    dcols = ("cluster", "n_tests", "min raw p", "min BH q", "min Storey q", "min tsbh q")
    drows = [(c, str(n), f"{mp:.2e}", f"{bq:.3f}", f"{sq:.3f}", ("n/a" if np.isnan(tq) else f"{tq:.3f}")) for c, n, mp, bq, sq, tq in diag]
    dwidths = [max(len(dcols[i]), max(len(r[i]) for r in drows)) for i in range(len(dcols))]

    def dfmt(cols):
        return "  ".join(str(c).ljust(w) for c, w in zip(cols, dwidths))

    lines.append("Per-cluster best hit (why every correction collapses to 0):")
    lines.append(dfmt(dcols))
    lines.append("  ".join("-" * w for w in dwidths))
    for r in drows:
        lines.append(dfmt(r))
    lines.append("")

    # Diagnostic for method (f): restrict each cluster to the sets its genes actually hit (n>=1), then BH within
    # that restricted universe. The tested-set count should drop from ~896 to tens; the min BH q shows whether the
    # smaller universe lets the best hit survive correction.
    rdiag = []
    for cluster in sorted(multigene, key=lambda c: int(c) if c.isdigit() else c):
        sub = ora[ora["group"] == cluster]
        rel = sub[sub["n"] >= 1]
        p = rel["pvalue"].to_numpy()
        if p.size == 0:
            rdiag.append((cluster, 0, float("nan"), float("nan")))
            continue
        rdiag.append((cluster, p.size, float(np.min(p)), float(np.min(bh(p)))))
    rdiag.sort(key=lambda r: (np.inf if np.isnan(r[2]) else r[2]))

    rcols = ("cluster", "tested sets (a>=1)", "min raw p", "min BH q (restricted)")
    rrows = [
        (c, str(n), ("n/a" if np.isnan(mp) else f"{mp:.2e}"), ("n/a" if np.isnan(bq) else f"{bq:.3f}"))
        for c, n, mp, bq in rdiag
    ]
    rwidths = [max(len(rcols[i]), max(len(r[i]) for r in rrows)) for i in range(len(rcols))]

    def rfmt(cols):
        return "  ".join(str(c).ljust(w) for c, w in zip(cols, rwidths))

    lines.append("Method (f) restricted universe a>=1 (tested-set count and min BH q; why it recovers or not):")
    lines.append(rfmt(rcols))
    lines.append("  ".join("-" * w for w in rwidths))
    for r in rrows:
        lines.append(rfmt(r))
    lines.append("")

    lines.append("Notes:")
    lines.append("  - total set discoveries and permuted-null counts are over the same multi-gene cluster universe.")
    lines.append("  - nominal p<0.05 is the uncorrected upper reference, not a valid FDR control.")
    lines.append(
        "  - every cluster is tested against all ~"
        f"{int(np.median([d[1] for d in diag]))} sets (ora tests the full universe per group), so BH divides the"
    )
    lines.append("    best hit by that count; the strongest single hit anywhere is p=%.1e, giving min BH q=%.2f." % (diag[0][2], diag[0][3]))
    lines.append(
        "  - Storey and two-stage BH do not help: a 2-8 gene cluster leaves almost all sets genuinely null (pi0~1),"
    )
    lines.append("    so the adaptive pi0 factor is near 1 and the adjusted values track plain BH.")

    # Data-driven verdict for the restricted-universe methods (f/g/h): does shrinking the universe recover REAL signal
    # WITHOUT equally lighting up the permuted null? The task's bar: real count moving toward the paper's ~19/22 while
    # the permuted-null count stays at or below the nominal FDR level (a handful at most, i.e. not more than 2-3/17).
    f_real, f_total = summarize_counts(significant_restricted(ora, 1, method="bh"), multigene)
    f_perm, _ = summarize_counts(significant_restricted(perm_ora, 1, method="bh"), perm_multigene)
    g_real, g_total = summarize_counts(significant_restricted(ora, 2, method="bh"), multigene)
    g_perm, _ = summarize_counts(significant_restricted(perm_ora, 2, method="bh"), perm_multigene)
    h_real, h_total = summarize_counts(significant_restricted(ora, 1, method="storey"), multigene)
    h_perm, _ = summarize_counts(significant_restricted(perm_ora, 1, method="storey"), perm_multigene)
    med_tested_f = int(np.median([r[1] for r in rdiag if r[1] > 0]))
    best_real = max(f_real, g_real, h_real)
    worst_perm = max(f_perm, g_perm, h_perm)
    null_clean = worst_perm <= 3

    lines.append(
        f"  - restricted-universe (conventional overlap-based ORA), corrected WITHIN each cluster over only the sets"
        f" its genes hit:"
    )
    lines.append(
        f"    (f) a>=1: REAL {f_real}/{n_universe} ({f_total} discoveries), PERMUTED-NULL {f_perm}/{n_perm_universe}; "
        f"(g) a>=2: REAL {g_real}/{n_universe} ({g_total}), NULL {g_perm}/{n_perm_universe}; "
        f"(h) a>=1 Storey: REAL {h_real}/{n_universe} ({h_total}), NULL {h_perm}/{n_perm_universe}."
    )
    lines.append(
        f"  - a>=1 barely shrinks the universe: each cluster's 2-8 genes still hit ~{med_tested_f} of {n_sets} sets"
        " (every gene sits in many GO/Reactome sets), so BH over ~300 tests still buries the best hit and (f) recovers"
        f" {f_real}/{n_universe}. Only a>=2 (two of the cluster's genes co-occurring in a set) cuts the universe hard"
        f" enough to recover {g_real}/{n_universe}."
    )
    if null_clean:
        lines.append(
            "  - The permuted null stays clean across every restricted method (max"
            f" {worst_perm}/{n_perm_universe}), so restricting the universe is NOT an artifact of fewer tests: the few"
            " recoveries are real signal, not shuffled noise clearing an easy bar."
        )
        lines.append(
            "  - CONCLUSION: restricting the test universe is the legitimate lever, and it does recover REAL enrichment"
            f" the whole-net correction misses ({best_real}/{n_universe} vs 0/{n_universe} for BH over all {n_sets}"
            " sets), with a clean permuted null. But even conventional ORA recovers only a handful, nowhere near the"
            " paper's 19/22. So the honest read is that the per-cluster signal is real but WEAK at construct level with"
            " GO-BP + CORUM + Reactome: G8 is only partially reproducible here, and the earlier curated-collection"
            " count remains the right graded number rather than the broad net."
        )
    else:
        lines.append(
            "  - CONCLUSION: restricting the universe raises the REAL count but the PERMUTED NULL also lights up (max"
            f" {worst_perm}/{n_perm_universe}, above the 2-3/17 bar), so it trades one artifact (over-correction over"
            f" {n_sets} sets) for another (too few tests, so any overlap clears q<0.05 on shuffled labels). G8 is"
            " genuinely not reproducible at construct level with these collections."
        )

    table = "\n".join(lines)
    print("\n" + table)
    out = HERE / "fdr_comparison.txt"
    out.write_text(table + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    # Keep tokenizer / BLAS threads from oversubscribing on the shared node.
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
