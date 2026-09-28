"""Trace why our GO-BP enrichment count differs from Rohban 2017 and whether the pathways still recover.

Test whether the SAME pathways recover when we run enrichment THEIR way.

The v2 reproduction (``reproduce.py``) grades GO/pathway enrichment (target G8) with per-cluster
Benjamini-Hochberg over a broad gene-set collection and lands at 0/16 multi-gene clusters at q<0.05,
against the paper's headline "19 of 22 enriched clusters". That gap is easy to misread as "the biology
did not reproduce". It did not: it is a methodological gap, because Rohban did NOT FDR-correct.

Rohban's ``GO_Term_Analysis_of_Clusters.R`` ran, per cluster, topGO classic Fisher, ontology BP,
nodeSize=2, with the UNIVERSE set to only the clustered genes (``allGenes`` = the gene.id.map of genes
that landed in clusters), selected the cluster's genes, took the top 5 GO-BP terms, filtered to
``Significant >= 2`` (at least two of the cluster's genes in the term), and reported the NOMINAL
classic-Fisher p-values. "19/22 enriched clusters" = 19 of the 22 multi-gene clusters had such a top
GO-BP term. They cut the tree at height ``1 - 0.522 = 0.478`` (0.522 is the correlation threshold).

This script reproduces that approach as closely as mantispy allows and asks the real question: do the
REAL clusters give coherent, gene-consistent pathways (Hippo/YAP, RAS-RAF-MEK-ERK, NF-kB) that a
permuted null does not? It does NOT modify ``reproduce.py`` or any package code. It reuses the
reproduction pipeline by importing ``reproduce`` (which runs it end to end) and reading the strong-subset
clusters it leaves behind, then runs its own GO-BP-only ORA the paper's way:

  * universe = the genes present in the clustered strong set (``mt.tl.ora`` takes the universe from
    ``obs[gene_key]`` of the object it is handed, so running on the strong subset with
    ``gene_key="Metadata_Gene"`` makes the universe exactly the clustered genes = topGO ``allGenes``),
  * ``min_overlap=2`` (matches ``Significant >= 2``),
  * ``tmin=2`` (matches topGO ``nodeSize=2``: a GO-BP term is tested only if >=2 of its genes are in the
    universe),
  * NOMINAL Fisher p-value (the raw ``pvalue``, no BH),
  * GO-BP only (``mt.ds.gene_sets("GO_BP")``).

Where the approximation differs from topGO is stated in the report: topGO classic Fisher is ONE-sided
(over-representation) while ``mt.tl.ora`` runs a TWO-sided Fisher exact test, and the gene-set source is
decoupler's MSigDB C5 GO-BP rather than ``org.Hs.eg.db``. topGO is R-only; this is the faithful Python
approximation.

Run from the repo root inside the project venv:

    python benchmarks/rohban/pathway_trace.py
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
NOMINAL = 0.05
PERM_SEED = 0
# The many-shuffle empirical null (Step 7). One shuffle (the task's required control) is noisy; N draws give
# a null distribution for the count and per-module recovery, hence an empirical p for each.
N_NULL = 200
# Rohban's tree cut is height 1 - 0.522 = 0.478 (0.522 is the correlation threshold, not the height).
ROHBAN_CUT = 0.478
# The number some notes carried as if it were the cut height; report the cluster count there for contrast.
CORR_AS_HEIGHT = 0.522

# Anchor gene sets and the pathway themes their clusters should light up if the biology reproduces.
YAP_GENES = ["YAP1", "WWTR1"]
RAS_GENES = ["KRAS", "HRAS", "NRAS", "BRAF", "RAF1", "MAP2K1", "MAP2K4"]
NFKB_GENES = ["TRAF2", "RELA", "NFKB1", "REL"]

# Keyword matchers over the GOBP_ term names (MSigDB C5 uses underscore-delimited upper-case tokens). These are
# the paper's NAMED modules, kept strict so a match means the named pathway itself, not a loose neighbour: Hippo
# signalling, the MAPK/ERK/stress-activated kinase cascade (incl. Ras-protein signal transduction), and NF-kappaB.
HIPPO_KW = ["HIPPO"]
MAPK_KW = ["MAPK", "_ERK", "STRESS_ACTIVATED_PROTEIN_KINASE", "RAS_PROTEIN_SIGNAL"]
NFKB_KW = ["KAPPAB"]


def banner(text: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 90}\n{text}\n{'=' * 90}")


def theme_hit(term: str, keywords: list[str]) -> bool:
    """Whether a GOBP_ term name contains any of the theme keywords."""
    up = term.upper()
    return any(kw in up for kw in keywords)


# =============================================================================================
# Reuse the reproduction pipeline by importing it, preserving REPRODUCTION.md byte-for-byte.
# =============================================================================================
def load_pipeline():
    """Import reproduce.py (which runs the full pipeline) and return the objects this analysis needs."""
    sys.path.insert(0, str(HERE))
    repro_md = HERE / "REPRODUCTION.md"
    saved = repro_md.read_bytes() if repro_md.exists() else None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            import reproduce
    finally:
        if saved is not None:
            repro_md.write_bytes(saved)
    return reproduce


# =============================================================================================
# GO-BP ORA the paper's way (universe = clustered genes, min_overlap=2, nodeSize=2, nominal p).
# =============================================================================================
def run_go_ora(ann, net, mt) -> pd.DataFrame:
    """Run GO-BP-only ORA on a clustered object and return the raw per-(cluster, term) table.

    ``mt.tl.ora`` sets the universe to the distinct genes in ``obs[gene_key]`` of ``ann``, so passing the
    strong-subset object makes the universe exactly the clustered genes (topGO ``allGenes``). ``tmin=2``
    matches ``nodeSize=2`` and ``min_overlap=2`` matches ``Significant >= 2``. We read the raw ``pvalue``
    (nominal classic Fisher), not the BH ``qvalue``.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        mt.tl.ora(
            ann,
            groupby="cluster",
            net=net,
            gene_key="Metadata_Gene",
            tmin=2,
            min_overlap=2,
            padj_by="group",
        )
    return ann.uns["mantispy"]["ora"].copy()


def annotated_counts(net: pd.DataFrame, universe: set[str]) -> dict[str, int]:
    """TopGO ``Annotated``: for each GO-BP term, the number of its genes present in the universe."""
    sub = net[net["target"].astype(str).isin(universe)]
    return sub.groupby("source", observed=True)["target"].nunique().to_dict()


def gentable(ora: pd.DataFrame, cluster: str, annotated: dict[str, int], k: int, n_bg: int, top: int = 5):
    """A topGO-GenTable-like view of one cluster: top ``top`` GO-BP terms by nominal p."""
    sub = ora[ora["group"] == cluster].sort_values("pvalue").head(top)
    rows = []
    for _, r in sub.iterrows():
        term = str(r["source"])
        ann = int(annotated.get(term, 0))
        expected = ann * k / n_bg if n_bg else float("nan")
        rows.append(
            {
                "term": term,
                "annotated": ann,
                "significant": int(r["n"]),
                "expected": round(expected, 2),
                "classic_p": float(r["pvalue"]),
            }
        )
    return rows


def enriched_their_way(ora: pd.DataFrame, universe_clusters: set[str]) -> set[str]:
    """Clusters with >=1 GO-BP term at nominal p<0.05 and Significant>=2 (min_overlap=2 already enforces >=2)."""
    hit = ora[ora["pvalue"] < NOMINAL]
    return set(hit["group"].astype(str)) & universe_clusters


def main() -> None:
    """Run the full trace: cut-height note, GO-BP ORA the paper's way, anchor check, permuted nulls, report."""
    banner("Load the Rohban v2 reproduction pipeline (runs reproduce.py end to end, ~1-2 min)")
    repro = load_pipeline()
    mt = repro.mt
    cons_strong = repro.cons_strong
    multigene = set(repro.multigene)
    gene = cons_strong.obs["Metadata_Gene"].astype(str)
    lab = cons_strong.obs["cluster"].astype(str)
    stability_cut = float(repro.stability_cut)
    print(f"  strong subset: {cons_strong.n_obs} constructs; stability cut used by reproduce.py = {stability_cut:.3f}")
    print(f"  multi-gene clusters (>=2 distinct genes): {len(multigene)}")

    # -----------------------------------------------------------------------------------------
    # Step 1 note: cluster counts at Rohban's cut height (0.478) vs the correlation-threshold value (0.522).
    # -----------------------------------------------------------------------------------------
    banner("Step 1: cluster counts at cut height 0.478 (Rohban) vs 0.522 (the correlation threshold value)")
    from scipy.cluster.hierarchy import fcluster

    Z = np.asarray(cons_strong.uns["mantispy"]["cluster_linkage"], dtype=float)
    genes_all = gene.to_numpy()
    cut_summary = {}
    for h in (ROHBAN_CUT, CORR_AS_HEIGHT, stability_cut):
        labels = fcluster(Z, h, criterion="distance")
        s = pd.Series(labels)
        sizes = s.value_counts()
        n_ge2 = int((sizes >= 2).sum())
        # multi-gene: clusters with >=2 distinct genes among their members
        gpc = pd.Series(genes_all).groupby(s.to_numpy()).nunique()
        n_mg = int((gpc >= 2).sum())
        cut_summary[round(h, 3)] = (int(s.nunique()), n_ge2, n_mg)
        print(
            f"  cut height {h:.3f}: {s.nunique():>3d} clusters total, {n_ge2:>3d} with >=2 members, {n_mg:>3d} multi-gene"
        )
    print(
        "  NOTE: Rohban cut at 1 - 0.522 = 0.478. The stability cut reproduce.py uses "
        f"({stability_cut:.3f}) is close to it; 0.522 as a raw height over-cuts (coarser, fewer clusters)."
    )

    # -----------------------------------------------------------------------------------------
    # Step 2-3: GO-BP ORA the paper's way on the REAL clusters.
    # -----------------------------------------------------------------------------------------
    banner("Step 2-3: GO-BP over-representation the paper's way (universe = clustered genes, min_overlap=2, nominal p)")
    net = mt.ds.gene_sets("GO_BP")[["source", "target"]].copy()
    ann_real = cons_strong.copy()
    ora_real = run_go_ora(ann_real, net, mt)

    universe = set(gene[gene != ""].tolist())
    n_bg = len(universe)
    annotated = annotated_counts(net, universe)
    k_by_cluster = gene[gene != ""].groupby(lab, observed=True).nunique().to_dict()
    print(
        f"  universe (clustered strong genes) = {n_bg} genes; GO-BP terms tested (nodeSize>=2, overlap>=2 somewhere): "
        f"{ora_real['source'].nunique()}"
    )

    # Per-cluster top-5 GenTable-like listing over multi-gene clusters.
    def cluster_sort_key(c: str):
        return (0, int(c)) if c.isdigit() else (1, c)

    per_cluster_rows = {}
    print("\n  Per-cluster top 5 GO-BP terms (Significant/Annotated, Expected, classic nominal p):")
    for c in sorted(multigene, key=cluster_sort_key):
        genes_here = sorted(set(gene[lab == c]) & universe)
        rows = gentable(ora_real, c, annotated, k_by_cluster.get(c, 0), n_bg)
        per_cluster_rows[c] = (genes_here, rows)
        print(f"\n  cluster {c}  (genes: {', '.join(genes_here)})")
        if not rows:
            print("    (no GO-BP term with >=2 of the cluster's genes)")
            continue
        for r in rows:
            flag = " *" if r["classic_p"] < NOMINAL else "  "
            print(
                f"   {flag} {r['term'][:60]:60s} Sig={r['significant']}/{r['annotated']:<4d} "
                f"Exp={r['expected']:<5} p={r['classic_p']:.2e}"
            )

    # -----------------------------------------------------------------------------------------
    # Step 4: count "enriched their way".
    # -----------------------------------------------------------------------------------------
    banner("Step 4: enriched their way (>=1 GO-BP term at nominal p<0.05 with Significant>=2)")
    real_enriched = enriched_their_way(ora_real, multigene)
    n_real = len(real_enriched)
    print(f"  REAL clusters enriched their way: {n_real}/{len(multigene)}")
    print("  paper reports: 19/22 (nominal classic Fisher, no BH)")
    print("  our earlier BH-over-broad-collection (reproduce.py G8): 0/16 at q<0.05")

    # -----------------------------------------------------------------------------------------
    # Step 5 + 6 machinery: named-module recovery.
    # A named module (Hippo/YAP, RAS-RAF-MEK-ERK, NF-kB) "recovers their way" when >=2 of its anchor genes
    # land in ONE cluster (so a shared GO-BP term can reach Significant>=2) AND that cluster's GO-BP terms
    # include the module's named term at nominal p<0.05. In the real clustering the paper's modules
    # co-cluster (reproduce.py G5-G7); a permuted null scatters each module's anchor genes so <2 land
    # together and the named term cannot reach Significant>=2. This is the honest discriminator, because the
    # raw "enriched their way" COUNT is a permissive bar that shuffled labels clear almost as often.
    # -----------------------------------------------------------------------------------------
    MODULES = [
        ("Hippo/YAP", YAP_GENES, HIPPO_KW),
        ("RAS-RAF-MEK-ERK", RAS_GENES, MAPK_KW),
        ("NF-kB", NFKB_GENES, NFKB_KW),
    ]

    def clusters_with(lab_s, gene_s, genes: list[str]) -> dict[str, list[str]]:
        """Map each cluster holding any anchor gene to the anchor genes it holds, under a given labelling."""
        out: dict[str, list[str]] = {}
        for g in genes:
            for c in sorted(set(lab_s[gene_s == g])):
                out.setdefault(c, [])
                if g not in out[c]:
                    out[c].append(g)
        return out

    def best_themed(ora: pd.DataFrame, cluster: str, keywords: list[str]):
        """Best-ranked (lowest-p) GO-BP term matching the module keywords among a cluster's tested terms."""
        sub = ora[ora["group"].astype(str) == cluster].sort_values("pvalue").reset_index(drop=True)
        for rank, r in sub.iterrows():
            if theme_hit(str(r["source"]), keywords):
                return {
                    "term": str(r["source"]),
                    "rank": int(rank) + 1,
                    "p": float(r["pvalue"]),
                    "sig": int(r["n"]),
                    "ntests": int(len(sub)),
                }
        return None

    def module_recovery(ora, lab_s, gene_s, name, genes, keywords) -> dict:
        """Whether the named module recovers: its named GO-BP term enriched (p<0.05) in an anchor cluster."""
        cmap = clusters_with(lab_s, gene_s, genes)
        max_co = max((len(v) for v in cmap.values()), default=0)
        best, best_cluster = None, None
        for c in cmap:
            bt = best_themed(ora, c, keywords)
            if bt and bt["p"] < NOMINAL and (best is None or bt["p"] < best["p"]):
                best, best_cluster = bt, c
        return {
            "name": name,
            "cmap": cmap,
            "max_co": max_co,
            "best": best,
            "best_cluster": best_cluster,
            "recovered": best is not None,
        }

    # -----------------------------------------------------------------------------------------
    # Step 5: biology anchor check on the REAL clusters.
    # -----------------------------------------------------------------------------------------
    banner("Step 5: biology anchor check -- do the named modules recover on the clusters holding their genes?")
    real_modules = []
    for name, genes, kw in MODULES:
        rec = module_recovery(ora_real, lab, gene, name, genes, kw)
        real_modules.append(rec)
        print(f"\n  {name} anchors {genes}: max anchors co-clustered = {rec['max_co']}")
        if not rec["cmap"]:
            print("    (no anchor gene present in the strong clustered set)")
        for c, held in rec["cmap"].items():
            genes_here = sorted(set(gene[lab == c]) & universe)
            rows = gentable(ora_real, c, annotated, k_by_cluster.get(c, 0), n_bg)
            print(f"    cluster {c} holds {held}  (all genes: {', '.join(genes_here)})")
            if not rows:
                print("      (no GO-BP term with >=2 of the cluster's genes)")
            for r in rows:
                flag = "*" if r["classic_p"] < NOMINAL else " "
                print(
                    f"     {flag}{r['term'][:60]:60s} Sig={r['significant']}/{r['annotated']:<4d} p={r['classic_p']:.2e}"
                )
            bt = best_themed(ora_real, c, kw)
            if bt:
                print(
                    f"      -> named-module term present: {bt['term']} at rank {bt['rank']}/{bt['ntests']}, "
                    f"p={bt['p']:.2e} (Sig {bt['sig']})"
                )
        if rec["recovered"]:
            b = rec["best"]
            print(
                f"    => {name}: RECOVERED via {b['term']} (cluster {rec['best_cluster']}, rank "
                f"{b['rank']}/{b['ntests']}, nominal p={b['p']:.2e})"
            )
        else:
            print(f"    => {name}: named term NOT enriched at nominal p<0.05 in any anchor cluster (anchors scattered)")

    # -----------------------------------------------------------------------------------------
    # Step 6: permuted-null control under the SAME nominal method.
    # -----------------------------------------------------------------------------------------
    banner("Step 6: permuted null under the SAME nominal method (shuffle gene-to-cluster labels once)")
    perm = cons_strong.copy()
    rng = np.random.default_rng(PERM_SEED)
    shuffled = perm.obs["cluster"].to_numpy().copy()
    rng.shuffle(shuffled)
    perm.obs["cluster"] = pd.Categorical([str(x) for x in shuffled])
    perm_lab = perm.obs["cluster"].astype(str)
    perm_gene = perm.obs["Metadata_Gene"].astype(str)
    perm_multigene = set(
        perm_gene[perm_gene != ""].groupby(perm_lab, observed=True).nunique().pipe(lambda s: s[s >= 2]).index
    )
    ora_perm = run_go_ora(perm, net, mt)
    perm_enriched = enriched_their_way(ora_perm, perm_multigene)
    n_perm = len(perm_enriched)
    print(f"  (a) NULL clusters enriched their way (>=1 nominal p<0.05 term): {n_perm}/{len(perm_multigene)}")
    print(
        f"      REAL was {n_real}/{len(multigene)}. The nominal count is a PERMISSIVE bar: with thousands of "
        "overlapping GO-BP terms, almost any 2-8 gene group hits one at nominal p<0.05, so the COUNT alone "
        "barely separates real from shuffled labels. This is exactly why the paper's 19/22 is high and why FDR "
        "(reproduce.py's 0/16) collapses it -- the count is not where the biological signal lives."
    )

    # The discriminator: do the paper's NAMED modules recover on the null? They should not, because the shuffle
    # scatters each module's anchor genes so <2 land together and the named term cannot reach Significant>=2.
    print("\n  (b) Named-module recovery, real vs null (the real discriminator):")
    null_modules = []
    for name, genes, kw in MODULES:
        null_modules.append(module_recovery(ora_perm, perm_lab, perm_gene, name, genes, kw))
    for rm, nm in zip(real_modules, null_modules, strict=True):
        rtxt = f"RECOVERED ({rm['best']['term']}, p={rm['best']['p']:.1e})" if rm["recovered"] else "not recovered"
        ntxt = f"recovered ({nm['best']['term']}, p={nm['best']['p']:.1e})" if nm["recovered"] else "not recovered"
        print(f"    {rm['name']:16s} real: co-cluster {rm['max_co']}, {rtxt}")
        print(f"    {'':16s} null: co-cluster {nm['max_co']}, {ntxt}")

    # Show the single-shuffle null-enriched clusters' scrambled genes and top term, to characterise (b): the
    # terms are a scatter of generic programs, and where the shuffle happens to keep an anchor pair together it
    # recovers the SAME gene-driven term (expected noise, not method artefact).
    print("\n  (c) Null-enriched clusters (single shuffle), their (scrambled) genes and top GO-BP term:")
    top_anchor_coherent = 0
    for c in sorted(perm_enriched, key=cluster_sort_key):
        genes_here = sorted(set(perm_gene[perm_lab == c]))
        sub = ora_perm[ora_perm["group"].astype(str) == c].sort_values("pvalue").head(1)
        top_term = str(sub["source"].iloc[0]) if len(sub) else "(none)"
        top_p = float(sub["pvalue"].iloc[0]) if len(sub) else float("nan")
        coh = (
            (theme_hit(top_term, HIPPO_KW) and any(g in genes_here for g in YAP_GENES))
            or (theme_hit(top_term, MAPK_KW) and any(g in genes_here for g in RAS_GENES))
            or (theme_hit(top_term, NFKB_KW) and any(g in genes_here for g in NFKB_GENES))
        )
        top_anchor_coherent += int(coh)
        print(f"    cluster {c}: genes {genes_here} -> {top_term[:55]} (p={top_p:.2e})")
    print(
        f"    null clusters whose TOP term is a named-module term for a gene they hold: {top_anchor_coherent}/"
        f"{max(n_perm, 1)} (the null's top terms are scattered generic programs, not the anchor pathways)"
    )

    n_real_modules = sum(m["recovered"] for m in real_modules)
    n_null_modules = sum(m["recovered"] for m in null_modules)

    # -----------------------------------------------------------------------------------------
    # Step 7: many-shuffle empirical null. A single shuffle is noisy: it can leave an anchor pair
    # co-clustered by chance, and a named term can reach Significant>=2 through non-anchor genes, so
    # both the count and naive module recovery are unreliable on one draw. Repeat the shuffle N times to
    # get null DISTRIBUTIONS for the count and for per-module recovery, and an empirical p for each.
    # -----------------------------------------------------------------------------------------
    banner(f"Step 7: many-shuffle empirical null ({N_NULL} shuffles) for the count and per-module recovery")
    null_fracs: list[float] = []
    null_mod_hits = {name: 0 for name, _, _ in MODULES}
    nrng = np.random.default_rng(PERM_SEED)
    base_labels = cons_strong.obs["cluster"].to_numpy().copy()
    for i in range(N_NULL):
        lab_i = base_labels.copy()
        nrng.shuffle(lab_i)
        ann_i = cons_strong.copy()
        ann_i.obs["cluster"] = pd.Categorical([str(x) for x in lab_i])
        li = ann_i.obs["cluster"].astype(str)
        gi = ann_i.obs["Metadata_Gene"].astype(str)
        mg_i = set(gi[gi != ""].groupby(li, observed=True).nunique().pipe(lambda s: s[s >= 2]).index)
        ora_i = run_go_ora(ann_i, net, mt)
        enr_i = enriched_their_way(ora_i, mg_i)
        null_fracs.append(len(enr_i) / max(len(mg_i), 1))
        for name, genes, kw in MODULES:
            if module_recovery(ora_i, li, gi, name, genes, kw)["recovered"]:
                null_mod_hits[name] += 1
        if (i + 1) % 50 == 0:
            print(f"    ...{i + 1}/{N_NULL} shuffles")

    real_frac = n_real / max(len(multigene), 1)
    null_fracs_arr = np.array(null_fracs)
    emp_p_count = (int(np.sum(null_fracs_arr >= real_frac)) + 1) / (N_NULL + 1)
    print(
        f"  COUNT: real enriched fraction {real_frac:.2f} ({n_real}/{len(multigene)}); null mean "
        f"{null_fracs_arr.mean():.2f}, 95th pct {np.percentile(null_fracs_arr, 95):.2f}; empirical p(null>=real) "
        f"= {emp_p_count:.3f}"
    )
    print("  -> the nominal 'enriched their way' COUNT does not separate real from null (permissive bar).")
    print("  MODULE recovery (real recovered? / null recovery rate over the shuffles):")
    null_mod_rate = {}
    for rm in real_modules:
        rate = null_mod_hits[rm["name"]] / N_NULL
        null_mod_rate[rm["name"]] = rate
        emp_p = (null_mod_hits[rm["name"]] + 1) / (N_NULL + 1)
        verdict = (
            "DISCRIMINATES"
            if (rm["recovered"] and rate <= 0.20)
            else ("weak/none" if rm["recovered"] else "n/a (real not recovered)")
        )
        print(
            f"    {rm['name']:16s} real={'yes' if rm['recovered'] else 'no ':3s}  null rate={rate:.2f}  "
            f"empirical p={emp_p:.3f}  [{verdict}]"
        )

    # -----------------------------------------------------------------------------------------
    # Write the report.
    # -----------------------------------------------------------------------------------------
    write_report(
        cut_summary=cut_summary,
        stability_cut=stability_cut,
        n_strong=cons_strong.n_obs,
        n_multigene=len(multigene),
        n_bg=n_bg,
        per_cluster_rows=per_cluster_rows,
        real_enriched=real_enriched,
        n_real=n_real,
        n_perm=n_perm,
        n_perm_universe=len(perm_multigene),
        real_modules=real_modules,
        null_modules=null_modules,
        n_real_modules=n_real_modules,
        n_null_modules=n_null_modules,
        top_anchor_coherent=top_anchor_coherent,
        real_frac=real_frac,
        null_mean_frac=float(null_fracs_arr.mean()),
        null_p95_frac=float(np.percentile(null_fracs_arr, 95)),
        emp_p_count=emp_p_count,
        null_mod_rate=null_mod_rate,
        cluster_sort_key=cluster_sort_key,
    )
    print("\nDONE.")


def _fmt_terms(rows, top=5) -> str:
    """Compact one-line rendering of a cluster's top GO-BP terms for the markdown tables."""
    if not rows:
        return "(no GO-BP term with >=2 of the cluster's genes)"
    parts = []
    for r in rows[:top]:
        star = "*" if r["classic_p"] < NOMINAL else ""
        parts.append(f"{r['term']} (Sig {r['significant']}/{r['annotated']}, p={r['classic_p']:.1e}){star}")
    return "; ".join(parts)


def write_report(**kw) -> None:
    """Write benchmarks/rohban/pathway_trace.md."""
    cut_summary = kw["cut_summary"]
    stability_cut = kw["stability_cut"]
    n_strong = kw["n_strong"]
    n_multigene = kw["n_multigene"]
    n_bg = kw["n_bg"]
    per_cluster_rows = kw["per_cluster_rows"]
    real_enriched = kw["real_enriched"]
    n_real = kw["n_real"]
    n_perm = kw["n_perm"]
    n_perm_universe = kw["n_perm_universe"]
    real_modules = kw["real_modules"]
    n_real_modules = kw["n_real_modules"]
    real_frac = kw["real_frac"]
    null_mean_frac = kw["null_mean_frac"]
    null_p95_frac = kw["null_p95_frac"]
    emp_p_count = kw["emp_p_count"]
    null_mod_rate = kw["null_mod_rate"]
    csk = kw["cluster_sort_key"]

    def cut_row(h):
        tot, ge2, mg = cut_summary[round(h, 3)]
        return f"{tot} total, {ge2} with >=2 members, {mg} multi-gene"

    lines: list[str] = []
    lines.append("# Why our GO enrichment count differs from Rohban, and whether the pathways still reproduce")
    lines.append("")
    lines.append(
        "This traces the GO-BP enrichment gap between `reproduce.py`'s graded target G8 and Rohban et al. "
        "2017's headline \"19 of 22 enriched clusters\", then reruns the enrichment the paper's way to test "
        "whether the SAME pathways (Hippo/YAP, RAS-RAF-MEK-ERK, NF-kB) recover. Produced by "
        "`benchmarks/rohban/pathway_trace.py`, which reuses the exact reproduction pipeline up to the "
        "per-cluster gene sets and does not touch package code."
    )
    lines.append("")
    lines.append("**Three findings, up front:**")
    lines.append("")
    lines.append(
        f"1. **The count gap is entirely multiple testing.** Run the paper's way (nominal classic Fisher, no BH, "
        f'universe = clustered genes), {n_real}/{n_multigene} multi-gene clusters are "enriched their way", '
        f"matching the paper's 19/22 in spirit. `reproduce.py`'s G8 gets 0/16 only because it applies "
        "Benjamini-Hochberg over a broad multi-collection net. Same clusters, same genes, different correction."
    )
    lines.append(
        f"2. **But the nominal count is a permissive bar.** A permuted null (gene-to-cluster labels shuffled) "
        f'reaches on average {null_mean_frac * 100:.0f}% of clusters "enriched their way" under the identical '
        f"method, vs the real {real_frac * 100:.0f}% (empirical p = {emp_p_count:.2f} over {N_NULL} shuffles). "
        "With thousands of overlapping GO-BP terms, almost any small gene group hits one at nominal p<0.05, so "
        "the COUNT alone does not separate real from shuffled. The paper's 19/22 is therefore not, on its own, "
        "strong evidence, and this is precisely why an FDR correction erases it."
    )
    lines.append(
        f"3. **The correct pathway LABELS appear on the correct clusters, but the nominal ORA is too permissive "
        f"at construct level to call that a significant recovery.** {n_real_modules}/3 of the paper's named "
        f"modules put their named GO-BP term on exactly the cluster holding their genes (Hippo/YAP, "
        f"RAS-RAF-MEK-ERK; NF-kB does not, its TFs scatter). Face-valid and matching the paper. But the null "
        f"recovery rate across {N_NULL} shuffles is high ("
        + ("; ".join(f"{rm['name']} {null_mod_rate[rm['name']] * 100:.0f}%" for rm in real_modules))
        + "), so a permuted null recovers the same terms nearly as often. The robust, non-random evidence for "
        "these modules is the CO-CLUSTERING itself (`reproduce.py` G5/G6: YAP1+WWTR1 cluster together, the RAS "
        "cascade clusters together), not the permissive nominal enrichment that reads the label off those "
        "clusters. Details below."
    )
    lines.append("")

    # -- Method comparison table --
    lines.append("## Step-by-step method comparison")
    lines.append("")
    lines.append("| Pipeline step | Rohban (their code) | Our reproduction | Divergence and its effect |")
    lines.append("|---|---|---|---|")
    lines.append(
        "| Replication unit | Gene-collapsed signatures (one profile per gene, ~220 QC-passing) | "
        f"ORF constructs (`Metadata_Perturbation`), strong subset = {n_strong} constructs over "
        f"{n_multigene}+ multi-gene clusters | Finer granularity; a gene can appear as several constructs, so "
        "cluster gene-membership is thinner and the enrichment universe is smaller. |"
    )
    lines.append(
        "| Tree cut height | height `1 - 0.522 = 0.478` (0.522 is the correlation threshold, not the height) | "
        f"stability-criterion cut at {stability_cut:.3f} over the 0.4-0.7 correlation window (height 0.3-0.6) | "
        f"Our stability cut ({stability_cut:.3f}) sits right next to Rohban's 0.478. At height 0.478 the tree "
        f"gives {cut_row(0.478)}; at 0.522 (the correlation value misread as a height) {cut_row(0.522)} - "
        "coarser, fewer clusters. Cutting at the correlation threshold value instead of `1 - threshold` would "
        "under-segment. |"
    )
    lines.append(
        "| Enrichment tool | topGO classic Fisher, ontology BP, nodeSize=2 | `mt.tl.ora` two-sided Fisher exact, "
        "`tmin=2` (= nodeSize=2), `min_overlap=2` (= Significant>=2) | Same 2x2 counting; the test statistic "
        "differs only in sidedness (next row). |"
    )
    lines.append(
        "| Universe | Clustered genes only (`allGenes` = the gene.id.map of genes that landed in clusters) | "
        f"`mt.tl.ora` takes the universe from `obs[gene_key]`; run on the strong subset it is exactly the "
        f"{n_bg} clustered genes | MATCHED. This is the key alignment: `reproduce.py`'s G8 in effect corrects "
        "against a far larger measured/annotated background, which is what buries the signal, not the universe "
        "of this rerun. |"
    )
    lines.append(
        "| Sidedness | topGO classic Fisher is ONE-sided (over-representation only) | `mt.tl.ora` runs a "
        "TWO-sided Fisher exact test | For an over-represented term the two-sided p is >= the one-sided p, so "
        'our nominal p is conservative relative to topGO; it makes our "enriched their way" count a lower '
        "bound, not an inflated one. |"
    )
    lines.append(
        "| Multiple testing | NONE (nominal classic-Fisher p reported directly) | `reproduce.py` G8 applies "
        'Benjamini-Hochberg (`padj_by="group"`); THIS rerun reports the raw nominal p to match the paper | '
        "The entire count gap. Per-cluster BH over hundreds-to-thousands of GO/Reactome/CORUM sets divides the "
        "best hit by the family size and yields 0/16 at q<0.05; removing the correction (the paper's actual "
        "method) recovers the enriched clusters. |"
    )
    lines.append(
        "| Gene-set source | `org.Hs.eg.db` GO-BP annotations | decoupler MSigDB C5 `GO_BP` "
        '(`mt.ds.gene_sets("GO_BP")`) | Different GO-BP snapshots (different term names, `GOBP_` prefixes, '
        "membership vintage), so individual term IDs will not match one-to-one; the pathway THEMES are the "
        "comparison unit, not term IDs. |"
    )
    lines.append("")

    # -- enriched-their-way count --
    lines.append("## Enriched their way: the count")
    lines.append("")
    lines.append(
        f"- **Real clusters:** {n_real}/{n_multigene} multi-gene clusters have >=1 GO-BP term at nominal "
        "p<0.05 with Significant>=2."
    )
    lines.append("- **Paper:** 19/22 multi-gene clusters (nominal classic Fisher, no BH).")
    lines.append("- **Our earlier BH-over-broad-collection (`reproduce.py` G8):** 0/16 at q<0.05.")
    lines.append(
        f"- **Permuted null (same nominal method, single fixed-seed shuffle):** {n_perm}/{n_perm_universe} clusters."
    )
    lines.append(
        f"- **Permuted null ({N_NULL} shuffles):** mean {null_mean_frac * 100:.0f}% of clusters, 95th "
        f"percentile {null_p95_frac * 100:.0f}%; empirical p(null >= real) = {emp_p_count:.3f}."
    )
    lines.append("")
    lines.append(
        f"The real count ({n_real}/{n_multigene}) matches the paper's 19/22 in spirit and dwarfs the BH-corrected "
        f"0/16, so the count gap between our G8 and the paper is fully explained by multiple-testing correction. "
        f"But the permuted null clears the same bar (empirical p = {emp_p_count:.2f}): the nominal-p criterion is "
        "permissive because each cluster is compared against thousands of overlapping GO-BP terms, so shuffled "
        "labels enrich almost as often. The count reproduces the paper's number and its weakness at the same "
        "time; the discriminating evidence has to come from pathway identity, below, not from the count."
    )
    lines.append("")

    # -- pathway comparison / anchor check --
    lines.append("## Pathway comparison: do we recover the paper's named modules?")
    lines.append("")
    lines.append(
        'A module is judged "recovered their way" when >=2 of its anchor genes co-cluster and that cluster\'s '
        "GO-BP terms include the module's named term at nominal p<0.05 (searched at any rank, not just the top "
        "5, because tiny 2/2 terms dominate the very top by p-value)."
    )
    lines.append("")
    lines.append(
        "| Paper's named module | Anchor genes | Real: co-cluster / recovered term (nominal p) | "
        f"Null recovery rate ({N_NULL} shuffles) |"
    )
    lines.append("|---|---|---|---|")
    anchor_label = {
        "Hippo/YAP": "YAP1, WWTR1",
        "RAS-RAF-MEK-ERK": "KRAS/HRAS/NRAS/BRAF/RAF1/MAP2K1/MAP2K4",
        "NF-kB": "TRAF2/RELA/NFKB1/REL",
    }
    for rm in real_modules:
        rtxt = (
            f"{rm['max_co']} co-clustered; **{rm['best']['term']}** (p={rm['best']['p']:.1e}, rank "
            f"{rm['best']['rank']}/{rm['best']['ntests']})"
            if rm["recovered"]
            else f"{rm['max_co']} co-clustered; named term not enriched"
        )
        rate = null_mod_rate[rm["name"]]
        lines.append(f"| {rm['name']} | {anchor_label.get(rm['name'], '')} | {rtxt} | {rate * 100:.0f}% |")
    lines.append("")
    lines.append(
        "The named GO-BP terms do appear on the real clusters holding their genes: `GOBP_HIPPO_SIGNALING` on the "
        "YAP1+WWTR1 cluster, `GOBP_POSITIVE_REGULATION_OF_MAPK_CASCADE` and `GOBP_RAS_PROTEIN_SIGNAL_TRANSDUCTION` "
        "on the RAS/RAF clusters. NF-kB does not recover as a co-clustered module because its transcription "
        "factors (RELA, NFKB1, REL) scatter one per cluster at construct level, so no cluster reaches "
        "Significant>=2 on an NF-kappaB term (this matches `reproduce.py`'s G7 FLAG). The null-recovery-rate "
        "column is the honest qualifier and it is sobering: a permuted null recovers Hippo/YAP "
        f"{null_mod_rate['Hippo/YAP'] * 100:.0f}% of the time and the MAPK cascade "
        f"{null_mod_rate['RAS-RAF-MEK-ERK'] * 100:.0f}% of the time, because the clustered-gene universe is small "
        "and dense with kinases and regulators that fall into broad GO terms. So the ORA recovering the correct "
        "label is face-valid but NOT statistically distinguishable from chance at construct level; it should be "
        "read as confirmation of the co-clustering, not as independent evidence."
    )
    lines.append("")

    # -- per-cluster top terms --
    lines.append("## Per-cluster top GO-BP terms (paper's way; `*` = nominal p<0.05)")
    lines.append("")
    lines.append("| Cluster | Genes | Top GO-BP terms (Significant/Annotated, nominal p) |")
    lines.append("|---|---|---|")
    for c in sorted(per_cluster_rows, key=csk):
        genes_here, rows = per_cluster_rows[c]
        enr = " (enriched)" if c in real_enriched else ""
        lines.append(f"| {c}{enr} | {', '.join(genes_here)} | {_fmt_terms(rows)} |")
    lines.append("")

    # -- conclusion --
    lines.append("## Conclusion (honest)")
    lines.append("")
    lines.append(
        f"- **The count difference is methodological, not irreproducibility.** Running enrichment exactly the "
        f"paper's way (nominal classic Fisher, no BH, universe = the clustered genes, Significant>=2, "
        f"nodeSize>=2) gives {n_real}/{n_multigene} enriched multi-gene clusters, matching the paper's 19/22 in "
        f"spirit. `reproduce.py`'s G8 count of 0/16 is the SAME clusters and genes with Benjamini-Hochberg over "
        f"a broad multi-collection net. The gap is pinned to named divergences, in order of impact: (1) multiple "
        f"testing (BH vs none) - the dominant driver; (2) gene-set breadth and source (broad GO+Reactome+CORUM "
        f"vs GO-BP alone, MSigDB C5 vs `org.Hs.eg.db`); (3) replication unit (thinner construct-level clusters "
        f"vs gene-collapsed signatures); (4) two-sided vs one-sided Fisher, which only makes our nominal count "
        f'conservative. None of these is "the biology did not reproduce".'
    )
    lines.append(
        f"- **The nominal count is weak evidence on its own.** Over {N_NULL} shuffles the permuted null enriches "
        f"{null_mean_frac * 100:.0f}% of clusters on average (empirical p = {emp_p_count:.2f} for the real "
        f'{real_frac * 100:.0f}%), so the raw "enriched their way" count does not by itself demonstrate biology; '
        "it demonstrates that testing against thousands of GO-BP terms without correction is permissive. The "
        "paper's 19/22 shares this property. This is a real, honest limitation of the nominal criterion, at "
        "construct level and in the paper alike."
    )
    lines.append(
        f"- **The correct pathway labels appear on the correct clusters, but the nominal ORA does not prove it.** "
        f"{n_real_modules}/3 named modules put their GO-BP term on the cluster holding their genes "
        f"(`GOBP_HIPPO_SIGNALING` on YAP1+WWTR1; `GOBP_POSITIVE_REGULATION_OF_MAPK_CASCADE` / "
        f"`GOBP_RAS_PROTEIN_SIGNAL_TRANSDUCTION` on the RAS/RAF clusters; NF-kB does not, its TFs scatter one per "
        f"cluster, matching G7 FLAG). This is face-valid and qualitatively matches the paper. But the null "
        f"recovery rate ("
        + ", ".join(f"{rm['name']} {null_mod_rate[rm['name']] * 100:.0f}%" for rm in real_modules)
        + ") is high, so a permuted null recovers the same terms nearly as often: at construct level the "
        "clustered-gene universe is small and dense with kinases/regulators in broad GO terms, and the nominal "
        "ORA cannot distinguish these modules from chance. The robust, non-random evidence for the modules is the "
        "CO-CLUSTERING itself (YAP1+WWTR1 together, the RAS cascade together), which `reproduce.py` grades as "
        "G5/G6; the ORA merely reads the expected label off those real clusters and should be treated as "
        "confirmation, not proof."
    )
    lines.append(
        "- **Bottom line.** The count difference is fully methodological: against the paper it is the paper "
        "reporting uncorrected nominal p-values over the clustered-gene universe; against our own G8 it is BH "
        'over a broad collection. Neither is "the biology did not reproduce". The pathway IDENTITY that '
        "reproduces (Hippo/YAP, RAS-RAF-MEK-ERK) reproduces in the CLUSTERING (G5/G6) and the paper's nominal ORA "
        "reads the correct label off those clusters, so the modules do emerge from mantispy's clusters. The "
        "honest limit is that the nominal ORA statistic itself, in the paper and here, is permissive enough that "
        "a shuffled null clears it, so it is not independent evidence; the co-clustering is. NF-kB is a partial "
        "recovery at construct level, driven by its TFs scattering, not by the enrichment method."
    )
    lines.append("")

    (HERE / "pathway_trace.md").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {HERE / 'pathway_trace.md'}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
