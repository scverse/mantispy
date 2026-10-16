"""Build the draft optical pooled screen tutorial as a notebook (executed separately)."""

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Optical pooled screens

An optical pooled screen (OPS) delivers a CRISPR library to one dish of cells, images their morphology, and then
reads each cell's guide by in-situ sequencing. Every well holds thousands of perturbations at once, so the unit of
analysis is no longer the well but the guide, and the well becomes the replicate.

This page analyses a subset of **Aconcagua**, a HeLa screen of 5,299 essential and DepMap-dependency genes imaged in
four channels: DNA (DAPI), microtubules (alpha-tubulin), DNA damage (gamma-H2AX) and F-actin (phalloidin). The cells
were segmented and measured with [brieflow](https://github.com/cheeseman-lab/brieflow). The subset keeps all 250
non-targeting guides, four annotated positive controls, 50 strong and 50 weak known hits, and 396 random library
genes, as profiles of each guide in each of 46 wells.

Along the way it meets three problems that are easy to miss in a pooled screen: profiles built from different
numbers of cells, a calibration that reuses its own controls, and the DNA damage that Cas9 causes simply by cutting.

The data are from Di Bernardo et al., Nat Commun 17, 6997 (2026), [doi:10.1038/s41467-026-73643-7](https://doi.org/10.1038/s41467-026-73643-7),
originally screened in Funk et al., Cell 185, 4634 (2022), and are available through the CZ Biohub OPS Explorer.
""")

code("""
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import plotly.express as px
import scanpy as sc
from scipy.stats import kstest

import mantispy as mt

warnings.filterwarnings("ignore")
SEED = 0

# DRAFT: replaced by mt.ds.aconcagua() once the dataset is hosted.
DATA = "acon/aconcagua_guide_well.h5ad"
adata = ad.read_h5ad(DATA)
adata
""")

md("""
One row per guide per well: the median of that guide's cells in that well, normalized against the non-targeting
cells of the same well with {func}`~mantispy.pp.normalize`. Mitotic cells were kept at their natural share, so a
profile pools interphase and mitotic cells as the screen saw them, and each row records its mitotic fraction.
""")

code("""
obs = adata.obs
design = (
    obs.assign(kind=np.where(obs["Metadata_Control_Type"] == "negcon", "control", obs["Metadata_Tier"].astype(str)))
    .groupby("kind", observed=True)
    .agg(guides=("Metadata_Guide", "nunique"), profiles=("Metadata_Guide", "size"), median_cells=("Metadata_CellCount", "median"))
)
design
""")

md("""
## Positive controls

Before calling anything, the screen's own positive controls should move the channel they were chosen for. The
knockouts of H2AX, TUBB and CFL1 remove or disturb the very protein a stain detects.
""")

code("""
controls = (obs["Metadata_Control_Type"] == "negcon").to_numpy()
checks = {
    "H2AX": "Nuclei_Intensity_MeanIntensity_GH2AX",
    "TUBB": "Cells_Intensity_MeanIntensity_TUBULIN",
    "CFL1": "Cells_Intensity_MeanIntensity_PHALLOIDIN",
}
rows = []
for gene, feature in checks.items():
    values = adata[:, feature].X.ravel()
    for label, mask in ((gene, (obs["Metadata_Gene"] == gene).to_numpy()), ("non-targeting", controls)):
        rows.append(pd.DataFrame({"feature": feature, "group": label, "z": values[mask], "gene": gene}))
frame = pd.concat(rows)
fig = px.box(frame, x="gene", y="z", color="group", points=False, title="Positive controls move their own channel")
fig.update_layout(yaxis_title="robust z (guide x well profiles)")
fig
""")

md("""
All three shift clearly, in the expected direction for gamma-H2AX and tubulin, and up for phalloidin after cofilin loss.
The fourth annotated control, ECT2, shows no change in DNA content at this level or in the single cells, which is
worth remembering when it comes to hit calling.

## Feature selection

{func}`~mantispy.pp.feature_select` flags features and {func}`~mantispy.pp.subset_features` keeps them. Brieflow's
discrete features, such as foci and neighbour counts, have no spread among the control cells of some wells, and
{func}`~mantispy.pp.normalize` flags them in `var["degenerate_scale"]` because dividing by a zero spread would
inflate them without bound. Selection drops those first.
""")

code("""
print("degenerate:", int(adata.var["degenerate_scale"].sum()), "of", adata.n_vars)
mt.pp.feature_select(adata)
adata = mt.pp.subset_features(adata)
adata.var["feature_group"].value_counts()
""")

md("""
About half of what remains is PFTAS texture, a brieflow family without a CellProfiler equivalent. Distances computed
on all features would be dominated by it, which is one reason the hit calling below works on principal components.

## Is there a plate or well effect left?

Normalizing each well against its own non-targeting cells removes offsets between wells. Whether anything remains is
a question for the controls: if plate or well still explained their variation, a batch correction such as
{func}`~mantispy.pp.sphere` would be needed.
""")

code("""
ctrl = adata[(adata.obs["Metadata_Control_Type"] == "negcon").to_numpy()].copy()
ctrl.X = np.nan_to_num(ctrl.X)
sc.pp.pca(ctrl, n_comps=10, random_state=SEED)
pcs = ctrl.obsm["X_pca"]


def explained(labels):
    labels = pd.Series(np.asarray(labels).astype(str))
    out = []
    for k in range(pcs.shape[1]):
        y = pcs[:, k]
        fitted = pd.Series(y).groupby(labels).transform("mean").to_numpy()
        out.append(1 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return np.array(out)


rng = np.random.default_rng(SEED)
plate = ctrl.obs["Metadata_Plate"].to_numpy()
well = (ctrl.obs["Metadata_Plate"].astype(str) + "_" + ctrl.obs["Metadata_Well"].astype(str)).to_numpy()
pd.DataFrame(
    {
        "plate": explained(plate),
        "plate, shuffled": explained(rng.permutation(plate)),
        "well": explained(well),
        "well, shuffled": explained(rng.permutation(well)),
    },
    index=[f"PC{k + 1}" for k in range(pcs.shape[1])],
).round(4)
""")

md("""
Plate and well explain no more of the control variation than shuffled labels do, so no further batch correction is
applied.

## Profiles must be built from comparable numbers of cells

A profile is a median of cells, and its noise falls with the number of cells behind it. If the controls were built
from more cells than the targeting guides, every targeting guide would look more active than the controls merely for
being noisier. The subset was drawn so that both have about ten cells per guide and well; the noise of a profile
against its cell count shows why that matching matters.
""")

code("""
spread = np.sqrt(np.nanmean(np.asarray(adata.X) ** 2, axis=1))
noise = pd.DataFrame(
    {
        "cells": adata.obs["Metadata_CellCount"].to_numpy(),
        "spread": spread,
        "kind": np.where(adata.obs["Metadata_Control_Type"] == "negcon", "non-targeting", "targeting"),
    }
)
noise["cell bin"] = pd.cut(noise["cells"], [4, 7, 10, 14, 20, 45]).astype(str)
summary = noise.groupby(["cell bin", "kind"], observed=True)["spread"].median().reset_index()
fig = px.line(summary, x="cell bin", y="spread", color="kind", markers=True, title="Profile noise falls with cell count")
fig.update_layout(yaxis_title="RMS of a profile's z-scores")
fig
""")

md("""
At the same cell count, control and targeting profiles are equally noisy. The extraction behind this dataset first
drew twice as many cells per control guide; with that mismatch, every random gene in the library was called a hit.

## From guides to genes

A gene is targeted by four guides, and its call should pool them. Each guide's profiles across the 46 wells are
first combined into one consensus profile.

The non-targeting guides then play three separate roles, split at random:

- **scale** (100 guides): {func}`~mantispy.tl.guide_activity` measures every guide against their spread,
- **null** (the remaining 102 guides): {func}`~mantispy.tl.aggregate_guides` calibrates each gene against random groups of four of them,
- **pseudo-genes** (12 groups of 4): scored like genes, they show how often the procedure calls a gene that does nothing.

Using the same guides for the scale and the null would score the null against itself.
""")

code("""
guides = mt.tl.aggregate(adata, by=("Metadata_Guide",), func="median", min_cells=1)
go = guides.obs
ntc = np.sort(go.loc[go["Metadata_Control_Type"] == "negcon", "Metadata_Guide"].astype(str).unique())
order = np.random.default_rng(SEED).permutation(ntc)
role = dict.fromkeys(order[:100], "ntc_scale") | dict.fromkeys(order[100:200], "nontargeting")
role |= {guide: f"PSEUDO{i // 4:02d}" for i, guide in enumerate(order[200:248])}
assigned = go["Metadata_Guide"].astype(str).map(role)
go["Analysis_Group"] = np.where(assigned.notna(), assigned, go["Metadata_Gene"].astype(str))
go["Analysis_Group"].str.replace(r"PSEUDO\\d+", "pseudo-gene", regex=True).value_counts().head(5)
""")

md("""
Activity is measured on 50 principal components, fitted on every guide without labels. Fitting them on the scale
guides alone would be a mistake that is easy to make: 50 components fitted to 100 guides describe those guides' own
noise, so in that space every other guide looks quieter than them, and held-out controls get p-values of one.
The check is simple: the held-out pseudo-gene guides should receive uniform p-values.
""")

code("""
guides.X = np.nan_to_num(guides.X)
sc.pp.pca(guides, n_comps=50, random_state=SEED)
mt.tl.guide_activity(guides, reference="ntc_scale", group="Analysis_Group", use_rep="X_pca")
pseudo = guides.obs["Analysis_Group"].str.startswith("PSEUDO").to_numpy()
p = guides.obs.loc[pseudo, "guide_activity"]
print(f"pseudo-gene guides: KS against uniform p = {kstest(p, 'uniform').pvalue:.2f}, share below 0.05 = {(p < 0.05).mean():.3f}")
""")

code("""
mt.tl.aggregate_guides(
    guides, score="guide_activity", guide="Metadata_Guide", gene="Analysis_Group", control="nontargeting", seed=SEED
)
calls = pd.DataFrame(guides.uns["mantispy"]["gene_aggregation"]).set_index("gene")
tier = dict(zip(go["Metadata_Gene"].astype(str), go["Metadata_Tier"].astype(str)))
calls["tier"] = ["pseudo-gene" if g.startswith("PSEUDO") else tier.get(g, "?") for g in calls.index]
calls.groupby("tier").agg(genes=("is_hit", "size"), called=("is_hit", "sum"), rate=("is_hit", "mean")).round(3)
""")

md("""
No pseudo-gene is called, and nearly every strong and weak known hit is. But more than half of the random genes are
called too. Some of that is real: the library holds only genes cells depend on. How much is real is the next question.

## Cas9 cutting is a phenotype of its own

A targeting guide makes Cas9 cut the genome; a non-targeting guide does not. The double-strand break alone raises
gamma-H2AX foci, whatever gene is cut. Genes the cells do not express are the cleanest test: knocking them out can do
nothing, so whatever they show is the cut. DepMap's expression for HeLa (model ACH-001086) names them.
""")

code("""
# DRAFT: download OmicsExpressionProteinCodingGenesTPMLogp1.csv from the DepMap data page (24Q4 on figshare, CC BY 4.0).
unexpressed = mt.io.unexpressed_genes("depmap/OmicsExpressionProteinCodingGenesTPMLogp1.csv", "ACH-001086")
calls["unexpressed"] = calls.index.isin(unexpressed)
print(f"{calls['unexpressed'].sum()} genes in this subset are not expressed in HeLa;")
print(f"{calls.loc[calls['unexpressed'], 'is_hit'].mean():.0%} of them are called against the non-targeting guides.")
""")

code("""
foci = "Nuclei_Intensity_MaxIntensity_GH2AX"
per_guide = pd.DataFrame(
    {
        "z": guides[:, foci].X.ravel(),
        "kind": np.where(go["Metadata_Control_Type"] == "negcon", "non-targeting",
                np.where(go["Metadata_Gene"].astype(str).isin(unexpressed), "unexpressed gene", go["Metadata_Tier"].astype(str))),
    }
)
fig = px.box(
    per_guide, x="kind", y="z", points=False,
    category_orders={"kind": ["non-targeting", "unexpressed gene", "random", "weak", "strong", "positive_control"]},
    title="Brightest gamma-H2AX focus per nucleus, per guide",
)
fig
""")

md("""
Guides against genes HeLa does not even express already carry more DNA damage foci than non-targeting guides. Genes
from multi-copy families, such as the olfactory receptor OR4F29 and AGAP9, carry more still, because their guides cut
at several loci at once.

The non-targeting null therefore answers *does this guide differ from not cutting*. The question a screen asks is
*does losing this gene do anything beyond the cut*, and the unexpressed genes are its null.
{func}`~mantispy.tl.empirical_fdr` calibrates each gene's combined statistic against them.
""")

code("""
genes = calls[calls["tier"] != "pseudo-gene"].copy()
table = ad.AnnData(obs=pd.DataFrame({"gene": genes.index, "score": genes["statistic"].to_numpy()}, index=genes.index))
mt.tl.empirical_fdr(table, control_genes=sorted(set(genes.index) & unexpressed), score="score", group="gene", alpha=0.05, criterion="q")
beyond = pd.DataFrame(table.uns["mantispy"]["empirical_fdr"]).set_index("group")
genes["beyond_cutting"] = beyond["is_hit"].reindex(genes.index).astype("boolean")
tested = genes[~genes["unexpressed"]]
tested.groupby("tier").agg(
    genes=("is_hit", "size"),
    vs_non_targeting=("is_hit", "mean"),
    beyond_cutting=("beyond_cutting", "mean"),
).round(3)
""")

md("""
Against the unexpressed genes, the strong known hits stay (about 98 %) while the random-gene rate falls from about
60 % to about 13 %. With only about 30 unexpressed genes in this subset the null is coarse: the smallest attainable
empirical p-value is about 1/33, which is why the cutoff is a q-value of 0.05.

## Mitosis is reported, not removed

Knockouts that arrest cells in mitosis also shift the pooled profile, because mitotic cells look different. Removing
that effect, or profiling interphase cells only, changes the calls very little here, and the activity of a gene is
nearly independent of its mitotic fraction. The mitotic fraction is best read beside each hit as its own phenotype.
""")

code("""
mitotic = adata.obs.groupby("Metadata_Gene", observed=True)["Metadata_MitoticFraction"].mean()
genes["mitotic_fraction"] = mitotic.reindex(genes.index)
fig = px.scatter(
    genes.reset_index(), x="statistic", y="mitotic_fraction", color="tier", hover_name="gene",
    title="Activity and mitotic fraction are separate readouts",
)
fig.add_hline(y=float(mitotic.get("nontargeting", np.nan)), line_dash="dot", annotation_text="non-targeting")
fig
""")

md("""
## Which channels moved

A hit call says that a gene does something; the features say what. Averaging the absolute z-scores of a gene's
profiles within each channel and compartment gives a compact signature.
""")

code("""
examples = ["RRM1", "DTL", "SKA2", "HAUS6", "ITGAV", "TLN1", "PSMB7", "SF3B3", "H2AX", "CFL1"]
profile = adata.to_df().groupby(adata.obs["Metadata_Gene"].astype(str).to_numpy()).median()
var = adata.var.assign(block=adata.var["object"].astype(str) + " " + adata.var["channel"].astype(str).replace("nan", "shape"))
signature = profile.loc[[g for g in examples if g in profile.index]].abs().T.groupby(var["block"].to_numpy()).mean().T
fig = px.imshow(signature, aspect="auto", color_continuous_scale="Blues", title="Mean |z| per compartment and channel")
fig
""")

md("""
Replication-stress genes (RRM1, DTL) light up gamma-H2AX and DNA, the spindle genes (SKA2, HAUS6) tubulin, and the
adhesion genes (ITGAV, TLN1) actin and shape.

## Comparison with the published calls

Brieflow publishes, for each gene, a bootstrap p-value per feature that resamples cells. Calling a gene when any of
its features passes FDR 0.05 treats every cell as an independent replicate, although cells in one well share that
well's conditions.
""")

code("""
# Brieflow's published interphase gene table, from the OPS Explorer bucket.
published = ad.read_h5ad("acon/brief/interphase.h5ad")  # DRAFT: fetched from the public bucket
threshold = published.uns["neg_log10_fdr_threshold"]
any_feature = pd.Series((published.layers["neg_log10_fdr"] > threshold).any(axis=1), index=published.obs_names)
ntc_rows = any_feature.index.str.startswith("nontargeting")
print(f"non-targeting guides called by brieflow (interphase): {any_feature[ntc_rows].mean():.0%}")
compare = genes.join(any_feature.rename("brieflow"), how="inner")
compare.groupby("tier").agg(
    genes=("brieflow", "size"),
    brieflow=("brieflow", "mean"),
    vs_non_targeting=("is_hit", "mean"),
    beyond_cutting=("beyond_cutting", "mean"),
).round(3)
""")

md("""
The strong and weak tiers were chosen from brieflow's own effect sizes, so brieflow recovering them is true by
construction and says nothing about either method. The fair comparisons are the controls and the random genes:
brieflow calls 17 % of its non-targeting guides and almost every random gene, including the unexpressed ones. With the
well as the replicate and the cut accounted for, about one random gene in eight remains. The two approaches agree on
the strong hits; they differ in what they count as evidence.

## What to take away

- In a pooled screen the guide is the unit and the well is the replicate.
- Build control and targeting profiles from comparable numbers of cells.
- Keep the guides that set the scale, form the null and test the calibration separate.
- A non-targeting null measures the effect of cutting; unexpressed genes measure what lies beyond it.
- Report the mitotic fraction beside the hits rather than folding it into them.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "mtenv", "display_name": "mantispy (scratch)", "language": "python"}
nbf.write(nb, "notebook/optical_pooled_screens.ipynb")
print(len(cells), "cells")
