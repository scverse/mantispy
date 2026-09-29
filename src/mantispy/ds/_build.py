"""Build the staged dataset variants that are rehosted on scverse-exampledata.

This is the single source of truth for the variants that the staged loaders fetch. Each entry in
:data:`STAGED` pairs a *builder* (rebuilds the variants from the raw pipeline) with a *shipped loader*
(returns the same filenames as loaded through the public ``mt.ds`` API) and a ``heavy`` flag (whether the
rebuild is too large to run on every pull request). ``scripts/build_staged_datasets.py`` writes what the
builders produce; ``scripts/check_staged_drift.py`` compares each shipped variant against a rebuild, so the
pipeline here and the uploaded bytes must not diverge. After a change to a builder, rebuild with
``--print-sha256``, re-upload to ``s3://scverse-exampledata/mantispy/<name>/`` and update the sha256 in
``registry.yaml``.

Imports of the public pipeline are deferred into the functions: this module lives inside the package, so a
top-level ``import mantispy`` would cycle through :mod:`mantispy.ds`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    import pandas as pd
    from anndata import AnnData


def _zero_nonfinite(adata: AnnData) -> AnnData:
    """Replace NaN and +-inf in ``X`` with zero, in place, and return the object."""
    from mantispy._core._reduce import get_matrix

    adata.X = np.nan_to_num(np.asarray(get_matrix(adata), dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return adata


def _stable(adata: AnnData) -> AnnData:
    """Rows sorted by ``obs_names`` and columns by ``var_names``, so a rebuild is order-stable."""
    rows = np.argsort(adata.obs_names.to_numpy(), kind="stable")
    cols = np.argsort(adata.var_names.to_numpy(), kind="stable")
    return adata[rows][:, cols].copy()


def _gene_consensus(block: AnnData, screened: np.ndarray) -> AnnData:
    """One modz consensus per gene over the screened wells of ``block``, NaN-zeroed."""
    from mantispy.tl._consensus import consensus

    gene = consensus(block[screened], by="Metadata_Gene", method="modz", correlation="spearman", min_replicates=2)
    return _zero_nonfinite(gene)


def _perturbation_consensus(block: AnnData) -> AnnData:
    """One modz consensus per ``Metadata_Perturbation`` over every well of ``block``, NaN-zeroed.

    Controls are kept: BBBC021's DMSO-at-a-concentration wells are legitimate perturbation units.
    """
    from mantispy.tl._consensus import consensus

    agg = consensus(block, by="Metadata_Perturbation", method="modz", correlation="spearman", min_replicates=2)
    return _zero_nonfinite(agg)


def build_rohban_base(cache_dir: str | Path | None = None) -> dict[str, AnnData]:
    """Returns {'rohban.h5ad': the raw wells the both-flags-False ``mt.ds.rohban`` used to assemble}."""
    from mantispy.ds._datasets import _rohban_raw

    return {"rohban.h5ad": _stable(_rohban_raw(cache_dir=cache_dir))}


def build_rohban_variants(cache_dir: str | Path | None = None) -> dict[str, AnnData]:
    """Returns {'rohban_selected.h5ad': ad, 'rohban_gene.h5ad': ad, 'rohban_gene_selected.h5ad': ad}."""
    from mantispy.ds._datasets import _rohban_raw
    from mantispy.pp._normalize import normalize
    from mantispy.pp._select import feature_select, subset_features

    adata = _rohban_raw(cache_dir=cache_dir)
    adata.obs["is_untreated"] = (adata.obs["Metadata_Perturbation_Type"].astype(str) == "untreated").to_numpy()
    normalize(adata, method="mad_robustize", by="Metadata_Plate", reference="is_untreated")
    _zero_nonfinite(adata)

    # Well-level feature-selected block, pycytominer's default operations.
    feature_select(adata)
    selected = _zero_nonfinite(subset_features(adata))

    # Gene-level consensus over the screened wells: untreated and transfection controls dropped.
    screened = (~adata.obs["is_untreated"].to_numpy()) & (~adata.obs["Metadata_Control"].to_numpy())
    return {
        "rohban_selected.h5ad": _stable(selected),
        "rohban_gene.h5ad": _stable(_gene_consensus(adata, screened)),
        "rohban_gene_selected.h5ad": _stable(_gene_consensus(selected, screened)),
    }


def _bbbc021_counts(paths: Sequence[Path]) -> pd.DataFrame:
    """Cells, and the fields that contributed them, per well, from the per-field ``Image.csv`` of the run the ljosa_2013 profiles aggregate."""
    import pandas as pd

    rows = []
    for path in paths:
        cells = pd.read_csv(path, usecols=["Count_Cells"])["Count_Cells"]
        rows.append((*path.parent.name.rsplit("-", 1), cells.sum(), (cells > 0).sum()))
    return pd.DataFrame(rows, columns=["Metadata_Plate", "Metadata_Well", "Metadata_CellCount", "Metadata_SiteCount"])


def _assemble_bbbc021(cache_dir: str | Path | None = None) -> AnnData:
    """Assemble the BBBC021 base object from the raw ljosa_2013 profiles, image table, MOA table and counts.

    This is the raw pipeline the both-flags-False ``mt.ds.bbbc021`` used to run before the base was staged; it
    lives here so the drift check can rebuild the hosted base from the raw inputs.
    """
    import pandas as pd

    from mantispy._core.frames import as_frame
    from mantispy._core.logging import get_logger
    from mantispy.ds._datasets import _files
    from mantispy.io._profiles import from_dataframe

    # The registry entry also lists the rehosted .h5ad variants the loader fetches; the rebuild needs only the
    # raw inputs, so those are filtered out before the positional unpack (profiles, images, moa, per-well counts).
    raw = _files("bbbc021", cache_dir, select=lambda name: not name.endswith(".h5ad"))
    profiles_path, images_path, moa_path, *fields = raw
    wells = (
        pd.read_csv(images_path)[
            [
                "Image_Metadata_Plate_DAPI",
                "Image_Metadata_Well_DAPI",
                "Image_Metadata_Compound",
                "Image_Metadata_Concentration",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "Image_Metadata_Plate_DAPI": "Metadata_Plate",
                "Image_Metadata_Well_DAPI": "Metadata_Well",
                "Image_Metadata_Compound": "Metadata_Compound",
                "Image_Metadata_Concentration": "Metadata_Concentration",
            }
        )
    )
    moa = pd.read_csv(moa_path).rename(
        columns={"compound": "Metadata_Compound", "concentration": "Metadata_Concentration", "moa": "Metadata_MOA"}
    )
    annotations = wells.merge(moa, on=["Metadata_Compound", "Metadata_Concentration"], how="left")

    profiles = pd.read_csv(profiles_path).rename(
        columns={"Image_Metadata_Plate": "Metadata_Plate", "Image_Metadata_Well": "Metadata_Well"}
    )
    merged = profiles.merge(annotations, on=["Metadata_Plate", "Metadata_Well"], how="left").merge(
        _bbbc021_counts(fields), on=["Metadata_Plate", "Metadata_Well"], how="left"
    )
    if unmatched := int(merged["Metadata_Compound"].isna().sum()):
        get_logger().warning("%d wells have no compound annotation and are dropped", unmatched)
        merged = merged[merged["Metadata_Compound"].notna()]

    adata = from_dataframe(merged, resolution="well")
    obs = as_frame(adata.obs)
    obs["Metadata_Control"] = (obs["Metadata_Compound"] == "DMSO").to_numpy()
    obs["Metadata_Perturbation"] = pd.Categorical(
        obs["Metadata_Compound"].astype(str) + "@" + obs["Metadata_Concentration"].astype(str)
    )
    obs["Metadata_Perturbation_Type"] = pd.Series("compound", index=obs.index, dtype="category")
    adata.uns["mantispy"]["dataset"] = "BBBC021"
    get_logger().info("BBBC021: %d wells x %d features", adata.n_obs, adata.n_vars)
    return adata


def build_bbbc021_variants(cache_dir: str | Path | None = None) -> dict[str, AnnData]:
    """Returns {'bbbc021.h5ad': ad, 'bbbc021_selected.h5ad': ad, 'bbbc021_agg.h5ad': ad, 'bbbc021_agg_selected.h5ad': ad}."""
    from mantispy.pp._select import feature_select, subset_features

    base = _stable(_assemble_bbbc021(cache_dir))

    # Well-level feature-selected block, pycytominer's default operations, on the raw (author-normalized) base.
    selected = base.copy()
    feature_select(selected)
    selected = _stable(subset_features(selected))

    # One modz consensus per compound-at-concentration, on the full and on the feature-selected block. DMSO
    # wells are kept: their concentration series are legitimate perturbation units, not a normalization control.
    return {
        "bbbc021.h5ad": base,
        "bbbc021_selected.h5ad": selected,
        "bbbc021_agg.h5ad": _stable(_perturbation_consensus(base)),
        "bbbc021_agg_selected.h5ad": _stable(_perturbation_consensus(selected)),
    }


def _shipped_rohban_base() -> dict[str, AnnData]:
    """The rohban base as fetched through the public ``mt.ds.rohban`` API (both flags False)."""
    from mantispy.ds._datasets import rohban

    return {"rohban.h5ad": rohban()}


def _shipped_rohban() -> dict[str, AnnData]:
    """The three rohban variants as fetched through the public ``mt.ds.rohban`` API."""
    from mantispy.ds._datasets import rohban

    return {
        "rohban_selected.h5ad": rohban(feature_selected=True),
        "rohban_gene.h5ad": rohban(aggregated=True),
        "rohban_gene_selected.h5ad": rohban(aggregated=True, feature_selected=True),
    }


def _shipped_bbbc021() -> dict[str, AnnData]:
    """The four bbbc021 variants as fetched through the public ``mt.ds.bbbc021`` API."""
    from mantispy.ds._datasets import bbbc021

    return {
        "bbbc021.h5ad": bbbc021(),
        "bbbc021_selected.h5ad": bbbc021(feature_selected=True),
        "bbbc021_agg.h5ad": bbbc021(aggregated=True),
        "bbbc021_agg_selected.h5ad": bbbc021(aggregated=True, feature_selected=True),
    }


def _assemble_neuropainting(cache_dir: str | Path | None = None) -> AnnData:
    """Astrocyte and neuron wells read down to the features the six plates share.

    The raw pipeline the both-flags-absent :func:`mt.ds.neuropainting` used before the base was staged; it lives
    here so the drift check can rebuild the hosted base from the raw per-plate tables.
    """
    from mantispy.ds._datasets import _profiles

    return _profiles("neuropainting", cache_dir, select=lambda name: not name.endswith(".h5ad"))


def build_neuropainting(cache_dir: str | Path | None = None) -> dict[str, AnnData]:
    """Returns {'neuropainting.h5ad': the wells the loader used to assemble}."""
    return {"neuropainting.h5ad": _stable(_assemble_neuropainting(cache_dir))}


def _shipped_neuropainting() -> dict[str, AnnData]:
    """The neuropainting base as fetched through the public ``mt.ds.neuropainting`` API."""
    from mantispy.ds._datasets import neuropainting

    return {"neuropainting.h5ad": neuropainting()}


def _assemble_chroma(cache_dir: str | Path | None = None) -> AnnData:
    """Assemble the chroma base: the extra-channel wells with their compound, concentration, MOA and control.

    The raw pipeline the loader used before the base was staged; it lives here so the drift check can rebuild
    the hosted base from the raw per-plate tables.
    """
    import numpy as np
    import pandas as pd

    from mantispy._core.frames import as_frame
    from mantispy._core.logging import get_logger
    from mantispy.ds._datasets import _profiles

    adata = _profiles("chroma", cache_dir, select=lambda name: not name.endswith(".h5ad"))
    obs = as_frame(adata.obs)
    control = (obs["Metadata_control_type"].astype(str) == "negcon").to_numpy()
    obs["Metadata_Control"] = control
    name = obs["Metadata_Common Name"].astype(str).to_numpy()
    dose = pd.to_numeric(obs["Metadata_mmoles_per_liter"], errors="coerce").to_numpy(dtype=float)
    compound = np.where(control, "DMSO", name)
    obs["Metadata_Compound"] = pd.Categorical(compound)
    obs["Metadata_Concentration"] = dose
    obs["Metadata_MOA"] = obs["Metadata_MoA"].astype("category")
    label = np.where(control, "DMSO", np.char.add(np.char.add(compound.astype(str), "@"), dose.astype(str)))
    obs["Metadata_Perturbation"] = pd.Categorical(label)
    obs["Metadata_Perturbation_Type"] = pd.Series("compound", index=obs.index, dtype="category")
    get_logger().info(
        "chroma: %d wells x %d features, %d compounds, %d control wells",
        adata.n_obs,
        adata.n_vars,
        int(pd.Series(name[~control]).nunique()),
        int(control.sum()),
    )
    return adata


def build_chroma(cache_dir: str | Path | None = None) -> dict[str, AnnData]:
    """Returns {'chroma.h5ad': the wells the loader used to assemble}."""
    return {"chroma.h5ad": _stable(_assemble_chroma(cache_dir))}


def _shipped_chroma() -> dict[str, AnnData]:
    """The chroma base as fetched through the public ``mt.ds.chroma`` API."""
    from mantispy.ds._datasets import chroma

    return {"chroma.h5ad": chroma()}


# One entry per staged dataset: (builder rebuilding the variants from the raw pipeline, loader returning the
# shipped variants through the public ``mt.ds`` API keyed by the same filenames, ``heavy`` marking a rebuild
# too large to run on every pull request). Later PRs stage a dataset by adding one entry here; the build and
# drift scripts loop this registry and need no per-dataset edit. ``heavy`` datasets rebuild on the cron drift
# job only (``check_staged_drift.py --include-heavy``); the rest rebuild on every PR.
STAGED: dict[str, tuple[Callable[..., dict[str, AnnData]], Callable[[], dict[str, AnnData]], bool]] = {
    "rohban": (build_rohban_variants, _shipped_rohban, False),
    "rohban_base": (build_rohban_base, _shipped_rohban_base, False),
    "bbbc021": (build_bbbc021_variants, _shipped_bbbc021, True),
    "neuropainting": (build_neuropainting, _shipped_neuropainting, False),
    "chroma": (build_chroma, _shipped_chroma, False),
}
