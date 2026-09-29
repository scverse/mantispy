"""Cell Painting Gallery accessions, downloaded once and read with :func:`mantispy.io.read_profiles`.

Every file is pinned by its sha256 in ``registry.yaml``, so a file changed upstream raises an error instead of loading different data.
Downloads land in :attr:`mantispy.settings.cache_dir`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from scverse_misc.datasets import fetch, parse_registry, register_loader

from mantispy._core.features import empty_annotation
from mantispy._core.frames import as_frame, categorize_metadata
from mantispy._core.logging import get_logger, report_drop
from mantispy._core.schema import SCHEMA_VERSION, stamp
from mantispy._settings import settings
from mantispy.io._jump import join_jump_annotation, read_jump
from mantispy.io._profiles import _UPSTREAM_COUNTS, _adopt_counts, from_dataframe, read, read_profiles, write
from mantispy.pp._select import subset_features

if TYPE_CHECKING:
    from anndata import AnnData
    from scverse_misc.datasets import DatasetEntry, DownloadCB
    from spatialdata import SpatialData

_BASE_URL, _DATASETS = parse_registry(Path(__file__).parent / "registry.yaml")
# scverse-misc registers loaders by type name across all packages in the process, so ours uses the package name.
_TYPE = "mantispy"

#: One plate from each of the eleven sources that ran Target-2, pinned so the default cannot move with the registry.
TARGET2_DEFAULT = (
    "1053600674",  # source_2
    "JCPQC051",  # source_3
    "BR00121438",  # source_4
    "ACPJUM012",  # source_5
    "110000294936",  # source_6
    "CP1-SC1-25",  # source_7
    "A1170384",  # source_8
    "GR00003394",  # source_9, the 1536-well plates
    "Dest210726-160150",  # source_10
    "LM37-70_1",  # source_11
    "CP-CC9-R1-29",  # source_13
)

#: Bumped whenever the assembled jump_cells object changes, so an older cached assembly is not reused.
_ASSEMBLY_VERSION = 3

_EXPORT_FOV = "BR00121438-J04-1"


@register_loader(_TYPE)
def _download(entry: DatasetEntry, target: Path, download: DownloadCB, /, **kwargs: object) -> list[Path]:
    """Download the files of `entry` that ``kwargs["select"]`` keeps, all of them without one, and return where they are."""
    select = cast("Callable[[str], bool] | None", kwargs.get("select"))
    return [Path(download(file)) for file in entry.files if select is None or select(file.name)]


def _files(name: str, cache_dir: str | Path | None = None, select: Callable[[str], bool] | None = None) -> list[Path]:
    return fetch(_DATASETS[name], cache_dir or settings.cache_dir, base_url=_BASE_URL, select=select)


def _plate(file_name: str) -> str | None:
    """The plate of a plate file named ``<batch>__<plate>__<file>``, or ``None`` when the name is not one (a rehosted variant)."""
    parts = file_name.split("__")
    return parts[1] if len(parts) >= 3 else None


def _plate_files(name: str, plates: Sequence[str] | None, cache_dir: str | Path | None) -> list[Path]:
    by_file = {file.name: _plate(file.name) for file in _DATASETS[name].files}
    known = {plate for plate in by_file.values() if plate is not None}
    if plates is not None and (unknown := sorted(set(plates) - known)):
        raise KeyError(f"{name} has no plate(s) {unknown}; available: {sorted(known)}")
    wanted = known if plates is None else set(plates)
    # Files that are not plate files (the rehosted variants) map to None and are never selected here.
    return _files(name, cache_dir, select=lambda file_name: by_file[file_name] in wanted)


def _read_counts(path: Path) -> pd.DataFrame:
    """The plate, well and counts, under mantispy's names, of a per-well table that may hold thousands of other columns."""
    header = pd.read_csv(path, nrows=0).columns
    wanted = [column for column in ("Metadata_Plate", "Metadata_Well", *_UPSTREAM_COUNTS) if column in header]
    counts = _adopt_counts(pd.read_csv(path, usecols=wanted, dtype={"Metadata_Plate": str}, engine="pyarrow"))
    return counts.drop(columns=list(_UPSTREAM_COUNTS), errors="ignore")


def _augmented(name: str, plates: Sequence[str] | None, cache_dir: str | Path | None) -> AnnData:
    """Read the ``*_augmented`` profiles of some plates, which already carry their platemap.

    Columns are intersected because plates of one screen can differ by a few features when a channel failed on one of them.
    """
    return read_profiles(_plate_files(name, plates, cache_dir), on_column_mismatch="intersect", resolution="well")


def _profiles(
    name: str, cache_dir: str | Path | None, select: Callable[[str], bool] | None = None, **kwargs: Any
) -> AnnData:
    """Stack the files of an accession that ``select`` keeps, with the reading arguments the registry records."""
    adata = read_profiles(
        _files(name, cache_dir, select=select), **{**_DATASETS[name].metadata.get("read", {}), **kwargs}
    )
    adata.uns["mantispy"]["dataset"] = _DATASETS[name].metadata["accession"]
    return adata


#: The (aggregated, feature_selected) combination each rehosted bbbc021 variant answers to.
_BBBC021_VARIANTS = {
    (False, False): "bbbc021.h5ad",  # 632 x 467, well level, all features
    (False, True): "bbbc021_selected.h5ad",  # well level, feature selected
    (True, False): "bbbc021_agg.h5ad",  # perturbation level (modz), all features
    (True, True): "bbbc021_agg_selected.h5ad",  # perturbation level (modz), feature selected
}


def bbbc021(
    cache_dir: str | Path | None = None, *, aggregated: bool = False, feature_selected: bool = False
) -> AnnData:
    """BBBC021, MCF-7 cells treated with small molecules, the standard mechanism-of-action benchmark.

    Well-level CellProfiler profiles (``cpg0010-caie-drugresponse``), joined to the compound, concentration and mechanism of action the Broad Bioimage Benchmark Collection publishes with the image set.
    The image set covers 113 compounds.
    This returns the annotated subset the benchmark uses: 38 compounds plus DMSO, 103 treatments (a compound at a concentration) across 12 mechanisms.

    The base and its variants are pre-built by ``scripts/build_staged_datasets.py`` and rehosted on ``scverse-exampledata``, so the loader fetches a single h5ad rather than reassembling the base from the raw per-well tables. The two flags select the variant:

    - both ``False``: the base, well-level profiles with every feature, the object the recipe tutorials start from.
    - ``feature_selected=True``: the well-level block after pycytominer-default feature selection.
    - ``aggregated=True``: one ``modz`` consensus (Spearman, ``min_replicates=2``) per ``Metadata_Perturbation`` over every well, DMSO-at-a-concentration included, so each compound-at-concentration is one profile.
    - ``aggregated=True, feature_selected=True``: that same consensus on the feature-selected block.

    Every well carries a mechanism, DMSO included; select treatments with ``adata[~adata.obs["Metadata_Control"]]``.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        aggregated: Return the perturbation-level ``modz`` consensus instead of the wells.
        feature_selected: Return the feature-selected block instead of all features.

    Returns:
        Wells by features at well resolution when both flags are ``False``, else the staged variant selected by the two flags, read with :func:`mantispy.io.read`.
        Every variant carries ``Metadata_Plate``, ``Metadata_Well``, ``Metadata_Compound``, ``Metadata_Concentration``, ``Metadata_MOA``, ``Metadata_Perturbation`` (compound at concentration), ``Metadata_Perturbation_Type`` (``"compound"``), ``Metadata_Control``, and ``Metadata_CellCount`` over the ``Metadata_SiteCount`` fields, of four imaged, that contributed cells.

    Raises:
        ValueError: ``aggregated`` or ``feature_selected`` is not a bool.

    References:
        :cite:t:`Caie_2010`, the image set.
        :cite:t:`Ljosa_2013`, these profiles and the benchmark.
        Images courtesy of Peter Caie and David Westwood, available from the Broad Bioimage Benchmark Collection :cite:p:`Ljosa_2012`.
    """
    for flag_name, flag in (("aggregated", aggregated), ("feature_selected", feature_selected)):
        if not isinstance(flag, bool):
            raise ValueError(f"{flag_name} must be a bool, got {type(flag).__name__}")
    target = _BBBC021_VARIANTS[aggregated, feature_selected]
    (path,) = _files("bbbc021", cache_dir, select=lambda name: name == target)
    return read(path)


#: The (aggregated, feature_selected) combination each rehosted rohban variant answers to.
_ROHBAN_VARIANTS = {
    (False, False): "rohban.h5ad",  # 1918 x 3616, well level, all features
    (False, True): "rohban_selected.h5ad",  # 1918 x 751, well level
    (True, False): "rohban_gene.h5ad",  # 190 x 3616, gene level, full features
    (True, True): "rohban_gene_selected.h5ad",  # 190 x 751, gene level, feature selected
}


def _rohban_raw(cache_dir: str | Path | None = None) -> AnnData:
    """Assemble the rohban base object (all five plates) from the raw ``*_augmented`` per-plate tables.

    This is the raw pipeline the both-flags-False :func:`rohban` used to run before the base was staged; it
    lives here so :func:`mantispy.ds._build.build_rohban_base` can rebuild the hosted base from the raw inputs.
    """
    adata = _augmented("rohban", None, cache_dir)
    obs = as_frame(adata.obs)
    role = obs["Metadata_ASSAY_WELL_ROLE"].astype(str)
    obs["Metadata_Control"] = (role == "CTRL").to_numpy()

    # The unit the screen varied is the ORF construct (Metadata_broad_sample), ~323 of them: the paper's
    # active set is construct-level, and several constructs can overexpress the same gene. The untreated
    # EMPTY wells carry no construct and form one "untreated" group; the control ORFs (Luciferase, LacZ,
    # eGFP) also carry no broad_sample, so they fall back to their human-readable Metadata_pert_name.
    construct = obs["Metadata_broad_sample"]
    gene = obs["Metadata_gene_name"].astype(str)
    allele = obs["Metadata_pert_name"].astype(str)
    untreated = (gene == "EMPTY").to_numpy()
    perturbation = np.where(
        construct.notna().to_numpy(),
        construct.astype(str).to_numpy(),
        np.where(untreated, "untreated", allele.to_numpy()),
    )
    obs["Metadata_Perturbation"] = pd.Categorical(perturbation)
    obs["Metadata_Perturbation_Type"] = pd.Categorical(np.where(untreated, "untreated", "orf"))
    obs["Metadata_Gene"] = gene.astype("category")
    obs["Metadata_Construct"] = construct
    obs["Metadata_Allele"] = allele.astype("category")
    adata.uns["mantispy"]["dataset"] = "cpg0017-rohban-pathways"
    get_logger().info(
        "rohban: %d wells x %d features, %d ORF constructs over %d genes",
        adata.n_obs,
        adata.n_vars,
        int(construct.nunique()),
        int(gene.nunique()),
    )
    return adata


def _subset_plates(adata: AnnData, name: str, plates: Sequence[str]) -> AnnData:
    """The wells of ``plates`` from an in-memory object, raising ``KeyError`` for a plate that is not present."""
    known = set(adata.obs["Metadata_Plate"].astype(str))
    if unknown := sorted(set(plates) - known):
        raise KeyError(f"{name} has no plate(s) {unknown}; available: {sorted(known)}")
    return adata[adata.obs["Metadata_Plate"].astype(str).isin(set(plates)).to_numpy()].copy()


def rohban(
    plates: Sequence[str] | None = None,
    cache_dir: str | Path | None = None,
    *,
    aggregated: bool = False,
    feature_selected: bool = False,
) -> AnnData:
    """An ORF overexpression screen, with the genes and cell counts that BBBC021 lacks.

    ``cpg0017-rohban-pathways``: U2OS cells, one ORF construct overexpressed per well, roughly ten replicate wells per construct over five plates.

    The base and its variants are pre-built by ``scripts/build_staged_datasets.py`` from the five raw plate tables and rehosted on ``scverse-exampledata`` (each under 7 MB), so the loader fetches a single h5ad rather than reassembling the base on every call. The two flags select the variant, which the variants build from per-plate ``mad_robustize`` normalization against the untreated wells:

    - both ``False``: the raw wells with every feature, 1,918 wells by 3,616 features.
    - ``feature_selected=True``: the well-level block after pycytominer-default feature selection, 1,918 wells by 751 features.
    - ``aggregated=True``: one ``modz`` consensus (Spearman, ``min_replicates=2``) per ``Metadata_Gene`` over the screened wells (untreated and transfection controls dropped), 190 genes by 3,616 features.
    - ``aggregated=True, feature_selected=True``: that same gene consensus on the feature-selected block, 190 genes by 751 features.

    Args:
        plates: Plate barcodes to load, all five when omitted.
            Only applies to the base; the hosted base is fetched once and subset to these plates in memory.
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        aggregated: Return the gene-level ``modz`` consensus instead of the raw wells.
        feature_selected: Return the feature-selected block instead of all features.

    Returns:
        The raw wells by features at well resolution when both flags are ``False``, else the staged variant selected by the two flags, read with :func:`mantispy.io.read`.

        The base object carries:

        ``Metadata_Perturbation``: the ORF construct (``Metadata_broad_sample``, ~323 of them), the unit the screen varied and what replicate wells share. Several constructs can overexpress the same gene, so this is finer than the gene; the paper's active set is construct-level. The control ORFs read as their ``Metadata_pert_name`` (``Luciferase_CTRL``, ``LacZ_CTRL``, ``eGFP_CTRL``) and the untreated EMPTY wells as ``"untreated"``.

        ``Metadata_Perturbation_Type``: ``"orf"`` for the overexpression constructs and controls, ``"untreated"`` for the EMPTY wells.

        ``Metadata_Gene`` (the overexpressed gene, 194 of them, so ``tl.pathway_coherence`` and ``tl.enrich_hits`` group by it), ``Metadata_Construct`` (the ``broad_sample``, missing on the controls and EMPTY wells) and ``Metadata_Allele`` (the human-readable ``pert_name``, which separates allele variants).

        ``Metadata_Control``, ``Metadata_CellCount``, ``Metadata_SiteCount`` and the screen's own ``Metadata_gene_name``, ``Metadata_GeneID`` and ``Metadata_ASSAY_WELL_ROLE``.

    Raises:
        KeyError: A plate is not one of the five.
        ValueError: ``aggregated`` or ``feature_selected`` is not a bool, or ``plates`` is given for a variant.

    Notes:
        ``Metadata_Control`` marks the wells transfected with a control ORF (Luciferase, LacZ and eGFP), the reference for normalization.
        The untreated wells (``Metadata_gene_name == "EMPTY"``) were never transfected and are not flagged; they carry ``Metadata_Perturbation == "untreated"``, so drop them if a gene-level analysis should not see them.
        Regroup replicates to the gene with ``groupby="Metadata_Gene"`` or ``tl.consensus(by="Metadata_Gene")``; the gene is never smeared into the perturbation id.

    References:
        :cite:t:`Rohban_2017`.
    """
    for flag_name, flag in (("aggregated", aggregated), ("feature_selected", feature_selected)):
        if not isinstance(flag, bool):
            raise ValueError(f"{flag_name} must be a bool, got {type(flag).__name__}")
    if (aggregated or feature_selected) and plates is not None:
        raise ValueError(
            "plates only applies to the raw wells; a pre-aggregated or feature-selected variant cannot be plate-subset"
        )
    target = _ROHBAN_VARIANTS[aggregated, feature_selected]
    (path,) = _files("rohban", cache_dir, select=lambda name: name == target)
    adata = read(path)
    if plates is not None:
        adata = _subset_plates(adata, "rohban", plates)
    return adata


def pki(plates: Sequence[str] | None = None, cache_dir: str | Path | None = None) -> AnnData:
    """Kinase inhibitors over a dose series, from the JUMP pilot.

    ``cpg0008-pki``: fifteen compounds over a seven-point dose range (eleven at three doses, four at one) in U2OS cells, over eight plates with 32 to 64 replicate wells per treatment.
    Downloads about 71 MB for all eight.

    Args:
        plates: Plate barcodes to load, all eight when omitted.
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.

    Returns:
        Wells by features at well resolution, with ``Metadata_Perturbation`` (compound at concentration), ``Metadata_Perturbation_Type`` (``"compound"``), ``Metadata_Compound``, ``Metadata_Concentration`` (the platemap's ``mmoles_per_liter``), ``Metadata_MOA``, ``Metadata_Control``, ``Metadata_CellCount`` and ``Metadata_SiteCount``.

    Raises:
        KeyError: A plate is not one of the eight.

    Notes:
        ``Metadata_Control`` marks the DMSO wells only.
        The positive controls (``Metadata_control_type == "poscon"``) are not flagged, because they are perturbations and should not be normalized against.
    """
    adata = _augmented("pki", plates, cache_dir)
    obs = as_frame(adata.obs)
    control = (obs["Metadata_control_type"].astype(str) == "negcon").to_numpy()
    obs["Metadata_Control"] = control
    # Control wells have no compound or dose, so without this each would become its own "nan@nan" perturbation.
    compound = np.where(control, "DMSO", obs["Metadata_broad_sample"].astype(str).to_numpy())
    dose = obs["Metadata_mmoles_per_liter"].to_numpy(dtype=float)
    obs["Metadata_Compound"] = pd.Categorical(compound)
    obs["Metadata_Concentration"] = dose
    obs["Metadata_MOA"] = obs.pop("Metadata_moa")
    label = np.where(control, "DMSO", np.char.add(np.char.add(compound.astype(str), "@"), dose.astype(str)))
    obs["Metadata_Perturbation"] = pd.Categorical(label)
    obs["Metadata_Perturbation_Type"] = pd.Series("compound", index=obs.index, dtype="category")
    adata.uns["mantispy"]["dataset"] = "cpg0008-pki"
    get_logger().info(
        "pki: %d wells x %d features, %d compounds x %d doses",
        adata.n_obs,
        adata.n_vars,
        int(obs.loc[~control, "Metadata_Compound"].nunique()),
        int(obs.loc[~control, "Metadata_Concentration"].nunique()),
    )
    return adata


def jump_target2(
    plates: Sequence[str] | None = TARGET2_DEFAULT, annotate: bool = True, cache_dir: str | Path | None = None
) -> AnnData:
    """JUMP-Target-2, one plate map run at many sites.

    The JUMP consortium :cite:p:`Chandrasekaran_2023` ran the same plate map in every participating laboratory, so differences between plates from different sources are technical.
    This makes it suited to studying batch and source effects.
    All 141 of its plates in ``cpg0016-jump`` are pinned here, from eleven sources and 107 batches, which lets :func:`~mantispy.tl.transport` separate a laboratory effect from a batch and a plate effect.
    source_9 ran it on 1536-well plates, the others on 384-well plates.

    Args:
        plates: Plate barcodes to load.
            The default takes one plate from each source, about 0.7 GB, most of it the per-well table each plate's cell counts are published in; ``None`` loads all 141, 9.4 GB.
        annotate: Join the JUMP annotation, which supplies ``Metadata_Perturbation`` and ``Metadata_Control``.
            Downloads another 14 MB.
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.

    Returns:
        One row per well at well resolution, carrying ``Metadata_Source``, ``Metadata_Batch``, ``Metadata_Plate``, ``Metadata_Well``, ``Metadata_CellCount``, ``Metadata_SiteCount`` and, when annotated, ``Metadata_JCP2022``, ``Metadata_Perturbation``, ``Metadata_Perturbation_Type`` (``"compound"``), ``Metadata_InChIKey`` and ``Metadata_Control`` (the DMSO wells).

    Raises:
        KeyError: A plate is not one of the 141.
    """
    paths = _plate_files("jump_target2", plates, cache_dir)
    # The profiles carry no count; each plate's backend table does, among 7,600 other columns.
    counts = pd.concat([_read_counts(path) for path in paths if path.suffix == ".csv"])
    profiles = [path for path in paths if path.suffix == ".parquet"]
    adata = read_jump(profiles, annotate=annotate, on_column_mismatch="intersect", platemap=counts)
    batches = {_plate(file.name): file.name.split("__")[0] for file in _DATASETS["jump_target2"].files}
    adata.obs["Metadata_Batch"] = [batches[str(plate)] for plate in adata.obs["Metadata_Plate"]]
    adata.uns["mantispy"]["dataset"] = "cpg0016-jump"
    get_logger().info(
        "jump_target2: %d wells x %d features from %d source(s)",
        adata.n_obs,
        adata.n_vars,
        adata.obs["Metadata_Source"].nunique(),
    )
    return adata


#: The rehosted pooled_rare variant each ``feature_selected`` flag answers to.
_POOLED_RARE_VARIANTS = {
    False: "pooled_rare.h5ad",  # 290 x 4417, per barcode, all features
    True: "pooled_rare_selected.h5ad",  # per barcode, feature selected
}


def pooled_rare(cache_dir: str | Path | None = None, *, feature_selected: bool = False) -> AnnData:
    """Pooled rare variants, 290 barcodes in a pooled screen.

    ``cpg0032-pooled-rare``, aggregated to one row per barcode rather than per well, so there is no plate or well in ``obs``.

    The base and its feature-selected block are pre-built by ``scripts/build_staged_datasets.py`` from the raw gene-normalized table and rehosted on ``scverse-exampledata``, so the loader fetches a single h5ad rather than reassembling it on every call.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        feature_selected: Return the block after pycytominer-default feature selection instead of all features.

    Returns:
        Barcodes by features, at perturbation resolution, read with :func:`mantispy.io.read`, with:

        ``Metadata_Perturbation``: the variant the barcode expresses (``Metadata_Foci_Barcode_MatchedTo_GeneCode``, 290 of them), the unit the library varied. It is a gene (``"ACTB"``) or a specific coding variant of it (``"ACTB E364K"``).

        ``Metadata_Perturbation_Type``: ``"orf"``, the variants being expressed from constructs.

        ``Metadata_Gene`` (the gene the variant belongs to, the token before the first space) and ``Metadata_Allele`` (the full variant label).

    Raises:
        ValueError: ``feature_selected`` is not a bool.
    """
    if not isinstance(feature_selected, bool):
        raise ValueError(f"feature_selected must be a bool, got {type(feature_selected).__name__}")
    target = _POOLED_RARE_VARIANTS[feature_selected]
    (path,) = _files("pooled_rare", cache_dir, select=lambda name: name == target)
    return read(path)


def neuropainting(cache_dir: str | Path | None = None) -> AnnData:
    """Astrocytes and neurons, 1,691 wells imaged at 20x and 63x.

    ``cpg0038-tegtmeyer-neuropainting``.
    One plate barcode appears in more than one batch, so the observations are not indexed by plate and well.

    The base is pre-built by ``scripts/build_staged_datasets.py`` from the six raw plate tables and rehosted on ``scverse-exampledata``, so the loader fetches a single h5ad rather than reassembling it on every call.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.

    Returns:
        Wells by features at well resolution, with ``Metadata_CellCount`` and ``Metadata_SiteCount``, read with :func:`mantispy.io.read`.

    Notes:
        No ``Metadata_Perturbation`` is set. This is a genotype-and-donor comparison (a control-versus-deletion contrast over patient and isogenic lines) rather than a reagent perturbation screen, and its genotype and line columns differ between plates, so the per-plate column intersect keeps none of them. Group with an explicit ``groupby=`` on the column the analysis needs.
    """
    (path,) = _files("neuropainting", cache_dir, select=lambda name: name == "neuropainting.h5ad")
    return read(path)


def chroma(cache_dir: str | Path | None = None) -> AnnData:
    """Alternative dyes, 3,455 wells across eight channels.

    ``cpg0029-chroma-pilot``, which images more channels than the five of the standard protocol.
    One plate barcode appears at several timepoints, so the observations are not indexed by plate and well.
    The plates hold a set of 91 compounds with known mechanisms; the point of the dataset is the extra dye channels rather than the compounds.

    The base is pre-built by ``scripts/build_staged_datasets.py`` from the raw per-plate tables and rehosted on ``scverse-exampledata``, so the loader fetches a single h5ad rather than reassembling it on every call.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.

    Returns:
        Wells by features at well resolution, with ``Metadata_CellCount`` and ``Metadata_SiteCount`` and, read with :func:`mantispy.io.read`:

        ``Metadata_Perturbation``: the compound at its concentration (``"<name>@<mmoles_per_liter>"``), with the ``negcon`` wells grouped as ``"DMSO"``.

        ``Metadata_Perturbation_Type``: ``"compound"``.

        ``Metadata_Compound`` (the compound's common name), ``Metadata_Concentration`` (the platemap's ``mmoles_per_liter``), ``Metadata_MOA`` (the mechanism) and ``Metadata_Control`` (the ``negcon`` wells).
    """
    (path,) = _files("chroma", cache_dir, select=lambda name: name == "chroma.h5ad")
    return read(path)


#: The spellings the four OASIS batches use for each column mantispy reads, since no two of them agree.
_OASIS_PLATEMAP_COLUMNS = {
    "Metadata_Compound": ("treatment", "Compound Name", "compound"),
    "Metadata_Concentration": ("concentration_uM", "assay_conc_uM", "compound_concentration"),
    "Metadata_CellLine": ("cell_line", "cell_type"),
}


def _decimals(value: float) -> int:
    """How many decimal places a level was written with, from its shortest exact repr."""
    return len(np.format_float_positional(value, trim="-").partition(".")[2])


def _aligned_doses(doses: pd.Series, compounds: pd.Series) -> pd.Series:
    """The dose each well was meant to get, where plate maps record one concentration to several precisions.

    The OASIS plate maps disagree on precision rather than on value: one batch writes the concentration the dilution actually produced, ``0.0152416`` uM, and another writes it rounded, ``0.015``.
    A level written to fewer decimals is therefore the same dose as the finer level that rounds to it.

    Matching runs within a compound, because a compound's levels are one dilution series and cannot collide.
    The one case it would get wrong is a ladder with two rungs inside a rounding step of each other, which a series coarser than two-fold never has.
    """

    def align(block: pd.Series) -> pd.Series:
        levels = np.sort(block.dropna().unique())
        levels = levels[levels > 0]
        places = [_decimals(level) for level in levels]
        lookup = {}
        for level, digits in zip(levels, places, strict=True):
            finer = [
                other
                for other, deeper in zip(levels, places, strict=True)
                if deeper > digits and round(other, digits) == level
            ]
            # The nearest one: rounding 0.006 up to 0.01 must not claim a level that was written as 0.01.
            lookup[level] = min(finer, key=lambda other: abs(other - level), default=level)
        return block.replace(lookup)

    # dropna=False, so a well whose compound is blank keeps the dose it was recorded with.
    return doses.groupby(compounds, observed=True, dropna=False).transform(align)


def _oasis_platemaps(cache_dir: str | Path | None) -> pd.DataFrame:
    """The plate maps of every OASIS batch, read down to plate, well, compound and concentration."""
    frames = []
    for path in _files("oasis_pilot", cache_dir, select=lambda name: name.endswith("__platemap.txt")):
        frame = pd.read_csv(path, sep="\t", dtype=str)
        found = {
            name: next((column for column in spellings if column in frame), None)
            for name, spellings in _OASIS_PLATEMAP_COLUMNS.items()
        }
        if found["Metadata_Compound"] is None or found["Metadata_Concentration"] is None:
            get_logger().warning("oasis_pilot: %s names no compound or concentration column", path.name)
            continue
        kept = pd.DataFrame(
            {
                "Metadata_plate_map_name": frame["plate_map_name"],
                "Metadata_Well": frame["well_position"],
                **{name: frame[column] if column else np.nan for name, column in found.items()},
            }
        )
        # Batches that name compounds in "Compound Name" leave it blank for controls and put DMSO or EMPTY in BROAD_ID.
        if "BROAD_ID" in frame:
            kept["Metadata_Compound"] = kept["Metadata_Compound"].fillna(frame["BROAD_ID"])
        frames.append(kept)
    platemap = pd.concat(frames, ignore_index=True)
    recorded = pd.to_numeric(platemap["Metadata_Concentration"], errors="coerce")
    platemap["Metadata_ConcentrationRecorded"] = recorded
    platemap["Metadata_Concentration"] = _aligned_doses(recorded, platemap["Metadata_Compound"])
    # One batch writes the line as HepRG and the others as HepaRG; two spellings would split every per-line grouping.
    platemap["Metadata_CellLine"] = platemap["Metadata_CellLine"].replace({"HepRG": "HepaRG"})
    return platemap


#: The rehosted OASIS-pilot variant each ``aggregated`` flag answers to (both need ``annotate=True``).
_OASIS_PILOT_VARIANTS = {
    False: "oasis_pilot.h5ad",  # 4604 x 99, well level, annotated
    True: "oasis_pilot_agg.h5ad",  # perturbation level (modz)
}


def oasis_pilot(annotate: bool = True, cache_dir: str | Path | None = None, *, aggregated: bool = False) -> AnnData:
    """OASIS pilot, 4,604 wells in U2OS and HepaRG, most compounds over a ten-point dose range.

    ``cpg0033-oasis-pilot``, twelve plates read down to the features they share, over four batches: two of assay development and two that dose 36 compounds in each cell line.
    It is the dose-response dataset of the package: 28 of those compounds carry six or more concentrations in both U2OS and HepaRG.

    The annotated base and its perturbation-level consensus are pre-built by ``scripts/build_staged_datasets.py`` from the raw profile and plate-map tables and rehosted on ``scverse-exampledata``, so the loader fetches a single h5ad rather than joining the plate maps on every call. Passing ``annotate=False`` still reads the raw profiles directly, without the annotation join.

    Args:
        annotate: Fetch the annotated base, which carries ``Metadata_Compound``, ``Metadata_Concentration``, ``Metadata_CellLine``, ``Metadata_Control`` (the DMSO wells), ``Metadata_Perturbation`` and ``Metadata_Perturbation_Type`` (``"compound"``).
            ``False`` reads the raw profiles without the plate-map join and cannot be aggregated.
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        aggregated: Return one ``modz`` consensus (Spearman, ``min_replicates=2``) per ``Metadata_Perturbation`` instead of the wells, DMSO kept as the ``"DMSO"`` perturbation. Needs ``annotate=True``.

    Returns:
        Wells by features at well resolution when ``aggregated`` is ``False``, else the perturbation-level consensus, indexed by plate and well, with ``Metadata_CellCount`` and ``Metadata_SiteCount``, read with :func:`mantispy.io.read`. The annotation columns above are present when ``annotate``.

    Raises:
        ValueError: ``aggregated`` is not a bool, or ``aggregated`` is asked for with ``annotate=False``.

    Notes:
        The four batches name their plate-map columns differently, so the join reads whichever of ``treatment``/``Compound Name``/``compound`` and ``concentration_uM``/``assay_conc_uM``/``compound_concentration`` each one carries.
        Concentrations are micromolar.

        The assay-development batch doses DMSO itself, so a control well there carries a concentration.
        ``Metadata_Control`` marks the compound, not the dose.

        The plate maps disagree on precision: one batch writes the concentration the dilution produced, ``0.0152416`` uM, and another writes it rounded, ``0.015``.
        ``Metadata_Concentration`` is the dose the well was meant to get, reading a coarser spelling as the finer level it rounds to within each compound, and it names the replicate groups.
        ``Metadata_ConcentrationRecorded`` keeps what the plate map wrote: read against that column, every dosed compound here carries eighteen levels where ten were plated, and a treatment's wells split across two spellings.
    """
    if not isinstance(aggregated, bool):
        raise ValueError(f"aggregated must be a bool, got {type(aggregated).__name__}")
    if aggregated and not annotate:
        raise ValueError("aggregated needs annotate=True: the consensus groups by the plate-map perturbation")
    if not annotate:
        return _profiles("oasis_pilot", cache_dir, select=lambda name: name.endswith(".csv.gz"))
    target = _OASIS_PILOT_VARIANTS[aggregated]
    (path,) = _files("oasis_pilot", cache_dir, select=lambda name: name == target)
    return read(path)


#: The feature sets JUMP-Lite publishes for one set of wells: five learned embeddings and the CellProfiler-equivalent ``cp_measure``.
JUMP_LITE_MODELS = ("openphenom", "dinov2", "dinov2_random", "subcell", "morphem", "cp_measure")


def jump_lite(
    model: str = "openphenom", annotate: bool = True, cache_dir: str | Path | None = None, **kwargs: Any
) -> AnnData:
    """JUMP-Lite Target-2: 1,536 wells, four imaging sites, one feature set at a time.

    ``cpg0016-jump``, the compact benchmark of :cite:t:`Munoz_2026`.
    Four plates of the JUMP Target-2 plate map, one from each of ``source_3``, ``source_4``, ``source_5`` and ``source_6``, so the four batches are four different laboratories running the same 302 compounds with 64 DMSO wells each.

    Every ``model`` covers the same 1,536 wells, which is what makes this a comparison rather than six datasets: the rows, their order and the metadata are identical and only the feature block changes.
    The files list the wells in a different order for every model, so the rows are sorted by source, plate and well.
    Five are learned embeddings and one, ``"cp_measure"``, is the CellProfiler-equivalent measurement of the same images.

    Args:
        model: Which feature set to read, one of ``ds.JUMP_LITE_MODELS``.
            ``"dinov2_random"`` is the same architecture with untrained weights, which is the null model the benchmark scores the others against.
        annotate: Join the JUMP well and compound tables, which name the compound of each well.
            Downloads about 14 MB once and caches it.
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features at well resolution, indexed by plate and well, with ``Metadata_Source``, ``Metadata_Batch``, ``Metadata_Plate``, ``Metadata_Well``, ``Metadata_CellCount`` and, when annotated, ``Metadata_JCP2022``, ``Metadata_Perturbation``, ``Metadata_Perturbation_Type`` (``"compound"``), ``Metadata_InChIKey`` and ``Metadata_Control``.

    Raises:
        ValueError: ``model`` is not one of ``ds.JUMP_LITE_MODELS``.

    Notes:
        A dimension of a learned embedding is a coordinate in the model's own basis, not a measurement with a name to parse, so for every model but ``"cp_measure"`` the annotation columns of ``var`` are supplied empty.
        Anything that reads ``var["feature_group"]`` or ``var["channel"]``, such as the feature families :func:`~mantispy.pl.effect_sizes` colours by, has nothing to work with on those.

        ``"cp_measure"`` is CellProfiler-style measurements and keeps its parsed compartment, feature group and channel.
        Its channel is the index cp_measure numbered its inputs by rather than the name of a stain, because the name lives in the acquisition metadata and not in the feature name.

        The embeddings are not normalized.
        They are the model's output on each well's images, so a per-plate control normalization is still the first step.

        The trained embeddings here carry the cell count in their leading components, where it can account for more of the variance than either the laboratory or the imaging site.
        The untrained ``"dinov2_random"`` does not, and neither does ``"cp_measure"``, whose per-cell measurements are averaged over the well.
        Measure it with :func:`~mantispy.metrics.evaluate_correction` before correcting for anything else, and read :doc:`/tutorials/multisite/learned_embeddings` on why removing it is not obviously right.

    References:
        :cite:t:`Munoz_2026`, :cite:t:`Chandrasekaran_2023`, :cite:t:`Weisbart_2024`.
    """
    if model not in JUMP_LITE_MODELS:
        raise ValueError(f"model must be one of {JUMP_LITE_MODELS}, got {model!r}")

    adata = _profiles("jump_lite", cache_dir, select=lambda name: name == f"{model}.parquet", **kwargs)
    if model != "cp_measure":
        empty = empty_annotation(adata.var_names)
        adata.var[empty.columns] = empty

    (counts_path,) = _files("jump_lite", cache_dir, select=lambda name: name == "cell_count.parquet")
    counts = pd.read_parquet(counts_path, columns=["Metadata_id", "cell_count"])

    obs = as_frame(adata.obs)
    joined = obs.merge(counts, on="Metadata_id", how="left", validate="1:1")
    joined.index = obs.index
    if unmatched := int(joined["cell_count"].isna().sum()):
        # Downstream, a missing count reads as a well with no cells.
        get_logger().warning("jump_lite(%s): %d well(s) have no cell count in the count table", model, unmatched)
    adata.obs = joined.rename(columns={"cell_count": "Metadata_CellCount"})
    order = as_frame(adata.obs).sort_values(["Metadata_Source", "Metadata_Plate", "Metadata_Well"], kind="stable").index
    adata = adata[order].copy()

    if annotate:
        from mantispy.pp._annotate import annotate_jump

        annotate_jump(adata)
    get_logger().info(
        "jump_lite(%s): %d wells x %d features over %d source(s)",
        model,
        adata.n_obs,
        adata.n_vars,
        int(as_frame(adata.obs)["Metadata_Source"].nunique()),
    )
    return adata


def jump_lite_targets(cache_dir: str | Path | None = None) -> pd.DataFrame:
    """The gene each JUMP compound is annotated to act on, as a set per target.

    RefChemDB annotations distributed with :cite:t:`Munoz_2026`, in the ``source``/``target`` shape :func:`~mantispy.metrics.known_relationships` reads, so two compounds annotated to the same gene count as a related pair.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.

    Returns:
        A frame with ``source``, the gene symbol, and ``target``, the ``Metadata_JCP2022`` of a compound annotated to it.
        One row per annotated pairing, over every JUMP compound rather than only those of :func:`jump_lite`.

    Notes:
        The annotation is sparse against a plate map: most compounds on the JUMP-Lite plates carry none, and only the targets shared by more than one compound contribute a pair, so the recall is computed over a minority of the plate.

    References:
        :cite:t:`Munoz_2026`.
    """
    (path,) = _files("jump_lite", cache_dir, select=lambda name: name == "refchem_annotations.parquet")
    frame = pd.read_parquet(path, columns=["target", "Metadata_JCP2022"])
    # Dropped before the cast, or every unannotated compound becomes the string "nan" and relates to every other.
    frame = frame.dropna().rename(columns={"target": "source", "Metadata_JCP2022": "target"}).astype(str)
    return frame.drop_duplicates().reset_index(drop=True)


def jump_crispr(annotate: bool = True, cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """The assembled JUMP CRISPR arm, 51,185 wells of knockouts in U2OS cells.

    ``cpg0016-jump-assembled``, the ``v1.0a`` well-position-corrected and feature-selected parquet, and the largest dataset here at 180 MB.
    jump-profiling-recipe has corrected it for well position and cell count, normalized it and selected its features, and has not yet sphered it.
    The cell counts are the ones the recipe regresses out of these profiles; their ``Cells_Count_Count`` feature was normalized with the rest and is no longer a count.

    Args:
        annotate: Join JUMP's CRISPR annotation, which names the gene each well's guides target and which wells are controls.
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, indexed by plate and well, with ``Metadata_JCP2022`` and ``Metadata_CellCount`` and, when annotated, ``Metadata_Gene`` and ``Metadata_Perturbation`` (the gene symbol; its guides are the replicates), ``Metadata_Perturbation_Type`` (``"crispr"``), ``Metadata_Control_Type`` (``"negcon"``, ``"poscon"`` or ``"trt"``), ``Metadata_Control`` (the no-guide and non-targeting wells) and ``Metadata_ChromosomeArm``.

    References:
        :cite:t:`Chandrasekaran_2023`.
    """
    (counts,) = _files("_jump_cell_counts", cache_dir)
    adata = _profiles("jump_crispr", cache_dir, platemap=_read_counts(counts), **kwargs)
    if annotate:
        from mantispy.pp._annotate import annotate_jump

        annotate_jump(adata, kind="crispr")
    return adata


def _finish_guide_screen(adata: AnnData, name: str) -> AnnData:
    """Record the accession, mark the perturbation type and log the shape shared by the single-cell guide screens."""
    # These are CRISPR guide screens: Metadata_Perturbation is the guide, set by each loader.
    adata.obs["Metadata_Perturbation_Type"] = pd.Series("crispr", index=adata.obs_names, dtype="category")
    adata.uns["mantispy"]["dataset"] = _DATASETS[name].metadata["accession"]
    get_logger().info(
        "%s: %d cells x %d features, %d gene(s) over %d guide(s)",
        name,
        adata.n_obs,
        adata.n_vars,
        adata.obs["Metadata_Gene"].nunique(),
        adata.obs["Metadata_sgRNA"].nunique(),
    )
    return adata


#: ``DAPI_FISH`` and ``DAPI_IF`` are the DAPI acquisitions of the DNA FISH and the immunofluorescence rounds.
_SCALLOPS_FEATURES = (
    "Nuclei_Intensity_MedianIntensity_ER",
    "Nuclei_Intensity_MedianIntensity_DAPI_IF",
    "Nuclei_Intensity_MedianIntensity_DAPI_FISH",
    "Nuclei_Spots_Count_ESR1",
    "Cells_Spots_Count_ESR1",
    "Nuclei_Spots_Count_CCND1",
    "Cells_Spots_Count_CCND1",
    "Nuclei_Spots_Count_GREB1",
    "Cells_Spots_Count_GREB1",
)

_SCALLOPS_SOURCE = (
    "gene_symbol",
    "sgRNA_id",
    "type",
    "plate",
    "well",
    "Condition",
    "Cells_Location_IntersectsBoundary_IF",
)


def scallops_arv471(cache_dir: str | Path | None = None) -> AnnData:
    """SCALLOPS ARV-471, single cells of an optical pooled screen under an estrogen-receptor degrader.

    The drug arm of a genome-scale optical pooled CRISPR screen from ``Genentech/scallops-manuscript``, its Figure 3 table.
    Cells express a guide library and are treated with ARV-471 (vepdegestrant), a PROTAC that recruits the CRL4-CRBN E3 ligase to the estrogen receptor and drives its degradation, then read by in-situ sequencing of the guide barcodes and a phenotype round that stains DNA and the estrogen receptor and counts ESR1, CCND1 and GREB1 transcripts.
    A guide that knocks out a gene the drug needs rescues the receptor, so cells carrying it keep the phenotype of an untreated cell.
    The genes with that known mechanism are the members of the ligase the PROTAC hijacks, ``CRBN``, ``DDB1``, ``CUL4A`` and ``CUL4B``, and ``ESR1`` itself, the drug's target.

    This loads only the ARV-471 condition, at single-cell resolution, so a hit is a guide whose cells sit away from the non-targeting cells in the phenotype space.
    The matched DMSO condition and the barcode-calling columns are left in the upstream file.
    Downloads about 205 MB once, checked against a pinned sha256, and subsets it on read.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.

    Returns:
        Cells by nine phenotype features at cell resolution, with:

        ``Metadata_Gene``: the gene the cell's guide targets, with the non-targeting guides written as ``"nontargeting"`` (the upstream ``NTC``), the spelling the analysis functions read.

        ``Metadata_sgRNA``: the guide identifier.

        ``Metadata_Perturbation``: the guide, so each guide is its own perturbation.

        ``Metadata_Perturbation_Type``: ``"crispr"``, the kind of screen this is.

        ``Metadata_Control_Type``: the schema's reserved control-type column, carrying the upstream ``type``, one of ``"target"`` (a screened gene), ``"ntc"`` (a non-targeting guide) or ``"neg"`` (a guide against an olfactory-receptor gene, a targeting negative control).
        The raw classes are kept rather than folded onto the reserved ``negcon``/``poscon``/``trt`` vocabulary, none of which fits the targeting negative cleanly.

        ``Metadata_Control``: ``True`` for the non-targeting guides, the reference :func:`~mantispy.tl.hit_calling` and normalization test against.
        The olfactory-receptor negatives are not flagged, so they can be scored as perturbations that should not move.

        ``Metadata_Plate``: the plate, ``A`` or ``B``.

        ``Metadata_Well``: the physical well, written as ``W03``.
        The raw well is an integer that the well vocabulary cannot parse, so it is padded and prefixed.
        The ARV-471 arm sits in one well per plate, so this is constant.

        The nine features are the ER and the two DAPI median intensities and the ESR1, CCND1 and GREB1 spot counts in the nucleus and the whole cell.
        The barcode, geometry (nucleus centers) and quality columns of the upstream table are dropped.

    Notes:
        Cells whose segmentation touches the field boundary (``Cells_Location_IntersectsBoundary_IF``) are cut off, so their intensities and spot counts undercount, and are dropped.
        Cells missing any phenotype feature are dropped too, so every returned cell has a full feature vector.

        A cell carries no count.
        Aggregate to a guide-level profile with ``mt.tl.aggregate(adata, by=("Metadata_Gene", "Metadata_sgRNA"))``, which writes ``Metadata_CellCount``.
    """
    (path,) = _files("scallops_arv471", cache_dir)
    df = pd.read_parquet(path, columns=[*_SCALLOPS_FEATURES, *_SCALLOPS_SOURCE])
    df = df[df["Condition"].astype(str) == "ARV-471"]
    df = df[~df["Cells_Location_IntersectsBoundary_IF"].astype(bool)]
    before = len(df)
    df = df.dropna(subset=list(_SCALLOPS_FEATURES))
    report_drop("cell(s) with a missing phenotype feature", before - len(df), before)

    gene = df["gene_symbol"].astype(str).to_numpy()
    guide = df["sgRNA_id"].astype(str).to_numpy()
    is_ntc = gene == "NTC"
    frame = df[list(_SCALLOPS_FEATURES)].reset_index(drop=True)
    frame["Metadata_Gene"] = np.where(is_ntc, "nontargeting", gene)
    frame["Metadata_sgRNA"] = guide
    frame["Metadata_Control_Type"] = df["type"].astype(str).to_numpy()
    frame["Metadata_Control"] = is_ntc
    frame["Metadata_Perturbation"] = guide
    frame["Metadata_Plate"] = df["plate"].astype(str).to_numpy()
    # The raw well is a rowless integer; prefix a synthetic row letter so from_dataframe's normalize_well can pad it.
    frame["Metadata_Well"] = ("W" + df["well"].astype(int).astype(str)).to_numpy()

    adata = from_dataframe(frame, resolution="cell")
    return _finish_guide_screen(adata, "scallops_arv471")


#: The upstream parquet stores these as its MultiIndex; every other column is a CellStats morphology feature.
_CP_POSH_METADATA = ("barcode", "gene_id", "treatment", "plate_well", "plate", "ID")

#: Upstream ``gene_id`` values of the non-targeting guides and the guides that cut an intergenic region.
_CP_POSH_CONTROLS = ("nontargeting", "intergenic")


def cp_posh(cache_dir: str | Path | None = None) -> AnnData:
    """Single cells of insitro cp-POSH, a broad-morphology pooled CRISPR Cell Painting screen.

    The 124-gene proof-of-concept dataset from ``insitro/cp-posh``: A549 cells carrying a pooled CRISPR-knockout library, stained with a six-channel Cell Painting panel (WGA, a mitochondrial probe, phalloidin, concanavalin A, DAPI and a marker round) and read by in-situ sequencing of the guide barcodes.
    Each cell gets a broad, untargeted morphology profile of about 1,278 CellStats features rather than the handful of hand-picked readouts a targeted screen keeps, so it is the broad-morphology complement to :func:`scallops_arv471`.

    The features are already well-normalized by the authors, so :func:`~mantispy.pp.normalize` is not needed before analysis; a per-plate control normalization would re-do work already done.
    Downloads about 1.6 GB once, checked against a pinned sha256.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.

    Returns:
        Cells by about 1,278 CellStats morphology features at cell resolution, indexed by the upstream cell ``ID``, with:

        ``Metadata_Gene``: the gene the cell's guide targets, taken from the upstream ``gene_id``.
        The two control classes keep their upstream spellings, ``"nontargeting"`` (the non-targeting guides) and ``"intergenic"`` (guides against intergenic regions); ``"nontargeting"`` is the spelling the analysis functions read.

        ``Metadata_sgRNA``: the guide, the upstream ``barcode``.

        ``Metadata_Perturbation``: the guide again, so each guide is its own perturbation, matching :func:`scallops_arv471`.

        ``Metadata_Perturbation_Type``: ``"crispr"``, the kind of screen this is.

        ``Metadata_Plate``: the plate, the upstream ``plate`` (``"EL37"``).

        ``Metadata_Well``: the physical well, such as ``"B04"``, taken from the upstream ``plate_well`` (``"EL37_B04"``) by dropping the plate prefix so the well vocabulary can parse it.

        ``Metadata_Control``: ``True`` for the non-targeting and intergenic guides, the reference :func:`~mantispy.tl.hit_calling` and the control normalization test against.

        The upstream ``treatment`` column is a constant (no small molecule) and is dropped, and the ``ID`` becomes the observation index.
        The known-mechanism genes, whose knockout moves cells away from the controls, are ``KIF18A``, the proteasome (``PSMB1``, ``PSMD4``), the mitochondrial ribosome (``MRPL43``, ``MRPS5``), the ARP2/3 complex (``ARPC4``, ``ACTR6``) and COPI (``COPE``, ``ARCN1``), scored against ``nontargeting`` and ``intergenic``.

    Notes:
        The CellStats feature names are insitro's own, not CellProfiler's ``<Object>_<Group>_<Feature>_<Channel>``, so the annotation columns of ``var`` are supplied empty rather than parsed.
        Left to the parser, a name such as ``nucleus_mask_height`` would read as the ``mask`` feature group of a ``nucleus`` object and invent feature families that are not there, the same reason the learned embeddings of :func:`jump_lite` carry an empty annotation.
        Anything that reads ``var["feature_group"]`` or ``var["channel"]`` has nothing to work with here.

        A cell carries no count.
        Aggregate to a guide-level profile with ``mt.tl.aggregate(adata, by=("Metadata_Gene", "Metadata_sgRNA"))``, which writes ``Metadata_CellCount``.
    """
    import anndata as ad

    (path,) = _files("cp_posh", cache_dir)
    df = pd.read_parquet(path).reset_index()
    features = [column for column in df.columns if column not in _CP_POSH_METADATA]

    gene = df["gene_id"].astype(str).to_numpy()
    guide = df["barcode"].astype(str).to_numpy()
    # plate_well is "<plate>_<well>", e.g. "EL37_B04"; the well is the segment after the last "_" (wells carry none).
    well = df["plate_well"].astype(str).str.rsplit("_", n=1).str[-1].to_numpy()
    obs = pd.DataFrame(
        {
            "Metadata_Gene": gene,
            "Metadata_sgRNA": guide,
            "Metadata_Perturbation": guide,
            "Metadata_Plate": df["plate"].astype(str).to_numpy(),
            "Metadata_Well": well,
            "Metadata_Control": np.isin(gene, _CP_POSH_CONTROLS),
        },
        index=pd.Index(df["ID"].astype(str).to_numpy()),
    )
    obs = categorize_metadata(obs)
    adata = ad.AnnData(X=df[features].to_numpy(dtype=np.float32), obs=obs, var=empty_annotation(features))
    stamp(adata, resolution="cell")
    return _finish_guide_screen(adata, "cp_posh")


def corum(cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Human protein complexes from CORUM, one row per complex and member gene.

    The complexes as the EFAAR benchmark of :cite:t:`Celik_2024` distributes them, in the ``source``/``target`` shape :func:`~mantispy.metrics.known_relationships` and :func:`~mantispy.tl.pathway_coherence` read, so two genes of one complex count as a related pair.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.

    Returns:
        A frame with ``source``, the complex, and ``target``, the symbol of a gene in it.

    Notes:
        CORUM is free for academic, non-commercial use, and the EFAAR copy is distributed under CC BY-NC 4.0.

    References:
        :cite:t:`Celik_2024`.
    """
    (path,) = _files("corum", cache_dir)
    lines = [line.split("\t") for line in path.read_text().splitlines() if line.strip()]
    rows = [(complex_name, gene) for complex_name, members in lines for gene in members.split()]
    return pd.DataFrame(rows, columns=["source", "target"]).drop_duplicates().reset_index(drop=True)


def _read_site(directory: Path, source: str, channels: Sequence[str]) -> AnnData:
    """One analysis directory as cells, checked against the ``<plate>-<well>-<site>`` name it is filed under.

    The JUMP pipeline gives ``Cytoplasm`` a ``Parent_Cells`` and a ``Parent_Nuclei`` and gives ``Cells`` no parent at all, so ``Cytoplasm`` is the only primary object that joins all three tables.
    One cytoplasm is one cell here, and the three tables have equal length.
    """
    adata = read_profiles(
        directory,
        primary_object="Cytoplasm",
        resolution="cell",
        channels=channels,
        index_columns=("Metadata_Plate", "Metadata_Well", "Metadata_Site", "Metadata_ObjectNumber"),
    )
    plate, well, site = directory.name.rsplit("-", 2)
    obs = as_frame(adata.obs)
    found = (str(obs["Metadata_Plate"].iloc[0]), str(obs["Metadata_Well"].iloc[0]), int(obs["Metadata_Site"].iloc[0]))
    if found != (plate, well, int(site)):
        raise ValueError(f"{directory.name} holds plate/well/site {found}, not what its name says")
    obs["Metadata_Source"] = source
    return adata


def _assemble_cells(entry: DatasetEntry, cache_dir: str | Path | None, *, annotate: bool) -> AnnData:
    """Read every analysis directory the entry pins and join them into one object."""
    import anndata as ad

    source = str(entry.metadata["source"])
    channels = [str(channel) for channel in entry.metadata["channels"]]
    paths = _files("jump_cells", cache_dir)
    parts = [_read_site(directory, source, channels) for directory in sorted({path.parent for path in paths})]
    # Every field of view numbers its own images from one, so each part's numbers are shifted past the ones before it.
    images, offset = [], 0
    for part in parts:
        table = part.uns["mantispy"]["image_table"]
        numbers = {number: offset + index + 1 for index, number in enumerate(table.index)}
        part.obs["Metadata_ImageNumber"] = part.obs["Metadata_ImageNumber"].astype(int).map(numbers)
        images.append(table.rename(index=numbers))
        offset += len(table)
    adata = ad.concat(parts, join="inner", merge="first", uns_merge="first")
    adata.uns["mantispy"]["image_table"] = pd.concat(images)
    n_parts, widest = len(parts), max(part.n_vars for part in parts)
    del parts
    report_drop("features measured in only some fields of view", widest - adata.n_vars, widest)

    if annotate:
        adata.obs = join_jump_annotation(as_frame(adata.obs)).set_axis(adata.obs_names)
        _mark_selected(adata)
    stamp(adata, resolution="cell")
    adata.uns["mantispy"]["dataset"] = entry.metadata["accession"]
    get_logger().info(
        "jump_cells: %d cells x %d features from %d field(s) of view in %d well(s)",
        adata.n_obs,
        adata.n_vars,
        n_parts,
        adata.obs["Metadata_Well"].nunique(),
    )
    return adata


def _select(adata: AnnData, path: Path) -> AnnData:
    """The selected features of `adata`, kept at `path` so the next call reads only those."""
    chosen = subset_features(adata)
    write(chosen, path)
    return chosen


def _mark_selected(adata: AnnData) -> None:
    """Write the feature-selection mask into ``var["selected"]``, as :func:`mantispy.pp.feature_select` does.

    The mask is computed on a normalized copy, so the values on `adata` stay as CellProfiler measured them.
    The recipe is the one the tutorials use.
    No step of it draws a random number, so the same pinned files always give the same mask.
    """
    from mantispy.pp._normalize import normalize
    from mantispy.pp._select import feature_select

    scratch = adata.copy()
    normalize(scratch, method="mad_robustize", by="Metadata_Plate", reference="negcon")
    scratch = scratch[:, ~scratch.var["degenerate_scale"].to_numpy()].copy()
    feature_select(scratch)
    kept = set(scratch.var_names[scratch.var["selected"].to_numpy()])
    adata.var["selected"] = adata.var_names.isin(kept)


def jump_export(cache_dir: str | Path | None = None) -> Path:
    """One real ``ExportToSpreadsheet`` directory, as CellProfiler wrote it.

    A single field of view of a DMSO well of ``BR00121438``: ``Image.csv`` plus the ``Cells``, ``Cytoplasm`` and ``Nuclei`` tables, unmodified, for reading with :func:`mantispy.io.read_profiles`.
    About 18 MB, and part of the download :func:`jump_cells` makes, so asking for both costs nothing extra.

    The images this was measured from are in :func:`jump_plate`, and the well-level profiles of the same plate are in :func:`jump_target2`.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.

    Returns:
        The directory, to pass to :func:`mantispy.io.read_profiles`.

    References:
        :cite:t:`Chandrasekaran_2023`.
    """
    paths = _files("jump_cells", cache_dir, select=lambda name: f"{_EXPORT_FOV}/" in name)
    if not paths:
        raise FileNotFoundError(f"no files for {_EXPORT_FOV} in the jump_cells registry entry")
    return paths[0].parent


def jump_cells(annotate: bool = True, selected: bool = False, cache_dir: str | Path | None = None) -> AnnData:
    """Single cells from one JUMP plate, as CellProfiler measured them.

    Twenty-four wells of ``BR00121438`` at four fields of view each: eight DMSO wells, four compounds with both of their replicate wells, and eight more compounds at one well.
    The strongest movers on this plate are cytotoxic, so ranking wells by distance alone selects for empty wells; every well here holds more than 120 cells in its first field.

    The same plate's well-level profiles are :func:`jump_target2`, so a profile aggregated from these cells can be compared with the one the consortium published.

    The first call downloads about 1.5 GB of CellProfiler output, reads 480 tables and writes the assembled object next to them, which takes a few minutes.
    Later calls read that one file.

    Args:
        annotate: Join the JUMP annotation, which supplies ``Metadata_Perturbation`` and ``Metadata_Control``.
            Downloads another 14 MB.
            Needed for `selected`, which is computed against the controls.
        selected: Return only the features ``var["selected"]`` marks, as :func:`mantispy.pp.subset_features` would.
            The subset is kept beside the whole object, so a notebook that only wants the reduced one reads 87 MB instead of 308 MB.
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.

    Raises:
        KeyError: `selected` was asked for without `annotate`, so there are no controls to select against.

    Returns:
        Cells by features at cell resolution, carrying ``Metadata_Source``, ``Metadata_Plate``, ``Metadata_Well``, ``Metadata_Site`` and, when annotated, ``Metadata_JCP2022``, ``Metadata_Perturbation``, ``Metadata_Perturbation_Type`` (``"compound"``), ``Metadata_InChIKey`` and ``Metadata_Control``.
        When annotated, ``var["selected"]`` marks the features feature selection keeps, so the object can be reduced with ``adata[:, adata.var["selected"]]`` the way scanpy's ``highly_variable`` is used.
        A cell carries no count.
        :func:`mantispy.tl.aggregate` writes ``Metadata_CellCount`` over the four fields read and a ``Metadata_SiteCount`` of four, so a well counts about four ninths of the cells :func:`jump_target2` gives it over all nine.

    References:
        :cite:t:`Chandrasekaran_2023`.
    """
    if selected and not annotate:
        raise KeyError("selected=True needs annotate=True: the mask is computed against the negative controls")

    entry = _DATASETS["jump_cells"]
    root = Path(cache_dir or settings.cache_dir)
    # The name fingerprints the schema, the pinned files and the assembly version, so a stale file is never read.
    material = f"{_ASSEMBLY_VERSION}:" + "".join(str(file.sha256) for file in entry.files)
    fingerprint = hashlib.sha256(material.encode()).hexdigest()[:12]
    stem = f"jump_cells-{SCHEMA_VERSION}-{fingerprint}-{'annotated' if annotate else 'raw'}"
    derived, subset = root / f"{stem}.h5ad", root / f"{stem}-selected.h5ad"
    if selected and subset.exists():
        return read(subset)
    if derived.exists():
        return _select(read(derived), subset) if selected else read(derived)

    adata = _assemble_cells(entry, cache_dir, annotate=annotate)
    write(adata, derived)
    return _select(adata, subset) if selected else adata


def jump_plate(cache_dir: str | Path | None = None, **kwargs: Any) -> SpatialData:
    """The images and segmentations behind one well of ``BR00121438``.

    Two fields of view of well ``O09``, a compound that changed the cells without killing them: the eight channel images of each field, the CellProfiler outlines they were segmented with, and the plate's ``load_data.csv``.
    About 44 MB, and needs the spatial extra.

    The cells measured from these fields are in :func:`jump_cells`, and the well-level profiles of the same plate in :func:`jump_target2`.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_plate`.

    Returns:
        The well, with its fields as Images, the Nuclei, Cells and Cytoplasm segmentations as Labels, and the ``cells`` Table.

    References:
        :cite:t:`Chandrasekaran_2023`.
    """
    from mantispy.io._plate import read_plate

    entry = _DATASETS["jump_plate"]
    paths = _files("jump_plate", cache_dir)
    # Every file is named by its gallery key, so stripping that key off a downloaded path gives the source tree's root.
    downloaded = Path(str(paths[0]).removesuffix(entry.files[0].name))
    root = downloaded / str(entry.metadata["accession"]) / str(entry.metadata["source"])
    return read_plate(
        root,
        str(entry.metadata["plate"]),
        batch=str(entry.metadata["batch"]),
        wells=[str(entry.metadata["well"])],
        profile=None,
        **kwargs,
    )
