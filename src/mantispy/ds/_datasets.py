"""Cell Painting Gallery accessions, downloaded once and read with :func:`mantispy.io.read_profiles`.

Every file is pinned by its sha256 in ``registry.yaml``, so a file changed upstream raises an error instead of loading different data.
Downloads land in :attr:`mantispy.settings.cache_dir`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from scverse_misc.datasets import fetch, parse_registry, register_loader

from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._settings import settings
from mantispy.io._jump import read_jump
from mantispy.io._profiles import from_dataframe, read_profiles

if TYPE_CHECKING:
    from anndata import AnnData
    from scverse_misc.datasets import DatasetEntry, DownloadCB

_BASE_URL, _DATASETS = parse_registry(Path(__file__).parent / "registry.yaml")
# scverse-misc registers loaders by type name across all packages in the process, so ours uses the package name.
_TYPE = "mantispy"

#: One plate from each of two sources, enough to see a source effect with a small download.
TARGET2_DEFAULT = ("BR00121438", "JCPQC051")


@register_loader(_TYPE)
def _download(entry: DatasetEntry, target: Path, download: DownloadCB, /, **kwargs: object) -> list[Path]:
    """Download the files of `entry` that ``kwargs["select"]`` keeps, all of them without one, and return where they are."""
    select = cast("Callable[[str], bool] | None", kwargs.get("select"))
    return [Path(download(file)) for file in entry.files if select is None or select(file.name)]


def _files(name: str, cache_dir: str | Path | None = None, select: Callable[[str], bool] | None = None) -> list[Path]:
    return fetch(_DATASETS[name], cache_dir or settings.cache_dir, base_url=_BASE_URL, select=select)


def _plate(file_name: str) -> str:
    """The plate of a registry file, which is named ``<batch>__<plate>__<file>``."""
    return file_name.split("__")[1]


def _plate_files(name: str, plates: Sequence[str] | None, cache_dir: str | Path | None) -> list[Path]:
    known = [_plate(file.name) for file in _DATASETS[name].files]
    if plates is not None and (unknown := sorted(set(plates) - set(known))):
        raise KeyError(f"{name} has no plate(s) {unknown}; available: {sorted(known)}")
    wanted = set(known if plates is None else plates)
    return _files(name, cache_dir, select=lambda file_name: _plate(file_name) in wanted)


def _augmented(name: str, plates: Sequence[str] | None, cache_dir: str | Path | None) -> AnnData:
    """Read the ``*_augmented`` profiles of some plates, which already carry their platemap.

    Columns are intersected because plates of one screen can differ by a few features when a channel failed on one of them.
    """
    adata = read_profiles(_plate_files(name, plates, cache_dir), on_column_mismatch="intersect", resolution="well")
    obs = as_frame(adata.obs)
    obs["Metadata_CellCount"] = obs.pop("Metadata_Count_Cells").to_numpy(dtype=float)
    return adata


def _profiles(name: str, cache_dir: str | Path | None, **kwargs: Any) -> AnnData:
    """Stack every file of an accession, with the reading arguments the registry records for it."""
    adata = read_profiles(_files(name, cache_dir), **{**_DATASETS[name].metadata.get("read", {}), **kwargs})
    adata.uns["mantispy"]["dataset"] = _DATASETS[name].metadata["accession"]
    return adata


def bbbc021(cache_dir: str | Path | None = None) -> AnnData:
    """BBBC021, MCF-7 cells treated with small molecules, the standard mechanism-of-action benchmark.

    Well-level CellProfiler profiles from Ljosa et al. 2013 (``cpg0010-caie-drugresponse``), joined to the compound, concentration and mechanism of action the Broad Bioimage Benchmark Collection publishes with the image set.
    The image set covers 113 compounds.
    This returns the annotated subset the benchmark uses: 38 compounds plus DMSO, 103 treatments (a compound at a concentration) across 12 mechanisms.
    Downloads about 10 MB.

    Every well carries a mechanism, DMSO included; select treatments with ``adata[~adata.obs["Metadata_Control"]]``.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.

    Returns:
        632 wells by 473 features at well resolution, with ``Metadata_Plate``, ``Metadata_Well``, ``Metadata_Compound``, ``Metadata_Concentration``, ``Metadata_MOA``, ``Metadata_Perturbation`` (compound at concentration) and ``Metadata_Control``.

    References:
        Caie et al. (2010) Mol Cancer Ther 9:1913, the image set.
        Ljosa et al. (2013) J Biomol Screen 18:1321, these profiles and the benchmark.
        Images courtesy of Peter Caie and David Westwood, available from the Broad Bioimage Benchmark Collection (Ljosa et al. 2012, Nature Methods 9:637).
    """
    profiles_path, images_path, moa_path = _files("bbbc021", cache_dir)
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
    merged = profiles.merge(annotations, on=["Metadata_Plate", "Metadata_Well"], how="left")
    if unmatched := int(merged["Metadata_Compound"].isna().sum()):
        get_logger().warning("%d wells have no compound annotation and are dropped", unmatched)
        merged = merged[merged["Metadata_Compound"].notna()]

    adata = from_dataframe(merged, resolution="well")
    obs = as_frame(adata.obs)
    obs["Metadata_Control"] = (obs["Metadata_Compound"] == "DMSO").to_numpy()
    # Replicates share a compound at a concentration; the mode= shorthands of mt.tl.map need this column.
    obs["Metadata_Perturbation"] = pd.Categorical(
        obs["Metadata_Compound"].astype(str) + "@" + obs["Metadata_Concentration"].astype(str)
    )
    adata.uns["mantispy"]["dataset"] = "BBBC021"
    get_logger().info("BBBC021: %d wells x %d features", adata.n_obs, adata.n_vars)
    return adata


def rohban(plates: Sequence[str] | None = None, cache_dir: str | Path | None = None) -> AnnData:
    """The Rohban 2017 ORF overexpression screen, with the genes and cell counts that BBBC021 lacks.

    ``cpg0017-rohban-pathways``: U2OS cells, one gene overexpressed per well, roughly ten replicate wells per gene over five plates.
    Downloads about 27 MB for all five.

    Args:
        plates: Plate barcodes to load, all five when omitted.
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.

    Returns:
        Wells by features at well resolution, with ``Metadata_Perturbation`` (the gene), ``Metadata_Control``, ``Metadata_CellCount`` and the screen's own ``Metadata_gene_name``, ``Metadata_GeneID`` and ``Metadata_ASSAY_WELL_ROLE``.

    Raises:
        KeyError: A plate is not one of the five.

    Notes:
        ``Metadata_Control`` marks the wells transfected with a control ORF (Luciferase, LacZ and eGFP), the reference for normalization.
        The untreated wells (``Metadata_gene_name == "EMPTY"``) were never transfected and are not flagged; drop them if a gene-level analysis should not see them.

    References:
        Rohban et al. (2017) eLife 6:e24060.
    """
    adata = _augmented("rohban", plates, cache_dir)
    obs = as_frame(adata.obs)
    obs["Metadata_Control"] = (obs["Metadata_ASSAY_WELL_ROLE"].astype(str) == "CTRL").to_numpy()
    obs["Metadata_Perturbation"] = obs["Metadata_gene_name"].astype(str).astype("category")
    adata.uns["mantispy"]["dataset"] = "cpg0017-rohban-pathways"
    get_logger().info(
        "rohban: %d wells x %d features, %d genes",
        adata.n_obs,
        adata.n_vars,
        adata.obs["Metadata_Perturbation"].nunique(),
    )
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
        Wells by features at well resolution, with ``Metadata_Perturbation`` (compound at concentration), ``Metadata_Compound``, ``Metadata_Concentration`` (the platemap's ``mmoles_per_liter``), ``Metadata_MOA``, ``Metadata_Control`` and ``Metadata_CellCount``.

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
    # Control wells have no compound or dose; without this each would become its own "nan@nan" perturbation
    # instead of joining one DMSO group.
    compound = np.where(control, "DMSO", obs["Metadata_broad_sample"].astype(str).to_numpy())
    dose = obs["Metadata_mmoles_per_liter"].to_numpy(dtype=float)
    obs["Metadata_Compound"] = pd.Categorical(compound)
    obs["Metadata_Concentration"] = dose
    obs["Metadata_MOA"] = obs.pop("Metadata_moa")
    label = np.where(control, "DMSO", np.char.add(np.char.add(compound.astype(str), "@"), dose.astype(str)))
    obs["Metadata_Perturbation"] = pd.Categorical(label)
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
    """JUMP-Target-2, one 384-well plate map run at many sites.

    The JUMP consortium ran the same plate map in every participating laboratory, so differences between plates from different sources are technical.
    This makes it suited to studying batch and source effects.
    Twelve of the 141 plates in ``cpg0016-jump`` are pinned here (three sources, two batches each, two plates per batch), which lets :func:`~mantispy.tl.transport` separate a laboratory effect from a plate effect.

    Args:
        plates: Plate barcodes to load.
            The default takes one plate from each of two sources, about 26 MB; ``None`` loads all twelve.
        annotate: Join the JUMP annotation, which supplies ``Metadata_Perturbation`` and ``Metadata_Control``.
            Downloads another 14 MB.
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.

    Returns:
        384 wells per plate at well resolution, carrying ``Metadata_Source``, ``Metadata_Batch``, ``Metadata_Plate``, ``Metadata_Well`` and, when annotated, ``Metadata_JCP2022``, ``Metadata_Perturbation``, ``Metadata_InChIKey`` and ``Metadata_Control`` (JUMP's 64 DMSO wells per plate).

    Raises:
        KeyError: A plate is not one of the twelve.
    """
    adata = read_jump(
        _plate_files("jump_target2", plates, cache_dir), annotate=annotate, on_column_mismatch="intersect"
    )
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


def pooled_rare(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """Pooled rare variants, 290 barcodes in a pooled screen.

    ``cpg0032-pooled-rare``, aggregated to one row per barcode rather than per well, so there is no plate or well in ``obs``.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Barcodes by features.
    """
    return _profiles("pooled_rare", cache_dir, **kwargs)


def luad(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """LUAD alleles, 6,144 wells of wild-type against mutant.

    ``cpg0031-caicedo-cmvip``, sixteen plates read down to the features they share.
    ``Metadata_x_mutation_status`` separates the mutants from the wild type.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, indexed by plate and well.
    """
    return _profiles("luad", cache_dir, **kwargs)


def agnp(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """Silver nanoparticles, 180 wells at four sizes and three doses.

    ``cpg0040-garcia-fossa-AgNP``, the smallest dataset here at 300 kB.
    ``Metadata_NPSize_nm`` holds the particle size and ``Metadata_Concentration_mgml`` the dose.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, indexed by plate and well.

    References:
        Garcia-Fossa et al. (2023), *Interpreting image-based profiles using similarity clustering and single-cell visualization*, Current Protocols.
    """
    return _profiles("agnp", cache_dir, **kwargs)


def neuropainting(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """Astrocytes and neurons, 1,691 wells imaged at 20x and 63x.

    ``cpg0038-tegtmeyer-neuropainting``.
    One plate barcode appears in more than one batch, so the observations are not indexed by plate and well.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features.
    """
    return _profiles("neuropainting", cache_dir, **kwargs)


def amish(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """An Amish cohort, 468 wells varying seeding density and timepoint.

    ``cpg0047-amish``.
    ``Metadata_density_cells_per_well`` and ``Metadata_timepoint_hours`` hold the two factors.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, indexed by plate and well.
    """
    return _profiles("amish", cache_dir, **kwargs)


def chroma(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """Alternative dyes, 3,455 wells across eight channels.

    ``cpg0029-chroma-pilot``, which images more channels than the five of the standard protocol.
    One plate barcode appears at several timepoints, so the observations are not indexed by plate and well.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features.
    """
    return _profiles("chroma", cache_dir, **kwargs)


def oasis_pilot(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """OASIS pilot, 4,604 wells in U2OS and HepaRG.

    ``cpg0033-oasis-pilot``, twelve plates read down to the features they share.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, indexed by plate and well.
    """
    return _profiles("oasis_pilot", cache_dir, **kwargs)


def miami(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """MIAMI, 2,775 wells of compounds in U2OS cells.

    ``cpg0006-miami``, fourteen plates read down to the features they share.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, indexed by plate and well.
    """
    return _profiles("miami", cache_dir, **kwargs)


def jump_crispr(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """The assembled JUMP CRISPR arm, 51,185 wells of knockouts.

    ``cpg0016-jump-assembled``, the ``v1.0a`` well-position-corrected and feature-selected parquet, and the largest dataset here at 180 MB.
    ``Metadata_JCP2022`` identifies the perturbation.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, indexed by plate and well.

    References:
        Chandrasekaran et al. (2024), *Three million images and morphological profiles of cells treated with matched chemical and genetic perturbations*, Nature Methods.
    """
    return _profiles("jump_crispr", cache_dir, **kwargs)
