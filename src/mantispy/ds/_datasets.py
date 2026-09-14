from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scverse_misc.datasets import fetch, parse_registry, register_loader

from mantispy._settings import settings
from mantispy.io._profiles import read_profiles

if TYPE_CHECKING:
    from anndata import AnnData
    from scverse_misc.datasets import DatasetEntry, DownloadCB

REGISTRY = Path(__file__).parent / "registry.yaml"

_BASE_URL, _DATASETS = parse_registry(REGISTRY)


@register_loader("profiles")
def _load_profiles(entry: DatasetEntry, target: Path, download: DownloadCB, /, **kwargs: Any) -> AnnData:
    """Download every profile file of a dataset and stack them with :func:`mantispy.io.read_profiles`."""
    paths = [download(file) for file in entry.files]
    return read_profiles(paths, **{**entry.metadata.get("read", {}), **kwargs})


@register_loader("bbbc021")
def _load_bbbc021(entry: DatasetEntry, target: Path, download: DownloadCB, /, **kwargs: Any) -> AnnData:
    """Read the Ljosa profiles and annotate every well with its compound, dose and mechanism of action."""
    profiles, images, moa = (download(entry.file(name=file.name)) for file in entry.files)
    adata = read_profiles(profiles, index_columns=("Plate", "Well"), **kwargs)

    wells = (
        pd.read_csv(images, usecols=lambda c: c.startswith("Image_Metadata_"))
        .rename(columns=lambda c: c.removeprefix("Image_Metadata_").removesuffix("_DAPI"))
        .drop_duplicates(subset=["Plate", "Well"])
    )
    annotation = wells.merge(
        pd.read_csv(moa).rename(columns=str.title).rename(columns={"Moa": "MOA"}),
        on=["Compound", "Concentration"],
        how="left",
    ).set_index(wells["Plate"].astype(str) + ":" + wells["Well"].astype(str))

    for column in ("Compound", "Concentration", "MOA"):
        adata.obs[column] = annotation[column].reindex(adata.obs_names)
    adata.obs["Control"] = np.asarray(adata.obs["Compound"] == "DMSO")
    return adata


def _fetch(name: str, cache_dir: str | Path | None, **kwargs: Any) -> AnnData:
    return fetch(_DATASETS[name], cache_dir or settings.cache_dir, base_url=_BASE_URL, **kwargs)


def bbbc021(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """BBBC021, 632 wells of compounds annotated with a mechanism of action.

    The reference benchmark for image-based profiling.
    The profiles are the Ljosa reanalysis published as ``cpg0010-caie-drugresponse``, annotated here with the
    compound, dose and mechanism of action that the Broad Bioimage Benchmark Collection publishes separately.
    ``obs`` gains ``Compound``, ``Concentration``, ``MOA`` and ``Control``, the last marking the 330 DMSO wells.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, indexed by plate and well.

    References:
        Caie et al. (2010), *High-content phenotypic profiling of drug response signatures across distinct cancer cells*, Molecular Cancer Therapeutics.
        Ljosa et al. (2012), *Annotated high-throughput microscopy image sets for validation*, Nature Methods.
    """
    return _fetch("bbbc021", cache_dir, **kwargs)


def pooled_rare(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """Pooled rare variants, 290 barcodes in a pooled screen.

    ``cpg0032-pooled-rare``, aggregated to one row per barcode rather than per well, so there is no plate or well in ``obs``.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, with the perturbation and plate metadata in ``obs``.
    """
    return _fetch("pooled_rare", cache_dir, **kwargs)


def rohban(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """TA-ORF, 1,918 wells of 323 genes overexpressed in U2OS cells.

    ``cpg0017-rohban-pathways``, five plates read down to the features they share.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, with the perturbation and plate metadata in ``obs``.

    References:
        Rohban et al. (2017), *Systematic morphological profiling of human gene and allele function via Cell Painting*, eLife.
    """
    return _fetch("rohban", cache_dir, **kwargs)


def luad(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """LUAD alleles, 6,144 wells of wild-type against mutant.

    ``cpg0031-caicedo-cmvip``, sixteen plates read down to the features they share.
        ``Metadata_x_mutation_status`` separates the mutants from the wild type.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, with the perturbation and plate metadata in ``obs``.
    """
    return _fetch("luad", cache_dir, **kwargs)


def agnp(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """Silver nanoparticles, 180 wells at four sizes and three doses.

    ``cpg0040-garcia-fossa-AgNP``, the smallest dataset here at 300 kB.
        ``Metadata_NPSize_nm`` holds the particle size and ``Metadata_Concentration_mgml`` the dose.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, with the perturbation and plate metadata in ``obs``.

    References:
        Garcia-Fossa et al. (2023), *Interpreting image-based profiles using similarity clustering and single-cell visualization*, Current Protocols.
    """
    return _fetch("agnp", cache_dir, **kwargs)


def neuropainting(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """Astrocytes and neurons, 1,691 wells imaged at 20x and 63x.

    ``cpg0038-tegtmeyer-neuropainting``.
        One plate barcode appears in more than one batch, so the observations are not indexed by plate and well.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, with the perturbation and plate metadata in ``obs``.
    """
    return _fetch("neuropainting", cache_dir, **kwargs)


def amish(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """An Amish cohort, 468 wells varying seeding density and timepoint.

    ``cpg0047-amish``.
        ``Metadata_density_cells_per_well`` and ``Metadata_timepoint_hours`` hold the two factors.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, with the perturbation and plate metadata in ``obs``.
    """
    return _fetch("amish", cache_dir, **kwargs)


def chroma(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """Alternative dyes, 3,455 wells across eight channels.

    ``cpg0029-chroma-pilot``, which images more channels than the five of the standard protocol.
        One plate barcode appears at several timepoints, so the observations are not indexed by plate and well.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, with the perturbation and plate metadata in ``obs``.
    """
    return _fetch("chroma", cache_dir, **kwargs)


def oasis_pilot(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """OASIS pilot, 4,604 wells in U2OS and HepaRG.

    ``cpg0033-oasis-pilot``, twelve plates read down to the features they share.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, with the perturbation and plate metadata in ``obs``.
    """
    return _fetch("oasis_pilot", cache_dir, **kwargs)


def miami(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """MIAMI, 2,775 wells of compounds in U2OS cells.

    ``cpg0006-miami``, fourteen plates read down to the features they share.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, with the perturbation and plate metadata in ``obs``.
    """
    return _fetch("miami", cache_dir, **kwargs)


def pki(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """Kinase inhibitors, 3,072 wells with a dose and a mechanism of action.

    ``cpg0008-pki``, eight plates read down to the features they share.
        ``Metadata_mmoles_per_liter`` holds the dose and ``Metadata_broad_sample`` the compound.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, with the perturbation and plate metadata in ``obs``.
    """
    return _fetch("pki", cache_dir, **kwargs)


def jump_crispr(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """The assembled JUMP CRISPR arm, 51,185 wells of knockouts.

    ``cpg0016-jump-assembled``, the ``v1.0a`` well-position-corrected and feature-selected parquet, 51,185 wells by 599 features.
        The largest dataset here at 180 MB.
        ``Metadata_JCP2022`` identifies the perturbation.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, with the perturbation and plate metadata in ``obs``.

    References:
        Chandrasekaran et al. (2024), *Three million images and morphological profiles of cells treated with matched chemical and genetic perturbations*, Nature Methods.
    """
    return _fetch("jump_crispr", cache_dir, **kwargs)
