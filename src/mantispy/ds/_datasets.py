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
from mantispy._core.frames import as_frame
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

#: One plate from each of the eleven sources that ran Target-2. Pinned rather than derived, so the default set
#: cannot move when a plate is added to the registry or the rows are reordered.
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
_ASSEMBLY_VERSION = 2

#: The field of view :func:`jump_export` hands out, a DMSO well of the plate :func:`jump_cells` reads.
_EXPORT_FOV = "BR00121438-J04-1"


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
    # A set, because a plate contributes several files and listing it once per file printed all 141 twice.
    known = {_plate(file.name) for file in _DATASETS[name].files}
    if plates is not None and (unknown := sorted(set(plates) - known)):
        raise KeyError(f"{name} has no plate(s) {unknown}; available: {sorted(known)}")
    wanted = known if plates is None else set(plates)
    return _files(name, cache_dir, select=lambda file_name: _plate(file_name) in wanted)


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


def bbbc021(cache_dir: str | Path | None = None) -> AnnData:
    """BBBC021, MCF-7 cells treated with small molecules, the standard mechanism-of-action benchmark.

    Well-level CellProfiler profiles (``cpg0010-caie-drugresponse``), joined to the compound, concentration and mechanism of action the Broad Bioimage Benchmark Collection publishes with the image set.
    The image set covers 113 compounds.
    This returns the annotated subset the benchmark uses: 38 compounds plus DMSO, 103 treatments (a compound at a concentration) across 12 mechanisms.
    Downloads about 22 MB, half of it the ``Image.csv`` of each well, which is where the cell counts are.

    Every well carries a mechanism, DMSO included; select treatments with ``adata[~adata.obs["Metadata_Control"]]``.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.

    Returns:
        632 wells by 473 features at well resolution, with ``Metadata_Plate``, ``Metadata_Well``, ``Metadata_Compound``, ``Metadata_Concentration``, ``Metadata_MOA``, ``Metadata_Perturbation`` (compound at concentration), ``Metadata_Control``, and ``Metadata_CellCount`` over the ``Metadata_SiteCount`` fields, of four imaged, that contributed cells.

    References:
        :cite:t:`Caie_2010`, the image set.
        :cite:t:`Ljosa_2013`, these profiles and the benchmark.
        Images courtesy of Peter Caie and David Westwood, available from the Broad Bioimage Benchmark Collection :cite:p:`Ljosa_2012`.
    """
    profiles_path, images_path, moa_path, *fields = _files("bbbc021", cache_dir)
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
    # Replicates share a compound at a concentration; the mode= shorthands of mt.tl.map need this column.
    obs["Metadata_Perturbation"] = pd.Categorical(
        obs["Metadata_Compound"].astype(str) + "@" + obs["Metadata_Concentration"].astype(str)
    )
    adata.uns["mantispy"]["dataset"] = "BBBC021"
    get_logger().info("BBBC021: %d wells x %d features", adata.n_obs, adata.n_vars)
    return adata


def _bbbc021_counts(paths: Sequence[Path]) -> pd.DataFrame:
    """Cells, and the fields that contributed them, per well, from the per-field ``Image.csv`` of the run the ljosa_2013 profiles aggregate."""
    rows = []
    for path in paths:
        cells = pd.read_csv(path, usecols=["Count_Cells"])["Count_Cells"]
        rows.append((*path.parent.name.rsplit("-", 1), cells.sum(), (cells > 0).sum()))
    return pd.DataFrame(rows, columns=["Metadata_Plate", "Metadata_Well", "Metadata_CellCount", "Metadata_SiteCount"])


def rohban(plates: Sequence[str] | None = None, cache_dir: str | Path | None = None) -> AnnData:
    """An ORF overexpression screen, with the genes and cell counts that BBBC021 lacks.

    ``cpg0017-rohban-pathways``: U2OS cells, one gene overexpressed per well, roughly ten replicate wells per gene over five plates.
    Downloads about 27 MB for all five.

    Args:
        plates: Plate barcodes to load, all five when omitted.
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.

    Returns:
        Wells by features at well resolution, with ``Metadata_Perturbation`` (the gene), ``Metadata_Control``, ``Metadata_CellCount``, ``Metadata_SiteCount`` and the screen's own ``Metadata_gene_name``, ``Metadata_GeneID`` and ``Metadata_ASSAY_WELL_ROLE``.

    Raises:
        KeyError: A plate is not one of the five.

    Notes:
        ``Metadata_Control`` marks the wells transfected with a control ORF (Luciferase, LacZ and eGFP), the reference for normalization.
        The untreated wells (``Metadata_gene_name == "EMPTY"``) were never transfected and are not flagged; drop them if a gene-level analysis should not see them.

    References:
        :cite:t:`Rohban_2017`.
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
        Wells by features at well resolution, with ``Metadata_Perturbation`` (compound at concentration), ``Metadata_Compound``, ``Metadata_Concentration`` (the platemap's ``mmoles_per_liter``), ``Metadata_MOA``, ``Metadata_Control``, ``Metadata_CellCount`` and ``Metadata_SiteCount``.

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
        One row per well at well resolution, carrying ``Metadata_Source``, ``Metadata_Batch``, ``Metadata_Plate``, ``Metadata_Well``, ``Metadata_CellCount``, ``Metadata_SiteCount`` and, when annotated, ``Metadata_JCP2022``, ``Metadata_Perturbation``, ``Metadata_InChIKey`` and ``Metadata_Control`` (the DMSO wells, 64 per 384-well plate and 256 on source_9's).

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
        Wells by features, indexed by plate and well, with ``Metadata_CellCount`` and ``Metadata_SiteCount``.
    """
    return _profiles("luad", cache_dir, **kwargs)


def agnp(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """Silver nanoparticles in Huh7 cells, 180 wells at two sizes and three timepoints.

    ``cpg0040-garcia-fossa-AgNP``, the smallest dataset here at 300 kB.
    Each size is given at one dose: 40 nm at 0.00012 mg/ml and 100 nm at 0.00036 mg/ml, in ``Metadata_NPSize_nm`` and ``Metadata_Concentration_mgml``.
    The 60 untreated wells, size and dose 0, are the ``negcon`` of ``Metadata_control_type``.
    ``Metadata_Time`` is 1, 15 or 30, with 20 wells of each condition at each.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, indexed by plate and well, with ``Metadata_CellCount`` and ``Metadata_SiteCount``.
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
        Wells by features, with ``Metadata_CellCount`` and ``Metadata_SiteCount``.
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
        Wells by features, indexed by plate and well, with ``Metadata_CellCount`` and ``Metadata_SiteCount``.
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
        Wells by features, with ``Metadata_CellCount`` and ``Metadata_SiteCount``.
    """
    return _profiles("chroma", cache_dir, **kwargs)


#: What each OASIS batch calls the columns mantispy reads. The four batches were laid out by different people and
#: none of them agree on a name, so each output column lists the spellings seen across them.
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

    The OASIS plate maps disagree on precision rather than on value: one batch writes the concentration the
    dilution actually produced, ``0.0152416`` uM, and another writes it rounded, ``0.015``. A level written to
    fewer decimals is therefore the same dose as the finer level that rounds to it, which is a statement about
    how the number was recorded rather than a tolerance fitted to the data.

    Matching runs within a compound, because a compound's levels are one dilution series and cannot collide.
    Across the whole plate map they can: berberine's 25 uM and the main ladder's 33.3 uM are a third apart and
    genuinely different doses, closer together than ``0.000762`` and ``0.001`` are, which are one dose.

    Rounding every level to a fixed precision cannot do this, and neither can a relative tolerance. The one case
    it would get wrong is a ladder with two rungs inside a rounding step of each other, which a series coarser
    than two-fold never has.
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
        # The batches that name compounds in "Compound Name" leave it blank for the wells that hold no compound and
        # put DMSO or EMPTY in the identifier column instead. Without this the controls read as unannotated wells.
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


def oasis_pilot(annotate: bool = True, cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """OASIS pilot, 4,604 wells in U2OS and HepaRG, most compounds over a ten-point dose range.

    ``cpg0033-oasis-pilot``, twelve plates read down to the features they share, over four batches: two of assay
    development and two that dose 36 compounds in each cell line. It is the dose-response dataset of the package:
    28 of those compounds carry six or more concentrations in both U2OS and HepaRG.

    Args:
        annotate: Join the plate maps, which supply ``Metadata_Compound``, ``Metadata_Concentration``,
            ``Metadata_CellLine``, ``Metadata_Control`` (the DMSO wells) and ``Metadata_Perturbation``.
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, indexed by plate and well, with ``Metadata_CellCount`` and ``Metadata_SiteCount``, and
        the annotation columns above when ``annotate``.

    Notes:
        The four batches name their plate-map columns differently, so the join reads whichever of
        ``treatment``/``Compound Name``/``compound`` and ``concentration_uM``/``assay_conc_uM``/``compound_concentration``
        each one carries. Concentrations are micromolar.

        The assay-development batch doses DMSO itself, so a control well there carries a concentration.
        ``Metadata_Control`` marks the compound, not the dose.

        The plate maps disagree on precision: one batch writes the concentration the dilution produced,
        ``0.0152416`` uM, and another writes it rounded, ``0.015``. ``Metadata_Concentration`` is the dose the
        well was meant to get, reading a coarser spelling as the finer level it rounds to within each compound,
        and it names the replicate groups. ``Metadata_ConcentrationRecorded`` keeps what the plate map wrote:
        read against that column, every dosed compound here carries eighteen levels where ten were plated, and a
        treatment's wells split across two spellings.
    """
    adata = _profiles("oasis_pilot", cache_dir, select=lambda name: name.endswith(".csv.gz"), **kwargs)
    if not annotate:
        return adata

    obs = as_frame(adata.obs)
    merged = obs.merge(_oasis_platemaps(cache_dir), on=["Metadata_plate_map_name", "Metadata_Well"], how="left")
    merged.index = obs.index
    if unmatched := int(merged["Metadata_Compound"].isna().sum()):
        get_logger().warning("oasis_pilot: %d of %d wells have no plate-map row", unmatched, len(merged))
    adata.obs["Metadata_Compound"] = merged["Metadata_Compound"].to_numpy()
    adata.obs["Metadata_Concentration"] = merged["Metadata_Concentration"].to_numpy(dtype=float)
    adata.obs["Metadata_ConcentrationRecorded"] = merged["Metadata_ConcentrationRecorded"].to_numpy(dtype=float)
    adata.obs["Metadata_CellLine"] = merged["Metadata_CellLine"].to_numpy()
    adata.obs["Metadata_Control"] = merged["Metadata_Compound"].astype(str).str.upper().eq("DMSO").to_numpy()
    # Replicates share a compound at a concentration, which is what the mode= shorthands of mt.tl.map compare.
    is_control = np.asarray(adata.obs["Metadata_Control"], dtype=bool)
    adata.obs["Metadata_Perturbation"] = pd.Categorical(
        np.where(
            is_control,
            "DMSO",
            merged["Metadata_Compound"].astype(str) + "@" + merged["Metadata_Concentration"].astype(str),
        )
    )
    get_logger().info(
        "OASIS pilot: %d wells x %d features, %d compounds over %d concentrations, %d control wells",
        adata.n_obs,
        adata.n_vars,
        int(merged.loc[~is_control, "Metadata_Compound"].nunique()),
        int(merged["Metadata_Concentration"].nunique()),
        int(is_control.sum()),
    )
    return adata


def miami(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """MIAMI, 2,775 wells of compounds in U2OS cells.

    ``cpg0006-miami``, fourteen plates read down to the features they share.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, indexed by plate and well, with ``Metadata_CellCount`` and ``Metadata_SiteCount``.
    """
    return _profiles("miami", cache_dir, **kwargs)


#: The feature sets JUMP-Lite publishes for one set of wells: five learned embeddings, and the
#: CellProfiler-equivalent measurements of ``cp_measure`` for comparison on the same rows.
JUMP_LITE_MODELS = ("openphenom", "dinov2", "dinov2_random", "subcell", "morphem", "cp_measure")


def jump_lite(
    model: str = "openphenom", annotate: bool = True, cache_dir: str | Path | None = None, **kwargs: Any
) -> AnnData:
    """JUMP-Lite Target-2: 1,536 wells, four imaging sites, one feature set at a time.

    ``cpg0016-jump``, the compact benchmark of :cite:t:`Munoz_2026`. Four plates of the JUMP Target-2 plate map, one from each of ``source_3``, ``source_4``, ``source_5`` and ``source_6``, so the four batches are four different laboratories running the same 302 compounds with 64 DMSO wells each.

    Every ``model`` covers the same 1,536 wells, which is what makes this a comparison rather than six datasets: the rows and the metadata are identical and only the feature block changes. Five are learned embeddings and one, ``"cp_measure"``, is the CellProfiler-equivalent measurement of the same images.

    Args:
        model: Which feature set to read, one of ``ds.JUMP_LITE_MODELS``. ``"dinov2_random"`` is the same architecture with untrained weights, which is the null model the benchmark scores the others against.
        annotate: Join the JUMP well and compound tables, which name the compound of each well.
            Downloads about 14 MB once and caches it.
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features at well resolution, indexed by plate and well, with ``Metadata_Source``, ``Metadata_Batch``, ``Metadata_Plate``, ``Metadata_Well``, ``Metadata_CellCount`` and, when annotated, ``Metadata_JCP2022``, ``Metadata_Perturbation``, ``Metadata_InChIKey`` and ``Metadata_Control``.

    Raises:
        ValueError: ``model`` is not one of ``ds.JUMP_LITE_MODELS``.

    Notes:
        A dimension of a learned embedding is a coordinate in the model's own basis, not a measurement with a name to parse, so for every model but ``"cp_measure"`` the annotation columns of ``var`` are supplied empty. Anything that reads ``var["feature_group"]`` or ``var["channel"]``, such as the feature families :func:`~mantispy.pl.effect_sizes` colours by, has nothing to work with on those. ``"cp_measure"`` is CellProfiler-style measurements and keeps its parsed annotation.

        The embeddings are not normalized. They are the model's output on each well's images, so a per-plate control normalization is still the first step.

        The trained embeddings here carry the cell count in their leading components, where it can account for more of the variance than either the laboratory or the imaging site. The untrained ``"dinov2_random"`` does not, and neither does ``"cp_measure"``, whose per-cell measurements are averaged over the well. Measure it with :func:`~mantispy.metrics.evaluate_correction` before correcting for anything else, and read :doc:`/tutorials/12_learned_embeddings` on why removing it is not obviously right.

    References:
        :cite:t:`Munoz_2026`, :cite:t:`Chandrasekaran_2023`, :cite:t:`Weisbart_2024`.
    """
    if model not in JUMP_LITE_MODELS:
        raise ValueError(f"model must be one of {JUMP_LITE_MODELS}, got {model!r}")

    adata = _profiles("jump_lite", cache_dir, select=lambda name: name == f"{model}.parquet", **kwargs)
    if model != "cp_measure":
        # An embedding dimension is a coordinate in a learned basis, not a measurement with a name
        # to parse. Left alone, the parser reads "openphenom_nahualX_17" as the "nahualX" feature
        # group of an "openphenom" object, and the model's own tensor names become feature families.
        empty = empty_annotation(adata.var_names)
        adata.var[empty.columns] = empty

    (counts_path,) = _files("jump_lite", cache_dir, select=lambda name: name == "cell_count.parquet")
    counts = pd.read_parquet(counts_path, columns=["Metadata_id", "cell_count"])

    obs = as_frame(adata.obs)
    joined = obs.merge(counts, on="Metadata_id", how="left", validate="1:1")
    joined.index = obs.index
    adata.obs = joined.rename(columns={"cell_count": "Metadata_CellCount"})

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
        A frame with ``source``, the gene symbol, and ``target``, the ``Metadata_JCP2022`` of a compound annotated to it. One row per annotated pairing, over every JUMP compound rather than only those of :func:`jump_lite`.

    Notes:
        The annotation is sparse against a plate map: most compounds on the JUMP-Lite plates carry none, and only the targets shared by more than one compound contribute a pair, so the recall is computed over a minority of the plate.

    References:
        :cite:t:`Munoz_2026`.
    """
    (path,) = _files("jump_lite", cache_dir, select=lambda name: name == "refchem_annotations.parquet")
    frame = pd.read_parquet(path, columns=["target", "Metadata_JCP2022"])
    # Dropped before the cast, or an unannotated compound becomes the literal string "nan" and
    # every one of them is then related to every other.
    frame = frame.dropna().rename(columns={"target": "source", "Metadata_JCP2022": "target"}).astype(str)
    return frame.drop_duplicates().reset_index(drop=True)


def jump_crispr(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """The assembled JUMP CRISPR arm, 51,185 wells of knockouts.

    ``cpg0016-jump-assembled``, the ``v1.0a`` well-position-corrected and feature-selected parquet, and the largest dataset here at 180 MB.
    ``Metadata_JCP2022`` identifies the perturbation.
    The cell counts are the ones jump-profiling-recipe regresses out of these profiles; their ``Cells_Count_Count`` feature was normalized with the rest and is no longer a count.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells by features, indexed by plate and well, with ``Metadata_CellCount``.

    References:
        :cite:t:`Chandrasekaran_2023`.
    """
    (counts,) = _files("_jump_cell_counts", cache_dir)
    return _profiles("jump_crispr", cache_dir, platemap=_read_counts(counts), **kwargs)


def _read_site(directory: Path, source: str, channels: Sequence[str]) -> AnnData:
    """One analysis directory as cells, checked against the ``<plate>-<well>-<site>`` name it is filed under.

    The JUMP pipeline gives ``Cytoplasm`` a ``Parent_Cells`` and a ``Parent_Nuclei`` and gives ``Cells`` no parent at all, so ``Cytoplasm`` is the only primary object that joins all three tables. One cytoplasm is one cell here, and the three tables have equal length.
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
    adata = ad.concat(parts, join="inner", merge="first", uns_merge="first")
    # The parts hold as much again as the result, and nothing below needs them.
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

    The mask is computed on a normalized copy, because variance and correlation are only comparable between features once each is on its own plate's control scale, and the values on `adata` stay as CellProfiler measured them. The recipe is the one the tutorials use: ``pp.normalize`` against the negative controls, drop what ``var["degenerate_scale"]`` flags, then ``pp.feature_select``. No step of it draws a random number, so the same pinned files always give the same mask.
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

    A single field of view of a DMSO well of ``BR00121438``: ``Image.csv`` plus the ``Cells``, ``Cytoplasm`` and ``Nuclei`` tables, unmodified, for reading with :func:`mantispy.io.read_profiles`. About 18 MB, and part of the download :func:`jump_cells` makes, so asking for both costs nothing extra.

    The images this was measured from are in :func:`jump_plate`, and the well-level profiles of the same plate are in :func:`jump_target2`.

    Args:
        cache_dir: Where to keep the download. Defaults to :attr:`mantispy.settings.cache_dir`.

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

    Twenty-four wells of ``BR00121438`` at four fields of view each: eight DMSO wells, four compounds with both of their replicate wells, and eight more compounds at one well. The strongest movers on this plate are cytotoxic, so ranking wells by distance alone selects for empty wells; every well here holds more than 120 cells in its first field.

    The same plate's well-level profiles are :func:`jump_target2`, so a profile aggregated from these cells can be compared with the one the consortium published.

    The first call downloads about 1.5 GB of CellProfiler output, reads 480 tables and writes the assembled object next to them, which takes a few minutes. Later calls read that one file.

    Args:
        annotate: Join the JUMP annotation, which supplies ``Metadata_Perturbation`` and ``Metadata_Control``.
            Downloads another 14 MB. Needed for `selected`, which is computed against the controls.
        selected: Return only the features ``var["selected"]`` marks, 1607 of 5857, as
            :func:`mantispy.pp.subset_features` would. The subset is kept beside the whole object, so a notebook that only wants the reduced one reads 87 MB instead of 308 MB.
        cache_dir: Where to keep the download. Defaults to :attr:`mantispy.settings.cache_dir`.

    Raises:
        KeyError: `selected` was asked for without `annotate`, so there are no controls to select against.

    Returns:
        Cells by features at cell resolution, carrying ``Metadata_Source``, ``Metadata_Plate``, ``Metadata_Well``, ``Metadata_Site`` and, when annotated, ``Metadata_JCP2022``, ``Metadata_Perturbation``, ``Metadata_InChIKey`` and ``Metadata_Control``. When annotated, ``var["selected"]`` marks the features feature selection keeps, so the object can be reduced with ``adata[:, adata.var["selected"]]`` the way scanpy's ``highly_variable`` is used.
        A cell carries no count. :func:`mantispy.tl.aggregate` writes ``Metadata_CellCount`` over the four fields read and a ``Metadata_SiteCount`` of four, so a well counts about four ninths of the cells :func:`jump_target2` gives it over all nine.

    References:
        :cite:t:`Chandrasekaran_2023`.
    """
    if selected and not annotate:
        raise KeyError("selected=True needs annotate=True: the mask is computed against the negative controls")

    entry = _DATASETS["jump_cells"]
    root = Path(cache_dir or settings.cache_dir)
    # The name carries the schema version and a fingerprint of the files the entry pins and of how they are
    # assembled, so no change to the schema, to which wells are read, or to what assembly produces can be
    # answered from a stale file.
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

    Two fields of view of well ``O09``, a compound that changed the cells without killing them: the eight channel images of each field, the CellProfiler outlines they were segmented with, and the plate's ``load_data.csv``. About 44 MB, and needs the spatial extra.

    The cells measured from these fields are in :func:`jump_cells`, and the well-level profiles of the same plate in :func:`jump_target2`.

    Args:
        cache_dir: Where to keep the download. Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_plate`.

    Returns:
        The well, with its fields as Images, the Nuclei, Cells and Cytoplasm segmentations as Labels, and the ``cells`` Table.

    References:
        :cite:t:`Chandrasekaran_2023`.
    """
    from mantispy.io._plate import read_plate

    entry = _DATASETS["jump_plate"]
    paths = _files("jump_plate", cache_dir)
    # Every file is named by its gallery key, so the download reconstructs the source tree and read_plate
    # reads it as it would read the bucket. Stripping that key off a downloaded path gives the tree's root,
    # rather than assuming where fetch put it.
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
