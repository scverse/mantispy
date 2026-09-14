from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

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


def _fetch(name: str, cache_dir: str | Path | None, **kwargs: Any) -> AnnData:
    return fetch(_DATASETS[name], cache_dir or settings.cache_dir, base_url=_BASE_URL, **kwargs)


def cpjump1(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """One plate of the JUMP pilot, 384 wells of compounds profiled in U2OS cells.

    Plate ``BR00116991`` of batch ``2020_11_04_CPJUMP1`` of ``cpg0000-jump-pilot``, as the gallery publishes it: normalized against the negative controls of the batch and feature-selected.
    The perturbation of each well is in ``obs``, ``Metadata_pert_iname`` naming the compound.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells × features, indexed by plate and well.

    References:
        Chandrasekaran et al. (2024), *Three million images and morphological profiles of cells treated with
        matched chemical and genetic perturbations*, Nature Methods.

    Examples:
        >>> import mantispy as mt
        >>> adata = mt.ds.cpjump1()  # doctest: +SKIP
    """
    return _fetch("cpjump1", cache_dir, **kwargs)


def lincs(cache_dir: str | Path | None = None, **kwargs: Any) -> AnnData:
    """Two plates of the LINCS Cell Painting dataset, compounds at six doses in A549 cells.

    Plates ``SQ00015116`` and ``SQ00015117`` of batch ``2016_04_01_a549_48hr_batch1`` of ``cpg0004-lincs``.
    Feature selection ran per plate, so the two disagree on their columns and are read down to the features they share, as reading more than one plate of a real screen usually requires.
    ``Metadata_mmoles_per_liter`` holds the dose and ``Metadata_pert_iname`` the compound.

    Args:
        cache_dir: Where to keep the download.
            Defaults to :attr:`mantispy.settings.cache_dir`.
        kwargs: Passed to :func:`mantispy.io.read_profiles`.

    Returns:
        Wells × features, indexed by plate and well.

    References:
        Way et al. (2022), *Morphology and gene expression profiling provide complementary information for
        mapping cell state*, Cell Systems.

    Examples:
        >>> import mantispy as mt
        >>> adata = mt.ds.lincs()  # doctest: +SKIP
        >>> adata.obs.groupby("pert_iname", observed=True).size().sort_values()  # doctest: +SKIP
    """
    return _fetch("lincs", cache_dir, **kwargs)
