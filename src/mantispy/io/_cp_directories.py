"""Read several CellProfiler exports, such as the one directory per site a pooled screen writes, into one profile."""

from __future__ import annotations

import importlib.util
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.provenance import record_params
from mantispy.io._cellprofiler import _kind, _profile, _read_tables


def _conform(adata: AnnData, var: pd.DataFrame, dtypes: Mapping[str, Any]) -> AnnData:
    """`adata` with the features of `var` and the ``obs`` columns of `dtypes`, in their order.

    A roll-up count it lacks is 0, as is an integer ``obs`` column such as the number of distinct barcodes; any other is missing.
    """
    columns = adata.var_names.get_indexer(var.index)
    counts = (var["aggregation"] == "count").to_numpy() if "aggregation" in var else np.zeros(len(var), dtype=bool)
    X = np.where(counts, 0.0, np.nan)[None, :].repeat(adata.n_obs, axis=0)
    X[:, columns >= 0] = np.asarray(adata.X)[:, columns[columns >= 0]]
    obs = as_frame(adata.obs)
    added = {
        column: pd.Series(0 if pd.api.types.is_integer_dtype(dtype) else None, index=obs.index, dtype=dtype)
        for column, dtype in dtypes.items()
        if column not in obs
    }
    obs = pd.concat([obs, pd.DataFrame(added, index=obs.index)], axis=1)[list(dtypes)]
    return AnnData(X=X, obs=obs, var=var, uns=adata.uns)


def read_directories(
    paths: Sequence[Path],
    *,
    read: Mapping[str, Any],
    profile: Mapping[str, Any],
    params: Mapping[str, Any],
    skip_non_exports: bool,
    out: str | Path | None,
) -> AnnData:
    """The profile of several exports, as :func:`~mantispy.io.read_cellprofiler` documents it.

    Each export is read and rolled up on its own, and either kept or, with `out`, written to a part file and dropped.
    The parts are then given the same features and ``obs`` columns and stacked once.
    """
    from mantispy.tl._aggregate_objects import categorical_annotation

    log = get_logger()
    root = profile["root"]
    target = Path(out) if out is not None else None
    if target is not None and importlib.util.find_spec("dask") is None:
        raise ImportError("read_cellprofiler(out=...) needs dask; install it with pip install 'mantispy[lazy]'")
    kinds: dict[Path, str] = {}
    others = []
    for path in paths:
        try:
            kinds[path] = _kind(path)
        except FileNotFoundError:
            others.append(str(path))
    if others and not skip_non_exports:
        raise FileNotFoundError(
            f"{len(others)} of {len(paths)} paths are not CellProfiler exports: {others}; "
            "pass skip_non_exports=True to skip them"
        )
    if others:
        log.info("skipped %d paths that are not CellProfiler exports: %s", len(others), others)
    if not kinds:
        raise FileNotFoundError(f"none of the {len(paths)} paths is a CellProfiler export")

    several = len(kinds) > 1
    seen: dict[str, Path] = {}
    images: list[pd.DataFrame] = []
    parts: list[AnnData] = []
    image_only: list[AnnData] = []
    without_root: list[str] = []
    files: list[Path] = []
    with (
        tempfile.TemporaryDirectory(dir=target.parent, prefix=f".{target.name}.")
        if target is not None
        else nullcontext(None)
    ) as scratch:
        for path, kind in kinds.items():
            tables = _read_tables(path, kind, run=path.name if several else None, **read)
            image_table = next(iter(tables.values())).uns["mantispy"]["image_table"]
            for image_id in image_table.index:
                if (other := seen.setdefault(image_id, path)) != path:
                    raise ValueError(
                        f"{other} and {path} both hold the image {image_id!r}; read them separately, or give them "
                        "plate, well and site metadata that tell them apart"
                    )
            images.append(image_table)
            if list(tables) == ["Image"]:
                image_only.append(_profile(tables, **profile))
            elif root not in tables:
                without_root.append(path.name)
            else:
                part = _profile(tables, **profile)
                if several and not profile["index_columns"]:
                    part.obs_names = [f"{path.name}:{name}" for name in part.obs_names]
                if scratch is not None:
                    files.append(Path(scratch) / f"{len(files)}.h5ad")
                    part.write_h5ad(files[-1])
                    # Only the annotation is kept; the matrix is on disk.
                    part = AnnData(obs=as_frame(part.obs), var=as_frame(part.var), uns=part.uns)
                parts.append(part)
        if without_root:
            log.info("found no %s in %s; their images are in uns['mantispy']['image_table']", root, without_root)
        if parts and image_only:
            log.info(
                "%d exports measure images only; their images are in uns['mantispy']['image_table']", len(image_only)
            )
        if not parts:
            if not image_only:
                raise FileNotFoundError(f"no table for {root!r} in any of the {len(kinds)} exports")
            parts = image_only
            if scratch is not None:
                files = [Path(scratch) / f"{position}.h5ad" for position in range(len(parts))]
                for part, file in zip(parts, files, strict=True):
                    part.write_h5ad(file)
        if (repeated := pd.Index(np.concatenate([part.obs_names for part in parts])).duplicated()).any():
            raise ValueError(f"{repeated.sum()} observations share an index; pass index_columns that identify them")

        variables = [as_frame(part.var) for part in parts]
        var = pd.concat(variables)
        var = var.loc[~var.index.duplicated()]
        # An export without a child keeps CellProfiler's own means of it, which the roll-up replaces in the others.
        if "aggregation" in var:
            rolled = var.loc[var["aggregation"].notna(), "object"].unique()
            means = tuple(f"{owner}_Mean_{child}_" for owner in var["object"].dropna().unique() for child in rolled)
            var = var.loc[~var.index.str.startswith(means)]
        var = categorical_annotation(var.copy(), variables)
        dtypes: dict[str, Any] = {}
        for part in parts:
            for column, dtype in as_frame(part.obs).dtypes.items():
                dtypes.setdefault(str(column), dtype)
        shell = AnnData(uns={"mantispy": {**parts[0].uns["mantispy"], "image_table": pd.concat(images)}})
        record_params(shell, "read_cellprofiler", {"path": [str(path) for path in kinds], **params})
        if target is None:
            result = ad.concat([_conform(part, var, dtypes) for part in parts], join="inner", merge="same")
            result.uns = shell.uns
            return result
        for part, file in zip(parts, files, strict=True):
            if list(part.var_names) != list(var.index) or list(part.obs.columns) != list(dtypes):
                _conform(ad.read_h5ad(file), var, dtypes).write_h5ad(file)
        return _write(files, target, var, dict(shell.uns))


def _write(files: Sequence[Path], out: Path, var: pd.DataFrame, uns: dict[str, Any]) -> AnnData:
    """Concatenate the part files into `out` once, write `var` and `uns` into it, and read it back lazily.

    ``concat_on_disk`` writes no ``uns``, and the parts share their features, so `var` is written once instead of being merged.
    """
    from anndata.experimental import concat_on_disk
    from anndata.io import write_elem

    from mantispy.io._profiles import read

    concat_on_disk([str(file) for file in files], out, join="inner")
    if out.suffix == ".zarr":
        import zarr

        group: Any = zarr.open_group(out, mode="a")
        context: Any = nullcontext(group)
    else:
        import h5py

        context = h5py.File(out, "a")
    with context as store:
        for key, value in (("var", var), ("uns", uns)):
            if key in store:
                del store[key]
            write_elem(store, key, value)
    return read(out, lazy=True)
