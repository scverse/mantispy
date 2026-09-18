"""A synthetic plate whose every injected effect is recorded as ground truth.

Used by the test suite and the tutorials, so both run offline and can check that a method recovers a known effect.

The default channel names are the Cell Painting ones, but any ``channels`` work, including a single channel.
"""

from __future__ import annotations

from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd

from mantispy._core.features import parse_feature_names
from mantispy._core.frames import categorize_metadata
from mantispy._core.plate import PLATE_FORMATS, well_col, well_name, well_row
from mantispy._core.schema import stamp

#: Default channel vocabulary (Cell Painting). No other code assumes these names.
DEFAULT_CHANNELS = ("DNA", "ER", "RNA", "AGP", "Mito")

_OBJECTS = ("Cells", "Nuclei", "Cytoplasm")
_SHAPE_FEATURES = ("Area", "Perimeter", "Eccentricity", "FormFactor", "Solidity")
_INTENSITY_FEATURES = ("MeanIntensity", "MaxIntensity", "IntegratedIntensity", "StdIntensity")
_TEXTURE_FEATURES = ("Contrast", "Correlation", "Entropy", "Variance")

#: How many robust standard deviations a degraded image's metrics are shifted by.
#: Large, because an out-of-focus image lies far from the rest.
BAD_IMAGE_SHIFT = 8.0

_IMAGE_QC_METRICS = {
    "FocusScore": (0.50, 0.02),
    "PowerLogLogSlope": (-2.0, 0.05),
    "PercentMaximal": (0.10, 0.01),
    "PercentMinimal": (0.05, 0.01),
    "Saturation": (0.02, 0.005),
}


def _feature_names(n_features: int, channels: Sequence[str]) -> list[str]:
    """CellProfiler-style names spanning the given channels."""
    names: list[str] = []
    for obj in _OBJECTS:
        names += [f"{obj}_AreaShape_{feature}" for feature in _SHAPE_FEATURES]
        for channel in channels:
            names += [f"{obj}_Intensity_{feature}_{channel}" for feature in _INTENSITY_FEATURES]
            names += [f"{obj}_Texture_{feature}_{channel}_3_00_256" for feature in _TEXTURE_FEATURES]
    if len(names) < n_features:
        raise ValueError(
            f"cannot generate {n_features} distinct names from {len(channels)} channel(s); "
            f"at most {len(names)} are available. Pass fewer features or more channels."
        )
    return names[:n_features]


def _wells(n_wells: int) -> list[str]:
    size = next((s for s in sorted(PLATE_FORMATS) if s >= n_wells), None)
    if size is None:
        raise ValueError(f"{n_wells} wells exceeds the largest standard plate format")
    n_rows, n_cols = PLATE_FORMATS[size]
    return [well_name(row, col) for row in range(n_rows) for col in range(n_cols)][:n_wells]


def _random_rotation(rng: np.random.Generator, size: int, strength: float) -> np.ndarray:
    """A rotation blended toward the identity, so batch effects are not purely additive."""
    q, _ = np.linalg.qr(rng.standard_normal((size, size)))
    return (1.0 - strength) * np.eye(size) + strength * q


def synthetic_plate(
    n_plates: int = 1,
    n_wells: int = 384,
    n_cells: int = 200,
    n_features: int = 60,
    channels: Sequence[str] = DEFAULT_CHANNELS,
    n_perturbations: int = 8,
    effect_size: float = 1.0,
    n_images_per_well: int = 1,
    n_batches: int = 1,
    plate_effect: float = 0.0,
    row_gradient: float = 0.0,
    col_gradient: float = 0.0,
    batch_effect: float = 0.0,
    batch_rotation: float = 0.0,
    confounder_effect: float = 0.0,
    n_bad_images: int = 0,
    n_correlated_pairs: int = 0,
    n_constant_features: int = 0,
    nan_fraction: float = 0.0,
    seed: int = 0,
) -> ad.AnnData:
    """Generate a synthetic plate with known ground truth.

    Every injected effect is recorded under ``uns["mantispy"]["truth"]`` so tests and tutorials can check that a method recovers it.

    Args:
        n_plates: Number of plates.
        n_wells: Wells per plate.
        n_cells: Cells per well, or the Poisson mean of that count when ``confounder_effect`` is set.
        n_features: Number of features, before correlated copies are added.
        channels: Channel vocabulary used to build feature names. Any names work.
        n_perturbations: Number of treatments; a ``DMSO`` negative control is always added.
        effect_size: Shift applied to the features affected by each perturbation.
        n_images_per_well: Fields of view per well, which sets ``Metadata_ImageNumber``.
        n_batches: Plates are assigned round-robin to this many batches.
        plate_effect: Shift added to every feature, multiplied by the plate's index.
        row_gradient: Shift added across the plate rows, for plate-position correction.
        col_gradient: Shift added across the plate columns, for plate-position correction.
        batch_effect: Standard deviation of a random per-batch offset added to every feature.
        batch_rotation: Strength of a random per-batch rotation of the first ``min(5, n_features)`` features.
        confounder_effect: Makes a fifth of the features scale with the well's cell count.
        n_bad_images: Images given a degraded image-quality profile and noisier cells.
        n_correlated_pairs: Near-copies of existing features to add, for feature selection to remove.
        n_constant_features: Features set to a constant, for feature selection to remove.
        nan_fraction: Fraction of entries set to NaN.
        seed: Seed for reproducibility.

    Returns:
        An :class:`~anndata.AnnData` at cell resolution, carrying every injected effect under ``uns["mantispy"]["truth"]``, the channel vocabulary under ``channels`` and the per-image quality metrics under ``image_table``.

    Raises:
        ValueError: `n_features` needs more distinct names than `channels` can spell, or `n_wells` exceeds the largest standard plate format.
    """
    rng = np.random.default_rng(seed)
    channels = list(channels)
    names = _feature_names(n_features, channels)
    wells = _wells(n_wells)

    perturbations = ["DMSO"] + [f"pert{i:02d}" for i in range(n_perturbations)]
    well_perturbation = [perturbations[i % len(perturbations)] for i in range(n_wells)]

    affected: dict[str, list[str]] = {"DMSO": []}
    for perturbation in perturbations[1:]:
        affected[perturbation] = sorted(rng.choice(names, size=max(1, n_features // 5), replace=False).tolist())

    blocks, obs_records = [], []
    image_counter = 0
    image_rows = []

    for plate_index in range(n_plates):
        plate = f"Plate{plate_index + 1:02d}"
        batch = f"Batch{plate_index % n_batches + 1}"
        for well_index, well in enumerate(wells):
            perturbation = well_perturbation[well_index]
            count = int(max(2, rng.poisson(n_cells))) if confounder_effect else n_cells

            block = rng.standard_normal((count, n_features))
            block += plate_effect * plate_index
            block += row_gradient * (well_row(well) / 8.0)
            block += col_gradient * (well_col(well) / 12.0)
            if affected[perturbation]:
                block[:, [names.index(name) for name in affected[perturbation]]] += effect_size

            images = [image_counter + i for i in range(n_images_per_well)]
            image_counter += n_images_per_well
            cell_images = np.array([images[i % len(images)] for i in range(count)])

            blocks.append(block)
            obs_records.append(
                pd.DataFrame(
                    {
                        "Metadata_Plate": plate,
                        "Metadata_Well": well,
                        "Metadata_Row": well_row(well),
                        "Metadata_Col": well_col(well),
                        "Metadata_Perturbation": perturbation,
                        "Metadata_Control": perturbation == "DMSO",
                        "Metadata_Batch": batch,
                        "Metadata_ImageNumber": cell_images,
                        "Metadata_CellCount": count,
                    }
                )
            )
            for image in images:
                image_rows.append({"ImageNumber": image, "Metadata_Plate": plate, "Metadata_Well": well})

    X = np.vstack(blocks)
    obs = pd.concat(obs_records, ignore_index=True)
    truth: dict = {"affected_features": affected, "effect_size": float(effect_size)}

    if confounder_effect:
        counts = obs["Metadata_CellCount"].to_numpy(dtype=float)
        standardized = (counts - counts.mean()) / (counts.std() or 1.0)
        confounded = names[: max(1, n_features // 5)]
        indices = [names.index(name) for name in confounded]
        X[:, indices] += confounder_effect * standardized[:, None]
        truth["confounded_features"] = confounded

    batch_offsets: dict[str, list[float]] = {}
    if batch_effect or batch_rotation:
        span = min(5, n_features)
        for batch in sorted(obs["Metadata_Batch"].unique()):
            rows = (obs["Metadata_Batch"] == batch).to_numpy()
            if batch_effect:
                offset = rng.normal(0.0, batch_effect, n_features)
                X[rows] += offset
                batch_offsets[batch] = offset.tolist()
            if batch_rotation:
                X[np.ix_(rows, np.arange(span))] = X[np.ix_(rows, np.arange(span))] @ _random_rotation(
                    rng, span, batch_rotation
                )
    truth["batch_offsets"] = batch_offsets

    image_table = pd.DataFrame(image_rows).set_index("ImageNumber")
    for metric, (location, spread) in _IMAGE_QC_METRICS.items():
        image_table[f"Image_ImageQuality_{metric}_{channels[0]}"] = rng.normal(location, spread, len(image_table))

    bad_images: list[int] = []
    if n_bad_images:
        bad_images = sorted(rng.choice(image_table.index.to_numpy(), size=n_bad_images, replace=False).tolist())
        focus = f"Image_ImageQuality_FocusScore_{channels[0]}"
        slope = f"Image_ImageQuality_PowerLogLogSlope_{channels[0]}"
        image_table.loc[bad_images, focus] -= BAD_IMAGE_SHIFT * _IMAGE_QC_METRICS["FocusScore"][1]
        image_table.loc[bad_images, slope] += BAD_IMAGE_SHIFT * _IMAGE_QC_METRICS["PowerLogLogSlope"][1]
        degraded = obs["Metadata_ImageNumber"].isin(bad_images).to_numpy()
        X[degraded] += rng.normal(0.0, 2.0, (int(degraded.sum()), n_features))
    truth["bad_images"] = bad_images

    correlated_pairs: list[tuple[str, str]] = []
    if n_correlated_pairs:
        extra = []
        for i in range(n_correlated_pairs):
            source = names[i % n_features]
            column = X[:, names.index(source)]
            extra.append(column + rng.normal(0.0, 0.05 * (column.std() or 1.0), column.size))
            correlated_pairs.append((source, f"{source}Copy{i}"))
        X = np.hstack([X, np.column_stack(extra)])
        names = names + [copy for _, copy in correlated_pairs]
    truth["correlated_pairs"] = correlated_pairs

    constant_features: list[str] = []
    if n_constant_features:
        constant_features = names[-n_constant_features - len(correlated_pairs) : len(names) - len(correlated_pairs)]
        for value, name in enumerate(constant_features, start=1):
            X[:, names.index(name)] = float(value)
    truth["constant_features"] = constant_features

    if nan_fraction:
        X[rng.random(X.shape) < nan_fraction] = np.nan

    obs = categorize_metadata(obs)
    obs.index = pd.Index([f"cell_{i}" for i in range(len(obs))])
    adata = ad.AnnData(
        X=X.astype(np.float32),
        obs=obs,
        var=parse_feature_names(names, channels=channels),
    )
    stamp(adata, resolution="cell")
    adata.uns["mantispy"]["channels"] = channels
    adata.uns["mantispy"]["image_table"] = image_table
    adata.uns["mantispy"]["truth"] = truth
    return adata
