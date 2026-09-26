from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import spatialdata as sd

import mantispy as mt
from mantispy._core.frames import as_frame
from mantispy.ds._datasets import _DATASETS, TARGET2_DEFAULT, _plate


@pytest.mark.parametrize(("n_wells", "n_sites"), [(1, 1), (4, 2)])
def test_blobs_is_a_plate_with_everything_on_it(n_wells: int, n_sites: int) -> None:
    sdata = mt.ds.blobs(n_wells=n_wells, n_sites=n_sites)

    assert len(sdata.images) == n_wells * n_sites
    assert len(sdata.labels) == n_wells * n_sites * 3
    assert set(sdata.tables) == {"cells", "wells"}
    assert sdata.tables["wells"].n_obs == n_wells
    assert len(sdata.shapes["BLOBS01_wells"]) == 96


def test_blobs_looks_like_a_plate_that_was_read() -> None:
    sdata = mt.ds.blobs(n_wells=2, n_sites=2)
    cells = sdata["BLOBS01_A01_s1_cells"].to_numpy()
    nuclei = sdata["BLOBS01_A01_s1_nuclei"].to_numpy()
    var = sdata.tables["cells"].var

    assert {"BLOBS01", "BLOBS01_A01", "BLOBS01_A01_s1"} <= set(sdata.coordinate_systems)
    assert sd.get_extent(sdata["BLOBS01_A02_s1_image"], coordinate_system="BLOBS01")["x"][0] == pytest.approx(9000.0)
    assert set(np.unique(nuclei)) <= set(np.unique(cells))
    assert np.array_equal(sdata["BLOBS01_A01_s1_cytoplasm"].to_numpy(), np.where(nuclei > 0, 0, cells))
    assert var.loc["Cells_Intensity_MeanIntensity_DNA", "channel"] == "DNA"
    assert var.loc["Cells_Correlation_Correlation_DNA_RNA", "channel"] == "DNA|RNA"
    for table in sdata.tables.values():
        assert mt.io.validate(table).ok, mt.io.validate(table).errors


def test_the_same_seed_gives_the_same_data() -> None:
    def build(seed: int) -> np.ndarray:
        return np.asarray(mt.ds.blobs(seed=seed).tables["cells"].X)

    assert np.array_equal(build(1), build(1))
    assert not np.array_equal(build(1), build(2))


@pytest.mark.parametrize("name", sorted(_DATASETS))
def test_every_registered_dataset_has_a_loader_and_hashed_files(name: str) -> None:
    entry = _DATASETS[name]

    assert entry.files
    assert all(file.sha256 and (file.s3_key or file.url) for file in entry.files)
    # names starting with an underscore hold files another loader reads, such as JUMP's annotation tables
    assert name.startswith("_") or getattr(mt.ds, name)


def test_jump_target2_default_names_one_pinned_plate_per_source() -> None:
    """A typo in a default barcode otherwise surfaces only over the network, on every tutorial call.

    The s3 key carries the source, so the one-plate-from-every-source invariant is checkable offline.
    """
    source_of = {
        _plate(file.name): file.s3_key.split("/")[1]
        for file in _DATASETS["jump_target2"].files
        if file.s3_key and file.s3_key.startswith("cpg0016-jump/")
    }
    assert set(TARGET2_DEFAULT) <= set(source_of)
    assert sorted(source_of[plate] for plate in TARGET2_DEFAULT) == sorted(set(source_of.values()))


@pytest.mark.network
@pytest.mark.slow
# Entries without a shape do not return an AnnData: jump_plate is a SpatialData and jump_export a directory.
@pytest.mark.parametrize("name", sorted(name for name, entry in _DATASETS.items() if "shape" in entry.metadata))
def test_the_downloads_read_back_at_the_shape_the_registry_claims(name: str) -> None:
    adata = getattr(mt.ds, name)()

    assert list(adata.shape) == _DATASETS[name].metadata["shape"]
    assert adata.obs_names.is_unique
    if name == "bbbc021":
        assert adata.obs["Metadata_MOA"].notna().all()
        assert adata.obs["Metadata_Control"].sum() == 330
    # Regression test for #63: every well-level dataset publishes an exact per-well count upstream.
    # jump_cells, pooled_rare, scallops_arv471 and cp_posh are cell- or barcode-resolution and carry no per-well count.
    if name not in ("jump_cells", "pooled_rare", "scallops_arv471", "cp_posh"):
        assert (adata.obs["Metadata_CellCount"] > 0).all()
        # jump-profiling-recipe's count table, which jump_crispr reads, has no field count, and
        # JUMP-Lite publishes one count per well rather than per field.
        if name not in ("jump_crispr", "jump_lite"):
            assert adata.obs["Metadata_SiteCount"].between(1, 36).all()


@pytest.mark.network
@pytest.mark.slow
def test_jump_export_is_an_export_directory_that_reads() -> None:
    """The fixture tutorial 1 reads: a real ExportToSpreadsheet directory, untouched."""
    directory = mt.ds.jump_export()

    assert {path.name for path in directory.glob("*.csv")} == {
        "Cells.csv",
        "Cytoplasm.csv",
        "Experiment.csv",
        "Image.csv",
        "Nuclei.csv",
    }
    # JUMP gives Cytoplasm both parents and Cells none, so Cytoplasm is the primary that joins all three.
    adata = mt.io.read_profiles(directory, primary_object="Cytoplasm")
    assert str(adata.obs["Metadata_Well"].iloc[0]) == "J04"
    assert mt.io.validate(adata).ok


@pytest.mark.network
@pytest.mark.slow
def test_jump_cells_holds_controls_and_treatments_at_cell_resolution() -> None:
    adata = mt.ds.jump_cells()

    assert adata.uns["mantispy"]["resolution"] == "cell"
    assert adata.obs["Metadata_Well"].nunique() == 24
    assert adata.obs["Metadata_Site"].nunique() == 4
    assert bool(adata.obs["Metadata_Control"].any()) and not bool(adata.obs["Metadata_Control"].all())
    assert adata.obs_names.is_unique
    assert mt.io.validate(adata).ok


@pytest.mark.network
@pytest.mark.slow
def test_jump_cells_keeps_the_quality_of_every_field_and_where_each_cell_sits() -> None:
    """Each field numbers its images from one, and stacking them kept the first field's quality alone."""
    adata = mt.ds.jump_cells()
    images = adata.uns["mantispy"]["image_table"]

    assert len(images) == 24 * 4 and images.index.is_unique
    assert adata.obs["Metadata_ImageNumber"].isin(images.index).all()
    assert np.isfinite(as_frame(adata.obs)[["Metadata_Center_X", "Metadata_Center_Y"]].to_numpy(dtype=float)).all()
    assert not any("_Center_" in name for name in adata.var_names)


@pytest.mark.network
@pytest.mark.slow
def test_jump_plate_reads_the_fields_that_were_downloaded() -> None:
    sdata = mt.ds.jump_plate()

    assert len(sdata.images) == 2
    assert len(sdata.labels) == 2 * 3
    assert sdata.tables["cells"].n_obs > 0
    # The channel vocabulary comes from load_data, so no invented channel reaches var.
    channels = {part for value in sdata.tables["cells"].var["channel"].dropna() for part in str(value).split("|")}
    assert channels <= {"AGP", "Brightfield", "Brightfield_H", "Brightfield_L", "DNA", "ER", "Mito", "RNA"}


@pytest.mark.network
@pytest.mark.slow
def test_jump_cells_can_hand_back_only_the_selected_features() -> None:
    full = mt.ds.jump_cells()
    selected = mt.ds.jump_cells(selected=True)

    assert selected.n_obs == full.n_obs
    assert selected.n_vars < full.n_vars
    assert list(selected.var_names) == list(full.var_names[full.var["selected"].to_numpy()])


def test_selecting_without_the_annotation_is_refused() -> None:
    """The mask is computed against the negative controls, which only the annotation names."""
    with pytest.raises(KeyError, match="needs annotate"):
        mt.ds.jump_cells(annotate=False, selected=True)


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.parametrize(
    ("model", "n_features"),
    [
        ("openphenom", 384),
        ("dinov2", 384),
        ("dinov2_random", 384),
        ("subcell", 1536),
        ("morphem", 1920),
        ("cp_measure", 2550),
    ],
)
def test_every_jump_lite_feature_set_reads_back_at_its_own_width(model: str, n_features: int) -> None:
    """The registry records one shape and the generic shape test only ever loads the default model, so the
    other five widths are asserted nowhere else. The rows are the same wells in all six."""
    adata = mt.ds.jump_lite(model=model, annotate=False)

    assert adata.shape == (1536, n_features)
    assert adata.obs["Metadata_CellCount"].notna().all()


def test_jump_lite_names_its_feature_sets():
    """The six feature sets cover the same wells, so a typo has to fail loudly rather than
    silently fall back to one of them."""
    with pytest.raises(ValueError, match="model must be one of"):
        mt.ds.jump_lite(model="openphenome")
    assert "cp_measure" in mt.ds.JUMP_LITE_MODELS


def test_jump_lite_returns_every_feature_set_in_one_row_order(tmp_path, monkeypatch):
    """Each feature set is distributed with the wells in its own order, so two of them stacked by position
    paired one well's features with another's. Read back, every set lists the wells in the same order."""
    from mantispy.ds import _datasets

    ids = ["P2_A01", "P1_B01", "P1_A01"]
    wells = pd.DataFrame(
        {
            "Metadata_Plate": ["P2", "P1", "P1"],
            "Metadata_Well": ["A01", "B01", "A01"],
            "Metadata_Source": ["source_4", "source_3", "source_3"],
            "Metadata_Batch": "b1",
            "Metadata_id": ids,
            "value": [2.0, 1.0, 0.0],
        }
    )
    wells.rename(columns={"value": "openphenom_0"}).to_parquet(tmp_path / "openphenom.parquet")
    shuffled = wells.iloc[[2, 0, 1]].reset_index(drop=True)
    shuffled.rename(columns={"value": "dinov2_0"}).to_parquet(tmp_path / "dinov2.parquet")
    pd.DataFrame({"Metadata_id": ids, "cell_count": [100, 110, 120]}).to_parquet(tmp_path / "cell_count.parquet")

    def files(name, cache_dir, select=None):
        return [path for path in sorted(tmp_path.glob("*.parquet")) if select is None or select(path.name)]

    monkeypatch.setattr(_datasets, "_files", files)

    first, second = (mt.ds.jump_lite(model=model, annotate=False) for model in ("openphenom", "dinov2"))

    assert list(first.obs_names) == list(second.obs_names)
    assert np.asarray(first.X)[:, 0].tolist() == np.asarray(second.X)[:, 0].tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize(("model", "parsed"), [("openphenom", False), ("cp_measure", True)])
def test_jump_lite_does_not_read_an_embedding_dimension_as_a_measurement(tmp_path, monkeypatch, model, parsed):
    """The parser finds structure in names that have none: it reads `openphenom_nahualX_17` as the
    `nahualX` feature group of an `openphenom` object, so the model's own tensor names became
    feature families and `scale` became the dimension index. cp_measure is real measurements and
    keeps its annotation."""
    from mantispy.ds import _datasets

    names = (
        [f"{model}_nahualX_{index}" for index in range(4)]
        if model == "openphenom"
        else ["cell_0/max/sizeshapeSolidity", "nuclei_3/max/intensityIntensity_MeanIntensity"]
    )
    wells = pd.DataFrame(
        {
            "Metadata_Plate": ["P1", "P1"],
            "Metadata_Well": ["A01", "A02"],
            "Metadata_Source": ["source_3", "source_3"],
            "Metadata_Batch": ["b1", "b1"],
            "Metadata_id": ["P1_A01", "P1_A02"],
            **{name: [float(index), float(index) + 1] for index, name in enumerate(names)},
        }
    )
    wells.to_parquet(tmp_path / f"{model}.parquet")
    pd.DataFrame({"Metadata_id": ["P1_A01", "P1_A02"], "cell_count": [120, 130]}).to_parquet(
        tmp_path / "cell_count.parquet"
    )

    def files(name, cache_dir, select=None):
        return [path for path in sorted(tmp_path.glob("*.parquet")) if select is None or select(path.name)]

    monkeypatch.setattr(_datasets, "_files", files)

    adata = mt.ds.jump_lite(model=model, annotate=False)

    assert bool(adata.var["feature_group"].notna().any()) is parsed
    # The compartment and the channel are what an embedding cannot offer, so they are what cp_measure has to keep.
    if parsed:
        assert list(adata.var["object"]) == ["cell", "nuclei"]
        assert list(adata.var["feature_group"]) == ["sizeshape", "intensity"]
        assert list(adata.var["channel"]) == ["0", "3"]
    assert list(adata.obs["Metadata_CellCount"]) == [120, 130]


def _write_scallops_fixture(path):
    """A small stand-in for the SCALLOPS upstream table, with the columns scallops_arv471 reads.

    It holds both conditions, a boundary-touching cell, and a cell missing a feature, so the loader's
    condition filter, boundary drop and NaN drop are all exercised. The ARV-471 arm has thirty
    non-targeting cells, two targeted genes and one olfactory-receptor negative control.
    """
    from mantispy.ds._datasets import _SCALLOPS_FEATURES

    rng = np.random.default_rng(0)
    # (gene_symbol, sgRNA_id, type, n_cells) for the clean ARV-471 cells.
    groups = [
        ("NTC", "NTC_1", "ntc", 15),
        ("NTC", "NTC_2", "ntc", 15),
        ("ESR1", "ESR1_1", "target", 8),
        ("ESR1", "ESR1_2", "target", 8),
        ("CRBN", "CRBN_1", "target", 8),
        ("CRBN", "CRBN_2", "target", 8),
        ("OR1L4", "OR1L4_1", "neg", 8),
    ]
    rows = []
    for gene, guide, kind, n in groups:
        for _ in range(n):
            rows.append((gene, guide, kind, "A", 3, "ARV-471", False))
    clean = len(rows)  # 70
    # DMSO cells (dropped by the condition filter), a boundary cell and two feature-NaN cells (dropped).
    rows += [("NTC", "NTC_1", "ntc", "A", 1, "DMSO", False) for _ in range(5)]
    rows += [("ESR1", "ESR1_1", "target", "A", 3, "ARV-471", True) for _ in range(3)]
    nan_rows = [("CRBN", "CRBN_1", "target", "A", 3, "ARV-471", False) for _ in range(2)]
    rows += nan_rows

    frame = pd.DataFrame(
        rows,
        columns=[
            "gene_symbol",
            "sgRNA_id",
            "type",
            "plate",
            "well",
            "Condition",
            "Cells_Location_IntersectsBoundary_IF",
        ],
    )
    for feature in _SCALLOPS_FEATURES:
        frame[feature] = rng.normal(size=len(frame))
    # Make the last two rows (the CRBN cells added above) miss a feature so the NaN drop removes them.
    frame.loc[frame.index[-2:], _SCALLOPS_FEATURES[0]] = np.nan
    frame.to_parquet(path)
    return clean


def _patch_scallops_files(monkeypatch, path):
    from mantispy.ds import _datasets

    monkeypatch.setattr(_datasets, "_files", lambda name, cache_dir, select=None: [path])


def test_scallops_arv471_loads_clean_cell_resolution(tmp_path, monkeypatch):
    """The loader keeps only the ARV-471 cells with a full feature vector and validates against the schema."""
    path = tmp_path / "fig3.pq"
    clean = _write_scallops_fixture(path)
    _patch_scallops_files(monkeypatch, path)

    adata = mt.ds.scallops_arv471()

    assert adata.shape == (clean, 9)
    assert adata.uns["mantispy"]["resolution"] == "cell"
    report = mt.io.validate(adata)
    assert report.ok, report.errors
    assert adata.obs_names.is_unique
    # The condition filter, boundary drop and NaN drop leave nothing but the clean ARV-471 cells.
    assert "Condition" not in adata.obs
    assert set(adata.obs["Metadata_ControlClass"].astype(str)) == {"ntc", "target", "neg"}


def test_scallops_arv471_maps_controls_and_guides(tmp_path, monkeypatch):
    """NTC becomes the non-targeting control, and every other guide is a targeted perturbation."""
    path = tmp_path / "fig3.pq"
    _write_scallops_fixture(path)
    _patch_scallops_files(monkeypatch, path)

    adata = mt.ds.scallops_arv471()
    obs = as_frame(adata.obs)

    assert "NTC" not in set(obs["Metadata_Gene"].astype(str))
    assert "nontargeting" in set(obs["Metadata_Gene"].astype(str))
    assert obs["Metadata_Gene"].nunique() == 4  # nontargeting, ESR1, CRBN, OR1L4
    assert obs["Metadata_sgRNA"].nunique() == 7
    assert list(obs["Metadata_Perturbation"].astype(str)) == list(obs["Metadata_sgRNA"].astype(str))
    # Both control and targeted cells are present, and only the NTC cells are marked control.
    control = obs["Metadata_Control"].to_numpy()
    assert control.dtype == bool
    assert control.any() and not control.all()
    assert control.sum() == 30
    assert set(obs.loc[control, "Metadata_Gene"].astype(str)) == {"nontargeting"}
    assert list(obs["Metadata_Well"].unique()) == ["W03"]


def test_scallops_arv471_runs_hit_calling(tmp_path, monkeypatch):
    """The object drives hit_calling against the non-targeting controls and returns a table of groups."""
    import inspect

    path = tmp_path / "fig3.pq"
    _write_scallops_fixture(path)
    _patch_scallops_files(monkeypatch, path)

    adata = mt.ds.scallops_arv471()

    kwargs = {"groupby": "Metadata_Gene", "reference": "negcon", "n_permutations": 50, "seed": 0, "copy": True}
    # block= is the well-block permutation null (#68); pass it once it reaches this build's signature.
    if "block" in inspect.signature(mt.tl.hit_calling).parameters:
        kwargs["block"] = "Metadata_sgRNA"
    result = mt.tl.hit_calling(adata, **kwargs)

    hits = result.uns["mantispy"]["hits"]
    assert isinstance(hits, pd.DataFrame)
    assert not hits.empty
    assert {"group", "is_hit"} <= set(hits.columns)


#: A handful of CellStats-style feature names for the cp_posh fixture. Their names carry no CellProfiler
#: structure, as the real ones do not, so the loader files them all into X with an empty var annotation.
_CP_POSH_FIXTURE_FEATURES = (
    "nucleus_mask_height",
    "nucleus_mask_area_pixel_sq",
    "cell_mask_eccentricity",
    "median_cell_mask_DAPI_pixel_intensity",
    "mean_cell_mask_WGA_pixel_intensity",
    "cyto_mask_solidity",
)


def _write_cp_posh_fixture(path):
    """A small stand-in for the cp-POSH upstream parquet, with the columns cp_posh reads.

    The metadata columns are written as the pandas MultiIndex, as the real file stores them, so the loader's
    ``reset_index`` brings them back. It holds both control classes (non-targeting and intergenic) and three
    targeted genes, so the control mapping and the gene and guide counts are all exercised.
    """
    from mantispy.ds._datasets import _CP_POSH_METADATA

    rng = np.random.default_rng(0)
    # (gene_id, barcode, n_cells); the targeted genes are shifted off the controls so hit_calling has something to find.
    groups = [
        ("nontargeting", "ntc_1", 20, 0.0),
        ("nontargeting", "ntc_2", 20, 0.0),
        ("intergenic", "int_1", 15, 0.0),
        ("intergenic", "int_2", 15, 0.0),
        ("KIF18A", "KIF18A_1", 10, 3.0),
        ("KIF18A", "KIF18A_2", 10, 3.0),
        ("PSMB1", "PSMB1_1", 10, 2.5),
        ("ARPC4", "ARPC4_1", 10, 2.0),
    ]
    rows, shifts = [], []
    for gene, guide, n, shift in groups:
        for _ in range(n):
            # In _CP_POSH_METADATA order: barcode (the guide), gene_id (the gene), treatment, plate_well, plate, ID.
            rows.append((guide, gene, "none", "EL37_B04", "EL37", f"EL37_B04_{len(rows)}"))
            shifts.append(shift)
    frame = pd.DataFrame(rows, columns=list(_CP_POSH_METADATA))
    for feature in _CP_POSH_FIXTURE_FEATURES:
        frame[feature] = rng.normal(size=len(frame)) + np.asarray(shifts)
    controls = int(sum(n for _, _, n, _ in groups[:4]))  # non-targeting + intergenic
    frame.set_index(list(_CP_POSH_METADATA)).to_parquet(path)
    return len(frame), controls


def _patch_cp_posh_files(monkeypatch, path):
    from mantispy.ds import _datasets

    monkeypatch.setattr(_datasets, "_files", lambda name, cache_dir, select=None: [path])


def test_cp_posh_loads_clean_cell_resolution(tmp_path, monkeypatch):
    """Every cell keeps a full feature vector and the object validates against the schema at cell resolution."""
    path = tmp_path / "cp_posh.pq"
    cells, _ = _write_cp_posh_fixture(path)
    _patch_cp_posh_files(monkeypatch, path)

    adata = mt.ds.cp_posh()

    assert adata.shape == (cells, len(_CP_POSH_FIXTURE_FEATURES))
    assert adata.uns["mantispy"]["resolution"] == "cell"
    assert adata.X.dtype == np.float32
    report = mt.io.validate(adata)
    assert report.ok, report.errors
    assert adata.obs_names.is_unique
    # The upstream metadata columns become obs, not features, and none leak into X or var.
    assert set(adata.var_names) == set(_CP_POSH_FIXTURE_FEATURES)
    assert not adata.var["is_feature"].isna().any()


def test_cp_posh_maps_controls_and_guides(tmp_path, monkeypatch):
    """Both control classes and the targeted genes are present, with only the controls flagged."""
    path = tmp_path / "cp_posh.pq"
    _, controls = _write_cp_posh_fixture(path)
    _patch_cp_posh_files(monkeypatch, path)

    adata = mt.ds.cp_posh()
    obs = as_frame(adata.obs)

    assert obs["Metadata_Gene"].nunique() == 5  # nontargeting, intergenic, KIF18A, PSMB1, ARPC4
    assert obs["Metadata_sgRNA"].nunique() == 8
    assert list(obs["Metadata_Perturbation"].astype(str)) == list(obs["Metadata_sgRNA"].astype(str))
    # Both control classes and a targeted gene are present.
    genes = set(obs["Metadata_Gene"].astype(str))
    assert {"nontargeting", "intergenic"} <= genes
    assert genes - {"nontargeting", "intergenic"}
    # Only the two control classes are flagged, and every other gene is a targeted perturbation.
    control = obs["Metadata_Control"].to_numpy()
    assert control.dtype == bool
    assert control.any() and not control.all()
    assert control.sum() == controls
    assert set(obs.loc[control, "Metadata_Gene"].astype(str)) == {"nontargeting", "intergenic"}
    # The well is read out of plate_well by dropping the plate prefix.
    assert list(obs["Metadata_Well"].unique()) == ["B04"]


def test_cp_posh_runs_hit_calling(tmp_path, monkeypatch):
    """The object drives hit_calling against the non-targeting and intergenic controls and returns a table of groups."""
    path = tmp_path / "cp_posh.pq"
    _write_cp_posh_fixture(path)
    _patch_cp_posh_files(monkeypatch, path)

    adata = mt.ds.cp_posh()

    result = mt.tl.hit_calling(adata, groupby="Metadata_Gene", reference="negcon", n_permutations=50, seed=0, copy=True)

    hits = result.uns["mantispy"]["hits"]
    assert isinstance(hits, pd.DataFrame)
    assert not hits.empty
    assert {"group", "is_hit"} <= set(hits.columns)
