"""JUMP profiles and the annotation join.

The join is tested offline against fabricated tables, since its failure cases (a
duplicated key, a well the annotation does not cover, the control flag) do not depend on
the network. The download is tested separately and marked.
"""

import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy.io import _jump


@pytest.fixture
def fake_metadata(monkeypatch):
    """Stand in for the two downloaded tables."""
    wells = pd.DataFrame(
        {
            "Metadata_Source": ["source_1"] * 4,
            "Metadata_Plate": ["P1"] * 4,
            "Metadata_Well": ["A01", "A02", "A03", "A04"],
            "Metadata_JCP2022": [_jump.NEGATIVE_CONTROL, "JCP2022_000001", "JCP2022_000002", _jump.NEGATIVE_CONTROL],
        }
    )
    compounds = pd.DataFrame(
        {
            "Metadata_JCP2022": ["JCP2022_000001", "JCP2022_000002"],
            "Metadata_InChIKey": ["AAA-KEY", "BBB-KEY"],
        }
    )
    monkeypatch.setattr(_jump, "jump_metadata", lambda name: wells if name == "well" else compounds)
    return wells


@pytest.fixture
def profiles(tmp_path):
    """A JUMP-shaped parquet: source, plate, well and features, nothing else."""
    frame = pd.DataFrame(
        {
            "Metadata_Source": ["source_1"] * 5,
            "Metadata_Plate": ["P1"] * 5,
            "Metadata_Well": ["A01", "A02", "A03", "A04", "A05"],
            "Cells_AreaShape_Area": np.arange(5.0),
            "Nuclei_Intensity_MeanIntensity_DNA": np.arange(5.0) * 2,
        }
    )
    path = tmp_path / "P1.parquet"
    frame.to_parquet(path)
    return path


def test_the_join_names_the_perturbation_and_the_controls(profiles, fake_metadata):
    adata = mt.io.read_jump(profiles)
    assert list(adata.obs["Metadata_JCP2022"].astype(str)[:4]) == list(fake_metadata["Metadata_JCP2022"])
    assert list(adata.obs["Metadata_Control"].to_numpy()[:4]) == [True, False, False, True]
    assert adata.obs["Metadata_Perturbation"].nunique() == 3  # two compounds, DMSO, and the unannotated well
    assert mt.io.validate(adata).ok, mt.io.validate(adata).errors


def test_a_well_the_annotation_misses_is_kept_and_logged(profiles, fake_metadata, caplog):
    with caplog.at_level("WARNING", logger="mantispy"):
        adata = mt.io.read_jump(profiles)
    assert adata.n_obs == 5  # A05 has no annotation and is still a measurement
    assert adata.obs["Metadata_JCP2022"].isna().sum() == 1
    assert "no JUMP annotation" in caplog.text


def test_reading_without_annotation_leaves_the_three_columns(profiles):
    adata = mt.io.read_jump(profiles, annotate=False)
    assert "Metadata_JCP2022" not in adata.obs
    assert set(adata.obs.columns) == {"Metadata_Source", "Metadata_Plate", "Metadata_Well"}


def test_annotating_something_that_is_not_jump_says_what_is_missing(cells):
    with pytest.raises(KeyError, match="mt.io.read_jump"):
        mt.pp.annotate_jump(cells)


def test_a_duplicated_annotation_row_is_refused(profiles, fake_metadata, monkeypatch):
    doubled = pd.concat([fake_metadata, fake_metadata.iloc[[0]]], ignore_index=True)
    monkeypatch.setattr(
        _jump,
        "jump_metadata",
        lambda name: doubled if name == "well" else pd.DataFrame({"Metadata_JCP2022": [], "Metadata_InChIKey": []}),
    )
    with pytest.raises(Exception, match="merge|duplicate|m:1"):
        mt.io.read_jump(profiles)


@pytest.fixture
def fake_crispr_metadata(monkeypatch):
    """Stand in for the CRISPR tables: genes, control types and loci."""
    tables = {
        "crispr": pd.DataFrame(
            {
                "Metadata_JCP2022": ["JCP2022_800001", "JCP2022_800002", "JCP2022_805264", "JCP2022_900001"],
                "Metadata_NCBI_Gene_ID": [None, None, "5347", "5690"],
                "Metadata_Symbol": ["no-guide", "non-targeting", "PLK1", "PSMB2"],
            }
        ),
        "perturbation_control": pd.DataFrame(
            {
                "Metadata_JCP2022": ["JCP2022_033924", "JCP2022_800001", "JCP2022_800002", "JCP2022_805264"],
                "Metadata_pert_type": ["negcon", "negcon", "negcon", "poscon"],
                "Metadata_Name": ["DMSO", "no-guide", "non-targeting", "PLK1"],
                "Metadata_modality": ["compound", "crispr", "crispr", "crispr"],
            }
        ),
        "gene_chromosome_map": pd.DataFrame(
            {"Approved_symbol": ["PLK1", "PSMB2"], "Locus": ["16p12.2", "1p34.3"], "Chromosome": ["16", "1"]}
        ),
    }
    monkeypatch.setattr(_jump, "jump_metadata", tables.__getitem__)


def test_a_crispr_well_is_named_by_its_gene_and_its_controls_by_their_type(fake_crispr_metadata):
    """Assembled profiles already carry Metadata_JCP2022, so no well table is needed to name them."""
    wells = mt.ds.synthetic_plate(n_wells=4, n_cells=1, n_features=3, seed=0)
    wells.obs["Metadata_JCP2022"] = ["JCP2022_800001", "JCP2022_800002", "JCP2022_805264", "JCP2022_900001"]
    mt.pp.annotate_jump(wells, kind="crispr")

    obs = wells.obs
    assert list(obs["Metadata_Perturbation"].astype(str)) == ["no-guide", "non-targeting", "PLK1", "PSMB2"]
    assert list(obs["Metadata_Gene"].astype(str)) == list(obs["Metadata_Perturbation"].astype(str))
    assert list(obs["Metadata_Control_Type"].astype(str)) == ["negcon", "negcon", "poscon", "trt"]
    assert list(obs["Metadata_Control"].to_numpy()) == [True, True, False, False]
    assert list(obs["Metadata_ChromosomeArm"].astype(object).fillna("")) == ["", "", "16p", "1p"]


def test_corum_reads_one_row_per_complex_and_member(tmp_path, monkeypatch):
    from mantispy.ds import _datasets

    path = tmp_path / "CORUM_clusters.tsv"
    path.write_text("20S proteasome\tPSMA1 PSMB2 PSMB2\nCCT complex\tTCP1 CCT2\n")
    monkeypatch.setattr(_datasets, "_files", lambda name, cache_dir=None, select=None: [path])

    complexes = mt.ds.corum()
    assert complexes.to_dict("list") == {
        "source": ["20S proteasome", "20S proteasome", "CCT complex", "CCT complex"],
        "target": ["PSMA1", "PSMB2", "TCP1", "CCT2"],
    }


@pytest.mark.network
@pytest.mark.slow
def test_jump_crispr_names_its_genes_and_controls():
    adata = mt.ds.jump_crispr()
    kinds = adata.obs["Metadata_Control_Type"].astype(str).value_counts().to_dict()
    assert kinds == {"trt": 43138, "negcon": 7478, "poscon": 569}
    assert adata.obs.loc[adata.obs["Metadata_Control_Type"] == "trt", "Metadata_Perturbation"].nunique() > 7900
    assert mt.io.validate(adata).ok, mt.io.validate(adata).errors


@pytest.mark.network
def test_jump_target2_loads_a_plate_from_every_source():
    # The shape and the per-well counts are asserted against the registry in test_ds.py.
    adata = mt.ds.jump_target2()
    assert adata.obs["Metadata_Source"].nunique() == 11
    assert adata.obs["Metadata_Perturbation"].nunique() == 302
    assert int(adata.obs["Metadata_Control"].sum()) == 898  # 256 of them on source_9's 1536-well plate
    assert mt.io.validate(adata).ok, mt.io.validate(adata).errors
    well = adata.obs[(adata.obs["Metadata_Plate"] == "JCPQC051") & (adata.obs["Metadata_Well"] == "A01")]
    assert well[["Metadata_CellCount", "Metadata_SiteCount"]].values.tolist() == [[1853.0, 9.0]]


@pytest.mark.network
def test_jump_target2_counts_a_source_that_publishes_only_the_object_count():
    """source_6's backend tables have Metadata_Object_Count and no Metadata_Count_Cells."""
    adata = mt.ds.jump_target2(plates=["110000294936"], annotate=False)
    assert (adata.obs["Metadata_CellCount"] > 0).all()


@pytest.mark.network
def test_jump_target2_rejects_a_plate_it_does_not_have():
    with pytest.raises(KeyError, match="available:"):
        mt.ds.jump_target2(plates=["nope"])
