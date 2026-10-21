import json

import anndata as ad
import numpy as np
import pandas as pd

import mantispy as mt
from mantispy._core.features import parse_feature_names
from mantispy._core.provenance import read_history


def _byo():
    return ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"]}, index=["0", "1"]),
        var=parse_feature_names(["Cells_AreaShape_Area"]),
    )


def test_stamp_seeds_history():
    a = mt.io.stamp(_byo(), resolution="well", feature_kind="measurement", copy=True)
    hist = a.uns["mantispy"]["history"]
    assert len(hist) == 1
    assert json.loads(hist[-1])["operation"] == "stamp"


def test_restamp_does_not_append():
    a = mt.io.stamp(_byo(), resolution="well", feature_kind="measurement", copy=True)
    mt.io.stamp(a)
    assert [r["operation"] for r in read_history(a)] == ["stamp"]


def test_history_round_trips_decodable(tmp_path):
    a = mt.io.stamp(_byo(), resolution="well", feature_kind="measurement", copy=True)
    p = tmp_path / "h.h5ad"
    mt.io.write(a, p)
    b = mt.io.read(p)
    assert [r["operation"] for r in read_history(b)] == ["stamp"]


def test_read_history_tolerates_empty_array():
    a = _byo()
    a.uns["mantispy"] = {"history": np.array([], dtype=np.float64)}
    assert read_history(a) == []
