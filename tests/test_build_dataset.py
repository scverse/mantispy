"""scripts/build_dataset.py, whose fingerprint is what a rebuilt dataset is checked against."""

import pathlib
import runpy

import mantispy as mt

fingerprint = runpy.run_path(str(pathlib.Path(__file__).parents[1] / "scripts/build_dataset.py"))["fingerprint"]


def test_the_fingerprint_survives_a_round_trip_and_sees_what_changed(wells, tmp_path):
    mt.io.write(wells, tmp_path / "wells.h5ad")
    assert fingerprint(mt.io.read(tmp_path / "wells.h5ad")) == fingerprint(wells)

    counted = wells.copy()
    counted.obs["Metadata_CellCount"] = counted.obs["Metadata_CellCount"] + 1
    restamped = wells.copy()
    restamped.uns["mantispy"]["resolution"] = "cell"
    assert fingerprint(wells) not in {fingerprint(counted), fingerprint(restamped)}
