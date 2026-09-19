"""scripts/build_dataset.py, whose fingerprint is what a rebuilt dataset is checked against."""

import pathlib
import runpy

import mantispy as mt

fingerprint = runpy.run_path(str(pathlib.Path(__file__).parents[1] / "scripts/build_dataset.py"))["fingerprint"]


def test_the_fingerprint_survives_a_round_trip_and_sees_one_changed_value(tmp_path):
    wells = mt.tl.aggregate(mt.ds.synthetic_plate(n_wells=8, n_cells=4, n_features=8, seed=0), min_cells=0)
    mt.io.write(wells, tmp_path / "wells.h5ad")
    assert fingerprint(mt.io.read(tmp_path / "wells.h5ad")) == fingerprint(wells)

    changed = wells.copy()
    changed.obs["Metadata_CellCount"] = changed.obs["Metadata_CellCount"] + 1
    assert fingerprint(changed) != fingerprint(wells)
