"""scripts/build_dataset.py, whose fingerprint is what a rebuilt dataset is checked against."""

import pathlib
import runpy

fingerprint = runpy.run_path(str(pathlib.Path(__file__).parents[1] / "scripts/build_dataset.py"))["fingerprint"]
