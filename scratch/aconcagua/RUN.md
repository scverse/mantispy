# Aconcagua pipeline: how to regenerate everything

Handoff bundle for the OPS case study. These are scratch scripts, not package code, so don't open a PR with this folder in it.
All scripts use paths relative to a work directory `W` that contains an `acon/` folder. Run them with `python -I` from `W`.
The data come from the public bucket `s3://ops-explorer-public/aconcagua/aconcagua/` (anonymous, us-west-2).
They were run on a 4-core, 15 GB node: extraction takes about 1.5 h, and each plate takes a few minutes.

| Step | Command | Output |
|---|---|---|
| 0 | `python -I get_meta.py acon/meta.parquet` | screen-wide per-cell metadata (about 600 MB) |
| 0 | `curl` `.../aconcagua/aconcagua/visualizations/{interphase,mitotic}/aggregated_data.h5ad` | `acon/brief/{interphase,mitotic}.h5ad`, brieflow's published gene tables |
| 0 | DepMap 24Q4: figshare files 51065297 (Model.csv) and 51065489 (OmicsExpressionProteinCodingGenesTPMLogp1.csv) | `depmap/`; HeLa is ACH-001086 |
| 1 | `python -I mitotic_index.py` | `acon/mitotic_index_genes.csv` (copy in `data/`) |
| 2 | `python -I plan_subset.py acon/subset_v3` | `genes.csv` and `keep_prob.parquet` (copies in `data/`; put them in `acon/subset_v3/` and skip this step) |
| 3 | `python -I extract_subset.py acon/subset_v3 4` | `acon/subset_v3/parts/rg*.parquet`; resumable, deterministic per-cell draw |
| 4 | `for p in 1..8: python -I process_plate.py acon/subset_v3 $p` | `profiles_plate{p}.h5ad` (guide x well) |
| 4b | `... process_plate.py acon/subset_v3 $p cells` | `cells_plate{p}.h5ad` (single cells, about 0.8 GB per plate) |
| 5 | `python -I merge_plates.py` | `acon/aconcagua_guide_well.h5ad` (84,125 x 1,483, about 500 MB) |
| 6 | `analysis_1.py`, `analysis_2.py`, `analysis_3.py`, `mitosis.py` | exploratory analyses behind the notebook |
| 7 | `python build_notebook.py`, then execute it | the draft page (`optical_pooled_screens_draft.ipynb` is that output with the outputs stripped) |

`convert_brieflow.py` maps brieflow/cp_emulator feature names to mantispy's grammar, and `process_plate.py` imports it.
`extract_tileblock.py` pulls every cell in 100 tiles of plate 1 well A1, intended for the single-cell QC case study.
