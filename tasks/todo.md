# Stage all remaining datasets on S3 (base + variants), flip loaders to fetch

Branch: feat/stage-all-datasets. Templates: #167 (staging machinery) + #168 (bbbc021/rohban).
I build finals + write code; coordinator uploads + pushes + opens PR.

## Pattern per dataset (mirror bbbc021/rohban)
- `_assemble_<name>` in `_build.py` = the raw pipeline (skip `.h5ad` inputs via select).
- `build_<name>_variants` returns `{filename: AnnData}`, each through `_stable` (+ `_zero_nonfinite` on agg/selected).
- `_shipped_<name>` fetches the same filenames through the public API.
- STAGED entry `(builder, shipped_loader, heavy)`; heavy=True if raw inputs > ~50 MB.
- `_<NAME>_VARIANTS` map in `_datasets.py`; loader validates bool flags, fetches the h5ad, `read()`.
- registry: append the h5ad rows (url + sha256) after the raw rows; raw rows stay for drift rebuild.
- build: `python scripts/build_staged_datasets.py --only <name> --print-sha256 --out $HOME/build_all/`.

## Order (smallest raw first, commit in groups)
- [x] neuropainting  (base only)            -- validated the whole pipeline
- [x] chroma         (base only)
- [x] pooled_rare    (base, selected)
- [x] oasis_pilot    (base, agg; annotate=False keeps raw path)
- [x] pki            (base, selected, agg, agg_selected); fixed stale gallery test 5857->5839
- [x] scallops_arv471(base, agg; cells->guide via tl.aggregate)
- [x] jump_crispr    (base, agg; guides->gene consensus; annotate=False keeps raw path)
- [x] jump_cells     (base, selected, agg; agg is cells->WELL per the audit; keep annotate=/selected=)
- [x] cp_posh        (base, selected, agg, agg_selected)
- [x] jump_target2   (default 11-plate base hosted; non-default plates= keeps the raw path)
- [x] jump_lite      (6 per-model keyed finals; annotate=False keeps raw path)

## Review
- All 11 datasets migrated, none deferred. 28 finals in $HOME/build_all/ (all sha256 in registry).
- jump_cells `aggregated`: the audit says aggregate by (Metadata_Plate, Metadata_Well) i.e. cells->well,
  which is also `tl.aggregate`'s default and lines up with the well-level jump_target2; the task text said
  "Metadata_Perturbation unit", the audit wins (noted per the follow-the-audit rule).
- pki base is (3072, 5839) from the live `_augmented`; the gallery test asserted (3072, 5857), which was
  stale against the pinned data (verified `_augmented("pki", None).n_vars == 5839`). Updated the assertion.
- Deterministic spot-checks (build twice, identical sha256): neuropainting, pki, scallops_arv471, jump_cells.
- api-guards + offline gallery tests pass; ruff clean; every built h5ad round-trips and `io.validate` is ok.

## Verify (mine)
- [ ] `import mantispy; import mantispy.ds._build` (no cycle)
- [ ] `pytest tests/test_api_guards.py -q` passes
- [ ] determinism spot-check (build twice, same sha256) on a few
- [ ] ruff check + ruff format --check on changed files
- Do NOT run check_staged_drift or network/gallery tests (404 until upload).

## Notes
- base must equal today's loader output modulo stable sort. Gallery test pins pki (3072,5857).
- Disk: home/cache mount ~37G free, 5.1G used. Clean cache between heavy datasets.
- No push, no PR, no aws. Report ends with `claude done`.
