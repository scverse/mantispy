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
- [ ] neuropainting  (base only)            -- validates the whole pipeline
- [ ] chroma         (base only)
- [ ] pooled_rare    (base, selected)
- [ ] oasis_pilot    (base, agg; keep annotate= raw path)
- [ ] pki            (base, selected, agg, agg_selected)
- [ ] scallops_arv471(base, agg; cells->guide via tl.aggregate)
- [ ] jump_crispr    (base, agg; guides->gene consensus; keep annotate= raw path)
- [ ] jump_cells     (base, selected, agg; HEAVY; keep annotate=/selected=)
- [ ] cp_posh        (base, selected, agg, agg_selected; HEAVY)
- [ ] jump_target2   (default base; keep plates= subset; parameterized) -- attempt/flag
- [ ] jump_lite      (per-model keyed finals) -- evaluate; SKIP if thin per-model read

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
