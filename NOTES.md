# mantispy implementation diary

**Purpose:** this file is the durable source of truth across Claude Code compaction.
Read it first in any new session. It records what is done, what is next, and every
binding decision that overrides the plan documents.

**Goal:** implement 0.1 → 0.2 → 0.3 in one continuous run. No releases in between.
A refactor pass for redundancy/elegance follows each of 0.1, 0.2, 0.3.
Target: usable at an upcoming hackathon, so the API must be elegant and the hot paths fast.

**Plans:** `docs/superpowers/plans/2026-09-07-mantispy-0.{1,2,3}.0-*.md`
The plans are the task breakdown. **Where this file disagrees with a plan, this file wins.**

---

## Status

| stage | state |
|---|---|
| 0.1 Foundation | not started |
| 0.1 refactor | not started |
| 0.2 Trustworthy profiles | not started |
| 0.2 refactor | not started |
| 0.3 Batch correction + evaluation | not started |
| 0.3 refactor | not started |

**Next action:** 0.1 Task 1 (repo scaffold), then Task 2 (`_core/plate.py`).

### Task ledger — 0.1

- [ ] 1 scaffold (pyproject, ruff, CI)
- [ ] 2 `_core/plate.py`
- [ ] 3 `_core/features.py` + blocklist
- [ ] 4 `_core/schema.py` + `spec/schema-0.1.json`
- [ ] 5 `_core/_utils.py`
- [ ] 6 `datasets/_synthetic.py`
- [ ] 7 `_core/_numba.py`
- [ ] 8 `_core/_chunks.py`
- [ ] 9 `io/_cellprofiler.py`
- [ ] 10 `io/_profiles.py`
- [ ] 11 `pp/_qc.py`
- [ ] 12 `pp/_normalize.py`
- [ ] 13 `tl/_aggregate.py`
- [ ] 14 `get/`
- [ ] 15 `pl/`
- [ ] 16 pycytominer equivalence
- [ ] 17 BBBC021 loader
- [ ] 18 tutorials 01–02
- [ ] 19 (skipped — no releases)

---

## Verified environment facts

Checked empirically in a scratch venv, not from memory:

| package | version | fact |
|---|---|---|
| scanpy | 1.12.4 | `sc.get.aggregate` supports mean/median/sum/var/count_nonzero, multi-key `by`, `mask=`, returns `obs["n_obs_aggregated"]`. **Propagates NaN.** 200k×1000×384 groups: mean 0.20 s, median 3.28 s (pandas 5.61 s) |
| scanpy | 1.12.4 | `sc.pp.scale(mask_obs=)` fits on a subset — but global only, no `by=`, and NaN-poisoning |
| scanpy | 1.12.4 | `sce.pp.harmony_integrate(adata, key, basis, adjusted_basis)` already wraps harmonypy |
| anndata | 0.13.3 | object-dtype all-NaN columns fail to write to h5ad |
| pycytominer | 1.7.1 | `standardize` = sklearn StandardScaler, **ddof=0** |
| pycytominer | 1.7.1 | `mad_robustize_epsilon` default `1e-18` |
| pycytominer | 1.7.1 | `variance_threshold` = sklearn VarianceThreshold (min_variance=1e-6). freq_cut/unique_cut is a **separate op named `frequency_threshold`** |
| pycytominer | 1.7.1 | `correlation_threshold` uses **signed** correlation and is **non-greedy** (set union over independently judged pairs). corr = −1.0 keeps both columns |
| pyod | 3.6.5 | ECOD is `sum over dims of elementwise max(U_l, U_r, U_skew)` — **sum(max), not max(sum)**. Verified: sum(max) diff 0.0, max(sum) diff 8.56 |

---

## Binding corrections to the plans

Numbered so commits and future sessions can cite them. All came from the max-effort
plan review (task `aa9d10e8`); C1/C4/C6/C7/C8 were independently re-verified above.

### Correctness — these break gates or delete user data

- **C1 `normalize` scale.** `standardize` must use **ddof=0** (pycytominer uses
  StandardScaler). Also: `standardize` and `robustize` currently divide by an
  unguarded scale — clamp a zero scale to `1.0` (sklearn's behaviour) so constant
  features yield `0.0`, not `inf`/`NaN`. `epsilon` stays on the `mad_robustize`
  branch only, default `1e-18`.
- **C2 `qc_pass` must not require zero NaN.** `qc_n_nan_features == 0` fails for
  essentially every real cell (partial NaN in Zernike/RadialDistribution is routine);
  combined with `filter_cells(qc_pass=True)` by default this silently deletes the whole
  dataset. Use a *fraction* threshold: `qc_nan_fraction <= max_nan_fraction` (default
  0.5), and make `filter_cells` log how many cells it dropped.
- **C3 `qc_area_outlier` +inf.** `np.nan_to_num(z, nan=0.0)` needs `posinf=0.0`;
  a zero Area MAD otherwise flags every cell. Same bug class: audit every
  `nan_to_num` call for `posinf`/`neginf` (also `pp/_image_qc.py` knn branch).
- **C4 ECOD is `sum(max)`.** Compute `O = maximum(U_l, U_r, U_skew)` elementwise, then
  `O.sum(axis=1)`. Handle `skew == 0` as pyod does (`U_l + U_r`), not by falling into
  the right-tail branch.
- **C5 parent/child join.** The parent link normally lives on the *primary* table
  (`Cells_Parent_Nuclei`), not the child. Look for the link in both directions; if
  neither exists, **raise** rather than silently merging on `ObjectNumber`. Also detect
  parents with *zero* children (group over the parent frame, not the child frame).
- **C6 `feature_select` op names.** Rename our freq/unique op to `frequency_threshold`
  and add a real `variance_threshold` (sklearn-style `min_variance`, default `1e-6`).
  Combine freq/unique with **OR** and strict `<`, matching
  `set(excluded_freq + excluded_unique)`.
- **C7 `correlation_threshold` is non-greedy and signed.** Compute the mean-correlation
  sum once from the full matrix, take the **signed** lower-triangle pairs above the
  threshold, and union the per-pair verdicts. Drop the "removal order is part of the
  contract" constraint from the 0.2 plan — pycytominer has no greedy pass.
- **C8 `sphere` regulariser.** pycytominer's `Spherize` works on **singular** values:
  pad null directions with the smallest nonzero singular value when `n <= d`, then
  `Sigma + epsilon`, `Lambda^(-1/2) = inv(S) * sqrt(n-1)`. Adding epsilon to a clipped
  eigenvalue diverges by 10×–3300×. The `-cor` branch must use ddof=0. Fixture needs
  enough control wells to be full rank (raise `n_wells` or lower `n_features`).
- **C9 `regress_out` must not fabricate data.** Do not `nan_to_num` then write back over
  the NaN positions. Fit per feature on the rows where that feature is observed, and
  write `NaN` back where it was `NaN`.
- **C10 `correct_plate_position` duplicate index assignment.** `grid[rows, cols] = ...`
  keeps only the last write when several cells share a well. Aggregate to the well grid
  (nanmedian) before the polish, at any resolution.
- **C11 `group_codes` multi-column path.** Drop the blanket `.astype(str)`: it turns NaN
  into a group literally named `"nan"` (defeating the `codes < 0` guard) and stringifies
  numeric metadata so `0.4` and `0.40` become distinct groups. Use
  `MultiIndex.from_frame` on the raw columns and check for NaN explicitly.
- **C12 anndata serialisation.** Two sites make `ms.io.write` raise:
  (a) object-dtype all-NaN `var` columns from the parser — use `pd.NA` in a
  `string`/`category` dtype, or drop empty columns before write;
  (b) `well_qc` storing a MultiIndex-indexed `DataFrame` in `uns` — flatten to columns.
  A round-trip test on `synthetic_plate()` after every `uns`/`var` writer is mandatory.
- **C13 controls have no producer.** `reference="negcon"` is the *default* of `sphere`
  and `grit`, and `Metadata_Perturbation` is the default of `feature_select`'s
  `noise_removal` and `tl.map`'s activity mode — but nothing in 0.1–0.3 creates either
  column. Pull `annotate_controls(adata, negcon=("DMSO",), perturbation_key=...)`
  forward from 0.4 into **0.1**, and add a `controls=` / `perturbation_key=` argument to
  `read_cellprofiler`. Also: `_reference_mask` doing `.to_numpy(dtype=bool)` on a string
  column silently yields all-True — validate the dtype.
- **C14 image QC has no plate column.** `read_cellprofiler` must carry
  `Metadata_Plate`/`Metadata_Well` into `uns["mantispy"]["image_table"]`, and `image_qc`
  must **raise** when `by=` is missing from the table instead of silently pooling plates.

### Performance — the 0.1 hot path

- **C15 `normalize` temporaries.** `((X - center[codes]) / scale[codes])` broadcasts
  float64 statistics into ~4 full-size float64 arrays (128 GB transient at the 1M×4000
  benchmark). Cast `center`/`scale` to float32 and fill a preallocated `out` in row
  chunks.
- **C16 numba kernel allocations.** `_dispatch` permutes and upcasts the whole matrix
  (`np.ascontiguousarray(X[order], dtype=np.float64)`) on *every* grouped statistic —
  twice for `mad_robustize`, four times for `robustize`. Pass `order` into the kernel
  instead of materialising the permutation; keep float32 accumulation with a float64
  accumulator only where it matters. Hoist the per-iteration `buf = np.empty(max_len)`
  out of `prange` (one buffer per thread) — 1.5M NRT allocations serialise the loop.
- **C17 the streaming seam is dead code.** All ~18 grouped reductions call
  `get_matrix(adata)` (materialising everything); none call `iter_groups`. The guard
  test only greps `adata.X[` under `pp/`, so it cannot fail. Make the seam
  **reduction-shaped** (`grouped_reduce`/`grouped_apply`) rather than matrix-shaped,
  and write it that way *before* 15 call sites exist. This is the single most important
  architectural decision for the 0.5 backed mode — get it right in 0.1.

### API / design — for the refactor passes

- **C18 drop `ms.pp.harmony`.** `sce.pp.harmony_integrate` already exists. Document it
  in tutorial 05 instead of wrapping a wrapper. Consider exposing `sc.pp.combat` as a
  second option (free, relevant).
- **C19 `tl.aggregate` provenance.** It copies only `uns["mantispy"]["channels"]`
  forward, destroying `truth` / `image_table` / `params`. Carry the whole
  `uns["mantispy"]` dict minus keys that are resolution-specific.
- **C20 `@records_params` decorator.** ~15 functions repeat the same
  copy → work → `record_params(...)` prologue/epilogue with hand-retyped parameter
  dicts (~120 lines of drift-prone restatement). One decorator that captures the bound
  arguments removes all of it. Do this in the 0.1 refactor, before the count grows.
- **C21 test hygiene.** 26 near-identical fixtures are pasted across test files while
  `conftest.py` holds none; ~17 of 25 plot tests are `isinstance(ax, Axes)` smoke tests
  that re-run whole pipelines to check a return type. Centralise fixtures in
  `conftest.py`; keep **one** parametrized smoke test for the plot namespace plus real
  assertions for the two or three plots with actual logic (`pl.plate` grid shape,
  `pl.map` threshold line). The user asked for *sufficient but not too many* tests —
  hold that line from the start rather than pruning later.
- **C22 misc small fixes.** `tl.map`'s `reference` parameter is accepted but never used
  — remove it. `evaluate_correction` emits two rows both `metric == "lisi"`, so
  `pl.metrics`' pivot raises — name them `ilisi`/`clisi`. `outliers(method="mad")`
  ignores `contamination` and hard-codes `z > 5` while its own test asserts a nonzero
  flag count. `validate()` requires `Metadata_Plate`/`Well`, so perturbation-resolution
  objects can never be written — make the requirement resolution-dependent.
  `ax.boxplot(tick_labels=)` needs matplotlib ≥ 3.9 (floor says 3.8). `pl.qc`
  `log10(qc_variance)` crashes on an all-NaN column and tutorial 02 calls it before
  `filter_features`. `_wells` uses `chr(65 + r)`, emitting garbage past row 26 for
  1536-well plates. The parser keeps only `numeric[0]`, so `Zernike_2_0` and
  `Zernike_2_2` collide under `standardize_feature_names`. `Nuclei_ObjectNumber` is
  classified `is_feature=True`. `Location_Center_X/_Y` are dropped from both `X` and
  `obs`, so `qc_is_border` can never fire on the reader's own output.
  The BBBC021 pooch registry is empty — Task 17 cannot run as written.
- **C23 four tests assert impossible values.** `test_filter_cells` exactly-5,
  `test_reference_negcon` pooled-across-plates median, `test_row_gradient` corr > 0.5
  against an actual ~0.47, and the scib `pcr` call without `n_comps` against a 10-PC
  fixture. Fix the assertions when reaching them.

---

## Standing decisions

- Reference libraries (pycytominer, pyod, scib, scmorph) are **test-only**; a guard test
  forbids importing them from `src/`.
- `keep_raw=False` by default — a `raw` layer doubles memory and the raw table is on disk.
- Resolution (`uns["mantispy"]["resolution"]`) is **advisory**: warn, never raise.
  Well level is the default and what the first tutorial shows.
- `tl.aggregate` delegates to `sc.get.aggregate` when the matrix is NaN-free, falls back
  to our NaN-skipping kernel otherwise. Both paths covered by the same equivalence test.
- No `ms.pp.pca` / `neighbors` / `umap` wrappers — scanpy calls, guarded by a test.
- scanpy floor `>=1.12` (the version `median` support was verified on).
- No releases, no tags, until all three stages are done.

---

## Session log

### 2026-09-07 — planning
- Analysed `mantispy_release_plan.md`; wrote three implementation plans (0.1/0.2/0.3).
- Verified scanpy reuse empirically; corrected the 0.1 plan's `sc.get.aggregate` claim
  (median *is* supported) and moved the scanpy floor to 1.12.
- Ran a max-effort review of the plans; 15 findings + a long tail, recorded above as
  C1–C23. Independently re-verified C1, C4, C6, C7, C8.
- Nothing implemented yet.
