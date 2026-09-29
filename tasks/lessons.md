# Lessons

- When staging a dataset's base (host + fetch), the base must equal the *live* loader output modulo the
  stable obs/var sort. Verify the shape against the actual assembly (`_augmented(...).n_vars`), not a
  docstring, audit, or an existing `@pytest.mark.network` shape assertion: pki's gallery test pinned
  (3072, 5857) but the pinned data yields 5839, so the test was stale. Trust the running code.
- Loaders that assemble via `_files`/`_profiles`/`_plate_files` with no `select` will pick up the new
  rehosted `.h5ad` rows once they are added to the registry entry. Every `_assemble_<name>` must filter them
  out (`select=lambda name: not name.endswith(".h5ad")`), and the fetch path must select the exact h5ad name.
- For a loader with an `annotate=`/`plates=`/`model=` parameter, host the default object and keep the
  non-default path reading the raw inputs unchanged; that keeps the public API stable without hosting every
  parameterization. Guard illegal combinations (e.g. `aggregated` needs `annotate=True`) before any fetch.
