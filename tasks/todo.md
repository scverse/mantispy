# Interpretation layer (single PR)

Branch `feat/interpretation-layer`. Reuse decoupler/scipy/scanpy only; add no dependency.

## Plan

- [ ] 1. `tl.cluster` (`tl/_cluster.py`): hierarchical clustering of per-perturbation profiles.
  - scipy `linkage(pdist(values, metric))` + `fcluster`; auto-cut by silhouette when both
    `distance_cut` and `n_clusters` are None; optional `method="leiden"` via scanpy.
  - obs[key_added] categorical labels; uns["mantispy"]["cluster_linkage"] = Z; uns["mantispy"]["cluster"] meta.
  - Tests: planted groups recover labels, linkage shape, auto-cut chosen.
- [ ] 2. `ds` resources (`ds/_resources.py`): cached wrappers over decoupler omnipath.
  - `gene_sets(name)` -> cached `dc.op.resource`/shortcuts (GO_BP, CORUM, Reactome, hallmark).
  - `interactions(source="CORUM")` -> within-complex gene-pair edges, cached parquet under cache_dir.
  - Tests: small cached fixture (no network); live fetch behind @pytest.mark.network + slow.
- [ ] 3. `tl.ora` (`tl/_ora.py`): thin wrapper over `dc.mt.ora`.
  - Per group in groupby, member genes vs net, universe = measured genes (n_up = n_bg - k).
  - tidy frame (group, source, n, odds_ratio, pvalue, qvalue) in uns.
  - Tests: synthetic net + planted membership -> seeded set is top hit.
- [ ] 4. `tl.network_enrichment` (`tl/_network.py`): fisher_exact of top similar pairs vs edges.
  - threshold top_quantile of off-diagonal similarity; overlap with reference edges; 2x2 + OR + p in uns.
  - Tests: planted edges significant; null not significant.
- [ ] 5. `pl.dendrogram` (`pl/_cluster.py`): scipy dendrogram of stored linkage on an Axes.
  - Interactive twin deferred (figure_factory recomputes from data, would diverge from stored Z).
- [ ] 6. Wire exports (tl/ds/pl __init__), docs/api.md autosummary + Stores rows.
- [ ] 7. Run `pytest -k "cluster or ora or network or resources or dendrogram"` green.
- [ ] 8. Push, open DRAFT PR.

## Notes / decisions
- decoupler ORA `n_up` is inverted: to select the top k features pass `n_up = n_bg - k` (see `tl/_enrich.py::_ora_n_up`). Verified against scipy `fisher_exact` (two-tailed).
- decoupler ORA returns BH-adjusted p across a row's sources -> used as `qvalue`; raw `pvalue` from scipy `fisher_exact` on the same 2x2 (universe = measured genes). odds_ratio = decoupler es (HA-corrected log OR).
- Never duck-type classes. No em-dashes in prose. Conventional commits, no attribution lines. Commit per component.

## Review

All five components implemented and wired, commits per component:
- `tl.cluster` (`tl/_cluster.py`): hierarchical + auto silhouette cut + optional leiden. 6 tests.
- `ds.gene_sets`/`ds.interactions` (`ds/_resources.py`): cached omnipath wrappers. 3 offline + 1 network test.
- `tl.ora` (`tl/_ora.py`): per-group ORA over `dc.mt.ora`, universe = measured genes. 4 tests.
- `tl.network_enrichment` (`tl/_network.py`): one-sided Fisher of top pairs vs edges. 3 tests.
- `pl.dendrogram` (`pl/_cluster.py`): static scipy dendrogram; interactive twin deferred. 2 tests.

Checks: `pytest -k "cluster or ora or network or resources or dendrogram"` -> 32 passed.
ruff clean, mypy clean on the five new files (remaining mypy notes are pre-existing env stub gaps in
untouched files). No new dependency added. Docs (`docs/api.md`) updated with autosummary + Stores rows.

Design note: decoupler ORA returns BH-adjusted p (per group, across sets) and uses inverted `n_up`
(`n_bg - k`), so `tl.ora` recomputes the raw two-tailed Fisher p from the same 2x2 and applies one BH
across the whole table, keeping pvalue/qvalue consistent. `interactions("CORUM")` reuses the packaged,
pinned `ds.corum()` complexes, so the default reference is offline-reproducible.
