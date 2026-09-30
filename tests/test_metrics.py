"""Correction metrics: PC-regression is native and pinned against a plain-numpy reference; evaluate_integration triages on what is installed, so its four cases are forced by blocking the imports."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # the integration heatmap draws a figure

import mantispy as mt


def _value(frame, metric):
    return float(frame.loc[frame["metric"] == metric, "value"].iloc[0])


def _pc_regression_pair(seed=0):
    """An embedding with a categorical and a numeric covariate, needing no optional dependency."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    n = 150
    values = rng.normal(size=(n, 6))
    obs = pd.DataFrame(
        {
            "cat": pd.Categorical([f"g{index % 3}" for index in range(n)]),
            "num": values[:, 0] * 0.5 + rng.normal(size=n),
        },
        index=[str(index) for index in range(n)],
    )
    adata = ad.AnnData(X=rng.normal(size=(n, 3)).astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = values
    return adata, values


def test_pc_regression_matches_a_plain_numpy_reference():
    """The numba kernels agree with a plain-numpy variance-weighted R^2 to 1e-6 for both a categorical and a numeric covariate."""
    from mantispy.metrics._common import r_squared

    adata, values = _pc_regression_pair()

    def reference(key):
        covariate = adata.obs[key]
        variances = values.var(axis=0, ddof=1)
        weights = variances / variances.sum()
        explained = np.array([r_squared(values[:, index], covariate) for index in range(values.shape[1])])
        return float(np.sum(weights * explained))

    for key in ("cat", "num"):
        got = float(mt.metrics.pc_regression(adata, key)["value"].iloc[0])
        assert got == pytest.approx(reference(key), abs=1e-6)


def test_pc_regression_is_one_when_the_only_component_is_the_covariate():
    """A single component that is exactly the covariate explains all of the variance, categorical or numeric."""
    import anndata as ad

    n = 60
    rng = np.random.default_rng(1)
    codes = np.array([index % 3 for index in range(n)])
    obs = pd.DataFrame(
        {
            "cat": pd.Categorical([f"g{code}" for code in codes]),
            "num": codes.astype(float) + rng.normal(scale=1e-6, size=n),
        },
        index=[str(index) for index in range(n)],
    )
    adata = ad.AnnData(X=rng.normal(size=(n, 2)).astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = codes.reshape(-1, 1).astype(float)

    assert mt.metrics.pc_regression(adata, "cat")["value"].iloc[0] == pytest.approx(1.0, abs=1e-9)
    assert mt.metrics.pc_regression(adata, "num")["value"].iloc[0] == pytest.approx(1.0, abs=1e-6)


def test_pc_regression_returns_nan_for_a_constant_covariate():
    """A single-value covariate has no variance to regress against, so its share is NaN with a warning, not a crash."""
    import anndata as ad

    rng = np.random.default_rng(2)
    obs = pd.DataFrame({"batch": ["only"] * 40}, index=[str(index) for index in range(40)])
    adata = ad.AnnData(X=rng.normal(size=(40, 3)).astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = rng.normal(size=(40, 4))

    with pytest.warns(UserWarning, match="PC-regression"):
        value = mt.metrics.pc_regression(adata, "batch")["value"].iloc[0]
    assert np.isnan(value)


def test_batch_variance_explained_covers_every_key():
    """One PC-regression row per key, stacked into a single tidy frame."""
    adata, _ = _pc_regression_pair()

    frame = mt.metrics.batch_variance_explained(adata, keys=["cat", "num"])
    assert set(frame["key"]) == {"cat", "num"}
    assert (frame["metric"] == "pc_regression").all()
    assert len(frame) == 2


def _scib_ready(n=120, seed=0):
    """A small object scib-metrics can benchmark: labels separable, two batches, an X_pca baseline."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    # Coprime counts so every label spans both batches; otherwise scib's batch silhouette is undefined.
    n_labels, n_batches = 3, 2
    labels = np.array([f"p{index % n_labels}" for index in range(n)])
    batches = np.array([f"b{index % n_batches}" for index in range(n)])
    signal = np.eye(6)[[int(label[1:]) for label in labels]] * 3.0  # labels separable, so bio metrics are defined
    emb = rng.normal(size=(n, 6)) + signal
    obs = pd.DataFrame(
        {"Metadata_Perturbation": labels, "Metadata_Batch": batches},
        index=[str(index) for index in range(n)],
    )
    adata = ad.AnnData(X=emb.astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = emb  # also the pre-integrated baseline the Benchmarker defaults to
    return adata


def _block_imports(monkeypatch, *blocked):
    """Make `import <name>` raise ImportError for each blocked top-level module, to force a triage branch."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] in blocked:
            raise ImportError(f"simulated missing {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_evaluate_integration_needs_at_least_one_engine(monkeypatch):
    """With neither scib-metrics nor copairs there is nothing to run, so it raises and names the extra to install."""
    _block_imports(monkeypatch, "scib_metrics", "copairs")
    with pytest.raises(ImportError, match=r"mantispy\[integration,map\]"):
        mt.metrics.evaluate_integration(_scib_ready())


def test_evaluate_integration_refuses_one_row_per_perturbation():
    """A consensus object has no replicate pairs to score, so it fails with a message naming the cause, not a traceback."""
    import anndata as ad

    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {"Metadata_Perturbation": [f"p{index}" for index in range(12)], "Metadata_Batch": ["b0", "b1"] * 6},
        index=[str(index) for index in range(12)],
    )
    adata = ad.AnnData(X=rng.normal(size=(12, 4)).astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = rng.normal(size=(12, 4))

    with pytest.raises(ValueError, match="one row per perturbation"):
        mt.metrics.evaluate_integration(adata)


def test_evaluate_integration_copairs_only_returns_a_per_rep_map_frame(monkeypatch):
    """Without scib-metrics it drops to a native mAP frame, one row per representation, and warns once."""
    pytest.importorskip("copairs")  # copairs declares requires-python <3.13
    adata = _scib_ready()
    adata.obsm["X_other"] = np.asarray(adata.obsm["X_pca"])[:, :4]
    _block_imports(monkeypatch, "scib_metrics")

    with pytest.warns(UserWarning, match="scib-metrics"):
        frame = mt.metrics.evaluate_integration(adata, reps=("X_pca", "X_other"))

    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["mean_average_precision"]  # the only metric column
    assert "pc_regression" not in frame.columns  # no native PC-regression fallback here
    assert list(frame.index) == ["X_pca", "X_other"]  # one value per rep
    assert np.isfinite(frame["mean_average_precision"]).all()


def test_evaluate_integration_scib_only_returns_a_frame_without_a_map_block(monkeypatch):
    """With scib-metrics but no copairs it returns the numeric frame with the aggregates and no mAP column."""
    pytest.importorskip("scib_metrics")

    adata = _scib_ready()
    _block_imports(monkeypatch, "copairs")

    frame = mt.metrics.evaluate_integration(adata, reps=("X_pca",))
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.index) == ["X_pca"]
    assert "mean_average_precision" not in frame.columns  # no retrieval block without copairs
    assert "Total+mAP" not in frame.columns  # and no mantispy total either
    assert {"Batch correction", "Bio conservation", "Total"} <= set(frame.columns)
    assert np.isfinite(frame["Total"]).all()


def test_evaluate_integration_both_returns_the_map_column_and_both_totals():
    """With both installed the frame carries the metric columns, the mAP column and scib's Total beside mantispy's Total+mAP."""
    pytest.importorskip("scib_metrics")
    pytest.importorskip("copairs")

    adata = _scib_ready()
    adata.obsm["X_other"] = np.asarray(adata.obsm["X_pca"])[:, :4]

    frame = mt.metrics.evaluate_integration(adata, reps=("X_pca", "X_other"))
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.index) == ["X_pca", "X_other"]
    assert "mean_average_precision" in frame.columns
    assert {"Total", "Total+mAP", "Batch correction", "Bio conservation"} <= set(frame.columns)
    assert np.isfinite(frame[["Total", "Total+mAP", "mean_average_precision"]].to_numpy()).all()


def test_map_is_one_for_separable_labels_and_drops_when_shuffled():
    """Ground truth: tight, well-separated label clusters retrieve perfectly, and shuffling the labels drops mAP to chance."""
    pytest.importorskip("copairs")
    import anndata as ad

    from mantispy.metrics._evaluate import _map_settings, _rep_map

    rng = np.random.default_rng(0)
    n_labels, per, dim = 5, 6, 8
    directions = rng.normal(size=(n_labels, dim)) * 5.0
    rows, labels, batches = [], [], []
    for label in range(n_labels):
        for replicate in range(per):
            rows.append(directions[label] + 0.01 * rng.normal(size=dim))
            labels.append(f"p{label}")
            batches.append(f"b{replicate % 2}")
    obs = pd.DataFrame(
        {"Metadata_Perturbation": labels, "Metadata_Batch": batches},
        index=[str(index) for index in range(len(rows))],
    )
    adata = ad.AnnData(np.asarray(rows, dtype=np.float32), obs=obs)
    adata.obsm["X_pca"] = np.asarray(rows)

    settings = _map_settings("replicability", "Metadata_Perturbation", "Metadata_Batch", None)
    separable = _rep_map(adata, "X_pca", settings)
    assert separable > 0.95

    shuffled = adata.copy()
    shuffled.obs["Metadata_Perturbation"] = rng.permutation(shuffled.obs["Metadata_Perturbation"].to_numpy())
    chance = _rep_map(shuffled, "X_pca", settings)
    assert chance < 0.6
    assert chance < separable


def test_scib_get_results_keeps_the_public_seam_we_read():
    """A scib-metrics upgrade that moved the Metric Type row or the aggregate columns would break the reader; fail here first."""
    pytest.importorskip("scib_metrics")
    from mantispy.metrics import _evaluate

    adata = _scib_ready()
    results = _evaluate._benchmark(
        adata,
        reps=("X_pca",),
        label_key="Metadata_Perturbation",
        batch_key="Metadata_Batch",
        min_max_scale=False,
    )

    assert "Metric Type" in results.index
    assert {"Batch correction", "Bio conservation", "Total"} <= set(results.columns)
    aggregate = results.loc["Metric Type", ["Batch correction", "Bio conservation", "Total"]]
    assert (aggregate == "Aggregate score").all()


def _gene_map(n_sets=4, per_set=3, n_background=24, n_features=16, noise=0.1, seed=0):
    """One profile per gene, where the genes of a set share a direction and the rest are noise."""
    import anndata as ad

    generator = np.random.default_rng(seed)
    names, rows, edges = [], [], []
    for index in range(n_sets):
        direction = generator.normal(size=n_features)
        for member in range(per_set):
            names.append(f"SET{index}G{member}")
            rows.append(direction + noise * generator.normal(size=n_features))
            edges.append({"source": f"complex{index}", "target": names[-1]})
    for index in range(n_background):
        names.append(f"BG{index}")
        rows.append(generator.normal(size=n_features))

    obs = pd.DataFrame({"Metadata_Perturbation": names}, index=pd.Index(names, name="gene"))
    return ad.AnnData(np.asarray(rows, dtype=np.float32), obs=obs), pd.DataFrame(edges)


def test_known_relationships_measures_chance_for_a_perturbation_in_many_sets():
    """A compound annotated to many targets draws many of the pairs.

    This one points away from every other profile, so its pairs sit in a tail whatever the annotation says: the recall is far above 2 x percentile, and only a shuffle that keeps how many sets each perturbation belongs to shows that it is chance.
    """
    import anndata as ad

    values = 3.0 + np.random.default_rng(0).normal(size=(60, 16))
    values[0] *= -1
    names = [f"G{index}" for index in range(60)]
    adata = ad.AnnData(values.astype(np.float32), obs=pd.DataFrame({"Metadata_Perturbation": names}, index=names))
    hub = [(f"hub{k}", member) for k in range(1, 20) for member in ("G0", f"G{k}")]
    rest = [(f"pair{k}", f"G{member}") for k in range(20) for member in (20 + 2 * k, 21 + 2 * k)]
    net = pd.DataFrame(hub + rest, columns=["source", "target"])

    result = mt.metrics.known_relationships(adata, net, n_permutations=100, seed=0).iloc[0]

    assert result["value"] > 0.3
    assert result["null"] > 0.3
    assert result["p_value"] > 0.05


def test_known_relationships_refuses_input_it_cannot_score():
    adata, net = _gene_map(seed=4)
    with pytest.raises(ValueError, match="aggregate first"):
        mt.metrics.known_relationships(adata[[0, 0, 1]].copy(), net)
    with pytest.raises(ValueError, match="relates no two"):
        mt.metrics.known_relationships(adata, net.assign(target="ABSENT" + net["target"]))
    with pytest.raises(KeyError, match="Metadata_Missing"):
        mt.metrics.known_relationships(adata, net, label_key="Metadata_Missing")
    with pytest.raises(ValueError, match=r"must be in \(0, 50\)"):
        mt.metrics.known_relationships(adata, net, percentile=150)


def test_known_relationships_names_each_annotation_source():
    """Every source is scored on its own, so stacking two unnamed rows gave a table that pl.metrics could not pivot: 'Index contains duplicate entries'."""
    adata, net = _gene_map(seed=7)
    rows = pd.concat(
        [
            mt.metrics.known_relationships(adata, net, name="corum"),
            mt.metrics.known_relationships(adata, net, name="reactome"),
        ]
    )

    assert list(rows["metric"]) == ["known_relationships:corum", "known_relationships:reactome"]
    mt.pl.metrics(rows)


def test_known_relationships_caps_what_one_set_expands_into(monkeypatch):
    """A set of n members is n(n-1)/2 pairs, so one set naming every gene in a genome would both dominate the recall and exhaust memory.

    The message has to name the way out.
    """
    adata, _ = _gene_map(seed=6)
    genes = adata.obs["Metadata_Perturbation"].to_numpy()
    everything = pd.DataFrame({"source": "all", "target": genes})

    monkeypatch.setattr(mt.metrics._relationships, "MAX_PAIRS", 5)
    with pytest.raises(ValueError, match="drop the largest ones"):
        mt.metrics.known_relationships(adata, everything)


def test_known_relationships_takes_a_pair_list_once_it_is_reshaped():
    """The reference relationship sets ship one pair per row.

    There is one annotation shape, so the conversion is the caller's, and it has to give the same answer as the sets do.
    """
    adata, _ = _gene_map(seed=5)
    genes = adata.obs["Metadata_Perturbation"].to_numpy()
    # Written both ways round, so the direction a pair appears in cannot change the answer.
    pairs = pd.DataFrame(
        [
            {"entity1": genes[3 * index + a], "entity2": genes[3 * index + b]}
            for index in range(4)
            for a, b in ((0, 1), (1, 0))
        ]
    )
    reshaped = pairs.assign(source=pairs.index.astype(str)).melt(id_vars="source", value_name="target")[
        ["source", "target"]
    ]
    sets = pd.DataFrame(
        [{"source": f"complex{index}", "target": genes[3 * index + member]} for index in range(4) for member in (0, 1)]
    )

    assert _value(mt.metrics.known_relationships(adata, reshaped), "known_relationships") == _value(
        mt.metrics.known_relationships(adata, sets), "known_relationships"
    )

    with pytest.raises(ValueError, match="net needs"):
        mt.metrics.known_relationships(adata, pairs)


def _carried_pair(seed=0):
    """An embedding and a named block on the same wells: three features are linear in the embedding, three are pure noise."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    n_obs, k = 60, 4
    names = [f"W{index}" for index in range(n_obs)]
    emb = rng.normal(size=(n_obs, k))

    adata = ad.AnnData(X=rng.normal(size=(n_obs, 3)).astype(np.float32), obs=pd.DataFrame(index=names))
    adata.obsm["X_emb"] = emb

    signal = emb @ rng.normal(size=(k, 3)) + 0.01 * rng.normal(size=(n_obs, 3))
    noise = rng.normal(size=(n_obs, 3))
    var = pd.DataFrame({"feature_group": ["signal"] * 3 + ["noise"] * 3}, index=[f"F{index}" for index in range(6)])
    reference = ad.AnnData(
        X=np.hstack([signal, noise]).astype(np.float32),
        obs=pd.DataFrame(index=names),
        var=var,
    )
    return adata, reference


def test_variance_carried_returns_one_row_per_feature_or_per_group():
    adata, reference = _carried_pair()

    per_feature = mt.metrics.variance_carried(adata, reference, use_rep="X_emb", groupby=None)
    assert list(per_feature.columns) == ["feature", "variance_carried"]
    assert len(per_feature) == reference.n_vars

    per_group = mt.metrics.variance_carried(adata, reference, use_rep="X_emb", groupby="feature_group")
    assert len(per_group) == 2
    assert "n_features" in per_group.columns


def test_variance_carried_refuses_input_it_cannot_score():
    adata, reference = _carried_pair()

    disjoint = reference.copy()
    disjoint.obs_names = [f"other{index}" for index in range(disjoint.n_obs)]
    with pytest.raises(ValueError, match="obs_names"):
        mt.metrics.variance_carried(adata, disjoint, use_rep="X_emb")

    with pytest.raises(ValueError, match="not a column"):
        mt.metrics.variance_carried(adata, reference, use_rep="X_emb", groupby="nope")


def test_variance_carried_rejects_non_unique_obs_names():
    """Duplicate obs_names raise an actionable error rather than an opaque pandas one."""
    adata, reference = _carried_pair()

    dup_reference = reference.copy()
    dup_reference.obs_names = ["W0"] * dup_reference.n_obs
    with pytest.raises(ValueError, match="not unique"):
        mt.metrics.variance_carried(adata, dup_reference, use_rep="X_emb")

    dup_adata = adata.copy()
    dup_adata.obs_names = ["W0"] * dup_adata.n_obs
    with pytest.raises(ValueError, match="not unique"):
        mt.metrics.variance_carried(dup_adata, reference, use_rep="X_emb")


def test_variance_carried_all_nan_when_fewer_shared_than_splits():
    """Fewer shared wells than n_splits yields all-NaN, not an opaque sklearn crash (issue #128)."""
    adata, reference = _carried_pair()
    few = adata[:3].copy()  # 3 shared wells, default n_splits=5

    frame = mt.metrics.variance_carried(few, reference, use_rep="X_emb", groupby=None)
    assert frame["variance_carried"].isna().all()


def test_variance_carried_survives_an_inf_target():
    """An inf in one target column drops only that row rather than aborting the run: neighbours still score (issue #128)."""
    adata, reference = _carried_pair()
    reference = reference.copy()
    block = np.asarray(reference.X).copy()
    block[0, 1] = np.inf
    reference.X = block.astype(np.float32)

    frame = mt.metrics.variance_carried(adata, reference, use_rep="X_emb", groupby=None)
    carried = frame.set_index("feature")["variance_carried"]
    assert carried["F0"] > 0.7
