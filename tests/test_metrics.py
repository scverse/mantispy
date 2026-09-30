"""Correction metrics: PC-regression is native and pinned against a plain-numpy reference; the batch-mixing rows come from scib-metrics, so the panel tests skip when it is absent and one test pins the graceful degradation."""

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

import mantispy as mt
from mantispy.ds import synthetic_plate


@pytest.fixture(scope="module")
def corrected():
    cells = synthetic_plate(
        n_plates=4,
        n_wells=96,
        n_cells=6,
        n_features=20,
        n_batches=2,
        batch_effect=4.0,
        n_perturbations=4,
        effect_size=3.0,
        seed=0,
    )
    wells = mt.tl.aggregate(cells, min_cells=0)
    sc.pp.pca(wells, n_comps=10)
    return wells


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


def test_evaluate_correction_compares_representations_and_names_the_map_row_honestly(corrected):
    """Reading one uns table once per representation gave every representation the same mAP, 0.7757 under both X_pca and X_other, inside the function whose purpose is comparing them."""
    pytest.importorskip("copairs")  # copairs declares requires-python <3.13
    corrected = corrected.copy()  # module-scoped fixture; this test writes obsm["X_other"] and tl.map writes uns
    corrected.obsm["X_other"] = np.asarray(corrected.obsm["X_pca"])[:, :5]
    mt.tl.map(corrected, mode="activity", null_size=200)  # scores X, neither representation

    frame = mt.metrics.evaluate_correction(corrected, reps=("X_pca", "X_other"), map_key="map")
    assert {"X_pca", "X_other"} <= set(frame["representation"])

    rows = frame[frame["metric"] == "mean_average_precision"]
    assert len(rows) == 1
    assert rows["representation"].iloc[0] == "X"


def test_evaluate_correction_survives_an_object_where_most_metrics_are_undefined():
    """A consensus object holds one row per perturbation, where the label silhouette, batch mixing and both LISIs are undefined at once, and the table still has to come back with whatever can be measured."""
    pytest.importorskip("scib_metrics")  # the four undefined rows this asserts on come from the scib panel
    import anndata as ad

    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "Metadata_Perturbation": ["a", "b", "c", "d"],
            "Metadata_Batch": ["B1", "B2", "B1", "B2"],
        },
        index=[str(index) for index in range(4)],
    )
    adata = ad.AnnData(X=rng.normal(size=(4, 5)).astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = rng.normal(size=(4, 3))

    with pytest.warns(UserWarning):
        frame = mt.metrics.evaluate_correction(adata)

    values = frame.set_index("metric")["value"]
    assert values[["silhouette_label", "silhouette_batch", "ilisi", "clisi"]].isna().all()
    assert np.isfinite(values["pc_regression"])


def test_evaluate_correction_returns_nan_for_pc_regression_on_a_single_batch():
    """scib's PCR raises on a constant covariate; a single-batch object must yield NaN, not abort the panel."""
    import anndata as ad

    rng = np.random.default_rng(0)
    obs = pd.DataFrame(
        {
            "Metadata_Perturbation": [f"p{index % 4}" for index in range(40)],
            "Metadata_Batch": ["only"] * 40,  # one batch: the covariate PC-regression scores is constant
        },
        index=[str(index) for index in range(40)],
    )
    adata = ad.AnnData(X=rng.normal(size=(40, 6)).astype(np.float32), obs=obs)
    adata.obsm["X_pca"] = rng.normal(size=(40, 4))

    with pytest.warns(UserWarning, match="PC-regression"):
        frame = mt.metrics.evaluate_correction(adata, perplexity=10)

    assert np.isnan(_value(frame, "pc_regression"))


def _batch_split(n=180, seed=0):
    """One object with two embeddings: batches mixed within labels, and batches shifted apart."""
    import anndata as ad

    rng = np.random.default_rng(seed)
    # Coprime counts so every label spans every batch; otherwise the batch silhouette is undefined.
    n_labels, n_batches = 5, 3
    labels = np.array([f"p{index % n_labels}" for index in range(n)])
    batches = np.array([f"b{index % n_batches}" for index in range(n)])
    signal = np.eye(8)[[int(label[1:]) for label in labels]] * 3.0  # labels separable in both reps
    base = rng.normal(size=(n, 8)) + signal
    shift = np.zeros((n, 8))
    shift[:, 7] = np.array([int(batch[1:]) for batch in batches]) * 8.0  # a strong batch axis

    obs = pd.DataFrame(
        {"Metadata_Perturbation": labels, "Metadata_Batch": batches},
        index=[str(index) for index in range(n)],
    )
    adata = ad.AnnData(X=base.astype(np.float32), obs=obs)
    adata.obsm["X_mixed"] = base
    adata.obsm["X_separated"] = base + shift
    return adata


def test_evaluate_correction_moves_the_batch_metrics_when_batches_separate():
    """A batch-separated representation scores worse on every batch metric than a batch-mixed one.

    This pins the delegated scib-metrics panel through the one public entry: the tidy schema,
    the metric rows and the directions the metrics move.
    """
    pytest.importorskip("scib_metrics")
    adata = _batch_split()

    frame = mt.metrics.evaluate_correction(
        adata,
        reps=("X_mixed", "X_separated"),
        label_key="Metadata_Perturbation",
        batch_key="Metadata_Batch",
        perplexity=10,
    )

    assert list(frame.columns) == ["metric", "representation", "key", "value", "better"]
    per_rep = frame.set_index(["metric", "representation"])["value"]
    for metric in ("silhouette_label", "silhouette_batch", "ilisi", "clisi", "pc_regression"):
        assert np.isfinite(per_rep[(metric, "X_mixed")])
        assert np.isfinite(per_rep[(metric, "X_separated")])

    # Mixing is better: higher iLISI and batch silhouette, lower PC-regression on the batch.
    assert per_rep[("ilisi", "X_mixed")] > per_rep[("ilisi", "X_separated")]
    assert per_rep[("silhouette_batch", "X_mixed")] > per_rep[("silhouette_batch", "X_separated")]
    assert per_rep[("pc_regression", "X_mixed")] < per_rep[("pc_regression", "X_separated")]


def test_evaluate_correction_degrades_to_native_rows_without_scib_metrics(corrected, monkeypatch):
    """Without scib-metrics the one public entry drops to the native PC-regression rows and warns once, naming what it left out, rather than raising a scib traceback."""
    import builtins

    real_import = builtins.__import__

    def _no_scib(name, *args, **kwargs):
        if name == "scib_metrics" or name.startswith("scib_metrics."):
            raise ImportError("simulated missing scib-metrics")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_scib)

    with pytest.warns(UserWarning) as record:
        frame = mt.metrics.evaluate_correction(corrected, reps=("X_pca",), perplexity=10)

    # Only the native rows survive; the batch-mixing rows are gone.
    assert set(frame["metric"]) == {"pc_regression"}
    assert list(frame.columns) == ["metric", "representation", "key", "value", "better"]
    assert np.isfinite(frame["value"]).all()

    # One message names the omitted metrics and the extra that adds them.
    messages = [str(warning.message) for warning in record]
    assert any("iLISI" in message and "mantispy[integration]" in message for message in messages)


def test_missing_representation_says_how_to_make_one(corrected):
    with pytest.raises(KeyError, match="sc.pp.pca"):
        mt.metrics.evaluate_correction(corrected, reps=("X_nope",), perplexity=10)


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


def test_evaluate_correction_reports_a_covariate_nothing_else_would_catch(corrected):
    """A representation can be dominated by something that is neither the batch nor the label.

    On the learned embeddings of `ds.jump_lite` the cell count explains several times more of the variance than the source does, and no other row of this table would say so.
    """
    generator = np.random.default_rng(0)
    embedding = np.asarray(corrected.obsm["X_pca"]).copy()
    corrected.obs["Metadata_CellCount"] = embedding[:, 0] * 10 + generator.normal(scale=0.01, size=corrected.n_obs)
    corrected.obs["Metadata_Unrelated"] = generator.normal(size=corrected.n_obs)

    frame = mt.metrics.evaluate_correction(
        corrected, reps=("X_pca",), covariates=("Metadata_CellCount", "Metadata_Unrelated"), perplexity=10
    )

    assert frame["metric"].is_unique  # a second row called "pc_regression" would collide
    dominant = _value(frame, "pc_regression:Metadata_CellCount")
    unrelated = _value(frame, "pc_regression:Metadata_Unrelated")
    batch = _value(frame, "pc_regression")
    assert dominant > batch and dominant > 0.5
    assert unrelated < 0.2

    # Whether a small share is better depends on what the covariate is, so no direction is claimed.
    covariate_rows = frame[frame["metric"].str.startswith("pc_regression:")]
    assert covariate_rows["better"].isna().all()
    assert not frame[~frame["metric"].str.startswith("pc_regression:")]["better"].isna().any()
