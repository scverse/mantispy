"""Parity of mt.tl.perturbation_score with Brieflow's perturbation score.

The reference is ``calculate_perturbation_scores`` from Brieflow v1.5.1,
``workflow/lib/aggregate/perturbation_score.py``, copied below. Brieflow draws an equal-sized control sample per
gene and scales it on that sample's controls; with as many controls as gene cells the sample is every control,
and on features already standardized on them the scaling is the identity, so both see the same rows and splits.
"""

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline

import mantispy as mt


def _brieflow_scores(
    cell_data, gene, feature_cols, perturbation_col, n_differential_features=200, minimum_cell_count=100
):
    """Brieflow v1.5.1 ``calculate_perturbation_scores``, unchanged apart from formatting."""
    if cell_data.shape[0] < minimum_cell_count:
        return pd.Series(np.nan, index=cell_data.index), np.nan
    y = cell_data[perturbation_col].eq(gene).fillna(False).astype(bool).astype(int).to_numpy()
    X_all = cell_data[feature_cols].to_numpy()
    k = min(n_differential_features, X_all.shape[1])
    n_splits = 5 if cell_data.shape[0] < minimum_cell_count * 2 else 10
    clf = make_pipeline(
        SelectKBest(score_func=f_classif, k=k),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    scores = cross_val_predict(clf, X_all, y, cv=cv, method="predict_proba")[:, 1]
    return pd.Series(scores, index=cell_data.index), roc_auc_score(y, scores)


def test_scores_and_auc_match_brieflow_on_the_same_sample():
    rng = np.random.default_rng(1)
    n, p = 150, 250  # more features than Brieflow keeps, so the in-fold selection matters
    x = rng.normal(0, 1, (2 * n, p))
    labels = rng.permutation(np.repeat(["nontargeting", "GENE"], n))
    x[labels == "GENE", :10] += rng.normal(0.6, 0.3, ((labels == "GENE").sum(), 1))
    control = labels == "nontargeting"
    x = (x - x[control].mean(0)) / x[control].std(0)
    obs = pd.DataFrame(
        {"Metadata_Gene": labels, "Metadata_Control_Type": np.where(control, "negcon", "treatment")},
        index=[f"c{i}" for i in range(2 * n)],
    )
    adata = AnnData(x.astype(np.float64), obs=obs)

    frame = pd.DataFrame(x, columns=[f"f{j}" for j in range(p)]).assign(gene=labels)
    expected, expected_auc = _brieflow_scores(frame, "GENE", [f"f{j}" for j in range(p)], "gene")

    mt.tl.perturbation_score(adata, n_folds=10)
    gene = labels == "GENE"
    np.testing.assert_allclose(adata.obs["perturbation_score"].to_numpy()[gene], expected.to_numpy()[gene], atol=1e-10)
    table = adata.uns["mantispy"]["perturbation_score"].set_index("group")
    np.testing.assert_allclose(table.loc["GENE", "auc"], expected_auc, atol=1e-12)
