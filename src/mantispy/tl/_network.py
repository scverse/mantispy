from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from mantispy._core.frames import as_frame
from mantispy._core.logging import get_logger
from mantispy._core.mutation import inplace_or_copy


@inplace_or_copy(expects=("well", "perturbation"))
def network_enrichment(
    adata: AnnData,
    similarity_key: str = "similarity",
    edges: pd.DataFrame | None = None,
    gene_key: str = "Metadata_Gene",
    top_quantile: float = 0.95,
    key_added: str = "network_enrichment",
    copy: bool = False,
) -> AnnData | None:
    """Test the most-similar perturbation pairs for enrichment of known interactions.

    Args:
        adata: Object whose ``obsp[similarity_key]`` holds a pairwise similarity, from :func:`~mantispy.tl.similarity`, and whose ``obs[gene_key]`` names each profile's gene.
        similarity_key: ``obsp`` key holding the pairwise similarity matrix.
        edges: A two-column frame of reference gene pairs.
            Its first two columns are read as the pair, in any order.
            Defaults to :func:`mantispy.ds.interactions` (CORUM within-complex pairs).
        gene_key: ``obs`` column holding the gene symbol.
        top_quantile: Quantile of the off-diagonal similarity above which a pair counts as a top pair.
            The default 0.95 takes the top 5%.
        key_added: Name for the output.
        copy: Return a modified copy instead of mutating in place.

    Returns:
        ``None``, or the modified copy.
        Writes ``uns["mantispy"][key_added]`` with the 2x2 contingency ``table`` (top vs not, known vs not), ``odds_ratio`` and ``pvalue`` (a one-sided Fisher exact test), the ``threshold`` similarity, and the pair counts ``n_top``, ``n_known`` and ``n_pairs``.

    Raises:
        KeyError: ``obsp`` has no ``similarity_key``, or ``obs`` has no ``gene_key``.
        ValueError: ``top_quantile`` is outside (0, 1), or the object has fewer than two annotated profiles.

    Notes:
        Only pairs whose two profiles both carry a gene are counted, so control wells with no gene are left out of the universe.
        The default reference needs the network on its first call to build the pinned CORUM snapshot; pass ``edges`` to avoid any fetch, or to test against a real protein-protein network.
    """
    from scipy.stats import fisher_exact

    if not 0.0 < top_quantile < 1.0:
        raise ValueError(f"top_quantile must be in (0, 1), got {top_quantile!r}")
    if similarity_key not in adata.obsp:
        raise KeyError(f"obsp has no {similarity_key!r}; run mt.tl.similarity first, which writes it")
    obs = as_frame(adata.obs)
    if gene_key not in obs:
        raise KeyError(f"obs has no column {gene_key!r} holding the gene symbol")

    if edges is None:
        from mantispy.ds._resources import interactions

        edges = interactions()
    edge_columns = list(edges.columns)[:2]
    reference = {tuple(sorted(pair)) for pair in edges[edge_columns].astype(str).to_numpy()}

    matrix = np.asarray(adata.obsp[similarity_key])
    upper = np.triu_indices(matrix.shape[0], k=1)
    genes = obs[gene_key].astype(str).to_numpy()
    # notna, not the str cast, catches NaN, None and pd.NA.
    present = obs[gene_key].notna().to_numpy() & (genes != "")
    gene_a, gene_b = genes[upper[0]], genes[upper[1]]

    valid = present[upper[0]] & present[upper[1]]
    n_pairs = int(valid.sum())
    if n_pairs < 1:
        raise ValueError("fewer than two annotated profiles to pair; check gene_key")

    similarity = matrix[upper][valid].astype(np.float64, copy=False)
    gene_a, gene_b = gene_a[valid], gene_b[valid]
    threshold = float(np.quantile(similarity, top_quantile))
    is_top = similarity >= threshold
    is_known = np.fromiter(
        (((a, b) if a <= b else (b, a)) in reference for a, b in zip(gene_a, gene_b, strict=True)),
        dtype=bool,
        count=len(gene_a),
    )
    n_top = int(is_top.sum())

    a = int(np.sum(is_top & is_known))
    b = int(np.sum(is_top & ~is_known))
    c = int(np.sum(~is_top & is_known))
    d = int(np.sum(~is_top & ~is_known))
    odds_ratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")

    adata.uns.setdefault("mantispy", {})[key_added] = {
        "table": [[a, b], [c, d]],
        "odds_ratio": float(odds_ratio),
        "pvalue": float(pvalue),
        "threshold": threshold,
        "n_top": n_top,
        "n_known": int(is_known.sum()),
        "n_pairs": n_pairs,
    }
    get_logger().info(
        "network_enrichment: %d known of %d top pair(s), odds ratio %.2f, p=%.2g",
        a,
        n_top,
        odds_ratio,
        pvalue,
    )
    return None
