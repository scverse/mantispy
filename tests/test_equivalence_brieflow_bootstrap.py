"""Parity of tl.bootstrap_features with Brieflow's per-gene bootstrap.

The reference is copied from Brieflow v1.5.1, ``workflow/lib/aggregate/bootstrap.py``: ``run_gene_bootstrap`` and the
helpers it calls, with only the imports and the degenerate-block check left out.
Both draw from numba's random state seeded with the gene's CRC32, so on the same seed the p-values agree exactly.
"""

import zlib

import anndata as ad
import numba
import numpy as np
import pandas as pd

import mantispy as mt
from mantispy._core.schema import stamp

BOOTSTRAP_BLOCK_SIMS = 1000


# --- Brieflow v1.5.1, workflow/lib/aggregate/bootstrap.py -------------------------------------------------------
def run_gene_bootstrap(construct_observed, construct_controls, sample_sizes, gene_observed, num_sims, seed):
    construct_observed = np.asarray(construct_observed, dtype=np.float64)
    gene_observed = np.asarray(gene_observed, dtype=np.float64)
    n_constructs, n_features = construct_observed.shape
    pools = [_control_pool_arrays(controls) for controls in construct_controls]
    block_size = min(num_sims, BOOTSTRAP_BLOCK_SIMS)
    nulls = np.empty((n_constructs, block_size, n_features), dtype=np.float64)
    construct_exceed = np.zeros((n_constructs, n_features), dtype=np.int64)
    gene_exceed = np.zeros(n_features, dtype=np.int64)

    _seed_numba(seed)
    done = 0
    while done < num_sims:
        n = min(block_size, num_sims - done)
        for c, (features_t, pool_lookup, pool_starts, pool_ends) in enumerate(pools):
            block = nulls[c, :n]
            _bootstrap_block_numba(features_t, pool_lookup, pool_starts, pool_ends, sample_sizes[c], block)
            construct_exceed[c] += (block > construct_observed[c]).sum(axis=0)
        gene_nulls = np.median(nulls[:, :n], axis=0)
        gene_exceed += (gene_nulls > gene_observed).sum(axis=0)
        done += n

    return (
        pvals_from_exceed_counts(construct_exceed, num_sims),
        pvals_from_exceed_counts(gene_exceed, num_sims),
    )


def pvals_from_exceed_counts(exceed_counts, num_sims):
    pvals_one_tail = np.asarray(exceed_counts, dtype=np.float64) / num_sims
    return np.where(pvals_one_tail > 0.5, 1 - pvals_one_tail, pvals_one_tail) * 2


def bootstrap_seed(gene_id):
    return zlib.crc32(str(gene_id).encode("utf-8"))


def _control_pool_arrays(controls_arr):
    control_ids = controls_arr[:, 0]
    controls_features = controls_arr[:, 1:].astype(np.float64)
    sort_order = np.argsort(control_ids, kind="stable")
    features_t = np.ascontiguousarray(controls_features[sort_order].T)
    _, counts = np.unique(control_ids[sort_order], return_counts=True)
    pool_ends = np.cumsum(counts).astype(np.int64)
    pool_starts = np.empty_like(pool_ends)
    pool_starts[0] = 0
    pool_starts[1:] = pool_ends[:-1]
    pool_lookup = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    return features_t, pool_lookup, pool_starts, pool_ends


@numba.njit
def _seed_numba(seed):
    np.random.seed(seed)


@numba.njit
def _bootstrap_block_numba(features_t, pool_lookup, pool_starts, pool_ends, sample_size, out):
    n_features, n_rows = features_t.shape
    rows = np.empty(sample_size, dtype=np.int64)
    buf = np.empty(sample_size, dtype=np.float64)
    for i in range(out.shape[0]):
        seed_row = np.random.randint(0, n_rows)
        pool_idx = pool_lookup[seed_row]
        start = pool_starts[pool_idx]
        pool_size = pool_ends[pool_idx] - start
        for k in range(sample_size):
            rows[k] = start + np.random.randint(0, pool_size)
        for j in range(n_features):
            has_nan = False
            for k in range(sample_size):
                value = features_t[j, rows[k]]
                buf[k] = value
                if np.isnan(value):
                    has_nan = True
            out[i, j] = np.nan if has_nan else _median_inplace(buf)


@numba.njit
def _median_inplace(buf):
    n = buf.shape[0]
    k = n // 2
    lo = 0
    hi = n - 1
    while lo < hi:
        pivot = buf[k]
        i = lo
        j = hi
        while i <= j:
            while buf[i] < pivot:
                i += 1
            while pivot < buf[j]:
                j -= 1
            if i <= j:
                tmp = buf[i]
                buf[i] = buf[j]
                buf[j] = tmp
                i += 1
                j -= 1
        if j < k:
            lo = i
        if k < i:
            hi = j
    upper = buf[k]
    if n % 2 == 1:
        return upper
    lower = buf[0]
    for t in range(1, k):
        if buf[t] > lower:
            lower = buf[t]
    return (lower + upper) / 2


# --- end of the Brieflow copy -----------------------------------------------------------------------------------


def _screen(seed: int = 0) -> ad.AnnData:
    """Five genes of two to four guides with uneven cell counts, some shifted, and six control guides."""
    rng = np.random.default_rng(seed)
    n_features = 6
    blocks, genes, guides = [], [], []
    for g in range(5):
        shift = rng.normal(scale=0.4, size=n_features) * (g % 2)
        for s in range(2 + g % 3):
            n = int(rng.integers(5, 40))
            blocks.append(rng.normal(size=(n, n_features)) + shift + rng.normal(scale=0.1, size=n_features))
            genes += [f"G{g}"] * n
            guides += [f"G{g}_{s}"] * n
    for s in range(6):
        n = int(rng.integers(20, 80))
        blocks.append(rng.normal(size=(n, n_features)) + rng.normal(scale=0.1, size=n_features))
        genes += ["nontargeting"] * n
        guides += [f"nt_{s}"] * n
    order = rng.permutation(len(genes))  # cells of a guide are not contiguous in a real object
    values = np.concatenate(blocks)[order]
    gene, guide = np.asarray(genes)[order], np.asarray(guides)[order]
    obs = pd.DataFrame(
        {
            "Metadata_Gene": gene,
            "Metadata_sgRNA": guide,
            "Metadata_Control_Type": np.where(gene == "nontargeting", "negcon", "treatment"),
        },
        index=[str(i) for i in range(gene.size)],
    )
    adata = ad.AnnData(values, obs=obs)
    adata.var_names = [f"Cells_Intensity_F{i}" for i in range(n_features)]
    stamp(adata, resolution="object")
    return adata


def test_pvalues_match_brieflow_on_the_same_seed():
    """Every gene and guide p-value equals Brieflow's, drawn on the same seeds over two blocks."""
    adata = _screen()
    n_sims = 2500
    mt.tl.bootstrap_features(adata, n_sims=n_sims)
    ours = adata.uns["mantispy"]["bootstrap"].pivot(index="group", columns="feature", values="pvalue")
    our_guides = adata.uns["mantispy"]["bootstrap_guides"].pivot(index="replicate", columns="feature", values="pvalue")

    values = adata.X
    obs = adata.obs
    is_control = (obs["Metadata_Control_Type"] == "negcon").to_numpy()
    controls = np.column_stack([obs["Metadata_sgRNA"].to_numpy()[is_control].astype(object), values[is_control]])
    features = list(adata.var_names)
    for gene in sorted(set(obs["Metadata_Gene"][~is_control])):
        in_gene = (obs["Metadata_Gene"] == gene).to_numpy()
        constructs = sorted(set(obs["Metadata_sgRNA"][in_gene]))
        cells = [values[(obs["Metadata_sgRNA"] == c).to_numpy()] for c in constructs]
        observed = np.array([np.median(c, axis=0) for c in cells])
        construct_p, gene_p = run_gene_bootstrap(
            observed,
            [controls] * len(constructs),
            [len(c) for c in cells],
            np.median(observed, axis=0),
            n_sims,
            bootstrap_seed(gene),
        )
        floor = 1 / n_sims
        np.testing.assert_array_equal(ours.loc[gene, features].to_numpy(), np.maximum(gene_p, floor))
        np.testing.assert_array_equal(our_guides.loc[constructs, features].to_numpy(), np.maximum(construct_p, floor))
