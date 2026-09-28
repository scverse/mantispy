"""Cached prior-knowledge resources.

``omnipath`` the package is not a dependency: everything here goes through decoupler, which talks to the web service directly.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from mantispy._core.logging import get_logger
from mantispy._settings import settings
from mantispy.ds._datasets import corum

#: Values are the ``collection`` labels the OmniPath ``MSigDB`` resource files each set under.
_MSIGDB_COLLECTIONS = {"GO_BP": "go_biological_process", "Reactome": "reactome_pathways"}


def _resources_dir(cache_dir: str | Path | None) -> Path:
    """The directory the pinned resource snapshots live in, created if missing."""
    directory = Path(cache_dir or settings.cache_dir) / "resources"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _cached(cache_key: str, builder, cache_dir: str | Path | None) -> pd.DataFrame:
    """Read a pinned resource snapshot, building and writing it on the first call."""
    path = _resources_dir(cache_dir) / f"{cache_key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frame = builder()
    # An empty resource is always a bad name or a failed fetch, and caching it would pin that for every later call.
    if frame.empty:
        raise ValueError(f"resource {cache_key!r} came back empty; check the name, and do not pin the result")
    frame.to_parquet(path, index=False)
    return frame


def _normalize_net(net: pd.DataFrame) -> pd.DataFrame:
    """A decoupler resource frame reduced to ``source``, ``target`` and ``weight``.

    Resources come back under different column names: gene-set nets already use ``source``/``target``, while an MSigDB-style frame uses ``geneset``/``genesymbol``.
    Anything else is refused with the columns it did carry.
    """
    lowered = net.rename(columns={column: str(column).lower() for column in net.columns})
    if {"source", "target"} <= set(lowered.columns):
        out = lowered[["source", "target"]]
    elif {"geneset", "genesymbol"} <= set(lowered.columns):
        out = lowered.rename(columns={"geneset": "source", "genesymbol": "target"})[["source", "target"]]
    else:
        raise ValueError(
            f"resource has columns {sorted(lowered.columns)}, not a source/target gene-set net; "
            "use a dedicated shortcut (hallmark, GO_BP, Reactome, CORUM) or another resource."
        )
    out = out.astype(str).drop_duplicates().reset_index(drop=True)
    out["weight"] = 1.0
    return out


def gene_sets(name: str = "hallmark", organism: str = "human", cache_dir: str | Path | None = None) -> pd.DataFrame:
    """A gene-set network from OmniPath, pinned to a local snapshot.

    Args:
        name: A friendly shortcut (``"hallmark"``, ``"GO_BP"``, ``"Reactome"``, ``"CORUM"``) or any OmniPath resource name that ``decoupler.op.show_resources`` lists (for example ``"MSigDB"``, ``"KEGG"``).
        organism: The organism the resource is fetched for.
            ``"CORUM"`` is human only.
        cache_dir: Where the snapshot is kept.
            Defaults to :attr:`mantispy.settings.cache_dir`.

    Returns:
        A frame with ``source`` (the set), ``target`` (a gene symbol) and ``weight`` (1.0), in the shape :func:`~mantispy.tl.ora` and :func:`~mantispy.tl.enrich` read.
        For ``"CORUM"`` the set is a complex.

    Notes:
        The first call fetches from the OmniPath web service (or, for ``"CORUM"``, downloads the packaged complexes) and writes a parquet snapshot; later calls read the snapshot, so CI and offline use never refetch.
        ``"GO_BP"`` and ``"Reactome"`` are collections of the large ``MSigDB`` resource.
    """

    def build() -> pd.DataFrame:
        import decoupler as dc

        if name == "hallmark":
            net = _normalize_net(dc.op.hallmark(organism=organism))
        elif name == "CORUM":
            net = corum(cache_dir).astype(str)
            net["weight"] = 1.0
        elif name in _MSIGDB_COLLECTIONS:
            collection = _MSIGDB_COLLECTIONS[name]
            msigdb = dc.op.resource("MSigDB", organism=organism)
            net = _normalize_net(msigdb[msigdb["collection"].astype(str) == collection])
        else:
            net = _normalize_net(dc.op.resource(name, organism=organism))
        get_logger().info("gene_sets(%s): %d set(s) over %d edge(s)", name, net["source"].nunique(), len(net))
        return net

    return _cached(f"gene_sets_{name}_{organism}", build, cache_dir)


def interactions(source: str = "CORUM", organism: str = "human", cache_dir: str | Path | None = None) -> pd.DataFrame:
    """Within-complex gene pairs from a complex resource, as an undirected edge list.

    Every pair of genes in the same complex becomes one edge, which is the reference :func:`~mantispy.tl.network_enrichment` tests the most-similar perturbation pairs against.

    Args:
        source: A complex resource :func:`gene_sets` can return as ``source`` (complex) and ``target`` (gene).
            ``"CORUM"`` (the default) uses the packaged CORUM complexes.
        organism: The organism the resource is fetched for.
        cache_dir: Where the snapshot is kept.
            Defaults to :attr:`mantispy.settings.cache_dir`.

    Returns:
        A frame with ``gene_a`` and ``gene_b`` (``gene_a < gene_b``), one row per unordered pair of genes that share a complex, deduplicated across complexes.

    Notes:
        The pinned snapshot means the default reference of :func:`~mantispy.tl.network_enrichment` is reproducible and needs no network after the first call.
        For a real protein-protein interaction network, pass a two-column edge frame as ``edges`` instead.
    """

    def build() -> pd.DataFrame:
        complexes = gene_sets(source, organism=organism, cache_dir=cache_dir)
        edges: set[tuple[str, str]] = set()
        for _, block in complexes.groupby("source", observed=True):
            members = sorted(set(block["target"].astype(str)))
            edges.update(combinations(members, 2))
        frame = pd.DataFrame(sorted(edges), columns=["gene_a", "gene_b"])
        get_logger().info(
            "interactions(%s): %d gene pair(s) from %d complex(es)", source, len(frame), complexes["source"].nunique()
        )
        return frame

    return _cached(f"interactions_{source}_{organism}", build, cache_dir)
