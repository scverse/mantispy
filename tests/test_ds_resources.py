"""Cached gene-set and interaction resources."""

import pandas as pd
import pytest

import mantispy as mt
from mantispy.ds._resources import _resources_dir


@pytest.fixture
def offline_cache(tmp_path):
    """A cache_dir with a small pinned gene-set net, so nothing here touches the network."""
    with mt.settings.override(cache_dir=tmp_path):
        net = pd.DataFrame(
            {
                "source": ["C1", "C1", "C1", "C2", "C2"],
                "target": ["A", "B", "C", "D", "E"],
                "weight": 1.0,
            }
        )
        net.to_parquet(_resources_dir(tmp_path) / "gene_sets_toy_human.parquet", index=False)
        yield tmp_path


def test_gene_sets_reads_the_pinned_snapshot(offline_cache):
    with mt.settings.override(cache_dir=offline_cache):
        net = mt.ds.gene_sets("toy")
    assert list(net.columns) == ["source", "target", "weight"]
    assert set(net["source"]) == {"C1", "C2"}
    assert set(net["target"]) == {"A", "B", "C", "D", "E"}


def test_interactions_expands_within_complex_pairs(offline_cache):
    with mt.settings.override(cache_dir=offline_cache):
        edges = mt.ds.interactions("toy")
    pairs = {tuple(row) for row in edges[["gene_a", "gene_b"]].to_numpy()}
    assert pairs == {("A", "B"), ("A", "C"), ("B", "C"), ("D", "E")}
    assert (edges["gene_a"] < edges["gene_b"]).all()


def test_interactions_writes_and_reuses_its_own_snapshot(offline_cache):
    with mt.settings.override(cache_dir=offline_cache):
        mt.ds.interactions("toy")
        # Second call reads the pinned snapshot, so removing the source net does not break it.
        (_resources_dir(offline_cache) / "gene_sets_toy_human.parquet").unlink()
        edges = mt.ds.interactions("toy")
    assert len(edges) == 4


@pytest.mark.network
@pytest.mark.slow
def test_gene_sets_hallmark_fetches_live(tmp_path):
    with mt.settings.override(cache_dir=tmp_path):
        net = mt.ds.gene_sets("hallmark")
    assert {"source", "target", "weight"} <= set(net.columns)
    assert net["source"].nunique() == 50  # MSigDB hallmark has 50 programs
