"""Tests for the data_loader module."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from dexposure_agent.data_loader import (
    SnapshotLoader,
    parse_date_range,
    raw_to_graph_snapshot,
    _extract_node_features,
)
from dexposure_agent.types import GraphSnapshot


# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_raw_snapshot(n_nodes: int = 5, n_links: int = 8) -> dict:
    """Build a minimal raw JSON snapshot dict."""
    nodes = []
    for i in range(n_nodes):
        comp = {f"token_{j}": 100.0 / (j + 1) for j in range(3)}
        nodes.append({"id": str(i), "size": float((i + 1) * 1000), "composition": comp})
    # Add an empty-size node to test filtering
    nodes.append({"id": "empty", "size": 0.0, "composition": {}})

    links = []
    for i in range(min(n_links, n_nodes * (n_nodes - 1))):
        src = str(i % n_nodes)
        tgt = str((i + 1) % n_nodes)
        if src != tgt:
            links.append({"source": src, "target": tgt, "size": float(i + 1) * 10})
    return {"nodes": nodes, "links": links}


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temp directory with mock JSON and meta CSV."""
    # 6 weekly snapshots spanning Jan-Feb 2025
    dates = [
        "2025-01-06", "2025-01-13", "2025-01-20", "2025-01-27",
        "2025-02-03", "2025-02-10",
    ]
    data = {}
    for date in dates:
        data[date] = _make_raw_snapshot(n_nodes=5, n_links=8)

    json_path = tmp_path / "historical-network_week_2025-01-01.json"
    json_path.write_text(json.dumps({"data": data}))

    meta_path = tmp_path / "meta_df.csv"
    meta_path.write_text("id,name,category\n0,Proto0,lending\n1,Proto1,dex\n2,Proto2,staking\n")

    return tmp_path


@pytest.fixture
def loader(tmp_data_dir: Path) -> SnapshotLoader:
    meta_path = tmp_data_dir / "meta_df.csv"
    return SnapshotLoader(
        data_dir=str(tmp_data_dir),
        meta_path=str(meta_path),
    )


# ── parse_date_range ─────────────────────────────────────────────────────


class TestParseDateRange:
    def test_basic(self):
        start, end = parse_date_range("2025-01~2025-08")
        assert start.year == 2025 and start.month == 1 and start.day == 1
        assert end.year == 2025 and end.month == 8 and end.day == 31

    def test_december_boundary(self):
        start, end = parse_date_range("2024-11~2024-12")
        assert end.month == 12 and end.day == 31

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="Expected"):
            parse_date_range("2025-01")

    def test_february(self):
        _, end = parse_date_range("2024-01~2024-02")
        assert end.month == 2 and end.day == 29  # 2024 is a leap year


# ── raw_to_graph_snapshot ────────────────────────────────────────────────


class TestRawToGraphSnapshot:
    def test_basic_conversion(self):
        raw = _make_raw_snapshot(n_nodes=3, n_links=4)
        snap = raw_to_graph_snapshot("2025-01-01", raw, meta_category={})
        # 3 non-empty + 1 empty node, all should be included (min_node_size=0)
        assert len(snap.nodes) == 4
        assert snap.date == "2025-01-01"
        assert len(snap.edges) > 0

    def test_min_node_size_filter(self):
        raw = _make_raw_snapshot(n_nodes=3, n_links=4)
        snap = raw_to_graph_snapshot("2025-01-01", raw, {}, min_node_size=500)
        # Node "0" has size=1000, "1"=2000, "2"=3000 -> all pass
        # "empty" has size=0 -> filtered out
        assert "empty" not in snap.nodes
        assert len(snap.nodes) == 3

    def test_edges_only_between_included_nodes(self):
        raw = _make_raw_snapshot(n_nodes=3, n_links=4)
        snap = raw_to_graph_snapshot("2025-01-01", raw, {}, min_node_size=2500)
        # Only node "2" (size=3000) survives
        for edge in snap.edges:
            assert edge.source in snap.nodes
            assert edge.target in snap.nodes

    def test_zero_weight_edges_excluded(self):
        raw = {"nodes": [{"id": "a", "size": 100}, {"id": "b", "size": 200}],
               "links": [{"source": "a", "target": "b", "size": 0.0}]}
        snap = raw_to_graph_snapshot("2025-01-01", raw, {})
        assert len(snap.edges) == 0

    def test_metadata_categories(self):
        raw = _make_raw_snapshot(n_nodes=2, n_links=1)
        meta = {"0": "lending", "1": "dex"}
        snap = raw_to_graph_snapshot("2025-01-01", raw, meta)
        assert snap.nodes["0"].category == "lending"
        assert snap.nodes["1"].category == "dex"

    def test_missing_metadata_defaults_to_unknown(self):
        raw = _make_raw_snapshot(n_nodes=2, n_links=1)
        snap = raw_to_graph_snapshot("2025-01-01", raw, meta_category={})
        for nf in snap.nodes.values():
            assert nf.category == "Unknown"


# ── _extract_node_features ───────────────────────────────────────────────


class TestExtractNodeFeatures:
    def test_normal_node(self):
        raw_node = {"id": "42", "size": 1000.0, "composition": {"ETH": 700, "USDC": 300}}
        nf = _extract_node_features(raw_node, {"42": "lending"})
        assert nf.log_size == pytest.approx(math.log1p(1000.0))
        assert nf.num_tokens == 2
        assert nf.max_share == pytest.approx(0.7, abs=0.01)
        assert nf.entropy > 0
        assert nf.category == "lending"

    def test_empty_node(self):
        raw_node = {"id": "x", "size": 0.0, "composition": {}}
        nf = _extract_node_features(raw_node, {})
        assert nf.log_size == pytest.approx(0.0)
        assert nf.num_tokens == 0
        assert nf.max_share == 0.0
        assert nf.entropy == 0.0
        assert nf.category == "Unknown"


# ── SnapshotLoader ───────────────────────────────────────────────────────


class TestSnapshotLoader:
    def test_dates_property(self, loader: SnapshotLoader):
        dates = loader.dates
        assert len(dates) == 6
        assert dates == sorted(dates)

    def test_load_all(self, loader: SnapshotLoader):
        snaps = loader.load()
        assert len(snaps) == 6
        assert all(isinstance(s, GraphSnapshot) for s in snaps)

    def test_load_with_date_range(self, loader: SnapshotLoader):
        snaps = loader.load(date_range="2025-01~2025-01")
        # Dates: 01-06, 01-13, 01-20, 01-27 are in January
        assert all(s.date.startswith("2025-01") for s in snaps)
        assert len(snaps) == 4

    def test_load_with_explicit_start_end(self, loader: SnapshotLoader):
        snaps = loader.load(start="2025-01-13", end="2025-01-27")
        dates = [s.date for s in snaps]
        assert "2025-01-06" not in dates
        assert "2025-01-13" in dates
        assert "2025-01-27" in dates

    def test_load_single(self, loader: SnapshotLoader):
        snap = loader.load_single("2025-01-06")
        assert snap.date == "2025-01-06"
        assert len(snap.nodes) > 0

    def test_load_single_not_found(self, loader: SnapshotLoader):
        with pytest.raises(KeyError):
            loader.load_single("2099-01-01")

    def test_build_baseline(self, loader: SnapshotLoader):
        # Baseline before 2025-02 with window=3 -> last 3 Jan snapshots
        baseline = loader.build_baseline(before="2025-02", window=3)
        assert len(baseline) == 3
        assert all(isinstance(m, dict) for m in baseline)
        # Each dict should have N1..N5
        for m in baseline:
            assert set(m.keys()) == {"N1", "N2", "N3", "N4", "N5"}

    def test_build_baseline_empty(self, loader: SnapshotLoader):
        baseline = loader.build_baseline(before="2020-01", window=26)
        assert baseline == []

    def test_iter_test_with_baselines(self, loader: SnapshotLoader):
        pairs = list(loader.iter_test_with_baselines("2025-02~2025-02", baseline_window=4))
        # February dates in our fixture: 2025-02-03, 2025-02-10
        assert len(pairs) == 2
        for snap, baseline in pairs:
            assert isinstance(snap, GraphSnapshot)
            assert isinstance(baseline, list)
            # Each test snapshot should have some baseline history
            assert len(baseline) > 0
            assert len(baseline) <= 4

    def test_min_node_size_propagates(self, tmp_data_dir: Path):
        meta_path = tmp_data_dir / "meta_df.csv"
        loader = SnapshotLoader(
            data_dir=str(tmp_data_dir),
            meta_path=str(meta_path),
            min_node_size=1500,
        )
        snaps = loader.load()
        for snap in snaps:
            # Node "0" has size=1000 -> filtered. "1"=2000, "2"=3000 survive.
            assert "0" not in snap.nodes
            assert "empty" not in snap.nodes


# ── Integration: real data (skipped if files not present) ────────────────


@pytest.fixture
def real_loader():
    data_dir = Path("data/")
    meta_path = Path("data/meta_df.csv")
    if not data_dir.exists() or not list(data_dir.glob("historical-network_week_*.json")):
        pytest.skip("Real data files not available")
    return SnapshotLoader(str(data_dir), str(meta_path), min_node_size=0.0)


class TestRealData:
    def test_load_recent(self, real_loader: SnapshotLoader):
        snaps = real_loader.load(date_range="2025-07~2025-08")
        assert len(snaps) > 0
        for snap in snaps:
            assert len(snap.nodes) > 100  # Plenty of protocols
            assert len(snap.edges) > 100

    def test_baseline_metrics(self, real_loader: SnapshotLoader):
        baseline = real_loader.build_baseline(before="2025-07", window=4)
        assert len(baseline) > 0
        for m in baseline:
            assert 0 <= m["N3"] <= 1  # Density in [0, 1]
            assert 0 <= m["N4"] <= 1  # Gini in [0, 1]
