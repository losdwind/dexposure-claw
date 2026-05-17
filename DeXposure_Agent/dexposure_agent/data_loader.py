"""Data loading pipeline for DeXposure-Agent experiments.

Bridges the raw JSON weekly snapshots (historical-network_week_*.json)
and protocol metadata (meta_df.csv) into the agent pipeline's
GraphSnapshot Pydantic objects.

Also provides helpers for:
- Date-range filtering (parse 'YYYY-MM~YYYY-MM' split strings)
- Baseline metric history (for rolling-window z-score comparisons)

Usage:
    loader = SnapshotLoader("data/", meta_path="data/meta_df.csv")
    snapshots = loader.load(date_range="2025-01~2025-08")
    baseline  = loader.build_baseline(before="2025-01", window=26)
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dexposure_agent.monitor import compute_metrics
from dexposure_agent.types import Edge, GraphSnapshot, NodeFeatures

logger = logging.getLogger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = "data/"
DEFAULT_META_PATH = "data/meta_df.csv"
DATA_FILE_GLOB = "historical-network_week_*.json"

EPS = 1e-12


# ── Date helpers ────────────────────────────────────────────────────────


def parse_date_range(range_str: str) -> tuple[datetime, datetime]:
    """Parse 'YYYY-MM~YYYY-MM' into (start_date, end_date) datetimes.

    The start is the first day of start_month; the end is the last day
    of end_month (approximated as first day of next month - 1 day).

    >>> parse_date_range("2025-01~2025-08")
    (datetime(2025, 1, 1), datetime(2025, 8, 31, ...))
    """
    parts = range_str.split("~")
    if len(parts) != 2:
        raise ValueError(f"Expected 'YYYY-MM~YYYY-MM', got: {range_str!r}")

    start = datetime.strptime(parts[0].strip(), "%Y-%m")
    end_month = datetime.strptime(parts[1].strip(), "%Y-%m")
    # End of month: go to first of next month, subtract 1 day
    if end_month.month == 12:
        end = datetime(end_month.year + 1, 1, 1)
    else:
        end = datetime(end_month.year, end_month.month + 1, 1)
    from datetime import timedelta
    end = end - timedelta(days=1)

    return start, end


def _parse_snapshot_date(date_str: str) -> datetime:
    """Parse a snapshot date string (YYYY-MM-DD) into a datetime."""
    return datetime.strptime(date_str, "%Y-%m-%d")


def _date_in_range(
    date_str: str, start: datetime | None, end: datetime | None
) -> bool:
    """Check if a date string falls within [start, end] inclusive."""
    if start is None and end is None:
        return True
    dt = _parse_snapshot_date(date_str)
    if start is not None and dt < start:
        return False
    if end is not None and dt > end:
        return False
    return True


# ── Node feature extraction ─────────────────────────────────────────────


def _extract_node_features(
    node: dict[str, Any],
    meta_category: dict[str, str],
) -> NodeFeatures:
    """Convert a raw JSON node dict into a NodeFeatures Pydantic object.

    Features:
      - log_size:  log(1 + TVL)
      - num_tokens: number of composition tokens
      - max_share:  max token share in composition
      - entropy:    Shannon entropy of composition shares
      - category:   from meta_df.csv (default 'Unknown')
    """
    node_id = str(node.get("id", ""))
    size = float(node.get("size", 0.0))
    comp = node.get("composition", {}) or {}

    log_size = math.log1p(max(size, 0.0))
    num_tokens = len(comp)

    if size > 0 and comp:
        values = np.array(list(comp.values()), dtype=np.float64)
        values = np.maximum(values, 0.0)
        total = values.sum() + EPS
        shares = values / total
        max_share = float(shares.max())
        entropy = float(-(shares * np.log(shares + EPS)).sum())
    else:
        max_share = 0.0
        entropy = 0.0

    category = meta_category.get(node_id, "Unknown")

    return NodeFeatures(
        log_size=log_size,
        num_tokens=num_tokens,
        max_share=max_share,
        entropy=entropy,
        category=category,
    )


# ── Snapshot conversion ──────────────────────────────────────────────────


def raw_to_graph_snapshot(
    date: str,
    snapshot: dict[str, Any],
    meta_category: dict[str, str],
    min_node_size: float = 0.0,
) -> GraphSnapshot:
    """Convert one raw JSON snapshot into a GraphSnapshot.

    Args:
        date: Snapshot date string (YYYY-MM-DD).
        snapshot: Raw dict with 'nodes' list and 'links' list.
        meta_category: node_id -> category string from metadata.
        min_node_size: Skip nodes with TVL below this (default 0 = keep all).

    Returns:
        GraphSnapshot with NodeFeatures and Edges.
    """
    raw_nodes = snapshot.get("nodes", [])
    raw_links = snapshot.get("links", [])

    # Pass 1: build node dict, applying size filter
    nodes: dict[str, NodeFeatures] = {}
    for raw_node in raw_nodes:
        node_id = str(raw_node.get("id", ""))
        if not node_id:
            continue
        size = float(raw_node.get("size", 0.0))
        if size < min_node_size:
            continue
        nodes[node_id] = _extract_node_features(raw_node, meta_category)

    node_set = set(nodes.keys())

    # Pass 2: build edges, only between included nodes
    edges: list[Edge] = []
    for raw_link in raw_links:
        src = str(raw_link.get("source", ""))
        tgt = str(raw_link.get("target", ""))
        weight = float(raw_link.get("size", 0.0))
        if src in node_set and tgt in node_set and weight > 0:
            edges.append(Edge(source=src, target=tgt, weight=weight))

    return GraphSnapshot(date=date, nodes=nodes, edges=edges)


# ── Main loader ──────────────────────────────────────────────────────────


class SnapshotLoader:
    """Loads and filters weekly graph snapshots for the agent pipeline.

    Typical usage:
        loader = SnapshotLoader("data/")
        test_snaps = loader.load(date_range="2025-01~2025-08")
        baseline   = loader.build_baseline(before="2025-01", window=26)
    """

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        meta_path: str = DEFAULT_META_PATH,
        min_node_size: float = 0.0,
    ):
        self.data_dir = Path(data_dir)
        # Resolve meta_path: if default and not found, try inside data_dir
        meta = Path(meta_path)
        if not meta.exists() and not meta.is_absolute():
            meta_in_data = self.data_dir / meta.name
            if meta_in_data.exists():
                meta = meta_in_data
        self.meta_path = meta
        self.min_node_size = min_node_size

        # Lazy-loaded caches
        self._meta_category: dict[str, str] | None = None
        self._raw_data: dict[str, dict] | None = None
        self._sorted_dates: list[str] | None = None

    # ── Metadata ─────────────────────────────────────────────────────

    def _load_metadata(self) -> dict[str, str]:
        """Load meta_df.csv -> {node_id: category}."""
        if self._meta_category is not None:
            return self._meta_category

        if not self.meta_path.exists():
            logger.warning("Metadata file not found: %s — using empty category map", self.meta_path)
            self._meta_category = {}
            return self._meta_category

        df = pd.read_csv(self.meta_path)
        df["id"] = df["id"].astype(str)
        self._meta_category = df.set_index("id")["category"].to_dict()
        logger.info("Loaded metadata: %d protocols from %s", len(self._meta_category), self.meta_path)
        return self._meta_category

    # ── Raw JSON loading ─────────────────────────────────────────────

    def _load_raw_data(self) -> dict[str, dict]:
        """Load all JSON files in data_dir, merge into {date: snapshot}.

        Tries ijson for streaming large files; falls back to json.load().
        """
        if self._raw_data is not None:
            return self._raw_data

        json_files = sorted(self.data_dir.glob(DATA_FILE_GLOB))
        if not json_files:
            raise FileNotFoundError(
                f"No {DATA_FILE_GLOB} files found in {self.data_dir}"
            )

        merged: dict[str, dict] = {}
        for path in json_files:
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info("Loading %s (%.1f MB)", path.name, size_mb)
            merged.update(self._read_json_file(path))

        self._raw_data = merged
        self._sorted_dates = sorted(merged.keys())
        logger.info(
            "Loaded %d weekly snapshots: %s ~ %s",
            len(self._sorted_dates),
            self._sorted_dates[0] if self._sorted_dates else "?",
            self._sorted_dates[-1] if self._sorted_dates else "?",
        )
        return self._raw_data

    @staticmethod
    def _read_json_file(path: Path) -> dict[str, dict]:
        """Read a single JSON file, returning {date: snapshot_dict}."""
        with path.open("rb") as f:
            try:
                import ijson
                return dict(ijson.kvitems(f, "data"))
            except ImportError:
                f.seek(0)
                payload = json.load(f)
                return payload.get("data", payload)
            except Exception:
                f.seek(0)
                payload = json.load(f)
                return payload.get("data", payload)

    # ── Public API ───────────────────────────────────────────────────

    @property
    def dates(self) -> list[str]:
        """All available snapshot dates, sorted chronologically."""
        self._load_raw_data()
        assert self._sorted_dates is not None
        return self._sorted_dates

    def load(
        self,
        date_range: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> list[GraphSnapshot]:
        """Load snapshots filtered by date range.

        Args:
            date_range: 'YYYY-MM~YYYY-MM' string (convenience).
            start: Explicit start date 'YYYY-MM-DD' (overrides date_range).
            end: Explicit end date 'YYYY-MM-DD' (overrides date_range).

        Returns:
            List of GraphSnapshot sorted by date, oldest first.
        """
        raw = self._load_raw_data()
        meta = self._load_metadata()

        # Resolve date bounds
        dt_start: datetime | None = None
        dt_end: datetime | None = None

        if date_range:
            dt_start, dt_end = parse_date_range(date_range)
        if start:
            dt_start = _parse_snapshot_date(start)
        if end:
            dt_end = _parse_snapshot_date(end)

        # Filter and convert
        snapshots: list[GraphSnapshot] = []
        for date_str in self.dates:
            if not _date_in_range(date_str, dt_start, dt_end):
                continue
            snap = raw_to_graph_snapshot(
                date_str, raw[date_str], meta, self.min_node_size
            )
            snapshots.append(snap)

        logger.info(
            "Loaded %d snapshots for range %s",
            len(snapshots),
            date_range or f"{start}~{end}",
        )
        return snapshots

    def load_single(self, date: str) -> GraphSnapshot:
        """Load a single snapshot by exact date string."""
        raw = self._load_raw_data()
        meta = self._load_metadata()

        if date not in raw:
            available = [d for d in self.dates if d.startswith(date[:7])]
            hint = f" (closest: {available})" if available else ""
            raise KeyError(f"No snapshot for date {date!r}{hint}")

        return raw_to_graph_snapshot(date, raw[date], meta, self.min_node_size)

    def build_baseline(
        self,
        before: str,
        window: int = 26,
    ) -> list[dict[str, float]]:
        """Build a baseline metric history for rolling-window z-score comparison.

        Loads `window` snapshots ending just before `before` date, computes
        Phi metrics (N1..N5) on each, returns list of metric dicts.

        Args:
            before: Date string 'YYYY-MM-DD' or 'YYYY-MM'. Snapshots strictly
                    before this date are used.
            window: Number of past snapshots to include (default 26 = ~6 months).

        Returns:
            List of {metric_id: value} dicts, oldest first.
        """
        raw = self._load_raw_data()
        meta = self._load_metadata()

        # Find snapshots before the cutoff
        if len(before) == 7:  # YYYY-MM
            cutoff = datetime.strptime(before, "%Y-%m")
        else:
            cutoff = _parse_snapshot_date(before)

        eligible_dates = [
            d for d in self.dates if _parse_snapshot_date(d) < cutoff
        ]
        selected = eligible_dates[-window:]  # last `window` dates

        if not selected:
            logger.warning("No snapshots found before %s for baseline", before)
            return []

        logger.info(
            "Building baseline: %d snapshots before %s (%s ~ %s)",
            len(selected), before, selected[0], selected[-1],
        )

        history: list[dict[str, float]] = []
        for date_str in selected:
            snap = raw_to_graph_snapshot(
                date_str, raw[date_str], meta, self.min_node_size
            )
            metrics = compute_metrics(snap)
            history.append(metrics)

        return history

    def iter_test_with_baselines(
        self,
        test_range: str,
        baseline_window: int = 26,
    ):
        """Iterate over test snapshots, yielding (snapshot, baseline_history) pairs.

        For each test snapshot at date t, the baseline is the preceding
        `baseline_window` snapshots' metric dicts. This is the main entry point
        for benchmark runners (b1_forecast..b6_robustness).

        Yields:
            (GraphSnapshot, list[dict[str, float]]) tuples.
        """
        raw = self._load_raw_data()
        meta = self._load_metadata()
        dt_start, dt_end = parse_date_range(test_range)

        # Pre-compute all metrics for the full timeline (cached)
        all_metrics: dict[str, dict[str, float]] = {}

        for date_str in self.dates:
            dt = _parse_snapshot_date(date_str)
            # We need metrics for dates before the test range (for baseline)
            # and within the test range (for ground truth comparison)
            if dt <= dt_end:
                snap = raw_to_graph_snapshot(
                    date_str, raw[date_str], meta, self.min_node_size
                )
                all_metrics[date_str] = compute_metrics(snap)

        # Yield test snapshots with their baselines
        for date_str in self.dates:
            if not _date_in_range(date_str, dt_start, dt_end):
                continue

            snap = raw_to_graph_snapshot(
                date_str, raw[date_str], meta, self.min_node_size
            )

            # Collect baseline: metrics from previous dates
            prior_dates = [
                d for d in self.dates
                if _parse_snapshot_date(d) < _parse_snapshot_date(date_str)
            ]
            baseline_dates = prior_dates[-baseline_window:]
            baseline = [all_metrics[d] for d in baseline_dates if d in all_metrics]

            yield snap, baseline
