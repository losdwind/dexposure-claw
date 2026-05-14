"""Shared experiment logging and progress tracking utility.

Provides:
- Structured loguru logging to both stderr and timestamped file
- tqdm progress bars with ETA for long-running loops
- Timing decorators and context managers
- Per-step metric summaries
- Experiment-level summary reports

Usage:
    from experiments.exp_logger import ExpLogger

    log = ExpLogger("B1", results_dir="results/")
    for snap in log.progress(snapshots, desc="Evaluating snapshots"):
        ...
        log.step("snapshot", date=snap.date, mae=0.03, rho=0.55)
    log.summary({"mean_mae": 0.03, "mean_rho": 0.55})
"""
from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator

from loguru import logger

# Try to import tqdm; fall back to a simple wrapper if unavailable
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        """Minimal tqdm fallback that logs progress every 10%."""
        def __init__(self, iterable=None, desc="", total=None, **kwargs):
            self.iterable = iterable
            self.desc = desc
            self.total = total or (len(iterable) if hasattr(iterable, '__len__') else None)
            self.n = 0
            self._last_pct = -1
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.n += 1
                if self.total:
                    pct = int(self.n / self.total * 100) // 10 * 10
                    if pct > self._last_pct:
                        self._last_pct = pct
                        logger.info(f"{self.desc}: {self.n}/{self.total} ({pct}%)")
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def set_postfix_str(self, s): pass
        def update(self, n=1): self.n += n


class ExpLogger:
    """Experiment logger with structured logging and progress tracking."""

    def __init__(
        self,
        benchmark: str,
        method: str = "",
        results_dir: str | Path = "results/",
        log_level: str = "INFO",
    ):
        self.benchmark = benchmark
        self.method = method
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._start_time = time.time()
        self._step_times: list[float] = []
        self._step_metrics: list[dict[str, Any]] = []

        # Setup loguru: file + stderr.
        # Filenames use "__" between benchmark and method so the segments stay
        # parseable even though the IDs themselves contain single underscores.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = (
            self.results_dir / f"{benchmark}__{method}__{timestamp}.log"
            if method
            else self.results_dir / f"{benchmark}__{timestamp}.log"
        )

        # Remove default logger, add our handlers
        # Note: we don't remove global handlers to avoid interfering with other loggers
        self._file_id = logger.add(
            str(self.log_file),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
            level="DEBUG",
            rotation="50 MB",
        )

        logger.info(
            "=" * 60 + "\n"
            f"  Experiment: {benchmark} | Method: {method or 'all'}\n"
            f"  Results dir: {self.results_dir}\n"
            f"  Log file: {self.log_file}\n"
            f"  Started: {datetime.now().isoformat()}\n"
            + "=" * 60
        )

    def progress(
        self,
        iterable: Iterable,
        desc: str = "",
        total: int | None = None,
        unit: str = "it",
    ) -> Iterator:
        """Wrap an iterable with tqdm progress bar + logging.

        Logs start, 25/50/75%, and completion with timing.
        """
        if total is None and hasattr(iterable, '__len__'):
            total = len(iterable)

        desc_full = f"{self.benchmark}"
        if self.method:
            desc_full += f"/{self.method}"
        if desc:
            desc_full += f" | {desc}"

        logger.info(f"Starting: {desc_full} ({total or '?'} {unit})")
        t0 = time.time()

        bar = tqdm(
            iterable,
            desc=desc_full,
            total=total,
            unit=unit,
            ncols=100,
            leave=True,
            file=sys.stderr,
        )

        milestones = set()
        for i, item in enumerate(bar):
            yield item

            # Log milestones at 25%, 50%, 75%
            if total and total > 4:
                pct = int((i + 1) / total * 100)
                for milestone in [25, 50, 75]:
                    if pct >= milestone and milestone not in milestones:
                        milestones.add(milestone)
                        elapsed = time.time() - t0
                        eta = elapsed / (i + 1) * (total - i - 1)
                        logger.info(
                            f"  [{desc_full}] {pct}% ({i+1}/{total}) "
                            f"elapsed={_fmt_time(elapsed)} ETA={_fmt_time(eta)}"
                        )

        elapsed = time.time() - t0
        logger.info(f"Completed: {desc_full} in {_fmt_time(elapsed)}")

    def step(self, label: str = "", **metrics: Any):
        """Log a single step with key metrics."""
        t = time.time()
        self._step_times.append(t)
        self._step_metrics.append({"label": label, **metrics})

        # Format metrics for log
        metric_str = " | ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        if label:
            logger.debug(f"  step: {label} | {metric_str}")
        else:
            logger.debug(f"  step: {metric_str}")

    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        logger.error(msg, *args, **kwargs)

    @contextmanager
    def timer(self, label: str):
        """Context manager that logs elapsed time for a block."""
        logger.info(f"  [{label}] starting...")
        t0 = time.time()
        yield
        elapsed = time.time() - t0
        logger.info(f"  [{label}] done in {_fmt_time(elapsed)}")

    def summary(self, metrics: dict[str, Any], extra: str = ""):
        """Log final summary with key metrics and timing."""
        total_elapsed = time.time() - self._start_time

        lines = [
            "",
            "=" * 60,
            f"  SUMMARY: {self.benchmark}",
        ]
        if self.method:
            lines.append(f"  Method: {self.method}")
        lines.append(f"  Total time: {_fmt_time(total_elapsed)}")
        lines.append(f"  Steps logged: {len(self._step_metrics)}")
        lines.append("-" * 60)
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k}: {v:.6f}")
            else:
                lines.append(f"  {k}: {v}")
        if extra:
            lines.append(f"  {extra}")
        lines.append("=" * 60)

        logger.info("\n".join(lines))

    def save_results(self, results: Any, filename: str | None = None):
        """Save results to JSON with metadata."""
        if filename is None:
            filename = (
                f"{self.benchmark}__{self.method}.json"
                if self.method
                else f"{self.benchmark}.json"
            )

        out_path = self.results_dir / filename
        total_elapsed = time.time() - self._start_time

        payload = {
            "benchmark": self.benchmark,
            "method": self.method,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(total_elapsed, 2),
            "log_file": str(self.log_file),
            "results": results,
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info(f"Results saved to {out_path}")
        return out_path

    def close(self):
        """Clean up logger handler."""
        try:
            logger.remove(self._file_id)
        except Exception:
            pass


def _fmt_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{int(s)}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)}h{int(m)}m{int(s)}s"
