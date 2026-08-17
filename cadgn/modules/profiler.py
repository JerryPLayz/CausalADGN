from __future__ import annotations
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Generator, Literal
import numpy as np
from scipy.stats import iqr, quantile

import torch


class Profiler:
    """
    Lightweight Section Timer for the CA-DGN training pipeline.
    Accumulates timings per named section across all steps and
    epochs, providing mean, std, min and max statistics.

    GPU Note:
        `sync_cuda=True` enforces `torch.cuda.synchronize()`, which imposes a small overhead.
    """

    def __init__(
            self,
            sync_cuda: bool = True,
            enabled: bool = True
    ):
        """

        :param sync_cuda: Synchronize CUDA before timing. Set False if timing overhead is excessive.
        :param enabled: When false, the context manager is a no-op and no timings will be recorded.
        """
        self.sync_cuda = sync_cuda and torch.cuda.is_available()
        self.enabled = enabled
        self._timings: dict[str, list[float]] = defaultdict(list)

    def reset(self) -> None:
        """
        Clear all accumulated timings.
        :return:
        """
        self._timings.clear()

    @contextmanager
    def section(self, name: str) -> Generator[None, None, None]:
        """
        :param name: Section label - used as key in stats output. Nested calls with the same name accumulate into the same list.
        """
        if not self.enabled:
            yield
            return
        if self.sync_cuda:
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        yield

        if self.sync_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        self._timings[name].append(elapsed)

    def stats(self) -> dict[str, dict[str, float]]:
        """
        Compute mean, std, min and max for each section
        All times are in seconds.
        """
        import statistics
        result = {}
        for name, times in self._timings.items():
            n = len(times)
            result[name] = {
                "mean": sum(times) / n,
                "std": statistics.stdev(times) if n > 1 else 0.0,
                "min": min(times),
                "max": max(times),
                "q1": quantile(times, 0.25),
                "q3": quantile(times, 0.75),
                "iqr": iqr(times),
                "total": sum(times),
                "n": n
            }
        return result

    def summary(
            self,
            sort_by: Literal["mean", "total", "max", "min", "iqr", "q1", "q3"] = "mean"):
        """
        Human-Readable Summary Table, sorted by a stat key, `sort_by`.
        :param sort_by: One of "mean", "total", "max", "min", "iqr", "q1", "q3"
        :return: Formatted string of the summary statistics.
        """
        s = self.stats()
        if not s:
            return "Profiler: No timings recorded"

        rows = sorted(
            s.items(),
            key=lambda x: x[1][sort_by],
            reverse=True
        )

        lines = [
            f"{'Section':<30} {'Mean':>9} {'Std':>9} "
            f"{'Min':>9} {'Max':>9} {'Q1':>9} {'Q3':>9} {'IQR':>9} {'Total':>10} {'N':>6}",
            "-"*115
        ]
        for name, d in rows:
            lines.append(
                f"{name:<30} "
                f"{d['mean']*1000:>8.2f}ms "
                f"{d['std']*1000:>8.2f}ms "
                f"{d['min']*1000:>8.2f}ms "
                f"{d['max']*1000:>8.2f}ms "
                f"{d['q1']*1000:>8.2f}ms "
                f"{d['q3']*1000:>8.2f}ms "
                f"{d['iqr']*1000:>8.2f}ms "
                f"{d['total']:>8.2f}s "
                f"{d['n']:>6d}"
            )
        return "\n".join(lines)

    def get_latest_n(self, key: str):
        return self._timings.get(key, [0])[-1]

