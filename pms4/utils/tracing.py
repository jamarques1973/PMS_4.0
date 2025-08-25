"""Execution tracing utilities for PMS 4.0.0.

Provides a lightweight tracer to measure function and block execution times.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional


@dataclass
class TraceEvent:
    name: str
    start_ts: float
    end_ts: float

    @property
    def duration_ms(self) -> float:
        return (self.end_ts - self.start_ts) * 1000.0


class Tracer:
    """Simple tracer that collects named timing events."""

    def __init__(self) -> None:
        self._events: List[TraceEvent] = []

    @contextmanager
    def span(self, name: str) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self._events.append(TraceEvent(name=name, start_ts=start, end_ts=end))

    def record(self, name: str, func, *args, **kwargs):  # type: ignore
        with self.span(name):
            return func(*args, **kwargs)

    def to_rows(self) -> List[Dict[str, float | str]]:
        return [
            {"name": e.name, "duration_ms": round(e.duration_ms, 3)} for e in self._events
        ]

    def clear(self) -> None:
        self._events.clear()

