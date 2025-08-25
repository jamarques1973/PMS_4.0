"""Execution engine for PMS 4.0.0.

Runs parsed code artifacts in isolated namespaces with tracing and logging.
"""

from __future__ import annotations

import builtins
import logging
from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, List, Optional

from ..utils.tracing import Tracer


logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    namespace: Dict[str, object]
    trace_rows: List[Dict[str, float | str]]


SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in [
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "len",
        "list",
        "map",
        "max",
        "min",
        "pow",
        "print",
        "range",
        "sum",
        "zip",
        "set",
        "tuple",
        "str",
        "repr",
        "sorted",
    ]
}


class ExecutionEngine:
    """Executes code strings within a controlled global namespace."""

    def __init__(self, allow_unsafe: bool = True) -> None:
        # Default to unsafe True to preserve original notebook behavior.
        # Can be restricted by passing allow_unsafe=False.
        self.allow_unsafe = allow_unsafe
        self.tracer = Tracer()

    def _make_globals(self) -> Dict[str, object]:
        g = {}
        g["__builtins__"] = builtins if self.allow_unsafe else MappingProxyType(SAFE_BUILTINS)
        return g

    def exec_code_blocks(self, labeled_blocks: List[tuple[str, str]]) -> ExecutionResult:
        namespace: Dict[str, object] = self._make_globals()
        for name, code in labeled_blocks:
            logger.debug("Executing block: %s", name)
            with self.tracer.span(f"exec:{name}"):
                exec(code, namespace, namespace)
        return ExecutionResult(namespace=namespace, trace_rows=self.tracer.to_rows())

