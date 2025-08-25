"""Global registry for PMS 4.0.0.

Manages global variables, module references, and cross-layer resources.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class GlobalRegistry:
    """Central registry to share global resources across PMS layers."""

    globals: Dict[str, Any] = field(default_factory=dict)
    modules: Dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        self.globals[key] = value

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self.globals.get(key, default)

    def register_module(self, name: str, module: Any) -> None:
        self.modules[name] = module

    def get_module(self, name: str) -> Any:
        return self.modules[name]


REGISTRY = GlobalRegistry()

