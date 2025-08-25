"""Configuration and parameter management for PMS 4.0.0.

This module defines the ParameterStore which centralizes configurable
parameters and provides load/save facilities from JSON and YAML.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore


DEFAULT_CONFIG_DIR = Path(os.environ.get("PMS4_CONFIG_DIR", "/workspace/config"))


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class ParameterStore:
    """Holds all configurable parameters for PMS.

    New parameters should be added here to ensure consistent usage across layers.
    """

    debug_enabled: bool = False
    allow_shell_commands: bool = False
    allow_pip_installs: bool = False
    data_dir: str = "/workspace/data"
    output_dir: str = "/workspace/output"
    input_notebook_path: str = "/workspace/input/PMS_3_6_0.ipynb"
    generated_package_dir: str = "/workspace/pms4/generated"
    logs_dir: str = "/workspace/logs"
    theme: str = "light"

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterStore":
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        init_kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for key, value in data.items():
            if key in known_fields:
                init_kwargs[key] = value
            else:
                extra[key] = value
        instance = cls(**init_kwargs)
        instance.extra.update(extra)
        return instance

    def apply_overrides(self, overrides: Optional[Dict[str, Any]] = None) -> None:
        if not overrides:
            return
        for key, value in overrides.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extra[key] = value

    def save_json(self, path: Optional[str | os.PathLike[str]] = None) -> Path:
        if path is None:
            ensure_directory(DEFAULT_CONFIG_DIR)
            path = DEFAULT_CONFIG_DIR / "pms4_config.json"
        target = Path(path)
        ensure_directory(target.parent)
        with target.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return target

    @classmethod
    def load_json(cls, path: Optional[str | os.PathLike[str]] = None) -> "ParameterStore":
        if path is None:
            path = DEFAULT_CONFIG_DIR / "pms4_config.json"
        target = Path(path)
        if not target.exists():
            return cls()
        with target.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_yaml(self, path: Optional[str | os.PathLike[str]] = None) -> Optional[Path]:
        if yaml is None:
            return None
        if path is None:
            ensure_directory(DEFAULT_CONFIG_DIR)
            path = DEFAULT_CONFIG_DIR / "pms4_config.yaml"
        target = Path(path)
        ensure_directory(target.parent)
        with target.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)  # type: ignore
        return target

    @classmethod
    def load_yaml(cls, path: Optional[str | os.PathLike[str]] = None) -> "ParameterStore":
        if yaml is None:
            return cls.load_json()
        if path is None:
            path = DEFAULT_CONFIG_DIR / "pms4_config.yaml"
        target = Path(path)
        if not target.exists():
            return cls()
        with target.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)  # type: ignore
        return cls.from_dict(data or {})

