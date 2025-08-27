from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel


class HPOConfig(BaseModel):
    engine: str = "none"  # none|random|grid|optuna|kerastuner|skopt
    n_trials: int = 25
    timeout_s: Optional[int] = None
    seed: int = 42


class HPOEngine(ABC):
    @abstractmethod
    def optimize(self, objective: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        ...


class NoHPO(HPOEngine):
    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def optimize(self, objective: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        # Just return provided params without optimization
        return self.params


def make_hpo_engine(cfg: HPOConfig, search_space: Callable[[Any], Dict[str, Any]] | Dict[str, Any]) -> HPOEngine:
    engine = cfg.engine.lower() if cfg and cfg.engine else "none"
    if engine == "none":
        params = search_space({}) if callable(search_space) else dict(search_space)
        return NoHPO(params)
    elif engine == "optuna":
        from .optuna_engine import OptunaEngine

        return OptunaEngine(cfg, search_space)
    elif engine == "random":
        from .random_engine import RandomSearchEngine

        return RandomSearchEngine(cfg, search_space)
    elif engine == "grid":
        from .grid_engine import GridSearchEngine

        return GridSearchEngine(cfg, search_space)
    elif engine == "kerastuner":
        from .kt_engine import KerasTunerEngine

        return KerasTunerEngine(cfg, search_space)
    else:
        raise ValueError(f"Unknown HPO engine: {engine}")

