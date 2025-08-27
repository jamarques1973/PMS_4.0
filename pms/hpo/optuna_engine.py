from __future__ import annotations

from typing import Any, Callable, Dict

import optuna

from .interface import HPOEngine, HPOConfig


class OptunaEngine(HPOEngine):
    def __init__(self, cfg: HPOConfig, search_space: Callable[[optuna.Trial], Dict[str, Any]]):
        self.cfg = cfg
        self.search_space = search_space

    def optimize(self, objective: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        def obj(trial: optuna.Trial) -> float:
            params = self.search_space(trial)
            return objective(params)

        sampler = optuna.samplers.TPESampler(seed=self.cfg.seed)
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
        study.optimize(obj, n_trials=self.cfg.n_trials, timeout=self.cfg.timeout_s)
        return study.best_params

