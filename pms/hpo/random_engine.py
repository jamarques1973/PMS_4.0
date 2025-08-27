from __future__ import annotations

import random
from typing import Any, Callable, Dict

from .interface import HPOEngine, HPOConfig


class RandomSearchEngine(HPOEngine):
    def __init__(self, cfg: HPOConfig, search_space: Callable[[Any], Dict[str, Any]] | Dict[str, Any]):
        self.cfg = cfg
        self.search_space = search_space

    def optimize(self, objective):
        best = None
        best_score = float("inf")
        rnd = random.Random(self.cfg.seed)
        for _ in range(self.cfg.n_trials):
            if callable(self.search_space):
                params = self.search_space({"random": rnd})
            else:
                params = {k: rnd.choice(v) if isinstance(v, (list, tuple)) else v for k, v in self.search_space.items()}
            score = objective(params)
            if score < best_score:
                best_score, best = score, params
        return best or (self.search_space({}) if callable(self.search_space) else dict(self.search_space))

