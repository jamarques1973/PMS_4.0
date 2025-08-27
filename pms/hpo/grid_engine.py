from __future__ import annotations

import itertools
from typing import Any, Dict

from .interface import HPOEngine, HPOConfig


class GridSearchEngine(HPOEngine):
    def __init__(self, cfg: HPOConfig, grid: Dict[str, Any]):
        self.cfg = cfg
        self.grid = grid

    def optimize(self, objective):
        keys = list(self.grid.keys())
        values = [v if isinstance(v, (list, tuple)) else [v] for v in self.grid.values()]
        best, best_score = None, float("inf")
        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            score = objective(params)
            if score < best_score:
                best_score, best = score, params
        return best if best is not None else {k: v[0] for k, v in zip(keys, values)}

