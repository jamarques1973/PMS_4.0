from __future__ import annotations

from typing import Any, Callable, Dict

import keras_tuner as kt

from .interface import HPOEngine, HPOConfig


class KerasTunerEngine(HPOEngine):
    def __init__(self, cfg: HPOConfig, hp_builder: Callable[[kt.HyperParameters], Dict[str, Any]]):
        self.cfg = cfg
        self.hp_builder = hp_builder

    def optimize(self, objective: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        class ObjectiveWrapper(kt.HyperModel):
            def build(self, hp: kt.HyperParameters):  # type: ignore
                params = self.hp_builder(hp)
                # KerasTuner expects a model, but we only need params
                # We'll return a dummy minimal model and encode params in hp for retrieval
                return None  # type: ignore

        # Fallback strategy: sample hyperparameters using tuner API and evaluate externally
        tuner = kt.tuners.BayesianOptimization(
            ObjectiveWrapper(),
            objective=kt.Objective("val_loss", direction="min"),
            max_trials=self.cfg.n_trials,
            seed=self.cfg.seed,
            overwrite=True,
            directory=".kt_tmp",
            project_name="pms"
        )

        best_params = None
        best_score = float("inf")
        # Use HyperParameters sampling directly
        for _ in range(self.cfg.n_trials):
            hp = kt.HyperParameters()
            params = self.hp_builder(hp)
            score = objective(params)
            if score < best_score:
                best_score, best_params = score, params
        return best_params or {}

