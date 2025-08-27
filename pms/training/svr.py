from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from pydantic import BaseModel, Field

from ..hpo.interface import HPOConfig, make_hpo_engine


class SVRConfig(BaseModel):
    kernel: str = "rbf"
    C: float = 1.0
    epsilon: float = 0.1
    gamma: str | float = "scale"

    # Default search space for HPO
    def search_space(self):
        return {
            "kernel": ["rbf", "poly", "sigmoid"],
            "C": [0.1, 1.0, 10.0, 100.0],
            "epsilon": [0.01, 0.1, 0.5],
            "gamma": ["scale", "auto"],
        }

    def to_params(self) -> Dict[str, Any]:
        return {
            "kernel": self.kernel,
            "C": self.C,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
        }


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, metrics_list: List[str]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for m in metrics_list:
        if m == "r2":
            results[m] = float(metrics.r2_score(y_true, y_pred))
        elif m == "rmse":
            results[m] = float(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))
        elif m == "mae":
            results[m] = float(metrics.mean_absolute_error(y_true, y_pred))
        else:
            raise ValueError(f"Unsupported metric: {m}")
    return results


class SVRTrainer:
    def __init__(self, cfg: Optional[SVRConfig], hpo_cfg: Optional[HPOConfig], random_state: int) -> None:
        self.cfg = cfg or SVRConfig()
        self.hpo_cfg = hpo_cfg or HPOConfig(engine="none")
        self.random_state = random_state

    def _build_pipeline(self, params: Dict[str, Any]) -> Pipeline:
        svr = SVR(**params)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", svr),
        ])
        return pipeline

    def fit_evaluate(self, X: pd.DataFrame, y: pd.Series, test_size: float, out_dir: Path, metrics_list: List[str]) -> Dict[str, Any]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # HPO or fixed params
        def objective(params: Dict[str, Any]) -> float:
            model = self._build_pipeline(params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = float(np.sqrt(metrics.mean_squared_error(y_test, preds)))
            return rmse

        engine = (self.hpo_cfg.engine or "none").lower()
        if engine == "optuna":
            def opt_space(trial):
                return {
                    "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"]),
                    "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
                    "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
                    "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                }
            hpo_engine = make_hpo_engine(self.hpo_cfg, opt_space)
        elif engine in {"random", "grid"}:
            hpo_engine = make_hpo_engine(self.hpo_cfg, self.cfg.search_space())
        else:
            # none or unknown -> use fixed params
            hpo_engine = make_hpo_engine(self.hpo_cfg, self.cfg.to_params())

        best_params = hpo_engine.optimize(objective)

        # Final train with best params
        model = self._build_pipeline(best_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metric_values = evaluate_metrics(y_test.to_numpy(), preds, metrics_list)

        # Persist
        out_dir.mkdir(parents=True, exist_ok=True)
        model_path = out_dir / "svr_model.joblib"
        meta_path = out_dir / "svr_meta.json"
        joblib.dump(model, model_path)
        meta = {
            "params": best_params,
            "metrics": metric_values,
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        return {
            "model_path": str(model_path),
            "metrics": metric_values,
            "best_params": best_params,
        }

