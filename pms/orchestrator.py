from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from .training.svr import SVRTrainer, SVRConfig
from .hpo.interface import HPOConfig


class DataConfig(BaseModel):
    input_path: Path
    target: str
    sep: str = ","
    index_col: Optional[str] = None
    features: Optional[list[str]] = None


class TrainConfig(BaseModel):
    task: str = Field(default="svr", description="Model type: svr|rf|xgb|nn|rnn")
    output_dir: Path = Path("artifacts")
    random_state: int = 42
    test_size: float = 0.2
    metrics: list[str] = ["r2", "rmse", "mae"]
    svr: Optional[SVRConfig] = None
    hpo: Optional[HPOConfig] = None


class OrchestratorConfig(BaseModel):
    data: DataConfig
    train: TrainConfig

    @staticmethod
    def from_yaml(path: Path) -> "OrchestratorConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return OrchestratorConfig.model_validate(raw)


class Orchestrator:
    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config

    def run(self) -> Dict[str, Any]:
        # 1) Load data
        df = pd.read_csv(self.config.data.input_path, sep=self.config.data.sep)
        if self.config.data.index_col and self.config.data.index_col in df.columns:
            df = df.set_index(self.config.data.index_col)
        features = self.config.data.features or [c for c in df.columns if c != self.config.data.target]
        X = df[features]
        y = df[self.config.data.target]

        # 2) Dispatch model training
        task = self.config.train.task.lower()
        output_dir = Path(self.config.train.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if task == "svr":
            trainer = SVRTrainer(self.config.train.svr, self.config.train.hpo, self.config.train.random_state)
            res = trainer.fit_evaluate(X, y, self.config.train.test_size, output_dir, self.config.train.metrics)
        else:
            raise NotImplementedError(f"Task not yet implemented: {task}")

        return res

