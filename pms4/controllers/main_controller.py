"""Main orchestration controller for PMS 4.0.0."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from ..core.parser import NotebookParser
from ..core.executor import ExecutionEngine
from ..core.registry import REGISTRY
from ..utils.config import ParameterStore
from ..utils.logging_setup import get_logger
from ..core.generator import write_generated_modules
from ..reports.variable_dictionary import build_variable_dictionary


logger = get_logger(__name__)


@dataclass
class PMSController:
    params: ParameterStore

    def __post_init__(self) -> None:
        logger.info("Initializing PMSController with input notebook: %s", self.params.input_notebook_path)
        REGISTRY.set("params", self.params)

    def analyze_notebook(self) -> List[NotebookParser]:
        parser = NotebookParser(self.params.input_notebook_path)
        sections = parser.parse()
        logger.info("Parsed notebook: %d sections", len(sections))
        REGISTRY.set("sections", sections)
        # Generate modules for transparency and reuse
        written = write_generated_modules(sections, self.params.generated_package_dir)
        logger.info("Generated %d module files at %s", len(written), self.params.generated_package_dir)
        return sections

    def execute(self) -> None:
        sections = REGISTRY.get("sections")
        if not sections:
            sections = self.analyze_notebook()
        engine = ExecutionEngine()
        code_blocks: List[tuple[str, str]] = []
        for section in sections:
            for art in section.artifacts:
                code_blocks.append((f"{section.index}:{art.kind}:{art.name}", art.source))
        result = engine.exec_code_blocks(code_blocks)
        REGISTRY.set("namespace", result.namespace)
        REGISTRY.set("trace", result.trace_rows)
        logger.info("Executed %d blocks", len(code_blocks))
        # Build variable dictionary
        try:
            import pandas as pd  # noqa: F401
            var_df = build_variable_dictionary(sections, result.namespace)
            REGISTRY.set("variable_dictionary", var_df)
        except Exception as exc:
            logger.warning("Could not build variable dictionary: %s", exc)

