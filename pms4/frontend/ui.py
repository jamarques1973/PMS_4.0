"""Voilà-compatible UI for PMS 4.0.0 using ipywidgets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import ipywidgets as widgets
from IPython.display import display
import pandas as pd

from ..controllers.main_controller import PMSController
from ..utils.config import ParameterStore
from ..utils.logging_setup import get_logger


logger = get_logger(__name__)


class PMSApp:
    """Builds the PMS UI and binds it to the controller."""

    def __init__(self, params: Optional[ParameterStore] = None) -> None:
        self.params = params or ParameterStore()
        self.controller = PMSController(self.params)
        self._build_ui()

    def _build_ui(self) -> None:
        self.title = widgets.HTML("<h2 style='margin:0'>PMS 4.0.0</h2>")

        self.load_btn = widgets.Button(description="Analizar", button_style="info", icon="search")
        self.run_btn = widgets.Button(description="Ejecutar", button_style="success", icon="play")
        self.save_btn = widgets.Button(description="Guardar Config", icon="save")

        self.input_path = widgets.Text(value=self.params.input_notebook_path, description="Notebook")
        self.theme_dd = widgets.Dropdown(options=["light", "dark"], value=self.params.theme, description="Tema")
        self.debug_cb = widgets.Checkbox(value=self.params.debug_enabled, description="Debug")

        self.log_out = widgets.Output(layout=dict(border="1px solid #ccc", height="200px", overflow="auto"))
        self.trace_out = widgets.Output(layout=dict(border="1px solid #ccc", height="200px", overflow="auto"))
        self.vars_out = widgets.Output(layout=dict(border="1px solid #ccc", height="300px", overflow="auto"))

        controls = widgets.HBox([self.load_btn, self.run_btn, self.save_btn])
        form = widgets.VBox([self.input_path, self.theme_dd, self.debug_cb])
        tabs = widgets.Tab(children=[self.log_out, self.trace_out, self.vars_out])
        tabs.set_title(0, "Log")
        tabs.set_title(1, "Trazas")
        tabs.set_title(2, "Variables")
        self.root = widgets.VBox([self.title, form, controls, tabs])

        self.load_btn.on_click(self._on_analyze)
        self.run_btn.on_click(self._on_execute)
        self.save_btn.on_click(self._on_save)

    def show(self) -> None:
        display(self.root)

    def _on_analyze(self, _):
        with self.log_out:
            print("Analizando notebook...")
        self.params.input_notebook_path = self.input_path.value
        sections = self.controller.analyze_notebook()
        with self.log_out:
            print(f"Secciones: {len(sections)}")

    def _on_execute(self, _):
        with self.log_out:
            print("Ejecutando artefactos...")
        self.controller.execute()
        with self.trace_out:
            from ..core.registry import REGISTRY

            rows = REGISTRY.get("trace") or []
            print(json.dumps(rows, indent=2, ensure_ascii=False))
        with self.vars_out:
            from ..core.registry import REGISTRY
            from ..reports.styling import style_dataframe
            var_df = REGISTRY.get("variable_dictionary")
            if isinstance(var_df, pd.DataFrame):
                display(style_dataframe(var_df, theme=self.theme_dd.value))
            else:
                print("No variable dictionary available.")

    def _on_save(self, _):
        self.params.input_notebook_path = self.input_path.value
        self.params.theme = self.theme_dd.value
        self.params.debug_enabled = self.debug_cb.value
        p = self.params.save_json()
        with self.log_out:
            print(f"Configuración guardada en: {p}")

