"""Consistent report styling utilities for PMS 4.0.0."""

from __future__ import annotations

import pandas as pd


def style_dataframe(df: pd.DataFrame, theme: str = "light") -> pd.io.formats.style.Styler:
    styler = df.style.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#222" if theme == "dark" else "#f5f5f5"),
                    ("color", "#fff" if theme == "dark" else "#333"),
                    ("font-weight", "600"),
                ],
            },
            {
                "selector": "td",
                "props": [("padding", "6px 10px"), ("border", "1px solid #ddd")],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("border", "1px solid #ddd"),
                    ("font-family", "Inter, Roboto, system-ui, -apple-system"),
                    ("font-size", "13px"),
                ],
            },
        ]
    )
    return styler

