"""Variable dictionary generation for PMS 4.0.0."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from ..core.parser import Section


def _section_to_module_name(index: int, name: str) -> str:
    import re

    def _sanitize_identifier(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"[^a-z0-9_]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        if not text:
            text = "section"
        if text[0].isdigit():
            text = f"s_{text}"
        return text

    return f"sec_{index:04d}_{_sanitize_identifier(name) or 'section'}"


def build_variable_dictionary(sections: List[Section], namespace: Dict[str, object]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    assigned_to_section: Dict[str, str] = {}
    for section in sections:
        sec_label = f"{section.index}:{section.name}"
        module_name = _section_to_module_name(section.index, section.name)
        for art in section.artifacts:
            if art.assignments:
                for var_name in art.assignments:
                    assigned_to_section.setdefault(var_name, f"{sec_label}|{module_name}")
            if art.local_assignments and art.kind == "function":
                for var_name in art.local_assignments:
                    rows.append({
                        "variable": var_name,
                        "description": "",
                        "type": "unknown",
                        "scope": "local",
                        "section": sec_label,
                        "module": module_name,
                        "component": art.name,
                    })

    for name, value in namespace.items():
        if name.startswith("__") and name.endswith("__"):
            continue
        type_name = type(value).__name__
        section_info = assigned_to_section.get(name, "unknown|unknown")
        sec, mod = section_info.split("|", 1) if "|" in section_info else (section_info, "unknown")
        rows.append({
            "variable": name,
            "description": "",
            "type": type_name,
            "scope": "global",
            "section": sec,
            "module": mod,
            "component": "",
        })

    df = pd.DataFrame(rows, columns=["variable", "description", "type", "scope", "section", "module", "component"]).sort_values(["scope", "variable"]) 
    return df

