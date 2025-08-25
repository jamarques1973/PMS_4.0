"""Code generator for PMS 4.0.0.

Writes parsed artifacts into Python modules under the generated package.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

from .parser import Section, CodeArtifact


def _sanitize_identifier(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "section"
    if text[0].isdigit():
        text = f"s_{text}"
    return text


def write_generated_modules(sections: List[Section], out_dir: str | Path) -> List[Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for section in sections:
        module_name = f"sec_{section.index:04d}_{_sanitize_identifier(section.name) or 'section'}"
        target = out_path / f"{module_name}.py"
        with target.open("w", encoding="utf-8") as f:
            f.write("# Auto-generated from PMS 3.6.0 by PMS 4.0.0\n")
            f.write(f"# Section {section.index}: {section.name}\n\n")
            for art in section.artifacts:
                f.write(f"# Artifact: {art.kind} {art.name}\n")
                f.write(art.source)
                f.write("\n\n")
        written.append(target)
    return written

