"""Notebook parser for PMS 4.0.0.

This module parses the PMS 3.6.0 notebook into structured sections and
code artifacts which can be transformed into layered modules.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


HEADING_RE = re.compile(r"^(#+)\s+(.*)")


@dataclass
class CodeArtifact:
    kind: str  # "function", "class", "assign", "exec"
    name: str
    source: str
    section: str
    hash: str
    assignments: List[str] = field(default_factory=list)
    local_assignments: List[str] = field(default_factory=list)


@dataclass
class Section:
    name: str
    index: int
    artifacts: List[CodeArtifact] = field(default_factory=list)

    def add_artifact(self, artifact: CodeArtifact) -> None:
        self.artifacts.append(artifact)


class NotebookParser:
    """Parses .ipynb files and extracts structured sections and code artifacts."""

    def __init__(self, notebook_path: str | Path) -> None:
        self.notebook_path = Path(notebook_path)
        self.sections: List[Section] = []
        self._artifact_hashes: Dict[str, CodeArtifact] = {}

    def load(self) -> Dict:
        with self.notebook_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _hash_source(source: str) -> str:
        return hashlib.sha256(source.strip().encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_source(source: str) -> str:
        lines = source.splitlines()
        if lines and lines[0].strip().startswith("%%"):
            lines = lines[1:]
        normalized = []
        for line in lines:
            if line.strip().startswith("%") or line.strip().startswith("!"):
                continue
            normalized.append(line)
        return "\n".join(normalized).strip("\n")

    def parse(self) -> List[Section]:
        nb = self.load()
        current_section = Section(name="root", index=0)
        self.sections = [current_section]

        for cell in nb.get("cells", []):
            cell_type = cell.get("cell_type")
            if cell_type == "markdown":
                text = "".join(cell.get("source", []))
                match = HEADING_RE.match(text.strip())
                if match:
                    level = len(match.group(1))
                    title = match.group(2).strip()
                    current_section = Section(name=title, index=len(self.sections))
                    self.sections.append(current_section)
                continue

            if cell_type != "code":
                continue

            source = "".join(cell.get("source", []))
            if not source.strip():
                continue
            normalized = self._normalize_source(source)
            if not normalized:
                continue

            try:
                node = ast.parse(normalized)
            except SyntaxError:
                artifact = CodeArtifact(
                    kind="exec",
                    name=f"exec_{len(current_section.artifacts)}",
                    source=normalized,
                    section=current_section.name,
                    hash=self._hash_source(normalized),
                )
                self._record_artifact(artifact, current_section)
                continue

            def collect_local_assignments(fn_node: ast.FunctionDef) -> List[str]:
                names: List[str] = []
                class AssignVisitor(ast.NodeVisitor):
                    def visit_Assign(self, n):  # type: ignore[override]
                        for t in n.targets:
                            if isinstance(t, ast.Name):
                                names.append(t.id)
                            elif isinstance(t, (ast.Tuple, ast.List)):
                                for elt in t.elts:
                                    if isinstance(elt, ast.Name):
                                        names.append(elt.id)
                        self.generic_visit(n)
                    def visit_AnnAssign(self, n):  # type: ignore[override]
                        t = n.target
                        if isinstance(t, ast.Name):
                            names.append(t.id)
                        self.generic_visit(n)
                AssignVisitor().visit(fn_node)
                return sorted(set(names))

            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    fragment = ast.get_source_segment(normalized, child) or ""
                    name = child.name
                    local_vars = collect_local_assignments(child)
                    artifact = CodeArtifact(
                        kind="function",
                        name=name,
                        source=fragment,
                        section=current_section.name,
                        hash=self._hash_source(fragment),
                        local_assignments=local_vars,
                    )
                    self._record_artifact(artifact, current_section)
                elif isinstance(child, ast.ClassDef):
                    fragment = ast.get_source_segment(normalized, child) or ""
                    artifact = CodeArtifact(
                        kind="class",
                        name=child.name,
                        source=fragment,
                        section=current_section.name,
                        hash=self._hash_source(fragment),
                    )
                    self._record_artifact(artifact, current_section)
                elif isinstance(child, (ast.Assign, ast.AnnAssign)):
                    fragment = ast.get_source_segment(normalized, child) or ""
                    assigned_names: List[str] = []
                    def _collect_names(target):
                        if isinstance(target, ast.Name):
                            assigned_names.append(target.id)
                        elif isinstance(target, (ast.Tuple, ast.List)):
                            for elt in target.elts:
                                _collect_names(elt)
                    if isinstance(child, ast.Assign):
                        for t in child.targets:
                            _collect_names(t)
                    elif isinstance(child, ast.AnnAssign) and child.target is not None:
                        _collect_names(child.target)
                    name_str = ",".join(assigned_names) if assigned_names else f"exec_{len(current_section.artifacts)}"
                    artifact = CodeArtifact(
                        kind="assign",
                        name=name_str,
                        source=fragment,
                        section=current_section.name,
                        hash=self._hash_source(fragment),
                        assignments=assigned_names,
                    )
                    self._record_artifact(artifact, current_section)
                else:
                    fragment = ast.get_source_segment(normalized, child) or ""
                    if not fragment.strip():
                        continue
                    artifact = CodeArtifact(
                        kind="exec",
                        name=f"exec_{len(current_section.artifacts)}",
                        source=fragment,
                        section=current_section.name,
                        hash=self._hash_source(fragment),
                    )
                    self._record_artifact(artifact, current_section)

        return self.sections

    def _record_artifact(self, artifact: CodeArtifact, section: Section) -> None:
        existing = self._artifact_hashes.get(artifact.hash)
        if existing is not None:
            return
        self._artifact_hashes[artifact.hash] = artifact
        section.add_artifact(artifact)

