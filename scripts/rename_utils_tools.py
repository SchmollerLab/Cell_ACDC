#!/usr/bin/env python3
"""Rename cellacdc.utils -> tools and cellacdc.utils -> utils in source files."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules"}

# Phase 1: batch-tool package (old utils -> tools). Run before utils -> utils.
TOOLS_PATTERNS: list[tuple[str, str]] = [
    (r"\bfrom \.\.utils import resize\b", "from ..tools import resize"),
    (r"\bfrom \.\.utils import base\b", "from ..tools import base"),
    (r"\bfrom \.\.utils\.", "from ..tools."),
    (r"\bfrom \.utils\.", "from .tools."),
    (r"\bfrom \.utils import", "from .tools import"),
    (r"\bfrom cellacdc\.utils\.", "from cellacdc.tools."),
    (r"\bfrom cellacdc\.utils import", "from cellacdc.tools import"),
    (r'"cellacdc/tools/', '"cellacdc/tools/'),
    (r"'cellacdc/tools/", "'cellacdc/tools/"),
    (r"\bcellacdc/utils/", "cellacdc/tools/"),
]

# Phase 2: helper package (utils -> utils).
UTILS_PATTERNS: list[tuple[str, str]] = [
    (r"\bmyutils\b", "utils"),
    (r'"cellacdc/utils/', '"cellacdc/tools/'),
    (r"'cellacdc/utils/", "'cellacdc/tools/"),
    (r"\bcellacdc/utils/", "cellacdc/tools/"),
    (r"\bcellacdc\.utils\b", "cellacdc.utils"),
]

# Phase 3: same-package imports inside tools/
TOOLS_INTERNAL: list[tuple[str, str]] = [
    (r"\bfrom \.\.tools import base\b", "from . import base"),
]


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if path.suffix != ".py":
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def apply_patterns(text: str, patterns: list[tuple[str, str]]) -> str:
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    return text


def fix_tools_package() -> None:
    tools_dir = ROOT / "cellacdc" / "tools"
    if not tools_dir.is_dir():
        return
    for path in tools_dir.rglob("*.py"):
        text = path.read_text()
        updated = apply_patterns(text, TOOLS_INTERNAL)
        if updated != text:
            path.write_text(updated)


def main() -> None:
    for path in iter_files():
        text = path.read_text()
        updated = apply_patterns(text, TOOLS_PATTERNS)
        updated = apply_patterns(updated, UTILS_PATTERNS)
        if updated != text:
            path.write_text(updated)
    fix_tools_package()
    print("Import rewrites complete.")


if __name__ == "__main__":
    main()
