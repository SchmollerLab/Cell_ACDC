#!/usr/bin/env python3
"""Fix parent-package imports in split submodules (from . -> from ..)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "cellacdc"

PACKAGES: dict[str, set[str]] = {
    "utils": {
        "dataframe",
        "install",
        "io",
        "logging",
        "misc",
        "models",
        "paths",
        "qt",
        "text",
        "version",
    },
    "workers": {
        "_base",
        "alignment",
        "data_prep",
        "gui",
        "io",
        "metrics",
        "segm",
        "tracking",
        "util",
    },
    "widgets": {"canvas", "controls", "toolbars"},
    "dialogs": {
        "_base",
        "export",
        "general",
        "measurements",
        "metadata",
        "models",
        "preprocess",
        "tracking",
    },
}


def fix_line(line: str, siblings: set[str]) -> str:
    m = re.match(r"^(\s*)from \. import (.+)$", line)
    if m:
        indent, rest = m.groups()
        return f"{indent}from .. import {rest}"

    m = re.match(r"^(\s*)from \.(\S+) import (.+)$", line)
    if not m:
        return line
    indent, module, rest = m.groups()
    top = module.split(".", 1)[0]
    if top in siblings:
        return line
    return f"{indent}from ..{module} import {rest}"


def fix_file(path: Path, siblings: set[str]) -> bool:
    lines = path.read_text().splitlines(keepends=True)
    new_lines = [fix_line(line, siblings) for line in lines]
    if new_lines != lines:
        path.write_text("".join(new_lines))
        return True
    return False


def main() -> None:
    for pkg, siblings in PACKAGES.items():
        pkg_dir = ROOT / pkg
        changed = 0
        for path in sorted(pkg_dir.glob("*.py")):
            if path.name == "__init__.py":
                continue
            if fix_file(path, siblings):
                changed += 1
        print(f"{pkg}: fixed {changed} files")


if __name__ == "__main__":
    main()
