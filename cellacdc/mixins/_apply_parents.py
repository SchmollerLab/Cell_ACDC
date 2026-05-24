#!/usr/bin/env python3
"""Apply upstream mixin parents to mixin class definitions."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIXINS = ROOT / "mixins"
GUI = ROOT / "gui.py"


def load_graph():
    spec = importlib.util.spec_from_file_location(
        "cellacdc_mixins_graph", MIXINS / "_graph.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GRAPH = load_graph()
MIXIN_PARENTS = GRAPH.MIXIN_PARENTS
class_name = GRAPH.class_name
file_module = GRAPH.file_module
guiwin_classes = GRAPH.guiwin_classes
guiwin_roots = GRAPH.guiwin_roots

FILE_CLASSES: dict[str, list[tuple[str, str]]] = {
    "combine": [("combine", "CombineGui"), ("combine_worker", "CombineWorker")],
}


def parent_imports(module: str, parents: tuple[str, ...]) -> list[str]:
    lines = []
    seen: set[str] = set()
    child_file = file_module(module)
    for p in parents:
        fm = file_module(p)
        if fm == child_file:
            continue
        cn = class_name(p)
        key = (fm, cn)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"from .{fm} import {cn}")
    return lines


def rewrite_class_bases(content: str, cls: str, parents: tuple[str, ...]) -> str:
    parent_names = [class_name(p) for p in parents]
    if parent_names:
        bases = ", ".join(parent_names)
        content = re.sub(
            rf"^class {cls}\([^)]*\)\:",
            f"class {cls}({bases}):",
            content,
            count=1,
            flags=re.MULTILINE,
        )
        content = re.sub(
            rf"^class {cls}\:",
            f"class {cls}({bases}):",
            content,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        content = re.sub(
            rf"^class {cls}\([^)]*\)\:",
            f"class {cls}:",
            content,
            count=1,
            flags=re.MULTILINE,
        )
    return content


def inject_imports(content: str, import_lines: list[str]) -> str:
    if not import_lines:
        # strip stale mixin imports
        return strip_old_mixin_imports(content)
    block = "\n".join(import_lines)
    content = strip_old_mixin_imports(content)
    idx = content.find("\n\nclass ")
    if idx == -1:
        raise ValueError("Could not find class definition anchor")
    return content[:idx] + "\n" + block + content[idx:]


def strip_old_mixin_imports(content: str) -> str:
    lines = []
    for line in content.splitlines(keepends=True):
        if re.match(r"from \.[a-z_]+ import [A-Z]", line):
            continue
        lines.append(line)
    return "".join(lines)


def apply_file(module: str, cls: str) -> None:
    parents = MIXIN_PARENTS.get(module, ())
    fp = MIXINS / f"{file_module(module)}.py"
    content = fp.read_text()
    content = rewrite_class_bases(content, cls, parents)
    content = inject_imports(content, parent_imports(module, parents))
    fp.write_text(content)
    print(f"  {module}({cls}) <- {[class_name(p) for p in parents]}")


def update_gui() -> None:
    classes = guiwin_classes()
    src = GUI.read_text()
    import_block = "from .mixins import (\n"
    import_block += "".join(f"    {c},\n" for c in classes)
    import_block += ")\n"
    src = re.sub(
        r"from \.mixins import \(\n.*?\n\)\n",
        import_block,
        src,
        count=1,
        flags=re.DOTALL,
    )
    bases = ",\n             ".join(classes)
    src = re.sub(
        r"class guiWin\(QMainWindow,\n(?:             .+\n)+?\):",
        f"class guiWin(QMainWindow,\n             {bases}):",
        src,
        count=1,
    )
    GUI.write_text(src)


def main() -> None:
    cycles = GRAPH.import_cycles()
    if cycles:
        raise SystemExit(f"Import cycles in mixin graph: {cycles}")
    for mod in sorted(MIXIN_PARENTS):
        if mod == "combine_worker":
            continue
        if mod in FILE_CLASSES:
            continue
        apply_file(mod, class_name(mod))
    for mod, cls in FILE_CLASSES["combine"]:
        apply_file(mod, cls)
    update_gui()
    print("\nguiWin roots:", guiwin_classes())


if __name__ == "__main__":
    main()
