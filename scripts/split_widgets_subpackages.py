#!/usr/bin/env python3
"""Split widgets/canvas.py, controls.py, and toolbars.py into subpackages."""

from __future__ import annotations

import ast
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WIDGETS = ROOT / "cellacdc" / "widgets"

CANVAS_MODULES: dict[str, set[str]] = {
    "histogram": {
        "BaseGradientEditorItemImage",
        "BaseGradientEditorItemLabels",
        "baseHistogramLUTitem",
        "myHistogramLUTitem",
        "overlayLabelsGradientWidget",
        "labelsGradientWidget",
        "myColorButton",
    },
    "rois": {"ROI", "ZoomROI", "DelROI", "PolyLineROI"},
    "plot_items": {
        "ContourItem",
        "BaseScatterPlotItem",
        "CustomAnnotationScatterPlotItem",
        "ScatterPlotItem",
        "myLabelItem",
        "LabelRoiCircularItem",
        "PlotCurveItem",
        "MainPlotItem",
        "GhostContourItem",
        "RulerPlotItem",
        "PointsScatterPlotItem",
        "RectItem",
        "LabelItem",
        "ScaleBar",
    },
    "images": {
        "BaseImageItem",
        "BaseLabelsImageItem",
        "OverlayImageItem",
        "ParentImageItem",
        "ChildImageItem",
        "labImageItem",
        "GhostMaskItem",
        "_ImShowImageItem",
    },
    "imshow": {"ImShow", "ImShowPlotItem"},
    "scrollbars": {
        "MouseCursor",
        "labelledQScrollbar",
        "navigateQScrollBar",
        "linkedQScrollbar",
        "sliderWithSpinBox",
        "ScrollBarWithNumericControl",
    },
}

CONTROLS_MODULES: dict[str, set[str]] = {
    "dialogs": {
        "QDialogListbox",
        "myMessageBox",
        "view_visualcpp_screenshot",
        "installJavaDialog",
        "selectTrackerGUI",
        "warnVisualCppRequired",
    },
    "inputs": {
        "ExpandableListBox",
        "QClickableLabel",
        "QCenteredComboBox",
        "AlphaNumericComboBox",
        "mySpinBox",
        "ShortcutLineEdit",
        "CenteredDoubleSpinbox",
        "readOnlyDoubleSpinbox",
        "readOnlySpinbox",
        "DoubleSpinBox",
        "SpinBox",
        "ReadOnlyLineEdit",
        "FloatLineEdit",
        "IntLineEdit",
        "LineEdit",
        "SearchLineEdit",
        "VectorLineEdit",
        "OddSpinBox",
        "KeySequenceFromText",
        "ComboBox",
        "WhitelistLineEdit",
        "highlightableQWidgetAction",
    },
    "metrics": {
        "_metricsQGBox",
        "channelMetricsQGBox",
        "PixelSizeGroupbox",
        "objPropsQGBox",
        "objIntesityMeasurQGBox",
        "SetMeasurementsGroupBox",
    },
    "forms": {
        "selectStartStopFrames",
        "formWidget",
        "CheckboxesGroupBox",
        "guiTabControl",
        "CopiableCommandWidget",
        "LabelsWidget",
        "SamInputPointsWidget",
        "FontSizeWidget",
        "RangeSelector",
        "PreProcessingSelector",
        "RescaleImageJroisGroupbox",
        "TimeWidget",
        "YeazV2SelectModelNameCombobox",
        "AutoSaveIntervalWidget",
        "CheckableWidget",
        "PostProcessSegmSlider",
        "PostProcessSegmSpinbox",
    },
    "panels": {
        "statusBarPermanentLabel",
        "listWidget",
        "OrderableListWidget",
        "KeptObjectIDsList",
        "Toggle",
        "ToggleTerminalButton",
        "expandCollapseButton",
        "ToggleVisibilityButton",
        "ToggleVisibilityCheckBox",
        "FeatureSelectorButton",
        "CheckableSpinBoxWidgets",
        "Label",
        "LatexLabel",
        "SwitchPlaneCombobox",
        "TimestampItem",
        "CheckableAction",
    },
}

TOOLBARS_MODULES: dict[str, set[str]] = {
    "_base": {
        "ToolBarSeparator",
        "ToolBar",
        "rightClickToolButton",
        "ToolButtonCustomColor",
        "GradientToolButton",
        "ToolButtonTextIcon",
        "customAnnotToolButton",
        "PointsLayerToolButton",
        "OverlayChannelToolButton",
        "SavePointsLayerButton",
        "ManualTrackingToolBar",
        "ManualBackgroundToolBar",
    },
    "feature": {
        "CopyLostObjectToolbar",
        "DrawClearRegionToolbar",
        "WhitelistIDsToolbar",
        "MagicPromptsToolbar",
        "PointsLayersToolbar",
        "PromptableModelPointsLayerToolbar",
        "OverlayToolbar",
        "HighlightedIDToolbar",
        "WandControlsToolbar",
    },
}


def extract_nodes(source: str) -> tuple[str, list[tuple[str, str, int, int]]]:
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    nodes: list[tuple[str, str, int, int]] = []
    first_start = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            end = getattr(node, "end_lineno", node.lineno)
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            nodes.append((node.name, kind, node.lineno, end))
            if first_start is None:
                first_start = node.lineno
    preamble_end = first_start - 1 if first_start else len(lines)
    return "".join(lines[:preamble_end]), nodes


def slice_nodes(source: str, nodes: list[tuple[str, str, int, int]], names: set[str]) -> str:
    lines = source.splitlines(keepends=True)
    chunks: list[str] = []
    for name, _kind, start, end in nodes:
        if name in names:
            chunks.append("".join(lines[start - 1 : end]))
    return "\n\n".join(chunks)


def clean_preamble(preamble: str) -> str:
    """Drop sibling-package imports; they are regenerated after the split."""
    out: list[str] = []
    skip = False
    for line in preamble.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if re.match(r"^from \.(canvas|controls|toolbars) import ", stripped):
            skip = stripped.rstrip().endswith("(")
            continue
        if skip:
            if ")" in stripped:
                skip = False
            continue
        if stripped.startswith("# Sibling imports"):
            break
        out.append(line)
    return "".join(out)


def deepen_imports(preamble: str) -> str:
    """widgets/<area>/<mod>.py needs one more parent level than widgets/<area>.py."""
    out: list[str] = []
    for line in preamble.splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        stripped = line.rstrip("\n")
        m = re.match(r"^(\s*)from \.\. import (.+)$", stripped)
        if m:
            indent, rest = m.groups()
            out.append(f"{indent}from ... import {rest}{newline}")
            continue
        m = re.match(r"^(\s*)from \.\.(\S+) import (.+)$", stripped)
        if m:
            indent, module, rest = m.groups()
            out.append(f"{indent}from ...{module} import {rest}{newline}")
            continue
        out.append(line)
    return "".join(out)


def write_module(path: Path, doc: str, preamble: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f'"""{doc}"""\n\n{preamble.rstrip()}\n\n{body.rstrip()}\n'
    path.write_text(content)


def split_area(
    src: Path,
    dest: Path,
    modules: dict[str, set[str]],
    doc: str,
) -> dict[str, list[str]]:
    source = src.read_text()
    preamble, nodes = extract_nodes(source)
    preamble = deepen_imports(clean_preamble(preamble))

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    exported: dict[str, list[str]] = {}
    for module, names in sorted(modules.items()):
        body = slice_nodes(source, nodes, names)
        if not body.strip():
            raise RuntimeError(f"No body extracted for {dest.name}/{module}.py")
        write_module(dest / f"{module}.py", f"{doc}: {module}.", preamble, body)
        exported[module] = sorted(names)

    init_lines = [f'"""{doc}."""', ""]
    all_names: list[str] = []
    for module in sorted(exported):
        names = exported[module]
        init_lines.append(f"from .{module} import (")
        for name in names:
            init_lines.append(f"    {name},")
            all_names.append(name)
        init_lines.append(")")
        init_lines.append("")
    init_lines.append("__all__ = [")
    for name in all_names:
        init_lines.append(f'    "{name}",')
    init_lines.append("]")
    (dest / "__init__.py").write_text("\n".join(init_lines) + "\n")
    return exported


def inject_widget_imports() -> None:
    assign: dict[str, tuple[str, str]] = {}
    for area in ("canvas", "controls", "toolbars"):
        for path in (WIDGETS / area).glob("*.py"):
            if path.name == "__init__.py":
                continue
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    assign[node.name] = (area, path.stem)

    for area in ("canvas", "controls", "toolbars"):
        for path in sorted((WIDGETS / area).glob("*.py")):
            if path.name == "__init__.py":
                continue
            mod = path.stem
            source = path.read_text()
            tree = ast.parse(source)

            top_needed: dict[tuple[str, str], set[str]] = defaultdict(set)
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                for base in node.bases:
                    for sub in ast.walk(base):
                        if (
                            isinstance(sub, ast.Name)
                            and sub.id in assign
                            and assign[sub.id] != (area, mod)
                        ):
                            top_needed[assign[sub.id]].add(sub.id)

            trailing_needed: dict[tuple[str, str], set[str]] = defaultdict(set)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id not in assign or assign[node.id] == (area, mod):
                        continue
                    loc = assign[node.id]
                    if node.id in top_needed.get(loc, set()):
                        continue
                    trailing_needed[loc].add(node.id)

            if not top_needed and not trailing_needed:
                continue

            def render(needed: dict[tuple[str, str], set[str]], prefix: str) -> str:
                lines: list[str] = []
                for sub_area, sub_mod in sorted(needed):
                    names = sorted(needed[(sub_area, sub_mod)])
                    if sub_area == area:
                        import_from = f".{sub_mod}"
                    else:
                        import_from = f"..{sub_area}.{sub_mod}"
                    lines.append(f"{prefix}from {import_from} import (")
                    for name in names:
                        lines.append(f"{prefix}    {name},")
                    lines.append(f"{prefix})")
                return "\n".join(lines) + ("\n\n" if lines else "")

            lines = source.splitlines(keepends=True)
            first_def = next(
                n.lineno
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            )
            top_block = render(top_needed, "")
            trailing_block = render(trailing_needed, "")
            if trailing_block:
                trailing_block = (
                    "\n# Cross-module imports (deferred to avoid import cycles)\n"
                    + trailing_block
                )
            updated = (
                "".join(lines[: first_def - 1])
                + top_block
                + "".join(lines[first_def - 1 :])
                + trailing_block
            )
            path.write_text(updated)


def rebuild_widgets_init() -> None:
    """Keep widgets/__init__.py as compatibility barrel."""
    header = '''"""GUI widgets package (canvas, controls, toolbars) + components re-exports."""

from ..components.palette import *  # noqa: F403
from ..components.progress import *  # noqa: F403
from ..components.buttons import *  # noqa: F403
from ..components.layout import *  # noqa: F403
from ..components.inputs_basic import *  # noqa: F403
from ..components.path_controls import *  # noqa: F403
from ..components.lists import *  # noqa: F403
from ..components.base import QBaseWindow, QBaseDialog  # noqa: F401

'''
    all_names: list[str] = []
    import_blocks: list[str] = []
    for area in ("canvas", "controls", "toolbars"):
        init_path = WIDGETS / area / "__init__.py"
        tree = ast.parse(init_path.read_text())
        names = [
            node.id
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == area
            for alias in node.names
        ]
        # parse from __all__
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            names = [
                                elt.value
                                for elt in node.value.elts
                                if isinstance(elt, ast.Constant)
                            ]
        import_blocks.append(f"from .{area} import (")
        for name in names:
            import_blocks.append(f"    {name},")
            all_names.append(name)
        import_blocks.append(")")
        import_blocks.append("")

    body = header + "\n".join(import_blocks) + "\n__all__ = [\n"
    for name in all_names:
        body += f'    "{name}",\n'
    body += "]\n"
    (WIDGETS / "__init__.py").write_text(body)


def main() -> None:
    # Pull toolbar classes that were left in controls.py into toolbars/_base.
    controls_src = (WIDGETS / "controls.py").read_text()
    toolbars_src = (WIDGETS / "toolbars.py").read_text()
    _, controls_nodes = extract_nodes(controls_src)
    _, toolbars_nodes = extract_nodes(toolbars_src)
    controls_names = {n for n, _, _, _ in controls_nodes}
    for name in ("ManualTrackingToolBar", "ManualBackgroundToolBar", "SavePointsLayerButton"):
        if name in controls_names and name not in TOOLBARS_MODULES["_base"]:
            TOOLBARS_MODULES["_base"].add(name)

    split_area(WIDGETS / "canvas.py", WIDGETS / "canvas", CANVAS_MODULES, "Canvas widgets")
    split_area(
        WIDGETS / "controls.py",
        WIDGETS / "controls",
        CONTROLS_MODULES,
        "Composite controls",
    )
    split_area(
        WIDGETS / "toolbars.py",
        WIDGETS / "toolbars",
        TOOLBARS_MODULES,
        "Toolbars",
    )

    for fname in ("canvas.py", "controls.py", "toolbars.py"):
        (WIDGETS / fname).unlink()

    inject_widget_imports()
    rebuild_widgets_init()
    print("widgets/ subpackage split complete.")


if __name__ == "__main__":
    main()
