#!/usr/bin/env python3
"""Split god files into packages while preserving public import paths."""

from __future__ import annotations

import ast
import re
import shutil
import textwrap
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELLACDC = ROOT / "cellacdc"


def extract_nodes(source: str) -> tuple[str, list[tuple[str, str, int, int]]]:
    """Return preamble and (name, kind, start, end) for each top-level def/class."""
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
    preamble = "".join(lines[:preamble_end])
    return preamble, nodes


def slice_nodes(source: str, nodes: list[tuple[str, str, int, int]], names: set[str]) -> str:
    lines = source.splitlines(keepends=True)
    chunks: list[str] = []
    for name, _kind, start, end in nodes:
        if name in names:
            chunks.append("".join(lines[start - 1 : end]))
    return "\n\n".join(chunks)


def write_module(
    path: Path,
    doc: str,
    preamble: str,
    body: str,
    *,
    siblings: set[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if siblings is not None:
        preamble = fix_preamble_imports(preamble, siblings)
    content = f'"""{doc}"""\n\n{preamble.rstrip()}\n\n{body.rstrip()}\n'
    path.write_text(content)


def assign_myutils(name: str) -> str:
    rules: list[tuple[str, str]] = [
        ("logging", r"log|Logger"),
        ("paths", r"path|folder|dir|recent|trim_path|explorer|filemaneger|gdrive|acdc_data|pos_folder|images_folder|PosStatus|pos_status"),
        ("install", r"install|gpu|pytorch|torch|java|javabridge|conda|pip|package|mamba|upgrade_javabridge|java_exists|download_java|check_install"),
        ("dataframe", r"df_|dataframe|acdc_df|ctc|reset_index|format_ID|cca_col|are_acdc_dfs|fix_acdc_df"),
        ("version", r"version|git|branch|date_from|info_version|salute|cellpose.*version|second_version"),
        ("models", r"model|download|Tracker|tracker|segm_params|init_tracker|ArgSpec|parse_model|insertModel|getModel|promptable|ModelArg|IntensityImgRequired"),
        ("qt", r"widget|Qt|Q[A-Z]|retain|cli_multi_choice|testQcore"),
        ("io", r"bytes|Memory|browse_docs|save_response|read_|write_|open_url|browse_url"),
        ("text", r"tooltip|instruction|html|text|string|trim|annot|elided|fstring|append_text|show_in_file"),
    ]
    for module, pat in rules:
        if re.search(pat, name):
            return module
    return "misc"


def assign_worker(name: str) -> str:
    mapping = {
        "worker_exception_handler": "_base",
        "workerLogger": "_base",
        "signals": "_base",
        "BaseWorkerUtil": "_base",
        "SimpleWorker": "_base",
        "AutoPilotWorker": "gui",
        "FindNextNewIdWorker": "gui",
        "StoreGuiStateWorker": "io",
        "AutoSaveWorker": "io",
        "LazyLoader": "io",
        "loadDataWorker": "io",
        "saveDataWorker": "io",
        "MoveTempFilesWorker": "io",
        "MigrateUserProfileWorker": "io",
        "relabelSequentialWorker": "io",
        "segmWorker": "segm",
        "segmVideoWorker": "segm",
        "SegForLostIDsWorker": "segm",
        "PostProcessSegmWorker": "segm",
        "MagicPromptsWorker": "segm",
        "FillHolesInSegWorker": "segm",
        "DelObjectsOutsideSegmROIWorker": "segm",
        "LabelRoiWorker": "segm",
        "CreateConnected3Dsegm": "segm",
        "trackingWorker": "tracking",
        "TrackSubCellObjectsWorker": "tracking",
        "ApplyTrackInfoWorker": "tracking",
        "ToSymDivWorker": "tracking",
        "CopyAllLostObjectsWorker": "tracking",
        "ComputeMetricsWorker": "metrics",
        "ComputeMetricsMultiChannelWorker": "metrics",
        "ConcatAcdcDfsWorker": "metrics",
        "ConcatSpotmaxDfsWorker": "metrics",
        "CountObjectsInSegm": "metrics",
        "GenerateMotherBudTotalTableWorker": "metrics",
        "CcaIntegrityCheckerWorker": "metrics",
        "reapplyDataPrepWorker": "data_prep",
        "DataPrepSaveBkgrDataWorker": "data_prep",
        "DataPrepCropWorker": "data_prep",
        "RestructMultiPosWorker": "data_prep",
        "RestructMultiTimepointsWorker": "data_prep",
        "ImagesToPositionsWorker": "data_prep",
        "CustomPreprocessWorkerGUI": "data_prep",
        "CombineChannelsWorkerGUI": "data_prep",
        "CustomPreprocessWorkerUtil": "data_prep",
        "CombineChannelsWorkerUtil": "data_prep",
        "SaveProcessedDataWorker": "data_prep",
        "SaveCombinedChannelsWorker": "data_prep",
        "FucciPreprocessWorker": "data_prep",
        "AlignDataWorker": "alignment",
        "AlignWorker": "alignment",
        "FromImajeJroiToSegmNpzWorker": "util",
        "ToImajeJroiWorker": "util",
        "ToObjCoordsWorker": "util",
        "Stack2DsegmTo3Dsegm": "util",
        "ResizeUtilWorker": "util",
        "FilterObjsFromCoordsTable": "util",
        "ApplyImageFilterWorker": "util",
        "ScreenRecorderWorker": "util",
    }
    return mapping.get(name, "util")


def assign_widget(name: str) -> str:
    if "Toolbar" in name or "ToolButton" in name or name in {
        "ToolBarSeparator",
        "ToolBar",
        "rightClickToolButton",
    }:
        return "toolbars"
    canvas_markers = (
        "pg.",
        "Plot",
        "ImageItem",
        "ROI",
        "Histogram",
        "Gradient",
        "Scatter",
        "Contour",
        "ScaleBar",
        "ImShow",
        "Ghost",
        "Ruler",
        "RectItem",
        "ColorButton",
        "LabelItem",
        "MainPlot",
        "MouseCursor",
        "ScrollBar",
        "sliderWithSpinBox",
    )
    if any(m in name for m in canvas_markers) or name in {
        "ContourItem",
        "BaseScatterPlotItem",
        "CustomAnnotationScatterPlotItem",
        "ScatterPlotItem",
        "myLabelItem",
        "PolyLineROI",
        "ZoomROI",
        "DelROI",
        "PlotCurveItem",
        "BaseGradientEditorItemImage",
        "BaseGradientEditorItemLabels",
        "baseHistogramLUTitem",
        "myHistogramLUTitem",
        "overlayLabelsGradientWidget",
        "labelsGradientWidget",
        "BaseImageItem",
        "BaseLabelsImageItem",
        "OverlayImageItem",
        "ParentImageItem",
        "ChildImageItem",
        "labImageItem",
        "labelledQScrollbar",
        "navigateQScrollBar",
        "linkedQScrollbar",
        "myColorButton",
        "ScrollBarWithNumericControl",
        "PointsScatterPlotItem",
        "LabelRoiCircularItem",
    }:
        return "canvas"
    return "controls"


def assign_dialog(name: str) -> str:
    if name in {"QBaseDialog", "ArgWidget"}:
        return "_base"
    if name in {"addCustomModelMessages", "addCustomPromptModelMessages"}:
        return "models"
    rules: list[tuple[str, str]] = [
        ("tracking", r"Tracker|Track|Cca|cca|editCca|lineage|ApplyTrack|MotherBud|SymDiv|manualSeparate|FindID|EditID|NumericEntry|swap|merge"),
        ("metadata", r"Metadata|metadata|XML|QDialogMetadata|filenameDialog|AppendText|EntriesWidget|ColumnNames|CropZ|CropTrange|CropT|Zslice|MultiTimePoint|TreeSelector|TreesSelector|MultiList|selectPositions|OrderableList|SelectFolders|OverlayLabels|AutoSaveInterval"),
        ("preprocess", r"PreProcess|CombineChannels|Fucci|ResizeUtil|InitFiji|ImageJRois|randomWalker|PostProcess|Threshold|Crop|Formula|DataPrepSubCrops|stopFrame|startStop|FutureFrames|FunctionParams|TestSegm|wandTolerance"),
        ("measurements", r"Metric|Measurement|combineMetrics|SetMeasurements|ComputeMetrics|GenerateMother|ApplyTrackTable|SelectFeatures|CombineFeatures"),
        ("export", r"Export|Video|Timestamp|ScaleBar|ViewText|pdDataFrame|ViewCcaTable|ObjectCount|Screen|Logo|ShortcutEditor"),
        ("models", r"Model|downloadModel|SelectPromptable|SelectModel|InstallPyTorch|Bayesian|DeltaTracker|CellACDCTracker|Promptable|QDialogModelParams|QInput|ChangeUserProfile|SelectAcdcDf|Restore"),
    ]
    for module, pat in rules:
        if re.search(pat, name):
            return module
    return "general"


def fix_preamble_imports(preamble: str, siblings: set[str]) -> str:
    """Rewrite cellacdc-root imports for package submodules."""
    out: list[str] = []
    for line in preamble.splitlines(keepends=True):
        newline = "\n" if line.endswith("\n") else ""
        stripped = line.rstrip("\n")
        m = re.match(r"^(\s*)from \. import (.+)$", stripped)
        if m:
            indent, rest = m.groups()
            out.append(f"{indent}from .. import {rest}{newline}")
            continue
        m = re.match(r"^(\s*)from \.(\S+) import (.+)$", stripped)
        if m:
            indent, module, rest = m.groups()
            top = module.split(".", 1)[0]
            if top in siblings:
                out.append(line)
            else:
                out.append(f"{indent}from ..{module} import {rest}{newline}")
            continue
        out.append(line)
    return "".join(out)


def inject_cross_imports(pkg_dir: Path) -> None:
    """Wire sibling symbols: top imports for bases, trailing imports for calls."""
    assign: dict[str, str] = {}
    for p in pkg_dir.glob("*.py"):
        if p.name == "__init__.py":
            continue
        for node in ast.parse(p.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                assign[node.name] = p.stem

    for p in sorted(pkg_dir.glob("*.py")):
        if p.name == "__init__.py":
            continue
        mod = p.stem
        source = p.read_text()
        tree = ast.parse(source)

        top_needed: dict[str, set[str]] = defaultdict(set)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                for sub in ast.walk(base):
                    if isinstance(sub, ast.Name) and sub.id in assign and assign[sub.id] != mod:
                        top_needed[assign[sub.id]].add(sub.id)

        trailing_needed: dict[str, set[str]] = defaultdict(set)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id not in assign or assign[node.id] == mod:
                    continue
                src = assign[node.id]
                if node.id in top_needed.get(src, set()):
                    continue
                trailing_needed[src].add(node.id)

        if not top_needed and not trailing_needed:
            continue

        def render_imports(needed: dict[str, set[str]], prefix: str) -> str:
            lines: list[str] = []
            for src_mod in sorted(needed):
                names = sorted(needed[src_mod])
                lines.append(f"{prefix}from .{src_mod} import (")
                for name in names:
                    lines.append(f"{prefix}    {name},")
                lines.append(f"{prefix})")
            return "\n".join(lines) + ("\n\n" if lines else "")

        lines = source.splitlines(keepends=True)
        first_def = next(
            node.lineno
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        top_block = render_imports(top_needed, "")
        trailing_block = render_imports(trailing_needed, "")
        if trailing_block:
            trailing_block = "\n# Sibling imports (deferred to avoid import cycles)\n" + trailing_block

        new_source = (
            "".join(lines[: first_def - 1])
            + top_block
            + "".join(lines[first_def - 1 :])
            + trailing_block
        )
        p.write_text(new_source)


def split_package(
    src_file: Path,
    pkg_dir: Path,
    assign_fn,
    module_doc: str,
    *,
    delete_src: bool = True,
    shim_file: Path | None = None,
    shim_import_from: str | None = None,
) -> dict[str, list[str]]:
    source = src_file.read_text()
    preamble, nodes = extract_nodes(source)
    groups: dict[str, set[str]] = defaultdict(set)
    for name, _kind, _s, _e in nodes:
        groups[assign_fn(name)].add(name)

    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True)

    exported: dict[str, list[str]] = {}
    sibling_stems = set(groups)
    for module, names in sorted(groups.items()):
        body = slice_nodes(source, nodes, names)
        if not body.strip():
            continue
        out = pkg_dir / f"{module}.py"
        write_module(
            out,
            f"{module_doc}: {module}.",
            preamble,
            body,
            siblings=sibling_stems,
        )
        exported[module] = sorted(names)

    init_lines = [
        f'"""{module_doc}."""',
        "",
    ]
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
    (pkg_dir / "__init__.py").write_text("\n".join(init_lines) + "\n")

    inject_cross_imports(pkg_dir)

    if delete_src:
        src_file.unlink()

    if shim_file is not None:
        imp = shim_import_from or pkg_dir.name
        shim = textwrap.dedent(
            f'''\
            """Compatibility shim; implementation lives in {imp}/."""

            from .{imp} import *  # noqa: F403
            '''
        )
        shim_file.write_text(shim)

    return exported


def split_widgets(src_file: Path, pkg_dir: Path) -> None:
    """widgets.py becomes a package that also re-exports components/."""
    source = src_file.read_text()
    # Keep import block through component re-exports as package preamble.
    marker = "\n\n\n\nclass ContourItem"
    idx = source.find(marker)
    if idx == -1:
        raise RuntimeError("Could not locate widgets split marker")
    header = source[: idx + 2]
    body_source = source[idx + 2 :]
    _empty, nodes = extract_nodes(body_source)

    groups: dict[str, set[str]] = defaultdict(set)
    for name, _kind, _s, _e in nodes:
        groups[assign_widget(name)].add(name)

    if pkg_dir.exists():
        shutil.rmtree(pkg_dir)
    pkg_dir.mkdir(parents=True)

    exported: dict[str, list[str]] = {}
    sibling_stems = set(groups)
    for module, names in sorted(groups.items()):
        chunk = slice_nodes(body_source, nodes, names)
        if not chunk.strip():
            continue
        write_module(
            pkg_dir / f"{module}.py",
            f"GUI widgets: {module}.",
            header,
            chunk,
            siblings=sibling_stems,
        )
        exported[module] = sorted(names)

    init_parts = [
        '"""GUI widgets package (controls, canvas, toolbars) + components re-exports."""',
        "",
        "from ..components.palette import *  # noqa: F403",
        "from ..components.progress import *  # noqa: F403",
        "from ..components.buttons import *  # noqa: F403",
        "from ..components.layout import *  # noqa: F403",
        "from ..components.inputs_basic import *  # noqa: F403",
        "from ..components.path_controls import *  # noqa: F403",
        "from ..components.lists import *  # noqa: F403",
        "from ..components.base import QBaseWindow, QBaseDialog  # noqa: F401",
        "",
    ]
    all_names: list[str] = []
    for module in sorted(exported):
        names = exported[module]
        init_parts.append(f"from .{module} import (")
        for name in names:
            init_parts.append(f"    {name},")
            all_names.append(name)
        init_parts.append(")")
        init_parts.append("")

    init_parts.append("__all__ = [")
    for name in all_names:
        init_parts.append(f'    "{name}",')
    init_parts.append("]")
    (pkg_dir / "__init__.py").write_text("\n".join(init_parts) + "\n")
    inject_cross_imports(pkg_dir)
    src_file.unlink()


def main() -> None:
    split_package(
        CELLACDC / "myutils.py",
        CELLACDC / "myutils",
        assign_myutils,
        "Cell-ACDC utility helpers",
        delete_src=True,
    )
    split_package(
        CELLACDC / "workers.py",
        CELLACDC / "workers",
        assign_worker,
        "Background Qt workers",
        delete_src=True,
    )
    split_widgets(CELLACDC / "widgets.py", CELLACDC / "widgets")
    split_package(
        CELLACDC / "apps.py",
        CELLACDC / "dialogs",
        assign_dialog,
        "Cell-ACDC dialog windows",
        delete_src=False,
        shim_file=CELLACDC / "apps.py",
        shim_import_from="dialogs",
    )
    print("Split complete.")


if __name__ == "__main__":
    main()
