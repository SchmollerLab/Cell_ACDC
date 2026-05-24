"""Mixin dependency graph and parent assignments for guiWin MRO."""

from __future__ import annotations

# Layered parent map (downstream -> upstream). Edges only go from higher layers
# to lower layers so importing a mixin loads its parents first without cycles.
_RAW_MIXIN_PARENTS: dict[str, tuple[str, ...]] = {
    # Layer 0 — foundation
    "display_decorations": (),
    "geometry": (),
    "main_menu": (),
    "measurements": (),
    "canvas_tool": (),
    "whitelist": (),
    "combine": (),
    # Layer 1 — display / chrome helpers
    "image_display": ("display_decorations",),
    "actions": ("image_display",),
    "status_hover": ("image_display",),
    "main_toolbar": ("actions",),
    "quick_settings": ("actions",),
    # Layer 2 — workers / session
    "worker": ("image_display", "status_hover"),
    "session": ("image_display", "worker"),
    "app_shell": ("actions", "session"),
    "tool_activation": ("image_display", "session", "worker"),
    # Layer 3 — tools & canvas primitives
    "brush_tools": ("geometry", "image_display", "tool_activation"),
    "canvas_context_menu": ("image_display",),
    "canvas_selection": ("canvas_tool", "geometry", "brush_tools"),
    "label_editing": ("image_display", "session", "tool_activation"),
    "undo_redo": ("session", "label_editing"),
    "points_layers": ("image_display", "brush_tools"),
    "mode_controls": ("session", "tool_activation"),
    "annotation_display": ("image_display", "tool_activation", "mode_controls"),
    # Layer 4 — canvas interaction stack
    "canvas_drawing": (
        "canvas_selection",
        "brush_tools",
        "label_editing",
        "image_display",
    ),
    "canvas_events": (
        "geometry",
        "canvas_context_menu",
        "canvas_selection",
        "brush_tools",
        "label_editing",
        "image_display",
    ),
    "canvas_hover": ("canvas_events", "brush_tools", "tool_activation"),
    "curvature_tools": ("brush_tools", "tool_activation", "undo_redo"),
    "draw_clear_region": ("label_editing", "undo_redo", "image_display"),
    "label_transform_tools": ("brush_tools", "label_editing", "image_display"),
    "label_roi": ("session", "image_display", "brush_tools"),
    # Layer 5 — domain features
    "cell_cycle": ("session", "label_editing", "undo_redo", "image_display"),
    "tracking": ("session", "label_editing", "tool_activation", "undo_redo"),
    "deleted_rois": ("session", "cell_cycle", "tool_activation"),
    "object_properties": ("cell_cycle", "image_display", "tracking"),
    "segmentation": ("session", "image_display", "tool_activation"),
    "preprocessing": ("image_display", "worker", "session"),
    "saving": ("session", "worker", "app_shell"),
    "graphics": ("image_display", "points_layers", "worker"),
    "lineage_interactions": ("annotation_display", "tracking", "image_display"),
    "custom_annotations": ("annotation_display", "object_properties"),
    "magic_prompts": ("graphics", "session", "worker"),
    # Layer 6 — high-level orchestrators
    "frame_navigation": (
        "session",
        "graphics",
        "label_editing",
        "display_decorations",
    ),
    "data_loading": (
        "app_shell",
        "session",
        "tool_activation",
        "layout_controls",
    ),
    "image_controls": ("image_display", "frame_navigation"),
    "window_events": (
        "app_shell",
        "frame_navigation",
        "label_editing",
        "tool_activation",
    ),
    "layout_controls": ("image_controls", "window_events", "label_roi"),
    "canvas_right_image": ("canvas_drawing", "canvas_events", "canvas_context_menu"),
    "object_search": ("frame_navigation", "graphics", "session"),
    "object_cleanup": ("cell_cycle", "session", "image_display"),
    "seg_for_lost_ids": ("segmentation", "frame_navigation", "label_editing", "session"),
    "exporting": ("app_shell", "frame_navigation", "session"),
    "combine_worker": ("combine", "graphics", "preprocessing", "worker"),
}


def _ancestors(
    module: str,
    graph: dict[str, tuple[str, ...]],
    cache: dict[str, frozenset[str]],
) -> frozenset[str]:
    if module not in cache:
        seen: set[str] = set()
        for parent in graph.get(module, ()):
            seen.add(parent)
            seen |= _ancestors(parent, graph, cache)
        cache[module] = frozenset(seen)
    return cache[module]


def _reduce_mixin_parents(
    raw: dict[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    """Drop direct parents already inherited through another direct parent."""
    cache: dict[str, frozenset[str]] = {}
    reduced: dict[str, tuple[str, ...]] = {}
    for module, parents in raw.items():
        kept = tuple(
            parent
            for parent in parents
            if not any(
                parent != other and parent in _ancestors(other, raw, cache)
                for other in parents
            )
        )
        reduced[module] = kept
    return reduced


MIXIN_PARENTS = _reduce_mixin_parents(_RAW_MIXIN_PARENTS)

MODULE_TO_CLASS: dict[str, str] = {
    "actions": "Actions",
    "annotation_display": "AnnotationDisplay",
    "app_shell": "AppShell",
    "brush_tools": "BrushTools",
    "canvas_context_menu": "CanvasContextMenu",
    "canvas_drawing": "CanvasDrawing",
    "canvas_events": "CanvasEvents",
    "canvas_hover": "CanvasHover",
    "canvas_right_image": "CanvasRightImage",
    "canvas_selection": "CanvasSelection",
    "canvas_tool": "CanvasTool",
    "cell_cycle": "CellCycle",
    "combine": "CombineGui",
    "combine_worker": "CombineWorker",
    "curvature_tools": "CurvatureTools",
    "custom_annotations": "CustomAnnotations",
    "data_loading": "DataLoading",
    "deleted_rois": "DeletedRois",
    "display_decorations": "DisplayDecorations",
    "draw_clear_region": "DrawClearRegion",
    "exporting": "Exporting",
    "frame_navigation": "FrameNavigation",
    "geometry": "Geometry",
    "graphics": "Graphics",
    "image_controls": "ImageControls",
    "image_display": "ImageDisplay",
    "label_editing": "LabelEditing",
    "label_roi": "LabelRoi",
    "label_transform_tools": "LabelTransformTools",
    "layout_controls": "LayoutControls",
    "lineage_interactions": "LineageInteractions",
    "magic_prompts": "MagicPrompts",
    "main_menu": "MainMenu",
    "main_toolbar": "MainToolbar",
    "measurements": "Measurements",
    "mode_controls": "ModeControls",
    "object_cleanup": "ObjectCleanup",
    "object_properties": "ObjectProperties",
    "object_search": "ObjectSearch",
    "points_layers": "PointsLayers",
    "preprocessing": "Preprocessing",
    "quick_settings": "QuickSettings",
    "saving": "Saving",
    "seg_for_lost_ids": "SegForLostIds",
    "segmentation": "Segmentation",
    "session": "Session",
    "status_hover": "StatusHover",
    "tool_activation": "ToolActivation",
    "tracking": "Tracking",
    "undo_redo": "UndoRedo",
    "whitelist": "WhitelistGui",
    "window_events": "WindowEvents",
    "worker": "Worker",
}

MODULE_FILE: dict[str, str] = {
    "combine": "combine",
    "combine_worker": "combine",
}


def class_name(module: str) -> str:
    return MODULE_TO_CLASS[module]


def file_module(module: str) -> str:
    return MODULE_FILE.get(module, module)


def guiwin_roots() -> list[str]:
    """Modules listed directly on guiWin (not inherited via another root)."""
    all_parents = {p for ps in MIXIN_PARENTS.values() for p in ps}
    roots = [m for m in MODULE_TO_CLASS if m not in all_parents]
    # combine is parent of combine_worker
    roots = [m for m in roots if m != "combine"]

    order = [
        "whitelist",
        "layout_controls",
        "data_loading",
        "canvas_right_image",
        "canvas_hover",
        "window_events",
        "frame_navigation",
        "graphics",
        "lineage_interactions",
        "custom_annotations",
        "magic_prompts",
        "object_search",
        "object_cleanup",
        "seg_for_lost_ids",
        "exporting",
        "combine_worker",
        "curvature_tools",
        "draw_clear_region",
        "label_transform_tools",
        "deleted_rois",
        "cell_cycle",
        "tracking",
        "segmentation",
        "preprocessing",
        "saving",
        "object_properties",
        "annotation_display",
        "mode_controls",
        "main_toolbar",
        "quick_settings",
        "main_menu",
        "measurements",
        "canvas_events",
        "canvas_drawing",
        "canvas_selection",
        "canvas_context_menu",
        "brush_tools",
        "canvas_tool",
        "label_editing",
        "label_roi",
        "tool_activation",
        "session",
        "worker",
        "app_shell",
        "points_layers",
        "image_controls",
        "image_display",
        "status_hover",
        "actions",
        "undo_redo",
        "geometry",
        "display_decorations",
    ]
    rank = {m: i for i, m in enumerate(order)}
    return sorted(roots, key=lambda m: rank.get(m, 999))


def guiwin_classes() -> list[str]:
    return [class_name(m) for m in guiwin_roots()]


def import_cycles() -> list[list[str]]:
    """Detect import cycles in the parent graph (child imports parent modules)."""
    graph = MIXIN_PARENTS
    mods = set(MODULE_TO_CLASS)
    cycles = []
    path: list[str] = []
    visited: set[str] = set()
    stack: set[str] = set()

    def dfs(node: str) -> None:
        if node in stack:
            cycles.append(path[path.index(node) :] + [node])
            return
        if node in visited:
            return
        visited.add(node)
        stack.add(node)
        path.append(node)
        for parent in graph.get(node, ()):
            dfs(parent)
        path.pop()
        stack.remove(node)

    for mod in mods:
        dfs(mod)
    return cycles
