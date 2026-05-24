"""Mixin bundles for composable GUI variants.

Each bundle lists the mixins declared directly on the window class. Upstream
parents are inherited automatically through the mixin dependency graph.
"""

from __future__ import annotations

# Minimal shell: logging, session settings, window chrome helpers.
BASIC_GUI_ROOTS: tuple[str, ...] = ("AppShell",)

# Load data and visualize images / labels (no annotation canvas stack).
VISUALIZATION_GUI_ROOTS: tuple[str, ...] = (
    "DataLoading",
    "MainMenu",
    "MainToolbar",
    "Saving",
    "Measurements",
    "ObjectSearch",
    "Exporting",
    "Preprocessing",
    "AnnotationDisplay",
    "QuickSettings",
    "UndoRedo",
    "CombineGui",
)

# Segmentation and annotation (no lineage tree, custom annotations, or measurements).
SEGMENTATION_GUI_ROOTS: tuple[str, ...] = (
    "WhitelistGui",
    "DataLoading",
    "CanvasRightImage",
    "CanvasHover",
    "MagicPrompts",
    "ObjectSearch",
    "SegForLostIds",
    "Exporting",
    "CombineWorker",
    "CurvatureTools",
    "DrawClearRegion",
    "LabelTransformTools",
    "DeletedRois",
    "Saving",
    "MainToolbar",
    "QuickSettings",
    "MainMenu",
    "AnnotationDisplay",
)
