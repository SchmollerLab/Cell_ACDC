import os
import operator

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import pyqtgraph as pg
from qtpy.QtGui import QFont

from .. import config, settings_folderpath
from .. import _palettes

LINEEDIT_WARNING_STYLESHEET = _palettes.lineedit_warning_stylesheet()
LINEEDIT_INVALID_ENTRY_STYLESHEET = _palettes.lineedit_invalid_entry_stylesheet()
TREEWIDGET_STYLESHEET = _palettes.TreeWidgetStyleSheet()
LISTWIDGET_STYLESHEET = _palettes.ListWidgetStyleSheet()
BASE_COLOR = _palettes.base_color()
PROGRESSBAR_QCOLOR = _palettes.QProgressBarColor()
PROGRESSBAR_HIGHLIGHTEDTEXT_QCOLOR = _palettes.QProgressBarHighlightedTextColor()
TEXT_COLOR = _palettes.text_float_rgba()

font = QFont()
font.setPixelSize(12)

custom_cmaps_filepath = os.path.join(settings_folderpath, "custom_colormaps.ini")

str_to_operator_mapper = {"+": operator.add, "-": operator.sub}

sign_int_mapper = {"+": 1, "-": -1}


def removeHSVcmaps():
    hsv_cmaps = []
    for g, grad in pg.graphicsItems.GradientEditorItem.Gradients.items():
        if grad["mode"] == "hsv":
            hsv_cmaps.append(g)
    for g in hsv_cmaps:
        del pg.graphicsItems.GradientEditorItem.Gradients[g]


def renamePgCmaps():
    Gradients = pg.graphicsItems.GradientEditorItem.Gradients
    try:
        Gradients["hot"] = Gradients.pop("thermal")
    except KeyError:
        pass
    try:
        Gradients.pop("greyclip")
    except KeyError:
        pass


def _tab20gradient():
    cmap = plt.get_cmap("tab20")
    ticks = [(t, tuple([int(v * 255) for v in cmap(t)])) for t in np.linspace(0, 1, 20)]
    gradient = {"ticks": ticks, "mode": "rgb"}
    return gradient


def _tab10gradient():
    cmap = plt.get_cmap("tab10")
    ticks = [(t, tuple([int(v * 255) for v in cmap(t)])) for t in np.linspace(0, 1, 20)]
    gradient = {"ticks": ticks, "mode": "rgb"}
    return gradient


def getCustomGradients(name="image"):
    CustomGradients = {}
    if not os.path.exists(custom_cmaps_filepath):
        return CustomGradients

    cp = config.ConfigParser()
    cp.read(custom_cmaps_filepath)
    for section in cp.sections():
        if not section.startswith(f"{name}"):
            continue

        cmap_name = section[len(f"{name}.") :]
        CustomGradients[cmap_name] = {"ticks": [], "mode": "rgb"}
        for option in cp.options(section):
            value = cp[section][option]
            pos, *rgb = value.split(",")
            rgb = tuple([int(c) for c in rgb])
            pos = float(pos)
            CustomGradients[cmap_name]["ticks"].append((pos, rgb))
    return CustomGradients


def addGradients():
    Gradients = pg.graphicsItems.GradientEditorItem.Gradients
    Gradients["cividis"] = {
        "ticks": [
            (0.0, (0, 34, 78, 255)),
            (0.25, (66, 78, 108, 255)),
            (0.5, (124, 123, 120, 255)),
            (0.75, (187, 173, 108, 255)),
            (1.0, (254, 232, 56, 255)),
        ],
        "mode": "rgb",
    }
    Gradients["cool"] = {
        "ticks": [(0.0, (0, 255, 255, 255)), (1.0, (255, 0, 255, 255))],
        "mode": "rgb",
    }
    Gradients["sunset"] = {
        "ticks": [
            (0.0, (71, 118, 148, 255)),
            (0.4, (222, 213, 141, 255)),
            (0.8, (229, 184, 155, 255)),
            (1.0, (240, 127, 97, 255)),
        ],
        "mode": "rgb",
    }
    Gradients["tab20"] = _tab20gradient()
    Gradients["tab10"] = _tab10gradient()
    cmaps = {}
    for name, gradient in Gradients.items():
        ticks = gradient["ticks"]
        colors = [tuple([v / 255 for v in tick[1]]) for tick in ticks]
        cmaps[name] = LinearSegmentedColormap.from_list(name, colors, N=256)
    return cmaps, Gradients


nonInvertibleCmaps = ["cool", "sunset", "bipolar"]

renamePgCmaps()
removeHSVcmaps()
cmaps, Gradients = addGradients()
GradientsLabels = Gradients.copy()
GradientsImage = Gradients.copy()
