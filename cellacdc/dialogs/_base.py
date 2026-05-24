"""Cell-ACDC dialog windows: _base."""

import os
import sys
import re
from typing import Literal, Callable, Dict, Iterable, List, Tuple
import datetime
import pathlib
from collections import defaultdict
import zipfile
from heapq import nlargest
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle, PathPatch, Path
import numpy as np
import scipy.interpolate

try:
    import tkinter as tk
except Exception as err:
    pass

import cv2
import traceback
from itertools import combinations, permutations
from collections import namedtuple
from natsort import natsorted

# from MyWidgets import Slider, Button, MyRadioButtons
from skimage.measure import label, regionprops
from functools import partial
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.exposure
import skimage.draw
import skimage.registration
import skimage.color
import skimage.segmentation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import time
import sympy as sp
import json
import html

import pyqtgraph as pg

pg.setConfigOption("imageAxisOrder", "row-major")

from qtpy import QtCore
from qtpy.QtGui import (
    QIcon,
    QFontMetrics,
    QKeySequence,
    QFont,
    QRegularExpressionValidator,
    QCursor,
    QKeyEvent,
    QPixmap,
    QFont,
    QPalette,
    QMouseEvent,
    QColor,
)
from qtpy.QtCore import (
    Qt,
    QSize,
    QEvent,
    Signal,
    QEventLoop,
    QTimer,
    QRegularExpression,
)
from qtpy.QtWidgets import (
    QFileDialog,
    QApplication,
    QMainWindow,
    QMenu,
    QLabel,
    QToolBar,
    QScrollBar,
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QDialog,
    QFormLayout,
    QListWidget,
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QSizePolicy,
    QComboBox,
    QSlider,
    QGridLayout,
    QSpinBox,
    QToolButton,
    QTableView,
    QTextBrowser,
    QDoubleSpinBox,
    QScrollArea,
    QFrame,
    QProgressBar,
    QGroupBox,
    QRadioButton,
    QDockWidget,
    QMessageBox,
    QStyle,
    QPlainTextEdit,
    QSpacerItem,
    QTreeWidget,
    QTreeWidgetItem,
    QTextEdit,
    QSplashScreen,
    QAction,
    QListWidgetItem,
    QActionGroup,
    QHeaderView,
    QStyledItemDelegate,
)
import qtpy.compat

from .. import exception_handler
from .. import load, prompts, core, measurements, html_utils
from .. import is_mac, is_win, is_linux, settings_folderpath, config
from .. import preproc_recipes_path, segm_recipes_path, combine_channels_recipes_path
from .. import is_conda_env
from .. import printl
from .. import colors
from .. import issues_url
from .. import myutils
from .. import qutils
from .. import _palettes
from .. import base_cca_dict
from .. import widgets
from .. import user_profile_path, promptable_models_path, models_path
from .. import features
from .. import _core
from .. import _types
from .. import plot
from .. import urls
from ..acdc_regex import float_regex, is_alphanumeric_filename, to_alphanumeric
from .. import _base_widgets
from .. import io
from .. import cca_functions
from .. import path

POSITIVE_FLOAT_REGEX = float_regex(allow_negative=False)
TREEWIDGET_STYLESHEET = _palettes.TreeWidgetStyleSheet()
LISTWIDGET_STYLESHEET = _palettes.ListWidgetStyleSheet()
BACKGROUND_RGBA = _palettes.get_disabled_colors()["Button"]

font = QFont()
font.setPixelSize(12)
italicFont = QFont()
italicFont.setPixelSize(12)
italicFont.setItalic(True)

class ArgWidget:
    def __init__(
        self, name, type, widget, defaultVal, valueSetter, valueGetter, changeSig=None
    ):
        self.name = name
        self.type = type
        self.widget = widget
        self.defaultVal = defaultVal
        self.valueSetter = valueSetter
        self.valueGetter = valueGetter
        if changeSig is not None:
            self.changeSig = changeSig


class QBaseDialog(_base_widgets.QBaseDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
