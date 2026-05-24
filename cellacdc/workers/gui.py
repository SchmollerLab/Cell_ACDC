"""Background Qt workers: gui."""

import re
import os
import shutil
import time
import json
import concurrent.futures
from functools import partial
from collections import defaultdict, deque
import itertools

from typing import Union, List, Dict, Callable, Any, Tuple, Iterable

from functools import wraps
import numpy as np
import pandas as pd
import h5py
import traceback

import skimage.io
import skimage.measure
import skimage.exposure

import queue

from tqdm import tqdm

from qtpy.QtCore import Signal, QObject, QMutex, QWaitCondition

from cellacdc import html_utils

from .. import load, myutils, core, prompts, printl, config, segm_re_pattern, io
from .. import transformation, measurements, cca_functions
from ..path import copy_or_move_tree
from .. import features, plot
from .. import core
from .. import cca_df_colnames, lineage_tree_cols, default_annot_df
from .. import cca_df_colnames_with_tree
from .. import cli
from ..utils import resize
from .. import segm_utils

DEBUG = False

class AutoPilotWorker(QObject):
    finished = Signal()
    critical = Signal(object)
    progress = Signal(str, object)
    sigStarted = Signal()
    sigStopTimer = Signal()

    def __init__(self, guiWin):
        QObject.__init__(self)
        self.logger = workerLogger(self.progress)
        self.guiWin = guiWin
        self.app = guiWin.app
        # self.timer = timer

    def timerCallback(self):
        pass

    def stop(self):
        self.sigStopTimer.emit()
        self.finished.emit()

    def run(self):
        self.sigStarted.emit()


class FindNextNewIdWorker(QObject):
    def __init__(self, posData, guiWin):
        QObject.__init__(self)
        self.signals = signals()
        self.logger = workerLogger(self.signals.progress)
        self.posData = posData
        self.guiWin = guiWin

    @worker_exception_handler
    def run(self):
        prev_IDs = None
        next_frame_i = -1
        for frame_i, data_dict in enumerate(self.posData.allData_li):
            lab = data_dict["labels"]
            rp = data_dict["regionprops"]
            IDs = data_dict["IDs"]
            if lab is None:
                lab = self.posData.segm_data[frame_i]
                rp = skimage.measure.regionprops(lab)
                IDs = [obj.label for obj in rp]

            if prev_IDs is None:
                prev_IDs = IDs
                continue

            newIDs = [ID for ID in IDs if ID not in prev_IDs]
            if newIDs:
                next_frame_i = frame_i
                break
            prev_IDs = IDs

        self.signals.finished.emit(next_frame_i)

# Sibling imports (deferred to avoid import cycles)
from ._base import (
    signals,
    workerLogger,
    worker_exception_handler,
)

