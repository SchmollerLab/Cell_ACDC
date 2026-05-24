"""Background Qt workers: _base."""

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

from .. import load, utils, core, prompts, printl, config, segm_re_pattern, io
from .. import transformation, measurements, cca_functions
from ..path import copy_or_move_tree
from .. import features, plot
from .. import core
from .. import cca_df_colnames, lineage_tree_cols, default_annot_df
from .. import cca_df_colnames_with_tree
from .. import cli
from ..tools import resize
from .. import segm_utils

DEBUG = False

def worker_exception_handler(func):
    @wraps(func)
    def run(self):
        try:
            func(self)
        except Exception as error:
            printl(traceback.format_exc())
            try:
                self.dataQ.clear()
            except Exception as err:
                pass

            # Some workers have both self.critical and self.signals.critical
            # errors but only one of them is connected --> emit both just
            # in case
            try:
                self.critical.emit((self, error))
            except Exception as err:
                self.signals.critical.emit((self, error))

            try:
                self.signals.critical.emit((self, error))
            except Exception as err:
                self.critical.emit((self, error))

            try:
                self.mutex.unlock()
            except Exception as err:
                pass

    return run


class workerLogger:
    def __init__(self, sigProcess):
        self.sigProcess = sigProcess

    def log(self, message, level="INFO"):
        try:
            self.sigProcess.emit(str(message), level)
        except Exception as err:
            print(message, level)
            try:
                traceback_format = traceback.format_exc()
                print(traceback_format)
            except Exception as err:
                pass
            printl(err)
        finally:
            pass

    def info(self, message):
        self.log(message, level="INFO")

    def warning(self, message):
        self.log(message, level="WARNING")

    def exception(self, message):
        self.log(message, level="EXCEPTION")


class signals(QObject):
    progress = Signal(str, object)
    finished = Signal(object)
    initProgressBar = Signal(int)
    progressBar = Signal(int)
    critical = Signal(object)
    dataIntegrityWarning = Signal(str)
    dataIntegrityCritical = Signal()
    sigLoadingFinished = Signal()
    sigLoadingNewChunk = Signal(object)
    resetInnerPbar = Signal(int)
    progress_tqdm = Signal(int)
    signal_close_tqdm = Signal()
    create_tqdm = Signal(int)
    innerProgressBar = Signal(int)
    sigPermissionError = Signal(str, object)
    sigSelectSegmFiles = Signal(object, object)
    sigSelectAcdcOutputFiles = Signal(object, object, str, bool, bool)
    sigSelectSpotmaxRun = Signal(object, object, object, str, bool, bool)
    sigSetMeasurements = Signal(object)
    sigInitAddMetrics = Signal(object, object)
    sigUpdatePbarDesc = Signal(str)
    sigComputeVolume = Signal(int, object)
    sigAskStopFrame = Signal(object)
    sigWarnMismatchSegmDataShape = Signal(object)
    sigErrorsReport = Signal(dict, dict, dict)
    sigMissingAcdcAnnot = Signal(dict)
    sigRecovery = Signal(object)
    sigInitInnerPbar = Signal(int)
    sigUpdateInnerPbar = Signal(int)
    sigSelectFile = Signal(str, str, str)
    sigAskCopyCca = Signal(str)
    sigSelectFilesWithText = Signal(str, object, str, object)
    sigAskRunNow = Signal(object)


class BaseWorkerUtil(QObject):
    progressBar = Signal(int, int, float)

    def __init__(self, mainWin):
        QObject.__init__(self)
        self.signals = signals()
        self.abort = False
        self.skipExp = False
        self.logger = workerLogger(self.signals.progress)
        self.mutex = QMutex()
        self.waitCond = QWaitCondition()
        self.mainWin = mainWin

    def emitSelectSegmFiles(self, exp_path, pos_foldernames):
        self.mutex.lock()
        self.signals.sigSelectSegmFiles.emit(exp_path, pos_foldernames)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    def emitSelectFilesWithText(self, exp_path, pos_foldernames, with_text, ext=None):
        self.mutex.lock()
        self.signals.sigSelectFilesWithText.emit(
            exp_path, pos_foldernames, with_text, ext
        )
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    def emitSelectFile(self, start_dir, caption="", filters="All files (*.)"):
        self.mutex.lock()
        self.signals.sigSelectFile.emit(start_dir, caption, filters)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    def emitSelectAcdcOutputFiles(
        self,
        exp_path,
        pos_foldernames,
        infoText="",
        allowSingleSelection=False,
        multiSelection=True,
    ):
        self.mutex.lock()
        self.signals.sigSelectAcdcOutputFiles.emit(
            exp_path, pos_foldernames, infoText, allowSingleSelection, multiSelection
        )
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    def emitSelectSpotmaxRun(
        self,
        exp_path,
        pos_foldernames,
        all_runs,
        infoText="",
        allowSingleSelection=True,
        multiSelection=True,
    ):
        self.mutex.lock()
        self.signals.sigSelectSpotmaxRun.emit(
            exp_path,
            pos_foldernames,
            all_runs,
            infoText,
            allowSingleSelection,
            multiSelection,
        )
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort


class SimpleWorker(QObject):
    def __init__(self, posData, func, func_args=None, func_kwargs=None):
        QObject.__init__(self)
        self.posData = posData
        self.signals = signals()
        self.output = {}

        if func_args is None:
            func_args = []

        if func_kwargs is None:
            func_kwargs = {}

        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs
        self.posData = posData

    @worker_exception_handler
    def run(self):
        self.result = self.func(self.posData, *self.func_args, **self.func_kwargs)
        self.signals.finished.emit(self.output)
