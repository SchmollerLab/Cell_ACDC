"""Cell-ACDC utility helpers: logging."""

import os
import re
import ast

import typing
from typing import Literal, List, Callable, Tuple, Dict

import pathlib
import difflib
import sys
import platform
import tempfile
import shutil
import traceback
import logging
import datetime
import time
import subprocess
import importlib
from uuid import uuid4
from importlib import import_module
from math import pow, ceil, floor
from functools import wraps, partial
from collections import namedtuple, Counter
from tqdm import tqdm
import requests
import zipfile
import json
import numpy as np
import pandas as pd
import skimage
import inspect

import traceback
import itertools
from packaging import version as packaging_version

from natsort import natsorted

import tifffile
import skimage.io
import skimage.measure

from .. import GUI_INSTALLED, KNOWN_EXTENSIONS, is_conda_env

from .. import core, load
from .. import html_utils, is_linux, is_win, is_mac, issues_url, is_mac_arm64
from .. import cellacdc_path, printl, acdc_fiji_path, logs_path, acdc_ffmpeg_path
from .. import user_profile_path, recentPaths_path
from .. import models_list_file_path, models_path
from .. import promptable_models_list_file_path, promptable_models_path
from .. import github_home_url
from .. import try_input_install_package
from .. import _warnings
from .. import urls
from .. import qrc_resources_path
from .. import settings_folderpath
from ..segmenters._cellpose_base import min_target_versions_cp

if GUI_INSTALLED:
    from qtpy.QtWidgets import QMessageBox
    from qtpy.QtCore import Signal, QObject, QCoreApplication

    from .. import widgets, apps
    from .. import config

ArgSpec = namedtuple("ArgSpec", ["name", "default", "type", "desc", "docstring"])

def get_logs_path():
    return logs_path


class Logger(logging.Logger):
    def __init__(self, module="base", name="cellacdc-logger", level=logging.DEBUG):
        super().__init__(f"{name}-{module}", level=level)
        self._stdout = sys.stdout
        self._stderr = StdErr(logger=self)
        sys.stderr = self._stderr
        self._levelToName = {
            50: "CRITICAL",
            40: "ERROR",
            30: "WARNING",
            20: "INFO",
            10: "DEBUG",
            0: "NOTSET",
        }

    def write(self, text, log_to_file=True, write_to_stdout=True):
        """Capture print statements, print to terminal and log text to
        the open log file

        Parameters
        ----------
        text : str
            Text to log
        log_to_file : bool, optional
            If True, call `info` method with `text`. Default is True
        """
        if write_to_stdout:
            self._stdout.write(text)

        if not log_to_file:
            return

        if text == "\n":
            return

        if not text:
            return

        self.debug(text)

    def close(self):
        for handler in self.handlers:
            handler.close()
            self.removeHandler(handler)
        sys.stdout = self._stdout
        self._stderr.close()

    def __del__(self):
        sys.stdout = self._stdout
        self._stderr.close()

    def info(self, text, *args, **kwargs):
        super().info(text, *args, **kwargs)
        try:
            self.write(f"{text}\n", log_to_file=False)
        except TypeError:
            # Sometimes the logger is patched (e.g., by spotiflow), which
            # triggers the TypeError because the patching function does not have
            # log_to_file argument
            self.write(f"{text}\n")

    def warning(self, text, *args, **kwargs):
        super().warning(text, *args, **kwargs)
        try:
            self.write(f"[WARNING]: {text}\n", log_to_file=False)
        except TypeError:
            # Sometimes the logger is patched (e.g., by spotiflow), which
            # triggers the TypeError because the patching function does not have
            # log_to_file argument
            self.write(f"[WARNING]: {text}\n")

    def error(self, text, *args, write_traceback=True, **kwargs):
        super().error(text, *args, **kwargs)
        self.write(traceback.format_exc())
        try:
            self.write(f"[ERROR]: {text}\n", log_to_file=False)
        except TypeError:
            # Sometimes the logger is patched (e.g., by spotiflow), which
            # triggers the TypeError because the patching function does not have
            # log_to_file argument
            self.write(f"[ERROR]: {text}\n")

    def plain(self, text, write_to_stdout=False):
        orig_formatters = [handler.formatter for handler in self.handlers]
        for handler in self.handlers:
            handler.setFormatter(logging.Formatter("%(message)s"))
        self.write(text, write_to_stdout=write_to_stdout)
        for handler in self.handlers:
            handler.setFormatter(orig_formatters.pop(0))

    def critical(self, text, *args, **kwargs):
        super().critical(text, *args, **kwargs)
        try:
            self.write(f"[CRITICAL]: {text}\n", log_to_file=False)
        except TypeError:
            # Sometimes the logger is patched (e.g., by spotiflow), which
            # triggers the TypeError because the patching function does not have
            # log_to_file argument
            self.write(f"[CRITICAL]: {text}\n")

    def exception(self, text, *args, write_traceback=True, **kwargs):
        super().exception(text, *args, **kwargs)
        self.write(traceback.format_exc())
        try:
            self.write(f"[ERROR]: {text}\n", log_to_file=False)
        except TypeError:
            # Sometimes the logger is patched (e.g., by spotiflow), which
            # triggers the TypeError because the patching function does not have
            # log_to_file argument
            self.write(f"[ERROR]: {text}\n")

    def log(self, level, text):
        if not isinstance(level, int):
            printl(level, text, type(level), type(text), sep="\n")
        super().log(level, text)
        levelName = self._levelToName.get(level, "INFO")
        getattr(self, levelName.lower())(text)

    def flush(self):
        self._stdout.flush()


def delete_older_log_files(logs_path):
    if not os.path.exists(logs_path):
        return

    log_files = os.listdir(logs_path)
    for log_file in log_files:
        if not log_file.endswith(".log"):
            continue

        log_filepath = os.path.join(logs_path, log_file)
        try:
            mtime = os.path.getmtime(log_filepath)
        except Exception as err:
            continue

        mdatetime = datetime.datetime.fromtimestamp(mtime)
        days = (datetime.datetime.now() - mdatetime).days
        if days < 7:
            continue

        try:
            os.remove(log_filepath)
        except Exception as err:
            continue


def _log_system_info(logger, log_path, is_cli=False, also_spotmax=False):
    logger.info(f'Initialized log file "{log_path}"')

    info_txt = get_info_version_text(is_cli=is_cli)

    logger.info(info_txt)

    if not also_spotmax:
        return

    from spotmax.utils import get_info_version_text as smax_info

    smax_info_txt = smax_info(include_platform=False)
    logger.info(smax_info_txt)


def setupLogger(module="base", logs_path=None, caller="Cell-ACDC"):
    if logs_path is None:
        logs_path = get_logs_path()

    logger = Logger(module=module)
    sys.stdout = logger

    delete_older_log_files(logs_path)
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)

    date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    id = uuid4()
    log_filename = f"{date_time}_{module}_{id}_stdout.log"
    log_path = os.path.join(logs_path, log_filename)

    output_file_handler = logging.FileHandler(log_path, mode="w")

    # Format your logs (optional)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s:\n"
        "------------------------\n"
        "%(message)s\n"
        "------------------------\n",
        datefmt="%d-%m-%Y, %H:%M:%S",
    )
    output_file_handler.setFormatter(formatter)

    logger.addHandler(output_file_handler)

    _log_system_info(logger, log_path, also_spotmax=caller != "Cell-ACDC")

    # if module == 'gui' and GUI_INSTALLED:
    #     qt_handler = widgets.QtHandler()
    #     qt_handler.setFormatter(logging.Formatter("%(message)s"))
    #     logger.addHandler(qt_handler)

    return logger, logs_path, log_path, log_filename


def log_segm_params(
    model_name,
    init_params,
    segm_params,
    logger_func=print,
    preproc_recipe=None,
    apply_post_process=False,
    standard_postprocess_kwargs=None,
    custom_postprocess_features=None,
):
    init_params_format = [
        f"  * {option} = {value}" for option, value in init_params.items()
    ]
    init_params_format = "\n".join(init_params_format)

    segm_params_format = [
        f"  * {option} = {value}" for option, value in segm_params.items()
    ]
    segm_params_format = "\n".join(segm_params_format)

    preproc_recipe_format = None
    if preproc_recipe is not None:
        preproc_recipe_format = []
        for s, step in enumerate(preproc_recipe):
            preproc_recipe_format.append(f"  * Step {s + 1}")
            method = step["method"]
            preproc_recipe_format.append(f"     - Method: {method}")
            for option, value in step["kwargs"].items():
                preproc_recipe_format.append(f"     - {option}: {value}")
        preproc_recipe_format = "\n".join(preproc_recipe_format)

    standard_postproc_format = None
    if apply_post_process and standard_postprocess_kwargs is not None:
        standard_postproc_format = [
            f"  * {option} = {value}"
            for option, value in standard_postprocess_kwargs.items()
        ]
        standard_postproc_format = "\n".join(standard_postproc_format)

    custom_postproc_format = None
    if apply_post_process and custom_postprocess_features is not None:
        custom_postproc_format = [
            f"  * {feature} = ({low}, {high})"
            for feature, (low, high) in custom_postprocess_features.items()
        ]
        custom_postproc_format = "\n".join(custom_postproc_format)

    separator = "-" * 100
    params_format = (
        f"{separator}\n"
        f"Model name: {model_name}\n\n"
        "Preprocessing recipe:\n\n"
        f"{preproc_recipe_format}\n\n"
        "Initialization parameters:\n\n"
        f"{init_params_format}\n\n"
        "Segmentation parameters:\n\n"
        f"{segm_params_format}\n\n"
        "Post-processing:\n\n"
        f"{standard_postproc_format}\n\n"
        "Custom post-processing:\n\n"
        f"{custom_postproc_format}\n"
        f"{separator}"
    )
    logger_func(params_format)

# Sibling imports (deferred to avoid import cycles)
from .misc import (
    StdErr,
)
from .version import (
    get_info_version_text,
)

