"""Cell-ACDC utility helpers: text."""

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

def get_trimmed_list(li: list, max_num_digits=10):
    if len(li) == 0:
        return "[]"

    tom_num_digits = sum([len(str(val)) for val in li])

    if tom_num_digits == 0:
        return f"[{', '.join(map(str, li))}]"

    avg_num_digits = tom_num_digits / len(li)
    max_num_vals = int(round(max_num_digits / avg_num_digits))

    if tom_num_digits > max_num_digits:
        front_vals = ceil(max_num_vals / 2)
        back_vals = max_num_vals // 2

        if front_vals + back_vals >= len(li):
            return f"[{', '.join(map(str, li))}]"

        li = li[:front_vals] + ["..."] + li[len(li) - back_vals :]

    return f"[{', '.join(map(str, li))}]"


def get_trimmed_dict(di: dict, max_num_digits=10):
    di_str = di.copy()
    total_num_digits = sum([len(str(key)) + len(str(val)) for key, val in di.items()])
    avg_num_digits = total_num_digits / len(di)
    max_num_vals = int(round(max_num_digits / avg_num_digits))
    if total_num_digits > max_num_digits:
        keys = list(di_str.keys())
        for key in keys[max_num_vals:-max_num_vals]:
            del di_str[key]
        di_str[keys[max_num_vals]] = "..."
    return f"[{', '.join([f'{key} -> {val}' for key, val in di_str.items()])}]"


def get_number_fstring_formatter(dtype, precision=4):
    if np.issubdtype(dtype, np.integer):
        return "d"
    else:
        return f".{precision}f"


def elided_text(text, max_len=50, elid_idx=None):
    if len(text) <= max_len:
        return text

    if elid_idx is None:
        elid_idx = int(max_len / 2)
    if elid_idx >= max_len:
        elid_idx = max_len - 1
    idx1 = elid_idx
    idx2 = elid_idx - max_len
    text = f"{text[:idx1]}...{text[idx2:]}"
    return text


def get_show_in_file_manager_text():
    if is_mac:
        return "Reveal in Finder"
    elif is_linux:
        return "Show in File Manager"
    elif is_win:
        return "Show in File Explorer"


def append_text_filename(filename: str, text_to_append: str):
    filename_noext, ext = os.path.splitext(filename)
    filename_out = f"{filename_noext}{text_to_append}{ext}"
    return filename_out
