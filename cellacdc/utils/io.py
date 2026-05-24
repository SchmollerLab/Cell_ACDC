"""Cell-ACDC utility helpers: io."""

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

def _bytes_to_MB(size_bytes):
    factor = pow(2, -20)
    size_MB = round(size_bytes * factor)
    return size_MB


def _bytes_to_GB(size_bytes):
    factor = pow(2, -30)
    size_GB = round(size_bytes * factor, 2)
    return size_GB


def getMemoryFootprint(files_list):
    required_memory = sum(
        [48 if file.endswith(".h5") else os.path.getsize(file) for file in files_list]
    )
    return required_memory


def browse_url(url):
    import webbrowser

    webbrowser.open(url)


def browse_docs():
    browse_url(urls.docs_homepage)


def save_response_content(
    response, destination, file_size=None, model_name="cellpose", progress=None
):
    print(f"Downloading {model_name} to: {os.path.dirname(destination)}")
    CHUNK_SIZE = 32768

    # Download to a temp folder in user path
    temp_folder = pathlib.Path.home().joinpath(".acdc_temp")
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)
    temp_dst = os.path.join(temp_folder, os.path.basename(destination))
    if file_size is not None and progress is not None:
        progress.emit(file_size, -1)
    pbar = tqdm(
        total=file_size, unit="B", unit_scale=True, unit_divisor=1024, ncols=100
    )
    with open(temp_dst, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
                if progress is not None:
                    progress.emit(-1, len(chunk))
    pbar.close()

    # Move to destination and delete temp folder
    destination_dir = os.path.dirname(destination)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)
    shutil.move(temp_dst, destination)
    shutil.rmtree(temp_folder)
