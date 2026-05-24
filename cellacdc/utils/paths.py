"""Cell-ACDC utility helpers: paths."""

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

def get_pos_status_acdc(pos_path):
    images_path = os.path.join(pos_path, "Images")
    ls = listdir(images_path)
    for file in ls:
        if file.endswith("acdc_output.csv"):
            acdc_df_path = os.path.join(images_path, file)
            break
    else:
        return ""

    acdc_df = pd.read_csv(acdc_df_path)
    last_tracked_i = acdc_df["frame_i"].max()
    last_cca_i = 0
    if "cell_cycle_stage" in acdc_df.columns:
        cca_df = acdc_df[["frame_i", "cell_cycle_stage"]].dropna()
        last_cca_i = cca_df["frame_i"].max()
    if last_cca_i > 0:
        return (
            f" (last tracked frame = {last_tracked_i + 1}, "
            f"last annotated frame = {last_cca_i + 1})"
        )
    else:
        return f" (last tracked frame = {last_tracked_i + 1})"


def get_pos_status_spotmax(pos_path):
    spotmax_out_path = os.path.join(pos_path, "spotMAX_output")
    is_smax_out_present = "Yes" if os.path.exists(spotmax_out_path) else "No"
    if os.path.exists(spotmax_out_path):
        return " (SpotMAX output exists)"
    else:
        return ""


def get_pos_status(pos_path, caller: Literal["Cell-ACDC", "SpotMAX"] = "Cell-ACDC"):
    if caller == "Cell-ACDC":
        return get_pos_status_acdc(pos_path)

    if caller == "SpotMAX":
        return get_pos_status_spotmax(pos_path)


def get_gdrive_path():
    if is_win:
        return os.path.join(f"G:{os.sep}", "My Drive")
    elif is_mac:
        return os.path.join(
            "/Users/francesco.padovani/Library/CloudStorage/"
            "GoogleDrive-padovaf@tcd.ie/My Drive"
        )


def get_acdc_data_path():
    Cell_ACDC_path = os.path.dirname(cellacdc_path)
    return os.path.join(Cell_ACDC_path, "data")


def get_open_filemaneger_os_string():
    if is_win:
        return "Show in Explorer..."
    elif is_mac:
        return "Reveal in Finder..."
    elif is_linux:
        return "Show in File Manager..."


def trim_path(path, depth=3, start_with_dots=True):
    path_li = os.path.abspath(path).split(os.sep)
    rel_path = f"{f'{os.sep}'.join(path_li[-depth:])}"
    if start_with_dots:
        return f"...{os.sep}{rel_path}"
    else:
        return rel_path


def get_pos_foldernames(exp_path, check_if_is_sub_folder=False):
    if not check_if_is_sub_folder:
        ls = listdir(exp_path)
        pos_foldernames = [
            pos for pos in ls if is_pos_folderpath(os.path.join(exp_path, pos))
        ]
    else:
        folder_type = determine_folder_type(exp_path)
        is_pos_folder, is_images_folder, _ = folder_type
        if is_pos_folder:
            return [os.path.basename(exp_path)]
        elif is_images_folder:
            pos_path = os.path.dirname(exp_path)
            if is_pos_folderpath(pos_path):
                return [os.path.basename(pos_path)]
            else:
                return []
        else:
            return get_pos_foldernames(exp_path)
    return pos_foldernames


def get_images_folderpath(folderpath):
    if os.path.isfile(folderpath):
        folderpath = os.path.dirname(folderpath)

    if folderpath.endswith("Images"):
        return folderpath

    images_folderpath = os.path.join(folderpath, "Images")
    if os.path.exists(images_folderpath):
        return images_folderpath

    return ""


def store_custom_model_path(model_file_path):
    model_file_path = model_file_path.replace("\\", "/")
    model_name = os.path.basename(os.path.dirname(model_file_path))
    cp = config.ConfigParser()
    if os.path.exists(models_list_file_path):
        cp.read(models_list_file_path)
    if model_name not in cp:
        cp[model_name] = {}
    cp[model_name]["path"] = model_file_path
    with open(models_list_file_path, "w") as configFile:
        cp.write(configFile)


def store_custom_promptable_model_path(promptable_model_file_path):
    model_file_path = promptable_model_file_path.replace("\\", "/")
    model_name = os.path.basename(os.path.dirname(model_file_path))
    cp = config.ConfigParser()
    if os.path.exists(promptable_models_list_file_path):
        cp.read(promptable_models_list_file_path)
    if model_name not in cp:
        cp[model_name] = {}
    cp[model_name]["path"] = model_file_path
    with open(promptable_models_list_file_path, "w") as configFile:
        cp.write(configFile)


def listdir(path) -> List[str]:
    return natsorted(
        [
            f
            for f in os.listdir(path)
            if not f.startswith(".")
            and not f == "desktop.ini"
            and not f == "recovery"
            and not f.endswith(".new.npz")
        ]
    )


def get_examples_path(which):
    if which == "time_lapse_2D":
        foldername = "TimeLapse_2D"
        url = "https://hmgubox2.helmholtz-muenchen.de/index.php/s/KgJQtsQKZJnWZjL/download/TimeLapse_2D.zip"
        file_size = 45143552
    elif which == "snapshots_3D":
        foldername = "Multi_3D_zStack_Analysed"
        url = "https://hmgubox2.helmholtz-muenchen.de/index.php/s/3RNjGiPwKcdnGtj/download/Yeast_Analysed_multi3D_zStacks.zip"
        file_size = 124822528
    else:
        return ""

    examples_path = os.path.join(user_profile_path, "acdc-examples")
    example_path = os.path.join(examples_path, foldername)
    return examples_path, example_path, url, file_size


def get_acdc_java_path():
    acdc_java_path = os.path.join(user_profile_path, "acdc-java")
    dot_acdc_java_path = os.path.join(user_profile_path, ".acdc-java")
    return acdc_java_path, dot_acdc_java_path


def get_model_path(model_name, create_temp_dir=True):
    if model_name == "Automatic thresholding":
        model_name == "thresholding"

    model_info_path = os.path.join(models_path, model_name, "model")

    if os.path.exists(model_info_path):
        for file in listdir(model_info_path):
            if file != "weights_location_path.txt":
                continue
            with open(os.path.join(model_info_path, file), "r") as txt:
                model_path = txt.read()
                model_path = os.path.expanduser(model_path)
            if not os.path.exists(model_path):
                model_path = _write_model_location_to_txt(model_name)
            else:
                break
        else:
            model_path = _write_model_location_to_txt(model_name)
    else:
        os.makedirs(model_info_path, exist_ok=True)
        model_path = _write_model_location_to_txt(model_name)

    model_path = migrate_to_new_user_profile_path(model_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    if not create_temp_dir:
        return "", model_path

    exists = check_model_exists(model_path, model_name)
    if exists:
        return "", model_path

    temp_zip_path = _create_temp_dir()
    return temp_zip_path, model_path


def _create_temp_dir():
    temp_model_path = tempfile.mkdtemp()
    temp_zip_path = os.path.join(temp_model_path, "model_temp.zip")
    return temp_zip_path


def check_v123_model_path(model_name):
    # Cell-ACDC v1.2.3 saved the weights inside the package,
    # while from v1.2.4 we save them on user folder. If we find the
    # weights in the package we move them to user folder without downloading
    # new ones.
    v123_model_path = os.path.join(models_path, model_name, "model")
    exists = check_model_exists(v123_model_path, model_name)
    if exists:
        return v123_model_path
    else:
        return ""


def is_old_user_profile_path(path_to_check: os.PathLike):
    from . import user_data_dir

    user_data_folderpath = user_data_dir()
    user_profile_path_txt = os.path.join(
        user_data_folderpath, "acdc_user_profile_location.txt"
    )
    if os.path.exists(user_profile_path_txt):
        return False

    from . import user_home_path

    user_home_path = user_home_path.replace("\\", "/")
    path_to_check = path_to_check.replace("\\", "/")
    return user_home_path == path_to_check


def migrate_to_new_user_profile_path(path_to_migrate: os.PathLike):
    parent_dir = os.path.dirname(path_to_migrate)
    if not is_old_user_profile_path(parent_dir):
        return path_to_migrate
    folder = os.path.basename(path_to_migrate)
    return os.path.join(user_profile_path, folder)


def determine_folder_type(folder_path):
    is_pos_folder = is_pos_folderpath(folder_path)
    is_images_folder = folder_path.endswith("Images") and listdir(folder_path)
    contains_images_folder = os.path.exists(os.path.join(folder_path, "Images"))
    contains_pos_folders = len(get_pos_foldernames(folder_path)) > 0
    if contains_pos_folders:
        is_pos_folder = False
        is_images_folder = False
    elif contains_images_folder and not is_pos_folder:
        # Folder created by loading an image
        is_images_folder = True
        folder_path = os.path.join(folder_path, "Images")

    return is_pos_folder, is_images_folder, folder_path


def to_relative_path(path, levels=3, prefix="..."):
    path = path.replace("\\", "/")
    parts = path.split("/")
    if levels >= len(parts):
        return path
    parts = parts[-levels:]
    rel_path = "/".join(parts)
    rel_path.replace("/", os.sep)
    if prefix:
        rel_path = f"{prefix}{os.sep}{rel_path}"
    return rel_path


def get_fiji_binary_filepath_mac(fiji_app_filepath):
    if not is_mac:
        return ""

    fiji_binary_path = os.path.join(
        fiji_app_filepath, "Contents", "MacOS", "ImageJ-macosx"
    )
    if os.path.exists(fiji_binary_path):
        return fiji_binary_path

    fiji_binary_path = os.path.join(
        fiji_app_filepath, "Contents", "MacOS", "fiji-macos"
    )
    if os.path.exists(fiji_binary_path):
        return fiji_binary_path

    return ""


def get_fiji_exec_folderpath() -> str:
    if not is_mac:
        return ""

    from cellacdc import fiji_location_filepath

    if os.path.exists(fiji_location_filepath):
        with open(fiji_location_filepath, "r") as txt:
            fiji_app_filepath = txt.read()

        return get_fiji_binary_filepath_mac(fiji_app_filepath)

    if os.path.exists("/Applications/Fiji.app"):
        return get_fiji_binary_filepath_mac("/Applications/Fiji.app")

    acdc_fiji_app_path = os.path.join(acdc_fiji_path, "Fiji.app")
    acdc_fiji_binary_path = get_fiji_binary_filepath_mac(acdc_fiji_app_path)

    return acdc_fiji_binary_path


def is_pos_folderpath(folderpath):
    """Determine if a path is a valid Cell-ACDC Position folder

    Parameters
    ----------
    folderpath : PathLike
        Path to check

    Returns
    -------
    bool
        True if the path is a valid Cell-ACDC Position folder, False otherwise

    Notes
    -----
    A valid Cell-ACDC Position folder must:
        - Have a name matching the pattern 'Position_<number>'
        - Be a directory
        - Contain an 'Images' subdirectory
        - The 'Images' subdirectory must not be empty
    """
    foldername = os.path.basename(folderpath)
    is_valid_pos_folder = (
        re.search(r"^Position_(\d+)$", foldername) is not None
        and os.path.isdir(folderpath)
        and os.path.exists(os.path.join(folderpath, "Images"))
        and listdir(os.path.join(folderpath, "Images"))
    )
    return is_valid_pos_folder


def validate_images_path(input_path: os.PathLike, create_dirs_tree=False):
    is_images_path = input_path.endswith("Images")
    parent_dir = os.path.dirname(input_path)
    parent_foldername = os.path.basename(parent_dir)
    is_pos_folder = re.search(
        r"^Position_(\d+)$", parent_foldername
    ) is not None and os.path.isdir(parent_dir)
    if not is_pos_folder:
        existing_pos_foldernames = get_pos_foldernames(input_path)
        pos_n = len(existing_pos_foldernames) + 1
        pos_folderpath = os.path.join(input_path, f"Position_{pos_n}")
        images_path = os.path.join(pos_folderpath, "Images")
    elif is_images_path:
        pos_folderpath = input_path
        images_path = os.path.join(pos_folderpath, "Images")
    else:
        images_path = input_path

    if create_dirs_tree:
        os.makedirs(images_path, exist_ok=True)

    return images_path

# Sibling imports (deferred to avoid import cycles)
from .models import (
    _write_model_location_to_txt,
    check_model_exists,
)

