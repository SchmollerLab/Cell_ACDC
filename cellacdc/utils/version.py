"""Cell-ACDC utility helpers: version."""

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

def get_salute_string():
    time_now = datetime.datetime.now().time()
    time_end_morning = datetime.time(12, 00, 00)
    time_end_lunch = datetime.time(13, 00, 00)
    time_end_afternoon = datetime.time(15, 00, 00)
    time_end_evening = datetime.time(20, 00, 00)
    time_end_night = datetime.time(4, 00, 00)
    if time_now >= time_end_night and time_now < time_end_morning:
        return "Have a good day!"
    elif time_now >= time_end_morning and time_now < time_end_lunch:
        return "Enjoy your lunch!"
    elif time_now >= time_end_lunch and time_now < time_end_afternoon:
        return "Have a good afternoon!"
    elif time_now >= time_end_afternoon and time_now < time_end_evening:
        return "Have a good evening!"
    else:
        return "Have a good night!"


def get_info_version_text(is_cli=False, cli_formatted_text=True):
    version = read_version()
    release_date = get_date_from_version(version, package="cellacdc")
    py_ver = sys.version_info
    env_folderpath = sys.prefix
    python_version = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    info_txts = [
        f"Version {version}",
        f"Released on: {release_date}",
        f'Installed in "{cellacdc_path}"',
        f'Environment folder: "{env_folderpath}"',
        f'User profile folder: "{user_profile_path}"',
        f'Settings folder: "{settings_folderpath}"',
        f"Python {python_version}",
        f"Platform: {platform.platform()}",
        f"System: {platform.system()}",
    ]
    if is_linux:
        try:
            distro_name = get_linux_distribution_name()
        except Exception as err:
            distro_name = "Undetermined"

        info_txts.append(f"Linux distribution: {distro_name}")

    if GUI_INSTALLED and not is_cli:
        info_txts.append(f'Icons from: "{qrc_resources_path}"')
        try:
            from qtpy import QtCore

            info_txts.append(f"Qt {QtCore.__version__}")
        except Exception as err:
            info_txts.append("Qt: Not installed")

    try:
        branch_name = get_git_branch_name()
        info_txts.append(f'Git branch: "{branch_name}"')
    except Exception as err:
        pass

    info_txts.append(f"Working directory: {os.getcwd()}")

    if not cli_formatted_text:
        return info_txts

    info_txts = [f"  - {txt}" for txt in info_txts]

    max_len = max([len(txt) for txt in info_txts]) + 2

    formatted_info_txts = []
    for txt in info_txts:
        horiz_spacing = " " * (max_len - len(txt))
        txt = f"{txt}{horiz_spacing}|"
        formatted_info_txts.append(txt)

    formatted_info_txts.insert(0, "Cell-ACDC info:\n")
    formatted_info_txts.insert(0, "=" * max_len)
    formatted_info_txts.append("=" * max_len)
    info_txt = "\n".join(formatted_info_txts)

    try:
        from spotmax.utils import get_info_version_text as smax_info

        smax_info_txt = smax_info(include_platform=False, is_cli=is_cli)
        info_txt += "\n\n" + smax_info_txt
    except ImportError:
        pass

    return info_txt


def read_version(logger=None, return_success=False):
    cellacdc_parent_path = os.path.dirname(cellacdc_path)
    cellacdc_parent_folder = os.path.basename(cellacdc_parent_path)
    if cellacdc_parent_folder == "site-packages":
        from . import __version__

        version = __version__
        success = True
    else:
        try:
            from setuptools_scm import get_version

            version = get_version(root="..", relative_to=__file__)
            success = True
        except Exception as e:
            if logger is None:
                logger = print
            logger("*" * 40)
            logger(traceback.format_exc())
            logger("-" * 40)
            logger(
                "[WARNING]: Cell-ACDC could not determine the current version. "
                "Returning the version determined at installation time. "
                "See details above."
            )
            logger("=" * 40)
            try:
                from . import _version

                version = _version.version
                success = False
            except Exception as e:
                version = "ND"
                success = False

    if return_success:
        return version, success
    else:
        return version


def get_date_from_version(version: str, package="cellacdc", debug=False):
    try:
        response = requests.get(f"https://pypi.org/pypi/{package}/json", timeout=2)
        res_json = response.json()
        pypi_releases_json = res_json["releases"]
        version_json = pypi_releases_json[version][0]
        upload_time = version_json["upload_time_iso_8601"]
        date = datetime.datetime.strptime(upload_time, r"%Y-%m-%dT%H:%M:%S.%fZ")
        date_str = date.strftime(r"%A %d %B %Y at %H:%M")
        return date_str
    except Exception as err:
        if debug:
            traceback.print_exc()

    try:
        # Locate the direct_url.json file for the package
        # installed with pip git+
        dist = importlib.metadata.distribution(package)
        dist_info_dir = dist._path  # internal path to .dist-info
        direct_url_path = os.path.join(dist_info_dir, "direct_url.json")

        with open(direct_url_path) as f:
            data = json.load(f)

        vcs_info = data["vcs_info"]
        commit_id = vcs_info.get("commit_id")
        url = data.get("url")

        parts = url.split("github.com/")[1].split(".git")[0]
        owner, repo = parts.split("/", 1)

        # Query GitHub API for commit date
        api_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_id}"
        response = requests.get(api_url)
        response.raise_for_status()

        commit_data = response.json()
        date_utc = commit_data["commit"]["committer"]["date"]

        date_str = format_commit_date_utc(date_utc)

        return date_str
    except Exception as err:
        if debug:
            traceback.print_exc()

    try:
        if package == "cellacdc":
            pkg_path = cellacdc_path
        elif package == "spotmax":
            from spotmax import spotmax_path

            pkg_path = spotmax_path
        commit_hash = re.findall(r"\+g([A-Za-z0-9]+)(\.d)?", version)[0][0]
        git_path = os.path.dirname(pkg_path)
        command = f"git -C {git_path} show {commit_hash}"
        commit_log = _subprocess_run_command(
            command, shell=False, callback="check_output"
        )
        commit_log = commit_log.decode()
        date_log = re.findall(r"Date:(.*) \+", commit_log)[0].strip()
        date = datetime.datetime.strptime(date_log, r"%a %b %d %H:%M:%S %Y")
        date_str = date.strftime(r"%A %d %B %Y at %H:%M")
        return date_str
    except Exception as err:
        if debug:
            traceback.print_exc()

    return "ND"


def get_git_branch_name():
    command = "git rev-parse --abbrev-ref HEAD"
    output = _subprocess_run_command(command, shell=False, callback="check_output")
    branch_name = output.decode().strip()
    return branch_name


def get_cellpose_major_version(errors="raise"):
    major_installed = None
    try:
        installed_version = get_package_version("cellpose")
        major_installed = int(installed_version.split(".")[0])
    except Exception as err:
        if errors == "raise":
            raise err

    return major_installed


def check_cellpose_version(version: str):
    if isinstance(version, int):
        version = f"{version}.0"

    major_requested = int(version.split(".")[0])
    cancel = False
    try:
        installed_version = get_package_version("cellpose")
        major_installed = int(installed_version.split(".")[0])
        is_version_correct = major_installed == major_requested
        if not is_version_correct:
            cancel = _warnings.warn_installing_different_cellpose_version(
                version, installed_version
            )
        if not is_second_version_greater(
            min_target_versions_cp[str(major_requested)], installed_version
        ):
            is_version_correct = False
    except Exception as err:
        is_version_correct = False

    if cancel:
        raise ModuleNotFoundError("Cellpose installation cancelled by the user.")
    return is_version_correct


def is_second_version_greater(
    target_version: str,
    current_version: str,
):
    """
    Compares two model versions and returns True if the current version is
    greater than or equal to the target version.
    """
    target_version = packaging_version.parse(target_version)
    current_version = packaging_version.parse(current_version)

    return current_version >= target_version


def is_pkg_version_within_range(package_version: str, min_version="", max_version=""):
    package_version_number = packaging_version.parse(package_version)
    is_greater_than_min = True
    if min_version:
        min_version_number = packaging_version.parse(min_version)
        is_greater_than_min = package_version_number >= min_version_number

    is_less_than_max = True
    if max_version:
        max_version_number = packaging_version.parse(max_version)
        is_less_than_max = package_version_number <= max_version_number

    return is_greater_than_min and is_less_than_max


def check_pkg_version(
    import_pkg_name, min_version, include_lower_version, raise_err=True
):
    is_version_correct = False
    try:
        installed_version = get_package_version(import_pkg_name)
        if include_lower_version:
            is_version_correct = packaging_version.parse(
                installed_version
            ) >= packaging_version.parse(min_version)
        else:
            is_version_correct = packaging_version.parse(
                installed_version
            ) > packaging_version.parse(min_version)
    except Exception as err:
        is_version_correct = False

    if raise_err and not is_version_correct:
        raise ModuleNotFoundError(f"{import_pkg_name}>{min_version} not installed.")
    else:
        return is_version_correct


def check_pkg_exact_version(import_pkg_name, version: str, raise_err=True):
    is_version_correct = False
    try:
        installed_version = get_package_version(import_pkg_name)
        is_version_correct = packaging_version.parse(
            installed_version
        ) == packaging_version.parse(version)
    except Exception as err:
        is_version_correct = False

    if raise_err and not is_version_correct:
        raise ModuleNotFoundError(f"{import_pkg_name}=={version} not installed.")
    else:
        return is_version_correct


def check_pkg_max_version(
    import_pkg_name, max_version, include_higher_version, raise_err=True
):
    is_version_correct = False
    try:
        from packaging import version

        installed_version = get_package_version(import_pkg_name)
        if include_higher_version:
            is_version_correct = packaging_version.parse(
                installed_version
            ) <= packaging_version.parse(max_version)
        else:
            is_version_correct = packaging_version.parse(
                installed_version
            ) < packaging_version.parse(max_version)
    except Exception as err:
        is_version_correct = False

    if raise_err and not is_version_correct:
        raise ModuleNotFoundError(f"{import_pkg_name}<={max_version} not installed.")
    else:
        return is_version_correct


def check_matplotlib_version(qparent=None):
    mpl_version = get_package_version("matplotlib")
    mpl_version_digits = mpl_version.split(".")

    mpl_major = int(mpl_version_digits[0])
    mpl_minor = int(mpl_version_digits[1])
    is_less_than_3_5 = mpl_major < 3 or (mpl_major >= 3 and mpl_minor < 5)
    if not is_less_than_3_5:
        return

    proceed = _install_package_msg("matplotlib", parent=qparent, upgrade=True)
    if not proceed:
        raise ModuleNotFoundError(f'User aborted "matplotlib" installation')
    import subprocess

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-U", "matplotlib"]
        )
    except Exception as e:
        printl(traceback.format_exc())
        _inform_install_package_failed("matplotlib", parent=qparent, do_exit=False)


def get_git_pull_checkout_cellacdc_version_commands(version=None):
    if version is None:
        version = read_version()
    commit_hash_idx = version.find("+g")
    is_dev_version = commit_hash_idx > 0
    if not is_dev_version:
        return []
    commit_hash = version[commit_hash_idx + 2 :].split(".")[0]
    commands = (
        f'cd "{os.path.dirname(cellacdc_path)}"',
        "git pull",
        f"git checkout {commit_hash}",
    )
    return commands


def _update_repo_with_git_command(package_name, repo_location):
    """Update repository using git command"""
    try:
        print(
            f"Updating {package_name} repository at {repo_location} using git command..."
        )

        # Change to repository directory
        original_cwd = os.getcwd()
        os.chdir(repo_location)

        stashed_changes = False

        # check if there is a portable git
        from .config import parser_args

        try:
            cp = parser_args
            if cp["install_details"] is not None:
                no_cli_install = True
                install_details = cp["install_details"]
                target_dir = install_details.get("target_dir", "")
                target_dir = target_dir.strip().strip('"').strip("'")
                target_dir = os.path.abspath(target_dir)
            else:
                no_cli_install = False
        except:
            no_cli_install = False
            pass

        if is_win and no_cli_install:
            git_loc = os.path.join(target_dir, "portable_git", "cmd", "git.exe")
            if not os.path.exists(git_loc):
                print(f"Portable git not found at {git_loc}. Using system git.")
                git_loc = "git"
        else:
            git_loc = "git"

        # Check if git is available
        if not shutil.which(git_loc):
            print(
                f"Git command not found. Please install git to update {package_name}."
            )
            return False

        try:
            # Check for uncommitted changes

            branch_result = subprocess.run(
                [git_loc, "branch", "--show-current"],
                capture_output=True,
                text=True,
                check=True,
            )
            current_branch = branch_result.stdout.strip()
            print(f"Current branch: {current_branch}")

            result = subprocess.run(
                [git_loc, "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout.strip():
                print(f"Repository {package_name} has uncommitted changes")
                print("Stashing changes before update...")
                subprocess.run([git_loc, "stash"], check=True)
                stashed_changes = True

            # Pull changes
            subprocess.run([git_loc, "pull"], check=True)
            print(f"Successfully updated {package_name}")

            # Pop stashed changes if any were stashed
            if stashed_changes:
                try:
                    subprocess.run([git_loc, "stash", "pop"], check=True)
                    print("Restored stashed changes")
                except subprocess.CalledProcessError as pop_error:
                    print(f"Warning: Could not restore stashed changes: {pop_error}")

            return True

        except subprocess.CalledProcessError as e:
            print(f"Git command failed for {package_name}: {e}")
            return False
        finally:
            os.chdir(original_cwd)

    except Exception as e:
        print(f"Error updating {package_name} with git command: {e}")
        return False

# Sibling imports (deferred to avoid import cycles)
from .install import (
    _inform_install_package_failed,
    _install_package_msg,
    get_package_version,
)
from .misc import (
    _subprocess_run_command,
    format_commit_date_utc,
    get_linux_distribution_name,
)

