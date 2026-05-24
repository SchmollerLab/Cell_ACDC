"""Cell-ACDC utility helpers: install."""

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

def check_git_installed(parent=None):
    try:
        subprocess.check_call(["git", "--version"], shell=True)
        return True
    except Exception as e:
        print("=" * 20)
        traceback.print_exc()
        print("=" * 20)
        git_url = "https://git-scm.com/book/en/v2/Getting-Started-Installing-Git"
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(f"""
            In order to install <code>javabridge</code> you first need to <b>install
            Git</b> (it was not found).<br><br>
            <b>Close Cell-ACDC</b> and follow the instructions
            {html_utils.tag("here", f'a href="{git_url}"')}.<br><br>
            <i><b>NOTE</b>: After installing Git you might need to <b>restart the
            terminal</b></i>.
        """)
        msg.warning(parent, "Git not installed", txt)
        return False


def install_java():
    try:
        subprocess.check_call(["javac", "-version"], shell=True)
        return False
    except Exception as e:
        from . import widgets

        win = widgets.installJavaDialog()
        win.exec_()
        return win.clickedButton == win.cancelButton


def install_javabridge(force_compile=False, attempt_uninstall_first=False):
    if attempt_uninstall_first:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "uninstall", "-y", "javabridge"]
            )
        except Exception as e:
            pass
    if sys.platform.startswith("win"):
        if force_compile:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-U",
                    "git+https://github.com/SchmollerLab/python-javabridge-acdc",
                ]
            )
        else:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-U",
                    "git+https://github.com/SchmollerLab/python-javabridge-windows",
                ]
            )
    elif is_mac:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-U",
                "git+https://github.com/SchmollerLab/python-javabridge-acdc",
            ]
        )
    elif is_linux:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-U",
                "git+https://github.com/LeeKamentsky/python-javabridge.git@master",
            ]
        )


def get_java_url():
    is_linux = sys.platform.startswith("linux")
    is_mac = sys.platform == "darwin"
    is_win = sys.platform.startswith("win")
    is_win64 = is_win and (os.environ["PROCESSOR_ARCHITECTURE"] == "AMD64")

    # https://drive.google.com/drive/u/0/folders/1MxhySsxB1aBrqb31QmLfVpq8z1vDyLbo
    if is_win64:
        os_foldername = "win64"
        unzipped_foldername = "java_portable_windows-0.1"
        file_size = 214798150
        # url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/eMyirTw8qG2wJMt/download/java_portable_windows-0.1.zip'
        url = "https://github.com/SchmollerLab/java_portable_windows/archive/refs/tags/v0.1.zip"
    elif is_mac:
        os_foldername = "macOS"
        unzipped_foldername = "java_portable_macos-0.1"
        url = "https://github.com/SchmollerLab/java_portable_macos/archive/refs/tags/v0.1.zip"
        # url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/SjZb8aommXgrECq/download/java_portable_macos-0.1.zip'
        file_size = 108478751
    elif is_linux:
        os_foldername = "linux"
        unzipped_foldername = "java_portable_linux-0.1"
        url = "https://github.com/SchmollerLab/java_portable_linux/archive/refs/tags/v0.1.zip"
        # url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/HjeQagixE2cjbZL/download/java_portable_linux-0.1.zip'
        file_size = 92520706
    return url, file_size, os_foldername, unzipped_foldername


def get_package_version(import_pkg_name):
    import importlib.metadata

    version = importlib.metadata.version(import_pkg_name)
    return version


def check_upgrade_javabridge():
    try:
        version = get_package_version("javabridge")
    except Exception as e:
        return
    patch = int(version.split(".")[2])
    if patch > 18:
        return
    install_javabridge()


def _java_exists(os_foldername):
    acdc_java_path, dot_acdc_java_path = get_acdc_java_path()
    os_acdc_java_path = os.path.join(acdc_java_path, os_foldername)
    if os.path.exists(os_acdc_java_path):
        for folder in os.listdir(os_acdc_java_path):
            if not folder.startswith("jre"):
                continue
            dir_path = os.path.join(os_acdc_java_path, folder)
            for file in os.listdir(dir_path):
                if file == "bin":
                    return dir_path

    # Some users still has the old .acdc folder --> check
    os_dot_acdc_java_path = os.path.join(dot_acdc_java_path, os_foldername)
    if os.path.exists(os_dot_acdc_java_path):
        for folder in os.listdir(os_dot_acdc_java_path):
            if not folder.startswith("jre"):
                continue
            dir_path = os.path.join(os_dot_acdc_java_path, folder)
            for file in os.listdir(dir_path):
                if file == "bin":
                    return dir_path
    return ""

    # Check if the user unzipped the javabridge_portable folder and not its content
    os_acdc_java_path = os.path.join(acdc_java_path, os_foldername)
    if os.path.exists(os_acdc_java_path):
        for folder in os.listdir(os_acdc_java_path):
            dir_path = os.path.join(os_acdc_java_path, folder)
            if folder.startswith("java_portable") and os.path.isdir(dir_path):
                # Move files one level up
                unzipped_path = os.path.join(os_acdc_java_path, folder)
                for name in os.listdir(unzipped_path):
                    # move files up one level
                    src = os.path.join(unzipped_path, name)
                    shutil.move(src, os_acdc_java_path)
                try:
                    shutil.rmtree(unzipped_path)
                except PermissionError as e:
                    pass
        # Check if what we moved one level up was actually java
        for folder in os.listdir(os_acdc_java_path):
            if not folder.startswith("jre"):
                continue
            dir_path = os.path.join(os_acdc_java_path, folder)
            for file in os.listdir(dir_path):
                if file == "bin":
                    return dir_path
    return ""


def download_java():
    url, file_size, os_foldername, unzipped_foldername = get_java_url()
    jre_path = _java_exists(os_foldername)
    jdk_path = _jdk_exists(jre_path)
    if os_foldername.startswith("win") and jre_path and jdk_path:
        return jre_path, jdk_path, url

    if jre_path:
        # on macOS jdk is the same as jre
        return jre_path, jre_path, url

    acdc_java_path, _ = get_acdc_java_path()
    os_acdc_java_path = os.path.join(acdc_java_path, os_foldername)
    temp_zip = os.path.join(os_acdc_java_path, "acdc_java_temp.zip")

    if not os.path.exists(os_acdc_java_path):
        os.makedirs(os_acdc_java_path, exist_ok=True)

    try:
        download_url(url, temp_zip, file_size=file_size, desc="Java")
        extract_zip(temp_zip, os_acdc_java_path)
    except Exception as e:
        print("=======================")
        traceback.print_exc()
        print("=======================")
    finally:
        os.remove(temp_zip)

    # Move files one level up
    unzipped_path = os.path.join(os_acdc_java_path, unzipped_foldername)
    for name in os.listdir(unzipped_path):
        # move files up one level
        src = os.path.join(unzipped_path, name)
        shutil.move(src, os_acdc_java_path)
    try:
        shutil.rmtree(unzipped_path)
    except PermissionError as e:
        pass

    jre_path = _java_exists(os_foldername)
    jdk_path = _jdk_exists(jre_path)
    return jre_path, jdk_path, url


def _install_homebrew_command():
    return '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'


def _brew_install_java_command():
    return "brew install --cask homebrew/cask-versions/adoptopenjdk8"


def _brew_install_hdf5():
    return "brew install hdf5"


def _apt_install_java_command():
    return "sudo apt-get install openjdk-8-jdk"


def _java_instructions_linux():
    s1 = html_utils.paragraph("""
        Run the following commands<br>
        in the Teminal <b>one by one:</b>
    """)

    s2 = html_utils.paragraph(f"""
        <code>{_apt_gcc_command().replace(" ", "&nbsp;")}</code>
    """)

    s3 = html_utils.paragraph(f"""
        <code>{_apt_update_command().replace(" ", "&nbsp;")}</code>
    """)

    s4 = html_utils.paragraph(f"""
        <code>{_apt_install_java_command().replace(" ", "&nbsp;")}</code>
    """)

    s5 = html_utils.paragraph("""
    The first command is used to install GCC, which is needed later.<br><br>
    The second and third commands are used is used to install
    Java Development Kit 8.<br><br>
    Follow the instructions on the terminal to complete
    installation.<br><br>
    """)
    return s1, s2, s3, s4


def _java_instructions_macOS():
    s1 = html_utils.paragraph("""
        Run the following commands<br>
        in the Teminal <b>one by one:</b>
    """)

    s2 = html_utils.paragraph(f"""
        <code>{_install_homebrew_command()}</code>
    """)

    s3 = html_utils.paragraph(f"""
        <code>{_brew_install_java_command().replace(" ", "&nbsp;")}</code>
    """)

    s4 = html_utils.paragraph("""
    The first command is used to install Homebrew<br>
    a package manager for macOS/Linux.<br><br>
    The second command is used to install Java 8.<br>
    Follow the instructions on the terminal to complete
    installation.<br><br>
    Alternatively,<b> you can install Java as a regular app</b><br>
    by downloading the app from
    <a href="https://hmgubox2.helmholtz-muenchen.de/index.php/s/AWWinWCTXwWTmEi">
        here
    </a>.
    """)
    return s1, s2, s3, s4


def _java_instructions_windows():
    jdk_url = f'"{jdk_windows_url()}"'
    cpp_url = f'"{cpp_windows_url()}"'
    s1 = html_utils.paragraph("""
        Download and install <code>Java Development Kit</code> and<br>
        <b>Microsoft C++ Build Tools</b> for Windows (links below).<br><br>
        <b>IMPORTANT</b>: when installing "Microsoft C++ Build Tools"<br>
        make sure to select <b>"Desktop development with C++"</b>.<br>
        Click "See the screenshot" for more details.<br>
    """)

    s2 = html_utils.paragraph(f"""
        Java Development Kit:
            <a href={jdk_url}>
                here
            </a>
    """)

    s3 = html_utils.paragraph(f"""
        Microsoft C++ Build Tools:
            <a href={cpp_url}>
                here
            </a>
    """)
    return s1, s2, s3


def install_javabridge_instructions_text():
    if is_win:
        return _java_instructions_windows()
    elif is_mac:
        return _java_instructions_macOS()
    elif is_linux:
        return _java_instructions_linux()


def install_javabridge_help(parent=None):
    msg = widgets.myMessageBox()
    txt = html_utils.paragraph(f"""
        Cell-ACDC is going to <b>download and install</b>
        <code>javabridge</code>.<br><br>
        Make sure you have an <b>active internet connection</b>,
        before continuing.
        Progress will be displayed on the terminal<br><br>
        <b>IMPORTANT:</b> If the installation fails, <b>please open an issue</b>
        on our
        <a href="https://github.com/SchmollerLab/Cell_ACDC/issues">
            GitHub page
        </a>.<br><br>
        Alternatively, you can cancel the process and try later.
    """)
    msg.setIcon()
    msg.setWindowTitle("Installing javabridge")
    msg.addText(txt)
    msg.addButton("   Ok   ")
    cancel = msg.addButton(" Cancel ")
    msg.exec_()
    return msg.clickedButton == cancel


def _install_pip_package(
    pkg_name: str,
    logger: Callable = print,
    install_dependencies: bool = True,
    force_binary: bool = True,
    pref_binary: bool = True,
) -> None:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        pkg_name,
    ]
    if force_binary:
        command.append("--only-binary=:all:")
    elif pref_binary:
        command.append("--prefer-binary")
    if not install_dependencies:
        command.append("--no-deps")
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        if "--only-binary=:all:" in str(e):
            logger(
                f"Error: {pkg_name} does not have a binary distribution available, trying preferred binary."
            )
            _install_pip_package(
                pkg_name=pkg_name,
                logger=logger,
                install_dependencies=install_dependencies,
                force_binary=False,
                pref_binary=True,
            )
        elif "--prefer-binary" in str(e):
            logger(
                f"Error: {pkg_name} does not have a preferred binary distribution available, trying source."
            )
            command.remove("--prefer-binary")
            command.append("--no-binary=:all:")
            _install_pip_package(
                pkg_name=pkg_name,
                logger=logger,
                install_dependencies=install_dependencies,
                force_binary=False,
                pref_binary=False,
            )
        else:
            logger(f"""Error: {pkg_name} installation failed. Please check the error message. This is probably due to the package 
                   not being available for your platform or python version.""")
            raise e


def uninstall_pip_package(pkg_name):
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", pkg_name])


def uninstall_omnipose_acdc():
    """Uninstall omnipose-acdc if present. Since v1.5.0 it is not needed."""
    import json

    pip_list_output = subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--format", "json"]
    )
    installed_packages = json.loads(pip_list_output)
    pkgs_to_uninstall = []
    for package_info in installed_packages:
        if package_info["name"] == "omnipose-acdc":
            pkgs_to_uninstall.append("omnipose-acdc")
        elif package_info["name"] == "cellpose-omni-acdc":
            pkgs_to_uninstall.append("cellpose-omni-acdc")

    for pkg_to_uninstall in pkgs_to_uninstall:
        uninstall_pip_package(pkg_to_uninstall)


def check_install_cellpose(
    version: Literal["2.0", "3.0", "4.0", "any"] = "2.0",
    version_to_install_if_missing: Literal["2.0", "3.0", "4.0"] = "4.0",
):
    if isinstance(version, int):
        version = f"{version}.0"

    check_install_torch()

    if version == "any":
        try:
            from cellpose import models

            return
        except Exception as err:
            version = version_to_install_if_missing  # after this the version will for sure be a valid format and not 'any'

    is_version_correct = check_cellpose_version(version)
    if is_version_correct:
        return

    major_version = int(version.split(".")[0])

    next_version = major_version + 1

    min_version = min_target_versions_cp[str(major_version)]

    check_install_package(
        "cellpose",
        max_version=f"{next_version}.0",
        min_version=min_version,
        include_lower_version=True,
    )

    purge_module("cellpose")


def check_install_baby():
    check_install_package(
        "TensorFlow",
        pypi_name="tensorflow",
        import_pkg_name="tensorflow",
        max_version="2.14",
    )
    check_install_package("baby", pypi_name="baby-seg", import_pkg_name="baby")


def check_install_nnInteractive():
    check_install_package("huggingface-hub")
    check_install_torch()
    check_install_package("nnInteractive")

    purge_module("nnInteractive")

    importlib.invalidate_caches()
    import nnInteractive

    importlib.reload(nnInteractive)


def check_install_microsam():
    check_install_package("micro-sam", pypi_name="micro_sam", installer="conda")


def check_install_yeaz():
    check_install_torch()
    check_install_package("yeaz")


def check_install_segment_anything():
    check_install_torch()
    check_install_package("segment_anything")


def check_install_sam2():
    check_install_torch()
    check_install_package("sam2")


def check_install_cellsam():
    check_install_torch()
    check_install_package(
        "cellSAM",
        pypi_name="git+https://github.com/vanvalenlab/cellSAM.git",
        import_pkg_name="cellSAM",
        note=(
            "CellSAM requires a DeepCell access token to download models.\n"
            "Set the DEEPCELL_ACCESS_TOKEN environment variable before use.\n"
            "Get your token at: https://deepcell.org"
        ),
    )


def install_package_conda(conda_pkg_name, channel="conda-forge"):
    if not is_conda_env():
        raise EnvironmentError("Cell-ACDC is not running in a `conda` environment.")
    conda_prefix, pip_prefix = get_pip_conda_prefix()
    conda_prefix = re.sub(
        r"(-c\sconda-forge\s?|--channel=conda-forge\s?)", f"-c {channel} ", conda_prefix
    )

    command = f"{conda_prefix} -y {conda_pkg_name}"
    _subprocess_run_command(command)


def check_install_omnipose():
    try:
        import_module("omnipose")
        return
    except ModuleNotFoundError:
        pass

    try:
        check_install_package("omnipose", pypi_name="omnipose_acdc")
    except Exception as err:
        install_package_conda("mahotas")
        _install_pip_package("omnipose-acdc")


def _warn_dll_torch(qparent=None):
    msg = widgets.myMessageBox()
    txt = html_utils.paragraph("""
    <b>An error message will occur after you close this message.</b><br>
    <b>Please save your data and restart Cell-ACDC.</b><br>
    Sorry for the inconvenience!<br>
    This error is not critical for the main functionality of Cell-ACDC,
    and only concerns the segmentation model. Your can save your data without
    a problem.<br>
    The specific reason is that PyTorch and QtPy have weird issues with 
    DLL conflicts.
    """)
    msg.information(
        qparent,
        "Please restart Cell-ACDC",
        txt,
        buttonsTexts=("Ok, I will save my data and restart Cell-ACDC"),
    )


def check_install_torch(is_cli=False, caller_name="Cell-ACDC", qparent=None):
    try:
        import torch
        import torchvision

        return

    except OSError as err:
        if "dll" in str(err):
            _warn_dll_torch(qparent=qparent)
            raise err
        else:
            traceback.print_exc()
    except Exception as err:
        traceback.print_exc()

    if is_cli:
        _install_pytorch_cli(caller_name=caller_name)
        return

    win = apps.InstallPyTorchDialog(parent=qparent, caller_name=caller_name)
    win.exec_()
    if win.cancel:
        _warnings.log_pytorch_not_installed()
        return

    command = win.command
    print(f'Running command: "{command}"')
    _run_command(command)

    try:
        import torch
    except OSError as e:
        if "dll" in str(e):
            _warn_dll_torch(qparent=qparent)
        raise e

    purge_module("torch")


def check_install_package(
    pkg_name: str,
    import_pkg_name: str = "",
    pypi_name="",
    note="",
    parent=None,
    raise_on_cancel=True,
    logger_func=print,
    is_cli=False,
    caller_name="Cell-ACDC",
    force_upgrade=False,
    upgrade=False,
    min_version="",
    max_version="",
    exact_version="",
    install_dependencies=True,
    return_outcome=False,
    installer: Literal["pip", "conda"] = "pip",
    include_higher_version: bool = False,
    include_lower_version: bool = False,
):
    """Try to import a package. If import fails, ask user to install it
    automatically.

    Parameters
    ----------
    pkg_name : str
        The name of the package that is displayed to the user.
    import_pkg_name : str, optional
        The name of the package as it should be imported (case sensitive).
        If empty string, `pkg_name` will be imported instead. Default is ''
    pypi_name : str, optional
        The name of the package to be installed with pip.
        If empty string, `pkg_name` will be installed instead. Default is ''
    note : str, optional
        Additional text to display to the user. Default is ''
    parent : QObject, optional
        Calling QtWidget. Default is None
    raise_on_cancel : bool, optional
        Raise exception if processed cancelled. Default is True
    logger_func : callable, optional
        Function used to log text. Default is print
    is_cli : bool, optional
        If True, message will be displayed in the terminal.
        If False, message will be displayed in a Qt message box.
        Default is False
    caller_name : str, optional
        Program calling this function. Default is 'Cell-ACDC'
    force_upgrade : bool, optional
        If True, we force the upgrade even if package is installed.
    upgrade : bool, optional
        If True, pip will upgrade the package. This value is True if
        `force_upgrade` is True. Without min_version and max_version
        it will never upgrade or downgrade the package.
    min_version : str, optional
        If not empty it must be a valid version `major[.minor][.patch]` where
        minor and patch are optional. If the installed package is older the
        upgrade will be forced.
    max_version : str, optional
        If not empty it must be a valid version `major[.minor][.patch]` where
        minor and patch are optional. If the installed package is newer the
        upgrade will be forced.
    exact_version : str, optional
        If not empty, install this exact version. It must be a valid
        `major[.minor][.patch]`.
    install_dependencies : bool, optional
        If False, the `--no-deps` flag will be added to the pip command.
    return_outcome : bool, optional
        If True, returns 1 on successfull action
    installer : str, optional
        Package manager to use to install the package. Either 'pip' or 'conda'.
        Default is 'pip'
    include_higher_version : bool, optional
        If True, if the higher version is installed, it will not be downgraded.
        Default is False
    include_lower_version : bool, optional
        If True, if the lower version is installed, it will not be upgraded.
        Default is False

    Raises
    ------
    ModuleNotFoundError
        Error raised if process is cancelled and `raise_on_cancel=True`.
    """
    if not import_pkg_name:
        import_pkg_name = pkg_name

    if not is_gui_running():
        is_cli = True

    try:  # check_pkg_version and check_pkg_max_version
        import_pkg_name = import_pkg_name.replace("-", "_")
        import_module(import_pkg_name)
        if force_upgrade:
            upgrade = True
            raise ModuleNotFoundError(
                f'User requested to forcefully upgrade the package "{pkg_name}"'
            )
        if exact_version:
            check_pkg_exact_version(import_pkg_name, exact_version)
        if min_version:
            check_pkg_version(import_pkg_name, min_version, include_lower_version)
        if max_version:
            check_pkg_max_version(import_pkg_name, max_version, include_higher_version)
    except ModuleNotFoundError:
        proceed = _install_package_msg(
            pkg_name,
            note=note,
            parent=parent,
            upgrade=upgrade,
            is_cli=is_cli,
            caller_name=caller_name,
            logger_func=logger_func,
            pkg_command=pypi_name,
            max_version=max_version,
            min_version=min_version,
            exact_version=exact_version,
            installer=installer,
            include_higher_version=include_higher_version,
            include_lower_version=include_lower_version,
        )
        if pypi_name:
            pkg_name = pypi_name
        if not proceed:
            if raise_on_cancel:
                raise ModuleNotFoundError(f"User aborted {pkg_name} installation")
            else:
                return traceback.format_exc()
        try:
            if pkg_name == "tensorflow":
                _install_tensorflow(max_version=max_version, min_version=min_version)
            elif pkg_name == "deepsea":
                _install_deepsea()
            elif pkg_name == "segment_anything":
                _install_segment_anything()
            elif pkg_name == "sam2":
                _install_sam2()
            else:
                pkg_command = _get_pkg_command_pip_install(
                    pkg_name,
                    exact_version=exact_version,
                    max_version=max_version,
                    min_version=min_version,
                    including_higher_version=include_higher_version,
                    including_lower_version=include_lower_version,
                )
                if installer == "pip":
                    _install_pip_package(
                        pkg_command, install_dependencies=install_dependencies
                    )
                else:
                    install_package_conda(pkg_command)
        except Exception as e:
            printl(traceback.format_exc())
            _inform_install_package_failed(
                pkg_name, parent=parent, do_exit=raise_on_cancel
            )
        if return_outcome:
            return True


def check_install_custom_dependencies(custom_install_requires, *args, **kwargs):
    """Used to install a package with custom dependencies, usefull if they have
    random pinned versions for their dependencies.

    For *args and **kwargs see `utils.check_install_package`.

    Parameters
    ----------
    custom_install_requires : list
        list of dependencies. Check either requirements.txt, setup.py,
        setup.cfg, pyproject.toml, or any other file that lists the dependencies.
        For formatting of the dependencies with min max version,
        use _get_pkg_command_pip_install.
    """
    kwargs["install_dependencies"] = False
    kwargs["return_outcome"] = True
    success = check_install_package(*args, **kwargs)
    if not success:
        return
    for pkg_name in custom_install_requires:
        _install_pip_package(pkg_name)


def _inform_install_package_failed(pkg_name, parent=None, do_exit=True):
    conda_prefix, pip_prefix = get_pip_conda_prefix()

    install_command = f"<code>{pip_prefix} --upgrade {pkg_name}</code>"
    txt = html_utils.paragraph(f"""
        Unfortunately, <b>installation of</b> <code>{pkg_name}</code> <b>returned an error</b>.<br><br>
        Try restarting Cell-ACDC. If it doesn't work, 
        please close Cell-ACDC and, with the <code>acdc</code> <b>environment ACTIVE</b>, 
        install <code>{pkg_name}</code> manually using the follwing command:<br><br>
        {install_command}<br><br>
        Thank you for your patience.
    """)
    msg = widgets.myMessageBox()
    msg.critical(parent, f"{pkg_name} installation failed", txt)
    print("*" * 50)
    print(
        f'[ERROR]: Installation of "{pkg_name}" failed. '
        f"Please, close Cell-ACDC and run the command "
        f"{pip_prefix} --upgrade {pkg_name}`"
    )
    print("^" * 50)


def _install_package_msg(
    pkg_name,
    note="",
    parent=None,
    upgrade=False,
    caller_name="Cell-ACDC",
    is_cli=False,
    pkg_command="",
    logger_func=print,
    exact_version="",
    max_version="",
    min_version="",
    installer: Literal["pip", "conda"] = "pip",
    include_higher_version: bool = False,
    include_lower_version: bool = False,
):
    if is_cli:
        proceed = _install_package_cli_msg(
            pkg_name,
            note=note,
            upgrade=upgrade,
            caller_name=caller_name,
            pkg_command=pkg_command,
            exact_version=exact_version,
            max_version=max_version,
            min_version=min_version,
            logger_func=logger_func,
            installer=installer,
            include_higher_version=include_higher_version,
            include_lower_version=include_lower_version,
        )
    else:
        proceed = _install_package_gui_msg(
            pkg_name,
            note=note,
            parent=parent,
            upgrade=upgrade,
            caller_name=caller_name,
            pkg_command=pkg_command,
            exact_version=exact_version,
            max_version=max_version,
            min_version=min_version,
            logger_func=logger_func,
            installer=installer,
            including_higher_version=include_higher_version,
            including_lower_version=include_lower_version,
        )
    return proceed


def _install_pytorch_cli(caller_name="Cell-ACDC", action="install", logger_func=print):
    separator = "-" * 60
    txt = (
        f"{separator}\n{caller_name} needs to {action} PyTorch\n\n"
        "You can choose to install it now or stop the process and install it "
        "later. To install it correctly, we need to know your preferences.\n"
    )
    logger_func(txt)
    questions = {
        "Choose your OS:": ("Windows", "Mac", "Linux"),
        "Package manager:": ("Pip"),
        "Compute platform:": (
            "CPU",
            "CUDA 11.8 (NVIDIA GPU)",
            "CUDA 12.1 (NVIDIA GPU)",
        ),
    }
    selected_command = get_pytorch_command()
    selected_preferences = []
    for question, choices in questions.items():
        input_txt = get_cli_multi_choice_question(question, choices)
        while True:
            answer = input(input_txt)
            if answer.lower() == "q":
                exit("Execution stopped by the user.")

            try:
                idx = int(answer) - 1
                if idx >= len(choices):
                    raise TypeError("Not a valid answer")
            except Exception as err:
                print("-" * 100)
                logger_func(
                    f'"{answer}" is not a valid answer.'
                    'Choose one of the options or "q" to quit.'
                )
                print("^" * 100)
                continue

            preference = choices[idx]
            selected_command = selected_command[preference]
            selected_preferences.append(preference)
            print("")
            break

    print("-" * 100)
    selected_preferences = ", ".join(selected_preferences)
    logger_func(f"Selected preferences: {selected_preferences}")
    print("-" * 100)
    logger_func(f"Command:\n\n{selected_command}\n")
    while True:
        answer = input("Do you want to run the command now ([y]/n)?: ")
        if answer.lower() == "n":
            exit("Execution stopped by the user.")

        if answer.lower() == "y" or not answer:
            break

        print("-" * 100)
        print(f'"{answer}" is not a valid answer. Choose "y" for yes or "n" for no.')
        print("^" * 100)

    if selected_command.startswith("conda"):
        try:
            subprocess.check_call([selected_command], shell=True)
        except Exception as err:
            cmd_list = selected_command.split()
            cmd_list = [cmd.strip('"') for cmd in cmd_list]
            cmd_list = [cmd.strip("'") for cmd in cmd_list]
            cmd_list = [cmd.lstrip(".") for cmd in cmd_list]
            subprocess.check_call(cmd_list, shell=True)
    else:
        cmd_list = selected_command.split()[1:]
        cmd_list = [cmd.strip('"') for cmd in cmd_list]
        cmd_list = [cmd.strip("'") for cmd in cmd_list]
        cmd_list = [cmd.lstrip(".") for cmd in cmd_list]
        subprocess.check_call([sys.executable, *cmd_list], shell=True)


def _get_pkg_command_pip_install(
    pkg_command,
    exact_version="",
    max_version="",
    min_version="",
    including_lower_version=False,
    including_higher_version=False,
):
    if exact_version:
        pkg_command = f"{pkg_command}=={exact_version}"
        return pkg_command

    if including_higher_version:
        sign_max = "<="
    else:
        sign_max = "<"
    if including_lower_version:
        sign_min = ">="
    else:
        sign_min = ">"
    if min_version:
        pkg_command = f"{pkg_command}{sign_min}{min_version}"
        if max_version:
            pkg_command = f"{pkg_command},"

    if max_version:
        pkg_command = f"{pkg_command}{sign_max}{max_version}"

    return pkg_command


def _install_package_cli_msg(
    pkg_name,
    note="",
    upgrade=False,
    caller_name="Cell-ACDC",
    logger_func=print,
    pkg_command="",
    exact_version="",
    max_version="",
    min_version="",
    installer: Literal["pip", "conda"] = "pip",
    include_lower_version=False,
    include_higher_version=False,
):
    if not pkg_command:
        pkg_command = pkg_name

    pkg_command = _get_pkg_command_pip_install(
        pkg_command,
        exact_version=exact_version,
        max_version=max_version,
        min_version=min_version,
        including_lower_version=include_lower_version,
        including_higher_version=include_higher_version,
    )

    if upgrade:
        action = "upgrade"
    else:
        action = "install"

    conda_prefix, pip_prefix = get_pip_conda_prefix()

    if installer == "pip":
        install_command = f"{pip_prefix} --upgrade {pkg_command}"
    elif installer == "conda":
        install_command = f"{conda_prefix} {pkg_command}"

    separator = "-" * 60
    txt = (
        f"{separator}\n{caller_name} needs to {action} {pkg_name}\n\n"
        "You can choose to install it now or stop the process and install it "
        "later with the following command:\n\n"
        f"{install_command}\n"
    )
    logger_func(txt)

    while True:
        answer = try_input_install_package(pkg_name, install_command)
        if not answer or answer.lower() == "y":
            return True

        if answer.lower() == "n":
            return False

        logger_func(
            f'{answer} is not a valid answer. Valid answers are "y" for Yes and '
            '"n" for No.'
        )


def _install_package_gui_msg(
    pkg_name,
    note="",
    parent=None,
    upgrade=False,
    caller_name="Cell-ACDC",
    pkg_command="",
    logger_func=None,
    exact_version="",
    max_version="",
    min_version="",
    including_lower_version=False,
    including_higher_version=False,
    installer: Literal["pip", "conda"] = "pip",
):
    msg = widgets.myMessageBox(parent=parent)
    if upgrade:
        install_text = "upgrade"
    else:
        install_text = "install"
    if pkg_name == "BayesianTracker":
        pkg_name = "btrack"

    if not pkg_command:
        pkg_command = pkg_name

    pkg_command = _get_pkg_command_pip_install(
        pkg_command,
        exact_version=exact_version,
        max_version=max_version,
        min_version=min_version,
        including_lower_version=including_lower_version,
        including_higher_version=including_higher_version,
    )

    conda_prefix, pip_prefix = get_pip_conda_prefix()

    if installer == "pip":
        command = f"{pip_prefix} --upgrade {pkg_command}"
    elif installer == "conda":
        command = f"{conda_prefix} {pkg_command}"

    command_html = command.lower().replace("<", "&lt;").replace(">", "&gt;")

    txt = html_utils.paragraph(f"""
        {caller_name} is going to <b>download and {install_text}</b>
        <code>{pkg_name}</code>.<br><br>
        Make sure you have an <b>active internet connection</b>,
        before continuing.<br>
        Progress will be displayed on the terminal<br><br>
        You might have to <b>restart {caller_name}</b>.<br><br>
        Alternatively, you can cancel the process and try later.<br><br>
        To install later, or if the installation fails, run the following 
        command:
    """)
    if note:
        txt = f"{txt}{note}"
    _, okButton = msg.information(
        parent,
        f"Install {pkg_name}",
        txt,
        buttonsTexts=("Cancel", "Ok"),
        commands=(command_html,),
    )
    return msg.clickedButton == okButton


def _install_tensorflow(max_version="", min_version=""):
    cpu = platform.processor()
    pkg_command = _get_pkg_command_pip_install(
        "tensorflow", max_version=max_version, min_version=min_version
    )
    conda_prefix, pip_prefix = get_pip_conda_prefix()

    if is_mac and cpu == "arm":
        args = [f'{conda_prefix} "{pkg_command}"']
        shell = True
    else:
        args = [sys.executable, "-m", "pip", "install", "-U", pkg_command]
        shell = False
    subprocess.check_call(args, shell=shell)

    # purge numpy
    purge_module("numpy")


def _install_segment_anything():
    args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        "--use-pep517",
        "git+https://github.com/facebookresearch/segment-anything.git",
    ]
    subprocess.check_call(args)


def _install_sam2():
    args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-U",
        "--use-pep517",
        "git+https://github.com/facebookresearch/sam2.git",
    ]
    subprocess.check_call(args)


def _install_deepsea():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deepsea"])


def get_pip_conda_prefix(list_return=False):
    from .config import parser_args

    try:
        cp = parser_args
        if cp["install_details"] is not None:
            no_cli_install = True
            install_details = cp["install_details"]
            venv_path = install_details["venv_path"]
            conda_path = install_details["conda_path"]
            if " " not in conda_path:
                conda_path = conda_path.strip('"').strip("'")
        else:
            no_cli_install = False
    except:
        no_cli_install = False
        pass

    if no_cli_install:
        conda_prefix = f"{conda_path} install -y -p {venv_path} -c conda-forge"
        exec_path = sys.executable
        if " " in exec_path:
            exec_path = f'"{exec_path}"'
        pip_prefix = f"{exec_path} -m pip install"
    else:
        conda_prefix = "conda install -y -c conda-forge"
        pip_prefix = "pip install"

    pip_list = [sys.executable, "-m", "pip", "install"]
    if no_cli_install:
        conda_list = [
            conda_path.strip('"').strip("'"),
            "install",
            "-y",
            "-p",
            venv_path.strip('"').strip("'"),
            "-c",
            "conda-forge",
        ]
    else:
        conda_list = ["conda", "install", "-y", "-c", "conda-forge"]
    if list_return:
        return conda_list, pip_list
    else:
        return conda_prefix, pip_prefix


def _warn_install_gpu(model_name, ask_installs, qparent=None):

    cellpose_cuda_url = (
        r"https://github.com/mouseland/cellpose#gpu-version-cuda-on-windows-or-linux"
    )
    torch_cuda_url = r"https://pytorch.org/get-started/locally/"
    direct_ml_url = r"https://microsoft.github.io/DirectML/"
    torch_directml_url = (
        r"https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows"
    )

    cellpose_href = f"{html_utils.href_tag('here', cellpose_cuda_url)}"
    torch_href = f"{html_utils.href_tag('here', torch_cuda_url)}"
    direct_ml_href = f"{html_utils.href_tag('direct_ml_DirectMLref', direct_ml_url)}"
    torch_directml_href = (
        f"{html_utils.href_tag('directml pytorch', torch_directml_url)}"
    )

    conda_prefix, pip_prefix = get_pip_conda_prefix()

    msg = widgets.myMessageBox(showCentered=False, wrapText=False)
    txt = html_utils.paragraph(f"""
        In order to use <code>{model_name}</code> with the GPU you need 
        to install a <b>PyTorch version which can use it</b>.<br>
        We recomment using CUDA over DirectML, but if you are using a Windows
        machine with an AMD GPU, you can use DirectML.<br>
        """)
    txt_cuda_title = html_utils.paragraph(f"<b>CUDA</b>", font_size="18px")

    pip_prefix = pip_prefix.replace("install -y", "uninstall")
    txt_cuda = html_utils.paragraph(f"""
        Check out these instructions {cellpose_href}, and {torch_href}.<br>
        First, uninstall the CPU version of PyTorch with the following command:
        <copiable>{pip_prefix} uninstall torch</copiable>
        <br>Then, install the CUDA version required by your GPU with the following 
        command (in this case 12.8):
        <copiable>{pip_prefix} torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128</copiable>
        <br>
        """)

    add_info = html_utils.to_admonition(
        f"""
        Pleae use the following table to find the correct link for the command.
        You can check the highest CUDA  <br> version supported on your system with the
        command <code>nvidia-smi</code> in the terminal.<br>

        {html_utils.table_style_header}
            <tr>
                <th>CUDA Version</th>
                <th>PyTorch Installation Link</th>
            </tr>
            <tr>
                <td>CUDA 11.8</td>
                <td><code>https://download.pytorch.org/whl/cu118</code></td>
            </tr>
            <tr>
                <td>CUDA 12.6</td>
                <td><code>https://download.pytorch.org/whl/cu126</code></td>
            </tr>
            <tr>
                <td>CUDA 12.8</td>
                <td><code>https://download.pytorch.org/whl/cu128</code></td>
            </tr>
        </table>
        """,
        "info",
    )

    txt_cuda = f"{txt_cuda}{add_info}"

    txt_directML_title = html_utils.paragraph(f"<b>DirectML</b>", font_size="18px")
    txt_directML = html_utils.paragraph(f"""
        Check out {direct_ml_href}, and {torch_directml_href} for more info.<br>
        Only supported on Windows 10/11 with Python 3.8-3.12.<br>
        Click the <b>Install DirectML</b> button to install DirectML.
        <br><br>
        """)

    txt_end = html_utils.paragraph(f"""
        How do you want to proceed?
    """)

    stopButton = widgets.cancelPushButton("Stop the process")
    directMLButton = widgets.okPushButton("Install DirectML")
    proceedButton = widgets.okPushButton("Proceed without GPU")

    buttons = [stopButton]

    if "cuda" in ask_installs:
        txt = f"{txt}{txt_cuda_title}{txt_cuda}"
    if "directML" in ask_installs:
        txt = f"{txt}{txt_directML_title}{txt_directML}"
        buttons.append(directMLButton)
    txt = f"{txt}{txt_end}"
    buttons.append(proceedButton)

    msg.warning(
        qparent,
        "PyTorch GPU version not installed",
        txt,
        buttonsTexts=buttons,
    )

    if msg.cancel:
        return False, False

    if msg.clickedButton == directMLButton:
        py_ver = sys.version_info
        if is_win and py_ver.major == 3 and py_ver.minor < 13:
            success = check_install_package(
                pkg_name="torch-directml",
                import_pkg_name="torch_directml",
                pypi_name="torch-directml",
                return_outcome=True,
            )
            purge_module("torch")
            return success, True
        else:
            msg = widgets.myMessageBox()
            msg.warning(
                qparent,
                "DirectML not supported",
                "DirectML is only supported on Python 3.8-3.12 and Windows 10/11",
            )
            return False, False

    if msg.clickedButton == stopButton:
        return False, False

    if msg.clickedButton == proceedButton:
        return True, False


def check_gpu_requested_segm_model(init_kwargs):
    gpu = init_kwargs.get("gpu", False)
    if gpu:
        return True

    device_type = init_kwargs.get("device_type", "cpu")
    return device_type == "gpu" or device_type == ""


def check_gpu_available(
    model_name,
    use_gpu,
    do_not_warn=False,
    qparent=None,
    cuda=False,
    directML=False,
    return_available_gpu_type=False,
):
    if not use_gpu:
        if return_available_gpu_type:
            return True, []
        else:
            return True

    ask_for_cuda = False
    if cuda:
        try:
            import torch

            if not torch.cuda.is_available():
                ask_for_cuda = True
            if not torch.cuda.device_count() > 0:
                ask_for_cuda = True
        except ModuleNotFoundError:
            ask_for_cuda = True

    ask_for_directML = False
    if directML:
        if is_win:
            try:
                import torch_directml

                if not torch_directml.is_available():
                    ask_for_directML = True
            except ModuleNotFoundError:
                ask_for_directML = True

    frameworks = _available_frameworks(model_name)
    ask_installs = set() if not ask_for_cuda else {"cuda"}
    ask_installs.update({"directML"} if ask_for_directML else set())
    framework_available = False
    available_frameworks_list = []
    for framework, model_compatible in frameworks.items():
        if not model_compatible:
            continue
        if framework == "cuda":
            import torch

            if not torch.cuda.is_available():
                ask_installs.add("cuda")
            elif not torch.cuda.device_count() > 0:
                ask_installs.add("cuda")
            else:
                framework_available = True
                available_frameworks_list.append("cuda")
        elif framework == "directML":
            if is_win:
                try:
                    import torch_directml

                    if not torch_directml.is_available():
                        ask_installs.add("directML")
                    else:
                        framework_available = True
                        available_frameworks_list.append("directML")
                except ModuleNotFoundError:
                    ask_installs.add("directML")
        elif is_mac_arm64:
            framework_available = True
            break

    if framework_available and not ask_for_cuda and not ask_for_directML:
        if return_available_gpu_type:
            return True, available_frameworks_list
        else:
            return True

    elif do_not_warn:
        if return_available_gpu_type:
            return False, available_frameworks_list
        else:
            return False

    proceed, directML_installed = _warn_install_gpu(
        model_name, ask_installs, qparent=qparent
    )
    if return_available_gpu_type:
        if directML_installed:
            available_frameworks_list.append("directML")
        return proceed, available_frameworks_list
    else:
        return proceed


def get_pip_install_cellacdc_version_command(version=None):
    conda_prefix, pip_prefix = get_pip_conda_prefix()

    if version is None:
        version = read_version()
    commit_hash_idx = version.find("+g")
    is_dev_version = commit_hash_idx > 0
    if is_dev_version:
        commit_hash = version[commit_hash_idx + 2 :].split(".")[0]
        command = f'{pip_prefix} --upgrade "git+{github_home_url}.git@{commit_hash}"'
        command_github = None
    else:
        command = f"{pip_prefix} --upgrade cellacdc=={version}"
        command_github = f'{pip_prefix} --upgrade "git+{urls.github_url}@{version}"'
    return command, command_github


def check_install_tapir():
    check_install_package(
        "tapnet", pypi_name="git+https://github.com/ElpadoCan/TAPIR.git"
    )


def check_install_trackastra():
    check_install_package(
        "Trackastra", import_pkg_name="trackastra", pypi_name="trackastra"
    )


def get_torch_device(gpu=False):
    import torch

    if torch.cuda.is_available() and gpu:
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def check_install_instanseg():
    check_install_package(
        pkg_name="InstanSeg", import_pkg_name="instanseg", pypi_name="instanseg-torch"
    )


def get_pytorch_command():
    """Get the command to install pytorch CPU or CUDA

    Returns
    -------
    dict
        Dictionary mapping OS to commands for installing PyTorch

    Notes
    -----
    As of Oct 2024, the `pytorch` channel on Anaconda was deprecated.
    See here https://github.com/pytorch/pytorch/issues/138506
    """
    conda_prefix, pip_prefix = get_pip_conda_prefix()

    pytorch_commands = {
        "Windows": {
            # 'Conda': {
            #     'CPU': f'{conda_prefix} pytorch torchvision cpuonly -c conda-forge',
            #     'CUDA 11.8 (NVIDIA GPU)': f'{conda_prefix} pytorch torchvision pytorch-cuda=11.8 -c conda-forge -c nvidia',
            #     'CUDA 12.1 (NVIDIA GPU)': f'{conda_prefix} pytorch torchvision pytorch-cuda=12.1 -c conda-forge -c nvidia'
            # },
            "Pip": {
                "CPU": f"{pip_prefix} torch torchvision",
                "CUDA 11.8 (NVIDIA GPU)": f"{pip_prefix} torch torchvision --index-url https://download.pytorch.org/whl/cu118",
                "CUDA 12.1 (NVIDIA GPU)": f"{pip_prefix} torch torchvision --index-url https://download.pytorch.org/whl/cu121",
            }
        },
        "Mac": {
            # 'Conda': {
            #     'CPU': f'{conda_prefix} pytorch torchvision cpuonly -c conda-forge',
            #     'CUDA 11.8 (NVIDIA GPU)': '[WARNING]: CUDA is not available on MacOS',
            #     'CUDA 12.1 (NVIDIA GPU)': '[WARNING]: CUDA is not available on MacOS'
            # },
            "Pip": {
                "CPU": f"{pip_prefix} torch torchvision",
                "CUDA 11.8 (NVIDIA GPU)": "[WARNING]: CUDA is not available on MacOS",
                "CUDA 12.1 (NVIDIA GPU)": "[WARNING]: CUDA is not available on MacOS",
            }
        },
        "Linux": {
            # 'Conda': {
            #     'CPU': f'{conda_prefix} pytorch torchvision cpuonly -c conda-forge',
            #     'CUDA 11.8 (NVIDIA GPU)': f'{conda_prefix} pytorch torchvision pytorch-cuda=11.8 -c conda-forge -c nvidia',
            #     'CUDA 12.1 (NVIDIA GPU)': f'{conda_prefix} pytorch torchvision pytorch-cuda=12.1 -c conda-forge -c nvidia'
            # },
            "Pip": {
                "CPU": f"{pip_prefix} torch torchvision --index-url https://download.pytorch.org/whl/cpu",
                "CUDA 11.8 (NVIDIA GPU)": f"{pip_prefix} torch torchvision --index-url https://download.pytorch.org/whl/cu118",
                "CUDA 12.1 (NVIDIA GPU)": f"{pip_prefix} torch torchvision",
            }
        },
    }

    return pytorch_commands


def get_package_info(package_name):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True,
            check=True,
        )

        info = {}
        for line in result.stdout.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()

        # Check if it's editable by looking at the location
        location = info.get("Location", "")
        editable_location = info.get("Editable project location", "")

        return {
            "installed": True,
            "editable": bool(editable_location),
            "location": location,
            "editable_location": editable_location,
        }

    except subprocess.CalledProcessError:
        return {"installed": False, "editable": False}


def update_package(parent, package_name):
    package_info = get_package_info(package_name)
    if not package_info["installed"]:
        printl(f"Package {package_name} is not installed.")
        return False
    editable = package_info.get("editable", False)
    if editable:
        return update_editable_package(parent, package_name, package_info)
    else:
        return update_not_editable_package(package_name, package_info)


def update_editable_package(parent, package_name, package_info):
    repo_location = package_info.get("editable_location", "")

    if not repo_location or not os.path.exists(repo_location):
        print(f"Repository location not found for {package_name}")
        return False

    return _update_repo_with_git_command(package_name, repo_location)


def update_not_editable_package(package_name, package_info):
    """Update a non-editable package using pip"""
    try:
        _, pip_list = get_pip_conda_prefix(list_return=True)
        command = pip_list + ["--upgrade ", package_name]

        print(f"Updating {package_name} using pip...")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"Successfully updated {package_name}")
            return True
        else:
            print(f"Failed to update {package_name}: {result.stderr}")
            return False

    except Exception as e:
        print(f"Error updating {package_name}: {e}")
        return False

# Sibling imports (deferred to avoid import cycles)
from .misc import (
    _apt_gcc_command,
    _apt_update_command,
    _available_frameworks,
    _jdk_exists,
    _run_command,
    _subprocess_run_command,
    cpp_windows_url,
    extract_zip,
    is_gui_running,
    jdk_windows_url,
    purge_module,
)
from .models import (
    download_url,
)
from .paths import (
    get_acdc_java_path,
)
from .qt import (
    get_cli_multi_choice_question,
)
from .version import (
    _update_repo_with_git_command,
    check_cellpose_version,
    check_pkg_exact_version,
    check_pkg_max_version,
    check_pkg_version,
    read_version,
)

