"""Cell-ACDC utility helpers: misc."""

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

def get_module_name(script_file_path):
    parts = pathlib.Path(script_file_path).parts
    parts = list(parts[parts.index("cellacdc") + 1 :])
    parts[-1] = os.path.splitext(parts[-1])[0]
    module = ".".join(parts)
    return module


def filterCommonStart(images_path):
    startNameLen = 6
    ls = listdir(images_path)
    if not ls:
        return []
    allFilesStartNames = [f[:startNameLen] for f in ls]
    mostCommonStart = Counter(allFilesStartNames).most_common(1)[0][0]
    commonStartFilenames = [f for f in ls if f.startswith(mostCommonStart)]
    return commonStartFilenames


def remove_known_extension(name):
    for ext in KNOWN_EXTENSIONS:
        if name.endswith(ext):
            return name[: -len(ext)], ext

    return name, ""


def getCustomAnnotTooltip(annotState):
    toolTip = (
        f"Name: {annotState['name']}\n\n"
        f"Type: {annotState['type']}\n\n"
        f"Usage: activate the button and RIGHT-CLICK on cell to annotate\n\n"
        f"Description: {annotState['description']}\n\n"
        f'SHORTCUT: "{annotState["shortcut"]}"'
    )
    return toolTip


def is_iterable(item):
    try:
        iter(item)
        return True
    except TypeError as e:
        return False


class utilClass:
    pass


class StdErr:
    def __init__(self, logger: Logger = None):
        self._sys_stderr = sys.stderr
        self._err_msg_line_buffer = []
        self._logger = logger

    def write(self, text: str):
        if text.startswith("Traceback"):
            print("-" * 100)

        self._sys_stderr.write(text)

        if not text:
            return

        self._err_msg_line_buffer.append(text)
        if not text.endswith("\n"):
            return

        # If the line ends with a newline, flush the buffer
        err_line = "".join(self._err_msg_line_buffer)
        if self._logger is not None:
            self._logger.plain(err_line, write_to_stdout=False)
        else:
            print(err_line)

        self._err_msg_line_buffer = []

    def flush(self):
        self._sys_stderr.flush()

    def close(self):
        """Close the StdErr stream"""
        sys.stderr = self._sys_stderr


def getMostRecentPath():
    if os.path.exists(recentPaths_path):
        df = pd.read_csv(recentPaths_path, index_col="index")
        if "opened_last_on" in df.columns:
            df = df.sort_values("opened_last_on", ascending=False)
        MostRecentPath = ""
        for path in df["path"]:
            if os.path.exists(path):
                MostRecentPath = path
                break
    else:
        MostRecentPath = ""
    return MostRecentPath


def addToRecentPaths(exp_path, logger=None):
    if not os.path.exists(exp_path):
        return
    exp_path = exp_path.replace("\\", "/")
    if os.path.exists(recentPaths_path):
        try:
            df = pd.read_csv(recentPaths_path, index_col="index")
            recentPaths = df["path"].to_list()
            if "opened_last_on" in df.columns:
                openedOn = df["opened_last_on"].to_list()
            else:
                openedOn = [np.nan] * len(recentPaths)
            if exp_path in recentPaths:
                pop_idx = recentPaths.index(exp_path)
                recentPaths.pop(pop_idx)
                openedOn.pop(pop_idx)
            recentPaths.insert(0, exp_path)
            openedOn.insert(0, datetime.datetime.now())
            # Keep max 40 recent paths
            if len(recentPaths) > 40:
                recentPaths.pop(-1)
                openedOn.pop(-1)
        except Exception as e:
            recentPaths = [exp_path]
            openedOn = [datetime.datetime.now()]
    else:
        recentPaths = [exp_path]
        openedOn = [datetime.datetime.now()]
    df = pd.DataFrame(
        {
            "path": recentPaths,
            "opened_last_on": pd.Series(openedOn, dtype="datetime64[ns]"),
        }
    )
    df.index.name = "index"
    df.to_csv(recentPaths_path)


def checkDataIntegrity(filenames, parent_path, parentQWidget=None):
    if not filenames:
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            "Cell-ACDC could not find any files in the folder "
            f"<b>{parent_path}</b>.<br><br>"
            "Please make sure that the folder contains at least one image file.<br><br>"
            "Thank you for your patience!"
        )
        msg.warning(parentQWidget, "Selected folder is emppty", txt)
        raise FileNotFoundError(f"No files found in the folder {parent_path}. ")

    char = filenames[0][:2]
    startWithSameChar = all([f.startswith(char) for f in filenames])
    if not startWithSameChar:
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(
            "Cell-ACDC detected files inside the folder "
            "that <b>do not start with the same, common basename</b>.<br><br>"
            "To ensure correct loading of the data, the folder where "
            "the file(s) is/are should either contain a single image file or"
            "only files that start with the same, common basename.<br><br>"
            "For example the following filenames:<br><br>"
            "<code>F014_s01_phase_contr.tif</code><br>"
            "<code>F014_s01_mCitrine.tif</code><br><br>"
            "are named correctly since they all start with the "
            'the common basename "F014_s01_". After the common basename you '
            'can write whatever text you want. In the example above, "phase_contr" '
            'and "mCitrine" are the channel names.<br><br>'
            "Data loading may still be successfull, so Cell-ACDC will "
            "still try to load data now.<br>"
        )
        filesFormat = [f"    - {file}" for file in filenames]
        filesFormat = "\n".join(filesFormat)
        detailsText = f"Files present in the folder {parent_path}:\n\n{filesFormat}"
        msg.addShowInFileManagerButton(parent_path, txt="Open folder...")
        msg.warning(
            parentQWidget,
            "Data structure compromised",
            txt,
            detailsText=detailsText,
            buttonsTexts=("Cancel", "Ok"),
        )
        if msg.cancel:
            raise TypeError("Process aborted by the user.")
        return False
    return True


def is_in_bounds(x, y, X, Y):
    in_bounds = x >= 0 and x < X and y >= 0 and y < Y
    return in_bounds


def showInExplorer(path):
    if is_mac:
        os.system(f'open "{path}"')
    elif is_linux:
        os.system(f'xdg-open "{path}"')
    else:
        os.startfile(path)


def exec_time(func):
    @wraps(func)
    def inner_function(self, *args, **kwargs):
        t0 = time.perf_counter()
        if func.__code__.co_argcount == 1 and func.__defaults__ is None:
            result = func(self)
        elif func.__code__.co_argcount > 1 and func.__defaults__ is None:
            result = func(self, *args)
        else:
            result = func(self, *args, **kwargs)
        t1 = time.perf_counter()
        s = f"{func.__name__} execution time = {(t1 - t0) * 1000:.3f} ms"
        printl(s, is_decorator=True)
        return result

    return inner_function


def setRetainSizePolicy(widget, retain=True):
    sp = widget.sizePolicy()
    sp.setRetainSizeWhenHidden(retain)
    widget.setSizePolicy(sp)


def getAcdcDfSegmPaths(images_path):
    ls = listdir(images_path)
    basename = getBasename(ls)
    paths = {}
    for file in ls:
        filePath = os.path.join(images_path, file)
        fileName, ext = os.path.splitext(file)
        endName = fileName[len(basename) :]
        if endName.find("acdc_output") != -1 and ext == ".csv":
            info_name = endName.replace("acdc_output", "")
            paths.setdefault(info_name, {})
            paths[info_name]["acdc_df_path"] = filePath
            paths[info_name]["acdc_df_filename"] = fileName
        elif endName.find("segm") != -1 and ext == ".npz":
            info_name = endName.replace("segm", "")
            paths.setdefault(info_name, {})
            paths[info_name]["segm_path"] = filePath
            paths[info_name]["segm_filename"] = fileName
    return paths


def getChannelFilePath(images_path, chName):
    file = ""
    alignedFilePath = ""
    tifFilePath = ""
    h5FilePath = ""
    for file in listdir(images_path):
        filePath = os.path.join(images_path, file)
        if file.endswith(f"{chName}_aligned.npz"):
            alignedFilePath = filePath
        elif file.endswith(f"{chName}.tif"):
            tifFilePath = filePath
        elif file.endswith(f"{chName}.h5"):
            h5FilePath = filePath
    if alignedFilePath:
        return alignedFilePath
    elif h5FilePath:
        return h5FilePath
    elif tifFilePath:
        return tifFilePath
    else:
        return ""


def get_chname_from_basename(filename, basename, remove_ext=True):
    if remove_ext:
        filename, ext = os.path.splitext(filename)
    chName = filename[len(basename) :]
    aligned_idx = chName.find("_aligned")
    if aligned_idx != -1:
        chName = chName[:aligned_idx]
    return chName


def getBaseAcdcDf(rp):
    zeros_list = [0] * len(rp)
    nones_list = [None] * len(rp)
    minus1_list = [-1] * len(rp)
    IDs = []
    xx_centroid = []
    yy_centroid = []
    zz_centroid = []
    for obj in rp:
        xc, yc = obj.centroid[-2:]
        IDs.append(obj.label)
        xx_centroid.append(xc)
        yy_centroid.append(yc)
        if len(obj.centroid) == 3:
            zc = obj.centroid[0]
            zz_centroid.append(zc)

    df = pd.DataFrame(
        {
            "Cell_ID": IDs,
            "is_cell_dead": zeros_list,
            "is_cell_excluded": zeros_list,
            "x_centroid": xx_centroid,
            "y_centroid": yy_centroid,
            "was_manually_edited": minus1_list,
        }
    ).set_index("Cell_ID")
    if zz_centroid:
        df["z_centroid"] = zz_centroid

    return df


def getBasenameAndChNames(images_path, useExt=None):
    _tempPosData = utilClass()
    _tempPosData.images_path = images_path
    load.loadData.getBasenameAndChNames(_tempPosData, useExt=useExt)
    return _tempPosData.basename, _tempPosData.chNames


def getBasename(files):
    basename = files[0]
    for file in files:
        # Determine the basename based on intersection of all files
        _, ext = os.path.splitext(file)
        sm = difflib.SequenceMatcher(None, file, basename)
        i, j, k = sm.find_longest_match(0, len(file), 0, len(basename))
        basename = file[i : i + k]
    return basename


def findalliter(patter, string):
    """Function used to return all re.findall objects in string"""
    m_test = re.findall(r"(\d+)_(.+)", string)
    m_iter = [m_test]
    while m_test:
        m_test = re.findall(r"(\d+)_(.+)", m_test[0][1])
        m_iter.append(m_test)
    return m_iter


def clipSelemMask(mask, shape, Yc, Xc, copy=True):
    if copy:
        mask = mask.copy()

    Y, X = shape
    h, w = mask.shape

    # Bottom, Left, Top, Right global coordinates of mask
    Y0, X0, Y1, X1 = Yc - (h / 2), Xc - (w / 2), Yc + (h / 2), Xc + (w / 2)
    mask_limits = [floor(Y0) + 1, floor(X0) + 1, floor(Y1) + 1, floor(X1) + 1]

    if Y0 >= 0 and X0 >= 0 and Y1 <= Y and X1 <= X:
        # Mask is withing shape boundaries, no need to clip
        ystart, xstart, yend, xend = mask_limits
        mask_slice = slice(ystart, yend), slice(xstart, xend)
        return mask, mask_slice

    if Y0 < 0:
        # Mask is exceeding at the bottom
        ystart = floor(abs(Y0))
        mask_limits[0] = 0
        mask = mask[ystart:]
    if X0 < 0:
        # Mask is exceeding at the left
        xstart = floor(abs(X0))
        mask_limits[1] = 0
        mask = mask[:, xstart:]
    if Y1 > Y:
        # Mask is exceeding at the top
        yend = ceil(abs(Y1)) - Y
        mask_limits[2] = Y
        mask = mask[:-yend]
    if X1 > X:
        # Mask is exceeding at the right
        xend = ceil(abs(X1)) - X
        mask_limits[3] = X
        mask = mask[:, :-xend]

    ystart, xstart, yend, xend = mask_limits
    mask_slice = slice(ystart, yend), slice(xstart, xend)
    return mask, mask_slice


def get_function_argspec(
    function,
    args_to_skip={
        "logger_func",
    },
):
    argspecs = inspect.getfullargspec(function)
    kwargs_type_hints = typing.get_type_hints(function)
    docstring = function.__doc__
    params = params_to_ArgSpec(
        argspecs, kwargs_type_hints, docstring, args_to_skip=args_to_skip
    )
    return params


def _get_doc_stop_idx(docstring, start_idx, next_param_name=None, debug=False):
    if debug:
        import pdb

        pdb.set_trace()

    if next_param_name is not None:
        doc_stop_idx = docstring.find(f"{next_param_name} : ")
        if doc_stop_idx > 1:
            return doc_stop_idx

    docstring_from_start = docstring[start_idx:]
    next_param_searched = re.search(r"\w+ : ", docstring_from_start)
    if next_param_searched is not None:
        return next_param_searched.start(0) + start_idx

    doc_stop_idx = docstring.find("Returns")
    if doc_stop_idx > 1:
        return doc_stop_idx

    doc_stop_idx = docstring.find("Notes")
    if doc_stop_idx > 1:
        return doc_stop_idx

    return -1


def add_segm_data_param(init_params, init_argspecs):
    if init_argspecs.defaults is None:
        num_kwargs = 0
    else:
        num_kwargs = len(init_argspecs.defaults)

    # Segm model requires segm data --> add it to params
    num_args = len(init_argspecs.args) - num_kwargs
    if num_args == 1:
        # Args is only self --> segm data not needed
        return init_params

    desc = (
        "This model requires an additional segmentation file as input.\n\n"
        "Please, select which segmentation file to provide to the model."
    )

    segm_data_argspec = ArgSpec(
        name="Auxiliary segmentation file",
        default="",
        type=str,
        desc=desc,
        docstring=None,
    )

    init_params.insert(0, segm_data_argspec)
    return init_params


def getDefault_SegmInfo_df(posData, filename):
    mid_slice = int(posData.SizeZ / 2)
    df = pd.DataFrame(
        {
            "filename": [filename] * posData.SizeT,
            "frame_i": range(posData.SizeT),
            "z_slice_used_dataPrep": [mid_slice] * posData.SizeT,
            "which_z_proj": ["single z-slice"] * posData.SizeT,
            "z_slice_used_gui": [mid_slice] * posData.SizeT,
            "which_z_proj_gui": ["single z-slice"] * posData.SizeT,
            "resegmented_in_gui": [False] * posData.SizeT,
            "is_from_dataPrep": [False] * posData.SizeT,
        }
    ).set_index(["filename", "frame_i"])
    return df


def _jdk_exists(jre_path):
    # If jre_path exists and it's windows search for ~/acdc-java/win64/jdk
    # or ~/.acdc-java/win64/jdk. If not Windows return jre_path
    if not jre_path:
        return ""
    os_acdc_java_path = os.path.dirname(jre_path)
    os_foldername = os.path.basename(os_acdc_java_path)
    if not os_foldername.startswith("win"):
        return jre_path
    if os.path.exists(os_acdc_java_path):
        for folder in os.listdir(os_acdc_java_path):
            if not folder.startswith("jdk"):
                continue
            dir_path = os.path.join(os_acdc_java_path, folder)
            for file in os.listdir(dir_path):
                if file == "bin":
                    return dir_path
    return ""


def showUserManual():
    manual_file_path = download_manual()
    showInExplorer(manual_file_path)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def extract_zip(zip_path, extract_to_path, verbose=True):
    if verbose:
        print(f"Extracting to {extract_to_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to_path)


def get_tiff_metadata(
    image_arr,
    SizeT=None,
    SizeZ=None,
    PhysicalSizeZ=None,
    PhysicalSizeX=None,
    PhysicalSizeY=None,
    TimeIncrement=None,
):
    SizeY, SizeX = image_arr.shape[-2:]
    Type = str(image_arr.dtype)

    metadata = {"Pixels": {"SizeX": SizeX, "SizeY": SizeY, "Type": Type}}

    axes = "YX"
    if SizeZ is not None and SizeZ > 1:
        axes = f"Z{axes}"
        metadata["Pixels"]["SizeZ"] = SizeZ

    if SizeT is not None and SizeT > 1:
        axes = f"T{axes}"
        metadata["Pixels"]["SizeT"] = SizeT

    metadata["axes"] = axes

    if PhysicalSizeX is not None:
        metadata["Pixels"]["PhysicalSizeX"] = PhysicalSizeX

    if PhysicalSizeY is not None:
        metadata["Pixels"]["PhysicalSizeY"] = PhysicalSizeY

    if PhysicalSizeZ is not None:
        metadata["Pixels"]["PhysicalSizeZ"] = PhysicalSizeZ

    if TimeIncrement is not None:
        metadata["Pixels"]["TimeIncrement"] = TimeIncrement

    return metadata


def to_tiff(
    new_path,
    data,
    SizeT=None,
    SizeZ=None,
    PhysicalSizeZ=None,
    PhysicalSizeX=None,
    PhysicalSizeY=None,
    TimeIncrement=None,
):
    valid_dtypes = (np.uint8, np.uint16, np.float32)
    is_valid_dtype = False
    for valid_dtype in valid_dtypes:
        if np.issubdtype(data.dtype, valid_dtype):
            is_valid_dtype = True
            break

    if not is_valid_dtype:
        data = data.astype(np.float32)

    metadata = get_tiff_metadata(
        data,
        SizeT=SizeT,
        SizeZ=SizeZ,
        PhysicalSizeZ=PhysicalSizeZ,
        PhysicalSizeX=PhysicalSizeX,
        PhysicalSizeY=PhysicalSizeY,
        TimeIncrement=TimeIncrement,
    )

    # # Potential alternative
    # hyperstack = tifffile.memmap(
    #     new_path,
    #     shape=img.shape,
    #     dtype=img.dtype,
    #     imagej=True,
    #     metadata={'axes': 'TZYX'},
    # )
    # hyperstack[:] = img
    # hyperstack.flush()

    try:
        tifffile.imwrite(new_path, data, metadata=metadata, imagej=True)
    except Exception as err:
        tifffile.imwrite(new_path, data)


def from_lab_to_obj_coords(lab):
    rp = skimage.measure.regionprops(lab)
    dfs = []
    keys = []
    for obj in rp:
        keys.append(obj.label)
        obj_coords = obj.coords
        ndim = obj_coords.shape[1]
        if ndim == 3:
            columns = ["z", "y", "x"]
        else:
            columns = ["y", "x"]
        df_obj = pd.DataFrame(data=obj_coords, columns=columns)
        dfs.append(df_obj)
    df = pd.concat(dfs, keys=keys, names=["Cell_ID", "idx"]).droplevel("idx")
    return df


def lab2d_to_rois(ImagejRoi, lab2D, ndigits, t=None, z=None):
    rp = skimage.measure.regionprops(lab2D)
    rois = []
    for obj in rp:
        cont = core.get_obj_contours(obj)
        yc, xc = obj.centroid
        x_str = str((int(xc))).zfill(ndigits)
        y_str = str((int(yc))).zfill(ndigits)
        name = f"{x_str}-{y_str}"
        if z is not None:
            z_str = str(z).zfill(ndigits)
            name = f"{z_str}-{name}"

        if t is not None:
            t_str = str(t).zfill(ndigits)
            name = f"{t_str}-{name}"

        name = f"id={obj.label}-{name}"

        roi = ImagejRoi.frompoints(cont, name=name, t=t, z=z, index=obj.label)
        rois.append(roi)
    return rois


def from_lab_to_imagej_rois(lab, ImagejRoi, t=0, SizeT=1, max_ID=None):
    if max_ID is None:
        max_ID = lab.max()

    if SizeT == 1:
        t = None

    SizeY, SizeX = lab.shape[-2:]
    ndigitsT = len(str(SizeT))
    ndigitsY = len(str(SizeY))
    ndigitsX = len(str(SizeX))

    if lab.ndim == 3:
        rois = []
        SizeZ = len(lab)
        ndigitsZ = len(str(SizeZ))
        ndigits = max(ndigitsT, ndigitsZ, ndigitsY, ndigitsX)
        for z, lab2D in enumerate(lab):
            z_rois = lab2d_to_rois(ImagejRoi, lab2D, ndigits, t=t, z=z)
        rois.extend(z_rois)
    else:
        ndigits = max(ndigitsT, ndigitsY, ndigitsX)
        rois = lab2d_to_rois(ImagejRoi, lab, ndigits, t=t)
    return rois


def from_imagej_rois_to_segm_data(
    TZYX_shape, ID_to_roi_mapper, rescale_rois_sizes, repeat_2d_rois_zslices_range
):
    SizeT, SizeZ, SizeY, SizeX = TZYX_shape
    segm_data = np.zeros(TZYX_shape, dtype=np.uint32)
    for ID, roi in ID_to_roi_mapper.items():
        name = roi.name
        name_parts = name.split("-")
        zz = [0]
        if len(name_parts) == 2 and SizeZ > 1:
            # 2D roi in 3D segm data --> place 2D roi on each z-slice
            zz = range(*repeat_2d_rois_zslices_range)

        elif len(name_parts) > 2 and SizeZ > 1:
            # 2D roi from a 3D roi --> place at requested z-slice
            zz = [int(name_parts[-3])]

        tt = [0] * len(zz)
        if SizeT > 1:
            tt = [roi.t_position] * len(zz)

        y0, x0 = roi.top, roi.left
        contours = roi.integer_coordinates + (x0, y0)
        xx = contours[:, 0]
        yy = contours[:, 1]
        if rescale_rois_sizes is not None:
            rescale_z = rescale_rois_sizes["Z"]
            rescale_y = rescale_rois_sizes["Y"]
            rescale_x = rescale_rois_sizes["X"]

            factor_z = rescale_z[1] / rescale_z[0]
            factor_y = rescale_y[1] / rescale_y[0]
            factor_x = rescale_x[1] / rescale_x[0]

            xx = np.clip(np.round(xx * factor_x).astype(int), 0, SizeX - 1)
            yy = np.clip(np.round(yy * factor_y).astype(int), 0, SizeY - 1)

        for t, z in zip(tt, zz):
            if rescale_rois_sizes is not None:
                z = round(z * factor_z)
                z = z if z < SizeZ else SizeZ
                z = z if z >= 0 else 0

            rr, cc = skimage.draw.polygon(yy, xx)
            segm_data[t, z, rr, cc] = ID

    return np.squeeze(segm_data)


def seconds_to_ETA(seconds):
    seconds = round(seconds)
    ETA = datetime.timedelta(seconds=seconds)
    ETA_split = str(ETA).split(":")
    if seconds < 0:
        ETA = "00h:00m:00s"
    elif seconds >= 86400:
        days, hhmmss = str(ETA).split(",")
        h, m, s = hhmmss.split(":")
        ETA = f"{days}, {int(h):02}h:{int(m):02}m:{int(s):02}s"
    else:
        h, m, s = str(ETA).split(":")
        ETA = f"{int(h):02}h:{int(m):02}m:{int(s):02}s"
    return ETA


def to_uint8(img):
    if img.dtype == np.uint8:
        return img
    img = np.round(img_to_float(img) * 255).astype(np.uint8)
    return img


def to_uint16(img):
    if img.dtype == np.uint16:
        return img
    img = np.round(img_to_float(img) * 65535).astype(np.uint16)
    return img


def img_to_float(img, force_dtype=None, force_missing_dtype=None, warn=True):
    input_img_dtype = img.dtype
    value = img[(0,) * img.ndim]
    img_max = np.max(img)
    # Check if float outside of -1, 1
    if img_max <= 1.0 and isinstance(value, (np.floating, float)):
        return img

    uint8_max = np.iinfo(np.uint8).max
    uint16_max = np.iinfo(np.uint16).max
    uint32_max = np.iinfo(np.uint32).max

    img = img.astype(float)
    if force_dtype is not None:
        dtype_max = np.iinfo(force_dtype).max
        img = img / dtype_max
    elif input_img_dtype == np.uint8:
        # Input image is 8-bit
        img = img / uint8_max
    elif input_img_dtype == np.uint16:
        # Input image is 16-bit
        img = img / uint16_max
    elif input_img_dtype == np.uint32:
        # Input image is 32-bit
        img = img / uint32_max
    elif force_missing_dtype is not None:
        img = img.astype(force_dtype)
    elif img_max <= uint8_max:
        # Input image is probably 8-bit
        if warn:
            _warnings.warn_image_overflow_dtype(input_img_dtype, img_max, "8-bit")
        img = img / uint8_max
    elif img_max <= uint16_max:
        # Input image is probably 16-bit
        if warn:
            _warnings.warn_image_overflow_dtype(input_img_dtype, img_max, "16-bit")
        img = img / uint16_max
    elif img_max <= uint32_max:
        # Input image is probably 32-bit
        if warn:
            _warnings.warn_image_overflow_dtype(input_img_dtype, img_max, "32-bit")
        img = img / uint32_max
    else:
        # Input image is a non-supported data type
        raise TypeError(
            f"The maximum value in the image is {img_max} which is greater than the "
            f"maximum value supported of {uint32_max} (32-bit). "
            "Please consider converting your images to 32-bit or 16-bit first."
        )
    return img


def float_img_to_dtype(img, dtype):
    if img.dtype == dtype:
        return img

    img_max = img.max()
    if img_max > 1.0:
        raise TypeError(
            "Images of float data type with values greater than 1.0 cannot "
            f"be safely casted to {dtype}. "
            f"The max value of the input image is {img_max:.3f}"
        )

    img_min = img.min()
    if img_min < -1.0:
        raise TypeError(
            "Images of float data type with values smaller than -1.0 cannot "
            f"be safely casted to {dtype}."
            f"The minumum value of the input image is {img_min:.3f}"
        )

    if dtype == np.uint8:
        return skimage.img_as_ubyte(img)

    if dtype == np.uint16:
        return skimage.img_as_uint(img)

    if dtype == np.float32:
        return img.astype(np.float32)

    if dtype == np.float64:
        return img.astype(np.float64)

    raise TypeError(
        f"Invalid output data type `{dtype}`. "
        "Valid output data types are `np.uint8` and `np.uint16`"
    )


def convert_to_dtype(data: np.ndarray, dtype):
    if data.dtype == dtype:
        return data
    val = data[tuple([0] * data.ndim)]
    if isinstance(val, (np.floating, float)):
        data = float_img_to_dtype(data, dtype)
    elif dtype == np.uint8:
        data = np.round(img_to_float(data) * 255).astype(np.uint8)
    elif dtype == np.uint16:
        data = np.round(img_to_float(data) * 65535).astype(np.uint16)
    else:
        raise TypeError(
            f"Invalid output data type `{dtype}`. "
            "Valid data types are floating-point format, `np.uint8` "
            "and `np.uint16`"
        )
    return data


def _apt_update_command():
    return "sudo apt-get update"


def _apt_gcc_command():
    return "sudo apt install python-dev gcc"


def jdk_windows_url():
    return "https://hmgubox2.helmholtz-muenchen.de/index.php/s/R62Ktcda6jWea2s"


def cpp_windows_url():
    return "https://visualstudio.microsoft.com/visual-cpp-build-tools/"


def check_napari_plugin(plugin_name, module_name, parent=None):
    try:
        import_module(module_name)
    except ModuleNotFoundError as e:
        url = "https://napari.org/stable/plugins/find_and_install_plugin.html#find-and-install-plugins"
        href = html_utils.href_tag("this guide", url)
        txt = html_utils.paragraph(f"""
            To correctly use this napari utility you need to <b>install the 
            plugin</b> called <code>{plugin_name}</code>.<br><br>
            Please, read {href} on how to install plugins in napari.<br><br>
            You will need to <b>restart</b> both napari and Cell-ACDC after installing 
            the plugin.<br><br>
            NOTE: in the text box in napari you will need to write the full name 
            <code>{plugin_name}</code> becasue it is NOT A SEARCH BOX.
        """)
        msg = widgets.myMessageBox()
        msg.critical(parent, f"Napari plugin required", txt)
        raise e


def purge_module(module_name):
    to_delete = [
        mod
        for mod in sys.modules
        if mod == module_name or mod.startswith(module_name + ".")
    ]
    for mod in to_delete:
        del sys.modules[mod]

    importlib.invalidate_caches()
    importlib.import_module(module_name)
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    else:
        raise ModuleNotFoundError(f"Module '{module_name}' not found in sys.modules.")


def is_gui_running():
    if not GUI_INSTALLED:
        return False

    return QCoreApplication.instance() is not None


def _subprocess_run_command(command, shell=True, callback="check_call"):
    func = getattr(subprocess, callback)
    try:
        out = func(command, shell=shell)
    except Exception as err:
        print(
            f"[WARNING]: Command `{command}` failed. Trying with `{command.split()}`..."
        )
        out = func(command.split(), shell=shell)

    return out


def _run_command(command: str | list[str], shell=False):
    if not isinstance(command, (str, list)):
        raise TypeError(
            f"Command must be a string or a list of strings, not {type(command)}"
        )

    command_str = None
    if isinstance(command, str):
        args_list = [command]
        command_str = command
    else:
        args_list = command
        if len(command) == 1:
            command_str = command[0]

    try:
        subprocess.check_call(args_list, shell=shell)
        return
    except Exception as err:
        pass

    if command_str is None:
        return

    try:
        subprocess.check_call(command_str, shell=shell)
        return
    except Exception as err:
        pass

    try:
        from . import acdc_regex

        args = acdc_regex.RE_SPLIT_SPACES_IGNORE_QUOTES.split(command_str)[1::2]
        subprocess.check_call(args, shell=shell)
        return
    except Exception as err:
        pass


def get_chained_attr(_object, _name):
    for attr in _name.split("."):
        _object = getattr(_object, attr)
    return _object


def get_fiji_base_command():
    command = None
    if is_mac:
        command = get_fiji_exec_folderpath()

    return command


def _init_fiji_cli():
    if is_win:
        return True

    fiji_app_folderpath = get_fiji_exec_folderpath()
    args_add_to_path = [f"chmod 755 {fiji_app_folderpath}"]
    try:
        subprocess.check_call(args_add_to_path, shell=True)
        return True
    except Exception as e:
        printl(f"Error occurred while setting permissions: {e}")
        return False


def test_fiji_base_command(logger_func=print):
    base_command = get_fiji_base_command()

    if base_command is None:
        logger_func("[WARNING]: Fiji is not present.")
        return False

    command = f"{base_command} --headless"
    return run_fiji_command(command=command, logger_func=logger_func)


def run_fiji_command(command=None, logger_func=print):
    if command is None:
        command = f"{get_fiji_base_command()} --headless"

    init_success = _init_fiji_cli()
    if not init_success:
        return False

    separator = "-" * 100
    commands = (command, command.split())
    for args in commands:
        logger_func(f'{separator}\nTrying Fiji command: "{args}"...\n{separator}\n')
        try:
            subprocess.check_call(args, shell=True)
            return True
        except Exception as err:
            continue
    return False


def import_segment_module(model_name):
    try:
        acdcSegment = import_module(f"cellacdc.segmenters.{model_name}.acdcSegment")
    except ModuleNotFoundError as e:
        # Check if custom model
        cp = config.ConfigParser()
        cp.read(models_list_file_path)
        model_path = cp[model_name]["path"]
        spec = importlib.util.spec_from_file_location("acdcSegment", model_path)
        acdcSegment = importlib.util.module_from_spec(spec)
        sys.modules["acdcSegment"] = acdcSegment
        spec.loader.exec_module(acdcSegment)
    return acdcSegment


def _available_frameworks(model_name):
    frameworks = {
        "cuda": (
            model_name.lower().find("cellpose") != -1
            or model_name.lower().find("omnipose") != -1
            or model_name.lower().find("deepsea") != -1
            or model_name.lower().find("segment_anything") != -1
            or model_name.lower().find("sam2") != -1
            or model_name.lower().find("yeaz") != -1
            or model_name.lower().find("yeaz_v2") != -1
        ),
        "directML": (
            model_name.lower().find("cellpose_v4") != -1
            or model_name.lower().find("cellpose_v3") != -1  # has its own way to check
        ),
    }
    return frameworks


def find_missing_integers(lst, max_range=None):
    if max_range is not None:
        max_range = lst[-1] + 1
    return [x for x in range(lst[0], max_range) if x not in lst]


def synthetic_image_geneator(size=(512, 512), f_x=1, f_y=1):
    Y, X = size
    x = np.linspace(0, 10, Y)
    y = np.linspace(0, 10, X)
    xx, yy = np.meshgrid(x, y)
    img = np.sin(f_x * xx) * np.cos(f_y * yy)
    return img


def get_slices_local_into_global_arr(bbox_coords, global_shape):
    slice_global_to_local = []
    slice_crop_local = []
    for (_min, _max), _D in zip(bbox_coords, global_shape):
        _min_crop, _max_crop = None, None
        if _min < 0:
            _min_crop = abs(_min)
            _min = 0
        if _max > _D:
            _max_crop = _D - _max
            _max = _D

        slice_global_to_local.append(slice(_min, _max))
        slice_crop_local.append(slice(_min_crop, _max_crop))

    return tuple(slice_global_to_local), tuple(slice_crop_local)


def format_cca_manual_changes(changes: dict):
    txt = ""
    for ID, changes_ID in changes.items():
        txt = f"{txt}* ID {ID}:\n"
        for col, (old_val, new_val) in changes_ID.items():
            txt = f"{txt}    - {col}: {old_val} --> {new_val}\n"
        txt = f"{txt}--------------------------------\n\n"
    return txt


def _parse_bool_str(value):
    if isinstance(value, bool):
        return value

    if value == "True":
        return True
    elif value == "False":
        return False


def init_input_points_df(posData, input_points_filepath):
    input_points_df = None
    if os.path.exists(input_points_filepath):
        input_points_df = pd.read_csv(input_points_filepath)
    else:
        # input_points_filepath is actually and endname
        for file in listdir(posData.images_path):
            if file.endswith(input_points_filepath):
                filepath = os.path.join(posData.images_path, file)
                input_points_df = pd.read_csv(filepath)
                break

    if input_points_df is None:
        raise FileNotFoundError(
            f'Could not find input points table from file "input_points_filepath" '
            "Perhaps, you forgot to save the table?"
        )

    for col in ("x", "y", "id"):
        if col not in input_points_df.columns:
            raise KeyError(
                f"Input points table is missing colum {col}. It must have "
                "the colums (x, y, id)"
            )

    return input_points_df


def pairwise(iterable):
    # pairwise('ABCDEFG') → AB BC CD DE EF FG
    iterator = iter(iterable)
    a = next(iterator, None)
    for b in iterator:
        yield a, b
        a = b


def _relabel_cca_dfs_and_segm_data(
    cca_dfs,
    IDs_mapper,
    asymm_tracked_segm,
    progressbar=True,
):
    # Rename Cell_ID index according to asymmetric cell div convention
    if progressbar:
        pbar = tqdm(
            desc="Applying asymmetric division", total=len(IDs_mapper), ncols=100
        )
    for key, (root_ID, parent_ID) in IDs_mapper.items():
        div_frame_i, daughter_ID = key
        for frame_i in range(div_frame_i, len(asymm_tracked_segm)):
            lab = asymm_tracked_segm[frame_i]
            rp = skimage.measure.regionprops(lab)
            rp_mapper = {obj.label: obj for obj in rp}
            obj_daught = rp_mapper.get(daughter_ID)
            mother_ID = root_ID if rp_mapper.get(root_ID) is None else parent_ID

            cca_dfs[frame_i].rename(index={daughter_ID: mother_ID}, inplace=True)

            if obj_daught is None:
                continue

            lab[obj_daught.slice][obj_daught.image] = mother_ID

        if progressbar:
            pbar.update()

    if progressbar:
        pbar.close()


def get_empty_stored_data_dict():
    return {
        "regionprops": None,
        "labels": None,
        "acdc_df": None,
        "delROIs_info": {"rois": [], "delMasks": [], "delIDsROI": [], "state": []},
        "IDs": [],
        "manually_edited_lab": {"lab": {}, "zoom_slice": None},
    }


def iterate_along_axes(arr, axes, arr_ndim=None):
    if arr_ndim is None:
        arr_ndim = arr.ndim
    axes = list(axes)
    front_axes = axes + [i for i in range(arr_ndim) if i not in axes]
    arr_moved = np.moveaxis(arr, front_axes, range(arr_ndim))
    iter_shape = arr_moved.shape[: len(axes)]
    for idx in np.ndindex(iter_shape):
        # Build the index for the original array
        full_idx = [slice(None)] * arr_ndim
        for axis, i in zip(axes, idx):
            full_idx[axis] = i
        yield tuple(full_idx)


def get_input_output_mapper(
    input_shape: Tuple[int],
    iterate_axes: Tuple[int],
    output_shape: Tuple[int],
    output_axes: Tuple[int],
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Creates list of tuples with the input and output indices

    Parameters
    ----------
    input_shape : Tuple[int]
        Shape of the input array
    iterate_axes : Tuple[int]
        Axes to iterate over
    output_shape : Tuple[int]
        Shape of the output array
    output_axes : Tuple[int]
        Axes of the output array
    """
    assert len(iterate_axes) == len(output_axes)

    iterate_shape = tuple(input_shape[axis] for axis in iterate_axes)
    mapper = []

    for idx_vals in itertools.product(*[range(s) for s in iterate_shape]):
        # Build full input index
        input_index = [slice(None)] * len(input_shape)
        for axis in iterate_axes:
            i = iterate_axes.index(axis)
            input_index[axis] = idx_vals[i]

        # Build full output index
        output_index = [slice(None)] * len(output_shape)
        for axis in output_axes:
            i = output_axes.index(axis)
            output_index[axis] = idx_vals[i]

        input_index = tuple(input_index)
        output_index = tuple(output_index)

        mapper.append((input_index, output_index))

    return mapper


def translateStrNone(*args):
    args = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, str):
            if arg.lower() == "none":
                args[i] = None
            elif arg.lower() == "true":
                args[i] = True
            elif arg.lower() == "false":
                args[i] = False

    return args


def try_kwargs(func, *args, **kwargs):
    """
    Attempt to call a function with the provided arguments and keyword arguments.

    If the function raises a TypeError due to unexpected keyword arguments,
    those arguments are dynamically removed, and the function is retried.
    This process continues until the function succeeds or no keyword arguments
    remain, in which case the exception is re-raised.

    Args:
        func (Callable): The function to call.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        Tuple[Any, List[str]]: A tuple containing:
            - The result of the function call (or None if it fails).
            - A list of keyword arguments that were removed.

    Raises:
        ValueError: If a keyword argument mentioned in the error message
            is not found in the provided kwargs.
        TypeError: If the function fails with a TypeError after all keyword
            arguments have been removed.
    """

    kwargs = kwargs.copy()  # Create a copy to avoid modifying the original
    removed_kwargs = []
    pattern = r"unexpected keyword argument ['\"](\w+)['\"]"
    while True:
        try:
            return func(*args, **kwargs), removed_kwargs
        except TypeError as e:
            match = re.search(pattern, str(e))
            if match:
                kwarg_name = match.group(1)
                if kwarg_name in kwargs:
                    del kwargs[kwarg_name]
                    removed_kwargs.append(kwarg_name)
                else:
                    raise ValueError(
                        f"Keyword argument '{kwarg_name}' not found in kwargs."
                    )
            else:
                raise e

            if len(kwargs) == 0:
                print(f"Function {func.__name__} failed with TypeError: {e}")
                raise e


def get_obj_by_label(rp, target_label):
    """
    Returns the object with the specified label from the given list of objects.

    Parameters
    ----------
    rp : list
        The list of objects to search through.
    target_label : str
        The label of the object to find.

    Returns
    -------
    object
        The object with the specified label, or None if not found.
    """
    for obj in rp:
        if obj.label == target_label:
            return obj
    return None


def find_distances_ID(rps, point=None, ID=None):
    """
    Calculate the distances between a given point and the centroids of a list of regionprops.

    Parameters
    ----------
    rps : list
        List of regionprops objects.
    point : tuple, optional
        The coordinates of the point. Defaults to None.
    ID : int, optional
        The label ID of the regionprops object. Defaults to None.

    Returns
    -------
    numpy.ndarray
        A matrix of distances between the point and the centroids.

    Raises
    ------
    ValueError
        If ID is not found in the list of regionprops (list of cells).
    ValueError
        If neither ID nor point is provided.
    ValueError
        If both ID and point are provided.
    """

    if ID is not None and point is None:
        try:
            point = [rp.centroid for rp in rps if rp.label == ID][0]
        except IndexError:
            raise ValueError(f"ID {ID} not found in regionprops (list of cells).")

    elif ID is None and point is None:
        raise ValueError("Either ID or point must be provided.")

    elif ID is not None and point is not None:
        raise ValueError("Only one of ID or point must be provided.")

    point = point[
        ::-1
    ]  # rp are in (y, x) format (or (z, y, x) for 3D data) so I need to reverse order
    point = np.array([point])
    centroids = np.array([rp.centroid for rp in rps])
    diff = point[:, np.newaxis] - centroids
    dist_matrix = np.linalg.norm(diff, axis=2)
    return dist_matrix


def sort_IDs_dist(rps, point=None, ID=None):
    """Sorts the IDs of regionprops based on their distances to a given point.

    Parameters
    ----------
    rps : list
        A list of regionprops objects representing cells.
    point : tuple, optional
        The coordinates of the point to calculate distances from.
        If not provided, it will be calculated based on the given ID.
    ID : int, optional
        The ID of the regionprops object to calculate distances from.
        If this and point are both provided, or neither, an error will be
        raised.

    Returns
    -------
    list
        A sorted list of IDs based on their distances to the given point.

    Raises
    ------
    ValueError
        If ID is not found in the list of regionprops objects.
    ValueError
        If neither ID nor point is provided.
    ValueError
        If both ID and point are provided.

    """
    if ID is not None and point is None:
        try:
            point = [rp.centroid for rp in rps if rp.label == ID][0]
        except IndexError:
            raise ValueError(f"ID {ID} not found in regionprops (list of cells).")

    elif ID is None and point is None:
        raise ValueError("Either ID or point must be provided.")

    elif ID is not None and point is not None:
        raise ValueError("Only one of ID or point must be provided.")

    IDs = [rp.label for rp in rps]
    if len(IDs) == 0:
        return []
    elif len(IDs) == 1:
        return IDs
    dist_matrix = find_distances_ID(rps, point=point)
    dist_matrix = np.squeeze(dist_matrix)

    sorted_ids = sorted(zip(dist_matrix, IDs))
    sorted_ids = [ID for _, ID in sorted_ids]
    return sorted_ids


def safe_get_or_call(obj, path: str):
    """Safely get nested attributes or call methods with literal args from a string path."""
    expr = ast.parse(path, mode="eval").body

    def _eval(node, current_obj):
        if isinstance(node, ast.Attribute):
            return getattr(_eval(node.value, current_obj), node.attr)
        elif isinstance(node, ast.Call):
            func = _eval(node.func, current_obj)
            args = [ast.literal_eval(arg) for arg in node.args]
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
            return func(*args, **kwargs)
        elif isinstance(node, ast.Name):
            # First name in chain is assumed to be from `obj`
            return getattr(current_obj, node.id)
        else:
            raise ValueError(f"Unsupported syntax: {ast.dump(node)}")

    return _eval(expr, obj)


def format_commit_date_utc(utc_str):
    # Parse the UTC date string (ISO 8601 format)
    dt = datetime.datetime.fromisoformat(utc_str.replace("Z", "+00:00"))

    # Convert to your local time zone (optional)
    local_dt = dt.astimezone()  # removes UTC offset if local

    # Format nicely
    return local_dt.strftime(r"%A %d %B %Y at %H:%M")


def get_linux_distribution_name():
    import csv

    RELEASE_DATA = {}
    with open("/etc/os-release") as f:
        reader = csv.reader(f, delimiter="=")
        for row in reader:
            if row:
                RELEASE_DATA[row[0]] = row[1]
    if RELEASE_DATA["ID"] in ["debian", "raspbian"]:
        with open("/etc/debian_version") as f:
            DEBIAN_VERSION = f.readline().strip()
        major_version = DEBIAN_VERSION.split(".")[0]
        version_split = RELEASE_DATA["VERSION"].split(" ", maxsplit=1)
        if version_split[0] == major_version:
            # Just major version shown, replace it with the full version
            RELEASE_DATA["VERSION"] = " ".join([DEBIAN_VERSION] + version_split[1:])

    name_version = f"{RELEASE_DATA['NAME']} {RELEASE_DATA['VERSION']}"

    return name_version


def reset_settings():
    question = (
        'Do you want to reset Cell-ACDC settings- type "h" for help - (y/[n]/h)? '
    )
    info_txt = (
        "If you reset Cell-ACDC settings, the folder below will be deleted.\n\n"
        "This means deeleting things like custom shortcuts, recent paths, last "
        "selections, and GUI preferences.\n\n"
        f'Settings folder path: "{settings_folderpath}"'
    )
    answer = "y"
    while True:
        try:
            answer = input(f"\n{question}")
        except Exception as err:
            break

        if answer == "n":
            print("*" * 100)
            return "Resetting Cell-ACDC settings cancelled."

        if answer == "y":
            break

        if answer == "h":
            print("-" * 100)
            print(f"\n{info_txt}")
            print("=" * 100)

        print(
            f'"{answer}" is not a valid answer. '
            'Type "y" for "yes", "n" for "no", or "h" for help.'
        )

    try:
        os.remove(settings_folderpath)
        print("*" * 100)
        out_txt = (
            "Cell-ACDC settings have been reset.\n\n"
            "The following folder was deleted:\n\n"
            f"{settings_folderpath}"
        )
    except Exception as err:
        traceback.print_exc()
        print("*" * 100)
        out_txt = (
            "**ERROR** occured when trying to remove the settings folder.\n\n"
            "To reset Cell-ACDC settings, please remove this folder:\n\n"
            f"{settings_folderpath}\n"
        )
        return out_txt


def separate_fluo_segment_channels(channels):
    segms_to_load = []
    channels_to_load = []
    current_segm = False
    for ch in channels:
        if ch == "current segm.":
            current_segm = True
        elif "segm" in ch:
            segms_to_load.append(ch)
        else:
            channels_to_load.append(ch)
    return segms_to_load, channels_to_load, current_segm

# Sibling imports (deferred to avoid import cycles)
from .logging import (
    Logger,
)
from .models import (
    download_manual,
    params_to_ArgSpec,
)
from .paths import (
    get_fiji_exec_folderpath,
    listdir,
)

