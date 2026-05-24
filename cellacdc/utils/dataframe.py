"""Cell-ACDC utility helpers: dataframe."""

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

def checked_reset_index(df):
    if df.index.names is None or df.index.names == [None]:
        return df.reset_index(drop=True)
    else:
        return df.reset_index()


def checked_reset_index_Cell_ID(df):
    if df.index.names == ["Cell_ID"]:
        return df
    df = checked_reset_index(df)
    return df.set_index("Cell_ID")


def get_cca_colname_desc():
    desc = {
        "Cell ID": (
            "ID of the segmented cell. All of the other columns "
            "are properties of this ID."
        ),
        "Cell cycle stage": ("G1 if the cell does NOT have a bud. S/G2/M if it does."),
        "Relative ID": (
            "ID of the bud related to the Cell ID (row). For cells in G1 write the "
            "bud ID it had in the previous cycle."
        ),
        "Generation number": (
            "Number of times the cell divided from a bud. For cells in the first "
            "frame write any number greater than 1."
        ),
        "Relationship": (
            "Relationship of the current Cell ID (row). "
            "Either <b>mother</b> or <b>bud</b>. An object is a bud if "
            "it didn't divide from the mother yet. All other instances "
            "(e.g., cell in G1) are still labelled as mother."
        ),
        "Emerging frame num.": (
            "Frame number at which the object emerged/appeared in the scene."
        ),
        "Division frame num.": (
            "Frame number at which the bud separated from the mother."
        ),
        "Is history known?": (
            "Cells that are already present in the first frame or appears "
            "from outside of the field of view, have some information missing. "
            "For example, for cells in the first frame we do not know how many "
            "times it budded and divided in the past. "
            "In these cases Is history known? is True."
        ),
    }
    return desc


def are_acdc_dfs_equal(df_left, df_right):
    if df_left.shape != df_right.shape:
        return False

    try:
        for col in df_left.columns:
            if col not in df_right.columns:
                return False

            try:
                eq_mask = np.isclose(df_left[col], df_right[col], equal_nan=True)
            except Exception as err:
                # Data type is string
                eq_mask = df_left[col] == df_right[col]

            nan_mask = (df_left[col].isna()) & (df_right[col].isna())
            equality_mask = (eq_mask) | (nan_mask)
            if not equality_mask.all():
                return False
    except Exception as err:
        return False

    return True


def fix_acdc_df_dtypes(acdc_df):
    acdc_df["is_cell_excluded"] = acdc_df["is_cell_excluded"].astype(bool)
    return acdc_df


def df_ctc_to_acdc_df(
    df_ctc,
    tracked_segm,
    cell_division_mode="Normal",
    return_list=False,
    progressbar=True,
):
    """Convert Cell Tracking Challenge DataFrame with annotated division to
    Cell-ACDC cell cycle annotations DataFrame.

    Parameters
    ----------
    df_ctc : pd.DataFrame
        DataFrame with {'label', 't1', 't2', 'parent'} columns where
        't1' is the frame index of cell division.
    tracked_segm : (T, Y, X) array of ints
        Array of tracked segmentation labels.
    cell_division_mode : {'Normal', 'Asymmetric'}, optional
        Type of cell division. `Normal` is the standard cell division,
        where the mother cell divides into two daughter cells. For the
        tracking, that means the two daughter cells get a new, unique ID
        each.

        `Asymmetric` means that the mother cell grows one daughter
        cell that eventually divides from the mother (e.g., budding yeast).
        For the tracking, this means that the mother cell ID keeps
        existing after division and the daughter cell gets a new, unique ID.

        If `Asymmetric`, the third returned element is the segmentation data
        with the asymmetric Cell IDs.
    return_list : bool, optional
        If `True`, the second returned element is the list of created dataframes,
        one per frame. Default is False
    progressbar : bool, optional
        If `True`, displays a tqdm progressbar. Default is True
    """
    cca_dfs = []
    keys = []
    df_ctc = df_ctc.set_index(["t1", "parent"])

    if cell_division_mode == "Asymmetric":
        asymm_tracked_segm = tracked_segm.copy()

    asymmetric_IDs_rename_mapper = {}
    if progressbar:
        pbar = tqdm(
            desc="Converting to Cell-ACDC format", total=len(tracked_segm), ncols=100
        )
    for frame_i, lab in enumerate(tracked_segm):
        rp = skimage.measure.regionprops(lab)
        IDs = [obj.label for obj in rp]
        cca_df = core.getBaseCca_df(IDs, with_tree_cols=True)
        keys.append(frame_i)
        if frame_i == 0:
            cca_dfs.append(cca_df)
            if progressbar:
                pbar.update()
            continue

        # Copy annotations from previous frames
        prev_cca_df = cca_dfs[frame_i - 1]
        old_IDs = cca_df.index.intersection(prev_cca_df.index)
        cca_df.loc[old_IDs] = prev_cca_df.loc[old_IDs]

        try:
            df_ctc_i = df_ctc.loc[frame_i]
        except KeyError as err:
            # No division detected --> nothing to annotate
            cca_dfs.append(cca_df)
            if progressbar:
                pbar.update()
            continue

        for parent_ID, df_ctc_i_pID in df_ctc_i.groupby(level=0):
            daughter_IDs = df_ctc_i_pID["label"].to_list()

            if parent_ID == 0:
                continue

            cca_df.loc[daughter_IDs, "parent_ID_tree"] = parent_ID
            cca_df.loc[daughter_IDs, "emerg_frame_i"] = frame_i
            cca_df.loc[daughter_IDs, "division_frame_i"] = frame_i

            root_ID = prev_cca_df.at[parent_ID, "root_ID_tree"]
            if root_ID == -1:
                root_ID = parent_ID
            cca_df.loc[daughter_IDs, "root_ID_tree"] = root_ID

            cca_df.loc[daughter_IDs[0], "sister_ID_tree"] = daughter_IDs[1]
            cca_df.loc[daughter_IDs[1], "sister_ID_tree"] = daughter_IDs[0]

            prev_gen_num = prev_cca_df.loc[parent_ID, "generation_num_tree"]
            cca_df.loc[daughter_IDs, "generation_num_tree"] = prev_gen_num + 1

            # Annotate division from df_ctc_i into
            if cell_division_mode == "Asymmetric":
                # Recycle the root_ID and assign it to one of the daughters
                replaced_daught_ID = daughter_IDs[1]
                key = (frame_i, replaced_daught_ID)
                asymmetric_IDs_rename_mapper[key] = (root_ID, parent_ID)

        cca_dfs.append(cca_df)

        if progressbar:
            pbar.update()

    if progressbar:
        pbar.close()

    if asymmetric_IDs_rename_mapper:
        _relabel_cca_dfs_and_segm_data(
            cca_dfs,
            asymmetric_IDs_rename_mapper,
            asymm_tracked_segm,
            progressbar=True,
        )

    cca_df = pd.concat(cca_dfs, keys=keys, names=["frame_i"])

    out = [cca_df, None, None]

    if return_list:
        out[1] = cca_dfs

    if cell_division_mode == "Asymmetric":
        out[2] = asymm_tracked_segm

    return out


def format_IDs(IDs):
    if isinstance(IDs, str):
        raise ValueError("IDs must not be a string")

    IDsRange = []
    text = ""
    sorted_vals = sorted(IDs)
    for i, e in enumerate(sorted_vals):
        e = int(e)
        # Get previous and next value (if possible)
        if i > 0:
            prevVal = sorted_vals[i - 1]
        else:
            prevVal = -1
        if i < len(sorted_vals) - 1:
            nextVal = sorted_vals[i + 1]
        else:
            nextVal = -1

        if e - prevVal == 1 or nextVal - e == 1:
            if not IDsRange:
                if nextVal - e == 1 and e - prevVal != 1:
                    # Current value is the first value of a new range
                    IDsRange = [e]
                else:
                    # Current value is the second element of a new range
                    IDsRange = [prevVal, e]
            else:
                if e - prevVal == 1:
                    # Current value is part of an ongoing range
                    IDsRange.append(e)
                else:
                    # Current value is the first element of a new range
                    # --> create range text and this element will
                    # be added to the new range at the next iter
                    start, stop = IDsRange[0], IDsRange[-1]
                    if stop - start > 1:
                        sep = "-"
                    else:
                        sep = ","
                    text = f"{text},{start}{sep}{stop}"
                    IDsRange = []
        else:
            # Current value doesn't belong to a range
            if IDsRange:
                # There was a range not added to text --> add it now
                start, stop = IDsRange[0], IDsRange[-1]
                if stop - start > 1:
                    sep = "-"
                else:
                    sep = ","
                text = f"{text},{start}{sep}{stop}"

            text = f"{text},{e}"
            IDsRange = []

    if IDsRange:
        # Last range was not added  --> add it now
        start, stop = IDsRange[0], IDsRange[-1]
        text = f"{text},{start}-{stop}"

    text = text[1:]

    return text

# Sibling imports (deferred to avoid import cycles)
from .misc import (
    _relabel_cca_dfs_and_segm_data,
)

