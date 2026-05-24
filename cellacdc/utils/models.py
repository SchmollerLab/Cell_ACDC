"""Cell-ACDC utility helpers: models."""

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

def get_add_custom_prompt_model_instructions():
    init_sh = html_utils.init_sh
    segment_sh = html_utils.segment_sh
    add_prompt_sh = html_utils.add_prompt_sh
    href = f'<a href="{issues_url}">here</a>'
    text = html_utils.paragraph(f"""
    To use a custom prompt model, you need to create a Python file with the name 
    <code>acdcPromptModel.py</code>.<br>
    Note that the folder name where you place this file will be used as the 
    model name.<br><br>
    In this file, you will implement a class called <code>Model</code> with 
    at least the <code>{init_sh}</code> to initialise the model,<br>
    the <code>{add_prompt_sh}</code> method to add prompts (points, boxes, etc.) 
    to the model, and the <code>{segment_sh}</code> method to run the 
    segmentation.<br><br>
    Have a look at the existing models in the <code>promptable_models</code> 
    folder for examples.<br><br>
    If it doesn't work, please report the issue {href} with the
    code you wrote. Thanks!
    """)
    return text


def get_add_custom_model_instructions():
    user_manual_url = "https://github.com/SchmollerLab/Cell_ACDC/blob/main/UserManual/Cell-ACDC_User_Manual.pdf"
    href_user_manual = f'<a href="{user_manual_url}">user manual</a>'
    href = f'<a href="{issues_url}">here</a>'
    class_sh = html_utils.class_sh
    def_sh = html_utils.def_sh
    kwargs_sh = html_utils.kwargs_sh
    Model_sh = html_utils.Model_sh
    segment_sh = html_utils.segment_sh
    predict_sh = html_utils.predict_sh
    init_sh = html_utils.init_sh
    myModel_sh = html_utils.myModel_sh
    return_sh = html_utils.return_sh
    equal_sh = html_utils.equal_sh
    open_par_sh = html_utils.open_par_sh
    close_par_sh = html_utils.close_par_sh
    image_sh = html_utils.image_sh
    from_sh = html_utils.from_sh
    import_sh = html_utils.import_sh
    s = html_utils.paragraph(f"""
    To use a custom model first <b>create a folder</b> with the name of your model.<br><br>
    Inside this new folder create a file named <code>acdcSegment.py</code>.<br><br>
    In the <code>acdcSegment.py</code> file you will <b>implement the model class</b>.<br><br>
    Have a look at the other existing models, but essentially you have to create
    a class called <code>Model</code> with at least<br>
    the <code>{init_sh}</code> and the <code>{segment_sh}</code> method.<br><br>
    The <code>{segment_sh}</code> method takes the image (2D or 3D) as an input and return the segmentation mask.<br><br>
    You can find more details in the {href_user_manual} at the section
    called <code>Adding segmentation models to the pipeline</code>.<br><br>
    Pseudo-code for the <code>acdcSegment.py</code> file:
    <pre><code>
    {from_sh} myModel {import_sh} {myModel_sh}

    {class_sh} {Model_sh}:
        {def_sh} {init_sh}(self, {kwargs_sh}):
            self.model {equal_sh} {myModel_sh}{open_par_sh}{close_par_sh}

        {def_sh} {segment_sh}(self, {image_sh}, {kwargs_sh}):
            labels {equal_sh} self.model.{predict_sh}{open_par_sh}{image_sh}{close_par_sh}
            {return_sh} labels
    </code></pre>
    
    If it doesn't work, please report the issue {href} with the
    code you wrote. Thanks.
    """)
    return s


def setDefaultValueArgSpecsFromKwargs(params: List[ArgSpec], kwargs: Dict[str, object]):
    new_params = []
    for param in params:
        new_value = kwargs.get(param.name)
        if new_value is None:
            new_params.append(param)
            continue

        new_param = ArgSpec(
            name=param.name,
            default=new_value,
            type=param.type,
            desc=param.desc,
            docstring=param.docstring,
        )
        new_params.append(new_param)
    return new_params


def insertModelArgSpec(
    params, param_name, param_value, param_type=None, desc="", docstring=""
):
    updated_params = []
    for param in params:
        if param.name == param_name:
            if param_type is None:
                param_type = param.type
            new_param = ArgSpec(
                name=param_name,
                default=param_value,
                type=param_type,
                desc=desc,
                docstring=docstring,
            )
            updated_params.append(new_param)
        else:
            updated_params.append(param)
    return updated_params


def getModelArgSpec(acdcSegment):
    init_ArgSpec = inspect.getfullargspec(acdcSegment.Model.__init__)
    init_kwargs_type_hints = typing.get_type_hints(acdcSegment.Model.__init__)
    init_doc = acdcSegment.Model.__init__.__doc__
    init_params = params_to_ArgSpec(init_ArgSpec, init_kwargs_type_hints, init_doc)
    init_params = add_segm_data_param(init_params, init_ArgSpec)

    segment_ArgSpec = inspect.getfullargspec(acdcSegment.Model.segment)
    segment_kwargs_type_hints = typing.get_type_hints(acdcSegment.Model.segment)
    try:
        segment_ArgSpec.args.remove("frame_i")
    except Exception as e:
        pass

    segment_doc = acdcSegment.Model.segment.__doc__
    segment_params = params_to_ArgSpec(
        segment_ArgSpec,
        segment_kwargs_type_hints,
        segment_doc,
    )

    return init_params, segment_params


def parse_model_param_doc(name, next_param_name=None, docstring=None):
    if not docstring:
        return ""

    try:
        # Extract parameter description from 'param : ...'
        start_text = f"{name} : "
        if docstring.find(start_text) == -1:
            # Parameter not present in docstring
            return ""

        doc_start_idx = docstring.find(start_text) + len(start_text)

        doc_stop_idx = _get_doc_stop_idx(
            docstring, doc_start_idx, next_param_name=next_param_name
        )
        if doc_stop_idx == -1:
            doc_stop_idx = len(docstring)

        param_doc = docstring[doc_start_idx:doc_stop_idx]

        # Start at first end of line
        param_doc = param_doc[param_doc.find("\n") + 1 :]

        # Replace multiples spaces with single space
        param_doc = re.sub(" +", " ", param_doc)

        # Remove trailing spaces
        param_doc = param_doc.strip()
    except Exception as err:
        param_doc = ""

    param_doc = param_doc.replace(", optional", "")

    return param_doc


def params_to_ArgSpec(fullargspecs, type_hints, docstring, args_to_skip=None):
    params = []

    if fullargspecs.defaults is None:
        return params

    if args_to_skip is None:
        args_to_skip = set()

    num_params = len(fullargspecs.args)
    ip = num_params - len(fullargspecs.defaults)
    if ip < 0:
        return params

    for arg, default in zip(fullargspecs.args[ip:], fullargspecs.defaults):
        if arg in args_to_skip:
            continue

        if arg in type_hints:
            _type = type_hints[arg]
        else:
            _type = type(default)

        next_param_name = None
        if ip + 1 < num_params:
            next_param_name = fullargspecs.args[ip + 1]

        param_doc = parse_model_param_doc(
            arg, next_param_name=next_param_name, docstring=docstring
        )
        param = ArgSpec(
            name=arg, default=default, type=_type, desc=param_doc, docstring=docstring
        )
        params.append(param)
        ip += 1
    return params


def getClassArgSpecs(classModule, runMethodName="run"):
    init_ArgSpec = inspect.getfullargspec(classModule.__init__)
    init_kwargs_type_hints = typing.get_type_hints(classModule.__init__)
    init_doc = classModule.__init__.__doc__
    init_params = params_to_ArgSpec(init_ArgSpec, init_kwargs_type_hints, init_doc)

    run_ArgSpec = inspect.getfullargspec(getattr(classModule, runMethodName))
    run_kwargs_type_hints = typing.get_type_hints(getattr(classModule, runMethodName))
    run_doc = getattr(classModule, runMethodName).__doc__
    run_params = params_to_ArgSpec(
        run_ArgSpec,
        run_kwargs_type_hints,
        run_doc,
        args_to_skip={"signals", "export_to"},
    )
    return init_params, run_params


def getTrackerArgSpec(trackerModule, realTime=False):
    init_ArgSpec = inspect.getfullargspec(trackerModule.tracker.__init__)
    init_kwargs_type_hints = typing.get_type_hints(trackerModule.tracker.__init__)
    init_doc = trackerModule.tracker.__init__.__doc__
    init_params = params_to_ArgSpec(init_ArgSpec, init_kwargs_type_hints, init_doc)
    if realTime:
        track_ArgSpec = inspect.getfullargspec(trackerModule.tracker.track_frame)
        track_kwargs_type_hints = typing.get_type_hints(
            trackerModule.tracker.track_frame
        )
        track_doc = trackerModule.tracker.track_frame.__doc__
    else:
        track_ArgSpec = inspect.getfullargspec(trackerModule.tracker.track)
        track_kwargs_type_hints = typing.get_type_hints(trackerModule.tracker.track)
        track_doc = trackerModule.tracker.track.__doc__

    track_params = params_to_ArgSpec(
        track_ArgSpec,
        track_kwargs_type_hints,
        track_doc,
        args_to_skip={"signals", "export_to"},
    )
    return init_params, track_params


def isIntensityImgRequiredForTracker(trackerModule):
    track_ArgSpec = inspect.getfullargspec(trackerModule.tracker.track)
    num_args = len(track_ArgSpec.args) - len(track_ArgSpec.defaults)
    # If the number of args is 3 then we have `self, labels, image` as args
    # which means the tracker requires the image
    return num_args == 3


def download_examples(which="time_lapse_2D", progress=None):
    examples_path, example_path, url, file_size = get_examples_path(which)
    if os.path.exists(example_path):
        if progress is not None:
            # display 100% progressbar
            progress.emit(0, 0)
        return example_path

    zip_dst = os.path.join(examples_path, "example_temp.zip")

    if not os.path.exists(examples_path):
        os.makedirs(examples_path, exist_ok=True)

    print(f"Downloading example to {example_path}")

    download_url(url, zip_dst, verbose=False, file_size=file_size, progress=progress)
    exctract_to = examples_path
    extract_zip(zip_dst, exctract_to)

    if progress is not None:
        # display 100% progressbar
        progress.emit(0, 0)

    # Remove downloaded zip archive
    os.remove(zip_dst)
    print("Example downloaded successfully")
    return example_path


def check_model_exists(model_path, model_name):
    try:
        import cellacdc

        m = model_name.lower()
        weights_filenames = getattr(cellacdc, f"{m}_weights_filenames")
        files_present = listdir(model_path)
        return all([f in files_present for f in weights_filenames])
    except Exception as e:
        return True


def _model_url(model_name, return_alternative=False):
    if model_name == "YeaZ":
        url = "https://hmgubox2.helmholtz-muenchen.de/index.php/s/8PMePcwJXmaMMS6/download/YeaZ_weights.zip"
        alternative_url = (
            "https://zenodo.org/record/6125825/files/YeaZ_weights.zip?download=1"
        )
        file_size = 693685011
    elif model_name == "YeastMate":
        url = "https://hmgubox2.helmholtz-muenchen.de/index.php/s/pMT8pAmMkNtN8BP/download/yeastmate_weights.zip"
        alternative_url = (
            "https://zenodo.org/record/6140067/files/yeastmate_weights.zip?download=1"
        )
        file_size = 164911104
    elif model_name == "segment_anything":
        url = [
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        ]
        file_size = [2564550879, 1249524736, 375042383]
        alternative_url = ""
    elif model_name == "YeaZ_v2":
        url = [
            "https://hmgubox2.helmholtz-muenchen.de/index.php/s/5PARckkcJcN9D3S/download/weights_budding_BF_multilab_0_1",
            "https://hmgubox2.helmholtz-muenchen.de/index.php/s/CTHq4HN3adyFbnE/download/weights_budding_PhC_multilab_0_1",
            "https://hmgubox2.helmholtz-muenchen.de/index.php/s/QTtBJycYnLQZsHQ/download/weights_fission_multilab_0_2",
        ]
        file_size = [124142981, 124143031, 124144759]
        alternative_url = "https://github.com/rahi-lab/YeaZ-GUI#installation"
    elif model_name == "DeepSea":
        url = [
            "https://github.com/abzargar/DeepSea/raw/master/deepsea/trained_models/segmentation.pth",
            "https://github.com/abzargar/DeepSea/raw/master/deepsea/trained_models/tracker.pth",
        ]
        file_size = [7988969, 8637439]
        alternative_url = ""
    elif model_name == "TAPIR":
        url = ["https://storage.googleapis.com/dm-tapnet/tapir_checkpoint.npy"]
        file_size = [124408122]
        alternative_url = ""
    elif model_name == "Cellpose_germlineNuclei":
        url = [
            "https://hmgubox2.helmholtz-muenchen.de/index.php/s/AXG6fFfD8o5GZ83/download/cellpose_germlineNuclei_2023"
        ]
        file_size = [26570752]
        alternative_url = ""
    elif model_name == "omnipose":
        url = [
            "https://hmgubox2.helmholtz-muenchen.de/index.php/s/DynLkocWRbQfyRp/download/bact_fluor_cptorch_0"
            "https://hmgubox2.helmholtz-muenchen.de/index.php/s/2248Eoyozp3Ezj2/download/bact_fluor_omnitorch_0",
            "https://hmgubox2.helmholtz-muenchen.de/index.php/s/GiacDfXGerxE7PT/download/bact_phase_omnitorch_0",
            "https://hmgubox2.helmholtz-muenchen.de/index.php/s/DDq8s3CgnG2Yw6H/download/cyto2_omnitorch_0",
            "https://hmgubox2.helmholtz-muenchen.de/index.php/s/MM5meM2J5HbWqXR/download/plant_cptorch_0",
            "https://hmgubox2.helmholtz-muenchen.de/index.php/s/aap7znrWq5sE6JQ/download/plant_omnitorch_0",
            "https://hmgubox2.helmholtz-muenchen.de/index.php/s/w5M46x9qr8zLHZH/download/size_cyto2_omnitorch_0.npy",
        ]
        file_size = [26558464, 26558464, 26558464, 26558464, 26558464, 75071488, 4096]
        alternative_url = ""
    elif model_name == "sam2":
        url = [
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        ]
        file_size = [155233385, 184211977, 319128965, 910600801]
        alternative_url = ""
    else:
        return
    if return_alternative:
        return url, alternative_url
    else:
        return url, file_size


def _download_segment_anything_models():
    urls, file_sizes = _model_url("segment_anything")
    temp_model_path = tempfile.mkdtemp()
    _, final_model_path = get_model_path("segment_anything", create_temp_dir=False)
    for url, file_size in zip(urls, file_sizes):
        filename = url.split("/")[-1]
        final_dst = os.path.join(final_model_path, filename)
        if os.path.exists(final_dst):
            continue

        temp_dst = os.path.join(temp_model_path, filename)
        download_url(
            url, temp_dst, file_size=file_size, desc="segment_anything", verbose=False
        )

        shutil.move(temp_dst, final_dst)


def _download_sam2_models():
    urls, file_sizes = _model_url("sam2")
    temp_model_path = tempfile.mkdtemp()
    _, final_model_path = get_model_path("sam2", create_temp_dir=False)
    for url, file_size in zip(urls, file_sizes):
        filename = url.split("/")[-1]
        final_dst = os.path.join(final_model_path, filename)
        if os.path.exists(final_dst):
            continue

        temp_dst = os.path.join(temp_model_path, filename)
        download_url(url, temp_dst, file_size=file_size, desc="sam2", verbose=False)

        shutil.move(temp_dst, final_dst)


def _download_deepsea_models():
    urls, file_sizes = _model_url("DeepSea")
    temp_model_path = tempfile.mkdtemp()
    _, final_model_path = get_model_path("deepsea", create_temp_dir=False)
    for url, file_size in zip(urls, file_sizes):
        filename = url.split("/")[-1]
        final_dst = os.path.join(final_model_path, filename)
        if os.path.exists(final_dst):
            continue

        temp_dst = os.path.join(temp_model_path, filename)
        download_url(url, temp_dst, file_size=file_size, desc="deepsea", verbose=False)

        shutil.move(temp_dst, final_dst)


def download_manual():
    manual_folder_path = os.path.join(user_profile_path, "acdc-manual")
    if not os.path.exists(manual_folder_path):
        os.makedirs(manual_folder_path, exist_ok=True)

    manual_file_path = os.path.join(user_profile_path, "Cell-ACDC_User_Manual.pdf")
    if not os.path.exists(manual_file_path):
        url = "https://github.com/SchmollerLab/Cell_ACDC/raw/main/UserManual/Cell-ACDC_User_Manual.pdf"
        download_url(url, manual_file_path, file_size=1727470)
    return manual_file_path


def download_bioformats_jar(qparent=None, logger_info=print, logger_exception=print):
    dst_filepath = os.path.join(
        cellacdc_path, "bioformats", "jars", "bioformats_package.jar"
    )
    if os.path.exists(dst_filepath):
        return True, dst_filepath
    urls_to_try = (urls.bioformats_jar_home_url, urls.bioformats_jar_hmgu_url)
    success = False
    for url in urls_to_try:
        try:
            logger_info(f"Downloading `bioformats_package.jar`...")
            download_url(url, dst_filepath, file_size=43233280)
            success = True
            break
        except Exception as err:
            success = False
            traceback_str = traceback.format_exc()
            logger_exception(traceback_str)
            continue

    if success:
        return True, dst_filepath

    _warnings.warn_download_bioformats_jar_failed(dst_filepath, qparent=qparent)
    raise ModuleNotFoundError(
        "Bioformats package jar could not be downloaded. Please, "
        f"download it from here {urls.bioformats_download_page} and "
        f'place it in the following path "{dst_filepath}". '
        "Thank you for your patience!"
    )
    return False, dst_filepath


def download_url(url, dst, desc="", file_size=None, verbose=True, progress=None):
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    CHUNK_SIZE = 32768
    if verbose:
        print(f"Downloading {desc} to: {os.path.dirname(dst)}")
    response = requests.get(url, stream=True, timeout=20, verify=False)
    if file_size is not None and progress is not None:
        progress.emit(file_size, -1)
    pbar = tqdm(
        total=file_size, unit="B", unit_scale=True, unit_divisor=1024, ncols=100
    )
    with open(dst, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            # if chunk:
            f.write(chunk)
            pbar.update(len(chunk))
            if progress is not None:
                progress.emit(-1, len(chunk))
    pbar.close()


def _write_model_location_to_txt(model_name):
    model_info_path = os.path.join(models_path, model_name, "model")
    model_path = os.path.join(user_profile_path, f"acdc-{model_name}")
    file = "weights_location_path.txt"
    with open(os.path.join(model_info_path, file), "w") as txt:
        txt.write(model_path)
    return os.path.expanduser(model_path)


def download_model(model_name):
    if model_name == "segment_anything":
        try:
            _download_segment_anything_models()
            return True
        except Exception as e:
            traceback.print_exc()
            return False
    elif model_name == "sam2":
        try:
            _download_sam2_models()
            return True
        except Exception as e:
            traceback.print_exc()
            return False
    elif model_name == "DeepSea":
        try:
            _download_deepsea_models()
            return True
        except Exception as e:
            traceback.print_exc()
            return False
    elif model_name == "TAPIR":
        try:
            _download_tapir_model()
            return True
        except Exception as e:
            traceback.print_exc()
            return False
    elif model_name == "YeaZ_v2":
        try:
            _download_yeaz_models()
            return True
        except Exception as e:
            traceback.print_exc()
            return False
    elif model_name == "Cellpose_germlineNuclei":
        try:
            _download_cellpose_germlineNuclei_model()
            return True
        except Exception as e:
            traceback.print_exc()
            return False
    elif model_name == "omnipose":
        try:
            _download_omnipose_models()
            return True
        except Exception as err:
            return False
    elif model_name != "YeastMate" and model_name != "YeaZ":
        # We manage only YeastMate and YeaZ
        return True

    try:
        # Check if model exists
        temp_zip_path, model_path = get_model_path(model_name)
        if not temp_zip_path:
            # Model exists return
            return True

        # Check if user has model in the old v1.2.3 location
        v123_model_path = check_v123_model_path(model_name)
        if v123_model_path:
            print(f"Weights files found in {v123_model_path}")
            print(f"--> moving to new location: {model_path}...")
            for file in listdir(v123_model_path):
                src = os.path.join(v123_model_path, file)
                dst = os.path.join(model_path, file)
                shutil.copy(src, dst)
            return True

        # Download model from url to tempDir/model_temp.zip
        temp_dir = os.path.dirname(temp_zip_path)
        url, file_size = _model_url(model_name)
        print(f"Downloading {model_name} to {model_path}")
        download_url(
            url, temp_zip_path, file_size=file_size, desc=model_name, verbose=False
        )

        # Extract zip file inside temp dir
        print(f"Extracting model...")
        extract_zip(temp_zip_path, temp_dir, verbose=False)

        # Move unzipped files to ~/acdc-{model_name} folder
        print(f"Moving files from temporary folder to {model_path}...")
        for file in listdir(temp_dir):
            if file.endswith(".zip"):
                continue
            src = os.path.join(temp_dir, file)
            dst = os.path.join(model_path, file)
            shutil.move(src, dst)

        # Remove temp directory
        print(f"Removing temporary folder...")
        shutil.rmtree(temp_dir)
        return True

    except Exception as e:
        traceback.print_exc()
        return False


def aliases_real_time_trackers(reverse=False):
    """
    Returns a dictionary with aliases for real-time trackers.
    """

    aliases = {
        "CellACDC_normal_division": "Cell-ACDC symmetric division",
        "CellACDC_2steps": "Cell-ACDC 2 steps",
    }

    if reverse:
        aliases = {v: k for k, v in aliases.items()}

    return aliases


def get_list_of_real_time_trackers():
    trackers = get_list_of_trackers()
    rt_trackers = []
    aliases = aliases_real_time_trackers()
    for tracker in trackers:
        if tracker == "CellACDC":
            continue
        if tracker == "YeaZ":
            continue
        tracker_filename = f"{tracker}_tracker.py"
        tracker_path = os.path.join(
            cellacdc_path, "trackers", tracker, tracker_filename
        )
        try:
            with open(tracker_path) as file:
                txt = file.read()
            if txt.find("def track_frame") != -1:
                rt_trackers.append(tracker)
        except Exception as e:
            continue

    for i, tracker in enumerate(rt_trackers):
        if tracker in aliases:
            rt_trackers[i] = aliases[tracker]

    return natsorted(rt_trackers, key=str.casefold)


def get_list_of_trackers():
    trackers_path = os.path.join(cellacdc_path, "trackers")
    trackers = []
    for name in listdir(trackers_path):
        _path = os.path.join(trackers_path, name)
        tracker_script_path = os.path.join(_path, f"{name}_tracker.py")
        is_valid_tracker = (
            os.path.isdir(_path)
            and os.path.exists(tracker_script_path)
            and not name.endswith("__")
        )

        if name.startswith("_"):
            continue

        if is_valid_tracker:
            trackers.append(name)
    return natsorted(trackers, key=str.casefold)


def get_list_of_models():
    models = set()
    for name in listdir(models_path):
        _path = os.path.join(models_path, name)
        if not os.path.exists(_path):
            continue

        if not os.path.isdir(_path):
            continue

        if name.endswith("__"):
            continue

        if name.startswith("_"):
            continue

        if name == "skip_segmentation":
            continue

        if not os.path.exists(os.path.join(_path, "acdcSegment.py")):
            continue

        if name == "thresholding":
            name = "Automatic thresholding"

        models.add(name)

    if not os.path.exists(models_list_file_path):
        return natsorted(list(models), key=str.casefold)

    cp = config.ConfigParser()
    cp.read(models_list_file_path)
    models.update(cp.sections())
    return natsorted(list(models), key=str.casefold)


def get_list_of_promptable_models():
    models = set()
    for name in listdir(promptable_models_path):
        _path = os.path.join(promptable_models_path, name)
        if not os.path.exists(_path):
            continue

        if not os.path.isdir(_path):
            continue

        if name.endswith("__"):
            continue

        if not os.path.exists(os.path.join(_path, "acdcPromptSegment.py")):
            continue

        models.add(name)

    if not os.path.exists(promptable_models_list_file_path):
        return natsorted(list(models), key=str.casefold)

    cp = config.ConfigParser()
    cp.read(promptable_models_list_file_path)
    models.update(cp.sections())
    return natsorted(list(models), key=str.casefold)


def download_fiji(logger_func=print):
    url = None
    if is_mac:
        url = "https://downloads.micron.ox.ac.uk/fiji_update/mirrors/fiji-latest/fiji-macosx.zip"
        file_size = 474_525_405

    if url is None:
        return

    if os.path.exists(get_fiji_exec_folderpath()):
        return

    os.makedirs(acdc_fiji_path)

    temp_dir = tempfile.mkdtemp()
    zip_dst = os.path.join(temp_dir, "fiji-macosx.zip")
    logger_func(f'Downloading Fiji to "{acdc_fiji_path}"...')
    download_url(url, zip_dst, verbose=False, file_size=file_size)
    extract_zip(zip_dst, acdc_fiji_path)

    return acdc_fiji_path


def import_tracker_module(tracker_name):
    module_name = f"cellacdc.trackers.{tracker_name}.{tracker_name}_tracker"
    tracker_module = import_module(module_name)
    return tracker_module


def download_ffmpeg():
    ffmpeg_folderpath = acdc_ffmpeg_path
    if is_win:
        url = "https://hmgubox2.helmholtz-muenchen.de/index.php/s/rXioWZpwjwn9JTT/download/windows_ffmpeg-7.0-full_build.zip"
        file_size = 173477888
        ffmep_exec_path = os.path.join(ffmpeg_folderpath, "bin", "ffmpeg.exe")
    elif is_mac:
        url = "https://hmgubox2.helmholtz-muenchen.de/index.php/s/We7rcTLzqAP4zf7/download/mac_ffmpeg.zip"
        file_size = 25288704
        ffmep_exec_path = os.path.join(ffmpeg_folderpath, "ffmpeg")
    elif is_linux:
        ffmep_exec_path = ""
        return ffmep_exec_path

    if os.path.exists(ffmep_exec_path):
        return ffmep_exec_path.replace("\\", os.sep).replace("/", os.sep)

    print("Downloading FFMPEG...")
    temp_dir = tempfile.mkdtemp()
    temp_zip_path = os.path.join(temp_dir, "acdc-ffmpeg.zip")

    download_url(
        url,
        temp_zip_path,
        verbose=True,
        file_size=file_size,
    )
    extract_zip(temp_zip_path, ffmpeg_folderpath)

    return ffmep_exec_path.replace("\\", os.sep).replace("/", os.sep)


def import_promptable_segment_module(model_name):
    try:
        acdcPromptSegment = import_module(
            f"cellacdc.segmenters_promptable.{model_name}.acdcPromptSegment"
        )
    except ModuleNotFoundError as e:
        # Check if custom model
        cp = config.ConfigParser()
        cp.read(promptable_models_list_file_path)
        model_path = cp[model_name]["path"]
        spec = importlib.util.spec_from_file_location("acdcPromptSegment", model_path)
        acdcPromptSegment = importlib.util.module_from_spec(spec)
        sys.modules["acdcPromptSegment"] = acdcPromptSegment
        spec.loader.exec_module(acdcPromptSegment)
    return acdcPromptSegment


def init_tracker(
    posData, trackerName, realTime=False, qparent=None, return_init_params=False
):
    from . import apps

    downloadWin = apps.downloadModel(trackerName, parent=qparent)
    downloadWin.download()

    trackerModule = import_tracker_module(trackerName)
    init_params = {}
    track_params = {}
    paramsWin = None
    if trackerName == "BayesianTracker":
        Y, X = posData.img_data_shape[-2:]
        if posData.isSegm3D:
            labShape = (posData.SizeZ, Y, X)
        else:
            labShape = (1, Y, X)
        paramsWin = apps.BayesianTrackerParamsWin(
            labShape,
            parent=qparent,
            channels=posData.chNames,
            currentChannelName=posData.user_ch_name,
        )
        paramsWin.exec_()
        if not paramsWin.cancel:
            init_params = paramsWin.params
            track_params["export_to"] = posData.get_btrack_export_path()
            if paramsWin.intensityImageChannel is not None:
                chName = paramsWin.intensityImageChannel
                track_params["image"] = posData.loadChannelData(chName)
                track_params["image_channel_name"] = chName
    elif trackerName == "CellACDC":
        paramsWin = apps.CellACDCTrackerParamsWin(parent=qparent)
        paramsWin.exec_()
        if not paramsWin.cancel:
            init_params = paramsWin.params
    elif trackerName == "delta":
        paramsWin = apps.DeltaTrackerParamsWin(posData=posData, parent=qparent)
        paramsWin.exec_()
        if not paramsWin.cancel:
            init_params = paramsWin.params
    else:
        init_argspecs, track_argspecs = getTrackerArgSpec(
            trackerModule, realTime=realTime
        )
        intensityImgRequiredForTracker = isIntensityImgRequiredForTracker(trackerModule)
        if init_argspecs or track_argspecs:
            try:
                url = trackerModule.url_help()
            except AttributeError:
                url = None
            try:
                channels = posData.chNames
            except Exception as e:
                channels = None
            try:
                currentChannelName = posData.user_ch_name
            except Exception as e:
                currentChannelName = None
            try:
                df_metadata = posData.metadata_df
            except Exception as e:
                df_metadata = None

            if not intensityImgRequiredForTracker:
                currentChannelName = None

            paramsWin = apps.QDialogModelParams(
                init_argspecs,
                track_argspecs,
                trackerName,
                url=url,
                channels=channels,
                is_tracker=True,
                currentChannelName=currentChannelName,
                df_metadata=df_metadata,
                posData=posData,
            )
            if not intensityImgRequiredForTracker and channels is not None:
                paramsWin.channelCombobox.setDisabled(True)

            paramsWin.exec_()
            if not paramsWin.cancel:
                init_params = paramsWin.init_kwargs
                track_params = paramsWin.model_kwargs
                if paramsWin.inputChannelName != "None":
                    chName = paramsWin.inputChannelName
                    track_params["image"] = posData.loadChannelData(chName)
                    track_params["image_channel_name"] = chName
        if "export_to_extension" in track_params:
            ext = track_params["export_to_extension"]
            track_params["export_to"] = posData.get_tracker_export_path(
                trackerName, ext
            )

    if paramsWin is not None and paramsWin.cancel:
        tracker = (None,)
        track_params = None
        init_params = None
    else:
        tracker = trackerModule.tracker(**init_params)

    if return_init_params:
        return tracker, track_params, init_params
    else:
        return tracker, track_params


def _download_tapir_model():
    urls, file_sizes = _model_url("TAPIR")
    temp_model_path = tempfile.mkdtemp()
    _, final_model_path = get_model_path("TAPIR", create_temp_dir=False)
    for url, file_size in zip(urls, file_sizes):
        filename = url.split("/")[-1]
        final_dst = os.path.join(final_model_path, filename)
        if os.path.exists(final_dst):
            continue

        temp_dst = os.path.join(temp_model_path, filename)
        download_url(url, temp_dst, file_size=file_size, desc="TAPIR", verbose=False)

        shutil.move(temp_dst, final_dst)


def _download_yeaz_models():
    urls, file_sizes = _model_url("YeaZ_v2")
    temp_model_path = tempfile.mkdtemp()
    _, final_model_path = get_model_path("YeaZ_v2", create_temp_dir=False)
    for url, file_size in zip(urls, file_sizes):
        filename = url.split("/")[-1]
        final_dst = os.path.join(final_model_path, filename)
        if os.path.exists(final_dst):
            continue

        temp_dst = os.path.join(temp_model_path, filename)
        download_url(url, temp_dst, file_size=file_size, desc="YeaZ_v2", verbose=False)

        shutil.move(temp_dst, final_dst)


def _download_cellpose_germlineNuclei_model():
    urls, file_sizes = _model_url("Cellpose_germlineNuclei")
    temp_model_path = tempfile.mkdtemp()
    _, final_model_path = get_model_path(
        "Cellpose_germlineNuclei", create_temp_dir=False
    )
    for url, file_size in zip(urls, file_sizes):
        filename = url.split("/")[-1]
        final_dst = os.path.join(final_model_path, filename)
        if os.path.exists(final_dst):
            continue

        temp_dst = os.path.join(temp_model_path, filename)
        download_url(
            url,
            temp_dst,
            file_size=file_size,
            desc="Cellpose_germlineNuclei",
            verbose=False,
        )

        shutil.move(temp_dst, final_dst)


def _download_omnipose_models():
    urls, file_sizes = _model_url("omnipose")
    temp_model_path = tempfile.mkdtemp()
    final_model_path = os.path.expanduser(r"~\.cellpose\models")
    for url, file_size in zip(urls, file_sizes):
        filename = url.split("/")[-1]
        final_dst = os.path.join(final_model_path, filename)
        if os.path.exists(final_dst):
            continue

        temp_dst = os.path.join(temp_model_path, filename)
        download_url(url, temp_dst, file_size=file_size, desc="omnipose", verbose=False)

        shutil.move(temp_dst, final_dst)


def init_prompt_segm_model(acdcPromptSegment, posData, init_kwargs):
    model = acdcPromptSegment.Model(**init_kwargs)
    return model


def init_segm_model(acdcSegment, posData, init_kwargs):
    segm_endname = init_kwargs.pop("segm_endname", "None")
    if segm_endname != "None":
        load_segm = True
        if not hasattr(posData, "segm_data"):
            load_segm = True
        elif posData.segm_npz_path.endswith(f"{segm_endname}.npz"):
            load_segm = False
        if not load_segm:
            segm_data = np.squeeze(posData.segm_data)
        else:
            segm_filepath, _ = load.get_path_from_endname(
                segm_endname, posData.images_path
            )
            printl(f'Loading segmentation data from "{segm_filepath}"...')
            segm_data = np.load(segm_filepath)["arr_0"]
    else:
        segm_data = None

    # Initialize input_points_df for models promptable with points
    input_points_filepath = init_kwargs.pop("input_points_path", "")
    if input_points_filepath:
        input_points_df = init_input_points_df(posData, input_points_filepath)
        init_kwargs["input_points_df"] = input_points_df

    try:
        # Models introduced before 1.3.2 do not have the segm_data as input
        kwargs = inspect.getfullargspec(acdcSegment.Model.__init__).args
        if "is_rgb" not in kwargs and "is_rgb" in init_kwargs:
            del init_kwargs["is_rgb"]
        model = acdcSegment.Model(**init_kwargs)

    except Exception as e:
        model = acdcSegment.Model(segm_data, **init_kwargs)

    if hasattr(model, "init_successful"):
        if not model.init_successful:
            return None
    return model


def parse_model_params(model_argspecs, model_params):
    parsed_model_params = {}
    for row, argspec in enumerate(model_argspecs):
        value = model_params.get(argspec.name)
        if value is None:
            continue
        if argspec.type == bool:
            value = _parse_bool_str(value)
        elif argspec.type == int:
            value = int(value)
        elif argspec.type == float:
            value = float(value)
        parsed_model_params[argspec.name] = value
    return parsed_model_params


def validate_tracker_input(tracker, segm_video_to_track):
    try:
        warning_text = tracker.validate_input(segm_video_to_track)
        return warning_text
    except Exception as err:
        printl(traceback.format_exc())
        pass
    return

# Sibling imports (deferred to avoid import cycles)
from .misc import (
    _get_doc_stop_idx,
    _parse_bool_str,
    add_segm_data_param,
    extract_zip,
    init_input_points_df,
)
from .paths import (
    check_v123_model_path,
    get_examples_path,
    get_fiji_exec_folderpath,
    get_model_path,
    listdir,
)

