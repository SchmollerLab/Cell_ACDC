# -*- coding: utf-8 -*-
"""
Module for configuration files and configuration parameters

@author: jblugagne
"""
import os as _os
import json as _json
from warnings import warn as _warn
from typing import Tuple, Optional

from . import __file__ as _delta_init  # Path to __init__ file

_DELTA_DIR = _os.path.dirname(_delta_init)
"DeLTA lib install directory"
_LOADED = None
"Which config file was loaded"

# Parameters:
presets: str = ""
"Type of analysis: can be '2D', 'mothermachine', or your own custom name for presets"
models: Tuple[str, ...] = ()
"Which models will be run"
model_file_rois: str = ""
"Model file for ROIs segmentation"
model_file_seg: str = ""
"Model file for cell segmentation"
model_file_track: str = ""
"Model file for cell tracking"
target_size_rois: Tuple[int, int] = (0, 0)
"ROI U-Net target/input image size"
target_size_seg: Tuple[int, int] = (0, 0)
"segmentation U-Net target/input image size"
target_size_track: Tuple[int, int] = (0, 0)
"tracking U-Net target/input image size"
training_set_rois: str = ""
"Path to ROIs U-Net training set"
training_set_seg: str = ""
"Path to segmentation training set"
training_set_track: str = ""
"Path to tracking training set"
eval_movie: str = ""
"Path to evaluation movie / image sequence"
rotation_correction: bool = False
"Flag to try to automatically correct image rotation (for microfluidic devices)"
drift_correction: bool = False
"Flag to correct drift over time (for microfluidic devices / ROIs)"
crop_windows: bool = False
"Flag to crop input image into windows of size target_size_seg for segmentation, otherwise resize them"
min_roi_area: float = 0.0
"Minimum area of detected ROIs, in pixels. Can be set to 0. (N/A for 2D)"
whole_frame_drift: bool = False
"If correcting for drift, use the entire frame instead of the region above the chambers"
min_cell_area: float = 0.0
"Minimum area of detected cells, in pixels. Can be set to 0"
save_format: Tuple[str, ...] = ()
"Format to save output data to"
TF_CPP_MIN_LOG_LEVEL: str = ""
"Debugging messages level from Tensorflow ('0' = most verbose to '3' = not verbose)"
memory_growth_limit: Optional[float] = None
"""If running into OOM issues or having trouble with cuDNN loading, try setting 
memory_growth_limit to a value in MB: (eg 1024, 2048...)"""
pipeline_seg_batch: int = 0
"""If running into OOM issues during segmentation with the pipeline, try lowering 
this value. You can also try to increase it to improve speed"""
pipeline_track_batch: int = 0
"""If running into OOM issues during tracking with the pipeline, try lowering 
this value. You can also try to increase it to improve speed"""

"""
IMPORTANT: Do not change the default parameters below. Update the .json files instead if necessary.
"""

_DEFAULTS_2D = dict(
    presets="2D",  # Type of analysis. Can be '2D', 'mothermachine', or your own custom name for presets
    models=("segmentation", "tracking"),  # Which models will be run
    model_file_rois="",  # Model file for ROIs segmentation (None for 2D)
    model_file_seg="D:/DeLTA_data/agar_pads/unet_pads_seg.hdf5",  # Model file for cell segmentation (placeholder)
    model_file_track="D:/DeLTA_data/agar_pads/unet_pads_track.hdf5",  # Model file for cell tracking(placeholder)
    target_size_rois=(512, 512),  # ROI U-Net target/input image size
    target_size_seg=(512, 512),  # segmentation U-Net target/input image size
    target_size_track=(256, 256),  # tracking U-Net target/input image size
    training_set_rois="",  # Path to ROIs U-Net training set (None for 2D)
    training_set_seg="D:/DeLTA_data/agar_pads/train/seg/",  # Path to segmentation training set (placeholder)
    training_set_track="D:/DeLTA_data/agar_pads/train/track/",  # Path to tracking training set (placeholder)
    eval_movie="D:/DeLTA_data/agar_pads/eval_movie/tifs/",  # Path to evaluation movie / image sequence
    rotation_correction=False,  # Flag to try to automatically correct image rotation (for microfluidic devices)
    drift_correction=False,  # Flag to correct drift over time (for microfluidic devices / ROIs)
    crop_windows=True,  # Flag to crop input image into windows of size target_size_seg for segmentation, otherwise resize them
    min_roi_area=500,  # Minimum area of detected ROIs, in pixels. Can be set to 0. (N/A for 2D)
    whole_frame_drift=False,  # If correcting for drift, use the entire frame instead of the region above the chambers
    min_cell_area=20,  # Minimum area of detected cells, in pixels. Can be set to 0
    save_format=("pickle", "legacy", "movie"),  # Format to save output data to.
    TF_CPP_MIN_LOG_LEVEL="2",  # Debugging messages level from Tensorflow ('0' = most verbose to '3' = not verbose)
    memory_growth_limit=None,  # If running into OOM issues or having trouble with cuDNN loading, try setting memory_growth_limit to a value in MB: (eg 1024, 2048...)
    pipeline_seg_batch=64,  # If running into OOM issues during segmentation with the pipeline, try lowering this value. You can also try to increase it to improve speed
    pipeline_track_batch=64,  # If running into OOM issues during tracking with the pipeline, try lowering this value. You can also try to increase it to improve speed
)

_DEFAULTS_mothermachine = dict(
    presets="mothermachine",  # Type of analysis. Can be '2D', 'mothermachine', or your own custom name for presets
    models=("rois", "segmentation", "tracking"),  # Which models will be run
    model_file_rois="D:/DeLTA_data/mother_machine/models/chambers_id_tessiechamp.hdf5",  # Model file for ROIs segmentation (placeholder)
    model_file_seg="D:/DeLTA_data/mother_machine/models/unet_moma_seg_multisets.hdf5",  # Model file for cell segmentation (placeholder)
    model_file_track="D:/DeLTA_data/mother_machine/models/unet_moma_track_v2.hdf5",  # Model file for cell tracking(placeholder)
    target_size_rois=(512, 512),  # ROI U-Net target/input image size
    target_size_seg=(256, 32),  # segmentation U-Net target/input image size
    target_size_track=(256, 32),  # tracking U-Net target/input image size
    training_set_rois="D:/DeLTA_data/mother_machine/training/chambers_seg_set/train",  # Path to ROIs U-Net training set (None for 2D)
    training_set_seg="D:/DeLTA_data/mother_machine/training/segmentation_set/train_multisets/",  # Path to segmentation training set (placeholder)
    training_set_track="D:/DeLTA_data/mother_machine/training/tracking_set/train_multisets",  # Path to tracking training set (placeholder)
    eval_movie="D:/DeLTA_data/mother_machine/eval_movie/tifs/",  # Path to evaluation movie / image sequence
    rotation_correction=True,  # Flag to try to automatically correct image rotation (for microfluidic devices)
    drift_correction=True,  # Flag to correct drift over time (for microfluidic devices / ROIs)
    crop_windows=False,  # Flag to crop input image into windows of size target_size_seg for segmentation, otherwise resize them
    min_roi_area=500,  # Minimum area of detected ROIs, in pixels. Can be set to 0. (N/A for 2D)
    whole_frame_drift=False,  # If correcting for drift, use the entire frame instead of the region above the chambers
    min_cell_area=20,  # Minimum area of detected cells, in pixels. Can be set to 0
    save_format=("pickle", "legacy", "movie"),  # Format to save output data to.
    TF_CPP_MIN_LOG_LEVEL="2",  # Debugging messages level from Tensorflow ('0' = most verbose to '3' = not verbose)
    memory_growth_limit=None,  # If running into OOM issues or having trouble with cuDNN loading, try setting memory_growth_limit to a value in MB: (eg 1024, 2048...)
    pipeline_seg_batch=64,  # If running into OOM issues during segmentation with the pipeline, try lowering this value. You can also try to increase it to improve speed
    pipeline_track_batch=64,  # If running into OOM issues during tracking with the pipeline, try lowering this value. You can also try to increase it to improve speed
)


def load_config(json_file: str = None, presets: str = "2D", config_level: str = None):
    """
    Loads json configuration files

    Parameters
    ----------
    json_file : str or None, optional
        Path to json file containing configuration. If None, load_config will
        search for local or global config files based on the presets value and
        config_level.
        The default is None.
    presets : str, optional
        If json_file is None, search for files named 'config_<presets>.json in
        the local user folder under .delta and then if not found in the global
        folder (ie the install folder).
        The default is '2D'.
    config_level : str or None, optional
        If 'local', look for preset config file in the local user folder under
        ~/.delta. If 'global', look in the delta install folder under
        assets/config. If None, look under local first and then global.
        The default is None.

    Returns
    -------
    None.

    """

    if presets == "2D":
        defaults = _DEFAULTS_2D
    elif presets == "mothermachine":
        defaults = _DEFAULTS_mothermachine
    else:
        raise ValueError(
            """Valid presets are '2D' and 'mothermachine'.
            If you implemented very different presets, please provide
            a config file to load_config() directely
            """
        )

    if json_file is None:

        # Is there a local/user config file for this preset?
        _json_file = _os.path.expanduser(
            _os.path.join("~/.delta", f"config_{presets}.json")
        )
        if _os.path.exists(_json_file) and (
            config_level is None or config_level == "local"
        ):
            json_file = _json_file

        # Is there a global config file for this preset?
        _os.path.join(_DELTA_DIR, "assets", "config")
        _json_file = _os.path.join(
            _DELTA_DIR, "assets", "config", f"config_{presets}.json"
        )
        if (
            json_file is None
            and _os.path.exists(_json_file)
            and (config_level is None or config_level == "global")
        ):
            json_file = _json_file

        # If no file was found, raise error:
        if json_file is None:
            raise ValueError(
                f"""Could not find a local or global config file for presets '{presets}'.
                Either use delta.assets.download_assets() or provide a json_file path
                """
            )

    variables = _read_json(json_file)

    # Check if it has the same parameters set as defaults:
    if set(variables.keys()) != set(defaults.keys()):
        _warn(
            "The config file keys differ from the defaults. This is most "
            "likely because the config file was generated for an earlier version"
            " of DeLTA. This may cause issues, to avoid this either use "
            "delta.assets.download_assets() again or update the config file "
            "manually to match the keys in the defaults. "
            "run print(delta.config._DEFAULTS_2D) "
            "or print(delta.config._DEFAULTS_mothermachine) for defaults."
        )

    # Update config variables:
    globals().update(variables)

    global _LOADED
    _LOADED = json_file

    # Tensorflow technical parameters:
    # Debugging messages level from Tensorflow ('0' = most verbose to '3' = not verbose)
    _os.environ["TF_CPP_MIN_LOG_LEVEL"] = TF_CPP_MIN_LOG_LEVEL

    # If running into OOM issues or having trouble with cuDNN loading, try setting
    # memory_growth_limit to a value in MB: (eg 1024, 2048...)
    if memory_growth_limit is not None:
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=memory_growth_limit
                        )
                    ],
                )
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)


def _read_json(json_file: str):

    # Load file:
    print(f"Loading configuration from: {json_file}")
    with open(json_file, "r") as f:
        variables = _json.loads(f.read())

    # Type cast:
    for k, v in variables.items():
        if isinstance(v, list):
            variables[k] = tuple(v)  # Always use tuples, not lists in config

    return variables
