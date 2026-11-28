"""Non-UI utilities of the measurements viewer (`measurements_viewer`).

Module-level functions load and normalize the positions' data (images
paths, `acdc_output` tables, last analyzed frame and frame masking),
evaluate masked per-cell equations, give signal colors and set up the
viewer's logger. `MeasurementsViewerUtils` collects the data and logic
methods of `measurements_viewer.MeasurementsViewerWin` that do not touch
widgets (tables, positions, configurations, predictors, time points, custom
annotations, phases and measurement steps). The window inherits it as a
mixin (right after ``QMainWindow``), so ``self`` is the viewer window and
the methods read its data attributes (``self.data``, ``self.acdc_df``,
``self.phaseTranslatorRules``, ...).

Keep this module free of widget code: methods that show dialogs, update
plots or widgets belong to `measurements_viewer`. The behavior of these
functions is specified in the `measurements_viewer` module docstring.
"""
import copy
import json
import keyword
import logging
import os
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyqtgraph as pg

from qtpy.QtCore import QSaveFile, QIODevice
from qtpy.QtGui import QColor

from cellacdc import load
from cellacdc.myutils import filesystem
from . import phase_translator
from cellacdc import (
    cca_df_colnames, core, timepoint_predictors, measurement_processing,
    background_values, settings_csv_path, qutils
)
from cellacdc.shared.dialogs import combine_measurements as dialogs_combine_measurements
from cellacdc.timepoint_predictors import TIMEPOINT_PREFIX, parse_flags
from .dialogs.conditional_formatting import (
    _FormattingConditionDialog,
    is_formatting_color,
)
from cellacdc.gui.mixins.custom_annotations import custom_annot_state_to_save
from .custom_annotation_predictors import CustomAnnotationPredictor
from . import position_data as viewer_position_data
from . import annotation_data as viewer_annotation_data
from .position_data import PositionDataUtils
from .annotation_data import AnnotationDataUtils


# Sections of a premade template that are used during setup only (not
# written to the instantiated configuration)
PREMADE_TEMPLATE_KEYS = (
    'channel_placeholders', 'input_placeholders', 'background_values',
    'channel_hints'
)


# Mean intensity column ``{NAME}_mean`` of a channel token in a premade
# template (a whole column name: not ``{NAME}_mean_x`` nor
# ``{A}_over_{B}_mean``)
PREMADE_MEAN_COLUMN_RE = re.compile(r'(?<![\w}])\{([A-Za-z0-9_]+)\}_mean(?!\w)')


# Default colors of the plotted columns: the colorblind-friendly Okabe-Ito
# palette without black (dark plot background) and with vermillion last,
# because the current frame line is red.
SIGNAL_COLORS = (
    '#56B4E9', '#E69F00', '#009E73', '#F0E442', '#0072B2', '#CC79A7',
    '#D55E00',
)


# On the light plot background (Invert black/white) the Okabe-Ito yellow is
# nearly invisible: black takes its place (same index, so colors stay stable)
LIGHT_BACKGROUND_SIGNAL_COLORS = tuple(
    '#000000' if color == '#F0E442' else color for color in SIGNAL_COLORS
)


# Further columns: tab10 without its red and brown, then evenly spaced hues
# that skip red (hues between 30 and 330 degrees)
EXTRA_SIGNAL_COLORS = (
    '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf',
)


_SIGNAL_FALLBACK_HUES = 9


def signal_color(index, light_background=False):
    """Default plotting color (QColor) number `index` of a plotted column.

    No default color is red (reserved for the current frame line). Pass
    `light_background=True` for the inverted (light) plot background.
    """
    palette = (
        LIGHT_BACKGROUND_SIGNAL_COLORS if light_background else SIGNAL_COLORS
    )
    if index < len(palette):
        return pg.mkColor(palette[index])
    index -= len(palette)
    if index < len(EXTRA_SIGNAL_COLORS):
        return pg.mkColor(EXTRA_SIGNAL_COLORS[index])
    index -= len(EXTRA_SIGNAL_COLORS)
    hues = _SIGNAL_FALLBACK_HUES
    # Each round of hues is shifted by a third of a step to stay distinct
    step = (index % hues) + 0.5 + (index // hues % 3)/3
    hue = (30 + 300*step/hues) / 360
    value = 0.75 if light_background else 1.0
    color = QColor()
    color.setHsvF(min(hue, 330/360), 0.85, value)
    return color


LOGGER_NAME = 'cellacdc-measurements-viewer'


class _StdoutLogHandler(logging.StreamHandler):
    """Handler that always writes to the *current* ``sys.stdout``.

    The stream is looked up at every record (assigning ``stream`` is
    ignored), so redirections done after the handler was created (pytest
    capture, the launcher's ``myutils.Logger``, which also writes the log
    file) keep receiving the messages.
    """
    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass

    def emit(self, record):
        stream = self.stream
        if stream is None:
            # E.g., pythonw without a console
            return
        try:
            text = self.format(record) + self.terminator
            try:
                stream.write(text)
            except UnicodeEncodeError:
                encoding = getattr(stream, 'encoding', None) or 'ascii'
                stream.write(
                    text.encode(encoding, errors='replace').decode(encoding)
                )
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


class _LogFormatter(logging.Formatter):
    """INFO records are printed as the bare message, other levels as
    ``[LEVEL]: message`` (with the traceback of ``logger.exception``).
    Messages that already start with ``[LEVEL]`` (e.g. of the mixins shared
    with the main GUI) are printed as they are."""
    def format(self, record):
        text = super().format(record)
        if record.levelno == logging.INFO:
            return text
        if text.lstrip().startswith(f'[{record.levelname}]'):
            return text
        return f'[{record.levelname}]: {text}'


def get_logger():
    """Logger of the measurements viewer, printing to the terminal.

    The stdout handler is attached only once (several windows share the
    logger) and the logger does not propagate to the root logger (whose
    handlers, if configured by a third-party library, would print every
    message twice), so use ``capsys``, not ``caplog``, in tests.
    """
    logger = logging.getLogger(LOGGER_NAME)
    hasHandler = any(
        getattr(handler, '_acdcMeasurementsViewerStdout', False)
        for handler in logger.handlers
    )
    if not hasHandler:
        handler = _StdoutLogHandler()
        handler._acdcMeasurementsViewerStdout = True
        handler.setFormatter(_LogFormatter('%(message)s'))
        logger.addHandler(handler)
        logger.propagate = False
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return logger


def get_images_paths(folderpath):
    """Get the list of `Images` folders from an experiment, Position or 
    Images folder path."""
    folderpath = os.path.normpath(folderpath)
    if os.path.isfile(folderpath):
        folderpath = os.path.dirname(folderpath)
    
    if os.path.basename(folderpath) == 'Images':
        return [folderpath]
    
    images_path = os.path.join(folderpath, 'Images')
    if os.path.isdir(images_path):
        return [images_path]
    
    pos_foldernames = filesystem.get_pos_foldernames(folderpath)
    images_paths = [
        os.path.join(folderpath, pos, 'Images') for pos in pos_foldernames
    ]
    return [path for path in images_paths if os.path.isdir(path)]


def get_channels_and_segm_endnames(images_path):
    """Return the channel names and the segmentation files end names 
    (e.g., 'segm' or 'segm_<name>') available in `images_path`."""
    basename, chNames = filesystem.getBasenameAndChNames(images_path)
    segm_files = load.get_segm_files(images_path)
    segm_endnames = load.get_endnames(basename, segm_files)
    segm_endnames = [
        endname.lstrip('_') for endname in segm_endnames 
        if not endname.endswith('_raw') and 'raw_postproc' not in endname
    ] or [endname.lstrip('_') for endname in segm_endnames]
    return chNames, sorted(set(segm_endnames))


def acdc_output_columns(images_path, segm_endname='segm'):
    """Column names (set) of the acdc_output table that `load_position`
    loads with the segmentation `segm_endname` of `images_path` (only the
    header is read), None if the file does not exist."""
    basename, _ = filesystem.getBasenameAndChNames(images_path)
    endname = (segm_endname or 'segm').replace('segm', 'acdc_output')
    filepath = os.path.join(images_path, f'{basename}{endname}.csv')
    if not os.path.exists(filepath):
        return None
    return set(pd.read_csv(filepath, nrows=0).columns)


def get_channel_from_filepath(file_path):
    """Return the channel name of an image file in an Images folder or None 
    if it cannot be determined."""
    images_path = os.path.dirname(file_path)
    filename, _ = os.path.splitext(os.path.basename(file_path))
    try:
        basename, chNames = filesystem.getBasenameAndChNames(images_path)
    except Exception as err:
        return
    
    matching = [ch for ch in chNames if filename.endswith(ch)]
    if not matching:
        return
    return max(matching, key=len)


def load_position(images_path, channel, segm_endname='segm'):
    """Load image, segmentation, acdc_output table and metadata of one
    Position.

    Image and segmentation data always have the time axis first and are
    cut to ``SizeT``, the number of frames of the shorter of both.
    """
    img_path = load.get_filename_from_channel(images_path, channel)
    if not img_path:
        raise FileNotFoundError(
            f'Could not find a file for channel "{channel}" '
            f'in "{images_path}".'
        )
    posData = load.loadData(img_path, channel)
    posData.getBasenameAndChNames()
    posData.buildPaths()
    posData.loadImgData()
    posData.loadOtherFiles(
        load_segm_data=True,
        load_acdc_df=True,
        load_metadata=True,
        loadSegmInfo=True,
        end_filename_segm=segm_endname
    )
    if not posData.segmFound:
        raise FileNotFoundError(
            f'Segmentation file ending with "{segm_endname}" not found '
            f'in "{images_path}".'
        )
    
    if posData.SizeZ > 1 and posData.img_data.ndim < 3:
        posData.SizeZ = 1
    
    if posData.SizeT == 1:
        posData.img_data = posData.img_data[np.newaxis]
        posData.segm_data = posData.segm_data[np.newaxis]
    
    posData.SizeT = min(len(posData.img_data), len(posData.segm_data))
    # Per-frame structures (e.g. `allData_li`) are built from `segm_data`
    # with `SizeT` items: frames beyond the shorter file are not shown
    posData.img_data = posData.img_data[:posData.SizeT]
    posData.segm_data = posData.segm_data[:posData.SizeT]
    normalize_loaded_acdc_df(posData)
    update_last_analyzed_frame(posData)
    posData.savedAcdcDf = acdc_df_snapshot(getattr(posData, 'acdc_df', None))
    posData.acdcDfFileStamp = viewer_position_data.acdc_df_file_stamp(
        getattr(posData, 'acdc_output_csv_path', None)
    )
    return posData


def normalize_loaded_acdc_df(posData):
    """Restore the viewer-owned columns of a table read from file (in place):
    parse the boolean ``timepoint_*`` and visited columns with `parse_flags`
    (the main GUI may save them as True/False/0 strings after adding cells)
    and restore
    unassigned phases (which the loader fills with 0 when the table has cell
    cycle columns) to '' and recognized ``phase_<name>`` markers to bool
    (`phase_translator.normalize_loaded_outputs`)."""
    acdc_df = getattr(posData, 'acdc_df', None)
    if acdc_df is None:
        return
    for column in acdc_df.columns:
        if column == viewer_annotation_data.VISITED_COLUMN or (
                isinstance(column, str) and column.startswith(TIMEPOINT_PREFIX)):
            acdc_df[column] = parse_flags(acdc_df[column])
    phase_translator.normalize_loaded_outputs(acdc_df)


LAST_ANALYZED_CCA = 'cell cycle annotated'


LAST_ANALYZED_TRACKED = 'tracked'


def last_analyzed_frame(acdc_df, SizeT=None):
    """Last frame index of a table that was analyzed in the main GUI and
    how it was found, as ``(frame_i, kind)``, or ``(None, None)`` for a
    missing or empty table.

    Like the main GUI, it is the last cell cycle annotated frame (the last
    frame with rows whose cell cycle columns are all filled, see
    `load.loadData.append_last_cca_frame`) when the table has a
    ``cell_cycle_stage`` column with annotated rows (kind
    `LAST_ANALYZED_CCA`), else the last tracked frame, i.e. the highest
    ``frame_i`` of the table (`LAST_ANALYZED_TRACKED`). Clipped to
    ``SizeT - 1`` when `SizeT` is given. Pure pandas (thread-safe)."""
    if acdc_df is None or acdc_df.empty:
        return None, None
    if 'frame_i' in acdc_df.columns:
        frames = acdc_df['frame_i']
    else:
        frames = pd.Series(acdc_df.index.get_level_values('frame_i'))
    frames = pd.to_numeric(frames, errors='coerce').to_numpy()
    last, kind = None, None
    if 'cell_cycle_stage' in acdc_df.columns:
        cca_cols = [col for col in cca_df_colnames if col in acdc_df.columns]
        annotated = acdc_df[cca_cols].notna().all(axis=1).to_numpy()
        annotated &= np.isfinite(frames)
        if annotated.any():
            last, kind = int(frames[annotated].max()), LAST_ANALYZED_CCA
    if last is None:
        finite = frames[np.isfinite(frames)]
        if not len(finite):
            return None, None
        last, kind = int(finite.max()), LAST_ANALYZED_TRACKED
    if SizeT is not None:
        last = min(last, int(SizeT) - 1)
    return last, kind


def min_frame(*frames):
    """Smallest of the last frame indices `frames`, where None means all
    frames (no limit). None if all are None."""
    frames = [frame for frame in frames if frame is not None]
    return min(frames) if frames else None


def float_if_bool(values):
    """Boolean `values` (Series) as float (1.0/0.0), so that NaN on the
    frames ignored by the calculations keeps the column numeric (an object
    column drops out of the plottable columns). Other values unchanged."""
    if (isinstance(values, pd.Series)
            and pd.api.types.is_bool_dtype(values.dtype)):
        return values.astype(float)
    return values


def position_key(posData):
    """Key of a position in the configuration (normalized Images path)."""
    path = getattr(posData, 'images_path', None)
    if not path:
        return None
    return os.path.normcase(os.path.abspath(path))


def blank_unanalyzed_outputs(df, columns, last_frame):
    """Set the `columns` of `df` to NaN (in place) on the rows after
    `last_frame` (outputs of measurement steps that must stay empty on the
    frames ignored by the calculations, e.g. read back as 0 from a saved
    table). Return the columns that changed."""
    if last_frame is None or df is None:
        return []
    later = df.index.get_level_values('frame_i').to_numpy() > last_frame
    if not later.any():
        return []
    changed = []
    for column in columns:
        if column not in df.columns or df[column][later].isna().all():
            continue
        df[column] = float_if_bool(df[column]).where(~later)
        changed.append(column)
    return changed


def evaluate_masked_cell_equation(df, expression, SizeT=None, last_frame=None):
    """Evaluate a combined measurement `expression` per cell
    (`apps.CombineMetricsEquationDialog.evaluateCellEquation`) on the rows
    of `df` up to `last_frame` with ``SizeT = last_frame + 1``. The result
    is indexed like those rows (assigning it to a column of the full table
    leaves NaN on the later rows). Without `last_frame`, it is exactly
    `evaluateCellEquation` on `df` with `SizeT`."""
    evaluate = dialogs_combine_measurements.CombineMetricsEquationDialog.evaluateCellEquation
    if last_frame is None:
        return evaluate(df, expression, SizeT=SizeT)
    source = viewer_position_data.mask_frames(df, last_frame)
    if source.empty:
        return pd.Series(np.nan, index=df.index)
    values = evaluate(source, expression, SizeT=last_frame + 1)
    if not isinstance(values, pd.Series):
        values = pd.Series(values, index=source.index)
    if len(source) < len(df):
        # NaN on the later rows once assigned: keep bool results numeric
        values = float_if_bool(values)
    return values


def update_last_analyzed_frame(posData):
    """Set ``posData.lastAnalyzedFrame`` and ``posData.lastAnalyzedKind``
    from its table (`last_analyzed_frame`). Called whenever a table is read
    from file (`load_position`, `MeasurementsViewerWin.reloadAcdcDfFile`).
    """
    posData.lastAnalyzedFrame, posData.lastAnalyzedKind = last_analyzed_frame(
        getattr(posData, 'acdc_df', None), getattr(posData, 'SizeT', None)
    )


def acdc_df_snapshot(acdc_df):
    """Copy of a table as it is on disk, used to detect unsaved changes."""
    if acdc_df is None or acdc_df.empty:
        return None
    return MeasurementsViewerUtils._standardizeAcdcDf(acdc_df).copy()


def position_name(images_path):
    """Name of the Position folder of an Images folder (for progress texts)."""
    images_path = os.path.normpath(str(images_path))
    if os.path.basename(images_path) == 'Images':
        images_path = os.path.dirname(images_path)
    return os.path.basename(images_path) or str(images_path)


def loaded_data_status(
        posData, pos_i=0, num_positions=1, channel=None, overlay_channels=(),
        configuration_filepath=None
    ):
    """Describe the displayed data for the viewer's status bar label (as the
    main GUI's `gui.guiWin.setStatusBarLabel`).

    Parameters
    ----------
    posData : `load.loadData` or None
        Displayed position. Missing attributes (e.g., of synthetic data)
        leave out their item; None (no data) gives empty texts.
    pos_i : int, optional
        Index of the displayed position. Default is 0.
    num_positions : int, optional
        Number of loaded positions: with more than one, the position item
        ends with ``(pos_i+1/num_positions)``. Default is 1.
    channel : str, optional
        Displayed channel. Default is ``posData.user_ch_name``.
    overlay_channels : iterable of str, optional
        Overlaid channels, in the given order (item left out if empty).
    configuration_filepath : str, optional
        File of the loaded or saved configuration (item left out if None).

    Returns
    -------
    text : str
        Single line with the items joined by `` || ``: ``Experiment: <exp
        folder>``, ``<Position folder> (i/N)``, ``Basename: ...``,
        ``Channel: ...``, ``Overlay channels: ...``, ``Segmentation file
        name: <segm file name without basename>`` and ``Configuration:
        <file name>``.
    tooltip : str
        The same items one per line, with the full Images folder (instead
        of the experiment folder) and configuration file paths. Plain text.
        A path item longer than a tool tip line (`path_tooltip_lines`) has
        its label alone on the first line and the path broken after
        separators on indented lines below; other lines are not wrapped yet
        (`qutils.wrap_tooltip`, which keeps the path lines).
    """
    if posData is None:
        return '', ''
    images_path = getattr(posData, 'images_path', None)
    pos_foldername = getattr(posData, 'pos_foldername', None)
    if not pos_foldername and images_path:
        pos_foldername = position_name(images_path)
    exp_path = getattr(posData, 'exp_path', None)
    exp_foldername = (
        os.path.basename(os.path.normpath(str(exp_path))) if exp_path else ''
    )
    basename = getattr(posData, 'basename', None) or ''
    if channel is None:
        channel = getattr(posData, 'user_ch_name', None)
    segm_npz_path = getattr(posData, 'segm_npz_path', None)
    segm_filename = os.path.basename(segm_npz_path) if segm_npz_path else ''
    segm_endname = segm_filename
    if basename and segm_filename.startswith(basename):
        segm_endname = segm_filename[len(basename):]
    overlay_channels = [str(ch) for ch in overlay_channels]

    items = []
    if pos_foldername:
        position = str(pos_foldername)
        if num_positions > 1:
            position = f'{position} ({pos_i+1}/{num_positions})'
        items.append(position)
    if basename:
        items.append(f'Basename: {basename}')
    if channel:
        items.append(f'Channel: {channel}')
    if overlay_channels:
        items.append(f'Overlay channels: {", ".join(overlay_channels)}')
    if segm_endname:
        items.append(f'Segmentation file name: {segm_endname}')

    text_items = list(items)
    tooltip_lines = list(items)
    if exp_foldername:
        text_items.insert(0, f'Experiment: {exp_foldername}')
    if images_path:
        tooltip_lines[:0] = path_tooltip_lines('Images folder', images_path)
    if configuration_filepath:
        text_items.append(
            f'Configuration: {os.path.basename(configuration_filepath)}'
        )
        tooltip_lines.extend(
            path_tooltip_lines('Configuration file', configuration_filepath)
        )
    return ' || '.join(text_items), '\n'.join(tooltip_lines)


def path_tooltip_lines(label, path, width=qutils.TOOLTIP_MAX_LINE_LENGTH):
    """Tool tip lines of the item ``<label>: <path>``: one line if it fits
    in `width` characters, else ``<label>:`` and the path broken after its
    separators on lines indented by four spaces (`qutils.wrap_path`), because
    `qutils.wrap_tooltip` never splits a path."""
    line = f'{label}: {path}'
    if len(line) <= width:
        return [line]
    return [f'{label}:', *qutils.wrap_path(path, width, indent=' '*4)]


def load_positions(
        images_paths, channel, segm_endname='segm', logger=print,
        progress=None, skip_errors=True
    ):
    """Load the positions with `load_position`, skipping (and reporting with
    `logger`) those that cannot be loaded, or raising the error if
    `skip_errors` is False. `progress(text)` is called before each position.
    """
    data = []
    for p, images_path in enumerate(images_paths):
        if progress is not None:
            progress(
                f'Loading {position_name(images_path)} '
                f'({p+1}/{len(images_paths)})...'
            )
        try:
            data.append(load_position(images_path, channel, segm_endname))
        except Exception as err:
            if not skip_errors:
                raise
            logger(f'[WARNING]: Skipping "{images_path}". {err}')
    return data


def precompute_position_regionprops(posData, logger_func=print):
    """Compute the regionprops of every frame of `posData` (the slow part of
    `MeasurementsViewerUtils.initPosAnnotData`) and keep them in
    ``posData._viewerPrecomputedRps`` for `initPosAnnotData`, which uses
    them if the segmentation data is still the same object.

    Thread-safe (no Qt): the viewer runs it in its loading worker thread for
    the position displayed first. Positions already initialized by the
    viewer are skipped.
    """
    if getattr(posData, '_isViewerAnnotDataInit', False):
        return
    segm_data = posData.segm_data
    # `count_objects_and_init_rps` fills `allData_li` of the object it gets:
    # use a holder, so that `posData` is changed only once all is computed
    holder = SimpleNamespace(
        segm_data=segm_data, allData_li=[None]*posData.SizeT
    )
    allIDs, _ = core.count_objects_and_init_rps(holder, logger_func)
    posData._viewerPrecomputedRps = (segm_data, holder.allData_li, set(allIDs))


def preload_overlay_channels(data, channels, progress=None):
    """Read the image (and background data) files of the overlay `channels`
    of every position in `data` into ``posData._viewerPreloadedFluo``
    (``{file path: (image data, background data)}``), where the viewer's
    `load_fluo_data` takes them instead of reading the files on the GUI
    thread. Thread-safe (no Qt). Channels without a file are skipped (the
    GUI thread reports them when loading the overlay)."""
    for posData in data:
        preloaded = {}
        for channel in channels:
            fluo_path = load.get_filename_from_channel(
                posData.images_path, channel
            )
            if not fluo_path or fluo_path in preloaded:
                continue
            if progress is not None:
                progress(
                    f'Loading overlay channel "{channel}" of '
                    f'{position_name(posData.images_path)}...'
                )
            preloaded[fluo_path] = load.load_image_and_bkgr_data(fluo_path)
        posData._viewerPreloadedFluo = preloaded


def load_display_positions(
        images_paths, channel, segm_endname='segm', skip_errors=True,
        progress=None, overlay_channels=()
    ):
    """Load the positions (`load_positions`), read their `overlay_channels`
    (`preload_overlay_channels`) and precompute the regionprops of the first
    one, which the viewer displays first (`precompute_position_regionprops`).
    Run in a worker thread by the viewer
    (`MeasurementsViewerWin.runInThread`)."""
    data = load_positions(
        images_paths, channel, segm_endname=segm_endname,
        progress=progress, skip_errors=skip_errors
    )
    if data and overlay_channels:
        preload_overlay_channels(data, overlay_channels, progress=progress)
    if data:
        posData = data[0]
        if progress is not None:
            progress(
                f'Computing the objects of {position_name(posData.images_path)}...'
                if hasattr(posData, 'images_path') else
                'Computing the objects of the first position...'
            )
        try:
            precompute_position_regionprops(
                posData, logger_func=progress or print
            )
        except Exception as err:
            # `initPosAnnotData` computes them again on the GUI thread
            # (where errors are reported as without precomputing)
            pass
    return data


class MeasurementsViewerUtils(
        PositionDataUtils,
        AnnotationDataUtils,
    ):
    """Data and logic methods of the measurements viewer window (see the
    module docstring)."""

    @staticmethod
    def validateTimeCourseYAxes(yAxes):
        """Validate the configuration's ``time_course_y_axes``
        ``{column: axis number}`` (optional: configurations saved before
        multiple y axes have none)."""
        if not isinstance(yAxes, dict) or any(
                not isinstance(col, str) or not col
                or type(number) is not int or number < 1
                for col, number in yAxes.items()):
            raise ValueError(
                'time_course_y_axes must map column names to y axis numbers '
                '(1, 2, ...).'
            )

    def yAxisNumber(self, col):
        """Time course y axis number of column `col` (1: default axis)."""
        return self.timeCourseYAxes.get(col, 1)

    def setYAxisNumber(self, col, number):
        if number == 1:
            self.timeCourseYAxes.pop(col, None)
        else:
            self.timeCourseYAxes[col] = number

    def mergeYAxisIntoDefault(self, number):
        """Put every column of the y axis `number` (also those that are not
        plotted) on the default axis 1."""
        for col, other in list(self.timeCourseYAxes.items()):
            if other == number:
                del self.timeCourseYAxes[col]

    def newYAxisNumber(self):
        """Number of a new y axis: after every number used by a column,
        also by columns that are not plotted (plotting them again must not
        put them on the new axis), so the new axis is the rightmost one."""
        return max([1, *self.timeCourseYAxes.values()]) + 1

    def timeCourseYAxesConfiguration(self, columns):
        """The configuration's ``time_course_y_axes``: the y axis number of
        every plotted time course column in `columns` (also 1), then of the
        other columns not on axis 1."""
        yAxes = {col: self.yAxisNumber(col) for col in columns}
        for col, number in sorted(self.timeCourseYAxes.items()):
            yAxes.setdefault(col, number)
        return yAxes

    def loadTimeCourseYAxes(self, yAxes):
        """Set `timeCourseYAxes` from a configuration's validated
        ``time_course_y_axes`` (axis 1 is the default, not stored)."""
        self.timeCourseYAxes = {
            col: number for col, number in yAxes.items() if number != 1
        }

    @staticmethod
    def validateConfiguration(configuration):
        if (not isinstance(configuration, dict)
                or configuration.get('version') not in (1, 2)):
            raise ValueError('Unsupported measurements viewer configuration.')
        MeasurementsViewerUtils.validateConfigurationInput(configuration)
        inputs = configuration.get('input_options', [])
        if not isinstance(inputs, list):
            raise ValueError('Invalid input options in configuration.')
        for option in inputs:
            MeasurementsViewerUtils.validateConfigurationInput(option)
        MeasurementsViewerUtils.validateMeasurementSteps(configuration)
        columns = configuration.get('time_course_columns')
        if not isinstance(columns, list) or any(
                not isinstance(item, str) or not item for item in columns):
            raise ValueError('Invalid configuration field: time_course_columns.')
        axes = configuration.get('measurements_axes')
        if (not isinstance(axes, dict) or set(axes) != {'X', 'Y'}
                or any(not isinstance(value, str) for value in axes.values())
                or (axes['X'] and axes['X'] == axes['Y'])):
            raise ValueError('Invalid comparison axes in configuration.')
        if (type(configuration.get('measurements_mode')) is not int
                or configuration['measurements_mode'] not in (0, 1)
                or type(configuration.get('normalize_measurements')) is not bool):
            raise ValueError('Invalid plotting options in configuration.')
        # Optional for configurations saved before custom colors were saved
        colors = configuration.get('measurements_colors', {})
        if not isinstance(colors, dict) or any(
                not isinstance(col, str) or not col
                or not isinstance(color, str)
                or not phase_translator.HEX_COLOR.fullmatch(color)
                for col, color in colors.items()):
            raise ValueError(
                'measurements_colors must map column names to colors like '
                '"#00ff00".'
            )
        MeasurementsViewerUtils.validateTimeCourseYAxes(
            configuration.get('time_course_y_axes', {})
        )
        MeasurementsViewerUtils.validateConditionalFormatting(
            configuration.get('conditional_formatting', {})
        )
        # Optional for configurations saved before time point identifiers
        identifiers = configuration.get('timepoint_identifiers', [])
        if not isinstance(identifiers, list):
            raise ValueError('Invalid time point identifiers in configuration.')
        for identifier in identifiers:
            timepoint_predictors.TimepointIdentifier.validate_dict(identifier)
        names = [identifier['name'] for identifier in identifiers]
        if len(set(names)) != len(names):
            raise ValueError('Duplicated time point identifier names.')
        viewed = configuration.get('viewed_cells', {})
        if not isinstance(viewed, dict) or any(
                not isinstance(path, str) or not path
                or not isinstance(IDs, list)
                or any(type(ID) is not int or ID <= 0 for ID in IDs)
                for path, IDs in viewed.items()):
            raise ValueError('Invalid viewed cells in configuration.')
        MeasurementsViewerUtils.validateCustomAnnotConfiguration(
            configuration.get('custom_annotations', [])
        )
        # Annotations overriding phases are inputs with `phase_<name>`
        # outputs: included in the rules' input/output clash check
        phase_translator.validate_rules(
            configuration.get('phase_translator_rules', []),
            annotation_overrides=[
                state['name']
                for state in configuration.get('custom_annotations', [])
                if state.get('overridesPhases', False)
            ]
        )
        phase_translator.validate_phase_colors(configuration.get('phase_colors', {}))
        if not isinstance(configuration.get('auto_apply_phases', False), bool):
            raise ValueError('auto_apply_phases must be true or false.')
        # Optional for configurations saved before the option existed
        # (loading them asks whether to ignore the unanalyzed frames)
        if not isinstance(configuration.get('mask_unanalyzed_frames', True), bool):
            raise ValueError('mask_unanalyzed_frames must be true or false.')
        stepsLastFrames = configuration.get('measurement_steps_last_frames', {})
        if not isinstance(stepsLastFrames, dict) or any(
                not isinstance(key, str) or not key
                or type(frame) is not int or frame < 0
                for key, frame in stepsLastFrames.items()):
            raise ValueError(
                'measurement_steps_last_frames must map Images folders to '
                'frame indices.'
            )
        predictors = {
            f'{TIMEPOINT_PREFIX}{identifier["name"]}': identifier['predictor']
            for identifier in identifiers
        }
        for state in configuration.get('custom_annotations', []):
            column = state['name']
            if column in predictors:
                raise ValueError(f'The predictor output "{column}" already exists.')
            predictors[column] = state.get('predictor') or {
                'type': timepoint_predictors.PREDICTOR_NONE
            }
        timepoint_predictors.PredictorDependencies.order(predictors)

    @staticmethod
    def validateConditionalFormatting(formatting):
        """Validate `{column: [{'condition': str, 'color': '#rrggbb'}]}`."""
        if not isinstance(formatting, dict):
            raise ValueError('Invalid conditional formatting in configuration.')
        for column, rules in formatting.items():
            if not isinstance(column, str) or not column:
                raise ValueError('Invalid conditional formatting column.')
            if not isinstance(rules, list):
                raise ValueError(
                    f'Invalid conditional formatting rules of "{column}".'
                )
            for rule in rules:
                if (not isinstance(rule, dict)
                        or set(rule) != {'condition', 'color'}
                        or not isinstance(rule['condition'], str)
                        or not rule['condition'].strip()
                        or not is_formatting_color(rule['color'])):
                    raise ValueError(
                        f'Invalid conditional formatting rule of "{column}": '
                        f'{rule!r}. Expected a non-empty condition and a '
                        'color like "#ff0000", "lighter" or "darker:250".'
                    )

    @staticmethod
    def evaluateConditionalFormatting(
            df_ID, rules, ID=None, SizeT=None, last_frame=None
        ):
        """Index of the first true rule at each row of one cell (-1: none).

        `df_ID` holds the rows of one cell in frame order (index `frame_i`).
        Each condition is evaluated like a predictor equation and must
        return bool or 0/1 (per row, or a scalar for the whole cell).
        Missing values count as False. Errors are raised. With `last_frame`
        (see `analysis_last_frame`), the conditions see only the rows up to
        that frame (with ``SizeT = last_frame + 1``) and the later rows get
        -1.
        """
        matched = np.full(len(df_ID), -1, dtype=int)
        if not rules or not len(df_ID):
            return matched
        cell = timepoint_predictors.CellTable.flatten(df_ID)
        if 'Cell_ID' not in cell.columns and ID is not None:
            cell['Cell_ID'] = ID
        rows = np.ones(len(cell), dtype=bool)
        if last_frame is not None and 'frame_i' in cell.columns:
            rows = cell['frame_i'].to_numpy() <= last_frame
            cell = cell[rows]
            SizeT = last_frame + 1
            if cell.empty:
                return matched
        cellMatched = np.full(len(cell), -1, dtype=int)
        variables = {'SizeT': SizeT} if SizeT is not None else None
        for i, rule in enumerate(rules):
            values = timepoint_predictors.CellEquation.evaluate(
                cell, rule['condition'], variables
            )
            mask = _FormattingConditionDialog.conditionMask(values, len(cell))
            cellMatched[(cellMatched < 0) & mask] = i
        matched[rows] = cellMatched
        return matched

    @staticmethod
    def validateConfigurationInput(configuration):
        if not isinstance(configuration, dict):
            raise ValueError('Invalid configuration input.')
        for key in (
                'images_paths', 'overlay_channels', 'visible_channels'):
            value = configuration.get(key)
            if not isinstance(value, list) or any(
                    not isinstance(item, str) or not item for item in value):
                raise ValueError(f'Invalid configuration field: {key}.')
        if not configuration['images_paths']:
            raise ValueError('The configuration contains no Images folders.')
        if not isinstance(configuration.get('channel'), str):
            raise ValueError('Invalid configuration field: channel.')
        segm = configuration.get('segm_endname')
        if not isinstance(segm, str) or not segm:
            raise ValueError('Invalid configuration field: segm_endname.')
        # An empty channel means "ask which channel to display when loading"
        available = {
            configuration['channel'], *configuration['overlay_channels']
        } - {''}
        if (not configuration['visible_channels']
                or not set(configuration['visible_channels']) <= available):
            raise ValueError('Invalid visible image channels in configuration.')

    @staticmethod
    def validateMeasurementSteps(configuration):
        orders = set()
        for section in ('measurement_processing', 'combined_measurements'):
            steps = configuration.get(section, [])
            if not isinstance(steps, list):
                raise ValueError(f'Invalid {section} in configuration.')
            for step in steps:
                if (not isinstance(step, dict)
                        or type(step.get('order')) is not int
                        or step['order'] < 0 or step['order'] in orders):
                    raise ValueError('Invalid or duplicate measurement step order.')
                orders.add(step['order'])
                if section == 'combined_measurements':
                    equations = step.get('equations')
                    if not isinstance(equations, dict) or not equations or any(
                            not isinstance(key, str) or not key
                            or not isinstance(value, str) or not value.strip()
                            for key, value in equations.items()):
                        raise ValueError('Invalid combined measurement equations.')
                    continue
                for key in ('columns', 'output_columns'):
                    values = step.get(key)
                    if (not isinstance(values, list) or not values
                            or any(not isinstance(value, str) or not value for value in values)
                            or len(values) != len(set(values))):
                        raise ValueError(f'Invalid processing {key}.')
                if len(step['columns']) != len(step['output_columns']):
                    raise ValueError('Processing input/output counts differ.')
                recipe = step.get('recipe')
                if not isinstance(recipe, list) or not recipe:
                    raise ValueError('Invalid measurement recipe.')
                for item in recipe:
                    if (not isinstance(item, dict)
                            or not isinstance(item.get('method'), str)
                            or item.get('method') not in measurement_processing.MEASUREMENT_PROCESSING_MAPPER
                            or not isinstance(item.get('kwargs'), dict)):
                        raise ValueError('Invalid measurement recipe step.')
        # Each output column is produced by exactly one step, otherwise
        # replaying the steps would overwrite a previous step's output
        outputs = [
            column
            for section in ('measurement_processing', 'combined_measurements')
            for step in configuration.get(section, [])
            for column in MeasurementsViewerUtils.measurementStepOutputColumns(step)
        ]
        duplicates = sorted({col for col in outputs if outputs.count(col) > 1})
        if duplicates:
            raise ValueError(
                'Duplicate measurement output name across steps: '
                + ', '.join(duplicates)
            )
        # Steps are replayed in their order after dropping all outputs, so
        # a step must not read the output of a step with a higher order
        steps = sorted(
            (
                step
                for section in ('measurement_processing', 'combined_measurements')
                for step in configuration.get(section, [])
            ),
            key=lambda step: step['order']
        )
        later = set()
        for step in reversed(steps):
            outputs = set(MeasurementsViewerUtils.measurementStepOutputColumns(step))
            reads = (
                set(MeasurementsViewerUtils.measurementStepInputColumns(step))
                - outputs
            ) & later
            if reads:
                raise ValueError(
                    'A measurement step reads outputs of a later step: '
                    + ', '.join(sorted(reads))
                )
            later.update(outputs)

    @staticmethod
    def measurementStepOutputColumns(step):
        """Output columns written by one measurement step (the processing
        `output_columns` or the names of the combined `equations`)."""
        if 'equations' in step:
            return list(step['equations'])
        return list(step['output_columns'])

    @staticmethod
    def measurementStepInputColumns(step):
        """Columns read by one measurement step (the processing `columns`
        or the inputs of the combined `equations`, as in
        `columnDependents`)."""
        if 'equations' not in step:
            return list(step['columns'])
        inputs = {}
        for expression in step['equations'].values():
            inputs.update(timepoint_predictors.PredictorDependencies.inputs({
                'type': timepoint_predictors.PREDICTOR_EQUATION,
                'equation': expression
            }))
        return list(inputs)

    @staticmethod
    def writeConfigurationFile(filepath, configuration):
        payload = json.dumps(configuration, indent=2).encode('utf-8')
        output = QSaveFile(filepath)
        if not output.open(QIODevice.WriteOnly):
            raise OSError(output.errorString())
        if output.write(payload) != len(payload):
            output.cancelWriting()
            raise OSError(output.errorString())
        if not output.commit():
            raise OSError(output.errorString())

    @staticmethod
    def validateChannelPlaceholders(placeholders):
        if not isinstance(placeholders, dict) or not placeholders:
            raise ValueError('Invalid channel_placeholders in configuration.')
        for name, description in placeholders.items():
            if (not isinstance(name, str)
                    or not re.fullmatch(r'[A-Za-z0-9_]+', name)
                    or not isinstance(description, str)):
                raise ValueError(f'Invalid channel placeholder "{name}".')

    @staticmethod
    def validateInputPlaceholders(inputs, channelNames=()):
        """Validate a template's `input_placeholders`:
        `{NAME: {"type": "float"|"int"|"str", "default": value,
        "description": str, "info": str}}` (description and info optional).
        Names follow the channel placeholder rules and must differ from them.
        """
        if not isinstance(inputs, dict):
            raise ValueError('Invalid input_placeholders in configuration.')
        for name, spec in inputs.items():
            if (not isinstance(name, str)
                    or not re.fullmatch(r'[A-Za-z0-9_]+', name)
                    or name in channelNames):
                raise ValueError(f'Invalid input placeholder "{name}".')
            if (not isinstance(spec, dict)
                    or not {'type', 'default'} <= set(spec)
                    or not set(spec) <= {'type', 'default', 'description', 'info'}
                    or any(not isinstance(spec.get(key, ''), str)
                           for key in ('description', 'info'))):
                raise ValueError(f'Invalid input placeholder "{name}".')
            kind, default = spec['type'], spec['default']
            valid = {
                'float': lambda v: (
                    isinstance(v, (int, float)) and not isinstance(v, bool)
                ),
                'int': lambda v: isinstance(v, int) and not isinstance(v, bool),
                'str': lambda v: isinstance(v, str),
            }
            if kind not in valid or not valid[kind](default):
                raise ValueError(
                    f'Invalid type or default of input placeholder "{name}".'
                )

    @staticmethod
    def validateChannelHints(hints, channelNames):
        """Validate a template's optional `channel_hints`:
        `{CHANNEL_PLACEHOLDER: [non-empty name part, ...]}`."""
        if hints is None:
            return {}
        if not isinstance(hints, dict):
            raise ValueError('Invalid channel_hints in configuration.')
        for name, parts in hints.items():
            if (name not in channelNames or not isinstance(parts, list)
                    or not all(isinstance(part, str) and part.strip()
                               for part in parts)):
                raise ValueError(f'Invalid channel_hints of "{name}".')
        return hints

    @staticmethod
    def validateBackgroundValuesSpec(spec, channelNames, inputs):
        """Validate a template's optional `background_values` section and
        return it with defaults:
        `{"statistic": "mean"|"median" (default "median"),
        "low_percentile": number (5), "high_percentile": number (95),
        "exclude_zero_pixels": bool (true),
        "inputs": {INPUT: {"channel": CHANNEL_PLACEHOLDER,
        "value": one of background_values.SUMMARY_VALUE_COLUMNS}}}`.
        Linked inputs must be float `input_placeholders`. Returns None
        when `spec` is None.
        """
        if spec is None:
            return None
        allowed = {
            'statistic', 'low_percentile', 'high_percentile',
            'exclude_zero_pixels', 'inputs'
        }
        if (not isinstance(spec, dict) or 'inputs' not in spec
                or not set(spec) <= allowed):
            raise ValueError('Invalid background_values in configuration.')
        result = {
            'statistic': 'median', 'low_percentile': 5.0,
            'high_percentile': 95.0, 'exclude_zero_pixels': True,
        }
        result.update({key: spec[key] for key in spec if key != 'inputs'})
        if result['statistic'] not in background_values.STATISTICS:
            raise ValueError(
                'Invalid statistic in background_values: '
                f'"{result["statistic"]}".'
            )
        if not isinstance(result['exclude_zero_pixels'], bool):
            raise ValueError(
                'Invalid exclude_zero_pixels in background_values.'
            )
        low, high = result['low_percentile'], result['high_percentile']
        if (any(isinstance(v, bool) or not isinstance(v, (int, float))
                for v in (low, high))
                or not 0 <= low < high <= 100):
            raise ValueError(
                'Invalid percentiles in background_values (0 <= '
                'low_percentile < high_percentile <= 100).'
            )
        result['low_percentile'] = float(low)
        result['high_percentile'] = float(high)
        links = spec['inputs']
        if not isinstance(links, dict) or not links:
            raise ValueError('Invalid background_values inputs.')
        for name, link in links.items():
            if name not in inputs or inputs[name].get('type') != 'float':
                raise ValueError(
                    f'background_values input "{name}" must be a float '
                    'input placeholder.'
                )
            if (not isinstance(link, dict)
                    or set(link) != {'channel', 'value'}
                    or link['channel'] not in channelNames
                    or link['value']
                    not in background_values.SUMMARY_VALUE_COLUMNS):
                raise ValueError(
                    f'Invalid background_values link of input "{name}".'
                )
        result['inputs'] = {name: dict(link) for name, link in links.items()}
        return result

    @classmethod
    def resolvePremadeTemplate(
            cls, template, mapping, values=None, meanSuffixes=None
        ):
        """Return the configuration of a premade `template` (not modified)
        with channel tokens replaced by `mapping` and input tokens by
        `values` (defaults for missing inputs). `meanSuffixes`
        (``{NAME: suffix}``, see `premadeMeanSuffixes`) replaces the mean
        intensity columns ``{NAME}_mean`` first (`substituteMeanColumns`).
        Raises ValueError if a token remains unresolved or a backtick-quoted
        name built from a token cannot be parsed in equations (see
        `isEquationName`)."""
        placeholders = template.get('channel_placeholders', {})
        inputs = template.get('input_placeholders', {})
        cls.validateInputPlaceholders(inputs, placeholders)
        replacements = {name: spec['default'] for name, spec in inputs.items()}
        replacements.update(values or {})
        replacements.update(mapping)
        result = {
            key: value for key, value in template.items()
            if key not in PREMADE_TEMPLATE_KEYS
        }
        if meanSuffixes:
            result = cls.substituteMeanColumns(result, meanSuffixes)
        quotedNames = set()
        result = cls.substituteChannelPlaceholders(
            result, replacements, quotedNames=quotedNames
        )
        invalid = sorted(
            name for name in quotedNames if not cls.isEquationName(name)
        )
        if invalid:
            raise ValueError(
                'These channels cannot be used in the equations of this '
                'configuration: a channel name must start with a letter or '
                'an underscore, also after a dash (e.g. "GFP" or '
                '"mKO2-hCdt1", not "561" or "GFP-488"), and must not '
                'contain quotes, brackets, "#" or backslashes. '
                'Rename the channel. '
                f'Invalid column names: {", ".join(invalid)}.'
            )
        unresolved = cls.unresolvedPlaceholders(
            result, [*placeholders, *inputs]
        )
        if unresolved:
            raise ValueError(
                f'Unresolved placeholders: {", ".join(unresolved)}.'
            )
        return result

    @staticmethod
    def isEquationName(name):
        """True if `name` can be used quoted in backticks in an equation
        (e.g. `mKO2-hCdt1_mean`).

        Rejected explicitly, the same on every Python version: empty names,
        names containing a backtick, and names with a digit as first
        character or right after a non-word character such as a dash
        (`561_mean`, `GFP-488_mean`). Python >= 3.12 cannot tokenize these
        inside backticks, while older tokenizers silently split them, so
        they are refused everywhere. Also rejected: quotes, brackets, `#`
        and backslashes (``' " ( ) [ ] { } # \\``). Python 3.10/3.11 parse
        a quoted name once but fail when it appears twice in an equation
        (the quotes pair up) and reject closing brackets, which 3.12+
        accept. Names passing these rules must also be parseable by pandas
        `eval` (extra guard)."""
        if not name or '`' in name:
            return False
        if re.search(r'(^|[^\w])\d', name):
            return False
        if re.search(r'[\'"#\\()\[\]{}]', name):
            return False
        try:
            pd.DataFrame({name: [0]}).eval(f'`{name}`')
        except Exception:
            return False
        return True

    @staticmethod
    def substituteChannelPlaceholders(obj, mapping, quotedNames=None):
        """Return a copy of `obj` with every `{NAME}` token replaced by
        `mapping[NAME]` in strings, dict keys and dict values (recursive).

        A string that is exactly one token becomes the mapped value itself
        (e.g. a number for input placeholders); tokens within longer strings
        and dict keys are replaced by the value's text. A backtick-quoted
        name containing tokens (`` `{RED}_mean` `` in an equation) loses its
        backticks if the result is a plain identifier (`mCherry_mean`) and
        keeps them otherwise (`` `mKO2-hCdt1_mean` ``); such quoted results
        are added to the `quotedNames` set if given."""
        def text(value):
            return value if isinstance(value, str) else repr(value)

        def replaceTokens(value):
            for name, replacement in mapping.items():
                value = value.replace(f'{{{name}}}', text(replacement))
            return value

        def unquote(match):
            inner = match.group(1)
            if not re.search(r'\{[A-Za-z0-9_]+\}', inner):
                return match.group(0)
            inner = replaceTokens(inner)
            if inner.isidentifier() and not keyword.iskeyword(inner):
                return inner
            if quotedNames is not None:
                quotedNames.add(inner)
            return f'`{inner}`'

        def substitute(value, isKey=False):
            if isinstance(value, str):
                match = re.fullmatch(r'\{([A-Za-z0-9_]+)\}', value)
                if match and match.group(1) in mapping and not isKey:
                    return copy.deepcopy(mapping[match.group(1)])
                value = re.sub(r'`([^`]*)`', unquote, value)
                return replaceTokens(value)
            if isinstance(value, dict):
                return {
                    substitute(key, isKey=True): substitute(item)
                    for key, item in value.items()
                }
            if isinstance(value, (list, tuple)):
                return [substitute(item) for item in value]
            return copy.deepcopy(value)
        return substitute(obj)

    @staticmethod
    def substituteMeanColumns(obj, meanSuffixes):
        """Return a copy of a premade template `obj` with the mean intensity
        column ``{NAME}_mean`` of each placeholder in `meanSuffixes`
        (``{NAME: suffix}``) replaced by ``{NAME}<suffix>`` (e.g.
        ``{RED}_mean_meanProj``) in strings and dict keys (recursive). Only
        whole column names are replaced (`PREMADE_MEAN_COLUMN_RE`)."""
        def replace(match):
            name = match.group(1)
            if name not in meanSuffixes:
                return match.group(0)
            return f'{{{name}}}{meanSuffixes[name]}'

        def substitute(value):
            if isinstance(value, str):
                return PREMADE_MEAN_COLUMN_RE.sub(replace, value)
            if isinstance(value, dict):
                return {
                    substitute(key): substitute(item)
                    for key, item in value.items()
                }
            if isinstance(value, (list, tuple)):
                return [substitute(item) for item in value]
            return copy.deepcopy(value)
        return substitute(obj)

    @staticmethod
    def premadeMeanSuffixes(template, mapping, images_paths, segm_endname):
        """Mean intensity columns of the positions' tables that replace the
        ``{NAME}_mean`` columns of a premade `template`, as
        ``{NAME: suffix}`` (only placeholders whose ``<channel>_mean`` is
        missing; `mapping` gives the channels).

        z-stack tables have no ``<channel>_mean``: as the background utility
        (`background_values.mean_source_columns`), ``<channel>_mean_3D``
        (3D segmentation) or else ``<channel>_mean_meanProj`` is used, the
        first one that every table (`acdc_output_columns` of `images_paths`
        with `segm_endname`) has. Positions without a table are skipped.
        Raises ValueError if the tables have none of these columns."""
        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for key, item in value.items():
                    yield from strings(key)
                    yield from strings(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from strings(item)

        body = {
            key: value for key, value in template.items()
            if key not in PREMADE_TEMPLATE_KEYS
        }
        names = sorted({
            match.group(1)
            for text in strings(body)
            for match in PREMADE_MEAN_COLUMN_RE.finditer(text)
            if match.group(1) in mapping
        })
        if not names:
            return {}
        tables = [
            columns for columns in (
                acdc_output_columns(path, segm_endname)
                for path in images_paths
            )
            if columns is not None
        ]
        if not tables:
            return {}
        suffixes, missing = {}, []
        for name in names:
            channel = mapping[name]
            candidates = list(dict.fromkeys([
                f'{channel}_mean',
                *background_values.mean_source_columns(channel, True),
                *background_values.mean_source_columns(channel),
            ]))
            column = next((
                col for col in candidates
                if all(col in columns for columns in tables)
            ), None)
            if column is None:
                missing.append(' / '.join(f'"{col}"' for col in candidates))
            elif column != f'{channel}_mean':
                suffixes[name] = column[len(channel):]
        if missing:
            raise ValueError(
                'This premade configuration needs the mean intensity of '
                'its channels, but the acdc_output tables of the selected '
                'Positions have none of these columns: '
                f'{"; ".join(missing)}. Compute the measurements of these '
                'channels first (e.g., in the launcher: Utilities > '
                'Measurements > Compute measurements for one or more '
                'experiments...).'
            )
        return suffixes

    @staticmethod
    def unresolvedPlaceholders(obj, names):
        """Return the `{NAME}` tokens of `names` still present in `obj`."""
        text = json.dumps(obj)
        return sorted(name for name in names if f'{{{name}}}' in text)

    @staticmethod
    def setConfigurationBaseChannel(configuration, base):
        """Make `base` the displayed channel; the previous base becomes an
        overlay so that it stays available and visible."""
        previous = configuration.get('channel')
        overlays = [
            ch for ch in configuration.get('overlay_channels', [])
            if ch != base
        ]
        visible = list(configuration.get('visible_channels', []))
        if previous and previous != base and previous not in overlays:
            overlays.insert(0, previous)
            if previous not in visible:
                visible.append(previous)
        visible = [base, *(ch for ch in visible if ch != base)]
        configuration.update(
            channel=base, overlay_channels=overlays, visible_channels=visible
        )

    @staticmethod
    def inputOptionKey(option):
        """Comparison key of an input option that ignores channel order.

        Two options with the same Images folders, channel, segmentation and
        overlay/visible channel sets are equal, even if the channels are
        listed in a different order (e.g. configured order vs. the sorted
        order written by `configuration()`). See `inputIdentityKey` for the
        key that ignores the overlay/visible channels.
        """
        return (
            *MeasurementsViewerUtils.inputIdentityKey(option),
            tuple(sorted(option.get('overlay_channels', []))),
            tuple(sorted(option.get('visible_channels', []))),
        )

    @staticmethod
    def inputIdentityKey(option):
        """Key of the experiment an input option loads.

        Options with the same Images folders, channel and segmentation load
        the same data and only differ in the displayed layers (overlay and
        visible channels). `configuration()` updates such an option in place
        with the current layers instead of adding another one (the loaded
        option, `loadedInputKey`, if several options share the experiment).
        """
        return (
            tuple(
                os.path.normcase(os.path.abspath(path))
                for path in option.get('images_paths', [])
            ),
            option.get('channel'),
            option.get('segm_endname'),
        )

    @classmethod
    def configurationInputOptions(cls, configuration):
        keys = (
            'images_paths', 'channel', 'segm_endname',
            'overlay_channels', 'visible_channels'
        )
        options = copy.deepcopy(configuration.get('input_options', []))
        current = {key: copy.deepcopy(configuration[key]) for key in keys}
        currentKey = cls.inputOptionKey(current)
        if all(cls.inputOptionKey(option) != currentKey for option in options):
            options.append(current)
        return options

    def configurationSteps(self):
        """The recorded measurement steps in the `configuration()` format
        (for `measurementStepOutputs`)."""
        return {
            'measurement_processing': self.measurementProcessingSteps,
            'combined_measurements': self.combinedMeasurementSteps,
        }

    @staticmethod
    def measurementStepOutputs(configuration):
        steps = sorted(
            (step['order'], section, step)
            for section in ('measurement_processing', 'combined_measurements')
            for step in configuration.get(section, [])
        )
        return list(dict.fromkeys(
            column for _, section, step in steps
            for column in (
                step['output_columns'] if section == 'measurement_processing'
                else step['equations']
            )
        ))

    @staticmethod
    def partiallyAnalyzedPositions(data):
        """Positions of `data` whose table was analyzed (cell cycle
        annotated, else tracked) only up to a frame before their last."""
        return [
            posData for posData in data
            if viewer_position_data.analysis_last_frame(posData) is not None
        ]

    @staticmethod
    def lastAnalyzedFrameText(posData):
        """'Position_1: last cell cycle annotated frame n. 5 of 20'."""
        name = position_name(
            getattr(posData, 'images_path', None)
            or getattr(posData, 'pos_path', '')
        )
        kind = getattr(posData, 'lastAnalyzedKind', None) or 'analyzed'
        return (
            f'{name}: last {kind} frame n. {posData.lastAnalyzedFrame+1} '
            f'of {posData.SizeT}'
        )

    @staticmethod
    def replayMeasurementSteps(
            posData, configuration, overwrite=False, last_frame=None
        ):
        """Calculate the recorded measurement steps of `configuration` on
        the table of `posData` (in a worker thread, no Qt). With
        `last_frame`, the steps use only the rows up to that frame (see
        `analysis_last_frame`) and leave NaN on the later rows."""
        steps = [
            (step['order'], section, step)
            for section in ('measurement_processing', 'combined_measurements')
            for step in configuration.get(section, [])
        ]
        if not steps:
            return
        source = getattr(posData, 'acdc_df', None)
        if source is None or source.empty:
            raise ValueError('The experiment has no measurements to process.')
        result = MeasurementsViewerUtils._standardizeAcdcDf(source).copy()
        if overwrite:
            outputs = MeasurementsViewerUtils.measurementStepOutputs(configuration)
            result = result.drop(columns=[c for c in outputs if c in result])
        for identifier in configuration.get('timepoint_identifiers', []):
            column = f'{TIMEPOINT_PREFIX}{identifier["name"]}'
            if column not in result:
                result[column] = False
        for state in configuration.get('custom_annotations', []):
            if state['name'] not in result:
                result[state['name']] = 0
        for _, section, step in sorted(steps):
            outputs = (
                step['output_columns'] if section == 'measurement_processing'
                else list(step['equations'])
            )
            collisions = set(outputs).intersection(result.columns)
            if collisions:
                raise ValueError(
                    'Workflow would overwrite existing measurements: '
                    + ', '.join(sorted(collisions))
                )
            if section == 'measurement_processing':
                kwargs = {}
                if last_frame is not None:
                    kwargs['max_frame_i'] = last_frame
                result = measurement_processing.apply_recipe(
                    result, step['columns'], step['recipe'],
                    step['output_columns'], **kwargs
                )
            else:
                equationSource = result.copy()
                for name, expression in step['equations'].items():
                    result[name] = evaluate_masked_cell_equation(
                        equationSource, expression,
                        SizeT=getattr(posData, 'SizeT', None),
                        last_frame=last_frame
                    )
        posData.acdc_df = result

    def load_fluo_data(self, fluo_path, isGuiThread=True):
        """Return the overlay channel data that the loading worker already
        read (`preload_overlay_channels`), otherwise read the file (as in
        `gui_overlay`)."""
        for posData in self.data:
            preloaded = getattr(posData, '_viewerPreloadedFluo', None)
            if preloaded and fluo_path in preloaded:
                return preloaded.pop(fluo_path)
        return super().load_fluo_data(fluo_path, isGuiThread=isGuiThread)

    @staticmethod
    def recordStepsLastFrame(stepsLastFrames, posData, last, replace=False):
        """Record in `stepsLastFrames` (`measurementStepsLastFrames`) that
        the measurement step outputs of `posData` were calculated only up
        to frame `last` (None: all frames). With `replace` (all steps
        recalculated) the previous record is replaced, else the outputs
        cover up to the smaller of both frames."""
        key = position_key(posData)
        if key is None:
            return
        if not replace:
            last = min_frame(stepsLastFrames.get(key), last)
        if last is None:
            stepsLastFrames.pop(key, None)
        else:
            stepsLastFrames[key] = int(last)

    @staticmethod
    def blankStepOutputs(posData, outputs, last):
        """Set the step `outputs` of `posData` to NaN after frame `last`,
        also in its "as on disk" snapshot (``savedAcdcDf``): restoring NaN
        read back as 0 is not an edit of the table."""
        blank_unanalyzed_outputs(getattr(posData, 'acdc_df', None), outputs, last)
        blank_unanalyzed_outputs(
            getattr(posData, 'savedAcdcDf', None), outputs, last
        )

    def blankSavedStepOutputs(self, data, outputs, mask, stepsLastFrames):
        """Saved step `outputs` of the positions `data` read from file and
        used as existing values: set them to NaN (`blankStepOutputs`) after
        the smaller of the frame they were calculated up to
        (`stepsLastFrames`) and the current last analyzed frame (with
        `mask`), and record that frame (`recordStepsLastFrame`). Return
        these frames (None: all frames) in the order of `data`."""
        lastFrames = []
        for posData in data:
            last = min_frame(
                stepsLastFrames.get(position_key(posData)),
                viewer_position_data.analysis_last_frame(posData, mask)
            )
            self.blankStepOutputs(posData, outputs, last)
            self.recordStepsLastFrame(stepsLastFrames, posData, last)
            lastFrames.append(last)
        return lastFrames

    @staticmethod
    def _standardizeAcdcDf(acdc_df):
        if list(acdc_df.index.names) == ['frame_i', 'Cell_ID']:
            return acdc_df
        acdc_df = acdc_df.reset_index()
        acdc_df = acdc_df.drop(columns='index', errors='ignore')
        return acdc_df.set_index(['frame_i', 'Cell_ID'])

    def measurementProcessingTargetTable(self, posData):
        """Current table of the position processed in the background, or
        None when the results are stale (position no longer loaded or table
        reloaded from file, or frames used by the calculations changed, since
        processing started)."""
        target, reloadCount, inputs, outputs, lastFrame = (
            self._measurementProcessingTarget
        )
        if posData is not target or not any(pos is posData for pos in self.data):
            return
        if getattr(posData, 'acdcDfReloadCount', 0) != reloadCount:
            return
        if self.analysisLastFrame(posData) != lastFrame:
            return
        return self.timepointsDf(posData)

    def recordedOutputConflicts(self, outputs):
        """Output names in `outputs` already written by a recorded step that
        does not write exactly the same outputs (such a partial overlap
        cannot be recorded, since each output column must be produced by
        exactly one step, see `validateMeasurementSteps`)."""
        outputs = set(outputs)
        conflicts = set()
        for saved in self.measurementProcessingSteps + self.combinedMeasurementSteps:
            savedOutputs = set(self.measurementStepOutputColumns(saved))
            if savedOutputs != outputs:
                conflicts.update(savedOutputs & outputs)
        return sorted(conflicts)

    def measurementStepReplacementOrder(self, step):
        """Order of `step` when it replaces the recorded step writing the
        same outputs, and the columns that prevent recording it.

        The replacement keeps the replaced step's order unless it reads
        outputs of recorded steps with a higher order: then it moves after
        all steps (max order + 1) when no other recorded step reads its
        outputs, otherwise it cannot be recorded and the conflicting input
        columns are returned. Returns (None, []) when `step` replaces no
        recorded step."""
        outputs = set(self.measurementStepOutputColumns(step))
        recorded = self.measurementProcessingSteps + self.combinedMeasurementSteps
        replaced = next((
            saved for saved in recorded
            if set(self.measurementStepOutputColumns(saved)) == outputs
        ), None)
        if replaced is None:
            return None, []
        others = [saved for saved in recorded if saved is not replaced]
        later = {
            column for saved in others if saved['order'] > replaced['order']
            for column in self.measurementStepOutputColumns(saved)
        }
        reads = (set(self.measurementStepInputColumns(step)) - outputs) & later
        if not reads:
            return replaced['order'], []
        if any(outputs.intersection(self.measurementStepInputColumns(saved))
               for saved in others):
            return replaced['order'], sorted(reads)
        return max(saved['order'] for saved in recorded) + 1, []

    def recordMeasurementStep(self, steps, step):
        """Record an applied measurement step in the configuration.

        Returns a note for the user when the step replaced a recorded step
        writing the same outputs (keeping its order, or moving it after all
        steps when it reads outputs of later steps, see
        `measurementStepReplacementOrder`), otherwise ''.
        """
        # Applying the same operation at another position is not a new recipe.
        if any({key: value for key, value in saved.items() if key != 'order'} == step
               for saved in steps):
            return ''
        outputs = set(self.measurementStepOutputColumns(step))
        allSteps = (self.measurementProcessingSteps, self.combinedMeasurementSteps)
        for savedSteps in allSteps:
            for i, saved in enumerate(savedSteps):
                if set(self.measurementStepOutputColumns(saved)) != outputs:
                    continue
                order, orderConflicts = self.measurementStepReplacementOrder(
                    step
                )
                if orderConflicts:
                    # Prevented before applying (see above)
                    self.logger.warning(
                        'Measurement step not recorded in the configuration: '
                        'it reads outputs of later steps: %s',
                        ', '.join(orderConflicts)
                    )
                    return ''
                # Same outputs with different parameters: the configuration
                # records the new parameters (keeping the order, so that later
                # steps reading these columns still run after it, unless the
                # new parameters read outputs of later steps).
                del savedSteps[i]
                steps.append({'order': order, **copy.deepcopy(step)})
                steps.sort(key=lambda item: item['order'])
                note = (
                    'The configuration now records the new parameters for '
                    + ', '.join(sorted(outputs)) + ' (replacing the previous '
                    'step; positions processed before keep their values '
                    'until recalculated).'
                )
                self.logger.info(note)
                return note
        conflicts = self.recordedOutputConflicts(outputs)
        if conflicts:
            # Prevented before applying; recording would make the
            # configuration impossible to replay.
            self.logger.warning(
                'Measurement step not recorded in the configuration: outputs '
                'already written by other steps: %s', ', '.join(conflicts)
            )
            return ''
        orders = [
            saved['order']
            for saved in self.measurementProcessingSteps + self.combinedMeasurementSteps
        ]
        steps.append({'order': max(orders, default=-1) + 1, **copy.deepcopy(step)})
        return ''

    def customAnnotConfiguration(self):
        return [
            custom_annot_state_to_save(annot['state'])
            for annot in self.customAnnotDict.values()
        ]

    @staticmethod
    def validateCustomAnnotConfiguration(customAnnotations):
        if not isinstance(customAnnotations, list):
            raise ValueError('Invalid custom annotations in configuration.')
        names = []
        for state in customAnnotations:
            if not isinstance(state, dict):
                raise ValueError('Invalid custom annotation in configuration.')
            name = state.get('name')
            if not isinstance(name, str) or not name:
                raise ValueError('Invalid custom annotation name.')
            for key in ('symbol', 'shortcut'):
                if not isinstance(state.get(key, ''), str):
                    raise ValueError(
                        f'Invalid {key} of custom annotation "{name}".'
                    )
            color = state.get('symbolColor')
            if (not isinstance(color, (list, tuple)) 
                    or len(color) not in (3, 4)
                    or any(type(value) is not int for value in color)):
                raise ValueError(
                    f'Invalid symbol color of custom annotation "{name}".'
                )
            if type(state.get('autoApply', False)) is not bool:
                raise ValueError(
                    f'Invalid auto apply of custom annotation "{name}".'
                )
            # Optional (absent: False), viewer only
            if type(state.get('overridesPhases', False)) is not bool:
                raise ValueError(
                    f'Invalid "Overrides phases" of custom annotation '
                    f'"{name}".'
                )
            predictor = state.get('predictor')
            if predictor is not None:
                CustomAnnotationPredictor.validate(predictor)
            names.append(name)
        if len(set(names)) != len(names):
            raise ValueError('Duplicated custom annotation names.')

    def formattingTestCell(self):
        """Rows and ID of the selected cell (or the first cell) to test
        formatting conditions on, or (None, None) without data."""
        df_ID = self.getMeasurementsDf()
        if df_ID is not None:
            return df_ID, self.selectedID
        if self.acdc_df is None or self.acdc_df.empty:
            return None, None
        ID, df_ID = next(
            timepoint_predictors.CellTable.iter_cells(self.acdc_df)
        )
        return df_ID, ID

    def validateFormattingCondition(self, condition):
        df_ID, ID = self.formattingTestCell()
        if df_ID is None:
            return
        self.evaluateConditionalFormatting(
            df_ID, [{'condition': condition, 'color': '#000000'}], ID=ID,
            SizeT=getattr(self.posData(), 'SizeT', None),
            last_frame=self.analysisLastFrame()
        )


INDEX_COLUMNS = ('frame_i', 'Cell_ID', 'time_seconds')
