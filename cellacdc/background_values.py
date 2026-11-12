"""Background intensity of fluorescence channels and normalization constants.

Used by the utility "Compute background intensity outside of segmented
objects..." (`utils/computeBackgroundValues.py`, worker
`workers.ComputeBackgroundValuesWorker`).

Method
------
1. **Per-frame background.** For every frame and channel, the background is
   the mean (and median) intensity of the pixels outside of all segmented
   objects (segmentation label 0). Use a segmentation where (nearly) all
   cells are segmented, otherwise unsegmented cells raise the background.
   Pixels with intensity 0 (e.g. padding added by the alignment of the frames)
   can be excluded. Frames without any segmented object get NaN, because
   their background would include every cell.
2. **Cell intensities.** The mean intensities of the cells are read from a
   selected acdc_output table, which is not modified (`cell_mean_values`;
   excluded cells are ignored). For z-stacks, Cell-ACDC has no
   ``<channel>_mean`` column: ``<channel>_mean_3D`` is used with 3D masks,
   and ``<channel>_mean_meanProj`` with 2D masks (the background over all
   z-slices is the background of the mean projection).
3. **Normalization constants.** One experiment-wide background
   (``constant_background``, median of the per-frame backgrounds) and the
   low and high percentiles (default 5th and 95th) of
   ``<channel>_mean - constant_background`` pooled over every cell, frame and
   position (``constant_min`` / ``constant_max``). They are the
   ``background_value`` / ``min_value`` / ``max_value`` of the "Min-max
   normalization" measurement processing step on ``<channel>_mean`` (e.g. in
   the premade FUCCI configuration of the measurements viewer), so that the
   same constants are used for all positions and conditions.

   A per-frame correction (adding ``<channel>_mean_bkgr_corrected``
   columns to the acdc_output table, with ``per_frame_min/max``
   percentiles) is disabled for now: its code is commented out because
   nothing uses it.
4. **Summary table.** The constants of all channels are saved in each
   experiment folder as ``<name>_summary.csv`` (one row per channel, columns
   `SUMMARY_COLUMNS`). `read_summary_table` reads it back, e.g. to fill the
   inputs of a premade measurements viewer configuration
   (`apps_measurements_viewer.ChannelPlaceholderMappingDialog`).
"""
import numpy as np
import pandas as pd

STATISTICS = ('mean', 'median')

# Values of the summary table that can fill normalization inputs
SUMMARY_VALUE_COLUMNS = ('constant_background', 'constant_min', 'constant_max')
SUMMARY_COLUMNS = (
    'channel', 'background_statistic', 'low_percentile', 'high_percentile',
    'num_cell_values', 'num_background_values', *SUMMARY_VALUE_COLUMNS,
)


def background_colname(channel, statistic):
    """Column of the per-frame background table with a channel statistic."""
    return f'{channel}_bkgr_{statistic}'


def num_pixels_colname(channel):
    """Column with the number of background pixels used per frame."""
    return f'{channel}_bkgr_num_pixels'


# Disabled for now (per-frame correction of acdc_output, not used):
# def corrected_colname(source_col):
#     """acdc_output column with the background-corrected `source_col`."""
#     return f'{source_col}_bkgr_corrected'


def is_background_table(csv_path):
    """True if `csv_path` is a table saved by `compute_background_values`."""
    try:
        columns = pd.read_csv(csv_path, nrows=0).columns
    except Exception:
        return False
    return list(columns[:2]) == ['frame_i', 'num_objects']


def read_summary_table(csv_path):
    """Read a summary table saved by the utility (``<name>_summary.csv``).

    Channel names are kept as text (CSV reading would turn e.g. ``561`` into
    a number and ``NA`` into NaN). Returns a DataFrame with one row per
    channel and numeric `SUMMARY_COLUMNS` (empty values are NaN). Raises
    ValueError with a message for the user when the file is not a summary
    table (e.g. a per-Position background table) or has no or repeated
    channels.
    """
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception as err:
        raise ValueError(f'The file cannot be read as a CSV table ({err}).')
    if list(df.columns[:2]) == ['frame_i', 'num_objects']:
        raise ValueError(
            'This is the background table of a single Position. Load the '
            'summary table saved in the experiment folder instead '
            '(<code>&lt;name&gt;_summary.csv</code>).'
        )
    missing = [col for col in SUMMARY_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            'This is not a summary table of the "Compute background '
            'intensity" utility. Missing columns: '
            f'{", ".join(missing)}.'
        )
    if df.empty:
        raise ValueError('The summary table has no channels.')
    duplicated = sorted(set(df['channel'][df['channel'].duplicated()]))
    if duplicated:
        raise ValueError(
            'The summary table has more than one row for the channels: '
            f'{", ".join(duplicated)}.'
        )
    numeric = [
        col for col in SUMMARY_COLUMNS
        if col not in ('channel', 'background_statistic')
    ]
    for col in numeric:
        df[col] = pd.to_numeric(
            df[col].str.strip().replace('', np.nan), errors='coerce'
        )
    return df.reset_index(drop=True)


def mean_source_columns(channel, is_segm_3D=False):
    """acdc_output mean-intensity columns that the background can correct.

    Candidates in order of preference: ``<channel>_mean_3D`` for 3D masks;
    ``<channel>_mean`` (2D data) and ``<channel>_mean_meanProj`` (z-stack
    with 2D masks) otherwise.
    """
    if is_segm_3D:
        return [f'{channel}_mean_3D']
    return [f'{channel}_mean', f'{channel}_mean_meanProj']


def as_frames(arr, size_t):
    """Return data with a leading time axis: ``(T, [Z,] Y, X)``.

    Singleton axes are removed, except the time axis of time-lapse data
    (``size_t > 1``), so that a segmentation saved with a single frame keeps
    its time axis. Non-timelapse data (``size_t == 1``) gets a time axis of
    length 1.

    Parameters
    ----------
    arr : numpy.ndarray
        Image or segmentation data of one Position.
    size_t : int
        Number of frames from the metadata.

    Returns
    -------
    numpy.ndarray
        Data with shape ``(T, [Z,] Y, X)``.
    """
    if size_t == 1:
        return np.squeeze(arr)[np.newaxis]
    if arr.ndim < 3:
        # Single frame of a time-lapse (time axis already squeezed)
        return arr[np.newaxis]
    shape = [arr.shape[0], *[dim for dim in arr.shape[1:] if dim != 1]]
    return arr.reshape(shape)


def _check_shapes(img, segm, channel):
    if img.ndim not in (3, 4):
        raise ValueError(
            f'Image data of channel "{channel}" must have shape '
            f'(T, Y, X) or (T, Z, Y, X), got {img.shape}.'
        )
    if segm.ndim not in (3, 4):
        raise ValueError(
            f'Segmentation data must have shape (T, Y, X) or (T, Z, Y, X), '
            f'got {segm.shape}.'
        )
    if segm.ndim == 4 and img.ndim == 3:
        raise ValueError(
            f'3D segmentation {segm.shape} cannot be used with the 2D images '
            f'of channel "{channel}" {img.shape}.'
        )
    if segm.shape[1:] != img.shape[1 + img.ndim - segm.ndim:]:
        raise ValueError(
            f'Shape mismatch between segmentation {segm.shape} and image data '
            f'of channel "{channel}" {img.shape}.'
        )


def compute_background_values(
        images, segm, exclude_zero_pixels=True, progress_callback=None
    ):
    """Mean and median intensity outside of the segmented objects per frame.

    Parameters
    ----------
    images : dict of {str: numpy.ndarray}
        Image data per channel with shape ``(T, Y, X)`` or ``(T, Z, Y, X)``.
    segm : numpy.ndarray
        Segmentation masks with shape ``(T, Y, X)`` or ``(T, Z, Y, X)``. 2D
        masks of z-stack images are applied to every z-slice. Only the
        frames present in both the images and the masks are processed.
    exclude_zero_pixels : bool, optional
        If True (default), pixels with intensity 0 are not background (e.g.
        padding added by the alignment of the frames).
    progress_callback : callable, optional
        Called with no arguments after each processed frame.

    Returns
    -------
    pandas.DataFrame
        One row per frame with the columns ``frame_i``, ``num_objects`` and
        per channel ``<channel>_bkgr_mean``, ``<channel>_bkgr_median`` and
        ``<channel>_bkgr_num_pixels``. Statistics are NaN in frames without
        segmented objects or without background pixels.
    """
    for channel, img in images.items():
        _check_shapes(img, segm, channel)

    num_frames = min([len(segm), *[len(img) for img in images.values()]])
    columns = {'frame_i': np.arange(num_frames), 'num_objects': []}
    for channel in images:
        for statistic in STATISTICS:
            columns[background_colname(channel, statistic)] = []
        columns[num_pixels_colname(channel)] = []

    for frame_i in range(num_frames):
        lab = segm[frame_i]
        num_objects = len(np.unique(lab[lab > 0]))
        columns['num_objects'].append(num_objects)
        bkgr_mask = lab == 0
        for channel, img in images.items():
            frame_img = img[frame_i]
            if frame_img.ndim > bkgr_mask.ndim:
                mask = np.broadcast_to(bkgr_mask, frame_img.shape)
            else:
                mask = bkgr_mask
            if exclude_zero_pixels:
                mask = mask & (frame_img != 0)
            values = frame_img[mask]
            num_pixels = values.size
            if num_objects == 0 or num_pixels == 0:
                mean_val = median_val = np.nan
            else:
                values = values.astype(np.float64, copy=False)
                mean_val = values.mean()
                median_val = np.median(values)
            columns[background_colname(channel, 'mean')].append(mean_val)
            columns[background_colname(channel, 'median')].append(median_val)
            columns[num_pixels_colname(channel)].append(num_pixels)
        if progress_callback is not None:
            progress_callback()

    return pd.DataFrame(columns)


def included_cells_mask(acdc_df):
    """Boolean mask of the acdc_output rows that are not excluded cells."""
    if 'is_cell_excluded' not in acdc_df.columns:
        return pd.Series(True, index=acdc_df.index)
    from .timepoint_predictors import parse_flags
    return ~parse_flags(acdc_df['is_cell_excluded'])


# Disabled for now (per-frame correction of acdc_output, not used).
# Re-enable together with `corrected_colname`,
# `workers.ComputeBackgroundValuesWorker._correctAcdcOutput` and the
# `per_frame_*` values of `normalization_constants`.
# def add_background_corrected_columns(
#         acdc_df, bkgr_df, channels, statistic, is_segm_3D=False
#     ):
#     """Add ``<source>_bkgr_corrected`` columns to an acdc_output table.
#
#     Parameters
#     ----------
#     acdc_df : pandas.DataFrame
#         acdc_output table with ``frame_i`` as a column or index level.
#     bkgr_df : pandas.DataFrame
#         Per-frame background table from `compute_background_values`.
#     channels : iterable of str
#         Channels to correct. The source column of each channel is the first
#         of `mean_source_columns` present in `acdc_df`. Channels without a
#         source column or without background values are skipped.
#     statistic : {'mean', 'median'}
#         Background statistic subtracted from the source column.
#     is_segm_3D : bool, optional
#         True when the background was computed with 3D masks.
#
#     Returns
#     -------
#     tuple of (pandas.DataFrame, dict of {str: str}, list of str)
#         Copy of `acdc_df` with the new (or replaced) columns, the source
#         column of each corrected channel and the skipped channels. Rows of
#         frames without a background value get NaN.
#     """
#     if statistic not in STATISTICS:
#         raise ValueError(f'statistic must be one of {STATISTICS}.')
#     acdc_df = acdc_df.copy()
#     if 'frame_i' in acdc_df.columns:
#         frames = acdc_df['frame_i']
#     else:
#         frames = pd.Series(
#             acdc_df.index.get_level_values('frame_i'), index=acdc_df.index
#         )
#     frames = pd.to_numeric(frames, errors='coerce')
#     bkgr_by_frame = bkgr_df.set_index('frame_i')
#     corrected, skipped = {}, []
#     for channel in channels:
#         source_cols = [
#             col for col in mean_source_columns(channel, is_segm_3D)
#             if col in acdc_df.columns
#         ]
#         bkgr_col = background_colname(channel, statistic)
#         if not source_cols or bkgr_col not in bkgr_by_frame:
#             skipped.append(channel)
#             continue
#         source_col = source_cols[0]
#         bkgr = frames.map(bkgr_by_frame[bkgr_col])
#         values = pd.to_numeric(acdc_df[source_col], errors='coerce')
#         acdc_df[corrected_colname(source_col)] = values - bkgr
#         corrected[channel] = source_col
#     return acdc_df, corrected, skipped


def cell_mean_values(acdc_df, channel, is_segm_3D=False):
    """Mean intensities of the included cells of `channel` in an acdc_output
    table (not modified).

    Returns ``(source_col, values)`` with the first of `mean_source_columns`
    present in `acdc_df` and its numeric values for the rows that are not
    excluded cells, or ``(None, empty array)`` if the table has none.
    """
    for source_col in mean_source_columns(channel, is_segm_3D):
        if source_col not in acdc_df.columns:
            continue
        values = pd.to_numeric(acdc_df[source_col], errors='coerce')
        included = included_cells_mask(acdc_df)
        return source_col, values[included].to_numpy(dtype=np.float64)
    return None, np.array([], dtype=np.float64)


def _percentiles(values, low, high):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    return tuple(np.percentile(values, [low, high]))


def normalization_constants(
        raw_means, backgrounds, low_percentile=5.0, high_percentile=95.0
    ):
    """Pooled min-max normalization constants per channel.

    Parameters
    ----------
    raw_means : dict of {str: list of array-like}
        Per channel, the mean intensities (source column, e.g.
        ``<channel>_mean``) of the included cells of every Position.
        Channels present only in `backgrounds` get a row with NaN min/max.
    backgrounds : dict of {str: list of array-like}
        Per channel, the per-frame background values of every Position.
    low_percentile, high_percentile : float, optional
        Percentiles mapped to 0 and 1 (default 5 and 95).

    Returns
    -------
    pandas.DataFrame
        One row per channel with ``channel``, ``low_percentile``,
        ``high_percentile``, ``num_cell_values``, ``num_background_values``,
        ``constant_background``, ``constant_min`` and ``constant_max`` (see
        the module docstring).
    """
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError(
            'Percentiles must satisfy 0 <= low < high <= 100, got '
            f'{low_percentile} and {high_percentile}.'
        )
    rows = []
    channels = dict.fromkeys([*backgrounds, *raw_means])
    for channel in channels:
        raw = _concat(raw_means.get(channel, []))
        bkgr = _concat(backgrounds.get(channel, []))
        bkgr = bkgr[np.isfinite(bkgr)]
        constant_bkgr = np.median(bkgr) if bkgr.size else np.nan
        raw_min, raw_max = _percentiles(raw, low_percentile, high_percentile)
        rows.append({
            'channel': channel,
            'low_percentile': low_percentile,
            'high_percentile': high_percentile,
            'num_cell_values': int(np.isfinite(raw).sum()),
            'num_background_values': int(bkgr.size),
            'constant_background': constant_bkgr,
            # Percentiles are shift-equivariant: P(x - c) = P(x) - c
            'constant_min': raw_min - constant_bkgr,
            'constant_max': raw_max - constant_bkgr,
        })
    return pd.DataFrame(rows)


def _concat(arrays):
    arrays = [np.asarray(arr, dtype=np.float64).ravel() for arr in arrays]
    if not arrays:
        return np.array([], dtype=np.float64)
    return np.concatenate(arrays)
