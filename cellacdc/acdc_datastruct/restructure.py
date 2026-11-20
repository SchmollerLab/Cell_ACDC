"""Qt-free core of "Re-structure image files" (loose TIFF files ->
``Position_<n>/Images``), shared by the GUI dialog
(``apps.ReStructTiffDataStructureSetupDialogue`` + ``dataReStruct.run``) and
the command line (``acdc-data restructure``).

The core functions report progress only through the optional, duck-typed
``signals`` object of ``workers.SimpleWorker``
(``initProgressBar.emit(n)``, ``progress.emit(text, level)``,
``progressBar.emit(k)``); headless callers pass
``callbacks.SignalsAdapter``. Recipes are the JSON files saved by the dialog
(dimension options + metadata).
"""
from __future__ import annotations

import copy
import csv
import json
import logging
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import tifffile

from . import errors, planning

logger = logging.getLogger(__name__)

DIMENSION_KEYS = ('channels', 'timepoints', 'z_slices')
MODES = ('not_present', 'per_file', 'in_file')
NAME_RE = re.compile(r'^[A-Za-z0-9_\-]+$')
"""Allowed channel names and basename (the dialog's alphaNumericLineEdit)."""
NAME_RULE = 'use only letters, digits, underscores (_) and dashes (-)'
RECIPE_PREFIX = 'datastruct_restructure_recipe_'
LAST_SETUP_FILENAME = '_last_setup.json'
TIME_UNIT_FACTORS = {
    'ms': 1e-3, 'seconds': 1.0, 'minutes': 60.0, 'hours': 3600.0
}
"""Time-unit labels of the dialog -> seconds (verbatim dialog table)."""
CLI_TIME_UNIT_TO_LABEL = {
    'ms': 'ms', 's': 'seconds', 'min': 'minutes', 'h': 'hours'
}
"""Command line time units -> dialog labels (recipes stay loadable in the
dialog)."""
DEFAULT_METADATA = {
    'PhysicalSizeX': 1.0,
    'PhysicalSizeY': 1.0,
    'PhysicalSizeZ': 1.0,
    'TimeIncrement': 1.0,
    'TimeIncrementUnit': 'seconds',
}
"""Metadata defaults of the dialog (um, um, um, value, unit label)."""
ON_EXISTING_CHOICES = ('error', 'overwrite', 'create-new')
_OME_PIXEL_TYPES = (
    'uint8', 'uint16', 'uint32', 'int8', 'int16', 'int32', 'float32',
    'float64', 'complex64', 'complex128', 'bool',
)
"""Numpy dtype names that ``tifffile.imwrite(..., ome=True)`` can store."""


def fail(message):
    raise errors.RestructureOptionsError(message)

def mode_for(key, dimensions):
    mode = dimensions[key].get('mode')
    allowed = {'not_present', 'per_file', 'in_file'}
    if mode not in allowed:
        fail(f'Unsupported mode {mode!r} for {key}.')
    return mode

def tiff_stem(path):
    name = path.name
    lower = name.lower()
    for suffix in ('.ome.tiff', '.ome.tif', '.tiff', '.tif'):
        if lower.endswith(suffix):
            return name[:-len(suffix)]
    return path.stem

def natural_key(value):
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r'(\d+)', value)
    )

def parse_index(stem, marker, path, dimension_name):
    if not marker:
        fail(
            f'No filename marker was provided for {dimension_name}; '
            f'file: {path.name}'
        )
    matches = list(
        re.finditer(re.escape(marker) + r'(\d+)', stem, flags=re.IGNORECASE)
    )
    if len(matches) != 1:
        fail(
            f'Expected one {dimension_name} marker {marker!r} followed '
            f'by digits in {path.name}; found {len(matches)}.'
        )
    match = matches[0]
    index = int(match.group(1))
    cleaned = stem[:match.start()] + stem[match.end():]
    return index, cleaned

def parse_channel(stem, channelSpecs, path):
    matches = []
    for channelIndex, channel in enumerate(channelSpecs):
        identifier = channel.get('filename_identifier')
        if not identifier:
            fail(
                f'No filename identifier was provided for channel '
                f'{channel["name"]!r}.'
            )
        found = list(
            re.finditer(re.escape(identifier), stem, flags=re.IGNORECASE)
        )
        if len(found) > 1:
            fail(
                f'Channel identifier {identifier!r} appears more than '
                f'once in {path.name}.'
            )
        if found:
            matches.append((channelIndex, found[0]))

    if len(matches) != 1:
        fail(
            f'Expected exactly one channel identifier in {path.name}; '
            f'found {len(matches)}.'
        )

    channelIndex, match = matches[0]
    cleaned = stem[:match.start()] + stem[match.end():]
    return channelIndex, cleaned

def axis_index(config, dimension_name, ndim):
    value = config.get('axis_index', config.get('axis'))
    if value is None or str(value).strip() == '':
        fail(f'Enter an axis index for {dimension_name}.')
    try:
        index = int(value)
    except (TypeError, ValueError):
        fail(
            f'The axis index for {dimension_name} must be an integer; '
            f'got {value!r}.'
        )
    if index < 0:
        index += ndim
    if not 0 <= index < ndim:
        fail(
            f'Axis index {value!r} for {dimension_name} is outside the '
            f'{ndim}-dimensional TIFF shape.'
        )
    return index

def get_layout(shape, dimensionConfigs, modes):
    axes = {}
    for key in ('channels', 'timepoints', 'z_slices'):
        if modes[key] == 'in_file':
            axes[key] = axis_index(
                dimensionConfigs[key], key, len(shape)
            )

    if len(set(axes.values())) != len(axes):
        fail('Each in-file dimension must use a different axis index.')

    spatialAxes = [
        index for index in range(len(shape))
        if index not in axes.values()
    ]
    if len(spatialAxes) != 2:
        fail(
            f'After selecting the in-file axes, expected two spatial '
            f'axes (Y, X), but found {len(spatialAxes)} for shape {shape}.'
        )

    return {
        'axes': axes,
        't_size': (
            shape[axes['timepoints']]
            if 'timepoints' in axes else 1
        ),
        'z_size': (
            shape[axes['z_slices']]
            if 'z_slices' in axes else 1
        ),
        'yx_shape': tuple(shape[index] for index in spatialAxes),
    }

def normalize_array(array, axes):
    """Return one channel as an array with shape (T, Z, Y, X)."""
    axisIds = list(range(array.ndim))

    channelAxis = axes.get('channels')
    if channelAxis is not None:
        array = np.take(array, 0, axis=channelAxis)
        axisIds.remove(channelAxis)

    orderedAxes = []
    for key in ('timepoints', 'z_slices'):
        axis = axes.get(key)
        if axis is not None:
            orderedAxes.append(axis)

    spatialAxes = [
        axis for axis in axisIds if axis not in orderedAxes
    ]
    orderedAxes.extend(spatialAxes)

    permutation = [axisIds.index(axis) for axis in orderedAxes]
    array = np.transpose(array, permutation)

    if 'timepoints' not in axes:
        array = np.expand_dims(array, axis=0)
    if 'z_slices' not in axes:
        array = np.expand_dims(array, axis=1)

    return array


# ------------------------------------------------------------------ options
def _check_name(name, what):
    if not NAME_RE.fullmatch(name):
        fail(f'Invalid {what} {name!r}: {NAME_RULE}.')


def normalize_restructure_options(opts: Mapping) -> dict:
    """Validated copy of the dialog's ``selectedOptions`` (a NEW dict; the
    'dimensions' entry is deep-copied, other values are shared).

    - 'source_folder' and 'destination_folder' are required;
      'overwrite_destination' / 'create_new_position_destination' default
      to False and cannot both be True.
    - Every dimension ('channels', 'timepoints', 'z_slices'; missing ->
      'not_present') has a mode of ``MODES``; per-file timepoints/z-slices
      need a 'filename_marker'; in-file dimensions an integer axis
      ('axis_index' or 'axis', kept as given).
    - Channel 'names' become ``[{'name': <stripped name>, ...}]`` (strings
      are accepted, default ``[{'name': 'Channel_1'}]``): non-empty, matching
      ``NAME_RE``, unique ignoring case, exactly one when channels are not
      present, and each with a 'filename_identifier' when per file.
    - A non-empty basename matches ``NAME_RE``.

    Raises ``RestructureOptionsError`` (the core's messages where the check
    already existed).
    """
    if not isinstance(opts, Mapping):
        fail('The re-structure options must be a mapping.')
    options = dict(opts)
    for key in ('source_folder', 'destination_folder'):
        if not options.get(key):
            fail(f'The {key.replace("_", " ")} is required.')

    overwrite = bool(options.get('overwrite_destination', False))
    createNew = bool(options.get('create_new_position_destination', False))
    if overwrite and createNew:
        fail(
            'Choose either to overwrite the existing Position folders or to '
            'create new ones, not both.'
        )
    options['overwrite_destination'] = overwrite
    options['create_new_position_destination'] = createNew

    rawDimensions = options.get('dimensions')
    if not isinstance(rawDimensions, Mapping):
        fail('The dimension options are required.')
    dimensions = {}
    for key in DIMENSION_KEYS:
        config = rawDimensions.get(key, {'mode': 'not_present'})
        if not isinstance(config, Mapping):
            fail(f'The options of {key} must be a mapping; got {config!r}.')
        dimensions[key] = copy.deepcopy(dict(config))
    modes = {key: mode_for(key, dimensions) for key in DIMENSION_KEYS}

    rawChannels = dimensions['channels'].get('names') or [
        {'name': 'Channel_1'}
    ]
    channelSpecs = []
    for item in rawChannels:
        spec = dict(item) if isinstance(item, Mapping) else {'name': item}
        name = spec.get('name')
        spec['name'] = '' if name is None else str(name).strip()
        channelSpecs.append(spec)
    channelNames = [spec['name'] for spec in channelSpecs]

    if any(not name for name in channelNames):
        fail('Provide a name for every channel.')
    for name in channelNames:
        _check_name(name, 'channel name')
    if len({name.casefold() for name in channelNames}) != len(channelNames):
        fail('Channel names must be unique.')
    if modes['channels'] == 'not_present' and len(channelNames) != 1:
        fail(
            'When channels are marked "Not present", provide exactly '
            'one channel name.'
        )
    if modes['channels'] == 'per_file':
        for spec in channelSpecs:
            if not spec.get('filename_identifier'):
                fail(
                    f'No filename identifier was provided for channel '
                    f'{spec["name"]!r}.'
                )
    dimensions['channels']['names'] = channelSpecs

    for key, dimension_name in (('timepoints', 'timepoint'),
                                ('z_slices', 'z-slice')):
        if modes[key] == 'per_file':
            if not dimensions[key].get('filename_marker'):
                fail(
                    f'No filename marker was provided for {dimension_name}.'
                )
    for key in DIMENSION_KEYS:
        if modes[key] != 'in_file':
            continue
        config = dimensions[key]
        value = config.get('axis_index', config.get('axis'))
        if value is None or str(value).strip() == '':
            fail(f'Enter an axis index for {key}.')
        try:
            int(value)
        except (TypeError, ValueError):
            fail(
                f'The axis index for {key} must be an integer; '
                f'got {value!r}.'
            )
    options['dimensions'] = dimensions

    basename = options.get('basename')
    if basename is not None:
        if not isinstance(basename, str):
            fail(f'The basename must be text; got {basename!r}.')
        if basename.strip():
            _check_name(basename.strip(), 'basename')
    return options


# --------------------------------------------------------------------- plan
@dataclass(frozen=True)
class PlannedChannel:
    """One output channel of a Position and the phase-1 header records
    (dicts: path, shape, dtype, axes, layout, channel/time/z index) of the
    input files it is assembled from."""
    name: str
    records: tuple[dict, ...]


@dataclass(frozen=True)
class PlannedPosition:
    """One output ``Position_<number>`` (``folder``) for the files whose
    stem reduces to ``identifier``; output shape (size_t, size_z, *yx);
    min_time/min_z = smallest per-file index (the Position's origin)."""
    identifier: str
    number: int
    folder: str
    size_t: int
    size_z: int
    yx: tuple[int, int]
    min_time: int
    min_z: int
    channels: tuple[PlannedChannel, ...]


@dataclass(frozen=True)
class RestructurePlan:
    """Everything ``execute_restructure_plan`` writes, validated.

    options       normalized options (``normalize_restructure_options``)
    prefix        output file prefix ('' or '<basename>_')
    overwrite     whether existing ``Images`` folders are replaced
    positions     planned Positions, in output order
    to_delete     existing ``Images`` folders that overwrite removes
    metadata_rows extra metadata.csv rows (description, text) from
                  'metadata_df'
    """
    options: dict
    prefix: str
    overwrite: bool
    positions: tuple[PlannedPosition, ...]
    to_delete: tuple[str, ...]
    metadata_rows: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict:
        """JSON-serializable summary (``--dry-run``, ``--format json``)."""
        return {
            'source_folder': str(self.options['source_folder']),
            'destination_folder': str(self.options['destination_folder']),
            'dimensions': copy.deepcopy(self.options['dimensions']),
            'prefix': self.prefix,
            'overwrite': self.overwrite,
            'create_new': self.options['create_new_position_destination'],
            'positions': [
                {
                    'identifier': position.identifier,
                    'number': position.number,
                    'folder': position.folder,
                    'size_t': position.size_t,
                    'size_z': position.size_z,
                    'yx': list(position.yx),
                    'min_time': position.min_time,
                    'min_z': position.min_z,
                    'channels': [
                        {
                            'name': channel.name,
                            'files': [
                                record['path'].name
                                for record in channel.records
                            ],
                            'image_file': _image_filepath(
                                position, self.prefix, channel.name
                            ),
                        }
                        for channel in position.channels
                    ],
                }
                for position in self.positions
            ],
            'to_delete': list(self.to_delete),
            'metadata_rows': [list(row) for row in self.metadata_rows],
        }


def _image_filepath(position: PlannedPosition, prefix: str,
                    channelName: str) -> str:
    return str(Path(position.folder) / 'Images' / f'{prefix}{channelName}.tif')


def _list_tiff_files(sourceFolder, selectedFilenames):
    if selectedFilenames is not None:
        selectedFilenames = set(selectedFilenames)

    tiffFiles = sorted(
        (
            path for path in sourceFolder.iterdir()
            if path.is_file()
            and path.name.lower().endswith(('.tif', '.tiff'))
            and (
                selectedFilenames is None
                or path.name in selectedFilenames
            )
        ),
        key=lambda path: natural_key(path.name),
    )

    if not tiffFiles:
        raise errors.SourceError(f'No TIFF files found in {sourceFolder}.')
    return tiffFiles


def _scan_headers(tiffFiles, dimensions, modes, channelSpecs):
    """Phase 1 (headers only): records grouped by position identifier."""
    channelNames = [item['name'] for item in channelSpecs]
    recordsByPosition = defaultdict(list)
    for path in tiffFiles:
        stem = tiff_stem(path)
        cleanedStem = stem
        timeIndex = None
        zIndex = None
        channelIndex = None

        if modes['timepoints'] == 'per_file':
            timeIndex, cleanedStem = parse_index(
                cleanedStem,
                dimensions['timepoints'].get('filename_marker'),
                path,
                'timepoint',
            )

        if modes['z_slices'] == 'per_file':
            zIndex, cleanedStem = parse_index(
                cleanedStem,
                dimensions['z_slices'].get('filename_marker'),
                path,
                'z-slice',
            )

        if modes['channels'] == 'per_file':
            channelIndex, cleanedStem = parse_channel(
                cleanedStem,
                channelSpecs,
                path,
            )
        elif modes['channels'] == 'not_present':
            channelIndex = 0

        # The position identifier is whatever remains after removing the
        # filename-encoded dimensions selected as "One file per ...".
        positionIdentifier = cleanedStem.strip(' _-.') or 'Position'

        with tifffile.TiffFile(path) as tif:
            if not tif.series:
                raise errors.SourceError(
                    f'No image series found in {path.name}.'
                )
            series = tif.series[0]
            shape = tuple(series.shape)
            dtype = np.dtype(series.dtype)

        layout = get_layout(shape, dimensions, modes)
        axes = layout['axes']

        if modes['channels'] == 'in_file':
            channelAxis = axes['channels']
            if shape[channelAxis] != len(channelNames):
                fail(
                    f'{path.name} has {shape[channelAxis]} channels on '
                    f'axis {channelAxis}, but {len(channelNames)} channel '
                    f'names were provided.'
                )

        recordsByPosition[positionIdentifier].append({
            'path': path,
            'shape': shape,
            'dtype': dtype,
            'axes': axes,
            'layout': layout,
            'channel_index': channelIndex,
            'time_index': timeIndex,
            'z_index': zIndex,
        })
    return recordsByPosition


def _describe_pixel_types(records) -> str:
    """'uint32: a.tif; int16: b.tif and 2 more' (error messages)."""
    filesByType = defaultdict(list)
    for record in records:
        filesByType[record['dtype'].name].append(record['path'].name)
    return '; '.join(
        f'{name}: {files[0]}'
        + (f' and {len(files) - 1} more' if len(files) > 1 else '')
        for name, files in filesByType.items()
    )


def _plan_position(positionIdentifier, records, number, folder, modes,
                   channelNames):
    """Phase-2 checks of one Position from the headers (Y/X, missing
    channel, out of bounds, overlaps, in the order of the legacy writer;
    then the output pixel type, which the legacy writer only rejected when
    writing)."""
    yxShapes = {
        record['layout']['yx_shape']
        for record in records
    }
    if len(yxShapes) != 1:
        raise errors.SourceError(
            f'Files grouped as position {positionIdentifier!r} have '
            f'different Y/X shapes: {sorted(yxShapes)}.'
        )
    yxShape = next(iter(yxShapes))

    if modes['timepoints'] == 'per_file':
        timeLabels = [record['time_index'] for record in records]
        minTime = min(timeLabels)
        outputT = max(timeLabels) - minTime + 1
    elif modes['timepoints'] == 'in_file':
        minTime = 0
        outputT = max(
            record['layout']['t_size'] for record in records
        )
    else:
        minTime = 0
        outputT = 1

    if modes['z_slices'] == 'per_file':
        zLabels = [record['z_index'] for record in records]
        minZ = min(zLabels)
        outputZ = max(zLabels) - minZ + 1
    elif modes['z_slices'] == 'in_file':
        minZ = 0
        outputZ = max(
            record['layout']['z_size'] for record in records
        )
    else:
        minZ = 0
        outputZ = 1

    channels = []
    for channelOutputIndex, channelName in enumerate(channelNames):
        channelRecords = [
            record for record in records
            if record['channel_index'] is None
            or record['channel_index'] == channelOutputIndex
        ]
        if not channelRecords:
            raise errors.SourceError(
                f'No input TIFFs matched channel {channelName!r} '
                f'for position {positionIdentifier!r}.'
            )

        occupied = np.zeros((outputT, outputZ), dtype=bool)
        for record in channelRecords:
            t0, z0 = _record_origin(record, modes, minTime, minZ)
            t1 = t0 + record['layout']['t_size']
            z1 = z0 + record['layout']['z_size']

            if t1 > outputT or z1 > outputZ:
                raise errors.SourceError(
                    f'Data from {record["path"].name} exceeds the '
                    f'assembled T/Z shape.'
                )

            if occupied[t0:t1, z0:z1].any():
                raise errors.SourceError(
                    f'Overlapping input files detected for position '
                    f'{positionIdentifier!r}, channel {channelName!r}: '
                    f'{record["path"].name}.'
                )
            occupied[t0:t1, z0:z1] = True
        channels.append(PlannedChannel(channelName, tuple(channelRecords)))

    # The writer stores every channel of the Position with this pixel type.
    pixelType = np.result_type(*[record['dtype'] for record in records])
    if pixelType.name not in _OME_PIXEL_TYPES:
        raise errors.SourceError(
            f'OME-TIFF cannot store the pixel type {pixelType.name!r} of '
            f'the files grouped as position {positionIdentifier!r} '
            f'({_describe_pixel_types(records)}). Supported pixel types: '
            f'{", ".join(_OME_PIXEL_TYPES)}.'
        )

    return PlannedPosition(
        identifier=positionIdentifier, number=number, folder=str(folder),
        size_t=outputT, size_z=outputZ, yx=yxShape, min_time=minTime,
        min_z=minZ, channels=tuple(channels),
    )


def _record_origin(record, modes, minTime, minZ):
    t0 = (
        record['time_index'] - minTime
        if modes['timepoints'] == 'per_file'
        else 0
    )
    z0 = (
        record['z_index'] - minZ
        if modes['z_slices'] == 'per_file'
        else 0
    )
    return t0, z0


def _metadata_rows(metadataDf):
    """Extra metadata.csv rows of the dialog's metadata DataFrame (every
    row except basename, SizeT, SizeZ and channel_*)."""
    if metadataDf is None:
        return ()
    alreadyWrittenMetadata = {'basename', 'SizeT', 'SizeZ'}
    rows = []
    for description, value in metadataDf['values'].items():
        if description in alreadyWrittenMetadata:
            continue
        if description.startswith('channel_'):
            continue
        rows.append((description, str(value)))
    return tuple(rows)


def plan_restructure(
        opts: Mapping, *, signals: object | None = None
    ) -> RestructurePlan:
    """Validate everything and plan the output, without writing anything.

    ``normalize_restructure_options``, source folder and TIFF listing, then
    ``signals.initProgressBar.emit(0)`` and
    ``signals.progress.emit('Processing TIFF files...', 'INFO')`` (where the
    legacy core emitted them; errors raised before emit nothing), then the
    header scan, the Position numbering (create new: after the highest
    valid existing Position, 1 on a destination without Positions) and
    every check the legacy core made while writing: existing Position
    folder without overwrite (``ExistingPositionsError``), Y/X mismatch,
    missing channel, out of bounds and overlaps (``SourceError``), with the
    legacy messages; also a pixel type that OME-TIFF cannot store
    (``SourceError``; the legacy writer failed on it while writing).
    """
    options = normalize_restructure_options(opts)
    sourceFolder = Path(options['source_folder'])
    destinationFolder = Path(options['destination_folder'])
    dimensions = options['dimensions']
    overwrite = options['overwrite_destination']
    create_new_positions = options['create_new_position_destination']

    if not sourceFolder.is_dir():
        raise errors.SourceError(
            f'Source folder does not exist: {sourceFolder}'
        )

    modes = {key: dimensions[key]['mode'] for key in DIMENSION_KEYS}
    channelSpecs = dimensions['channels']['names']
    channelNames = [item['name'] for item in channelSpecs]

    tiffFiles = _list_tiff_files(sourceFolder, options.get('files'))

    if signals is not None:
        signals.initProgressBar.emit(0)
        signals.progress.emit('Processing TIFF files...', 'INFO')

    recordsByPosition = _scan_headers(
        tiffFiles, dimensions, modes, channelSpecs
    )
    positionKeys = sorted(recordsByPosition, key=natural_key)

    basename = (options.get('basename') or '').strip().rstrip('_')
    filenamePrefix = f'{basename}_' if basename else ''

    posNumberShift = 0
    if create_new_positions:
        posNumberShift = max(
            planning.existing_position_numbers(str(destinationFolder)),
            default=0
        )

    numbers = [
        positionNumber + posNumberShift
        for positionNumber in range(1, len(positionKeys) + 1)
    ]
    existing = [
        number for number in numbers
        if (destinationFolder / f'Position_{number}').exists()
    ]
    positions = []
    for number, positionIdentifier in zip(numbers, positionKeys):
        positionFolder = destinationFolder / f'Position_{number}'
        if positionFolder.exists() and not overwrite:
            raise errors.ExistingPositionsError(
                f'Output Position folder already exist: {positionFolder}. ',
                existing=existing, choices=('overwrite', 'create-new'),
                hint=(
                    'Overwrite them (their Images folders are deleted) or '
                    'create new Position folders.'
                ),
            )
        positions.append(_plan_position(
            positionIdentifier, recordsByPosition[positionIdentifier],
            number, positionFolder, modes, channelNames,
        ))

    toDelete = tuple(
        str(Path(position.folder) / 'Images') for position in positions
        if overwrite and (Path(position.folder) / 'Images').exists()
    )
    return RestructurePlan(
        options=options, prefix=filenamePrefix, overwrite=overwrite,
        positions=tuple(positions), to_delete=toDelete,
        metadata_rows=_metadata_rows(options.get('metadata_df')),
    )


# ------------------------------------------------------------------ execute
def _squeeze_time_and_z(array):
    """Drop the T and Z axes of a (T, Z, Y, X) array when their size is 1
    (Y and X are always kept)."""
    singletons = tuple(axis for axis in (0, 1) if array.shape[axis] == 1)
    return np.squeeze(array, axis=singletons)


def execute_restructure_plan(
        plan: RestructurePlan, *, signals: object | None = None
    ) -> dict:
    """Write the planned Positions: per Position, remove ``Images`` when
    overwriting, create the folders, assemble and write one OME-TIFF per
    channel (``<prefix><channel>.tif``, singleton T/Z squeezed) and
    ``<prefix>metadata.csv``. Same signals, in the same order, as the legacy
    core (``initProgressBar.emit(<number of Positions>)``, per Position
    ``progress.emit('Creating Position folder ...', 'INFO')`` and
    ``progressBar.emit(1)``).

    Returns ``{'position_folders': {identifier: folder},
    'image_files': [paths]}``.
    """
    destinationFolder = Path(plan.options['destination_folder'])
    dimensions = plan.options['dimensions']
    modes = {key: dimensions[key]['mode'] for key in DIMENSION_KEYS}
    channelNames = [channel['name'] for channel in dimensions['channels'][
        'names'
    ]]
    overwrite = plan.overwrite
    filenamePrefix = plan.prefix
    destinationFolder.mkdir(parents=True, exist_ok=True)

    createdFiles = []
    positionFolders = {}

    if signals is not None:
        signals.initProgressBar.emit(len(plan.positions))

    for position in plan.positions:
        positionFolder = Path(position.folder)

        if signals is not None:
            signals.progress.emit(
                f'Creating Position folder {positionFolder.name}...', 'INFO'
            )

        imagesFolder = positionFolder / 'Images'
        if overwrite:
            try:
                shutil.rmtree(imagesFolder)
            except Exception:
                pass

        positionFolder.mkdir(parents=True, exist_ok=overwrite)
        imagesFolder.mkdir(parents=True, exist_ok=overwrite)
        positionFolders[position.identifier] = str(positionFolder)

        yxShape = position.yx
        outputT, outputZ = position.size_t, position.size_z
        outputDtype = np.result_type(*[
            record['dtype']
            for channel in position.channels for record in channel.records
        ])

        for channelOutputIndex, channel in enumerate(position.channels):
            output = np.zeros(
                (outputT, outputZ, yxShape[0], yxShape[1]),
                dtype=outputDtype,
            )

            for record in channel.records:
                with tifffile.TiffFile(record['path']) as tif:
                    array = tif.series[0].asarray()

                if tuple(array.shape) != record['shape']:
                    raise ValueError(
                        f'TIFF shape changed while reading '
                        f'{record["path"].name}: expected '
                        f'{record["shape"]}, got {tuple(array.shape)}.'
                    )

                axes = dict(record['axes'])

                if modes['channels'] == 'in_file':
                    channelAxis = axes.pop('channels')
                    array = np.take(
                        array,
                        channelOutputIndex,
                        axis=channelAxis,
                    )

                    # Taking an axis shifts the indices of later axes left.
                    axes = {
                        key: axis - 1 if axis > channelAxis else axis
                        for key, axis in axes.items()
                    }

                array = normalize_array(array, axes)

                if tuple(array.shape[2:]) != yxShape:
                    raise ValueError(
                        f'Unexpected spatial shape in '
                        f'{record["path"].name}: '
                        f'{tuple(array.shape[2:])}.'
                    )

                t0, z0 = _record_origin(
                    record, modes, position.min_time, position.min_z
                )
                t1 = t0 + array.shape[0]
                z1 = z0 + array.shape[1]
                output[t0:t1, z0:z1, :, :] = array

            outputPath = (
                imagesFolder / f'{filenamePrefix}{channel.name}.tif'
            )

            tifffile.imwrite(
                outputPath,
                _squeeze_time_and_z(output),
                ome=True,
                metadata={'Name': channel.name},
            )
            createdFiles.append(str(outputPath))

        SizeT, SizeZ = output.shape[:2]
        metadataPath = imagesFolder / f'{filenamePrefix}metadata.csv'
        with metadataPath.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(('Description', 'values'))
            writer.writerow(('basename', filenamePrefix))
            writer.writerow(('SizeT', str(SizeT)))
            writer.writerow(('SizeZ', str(SizeZ)))
            for c, channelName in enumerate(channelNames):
                writer.writerow((f'channel_{c}_name', channelName))
            for description, value in plan.metadata_rows:
                writer.writerow((description, value))

        if signals is not None:
            signals.progressBar.emit(1)

    out = {
        'position_folders': positionFolders,
        'image_files': createdFiles,
    }

    return out


def create_datastruct_from_restruct_tiff_dialogue_options(
        selectedOptions, *, signals=None
    ):
    """Re-structure loose TIFF files as the dialog's options say:
    ``execute_restructure_plan(plan_restructure(...))``, so every validation
    error is raised before anything is written. A selected
    'segmentation_mask_folder' is not used: a WARNING says so (through
    ``signals`` when given)."""
    plan = plan_restructure(selectedOptions, signals=signals)
    segmFolder = selectedOptions.get('segmentation_mask_folder')
    if segmFolder:
        text = (
            f'The segmentation mask folder "{segmFolder}" is ignored: '
            '"Re-structure image files" only re-structures the TIFF files '
            'of the source folder.'
        )
        if signals is not None:
            signals.progress.emit(text, 'WARNING')
        else:
            logger.warning(text)
    return execute_restructure_plan(plan, signals=signals)


# ----------------------------------------------------------------- metadata
def build_metadata_df(
        *, physical_size_x: float, physical_size_y: float,
        physical_size_z: float, time_increment: float,
        time_increment_unit: str, channel_names: Sequence[str]
    ) -> pd.DataFrame:
    """Cell-ACDC metadata DataFrame of the dialog's metadata group.

    Index 'Description', single column 'values' (like ``metadata.csv``):
    rows TimeIncrement (converted to seconds with ``TIME_UNIT_FACTORS``),
    PhysicalSizeZ, PhysicalSizeY, PhysicalSizeX (um) and
    ``channel_<c>_name``. ``time_increment_unit`` is a dialog label
    ('ms', 'seconds', 'minutes', 'hours'); anything else raises
    ``RestructureOptionsError``.
    """
    if time_increment_unit not in TIME_UNIT_FACTORS:
        raise errors.RestructureOptionsError(
            f'Unknown time increment unit {time_increment_unit!r}; '
            f'expected one of {", ".join(TIME_UNIT_FACTORS)}.'
        )
    toSecondsFactor = TIME_UNIT_FACTORS[time_increment_unit]
    values = {
        'TimeIncrement': time_increment*toSecondsFactor,
        'PhysicalSizeZ': physical_size_z,
        'PhysicalSizeY': physical_size_y,
        'PhysicalSizeX': physical_size_x,
    }
    for c, name in enumerate(channel_names):
        values[f'channel_{c}_name'] = name

    df = pd.DataFrame({
        'Description': list(values.keys()),
        'values': list(values.values())
    }).set_index('Description')
    return df


# ------------------------------------------------------------------ recipes
def recipes_folder() -> str:
    """Folder of the recipes saved by the dialog
    (``cellacdc.restruct_files_recipes_path``, read at call time)."""
    from .. import restruct_files_recipes_path
    return restruct_files_recipes_path


def recipe_filenames(folder: str) -> list[str]:
    """File names of the recipes saved in ``folder``
    (``<RECIPE_PREFIX>NAME.json``, '.json' in any case), natsorted as the
    dialog's "Load recipe" lists them; [] when ``folder`` is not a
    folder."""
    from natsort import natsorted

    if not os.path.isdir(folder):
        return []
    return natsorted(
        filename for filename in os.listdir(folder)
        if filename.startswith(RECIPE_PREFIX)
        and filename.lower().endswith('.json')
    )


def _recipe_names(folder: str) -> list[str]:
    """NAMEs of the ``<RECIPE_PREFIX>NAME.json`` files in ``folder``."""
    return [
        filename[len(RECIPE_PREFIX):-len('.json')]
        for filename in recipe_filenames(folder)
    ]


def last_setup_path(folder: str | None = None) -> str:
    """Path of the setup the dialog saved at its last Ok
    (``<folder>/LAST_SETUP_FILENAME``; ``folder`` defaults to
    ``recipes_folder()``): the dialog's "Load last parameters" and
    ``resolve_recipe_spec('last')``.

    Raises
    ------
    RestructureOptionsError
        The file does not exist.
    """
    if folder is None:
        folder = recipes_folder()
    path = os.path.join(folder, LAST_SETUP_FILENAME)
    if os.path.isfile(path):
        return path
    raise errors.RestructureOptionsError(
        f'No last "Re-structure image files" setup found ({path}).',
        hint='Press Ok in the dialog once, or give a recipe file or name.'
    )


def resolve_recipe_spec(spec: str, folder: str | None = None) -> str:
    """Path of the recipe given on the command line.

    ``spec`` is an existing file path (returned as an absolute path),
    'last' (the dialog's last setup, ``LAST_SETUP_FILENAME``) or NAME
    (``<folder>/<RECIPE_PREFIX>NAME.json``; a trailing '.json' is accepted).
    ``folder`` defaults to ``recipes_folder()``. Anything else raises
    ``RestructureOptionsError`` listing the saved recipe names.
    """
    if os.path.isfile(spec):
        return os.path.abspath(spec)
    if folder is None:
        folder = recipes_folder()
    if spec == 'last':
        return last_setup_path(folder)
    name = spec[:-len('.json')] if spec.lower().endswith('.json') else spec
    path = os.path.join(folder, f'{RECIPE_PREFIX}{name}.json')
    if os.path.isfile(path):
        return path
    names = _recipe_names(folder)
    available = ', '.join(names) if names else 'none'
    raise errors.RestructureOptionsError(
        f'Recipe {spec!r} is neither a file nor a recipe saved in {folder} '
        f"(saved recipes: {available}; or 'last').",
    )


def load_recipe(path: str) -> dict:
    """Recipe saved by the dialog (or by ``save_recipe``): the dimension
    options ('channels', 'timepoints', 'z_slices'; in-file axes under
    'axis' or 'axis_index', as int or str) plus 'metadata'. Returned
    unchanged; unreadable files or non-object JSON raise
    ``RestructureOptionsError``."""
    try:
        with open(path, 'r', encoding='utf-8-sig') as file:
            recipe = json.load(file)
    except (OSError, ValueError) as err:
        raise errors.RestructureOptionsError(
            f'Cannot read the recipe {path}: {err}'
        ) from err
    if not isinstance(recipe, dict):
        raise errors.RestructureOptionsError(
            f'The recipe {path} does not contain a JSON object.'
        )
    return recipe


def save_recipe(
        path: str, dimension_options: Mapping, metadata: Mapping
    ) -> None:
    """Write ``{**dimension_options, 'metadata': metadata}`` as the dialog
    does (``json.dump(..., indent=2)``, text mode: byte-identical files).
    The parent folder is created when missing."""
    recipe = {**dimension_options, 'metadata': dict(metadata)}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(recipe, file, indent=2)


def _channel_names(channels: Mapping) -> list[str]:
    """Output channel names of a 'channels' dimension option (as the core
    reads them; default ['Channel_1'])."""
    rawChannels = channels.get('names') or [{'name': 'Channel_1'}]
    return [
        item.get('name', '').strip() if isinstance(item, Mapping)
        else str(item).strip()
        for item in rawChannels
    ]


def _recipe_metadata(recipe: Mapping, metadata: Mapping | None) -> dict:
    """``DEFAULT_METADATA`` updated by the recipe's 'metadata' and then by
    ``metadata`` (None values are ignored)."""
    unknown = sorted(set(metadata or {}) - set(DEFAULT_METADATA))
    if unknown:
        raise errors.RestructureOptionsError(
            f'Unknown metadata keys {unknown}; expected '
            f'{", ".join(DEFAULT_METADATA)}.'
        )
    recipeMetadata = recipe.get('metadata')
    if recipeMetadata is None:
        recipeMetadata = {}
    if not isinstance(recipeMetadata, Mapping):
        raise errors.RestructureOptionsError(
            "The recipe 'metadata' entry must be a JSON object."
        )
    merged = dict(DEFAULT_METADATA)
    for source in (recipeMetadata, metadata or {}):
        for key in DEFAULT_METADATA:
            if source.get(key) is not None:
                merged[key] = source[key]
    for key in ('PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeZ',
                'TimeIncrement'):
        try:
            merged[key] = float(merged[key])
        except (TypeError, ValueError):
            raise errors.RestructureOptionsError(
                f'{key} must be a number; got {merged[key]!r}.'
            ) from None
    return merged


def options_from_recipe(
        recipe: Mapping, *, source_folder: str, destination_folder: str,
        basename: str | None = None, files: Sequence[str] | None = None,
        on_existing: Literal['error', 'overwrite', 'create-new'] = 'error',
        metadata: Mapping | None = None
    ) -> dict:
    """``selectedOptions`` for the core, as the dialog builds them, from a
    recipe (``load_recipe``) plus what the recipe does not hold.

    Missing dimensions are 'not_present'. ``on_existing`` maps to
    ``overwrite_destination`` / ``create_new_position_destination``.
    ``metadata`` entries replace the recipe's ones (``DEFAULT_METADATA``
    for the rest); the merged values give 'metadata' and the dialog-style
    'metadata_df' (``build_metadata_df``).
    """
    if not isinstance(recipe, Mapping):
        raise errors.RestructureOptionsError(
            'A recipe must be a JSON object.'
        )
    if on_existing not in ON_EXISTING_CHOICES:
        raise errors.RestructureOptionsError(
            f'Unknown on_existing {on_existing!r}; expected one of '
            f'{", ".join(ON_EXISTING_CHOICES)}.'
        )
    dimensions = {}
    for key in DIMENSION_KEYS:
        dimension = recipe.get(key) or {'mode': 'not_present'}
        if not isinstance(dimension, Mapping):
            raise errors.RestructureOptionsError(
                f'The recipe entry {key!r} must be a JSON object.'
            )
        dimensions[key] = copy.deepcopy(dict(dimension))
    merged = _recipe_metadata(recipe, metadata)
    metadataDf = build_metadata_df(
        physical_size_x=merged['PhysicalSizeX'],
        physical_size_y=merged['PhysicalSizeY'],
        physical_size_z=merged['PhysicalSizeZ'],
        time_increment=merged['TimeIncrement'],
        time_increment_unit=merged['TimeIncrementUnit'],
        channel_names=_channel_names(dimensions['channels']),
    )
    return {
        'source_folder': source_folder,
        'destination_folder': destination_folder,
        'overwrite_destination': on_existing == 'overwrite',
        'create_new_position_destination': on_existing == 'create-new',
        'dimensions': dimensions,
        'basename': basename,
        'files': None if files is None else list(files),
        'metadata': merged,
        'metadata_df': metadataDf,
    }
