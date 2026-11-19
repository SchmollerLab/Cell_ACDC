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
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import tifffile

from .. import myutils
from . import errors

DIMENSION_KEYS = ('channels', 'timepoints', 'z_slices')
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


def fail(message):
    raise ValueError(message)

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

def create_datastruct_from_restruct_tiff_dialogue_options(
        selectedOptions, *, signals=None
    ):
    sourceFolder = Path(selectedOptions['source_folder'])
    destinationFolder = Path(selectedOptions['destination_folder'])
    dimensions = selectedOptions['dimensions']
    overwrite = selectedOptions['overwrite_destination']
    create_new_positions = selectedOptions['create_new_position_destination']

    if not sourceFolder.is_dir():
        raise ValueError(f'Source folder does not exist: {sourceFolder}')

    modes = {
        key: dimensions[key]['mode']
        for key in ('channels', 'timepoints', 'z_slices')
    }

    channelConfig = dimensions['channels']
    rawChannels = channelConfig.get('names') or [
        {'name': 'Channel_1'}
    ]
    channelSpecs = [
        item if isinstance(item, dict) else {'name': str(item)}
        for item in rawChannels
    ]
    channelNames = [item.get('name', '').strip() for item in channelSpecs]

    if not channelNames or any(not name for name in channelNames):
        raise ValueError('Provide a name for every channel.')

    if len({name.casefold() for name in channelNames}) != len(channelNames):
        raise ValueError('Channel names must be unique.')

    if modes['channels'] == 'not_present' and len(channelNames) != 1:
        raise ValueError(
            'When channels are marked "Not present", provide exactly '
            'one channel name.'
        )

    selectedFilenames = selectedOptions.get('files')
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
        raise ValueError(f'No TIFF files found in {sourceFolder}.')

    if signals is not None:
        signals.initProgressBar.emit(0)
        signals.progress.emit('Processing TIFF files...', 'INFO')

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
                raise ValueError(f'No image series found in {path.name}.')
            series = tif.series[0]
            shape = tuple(series.shape)
            dtype = np.dtype(series.dtype)

        layout = get_layout(shape, dimensions, modes)
        axes = layout['axes']

        if modes['channels'] == 'in_file':
            channelAxis = axes['channels']
            if shape[channelAxis] != len(channelNames):
                raise ValueError(
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

    positionKeys = sorted(recordsByPosition, key=natural_key)
    destinationFolder.mkdir(parents=True, exist_ok=True)

    basename = (selectedOptions.get('basename') or '').strip().rstrip('_')
    filenamePrefix = f'{basename}_' if basename else ''

    # User-provided metadata (PhysicalSizeX/Y/Z in μm, TimeIncrement in
    # seconds) from the setup dialogue as a Cell-ACDC metadata DataFrame
    metadataDf = selectedOptions.get('metadata_df')
    alreadyWrittenMetadata = {'basename', 'SizeT', 'SizeZ'}

    createdFiles = []
    positionFolders = {}
    posNumberShift = 0
    if create_new_positions:
        pos_foldernames = myutils.get_pos_foldernames(destinationFolder)
        last_pos_n = int(pos_foldernames[-1].split('_')[-1])
        posNumberShift = last_pos_n

    if signals is not None:
        signals.initProgressBar.emit(len(positionKeys))

    for positionNumber, positionIdentifier in enumerate(positionKeys, start=1):
        records = recordsByPosition[positionIdentifier]

        posFoldername = f'Position_{positionNumber+posNumberShift}'
        positionFolder = destinationFolder / posFoldername

        if positionFolder.exists() and not overwrite:
            raise FileExistsError(
                f'Output Position folder already exist: {positionFolder}. '
            )

        if signals is not None:
            signals.progress.emit(
                f'Creating Position folder {posFoldername}...', 'INFO'
            )

        imagesFolder = positionFolder / 'Images'
        if overwrite:
            try:
                shutil.rmtree(imagesFolder)
            except Exception:
                pass

        positionFolder.mkdir(parents=True, exist_ok=overwrite)
        imagesFolder.mkdir(parents=True, exist_ok=overwrite)
        positionFolders[positionIdentifier] = str(positionFolder)

        yxShapes = {
            record['layout']['yx_shape']
            for record in records
        }
        if len(yxShapes) != 1:
            raise ValueError(
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

        outputDtype = np.result_type(
            *[record['dtype'] for record in records]
        )

        for channelOutputIndex, channelName in enumerate(channelNames):
            channelRecords = [
                record for record in records
                if record['channel_index'] is None
                or record['channel_index'] == channelOutputIndex
            ]
            if not channelRecords:
                raise ValueError(
                    f'No input TIFFs matched channel {channelName!r} '
                    f'for position {positionIdentifier!r}.'
                )

            output = np.zeros(
                (outputT, outputZ, yxShape[0], yxShape[1]),
                dtype=outputDtype,
            )
            occupied = np.zeros((outputT, outputZ), dtype=bool)

            for record in channelRecords:
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
                t1 = t0 + array.shape[0]
                z1 = z0 + array.shape[1]

                if t1 > outputT or z1 > outputZ:
                    raise ValueError(
                        f'Data from {record["path"].name} exceeds the '
                        f'assembled T/Z shape.'
                    )

                if occupied[t0:t1, z0:z1].any():
                    raise ValueError(
                        f'Overlapping input files detected for position '
                        f'{positionIdentifier!r}, channel {channelName!r}: '
                        f'{record["path"].name}.'
                    )

                output[t0:t1, z0:z1, :, :] = array
                occupied[t0:t1, z0:z1] = True

            outputPath = (
                imagesFolder / f'{filenamePrefix}{channelName}.tif'
            )

            tifffile.imwrite(
                outputPath,
                np.squeeze(output),
                ome=True,
                metadata={'Name': channelName},
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
            for c, channel in enumerate(channelNames):
                writer.writerow((f'channel_{c}_name', channel))
            if metadataDf is not None:
                for description, value in metadataDf['values'].items():
                    if description in alreadyWrittenMetadata:
                        continue
                    if description.startswith('channel_'):
                        continue
                    writer.writerow((description, str(value)))

        if signals is not None:
            signals.progressBar.emit(1)

    out = {
        'position_folders': positionFolders,
        'image_files': createdFiles,
    }

    return out


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
