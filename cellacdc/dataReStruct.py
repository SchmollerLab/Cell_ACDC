import os
import re
import csv
import traceback
import tifffile

from functools import partial

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io
from natsort import natsorted
from qtpy.QtCore import QThread
from qtpy.QtWidgets import QFileDialog

from . import _main
from . import apps, html_utils, myutils, printl, widgets, workers
from . import exception_handler

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

def create_cellacdc_structure(selectedOptions, *, signals=None):
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

    tiffFiles = sorted(
        (
            path for path in sourceFolder.iterdir()
            if path.is_file()
            and path.name.lower().endswith(('.tif', '.tiff'))
        ),
        key=lambda path: natural_key(path.name),
    )
    if not tiffFiles:
        raise ValueError(f'No TIFF files found in {sourceFolder}.')

    if signals is not None:
        signals.initProgressBar.emit(len(tiffFiles))

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

        if signals is not None:
            signals.progressBar.emit(1)

    positionKeys = sorted(recordsByPosition, key=natural_key)
    destinationFolder.mkdir(parents=True, exist_ok=True)

    if not overwrite:
        existingFolders = [
            destinationFolder / f'Position_{index}'
            for index in range(1, len(positionKeys) + 1)
            if (destinationFolder / f'Position_{index}').exists()
        ]
        if existingFolders:
            raise FileExistsError(
                'Output Position folders already exist: '
                + ', '.join(str(path) for path in existingFolders)
                + '. Pass overwrite=True to replace matching image files.'
            )

    basename = (selectedOptions.get('basename') or '').strip().rstrip('_')
    filenamePrefix = f'{basename}_' if basename else ''

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

        positionFolder = (
            destinationFolder / f'Position_{positionNumber+posNumberShift}'
        )
        imagesFolder = positionFolder / 'Images'
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
        metadataPath = imagesFolder / f'{filenamePrefix}_metadata.csv'
        with metadataPath.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(('Description', 'values'))
            writer.writerow(('basename', filenamePrefix))
            writer.writerow(('SizeT', str(SizeT)))
            writer.writerow(('SizeZ', str(SizeZ)))
        
        if signals is not None:
            signals.progressBar.emit(1)

    out = {
        'position_folders': positionFolders,
        'image_files': createdFiles,
    }

    return out

def _log(mainWin, text):
    mainWin.log(text)

@exception_handler
def run(mainWin: '_main.mainWin'):
    setupWindow = apps.ReStructTiffDataStructureSetupDialogue(parent=mainWin)
    setupWindow.exec_()
    if setupWindow.cancel:
        return False

    mainWin._restructure_thread = QThread()
    mainWin._restructure_worker = workers.SimpleWorker(
        None, create_cellacdc_structure,
        func_args=(setupWindow.selectedOptions,),
    )
    mainWin._restructure_worker.set_func_kwargs(
        {'signals': mainWin._restructure_worker.signals}
    )
    mainWin._restructure_worker.moveToThread(mainWin._restructure_thread)
    mainWin._restructure_worker.signals.finished.connect(
        mainWin._restructure_thread.quit
    )
    mainWin._restructure_worker.signals.finished.connect(
        mainWin._restructure_worker.deleteLater
    )
    mainWin._restructure_thread.finished.connect(
        mainWin._restructure_thread.deleteLater
    )

    mainWin._restructure_worker.signals.critical.connect(
        mainWin.workerCritical
    )
    mainWin._restructure_worker.signals.initProgressBar.connect(
        mainWin.workerInitProgressbar
    )
    mainWin._restructure_worker.signals.progressBar.connect(
        mainWin.workerUpdateProgressbar
    )
    mainWin._restructure_worker.signals.progress.connect(
        mainWin.workerProgress
    )
    mainWin._restructure_worker.signals.finished.connect(
        mainWin.workerFinished
    )

    mainWin._restructure_thread.started.connect(mainWin._restructure_worker.run)
    mainWin._restructure_thread.start()

    return True

def saveTiff(filePath, data, waitCond):
    myutils.to_tiff(filePath, data)
    waitCond.wakeAll()
    del data