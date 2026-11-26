"""File-system writes of a conversion: metadata.csv (legacy pandas code,
byte-identical text), metadataXML.txt, Images folders, raw-file renames and
moves, and the one-file-per-channel "other files".

Imports ``planning`` (one direction only).
"""
from __future__ import annotations

import csv
import logging
import os
import shutil
import traceback
from typing import Callable, Sequence

import pandas as pd

from .. import io
from . import errors, planning

logger = logging.getLogger(__name__)


def build_metadata_df(spec: planning.MetadataCsvSpec) -> pd.DataFrame:
    """metadata.csv DataFrame (index 'Description', column 'values'):
    LensNA, SizeT (saved frames), SizeZ, TimeIncrement, PhysicalSizeZ/Y/X,
    basename, then ``channel_{c}_name`` and ``channel_{c}_emWavelen`` of the
    saved channels. Same construction as the legacy ``saveData``, so
    ``to_csv`` gives the same text."""
    df = pd.DataFrame({
        'LensNA': spec.lens_na,
        'SizeT': spec.saved_size_t,
        'SizeZ': spec.size_z,
        'TimeIncrement': spec.time_increment_s,
        'PhysicalSizeZ': spec.physical_size_z_um,
        'PhysicalSizeY': spec.physical_size_y_um,
        'PhysicalSizeX': spec.physical_size_x_um,
        'basename': spec.basename
    }, index=['values']).T
    df.index.name = 'Description'

    size_c = len(spec.channel_names)
    ch_metadata = [
        chName for c, chName in enumerate(spec.channel_names)
        if spec.save_channels[c]
    ]
    description = [
        f'channel_{c}_name' for c in range(size_c)
        if spec.save_channels[c]
    ]
    ch_metadata.extend([
        wavelen for c, wavelen in enumerate(spec.em_wavelens)
        if spec.save_channels[c]
    ])
    description.extend([
        f'channel_{c}_emWavelen' for c in range(size_c)
        if spec.save_channels[c]
    ])

    df_channelNames = pd.DataFrame({
        'Description': description,
        'values': ch_metadata
    }).set_index('Description')

    df = pd.concat([df, df_channelNames])
    return df


def write_metadata_csv(path: str, spec: planning.MetadataCsvSpec) -> None:
    """Write ``build_metadata_df(spec)`` to ``path``. When the file exists,
    the name and emission-wavelength rows of the channels NOT saved now are
    copied from it (adding files to an existing Position keeps the other
    channels); rows missing there are skipped (logged at DEBUG)."""
    df = build_metadata_df(spec)
    if os.path.exists(path):
        existing_df = pd.read_csv(path).set_index('Description')
        for c in range(len(spec.channel_names)):
            if spec.save_channels[c]:
                continue
            for row in (f'channel_{c}_name', f'channel_{c}_emWavelen'):
                try:
                    df.at[row, 'values'] = existing_df.at[row, 'values']
                except Exception as err:
                    logger.debug(
                        f'Row "{row}" not copied from {path}: {err!r}'
                    )
    df.to_csv(path)


def read_metadata_csv(path: str) -> dict[str, str]:
    """``{Description: values}`` of a metadata.csv, values as text."""
    with open(path, 'r', newline='', encoding='utf-8') as file:
        rows = [row for row in csv.reader(file) if row]
    return {row[0]: row[1] if len(row) > 1 else '' for row in rows[1:]}


def write_metadata_xml(path: str, text: str) -> None:
    """Write the metadataXML text (utf-8, as the legacy worker)."""
    with open(path, 'w', encoding='utf-8') as txt:
        txt.write(text)


def prepare_images_folder(images_path: str, *, clear: bool) -> None:
    """Create ``images_path``; with ``clear`` an existing folder is deleted
    first (overwrite removes only ``Position_N/Images``)."""
    if clear and os.path.exists(images_path):
        shutil.rmtree(images_path)
    os.makedirs(images_path, exist_ok=True)


def execute_source_mutations(
        m: planning.SourceMutations, *, log: Callable[..., None]
    ) -> list[tuple[str, str]]:
    """Rename the raw files (``os.rename``), then move them into
    ``<dst>/raw_microscopy_files`` with ``io.move_raw_microscopy_file``
    (companion folder included), creating the folders when needed.

    Returns the ``(src, dst)`` moves done. Moves already done stay done
    when a later one fails (GUI behaviour).

    Raises
    ------
    MoveRawFilesError
        ``permission=True`` for a ``PermissionError``, else ``False`` for
        any other ``OSError``; ``moved`` lists the moves done.
    """
    for old, new in m.renames:
        log(
            f'Renaming "{os.path.basename(old)}" to '
            f'"{os.path.basename(new)}"...', 'INFO'
        )
        try:
            os.rename(old, new)
        except OSError as err:
            raise _move_error(err, old, new, [], action='renaming') from err

    moved: list[tuple[str, str]] = []
    for move in m.moves:
        logger.debug(f'Moving raw file {move.src} to {move.dst}')
        try:
            os.makedirs(os.path.dirname(move.dst), exist_ok=True)
            io.move_raw_microscopy_file(move.src, os.path.dirname(move.dst))
        except OSError as err:
            raise _move_error(
                err, move.src, move.dst, moved, action='moving'
            ) from err
        moved.append((move.src, move.dst))
    return moved


def _move_error(
        err: OSError, src: str, dst: str, moved: list[tuple[str, str]], *,
        action: str
    ) -> errors.MoveRawFilesError:
    return errors.MoveRawFilesError(
        f'Error while {action} the raw file "{src}" to "{dst}": {err}',
        moved=moved, failed=src,
        permission=isinstance(err, PermissionError),
        hint=(
            'Close every program that uses the file and check the folder '
            'permissions.'
        ),
        details=traceback.format_exc()
    )


def execute_other_files(
        actions: Sequence[planning.OtherFileAction], *,
        log: Callable[..., None]
    ) -> list[str]:
    """Move or copy (``shutil.move``/``shutil.copy``) the other files of a
    one-file-per-channel position. Errors are logged as WARNING text, not
    raised. Returns the destination paths written."""
    done = []
    for action in actions:
        transfer = shutil.move if action.mode == 'move' else shutil.copy
        try:
            transfer(action.src, action.dst)
        except Exception as err:
            log(
                f'Could not {action.mode} "{action.src}" to '
                f'"{action.dst}": {err}', 'WARNING'
            )
            continue
        done.append(action.dst)
    return done
