"""Channel-name sanitizing/validation and Cell-ACDC output file names
(standard library only)."""
from __future__ import annotations

import os
from typing import Sequence


def sanitize_name(text: str) -> str:
    """Replace every character that is not alphanumeric or ``_`` with ``_``
    and strip trailing ``_`` (the rule of the metadata dialog,
    ``QDialogMetadataXML.removeInvalidCharacters``).

    ``''`` gives ``''``; e.g. ``'GFP 1!'`` gives ``'GFP_1'``.
    """
    sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in text)
    return sanitized.rstrip('_')


def validate_channel_names(
        names: Sequence[str], save: Sequence[bool] | None = None
    ) -> list[str]:
    """Check the (already sanitized) names of the saved channels.

    Saved names must be non-empty and pairwise distinct ignoring case
    (``casefold()``), because files differing only by case overwrite each
    other on Windows and macOS. ``save`` defaults to every channel saved.

    Returns the human-readable problems (``[]`` = valid); raises nothing.
    """
    if save is None:
        save = [True] * len(names)
    problems = []
    seen: dict[str, int] = {}
    for c, (name, is_saved) in enumerate(zip(names, save)):
        if not is_saved:
            continue
        if not name:
            problems.append(
                f'Channel {c} name is empty (after replacing characters '
                'that are not letters, numbers or "_").'
            )
            continue
        key = name.casefold()
        if key in seen:
            problems.append(
                f'Channel {seen[key]} and channel {c} have the same name '
                f'"{name}" (names are compared ignoring case, after '
                'replacing invalid characters with "_").'
            )
        else:
            seen[key] = c
    return problems


def raw_file_stem(filename: str) -> str:
    """Basename without extension; a trailing ``.ome`` of the stem is also
    removed (case-insensitive).

    ``'x_1_GFP.ome.tif'`` gives ``'x_1_GFP'``, ``'a.CZI'`` gives ``'a'``.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    if stem.lower().endswith('.ome'):
        stem = stem[:-len('.ome')]
    return stem


def output_stem(filename: str) -> str:
    """Stem used in output file names (legacy ``getFilename``):
    ``os.path.splitext(filename)[0]`` with dots replaced by ``_``, e.g.
    ``'x.ome.tif'`` gives ``'x_ome'``."""
    return os.path.splitext(filename)[0].replace('.', '_')


def pos_str(number: int, num_positions: int) -> str:
    """Position number zero-padded to the number of digits of
    ``num_positions`` (legacy padding)."""
    return str(number).zfill(len(str(num_positions)))


def position_basename(stem: str, pos: str, image_name: str = '') -> str:
    """Basename shared by every file of a Position: ``'<stem>_s<pos>_'``, or
    ``'<stem>_<image_name>_s<pos>_'`` when ``image_name`` is not empty.
    ``stem`` is ``output_stem(...)``."""
    if image_name:
        return f'{stem}_{image_name}_s{pos}_'
    return f'{stem}_s{pos}_'


def position_filename(basename: str, suffix: str, ext: str) -> str:
    """File name ``f'{basename}{suffix}{ext}'``, e.g.
    ``('x_s1_', 'metadata', '.csv')`` gives ``'x_s1_metadata.csv'``."""
    return f'{basename}{suffix}{ext}'
