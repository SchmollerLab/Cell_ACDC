"""Raw microscopy files: layouts, listing and checks of the source folder,
source selection for the command line, one-file-per-channel name parsing and
metadata units (one metadata read + one confirmation each).

Read-only: nothing here changes the file system.
"""
from __future__ import annotations

import collections
import os
import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Collection, Literal, Mapping, Sequence

from natsort import natsorted

from .. import acdc_regex, io, myutils
from . import errors, naming

RAW_MICROSCOPY_FILES_FOLDERNAME = 'raw_microscopy_files'
COMPOUND_EXTENSIONS = ('.ome.tiff', '.ome.tif', '.ome.zarr')
ALLOWED_DOUBLE_EXTENSIONS = ('.ome.tif',)
_POSITION_CHANNEL_RE = re.compile(r'(\d+)_(.+)')


class RawLayout(IntEnum):
    """How positions and channels are spread over the raw files (the
    "Raw data structure" of the setup dialog)."""
    SINGLE_FILE_MULTI_POS = 0
    ONE_FILE_PER_POS = 1
    ONE_FILE_PER_CHANNEL = 2

    @classmethod
    def from_cli(cls, name: str) -> 'RawLayout':
        """Layout of a command line name ('single-multi-pos', 'one-per-pos',
        'one-per-channel')."""
        for layout, cli_name in _CLI_NAMES.items():
            if cli_name == name:
                return layout
        raise errors.UsageError(
            f'Unknown layout "{name}" '
            f'(choose from {", ".join(_CLI_NAMES.values())}).'
        )

    @property
    def cli_name(self) -> str:
        """Command line name of the layout."""
        return _CLI_NAMES[self]


_CLI_NAMES = {
    RawLayout.SINGLE_FILE_MULTI_POS: 'single-multi-pos',
    RawLayout.ONE_FILE_PER_POS: 'one-per-pos',
    RawLayout.ONE_FILE_PER_CHANNEL: 'one-per-channel',
}


def list_source_files(src_dir: str) -> list[str]:
    """Natural-sorted regular files of ``src_dir`` as the setup dialog lists
    them (``myutils.listdir``: no dot files, ``desktop.ini``, ``recovery``,
    ``*.new.npz``; no folders).

    Raises
    ------
    SourceError
        ``src_dir`` is not an existing folder.
    """
    if not os.path.isdir(src_dir):
        raise errors.SourceError(
            f'The source folder does not exist: {src_dir}'
        )
    return [
        filename for filename in natsorted(myutils.listdir(src_dir))
        if os.path.isfile(os.path.join(src_dir, filename))
    ]


def normalized_extension(filename: str) -> str:
    """Lower-case extension, compound for '.ome.tif', '.ome.tiff' and
    '.ome.zarr' (e.g. 'a.OME.TIF' gives '.ome.tif', 'b.CZI' gives '.czi')."""
    lower = filename.lower()
    for compound in COMPOUND_EXTENSIONS:
        if lower.endswith(compound):
            return compound
    return os.path.splitext(lower)[1]


def extension_counts(files: Sequence[str]) -> collections.Counter:
    """Counter of ``os.path.splitext(f)[1]`` (case kept, as the setup
    dialog counts extensions)."""
    return collections.Counter(os.path.splitext(f)[1] for f in files)


def find_invalid_filenames(files: Sequence[str]) -> list[str]:
    """Files whose name the setup dialog rejects
    (``acdc_regex.is_alphanumeric_filename(f, allowed=('.ome.tif',))``)."""
    return [
        f for f in files
        if not acdc_regex.is_alphanumeric_filename(
            f, allowed=ALLOWED_DOUBLE_EXTENSIONS
        )
    ]


def planned_valid_filename(
        filename: str, replacement: Literal['_', '-']
    ) -> str:
    """``filename`` with each run of invalid characters that
    ``io._validate_filename`` reports (spaces included) replaced by
    ``replacement``, one ``str.replace`` per run: the rule of the setup
    dialog's rename buttons (main's
    ``io.rename_files_replace_invalid_chars``). As on main, a leading run
    is kept ('+a.tif' is not renamed) and overlapping runs can keep a
    character ('a +b+c.tif' can become 'a _b_c.tif'). Renames nothing."""
    new_filename = filename
    for chars in io._validate_filename(filename, is_path=False):
        new_filename = new_filename.replace(chars, replacement)
    return new_filename


def plan_invalid_name_renames(
        src_dir: str, files: Sequence[str], replacement: Literal['_', '-']
    ) -> dict[str, str]:
    """Renames that replace the invalid characters of ``files``
    (``planned_valid_filename``), as the setup dialog's rename buttons and
    ``acdc-data convert --rename-invalid`` do: old name -> new name of every
    file whose name changes, in the order of ``files``. Renames nothing.

    Raises
    ------
    SourceError
        Listing every collision: a new name is already taken in ``src_dir``,
        or two files would get the same name (ignoring case; two names that
        differ only in case collide only when one of them comes from a
        rename).
    """
    renames: dict[str, str] = {}
    for f in files:
        new_name = planned_valid_filename(f, replacement)
        if new_name != f:
            renames[f] = new_name
    _check_rename_collisions(src_dir, files, renames)
    return renames


def supported_extensions() -> frozenset[str]:
    """Lower-cased extensions supported by the installed BioIO plugins and
    by Bio-Formats (``_utils.get_supported_image_extensions``)."""
    from ..acdc_bioio_bioformats import _utils
    return frozenset(
        ext.strip().lower()
        for ext in _utils.get_supported_image_extensions()
    )


def find_unsupported_files(
        files: Sequence[str], supported: Collection[str] | None = None
    ) -> list[str]:
    """Files whose extension is not supported (case-insensitive). A
    compound extension is accepted when either its last part ('.tif' of
    '.ome.tif') or the compound itself is supported."""
    if supported is None:
        supported = supported_extensions()
    supported = {ext.strip().lower() for ext in supported}
    unsupported = []
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in supported or normalized_extension(f) in supported:
            continue
        unsupported.append(f)
    return unsupported


def one_file_per_extension(files: Sequence[str]) -> list[str]:
    """First file of every distinct ``normalized_extension`` in the given
    order (``files[0]`` always comes first). Used to check/install the
    reader plugin of every extension before conversion."""
    seen = set()
    selected = []
    for f in files:
        ext = normalized_extension(f)
        if ext in seen:
            continue
        seen.add(ext)
        selected.append(f)
    return selected


@dataclass(frozen=True)
class SourceSelection:
    """Source files selected by the command line options."""
    files: tuple[str, ...]          # selected ORIGINAL names, natsorted
    renames: Mapping[str, str]      # original -> planned name (rename_invalid)
    ignored: tuple[str, ...]        # files in src not selected


def _format_counts(counts: collections.Counter) -> str:
    return ', '.join(
        f'{ext or "(no extension)"}: {n}' for ext, n in counts.most_common()
    )


def _bullet_list(items: Sequence[str]) -> str:
    return '\n'.join(f'  - {item}' for item in items)


def _select_by_extension(
        listing: Sequence[str], ext: str | None
    ) -> list[str]:
    if ext is None:
        return list(listing)
    ext = ext.strip().lower()
    if not ext.startswith('.'):
        ext = f'.{ext}'
    return [f for f in listing if f.lower().endswith(ext)]


def select_source_files(
        src_dir: str, layout: RawLayout, *,
        files: Sequence[str] | None = None, ext: str | None = None,
        rename_invalid: Literal['_', '-'] | None = None
    ) -> SourceSelection:
    """Select the raw files of ``src_dir`` like the setup dialog does, but
    without prompts (command line). Read-only.

    Parameters
    ----------
    src_dir : str
        Source folder.
    layout : RawLayout
        Raw data structure; single-multi-pos needs exactly one file.
    files : sequence of str, optional
        Only these file names (``--files``).
    ext : str, optional
        Only files with this extension (case-insensitive, ``--ext``);
        required when the folder has mixed extensions and ``files`` is not
        given.
    rename_invalid : '_' or '-', optional
        Plan renaming the files (as the dialog's rename buttons do) when
        some names are invalid; without it invalid names are an error.

    Raises
    ------
    SourceError
        Source folder missing or empty; a requested file is missing or does
        not have ``ext``; mixed extensions without ``files``/``ext``; no
        file with ``ext``; single-multi-pos with a number of files other
        than one; unsupported files; invalid names without
        ``rename_invalid``; planned renames collide.
    """
    listing = list_source_files(src_dir)
    if not listing:
        raise errors.SourceError(
            f'The source folder does not contain any file: {src_dir}'
        )

    if files is not None:
        requested = natsorted(dict.fromkeys(files))
        missing = [f for f in requested if f not in listing]
        if missing:
            raise errors.SourceError(
                'These files were not found in the source folder '
                f'{src_dir}:\n{_bullet_list(missing)}',
                hint='--files takes file names relative to the input folder.'
            )
        selected = _select_by_extension(requested, ext)
        wrong_ext = [f for f in requested if f not in selected]
        if wrong_ext:
            raise errors.SourceError(
                f'These files do not have the extension "{ext}":\n'
                f'{_bullet_list(wrong_ext)}'
            )
    elif ext is not None:
        selected = _select_by_extension(listing, ext)
        if not selected:
            raise errors.SourceError(
                f'No file with the extension "{ext}" in {src_dir} '
                f'(found {_format_counts(extension_counts(listing))}).'
            )
    else:
        counts = extension_counts(listing)
        if len(counts) > 1:
            raise errors.SourceError(
                'The source folder contains files with different '
                f'extensions ({_format_counts(counts)}).',
                hint='Select the files with --ext or --files.'
            )
        selected = list(listing)

    if layout == RawLayout.SINGLE_FILE_MULTI_POS and len(selected) != 1:
        raise errors.SourceError(
            'The layout single-multi-pos needs exactly one file, '
            f'{len(selected)} were selected:\n{_bullet_list(selected)}',
            hint='Select the file with --files.'
        )

    unsupported = find_unsupported_files(selected)
    if unsupported:
        raise errors.SourceError(
            'These files have an extension that is not supported:\n'
            f'{_bullet_list(unsupported)}',
            hint='Select other files with --ext or --files.'
        )

    renames: dict[str, str] = {}
    invalid = find_invalid_filenames(selected)
    if invalid and rename_invalid is None:
        raise errors.SourceError(
            'These file names contain invalid characters (valid are '
            'letters, numbers, spaces, "_", "-" and one "."):\n'
            f'{_bullet_list(invalid)}',
            hint=(
                'Rename the files, or re-run with --rename-invalid '
                'underscore (or dash).'
            )
        )
    if invalid:
        # Like the setup dialog: once a name is invalid, every selected
        # file is renamed with the same rule
        renames = plan_invalid_name_renames(src_dir, selected, rename_invalid)

    selected_set = set(selected)
    return SourceSelection(
        files=tuple(selected),
        renames=renames,
        ignored=tuple(f for f in listing if f not in selected_set),
    )


def name_collisions(
        files: Sequence[str], final_names: Mapping[str, str], *,
        moved_to: str | None) -> list[str]:
    """One problem per colliding pair of ``files`` (``final_names``: file
    -> final name, the file itself when it is not renamed), each pair once,
    in the order of the second file of the pair. Equal final names collide
    ('A and B would both be named X'). A name that comes from a rename
    collides with every other final name that is equal to it ignoring case
    ('A and B would be named X and x, which differ only in case'). Two
    unchanged names that differ only in case already coexist in the source
    folder (case-sensitive file system), so they collide only when both
    files are moved into ``moved_to``, whose file system may ignore case.

    Used by ``plan_invalid_name_renames`` (the setup dialog's rename
    buttons and ``acdc-data convert --rename-invalid``) and by
    ``planning.plan_source_mutations``, so both report the same pairs with
    the same words."""
    problems = []
    by_key: dict[str, list[str]] = {}
    for filename in files:
        name = final_names[filename]
        others = by_key.setdefault(name.casefold(), [])
        for other in others:
            other_name = final_names[other]
            if other_name == name:
                problems.append(
                    f'{other} and {filename} would both be named {name}'
                )
            elif other_name != other or name != filename:
                problems.append(
                    f'{other} and {filename} would be named {other_name} '
                    f'and {name}, which differ only in case'
                )
            elif moved_to is not None:
                problems.append(
                    f'{other} and {filename} differ only in case and cannot '
                    f'both be moved to {moved_to} (its file system may '
                    'ignore case)'
                )
        others.append(filename)
    return problems


def _check_rename_collisions(
        src_dir: str, selected: Sequence[str], renames: Mapping[str, str]
    ) -> None:
    final_names = {f: renames.get(f, f) for f in selected}
    collisions = []
    for f in selected:
        final = final_names[f]
        if final != f and os.path.lexists(os.path.join(src_dir, final)):
            collisions.append(f'{f} -> {final}: {final} already exists')
    collisions.extend(name_collisions(selected, final_names, moved_to=None))
    if collisions:
        raise errors.SourceError(
            'Renaming the files with invalid characters would overwrite '
            f'files:\n{_bullet_list(collisions)}',
            hint='Rename the files yourself.'
        )


@dataclass(frozen=True)
class OnePerChannelIndex:
    """Positions and channels parsed from one-file-per-channel names
    ``<basename><position>_<channel>.<ext>``."""
    basename: str                           # e.g. 'ASY015_' (legacy)
    position_ids: tuple[int, ...]           # sorted ints ('01' -> 1)
    position_tokens: Mapping[int, str]      # id -> digits as written ('01')
    channel_tokens: tuple[str, ...]         # RAW tokens, natsorted
    files: Mapping[tuple[int, str], str]    # (id, raw token) -> filename

    def missing(
            self, tokens: Sequence[str] | None = None
        ) -> list[tuple[int, str]]:
        """(position id, token) pairs without a file, for ``tokens``
        (default: every channel token), in position then token order."""
        tokens = self.channel_tokens if tokens is None else tokens
        return [
            (pos_id, token)
            for pos_id in self.position_ids
            for token in tokens
            if (pos_id, token) not in self.files
        ]

    def to_dict(self) -> dict:
        """JSON-able dict (inverse of ``from_dict``)."""
        return {
            'basename': self.basename,
            'position_ids': list(self.position_ids),
            'position_tokens': {
                str(pos_id): token
                for pos_id, token in self.position_tokens.items()
            },
            'channel_tokens': list(self.channel_tokens),
            'files': [
                [pos_id, token, filename]
                for (pos_id, token), filename in self.files.items()
            ],
        }

    @classmethod
    def from_dict(cls, d: Mapping) -> 'OnePerChannelIndex':
        """Index from ``to_dict()`` output."""
        return cls(
            basename=d['basename'],
            position_ids=tuple(int(pos_id) for pos_id in d['position_ids']),
            position_tokens={
                int(pos_id): token
                for pos_id, token in d['position_tokens'].items()
            },
            channel_tokens=tuple(d['channel_tokens']),
            files={
                (int(pos_id), token): filename
                for pos_id, token, filename in d['files']
            },
        )


def _findalliter(text: str) -> list[list[tuple[str, str]]]:
    """Port of ``myutils.findalliter``: repeated ``(\\d+)_(.+)`` matches,
    each on the remainder of the previous one; the last entry is ``[]``."""
    match = _POSITION_CHANNEL_RE.findall(text)
    matches = [match]
    while match:
        match = _POSITION_CHANNEL_RE.findall(match[0][1])
        matches.append(match)
    return matches


def split_position_channel(stem: str) -> tuple[str, str] | None:
    """Deepest ``(\\d+)_(.+)`` match of ``stem``: (position digits as
    written, channel token), or None without a match.

    'ASY015_1_GFP' gives ('1', 'GFP'); 'exp_01_mNeon-Green' gives
    ('01', 'mNeon-Green').
    """
    matches = _findalliter(stem)
    if len(matches) <= 1:
        return None
    digits, token = matches[-2][0]
    return digits, token


def parse_one_per_channel_filenames(
        filenames: Sequence[str]
    ) -> OnePerChannelIndex:
    """Parse one-file-per-channel names.

    For each file, ``stem = naming.raw_file_stem(f)`` ('.ome' removed),
    (digits, token) = ``split_position_channel(stem)`` and the prefix is the
    stem without the suffix ``f'{digits}_{token}'``; the basename is
    ``myutils.getBasename`` of the prefixes (legacy). Missing (position,
    channel) pairs are not an error here (see ``OnePerChannelIndex.missing``).

    Raises
    ------
    FilenamePatternError
        No file, a name without the pattern, an empty basename, or two files
        for the same (position, channel).
    """
    if not filenames:
        raise errors.FilenamePatternError(
            'No files to parse for the one-file-per-channel layout.'
        )
    hint = (
        'One-file-per-channel names must look like '
        '<basename><position number>_<channel>.<ext>, '
        'e.g. ASY015_1_GFP.tif.'
    )
    no_pattern = []
    parsed = []
    for filename in filenames:
        stem = naming.raw_file_stem(filename)
        match = split_position_channel(stem)
        if match is None:
            no_pattern.append(filename)
            continue
        digits, token = match
        prefix = stem[:len(stem) - len(f'{digits}_{token}')]
        parsed.append((filename, prefix, digits, token))
    if no_pattern:
        raise errors.FilenamePatternError(
            'These file names do not contain a position number followed by '
            f'"_" and a channel name:\n{_bullet_list(no_pattern)}',
            files=no_pattern, hint=hint, details='\n'.join(no_pattern)
        )

    basename = myutils.getBasename([prefix for _, prefix, _, _ in parsed])
    if not basename:
        parsed_names = [filename for filename, _, _, _ in parsed]
        raise errors.FilenamePatternError(
            'The file names do not share a common basename before the '
            'position number.',
            files=parsed_names, hint=hint, details='\n'.join(parsed_names)
        )

    files: dict[tuple[int, str], str] = {}
    position_tokens: dict[int, str] = {}
    duplicates: list[tuple[str, str]] = []
    for filename, _, digits, token in parsed:
        key = (int(digits), token)
        if key in files:
            duplicates.append((files[key], filename))
            continue
        files[key] = filename
        position_tokens.setdefault(int(digits), digits)
    if duplicates:
        pairs = [f'{first} and {second}' for first, second in duplicates]
        raise errors.FilenamePatternError(
            'Several files have the same position number and channel:\n'
            f'{_bullet_list(pairs)}',
            files=[f for pair in duplicates for f in pair],
            hint=hint, details='\n'.join(pairs)
        )

    return OnePerChannelIndex(
        basename=basename,
        position_ids=tuple(sorted(position_tokens)),
        position_tokens=position_tokens,
        channel_tokens=tuple(natsorted({token for _, token in files})),
        files=files,
    )


@dataclass(frozen=True)
class MetadataUnit:
    """One metadata read and one metadata confirmation."""
    index: int
    label: str                  # filename (dialog title); layout 2: basename
    metadata_file: str          # ORIGINAL filename whose metadata is read
    files: tuple[str, ...]      # raw files covered
    file_ordinal: int | None    # layout 1: 1-based ordinal (= input_id)


def metadata_units(
        layout: RawLayout, files: Sequence[str],
        one_per_channel: OnePerChannelIndex | None
    ) -> list[MetadataUnit]:
    """Metadata units of a conversion: layout 0 one unit (the single file);
    layout 1 one unit per file; layout 2 one unit for all files, reading the
    metadata of the first natsorted file (legacy). ``one_per_channel``
    (layout 2) is parsed from ``files`` when None.

    Raises
    ------
    SourceError
        Layout 0 with a number of files other than one, or no file.
    """
    if not files:
        raise errors.SourceError('No raw files to convert.')
    if layout == RawLayout.SINGLE_FILE_MULTI_POS:
        if len(files) != 1:
            raise errors.SourceError(
                'The layout single-multi-pos needs exactly one file, '
                f'got {len(files)}.'
            )
        return [MetadataUnit(0, files[0], files[0], (files[0],), None)]
    if layout == RawLayout.ONE_FILE_PER_POS:
        return [
            MetadataUnit(i, f, f, (f,), i + 1) for i, f in enumerate(files)
        ]
    if one_per_channel is None:
        one_per_channel = parse_one_per_channel_filenames(files)
    first_file = natsorted(files)[0]
    return [MetadataUnit(
        0, one_per_channel.basename, first_file, tuple(files), None
    )]
