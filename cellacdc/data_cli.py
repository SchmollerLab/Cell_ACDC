"""``acdc-data``: Cell-ACDC data-structure tools for the command line.

Commands (the same core as the GUI, ``cellacdc.acdc_datastruct``):

- ``metadata``: read the metadata of one raw microscopy file (in a worker
  process) and print it as text, JSON or the metadata.csv rows.
- ``convert``: convert raw microscopy files into ``Position_n/Images``
  folders, like module 0 of the GUI (``convert.convert`` with the 'upfront'
  schedule: read every file, plan and check everything, then write).
- ``restructure``: organize loose TIFF files into ``Position_n/Images``
  folders, like "Re-structure image files" (``restructure`` core, in this
  process).

This module only parses the command line, wires the core together and owns
the process conventions: standard output carries only the result (in the
``--format`` of the command), logs go to standard error and to a log file,
and the exit code is 0 (success), 1 (the run failed), 2 (invalid command
line, or options that do not fit the input files: nothing was changed) or
130 (interrupted).

Standard library only at module level: the core is imported inside the
functions that use it (the parser takes the names of the layouts and units
from it, the command handlers run it), so an error of the core while a
command runs is reported like any other error.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import io
import json
import locale
import logging
import math
import os
import shlex
import subprocess
import sys
import traceback
import uuid
from typing import TYPE_CHECKING, Callable, Sequence

if TYPE_CHECKING:
    from .acdc_datastruct.backends import ConvertBackend

PROG = 'acdc-data'
DOCS_PAGE = 'https://cell-acdc.readthedocs.io/en/latest/data-cli.html'
METADATA_SCHEMA = 'acdc-data/metadata/1'
CONVERT_SCHEMA = 'acdc-data/convert/1'
RESTRUCTURE_SCHEMA = 'acdc-data/restructure/1'

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

LOGGER_NAME = 'cellacdc.data_cli'
CORE_LOGGER_NAME = 'cellacdc.acdc_datastruct'
"""Logger of the shared core; it gets the command line handlers too."""

RENAME_REPLACEMENTS = {'underscore': '_', 'dash': '-'}

_STDOUT_FD = 1
_STDERR_FD = 2
_LOG_LEVELS = {
    'DEBUG': logging.DEBUG, 'INFO': logging.INFO,
    'WARNING': logging.WARNING, 'ERROR': logging.ERROR,
}

logger = logging.getLogger(LOGGER_NAME)


# =============================================================== parser
_REMOVED_SEE = (
    'see "Differences from pull request #1109" in the acdc-data '
    f'documentation, {DOCS_PAGE}'
)
_SIZES_HINT = (
    'sizes are read from the file and cannot be overridden; use '
    '--time-range, --positions, --save-channels to select data'
)
_TRUST_HINT = (
    'file metadata is always used; override single values with the '
    'metadata flags or --metadata-csv'
)
_CONVERT_BASENAME_HINT = (
    'output names follow the GUI (<file stem>_s<NN>_); the one-per-channel '
    'basename is detected from the file names'
)
_RESTRUCTURE_REMOVED = {
    '--layout': (
        'use the dimension flags (--channel, --channel-axis, --time-marker, '
        '--time-axis, --z-marker, --z-axis) or --recipe'
    ),
    '--action': (
        'moving the source files is not supported: the TIFF files are only '
        'read and new files are written'
    ),
    '--segm-folder': (
        'importing segmentation masks is not supported'
    ),
    '--channels': 'repeat --channel NAME[=ID] (one per channel)',
}


class _RemovedOption(argparse.Action):
    """Hidden option of the first acdc-data version (pull request #1109):
    any use is a usage error (exit code 2) that names the replacement."""
    def __init__(self, option_strings, dest, *, hint: str, nargs='?',
                 **kwargs) -> None:
        super().__init__(
            option_strings, dest, nargs=nargs, default=argparse.SUPPRESS,
            help=argparse.SUPPRESS, **kwargs
        )
        self.hint = hint

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(
            f'{option_string} was removed: {self.hint} ({_REMOVED_SEE}).'
        )


class _VersionRequested(Exception):
    """``--version`` was given (handled by ``main``)."""


class _VersionAction(argparse.Action):
    """``--version``: print the Cell-ACDC version (needs ``import
    cellacdc``, which ``main`` does with standard output guarded)."""
    def __init__(self, option_strings, dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS, help=None) -> None:
        super().__init__(
            option_strings, dest, nargs=0, default=default, help=help
        )

    def __call__(self, parser, namespace, values, option_string=None):
        raise _VersionRequested()


def _convert_stdout_format(value: str) -> str:
    """``convert --format``: 'tif'/'h5' of the first version get the
    ``--image-format`` hint instead of argparse's "invalid choice"."""
    if value in ('tif', 'h5'):
        raise argparse.ArgumentTypeError(
            f'"--format {value}" was removed: use --image-format {value}; '
            '--format now selects the format of standard output (text or '
            f'json) ({_REMOVED_SEE})'
        )
    return value


def _new_parser(subparsers, name: str, *, help: str,
                description: str) -> argparse.ArgumentParser:
    return subparsers.add_parser(
        name, help=help, description=description, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def add_common_options(parser: argparse.ArgumentParser) -> None:
    """Add the options every command accepts (``-v``, ``-q``,
    ``--log-file``, ``--no-log-file``, ``--yes``, ``--debug``)."""
    group = parser.add_argument_group('common options')
    verbosity = group.add_mutually_exclusive_group()
    verbosity.add_argument(
        '-v', '--verbose', action='store_true',
        help='Show debug messages on stderr.'
    )
    verbosity.add_argument(
        '-q', '--quiet', action='store_true',
        help='Show only warnings and errors on stderr; no progress bar.'
    )
    log_file = group.add_mutually_exclusive_group()
    log_file.add_argument(
        '--log-file', metavar='PATH',
        help=(
            'Write the log to PATH (default: <cellacdc logs folder>/'
            '<YYYY-mm-dd_HH-MM-SS>_acdc-data-<command>_<id>.log).'
        )
    )
    log_file.add_argument(
        '--no-log-file', action='store_true',
        help='Do not write a log file.'
    )
    group.add_argument(
        '--yes', action='store_true',
        help=(
            'Install missing reader packages (bioio plugins) without asking '
            '(metadata and convert).'
        )
    )
    group.add_argument(
        '--debug', action='store_true',
        help="Enable Cell-ACDC debug mode (sets config.parser_args['debug'])."
    )


def _add_metadata_parser(subparsers) -> None:
    parser = _new_parser(
        subparsers, 'metadata',
        help='Read and print the metadata of one raw microscopy file.',
        description=(
            'Read the metadata of one raw microscopy file in a worker '
            'process, apply the same defaults as the GUI and print the '
            'values that convert would use.'
        ),
    )
    parser.add_argument(
        '-i', '--input', required=True, metavar='INPUT',
        help=(
            'Raw microscopy file, or a folder containing exactly one '
            'supported file.'
        )
    )
    parser.add_argument(
        '--format', choices=('text', 'json', 'csv'), default='text',
        help=(
            'Output on stdout (default: text). text: key: value lines. '
            'json: resolved values, defaulted fields, warnings and raw '
            'reader values. csv: the Description,values rows that convert '
            'would write without overrides (usable with convert '
            '--metadata-csv).'
        )
    )
    parser.add_argument(
        '-o', '--output', metavar='DIR',
        help=(
            'Also write <basename>metadata.csv and <basename>metadataXML.txt '
            'into DIR (any --format).'
        )
    )
    parser.add_argument(
        '--basename', metavar='TEXT',
        help=(
            'Basename of the files written with --output and of the csv '
            'basename row (default: <file stem>_, dots in the stem become '
            '_).'
        )
    )
    parser.add_argument(
        '--include-xml', action='store_true',
        help='json only: include the full metadataXML text.'
    )
    add_common_options(parser)


def _add_convert_parser(subparsers) -> None:
    from .acdc_datastruct import sources, units
    parser = _new_parser(
        subparsers, 'convert',
        help='Convert raw microscopy files into Position_n folders.',
        description=(
            'Convert the raw microscopy files of SRC into Position_n/Images '
            'folders in DST, like module 0 of the GUI ("Create data '
            'structure from microscopy/image file(s)"). One worker process '
            'reads the metadata of all files, every position is planned, a '
            'second worker process checks every raw file against the plan, '
            'then a third one writes all positions. When a check fails '
            'nothing is changed (exit code 2).'
        ),
    )
    inputs = parser.add_argument_group('input options')
    inputs.add_argument(
        '-i', '--input', required=True, metavar='SRC',
        help='Folder with the raw microscopy files.'
    )
    inputs.add_argument(
        '-o', '--output', required=True, metavar='DST',
        help=(
            'Experiment folder that receives Position_N folders (created '
            'when writing starts).'
        )
    )
    inputs.add_argument(
        '--layout', required=True,
        choices=tuple(layout.cli_name for layout in sources.RawLayout),
        help=(
            'single-multi-pos: one file with several positions (scenes). '
            'one-per-pos: one file per position. one-per-channel: one file '
            'per channel, named <basename><N>_<channel>.'
        )
    )
    inputs.add_argument(
        '--files', nargs='+', metavar='NAME',
        help=(
            'Only use these file names (in natural sort order). Required '
            'for single-multi-pos when SRC has several files.'
        )
    )
    inputs.add_argument(
        '--ext', metavar='EXT',
        help=(
            'Only use files with this extension (case-insensitive, e.g. '
            '.czi or .ome.tif). Required when SRC has mixed extensions and '
            '--files is not given.'
        )
    )
    inputs.add_argument(
        '--rename-invalid', choices=tuple(RENAME_REPLACEMENTS),
        help=(
            'Rename files whose names contain invalid characters (as the GUI '
            'offers), replacing them with _ or -. Without it such files are '
            'an error. Renaming happens right before writing.'
        )
    )
    inputs.add_argument(
        '--other-files', choices=('leave', 'move', 'copy'),
        help=(
            'one-per-channel only: what to do with other files of each '
            'position, e.g. notes (default: leave). Error with the other '
            'layouts.'
        )
    )

    outputs = parser.add_argument_group('output options')
    outputs.add_argument(
        '--on-existing', choices=('error', 'overwrite', 'add', 'create-new'),
        default='error',
        help=(
            'When DST already has Position folders (default: error). error: '
            'stop and list them. overwrite: DELETE Position_N/Images '
            '(including segmentation and acdc_output files) of every '
            'position written. add: write into existing folders and merge '
            'metadata.csv. create-new: number new positions after the '
            'highest existing one.'
        )
    )
    outputs.add_argument(
        '--move-raw', action=argparse.BooleanOptionalAction, default=None,
        help=(
            'Move the raw files into DST/raw_microscopy_files right before '
            'writing (GUI behaviour; symlink files then point to the moved '
            'files). Default: off.'
        )
    )
    outputs.add_argument(
        '--use-symlinks', action='store_true',
        help=(
            'Write <basename>symlink.ini files pointing to the raw files '
            'instead of image files. Not combinable with --image-format h5.'
        )
    )
    outputs.add_argument(
        '--image-format', choices=('tif', 'h5'), default='tif',
        help=(
            'Output image format (default: tif; renamed from the old '
            '--format).'
        )
    )
    outputs.add_argument(
        '--lazy-load', action=argparse.BooleanOptionalAction, default=None,
        help=(
            '--no-lazy-load loads each position into RAM at once (GUI "Load '
            'entire field of view in RAM"). Default: lazy.'
        )
    )

    values = parser.add_argument_group(
        'metadata options (default: the value read from the file)'
    )
    values.add_argument(
        '--lens-na', type=float, metavar='F',
        help='Objective numerical aperture (file, else 1.4).'
    )
    values.add_argument(
        '--time-increment', type=float, metavar='F',
        help='Frame interval, in --time-unit (file, else 1.0 s).'
    )
    values.add_argument(
        '--time-unit', choices=units.CLI_TIME_UNITS,
        help='Unit of --time-increment (default: s).'
    )
    for axis in ('x', 'y', 'z'):
        values.add_argument(
            f'--physical-size-{axis}', type=float, metavar='F',
            help=(
                f'Pixel size along {axis.upper()}, in --physical-size-unit '
                '(file, else 1.0 um).'
            )
        )
    values.add_argument(
        '--physical-size-unit', choices=units.CLI_LENGTH_UNITS,
        help='Unit of --physical-size-x/-y/-z (default: um).'
    )
    values.add_argument(
        '--channels', metavar='A,B,...',
        help=(
            'Output channel names, one per channel (default: the file '
            'names; one-per-channel: the file name tokens). Non-alphanumeric '
            'characters become _. Must be non-empty and unique after that.'
        )
    )
    values.add_argument(
        '--em-wavelens', metavar='NM,NM,...',
        help=(
            'Emission wavelengths, one per channel (file, else 0.0; '
            'one-per-channel: 500.0).'
        )
    )
    values.add_argument(
        '--save-channels', metavar='A,B,...',
        help='Channel names (after --channels) to save (default: all).'
    )
    values.add_argument(
        '--positions', metavar='LIST',
        help=(
            'Input positions to save (default: all): N, Position_N or A-B, '
            'comma-separated. single-multi-pos: scene number; one-per-pos: '
            'file order (1 = first file); one-per-channel: number in the '
            'file name.'
        )
    )
    values.add_argument(
        '--time-range', metavar='START:END',
        help=(
            'Frames to save (default: all), 1-based and inclusive like the '
            'GUI (e.g. 5:20, 5:, :20, 7).'
        )
    )
    values.add_argument(
        '--add-image-name', action='store_true',
        help=(
            'Add the series image name (OME-TIFF) to every output file name. '
            'Error (exit 2) when a selected file has no image name.'
        )
    )
    values.add_argument(
        '--metadata-csv', metavar='PATH',
        help=(
            'Description,values overrides (as written by metadata --format '
            'csv). Explicit flags win over the CSV, the CSV wins over the '
            'file.'
        )
    )

    run = parser.add_argument_group('run options')
    run.add_argument(
        '--dry-run', action='store_true',
        help=(
            'Read the metadata, check the raw files and print the plan '
            '(positions, names, files to move, rename or delete); change '
            'nothing.'
        )
    )
    run.add_argument(
        '--format', type=_convert_stdout_format, choices=('text', 'json'),
        default='text',
        help='Format of the plan/result summary on stdout (default: text).'
    )
    run.add_argument(
        '--keep-temp', action='store_true',
        help='Keep the worker job folders (debugging).'
    )

    for size in ('t', 'z', 'c', 's'):
        parser.add_argument(
            f'--size-{size}', action=_RemovedOption, hint=_SIZES_HINT
        )
    for flag in ('--trust-metadata', '--no-trust-metadata'):
        parser.add_argument(
            flag, action=_RemovedOption, hint=_TRUST_HINT, nargs=0
        )
    parser.add_argument(
        '--basename', action=_RemovedOption, hint=_CONVERT_BASENAME_HINT
    )
    add_common_options(parser)


def _add_restructure_parser(subparsers) -> None:
    from .acdc_datastruct import units
    parser = _new_parser(
        subparsers, 'restructure',
        help='Organize loose TIFF files into Position_n folders.',
        description=(
            'Organize the loose TIFF files of SRC into Position_n/Images '
            'folders in DST, like "Re-structure image files" in the GUI. '
            'Every check runs before anything is written (exit code 2 when '
            'one fails).'
        ),
    )
    parser.add_argument(
        '-i', '--input', required=True, metavar='SRC',
        help='TIFF folder (not recursive).'
    )
    parser.add_argument(
        '-o', '--output', required=True, metavar='DST',
        help='Destination experiment folder.'
    )
    parser.add_argument(
        '--recipe', metavar='SPEC',
        help=(
            'Recipe JSON saved by the GUI "Re-structure image files" dialog: '
            "a file path, 'last' (last GUI setup) or NAME (saved recipe). "
            'Flags below replace the matching recipe entries.'
        )
    )
    dimensions = parser.add_argument_group('dimension options')
    dimensions.add_argument(
        '--channel', action='append', metavar='NAME[=ID]',
        help=(
            'Channel name and, for one-file-per-channel input, the text '
            'identifying it in file names. Repeat for every channel '
            '(default: one channel, Channel_1). If one --channel has =ID, '
            'all must.'
        )
    )
    dimensions.add_argument(
        '--channel-axis', type=int, metavar='N',
        help=(
            'Channels are axis N inside each TIFF (give one --channel per '
            'channel, without =ID).'
        )
    )
    timepoints = dimensions.add_mutually_exclusive_group()
    timepoints.add_argument(
        '--time-marker', metavar='TEXT',
        help=(
            'Timepoints: one file per frame, TEXT followed by the frame '
            'number in the file names (e.g. _t). Default: not present.'
        )
    )
    timepoints.add_argument(
        '--time-axis', type=int, metavar='N',
        help='Timepoints are axis N inside each TIFF.'
    )
    z_slices = dimensions.add_mutually_exclusive_group()
    z_slices.add_argument(
        '--z-marker', metavar='TEXT',
        help=(
            'Z-slices: one file per slice, TEXT followed by the slice number '
            'in the file names. Default: not present.'
        )
    )
    z_slices.add_argument(
        '--z-axis', type=int, metavar='N',
        help='Z-slices are axis N inside each TIFF.'
    )
    parser.add_argument(
        '--basename', metavar='TEXT',
        help='Prefix of the output files (letters, digits, _ and - only).'
    )
    parser.add_argument(
        '--files', nargs='+', metavar='NAME',
        help='Only use these TIFF files of SRC (default: all TIFFs).'
    )
    parser.add_argument(
        '--on-existing', choices=('error', 'overwrite', 'create-new'),
        default='error',
        help=(
            'When the Position folders to write exist (default: error). '
            'overwrite DELETES Position_N/Images of every position written; '
            'create-new numbers after the highest existing one.'
        )
    )
    values = parser.add_argument_group(
        'metadata.csv options (default: the recipe, else 1.0)'
    )
    for axis in ('x', 'y', 'z'):
        values.add_argument(
            f'--physical-size-{axis}', type=float, metavar='UM',
            help=f'Pixel size along {axis.upper()} in um.'
        )
    values.add_argument(
        '--time-increment', type=float, metavar='V',
        help='Frame interval, in --time-unit.'
    )
    values.add_argument(
        '--time-unit', choices=units.CLI_TIME_UNITS,
        help='Unit of the frame interval (default: the recipe, else s).'
    )
    run = parser.add_argument_group('run options')
    run.add_argument(
        '--dry-run', action='store_true',
        help=(
            'Print identifier -> Position_N, files per channel, T/Z sizes '
            'and folders to delete; write nothing in SRC or DST '
            '(--save-recipe still writes the recipe file).'
        )
    )
    run.add_argument(
        '--save-recipe', metavar='SPEC',
        help=(
            'Write the effective recipe in the GUI format (a path, or NAME '
            'in the GUI recipes folder).'
        )
    )
    run.add_argument(
        '--format', choices=('text', 'json'), default='text',
        help='Format of the plan/result summary on stdout (default: text).'
    )
    for flag, hint in _RESTRUCTURE_REMOVED.items():
        parser.add_argument(flag, action=_RemovedOption, hint=hint)
    add_common_options(parser)


def build_parser(prog: str = PROG) -> argparse.ArgumentParser:
    """The ``acdc-data`` argument parser (subcommands ``metadata``,
    ``convert``, ``restructure``; common options after the subcommand)."""
    parser = argparse.ArgumentParser(
        prog=prog, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            'Cell-ACDC data-structure tools for the command line (no GUI '
            'needed). The commands run the same code as the GUI modules.'
        ),
        epilog=(
            f'Run "{prog} COMMAND -h" for the options of a command.\n\n'
            'Exit codes: 0 success; 1 the run failed; 2 invalid command line\n'
            'or options that do not fit the input files (source and\n'
            'destination unchanged); 130 interrupted.\n'
            'Standard output carries only the result (--format); logs go to\n'
            'standard error and to a log file.\n\n'
            f'Documentation: {DOCS_PAGE}'
        ),
    )
    parser.add_argument(
        '--version', action=_VersionAction,
        help='Print the Cell-ACDC version and exit.'
    )
    subparsers = parser.add_subparsers(
        dest='command', required=True, metavar='COMMAND'
    )
    _add_metadata_parser(subparsers)
    _add_convert_parser(subparsers)
    _add_restructure_parser(subparsers)
    return parser


# ================================================================= main
def main(argv: Sequence[str] | None = None, *, prog: str = PROG,
         backend_factory: Callable[[], ConvertBackend] | None = None) -> int:
    """Run ``acdc-data`` with ``argv`` (default ``sys.argv[1:]``) and return
    the exit code (never raises ``SystemExit``).

    ``backend_factory`` replaces the production worker backend of
    ``metadata`` and ``convert`` (test seam).
    """
    parser = build_parser(prog)
    try:
        args = parser.parse_args(argv)
    except _VersionRequested:
        return _print_version(prog)
    except SystemExit as exit_:
        return _system_exit_code(exit_)
    return _run(args, backend_factory)


run = main


def _system_exit_code(exit_: SystemExit) -> int:
    code = exit_.code
    if code is None:
        return EXIT_OK
    if isinstance(code, int):
        return code
    _write_stream(sys.stderr, f'{code}\n')
    return EXIT_FAILED


def _print_version(prog: str) -> int:
    guard = _StdoutGuard(_payload_encoding('text'))
    text, exit_code = '', EXIT_OK
    with guard:
        try:
            import cellacdc
            text = f'{prog} (Cell-ACDC) {cellacdc.__version__}\n'
        except Exception as err:
            _write_stream(sys.stderr, f'ERROR: {type(err).__name__}: {err}\n')
            exit_code = EXIT_FAILED
    guard.write(text)
    return exit_code


def _run(args: argparse.Namespace,
         backend_factory: Callable[[], ConvertBackend] | None) -> int:
    """Run the parsed command with standard output reserved for its
    payload."""
    _normalize_paths(args)
    guard = _StdoutGuard(_payload_encoding(args.format))
    exit_code, payload = EXIT_FAILED, ''
    try:
        with guard:
            session = _Session(args, backend_factory)
            try:
                command = _COMMANDS[args.command](session)
                exit_code, payload = command.execute()
            finally:
                session.close(exit_code)
    except KeyboardInterrupt:
        _write_stream(sys.stderr, 'ERROR: Interrupted (Ctrl+C).\n')
        exit_code, payload = EXIT_INTERRUPTED, ''
    guard.write(payload)
    return exit_code


def _absolute(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _is_path_spec(spec: str) -> bool:
    """A ``--save-recipe`` SPEC naming a file (not a recipe NAME)."""
    separators = {'/', os.sep, os.altsep} - {None}
    return (
        any(sep in spec for sep in separators) or spec.startswith('~')
        or spec.lower().endswith('.json')
    )


def _normalize_paths(args: argparse.Namespace) -> None:
    """Make every path option absolute (``~`` expanded) before any request
    is built: the worker children run in a run folder of their own, and
    symlink.ini files store absolute source paths."""
    for name in ('input', 'output', 'metadata_csv', 'log_file'):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, _absolute(value))
    recipe = getattr(args, 'recipe', None)
    if recipe is not None and os.path.isfile(os.path.expanduser(recipe)):
        args.recipe = _absolute(recipe)
    save_recipe = getattr(args, 'save_recipe', None)
    if save_recipe is not None and _is_path_spec(save_recipe):
        args.save_recipe = _absolute(save_recipe)


# ======================================================= process plumbing
def _payload_encoding(stdout_format: str) -> str:
    """ASCII for JSON (written with ``ensure_ascii``); UTF-8 for CSV, the
    encoding of metadata.csv files and of ``convert --metadata-csv`` (so
    ``metadata --format csv > file`` round-trips also when standard output
    is a cp1252 file or pipe); else the encoding of standard output
    (``PYTHONIOENCODING`` and UTF-8 mode are honoured)."""
    if stdout_format == 'json':
        return 'ascii'
    if stdout_format == 'csv':
        return 'utf-8'
    encoding = getattr(sys.stdout, 'encoding', None)
    return encoding or locale.getpreferredencoding(False)


def _fileno(stream) -> int | None:
    try:
        return stream.fileno()
    except (AttributeError, OSError, ValueError):
        return None


def _flush(stream) -> None:
    try:
        stream.flush()
    except (AttributeError, OSError, ValueError):
        pass


def _write_stream(stream, text: str) -> None:
    if stream is None:
        return
    try:
        stream.write(text)
        stream.flush()
    except (OSError, ValueError):
        pass


class _StdoutGuard:
    """Standard output carries only the payload.

    While the command runs, file descriptor 1 points to standard error
    (``os.dup2``: catches C-level writes and child processes) and
    ``sys.stdout`` is ``sys.stderr`` (prints, ``printl``, prompts); the
    payload goes, at the end, to a duplicate of the original descriptor 1
    opened with ``encoding``, ``errors='backslashreplace'`` and
    ``newline=''`` (no ``\\r\\r\\n`` on Windows). Without a usable
    ``sys.stdout.fileno()`` (e.g. pytest's ``capsys``) only the Python-level
    redirect is used and the payload goes to the original ``sys.stdout``.
    """
    def __init__(self, encoding: str) -> None:
        self.encoding = encoding
        self._stdout = None
        self._payload = None
        self._saved_fd: int | None = None
        self._redirect = None

    def __enter__(self) -> '_StdoutGuard':
        self._stdout = sys.stdout
        if _fileno(sys.stdout) is not None:
            self._swap_descriptors()
        self._redirect = contextlib.redirect_stdout(sys.stderr)
        self._redirect.__enter__()
        return self

    def _swap_descriptors(self) -> None:
        _flush(sys.stdout)
        payload_fd = saved_fd = None
        try:
            payload_fd = os.dup(_STDOUT_FD)
            saved_fd = os.dup(_STDOUT_FD)
            os.dup2(_STDERR_FD, _STDOUT_FD)
        except OSError:
            for fd in (payload_fd, saved_fd):
                if fd is not None:
                    os.close(fd)
            return
        self._saved_fd = saved_fd
        self._payload = os.fdopen(
            payload_fd, 'w', encoding=self.encoding,
            errors='backslashreplace', newline=''
        )

    def __exit__(self, *exc_info) -> bool:
        self._redirect.__exit__(*exc_info)
        if self._saved_fd is not None:
            for stream in (sys.stderr, self._stdout, sys.__stdout__):
                _flush(stream)
            os.dup2(self._saved_fd, _STDOUT_FD)
            os.close(self._saved_fd)
            self._saved_fd = None
        return False

    def write(self, text: str) -> None:
        """Write the payload once, after the guarded block (an empty text
        just closes the payload stream)."""
        if self._payload is not None:
            try:
                self._payload.write(text)
                self._payload.flush()
            finally:
                self._payload.close()
                self._payload = None
        elif text:
            _write_stream(self._stdout, text)


class _Progress:
    """tqdm progress bar on standard error driven by the core's
    ``progress_init``/``progress_step`` (none with ``--quiet``; tqdm hides
    it when standard error is not a terminal)."""
    def __init__(self, stream, *, enabled: bool) -> None:
        self.stream = stream
        self.enabled = enabled
        self.bar = None

    def init(self, total: int) -> None:
        self.close()
        if not self.enabled or total < 1:
            return
        from tqdm import tqdm
        self.bar = tqdm(total=total, file=self.stream, disable=None)

    def step(self, increment: int = 1) -> None:
        if self.bar is not None:
            self.bar.update(increment)

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None


class _ConsoleFormatter(logging.Formatter):
    """Message only, with a 'WARNING: ' or 'ERROR: ' prefix."""
    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if record.levelno >= logging.ERROR:
            return f'ERROR: {text}'
        if record.levelno >= logging.WARNING:
            return f'WARNING: {text}'
        return text


class _ConsoleHandler(logging.StreamHandler):
    """Standard error handler that clears the progress bar while it
    writes."""
    def __init__(self, stream, progress: _Progress) -> None:
        super().__init__(stream)
        self.progress = progress

    def emit(self, record: logging.LogRecord) -> None:
        bar = self.progress.bar
        if bar is None:
            super().emit(record)
            return
        with type(bar).external_write_mode(file=self.stream):
            super().emit(record)


class _Session:
    """Logging, progress bar, backend and error reporting of one command
    run."""
    def __init__(
            self, args: argparse.Namespace,
            backend_factory: Callable[[], ConvertBackend] | None
        ) -> None:
        self.args = args
        self.backend_factory = backend_factory
        self.log_file: str | None = None
        self.stderr = sys.stderr
        self.progress = _Progress(self.stderr, enabled=not args.quiet)
        self._handlers: list[logging.Handler] = []
        self._loggers = [logger, logging.getLogger(CORE_LOGGER_NAME)]
        self._logger_state = [(lg.level, lg.propagate) for lg in self._loggers]
        logger.propagate = False
        for lg in self._loggers:
            lg.setLevel(logging.DEBUG)
        console = _ConsoleHandler(self.stderr, self.progress)
        console.setFormatter(_ConsoleFormatter('%(message)s'))
        if args.verbose:
            console.setLevel(logging.DEBUG)
        elif args.quiet:
            console.setLevel(logging.WARNING)
        else:
            console.setLevel(logging.INFO)
        self._add_handler(console)

    def _add_handler(self, handler: logging.Handler) -> None:
        self._handlers.append(handler)
        for lg in self._loggers:
            lg.addHandler(handler)

    def start(self) -> None:
        """Open the log file and apply ``--debug``."""
        if not self.args.no_log_file:
            self._open_log_file()
        import cellacdc
        logger.debug(
            f'{PROG} {self.args.command} (Cell-ACDC {cellacdc.__version__}, '
            f'Python {sys.version.split()[0]}): {vars(self.args)}'
        )
        if self.args.debug:
            from . import config
            config.parser_args['debug'] = True

    def _open_log_file(self) -> None:
        explicit = self.args.log_file is not None
        if explicit:
            path = self.args.log_file
        else:
            import cellacdc
            stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            path = os.path.join(
                cellacdc.logs_path,
                f'{stamp}_{PROG}-{self.args.command}_{uuid.uuid4().hex[:8]}'
                '.log'
            )
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            handler = logging.FileHandler(path, encoding='utf-8')
        except OSError as err:
            if explicit:
                from .acdc_datastruct import errors
                raise errors.UsageError(
                    f'Cannot write the log file {path}: {err}',
                    hint='Choose another --log-file or use --no-log-file.'
                ) from err
            logger.warning(
                f'Cannot write the log file {path} ({err}); continuing '
                'without a log file.'
            )
            return
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s: %(message)s'
        ))
        self._add_handler(handler)
        self.log_file = path
        if not explicit:
            from . import myutils
            try:
                myutils.delete_older_log_files(os.path.dirname(path))
            except Exception as err:
                logger.debug(f'Old log files not deleted: {err!r}')

    def close(self, exit_code: int) -> None:
        """Remove the handlers; on failure the last stderr line names the
        log file."""
        self.progress.close()
        for handler in self._handlers:
            for lg in self._loggers:
                lg.removeHandler(handler)
            handler.close()
        self._handlers = []
        for lg, (level, propagate) in zip(self._loggers, self._logger_state):
            lg.setLevel(level)
            lg.propagate = propagate
        if exit_code != EXIT_OK and self.log_file is not None:
            _write_stream(self.stderr, f'See the log file: {self.log_file}\n')

    # ------------------------------------------------------------ helpers
    def log(self, text: str, level: str = 'INFO') -> None:
        """``ProgressCallbacks.log``."""
        logger.log(_LOG_LEVELS.get(str(level).upper(), logging.INFO), text)

    def callback_kwargs(self) -> dict:
        """``log``/``progress_init``/``progress_step`` for
        ``ProgressCallbacks``."""
        return {
            'log': self.log, 'progress_init': self.progress.init,
            'progress_step': self.progress.step,
        }

    def backend(self, *, keep_temp: bool = False) -> ConvertBackend:
        """The test backend, else the production worker backend (child
        output logged at DEBUG)."""
        if self.backend_factory is not None:
            return self.backend_factory()
        from .acdc_datastruct import backends, runner
        return backends.default_backend(
            for_gui=False, on_output=logger.debug,
            options=runner.RunnerOptions(keep_run_dir=keep_temp)
        )

    def check_reader_packages(
            self, backend: ConvertBackend, paths: Sequence[str], *,
            lazy_load: bool
        ) -> None:
        """Make sure the reader packages of ``paths`` are installed before
        the first worker child starts: install them with ``--yes``, ask in
        a terminal, else ``MissingDependencyError``.

        Skipped for a backend with a reader factory (test hook: it replaces
        bioio, so no reader package is needed)."""
        if getattr(backend, 'reader_factory', None) is not None:
            logger.debug('Reader packages not checked: test reader factory.')
            return
        from .acdc_bioio_bioformats import install
        from .acdc_datastruct import errors
        specs = install.required_reader_packages(paths, lazy_load=lazy_load)
        missing = install.missing_packages(specs)
        if not missing:
            return
        names = ', '.join(spec.pypi_name for spec in missing)
        command = _install_command(missing)
        if not self.args.yes and not self._interactive():
            raise errors.MissingDependencyError(
                f'Missing reader packages: {names}. Install them with:\n'
                f'  {command}',
                packages=[spec.pypi_name for spec in missing],
                install_command=command,
                hint='Or re-run the command with --yes to install them.'
            )
        logger.info(f'Missing reader packages: {names}')
        _install_reader_packages(missing, paths, auto_install=self.args.yes)
        still_missing = install.missing_packages(specs)
        if still_missing:
            names = ', '.join(spec.pypi_name for spec in still_missing)
            raise errors.MissingDependencyError(
                f'These reader packages are still missing: {names}',
                packages=[spec.pypi_name for spec in still_missing],
                install_command=_install_command(still_missing),
            )

    def _interactive(self) -> bool:
        return _isatty(sys.stdin) and _isatty(self.stderr)

    def report(self, err: BaseException) -> int:
        """Log ``err`` and return its exit code."""
        exit_code = _exit_code_for(err)
        name, message, hint, details = _describe_error(err)
        if exit_code == EXIT_FAILED and not _is_core_error(err):
            message = f'{name}: {message}'
        logger.error(message if hint is None else f'{message}\n{hint}')
        if details:
            logger.debug(details)
        return exit_code


def _isatty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _shell_join(argv: Sequence[str]) -> str:
    if os.name == 'nt':
        return subprocess.list2cmdline(list(argv))
    return shlex.join(argv)


def _install_command(specs) -> str:
    """Commands that install ``specs`` (conda ones first)."""
    conda = [spec.pypi_name for spec in specs if spec.installer == 'conda']
    pip = [spec.pypi_name for spec in specs if spec.installer != 'conda']
    commands = []
    if conda:
        commands.append(
            _shell_join(['conda', 'install', '-c', 'conda-forge', *conda])
        )
    if pip:
        commands.append(
            _shell_join([sys.executable, '-m', 'pip', 'install', *pip])
        )
    return '\n  '.join(commands)


def _install_reader_packages(missing, paths: Sequence[str], *,
                             auto_install: bool) -> None:
    """Install ``missing`` with the GUI helpers (and their version pins):
    the packages of a file extension through
    ``install.install_reader_dependencies`` for the first file of that
    extension, the others (bioio, zarr) through
    ``myutils.check_install_package``."""
    from . import myutils
    from .acdc_bioio_bioformats import install
    from .acdc_datastruct import errors, sources

    def failure(names: str) -> errors.MissingDependencyError:
        return errors.MissingDependencyError(
            f'Installing {names} failed.', packages=names.split(', '),
            install_command=_install_command(missing)
        )

    missing_names = {spec.pypi_name for spec in missing}
    base = {
        spec.pypi_name
        for spec in install.required_reader_packages([], lazy_load=False)
    }
    per_file = {}
    for path in sources.one_file_per_extension(paths):
        needed = {
            spec.pypi_name
            for spec in install.required_reader_packages(
                [path], lazy_load=False
            )
        } - base
        if needed & missing_names:
            per_file[path] = needed & missing_names
    tied = set().union(*per_file.values())
    for spec in missing:
        if spec.pypi_name in tied:
            continue
        try:
            myutils.check_install_package(
                spec.pypi_name, import_pkg_name=spec.import_name,
                installer=spec.installer, is_cli=True,
                auto_install=auto_install, logger_func=logger.info
            )
        except Exception as err:
            raise failure(spec.pypi_name) from err
    for path, names in per_file.items():
        install.install_reader_dependencies(
            path, failure(', '.join(sorted(names))),
            auto_install=auto_install, logger_func=logger.info
        )


def _core_modules():
    """``(errors, runner)`` of the core, or ``(None, None)`` when they cannot
    be imported (broken installation)."""
    try:
        from .acdc_datastruct import errors, runner
    except Exception:
        return None, None
    return errors, runner


def _is_core_error(err: BaseException) -> bool:
    errors, runner = _core_modules()
    return errors is not None and isinstance(
        err, (errors.DataStructError, runner.WorkerError)
    )


def _exit_code_for(err: BaseException) -> int:
    """2 for ``UsageError``, 130 for an interruption, else 1."""
    if isinstance(err, KeyboardInterrupt):
        return EXIT_INTERRUPTED
    if isinstance(err, SystemExit):
        return _system_exit_code(err) or EXIT_FAILED
    errors, _ = _core_modules()
    if errors is not None:
        if isinstance(err, errors.ConversionCancelled):
            return EXIT_INTERRUPTED
        if isinstance(err, errors.UsageError):
            return EXIT_USAGE
    return EXIT_FAILED


def _describe_error(
        err: BaseException
    ) -> tuple[str, str, str | None, str | None]:
    """(type name, message, hint, details) of ``err`` for the log and the
    JSON ``error`` object."""
    name = type(err).__name__
    if isinstance(err, KeyboardInterrupt):
        return name, 'Interrupted (Ctrl+C).', None, None
    if isinstance(err, SystemExit):
        return name, f'Exited with code {err.code!r}.', None, None
    errors, runner = _core_modules()
    if errors is not None and isinstance(err, errors.DataStructError):
        return name, err.message, err.hint, err.details
    if runner is not None and isinstance(err, runner.WorkerError):
        first, _, rest = str(err).partition('\n')
        return name, first, err.hint, rest or None
    details = ''.join(
        traceback.format_exception(type(err), err, err.__traceback__)
    )
    return name, str(err) or name, None, details


def _error_dict(err: BaseException | None) -> dict | None:
    if err is None:
        return None
    name, message, hint, _ = _describe_error(err)
    return {'type': name, 'message': message, 'hint': hint}


def _json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True) + '\n'


def _usage_error(message: str, hint: str | None = None):
    from .acdc_datastruct import errors
    return errors.UsageError(message, hint=hint)


def _split_list(text: str | None, flag: str) -> tuple[str, ...] | None:
    """Comma-separated values of ``flag`` (stripped; no empty item)."""
    if text is None:
        return None
    items = tuple(item.strip() for item in text.split(','))
    if not all(items):
        raise _usage_error(
            f'Invalid {flag} "{text}": empty item.',
            hint=f'Use comma-separated values, e.g. {flag} A,B.'
        )
    return items


def _float_list(text: str | None, flag: str) -> tuple[float, ...] | None:
    """Comma-separated finite numbers >= 0 of ``flag`` (wavelengths)."""
    items = _split_list(text, flag)
    if items is None:
        return None
    hint = f'Use comma-separated numbers >= 0, e.g. {flag} 510,610.'
    try:
        values = tuple(float(item) for item in items)
    except ValueError:
        raise _usage_error(
            f'Invalid {flag} "{text}": every item must be a number.',
            hint=hint
        ) from None
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise _usage_error(
            f'Invalid {flag} "{text}": every item must be a finite number '
            '>= 0.', hint=hint
        )
    return values


# argparse dests of the metadata numbers that must be finite and > 0 (the
# GUI dialogs cannot produce negative, NaN or infinite values either)
_POSITIVE_OPTIONS = (
    'lens_na', 'time_increment', 'physical_size_x', 'physical_size_y',
    'physical_size_z',
)


def _check_positive_options(args: argparse.Namespace) -> None:
    """``UsageError`` for a metadata number of ``_POSITIVE_OPTIONS`` (those
    the command has) that is not a finite number > 0."""
    for name in _POSITIVE_OPTIONS:
        value = getattr(args, name, None)
        if value is not None and not (math.isfinite(value) and value > 0):
            flag = '--' + name.replace('_', '-')
            raise _usage_error(
                f'Invalid {flag} {value}: it must be a finite number '
                'greater than 0.'
            )


def _join(values) -> str:
    return ', '.join(str(value) for value in values)


class _Command:
    """One command run: ``run`` does the work (raising on failure) and
    ``payload`` renders standard output for the outcome."""
    def __init__(self, session: _Session) -> None:
        self.session = session
        self.args = session.args

    def run(self) -> None:
        raise NotImplementedError

    def payload(self, error: BaseException | None, exit_code: int) -> str:
        raise NotImplementedError

    def execute(self) -> tuple[int, str]:
        """(exit code, payload) of the command."""
        error = None
        exit_code = EXIT_OK
        try:
            self.session.start()
            self.run()
        except (Exception, KeyboardInterrupt, SystemExit) as err:
            error = err
            exit_code = self.session.report(err)
        finally:
            self.session.progress.close()
        try:
            return exit_code, self.payload(error, exit_code)
        except Exception as err:
            logger.error(f'The result could not be printed: {err!r}')
            logger.debug(traceback.format_exc())
            return exit_code or EXIT_FAILED, ''


# ============================================================== metadata
_DEFAULTED_FLAGS = {
    'LensNA': '--lens-na',
    'TimeIncrement': '--time-increment',
    'TimeIncrementUnit': '--time-increment and --time-unit',
    'PhysicalSizeX': '--physical-size-x',
    'PhysicalSizeY': '--physical-size-y',
    'PhysicalSizeZ': '--physical-size-z',
    'PhysicalSizeUnit': '--physical-size-x/-y/-z and --physical-size-unit',
    'chNames': '--channels',
    'EmissionWavelength': '--em-wavelens',
}
_DEFAULTED_LABELS = {
    'chNames': 'Channel name', 'EmissionWavelength': 'Emission wavelength',
}
# Defaulted/warning field (GUI label or internal name) -> key of the
# ``metadata`` object, as the JSON output names it
_JSON_FIELD_KEYS = {
    'Image Name': 'ImageName', 'chNames': 'channel_names',
    'EmissionWavelength': 'em_wavelens',
}


def _json_field(name: str) -> str:
    return _JSON_FIELD_KEYS.get(name, name)


def _defaulted_text(field: str, filename: str, *, command: str) -> str:
    """Warning for a field that fell back to its default, with the
    ``convert`` flag that sets it (when there is one)."""
    label = _DEFAULTED_LABELS.get(field, field)
    text = (
        f'{label} not found in the metadata of {filename}; using the '
        'default'
    )
    flag = _DEFAULTED_FLAGS.get(field)
    if flag is not None:
        text = f'{text} (set it with {command}{flag})'
    return f'{text}.'


def _ordered_defaulted(defaulted) -> list[str]:
    from .acdc_datastruct import metadata
    order = metadata.WARNING_FIELDS + ('chNames', 'EmissionWavelength')
    return sorted(
        defaulted,
        key=lambda f: (order.index(f) if f in order else len(order), f)
    )


def _metadata_file(path: str) -> str:
    """The raw file of ``metadata -i``: the file itself, or the only
    supported file of a folder."""
    from .acdc_datastruct import errors, sources
    if os.path.isdir(path):
        listing = sources.list_source_files(path)
        unsupported = set(sources.find_unsupported_files(listing))
        supported = [f for f in listing if f not in unsupported]
        if len(supported) != 1:
            found = '\n'.join(f'  - {f}' for f in supported) or '  (none)'
            raise errors.SourceError(
                f'The folder {path} must contain exactly one supported raw '
                f'file, found {len(supported)}:\n{found}',
                hint='Pass the file itself with -i.'
            )
        return os.path.join(path, supported[0])
    if not os.path.isfile(path):
        raise errors.SourceError(f'The input does not exist: {path}')
    if sources.find_unsupported_files([os.path.basename(path)]):
        raise errors.SourceError(
            f'The file extension of {path} is not supported.'
        )
    return path


def _converted(func, value, unit):
    """``units`` conversion, or None when the unit is unknown."""
    from .acdc_datastruct import errors
    try:
        return func(value, unit)
    except errors.SettingsError:
        return None


def _metadata_values(resolved) -> dict:
    """The ``metadata`` object of ``acdc-data/metadata/1``."""
    from .acdc_datastruct import naming, units
    to_um = units.physical_size_to_um
    unit = resolved.physical_size_unit
    return {
        'LensNA': resolved.lens_na,
        'SizeS': resolved.size_s,
        'SizeT': resolved.size_t,
        'SizeZ': resolved.size_z,
        'SizeC': resolved.size_c,
        'TimeIncrement': resolved.time_increment,
        'TimeIncrementUnit': resolved.time_increment_unit,
        'TimeIncrement_s': _converted(
            units.time_increment_to_seconds, resolved.time_increment,
            resolved.time_increment_unit
        ),
        'PhysicalSizeX': resolved.physical_size_x,
        'PhysicalSizeY': resolved.physical_size_y,
        'PhysicalSizeZ': resolved.physical_size_z,
        'PhysicalSizeUnit': unit,
        'PhysicalSizeX_um': _converted(to_um, resolved.physical_size_x, unit),
        'PhysicalSizeY_um': _converted(to_um, resolved.physical_size_y, unit),
        'PhysicalSizeZ_um': _converted(to_um, resolved.physical_size_z, unit),
        'ImageName': resolved.image_name,
        'channel_names': list(resolved.channel_names),
        'channel_names_sanitized': [
            naming.sanitize_name(name) for name in resolved.channel_names
        ],
        'em_wavelens': list(resolved.em_wavelens),
    }


def _text_value(value) -> str:
    if value is None:
        return 'unknown'
    if isinstance(value, (list, tuple)):
        return _join(value)
    return str(value)


class _MetadataCommand(_Command):
    """``acdc-data metadata``: ``read_file_metadata`` (one worker child),
    then the text, JSON or metadata.csv rows of the file."""
    def __init__(self, session: _Session) -> None:
        super().__init__(session)
        self.text = ''

    def run(self) -> None:
        from .acdc_datastruct import convert, metadata, naming, planning
        from .acdc_datastruct import sources, writers
        args = self.args
        if args.include_xml and args.format != 'json':
            raise _usage_error('--include-xml needs --format json.')
        if (
            args.basename is not None and args.output is None
            and args.format != 'csv'
        ):
            raise _usage_error(
                '--basename needs --output or --format csv (it names the '
                'written files and the csv basename row).'
            )
        path = _metadata_file(args.input)
        name = os.path.basename(path)
        backend = self.session.backend()
        self.session.check_reader_packages(backend, [path], lazy_load=False)
        raw, resolved, warnings = convert.read_file_metadata(
            path, backend=backend
        )
        defaulted = _ordered_defaulted(resolved.defaulted)
        for field in defaulted:
            logger.warning(
                _defaulted_text(field, name, command='acdc-data convert ')
            )
        basename = args.basename
        if basename is None:
            basename = f'{naming.output_stem(name)}_'
        csv_text = ''
        if args.format == 'csv' or args.output is not None:
            settings, texts = metadata.settings_from_metadata(
                resolved, None, layout=sources.RawLayout.ONE_FILE_PER_POS,
                context=name, scene_names=raw.scene_names
            )
            for text in texts:
                logger.info(text)
            spec = planning.metadata_csv_spec(
                settings, basename=basename,
                saved_size_t=settings.saved_size_t(raw.size_t)
            )
            buffer = io.StringIO()
            writers.build_metadata_df(spec).to_csv(
                buffer, lineterminator='\n'
            )
            csv_text = buffer.getvalue()
            if args.output is not None:
                os.makedirs(args.output, exist_ok=True)
                csv_path = os.path.join(args.output, naming.position_filename(
                    basename, 'metadata', '.csv'
                ))
                xml_path = os.path.join(args.output, naming.position_filename(
                    basename, 'metadataXML', '.txt'
                ))
                writers.write_metadata_csv(csv_path, spec)
                writers.write_metadata_xml(xml_path, raw.metadata_xml)
                logger.info(f'Wrote {csv_path} and {xml_path}')
        values = _metadata_values(resolved)
        if args.format == 'csv':
            self.text = csv_text
        elif args.format == 'json':
            raw_dict = raw.to_dict()
            raw_dict.pop('metadata_xml', None)
            data = {
                'schema': METADATA_SCHEMA,
                'file': path,
                'reader': raw.reader,
                'metadata': values,
                'defaulted': [_json_field(field) for field in defaulted],
                'warnings': [
                    {
                        'field': _json_field(w.field), 'file': w.path,
                        'message': w.message,
                    }
                    for w in warnings
                ],
                'raw': raw_dict,
                'metadata_xml_length': len(raw.metadata_xml),
            }
            if args.include_xml:
                data['metadata_xml'] = raw.metadata_xml
            self.text = _json(data)
        else:
            lines = [f'file: {path}', f'reader: {raw.reader}']
            lines.extend(
                f'{key}: {_text_value(value)}'.rstrip()
                for key, value in values.items()
            )
            lines.append(f'defaulted: {_join(defaulted)}'.rstrip())
            self.text = '\n'.join(lines) + '\n'

    def payload(self, error: BaseException | None, exit_code: int) -> str:
        return '' if error is not None else self.text


# =============================================================== convert
class NoPositionWrittenError(RuntimeError):
    """``convert`` finished without writing any Position (exit code 1)."""


class _ConvertCommand(_Command):
    """``acdc-data convert``: ``select_source_files`` -> ``prepare`` ->
    reader packages -> ``convert`` (upfront schedule)."""
    def __init__(self, session: _Session) -> None:
        super().__init__(session)
        self.result = None

    def run(self) -> None:
        from .acdc_datastruct import convert, planning, sources
        args = self.args
        layout = sources.RawLayout.from_cli(args.layout)
        self._check_options(layout)
        overrides = self._overrides()
        selection = sources.select_source_files(
            args.input, layout, files=args.files, ext=args.ext,
            rename_invalid=RENAME_REPLACEMENTS.get(args.rename_invalid)
        )
        if selection.ignored:
            logger.info(
                f'{len(selection.ignored)} file(s) of the input folder are '
                'not used.'
            )
            logger.debug(f'Not used: {_join(selection.ignored)}')
        request = convert.ConvertRequest(
            src_dir=args.input, dst_dir=args.output, layout=layout,
            files=selection.files,
            on_existing=planning.OnExisting(args.on_existing),
            move_raw=bool(args.move_raw), renames=selection.renames,
            use_symlinks=args.use_symlinks,
            lazy_load=args.lazy_load is not False,
            other_files=args.other_files or 'leave', schedule='upfront',
            dry_run=args.dry_run,
        )
        convert.prepare(request)
        backend = self.session.backend(keep_temp=args.keep_temp)
        self.session.check_reader_packages(
            backend,
            [os.path.join(args.input, name) for name in selection.files],
            lazy_load=request.lazy_load
        )
        callbacks = convert.ConvertCallbacks(
            **self.session.callback_kwargs(),
            confirm=convert.confirm_from_metadata(
                overrides, on_warning=logger.warning
            ),
            warn=self._warn,
        )
        self.result = convert.convert(
            request, callbacks=callbacks, backend=backend
        )
        self.session.progress.close()
        count = len(self.result.jobs)
        if args.dry_run:
            logger.info(
                f'Dry run: {count} position(s) planned, nothing was changed.'
            )
        elif not self.result.ok:
            raise NoPositionWrittenError('No position was written.')
        else:
            logger.info(
                f'{len(self.result.written)} position(s) written to '
                f'{args.output}'
            )

    @staticmethod
    def _warn(warning) -> None:
        logger.warning(_defaulted_text(
            warning.field, os.path.basename(warning.path), command=''
        ))

    def _check_options(self, layout) -> None:
        """Flag combinations that cannot work; out-of-range numbers."""
        from .acdc_datastruct import sources
        args = self.args
        _check_positive_options(args)
        if (
            args.other_files is not None
            and layout != sources.RawLayout.ONE_FILE_PER_CHANNEL
        ):
            raise _usage_error(
                '--other-files is only for --layout one-per-channel.'
            )
        if args.use_symlinks and args.image_format == 'h5':
            raise _usage_error(
                '--use-symlinks cannot be combined with --image-format h5 '
                '(symlink files point to the raw files instead of writing '
                'image files).'
            )
        if args.time_unit is not None and args.time_increment is None:
            raise _usage_error('--time-unit needs --time-increment.')
        sizes = (args.physical_size_x, args.physical_size_y,
                 args.physical_size_z)
        if args.physical_size_unit is not None and sizes == (None,) * 3:
            raise _usage_error(
                '--physical-size-unit needs --physical-size-x, -y or -z.'
            )

    def _overrides(self):
        """``MetadataOverrides`` of the metadata flags, merged over
        ``--metadata-csv``."""
        from .acdc_datastruct import metadata, units
        args = self.args
        time_increment_s = None
        if args.time_increment is not None:
            time_increment_s = units.time_increment_to_seconds(
                args.time_increment, args.time_unit or 's'
            )
        sizes_um = [
            None if value is None else units.physical_size_to_um(
                value, args.physical_size_unit or 'um'
            )
            for value in (args.physical_size_x, args.physical_size_y,
                          args.physical_size_z)
        ]
        time_range = None
        if args.time_range is not None:
            time_range = metadata.parse_time_range_spec(args.time_range)
        positions = None
        if args.positions is not None:
            positions = metadata.parse_positions_spec(args.positions)
        overrides = metadata.MetadataOverrides(
            lens_na=args.lens_na,
            time_increment_s=time_increment_s,
            physical_size_x_um=sizes_um[0],
            physical_size_y_um=sizes_um[1],
            physical_size_z_um=sizes_um[2],
            channel_names=_split_list(args.channels, '--channels'),
            em_wavelens=_float_list(args.em_wavelens, '--em-wavelens'),
            save_channels=_split_list(args.save_channels, '--save-channels'),
            time_range=time_range,
            positions=positions,
            to_h5=args.image_format == 'h5',
            add_image_name=args.add_image_name,
        )
        if args.metadata_csv is None:
            return overrides
        csv_overrides, warnings = metadata.overrides_from_metadata_csv(
            args.metadata_csv
        )
        for text in warnings:
            logger.warning(text)
        return overrides.merged_over(csv_overrides)

    def _summary(self, error: BaseException | None) -> dict:
        """``ConvertResult.to_dict()`` of the result (after a failure: what
        was done before it), without the GUI-only 'cancelled'."""
        from .acdc_datastruct import convert
        result = self.result
        if error is not None:
            result = convert.partial_result(error) or result
        if result is None:
            result = convert.ConvertResult(dry_run=self.args.dry_run)
        summary = result.to_dict()
        summary.pop('cancelled', None)
        summary.pop('dry_run', None)
        return summary

    def payload(self, error: BaseException | None, exit_code: int) -> str:
        args = self.args
        try:
            summary = self._summary(error)
        except Exception:
            summary = None
        if args.format == 'json':
            data = {
                'schema': CONVERT_SCHEMA,
                'command': 'convert',
                'status': 'ok' if exit_code == EXIT_OK else 'error',
                'exit_code': exit_code,
                'dry_run': args.dry_run,
                'source': args.input,
                'destination': args.output,
                'layout': args.layout,
            }
            data.update(summary or {})
            if 'warnings' in data:
                data['warnings'] = [
                    dict(w, field=_json_field(w['field']))
                    for w in data['warnings']
                ]
            data['error'] = _error_dict(error)
            data['log_file'] = self.session.log_file
            return _json(data)
        if summary is None:
            return ''
        changed = any(
            summary[key] for key in
            ('positions', 'moved', 'renamed', 'deleted_images_folders')
        )
        if error is not None and (args.dry_run or not changed):
            return ''
        return self._text(summary, failed=error is not None)

    def _text(self, summary: dict, *, failed: bool) -> str:
        args = self.args
        planned = args.dry_run
        lines = []
        if failed:
            lines.append('The conversion failed after these changes:')
        elif planned:
            lines.append('Dry run: nothing was changed.')
        lines.extend([
            f'Source: {args.input}', f'Destination: {args.output}',
            f'Layout: {args.layout}',
        ])
        positions = summary['positions']
        title = 'Planned positions' if planned else 'Written positions'
        lines.append(f'{title}: {len(positions)}')
        for position in positions:
            details = [f'series {position["series"]}']
            if position['channels']:
                details.append(f'channels {_join(position["channels"])}')
            if position['time_range']:
                first, last = position['time_range']
                details.append(f'frames {first}-{last}')
            lines.append(
                f'  {position["source_file"]} (input {position["input_id"]})'
                f' -> {position["output"]} ({"; ".join(details)})'
            )
            lines.append(f'    {position["images_path"]}')
            for action in position['other_files']:
                lines.append(
                    f'    {action["mode"]} {action["src"]} -> {action["dst"]}'
                )
        if summary['skipped']:
            lines.append(
                f'Input positions not selected: {_join(summary["skipped"])}'
            )
        sections = (
            ('renamed', 'Raw files to rename' if planned else
             'Renamed raw files'),
            ('moved', 'Raw files to move' if planned else 'Moved raw files'),
        )
        for key, title in sections:
            if summary[key]:
                lines.append(f'{title}:')
                lines.extend(f'  {old} -> {new}' for old, new in summary[key])
        if summary['deleted_images_folders']:
            if planned:
                title = 'Images folders to delete:'
            elif failed:
                # A folder can be partly deleted (e.g. a file open elsewhere)
                title = 'Deleted or partly deleted Images folders:'
            else:
                title = 'Deleted Images folders:'
            lines.append(title)
            lines.extend(
                f'  {path}' for path in summary['deleted_images_folders']
            )
        return '\n'.join(lines) + '\n'


# =========================================================== restructure
def _split_channel(text: str) -> tuple[str, str | None]:
    """``--channel NAME[=ID]`` -> (name, identifier or None)."""
    name, sep, identifier = text.partition('=')
    return name.strip(), (identifier.strip() if sep else None)


def _restructure_dimensions(args: argparse.Namespace) -> dict:
    """Dimension entries (the dialog's recipe format, axes as strings) of
    the dimension flags; a flag replaces the whole recipe entry of its
    dimension."""
    dimensions = {}
    channels = [_split_channel(text) for text in args.channel or ()]
    if channels or args.channel_axis is not None:
        identifiers = [identifier for _, identifier in channels]
        names = [{'name': name} for name, _ in channels]
        if args.channel_axis is not None:
            if not channels:
                raise _usage_error(
                    '--channel-axis needs one --channel NAME per channel (in '
                    'the order of the axis).'
                )
            if any(identifier is not None for identifier in identifiers):
                raise _usage_error(
                    '--channel NAME=ID cannot be combined with '
                    '--channel-axis: give the channel names only.'
                )
            entry = {'mode': 'in_file', 'axis': str(args.channel_axis)}
        elif all(identifier is not None for identifier in identifiers):
            entry = {'mode': 'per_file'}
            for item, identifier in zip(names, identifiers):
                item['filename_identifier'] = identifier
        elif any(identifier is not None for identifier in identifiers):
            raise _usage_error(
                'If one --channel has =ID, all must (one file per channel).'
            )
        else:
            entry = {'mode': 'not_present'}
        entry['count'] = len(names)
        entry['names'] = names
        dimensions['channels'] = entry
    for key, marker, axis in (
            ('timepoints', args.time_marker, args.time_axis),
            ('z_slices', args.z_marker, args.z_axis)):
        if marker is not None:
            dimensions[key] = {'mode': 'per_file', 'filename_marker': marker}
        elif axis is not None:
            dimensions[key] = {'mode': 'in_file', 'axis': str(axis)}
    return dimensions


def _check_tiff_names(src_dir: str, files: Sequence[str] | None) -> None:
    """``restructure --files`` names TIFF files directly inside SRC, spelled
    exactly as in the folder listing: the core keeps only exact matches and
    silently ignores other names, also a name in another case on a
    case-insensitive file system (Windows, macOS)."""
    if files is None or not os.path.isdir(src_dir):
        return
    with os.scandir(src_dir) as entries:
        tiffs = {
            entry.name for entry in entries
            if entry.is_file()
            and entry.name.lower().endswith(('.tif', '.tiff'))
        }
    bad = [name for name in files if name not in tiffs]
    if bad:
        from .acdc_datastruct import errors
        by_lower = {name.lower(): name for name in tiffs}
        listing = '\n'.join(
            f'  - {name}' + (
                f' (the file is named {by_lower[name.lower()]})'
                if name.lower() in by_lower else ''
            )
            for name in bad
        )
        raise errors.SourceError(
            f'These --files are not TIFF files of the input folder '
            f'{src_dir}:\n{listing}',
            hint='--files takes names of .tif/.tiff files inside the input '
                 'folder.'
        )


class _RestructureCommand(_Command):
    """``acdc-data restructure``: recipe + flags -> ``options_from_recipe``
    -> ``plan_restructure`` -> (``save_recipe``) ->
    ``execute_restructure_plan``."""
    def __init__(self, session: _Session) -> None:
        super().__init__(session)
        self.plan = None
        self.output = None
        self.executing = False
        self.recipe_path = None

    def run(self) -> None:
        from .acdc_datastruct import callbacks, errors, restructure
        args = self.args
        _check_positive_options(args)
        recipe = {}
        if args.recipe is not None:
            path = restructure.resolve_recipe_spec(args.recipe)
            logger.info(f'Recipe: {path}')
            recipe = restructure.load_recipe(path)
        merged = {**recipe, **_restructure_dimensions(args)}
        _check_tiff_names(args.input, args.files)
        time_unit = None
        if args.time_unit is not None:
            time_unit = restructure.CLI_TIME_UNIT_TO_LABEL[args.time_unit]
        options = restructure.options_from_recipe(
            merged, source_folder=args.input, destination_folder=args.output,
            basename=args.basename, files=args.files,
            on_existing=args.on_existing,
            metadata={
                'PhysicalSizeX': args.physical_size_x,
                'PhysicalSizeY': args.physical_size_y,
                'PhysicalSizeZ': args.physical_size_z,
                'TimeIncrement': args.time_increment,
                'TimeIncrementUnit': time_unit,
            }
        )
        signals = callbacks.SignalsAdapter(
            callbacks.ProgressCallbacks(**self.session.callback_kwargs())
        )
        try:
            self.plan = restructure.plan_restructure(options, signals=signals)
        except errors.ExistingPositionsError as err:
            if args.on_existing == 'create-new':
                # create-new numbers after the last valid Position, so the
                # colliding folders have no images (overwrite would not
                # target them but the valid Positions)
                names = ', '.join(f'Position_{n}' for n in err.existing)
                err.hint = (
                    f'{names}: Position folder(s) without images, which '
                    'create-new does not count as existing positions. '
                    'Remove or rename them and run again.'
                )
            else:
                err.hint = (
                    f'Choose what to do with --on-existing: '
                    f'{" or ".join(err.choices)} ({err.hint})'
                )
            raise
        if args.save_recipe is not None:
            self.recipe_path = self._recipe_path(args.save_recipe)
            restructure.save_recipe(
                self.recipe_path,
                {
                    key: merged[key] for key in restructure.DIMENSION_KEYS
                    if key in merged
                },
                options['metadata']
            )
            logger.info(f'Recipe saved: {self.recipe_path}')
        if args.dry_run:
            logger.info(
                f'Dry run: {len(self.plan.positions)} position(s) planned, '
                'nothing was written.'
            )
            return
        self.executing = True
        self.output = restructure.execute_restructure_plan(
            self.plan, signals=signals
        )
        self.session.progress.close()
        logger.info(
            f'{len(self.output["position_folders"])} position(s) written to '
            f'{args.output}'
        )

    @staticmethod
    def _recipe_path(spec: str) -> str:
        """``--save-recipe``: a path (made absolute when parsing) or NAME in
        the GUI recipes folder."""
        if os.path.isabs(spec):
            return spec
        from .acdc_datastruct import restructure
        return os.path.join(
            restructure.recipes_folder(),
            f'{restructure.RECIPE_PREFIX}{spec}.json'
        )

    def _written(self, error: BaseException | None) -> tuple:
        """(position_folders, image_files, deleted_images_folders); None
        when a write failed part way (unknown)."""
        if self.output is not None:
            return (dict(self.output['position_folders']),
                    list(self.output['image_files']),
                    list(self.plan.to_delete))
        if self.executing:
            return None, None, None
        deleted = []
        if error is None and self.plan is not None:
            deleted = list(self.plan.to_delete)
        return {}, [], deleted

    def payload(self, error: BaseException | None, exit_code: int) -> str:
        args = self.args
        folders, images, deleted = self._written(error)
        if args.format == 'json':
            data = {'schema': RESTRUCTURE_SCHEMA}
            if self.plan is not None:
                data.update(self.plan.to_dict())
            data.update({
                'status': 'ok' if exit_code == EXIT_OK else 'error',
                'exit_code': exit_code,
                'position_folders': folders,
                'image_files': images,
                'deleted_images_folders': deleted,
                'error': _error_dict(error),
                'log_file': self.session.log_file,
            })
            return _json(data)
        if error is not None:
            return ''
        return self._text(images, deleted)

    def _text(self, images: list, deleted: list) -> str:
        args = self.args
        plan = self.plan.to_dict()
        lines = ['Dry run: nothing was written.'] if args.dry_run else []
        lines.extend([f'Source: {args.input}', f'Destination: {args.output}'])
        positions = plan['positions']
        title = 'Planned positions' if args.dry_run else 'Written positions'
        lines.append(f'{title}: {len(positions)}')
        for position in positions:
            size_y, size_x = position['yx']
            lines.append(
                f'  {position["identifier"]} -> '
                f'{os.path.basename(position["folder"])} '
                f'(T={position["size_t"]}, Z={position["size_z"]}, '
                f'Y={size_y}, X={size_x})'
            )
            for channel in position['channels']:
                lines.append(
                    f'    {channel["name"]}: {len(channel["files"])} file(s) '
                    f'-> {channel["image_file"]}'
                )
                lines.append(f'      {_join(channel["files"])}')
        if deleted:
            lines.append(
                'Images folders to delete:' if args.dry_run
                else 'Deleted Images folders:'
            )
            lines.extend(f'  {path}' for path in deleted)
        if not args.dry_run:
            lines.append(f'Image files written: {len(images)}')
        if self.recipe_path is not None:
            lines.append(f'Recipe saved: {self.recipe_path}')
        return '\n'.join(lines) + '\n'


_COMMANDS = {
    'metadata': _MetadataCommand,
    'convert': _ConvertCommand,
    'restructure': _RestructureCommand,
}


if __name__ == '__main__':
    sys.exit(main())
