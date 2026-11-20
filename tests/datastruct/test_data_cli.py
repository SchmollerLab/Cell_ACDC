"""`acdc-data` in-process.

`main([...], backend_factory=<InProcessBackend + FakeBioImage>)` with
`capsys`/`capfd`: parser, ASCII help, `--version`, return codes without
`SystemExit`, every hidden removed flag, the three commands and their
JSON/text/CSV payloads, exit 2 for every `UsageError` class with the source
and destination unchanged, exit 1 for runtime failures (incl. a missing
reader package without a terminal, zero positions written), exit 130 for
Ctrl+C, standard output carrying only the payload (also with prints and
C-level writes in the parent), the common options, the reader-package
handling, `--keep-temp`, the number of worker tasks of `convert`, the
parent never opening a raw file and relative paths through the real
`SubprocessBackend` (`subprocess_heavy`), the console-script registration
and `test_every_flag_is_exercised`.
"""
from __future__ import annotations

import argparse
import configparser
import inspect
import io
import json
import os
import re
import sys
from pathlib import Path

import pytest
import tifffile

from cellacdc import data_cli, myutils
from cellacdc.acdc_bioio_bioformats import install, reader
from cellacdc.acdc_datastruct import (
    backends, errors, restructure, runner, sources, units, writers
)

from . import conditions, fake_bioio, synthetic

FACTORY = fake_bioio.FakeBioImage
REPO_ROOT = Path(conditions.REPO_ROOT)
LOG_NAME = r'\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d_acdc-data-%s_[0-9a-f]{8}\.log'
METADATA_KEYS = [
    'LensNA', 'SizeS', 'SizeT', 'SizeZ', 'SizeC', 'TimeIncrement',
    'TimeIncrementUnit', 'TimeIncrement_s', 'PhysicalSizeX', 'PhysicalSizeY',
    'PhysicalSizeZ', 'PhysicalSizeUnit', 'PhysicalSizeX_um',
    'PhysicalSizeY_um', 'PhysicalSizeZ_um', 'ImageName', 'channel_names',
    'channel_names_sanitized', 'em_wavelens',
]
CONVERT_KEYS = [
    'schema', 'command', 'status', 'exit_code', 'dry_run', 'source',
    'destination', 'layout', 'positions', 'skipped', 'moved', 'renamed',
    'deleted_images_folders', 'warnings', 'error', 'log_file',
]
POSITION_KEYS = [
    'input_id', 'output', 'source_file', 'series', 'images_path', 'channels',
    'time_range', 'files', 'other_files',
]
PLAN_KEYS = [
    'source_folder', 'destination_folder', 'dimensions', 'prefix',
    'overwrite', 'create_new', 'positions', 'to_delete', 'metadata_rows',
]
RESTRUCTURE_RESULT_KEYS = [
    'status', 'exit_code', 'position_folders', 'image_files',
    'deleted_images_folders', 'error', 'log_file',
]
PER_FILE_CHANNELS = [
    '--channel', 'GFP=_GFP', '--channel', 'mCherry=_mCh', '--time-marker',
    '_t',
]
DIALOG_RECIPE = {
    'channels': {'mode': 'per_file', 'count': 2, 'names': [
        {'name': 'GFP', 'filename_identifier': '_GFP'},
        {'name': 'mCherry', 'filename_identifier': '_mCh'},
    ]},
    'timepoints': {'mode': 'per_file', 'filename_marker': '_t'},
    'z_slices': {'mode': 'not_present'},
}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Log files and recipes in tmp_path; no worker test hooks; the debug
    flag of config restored."""
    import cellacdc
    from cellacdc import config
    monkeypatch.setattr(cellacdc, 'logs_path', str(tmp_path / 'logs'))
    monkeypatch.setattr(
        cellacdc, 'restruct_files_recipes_path', str(tmp_path / 'recipes')
    )
    monkeypatch.setitem(
        config.parser_args, 'debug', config.parser_args.get('debug', False)
    )
    monkeypatch.delenv(runner.TEST_HOOKS_ENV, raising=False)
    monkeypatch.delenv(backends.READER_FACTORY_ENV, raising=False)
    FACTORY.reset_log()
    yield
    FACTORY.reset_log()


# ------------------------------------------------------------------ helpers
def fake_backend() -> backends.InProcessBackend:
    return backends.InProcessBackend(reader_factory=FACTORY)


def cli(*argv, backend_factory=fake_backend) -> int:
    return data_cli.main(
        [str(arg) for arg in argv], backend_factory=backend_factory
    )


def fake(folder, name: str, **spec) -> str:
    os.makedirs(folder, exist_ok=True)
    spec.setdefault('channel_names', ['GFP', 'mCherry'][:spec.get('C', 2)])
    fake_bioio.write_fake_raw(
        os.path.join(folder, name), fake_bioio.make_spec(**spec)
    )
    return name


def one_per_pos(src, count: int = 2, **spec) -> list[str]:
    return [
        fake(src, f'pos{i + 1}.czi', s_offset=i, **spec)
        for i in range(count)
    ]


def one_per_channel(src, positions=(1, 2), channels=('GFP', 'mCh'),
                    skip=()) -> list[str]:
    return [
        fake(src, f'ASY015_{position}_{channel}.czi', C=1, s_offset=position,
             c_offset=c, channel_names=[channel])
        for position in positions for c, channel in enumerate(channels)
        if (position, channel) not in skip
    ]


def loose_tiffs(folder, **kwargs) -> list[str]:
    options = dict(
        pattern='pos{pos}_{ch}_t{t}.tif', positions=(1, 2),
        channels=('GFP', 'mCh'), timepoints=(1, 2, 3)
    )
    options.update(kwargs)
    return synthetic.make_loose_tiff_set(str(folder), **options)


def convert_args(src, dst, *extra, layout='one-per-pos') -> list:
    return ['convert', '-i', src, '-o', dst, '--layout', layout, *extra]


def snapshot(*roots) -> dict:
    return conditions.tree_snapshot(*[str(root) for root in roots])


def subparsers(parser: argparse.ArgumentParser) -> dict:
    action = next(
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return dict(action.choices)


def plane_origins(path) -> list[tuple[int, int, int, int]]:
    array = tifffile.imread(str(path))
    planes = array.reshape(-1, *array.shape[-2:])
    return [synthetic.decode_pixel(plane[0, 0]) for plane in planes]


class SpyBackend:
    """Fake-reader backend that records the tasks; `failures` maps a task
    name to the exception it raises."""
    def __init__(self, **failures) -> None:
        self.inner = fake_backend()
        self.reader_factory = self.inner.reader_factory
        self.calls: list[str] = []
        self.failures = failures

    def _call(self, name: str, *args, **kwargs):
        self.calls.append(name)
        if name in self.failures:
            raise self.failures[name]
        return getattr(self.inner, name)(*args, **kwargs)

    def init_reader(self, path):
        return self._call('init_reader', path)

    def read_metadata(self, paths):
        return self._call('read_metadata', paths)

    def read_sample(self, path, **kwargs):
        return self._call('read_sample', path, **kwargs)

    def check_positions(self, jobs):
        return self._call('check_positions', jobs)

    def save_positions(self, jobs, texts, *, on_event):
        return self._call('save_positions', jobs, texts, on_event=on_event)


class SilentSaveBackend(SpyBackend):
    """Writes nothing and reports no Position."""
    def save_positions(self, jobs, texts, *, on_event):
        self.calls.append('save_positions')
        return []


class NoisyBackend(SpyBackend):
    """Prints and writes to file descriptor 1 while reading."""
    def read_metadata(self, paths):
        print('PRINTED')
        os.write(1, b'C-LEVEL\n')
        return super().read_metadata(paths)


class ProductionLikeBackend(SpyBackend):
    """Reads with the fake but has no reader factory, like the production
    backend, so the CLI checks the reader packages."""
    def __init__(self) -> None:
        super().__init__()
        self.reader_factory = None


# =================================================================== parser
def test_prog_and_commands():
    parser = data_cli.build_parser()
    assert parser.prog == 'acdc-data'
    assert set(subparsers(parser)) == {'metadata', 'convert', 'restructure'}
    assert data_cli.run is data_cli.main


@pytest.mark.parametrize('argv, hidden', [
    (['-h'], []),
    (['--help'], []),
    (['metadata', '-h'], []),
    (['convert', '-h'], ['--size-t', '--trust-metadata', '--basename']),
    (['restructure', '--help'],
     ['--layout', '--action', '--segm-folder', '--channels ']),
])
def test_help_is_ascii(argv, hidden, capsys):
    assert data_cli.main(argv) == 0
    out = capsys.readouterr().out
    assert out.isascii()
    command = '' if argv[0].startswith('-') else f' {argv[0]}'
    assert out.startswith(f'usage: acdc-data{command}')
    for flag in hidden:
        assert flag not in out


def test_parser_names_come_from_the_core():
    """--layout and the unit choices are the core's names (one definition);
    every restructure --time-unit has a dialog label."""
    commands = subparsers(data_cli.build_parser())

    def choices(command: str, dest: str) -> tuple:
        return tuple(next(
            action.choices for action in commands[command]._actions
            if action.dest == dest
        ))

    assert choices('convert', 'layout') == tuple(
        layout.cli_name for layout in sources.RawLayout
    ) == ('single-multi-pos', 'one-per-pos', 'one-per-channel')
    assert choices('convert', 'time_unit') == units.CLI_TIME_UNITS
    assert choices('convert', 'physical_size_unit') == units.CLI_LENGTH_UNITS
    assert choices('restructure', 'time_unit') == units.CLI_TIME_UNITS
    assert tuple(restructure.CLI_TIME_UNIT_TO_LABEL) == units.CLI_TIME_UNITS


def test_version_and_usage_errors(capsys):
    assert data_cli.main(['--version']) == 0
    out = capsys.readouterr().out
    assert re.fullmatch(r'acdc-data \(Cell-ACDC\) \S+\n', out)
    # argparse errors are returned, never raised as SystemExit
    assert data_cli.main(['bogus']) == 2
    assert data_cli.main([]) == 2
    assert data_cli.main(['convert', '-i', 'x']) == 2
    assert data_cli.main(['metadata', '-i', 'x', '-v', '-q']) == 2
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'usage: acdc-data' in captured.err


@pytest.mark.parametrize('command, flag, hint', [
    ('convert', ['--size-t', '5'], 'sizes are read from the file'),
    ('convert', ['--size-z', '1'], 'use --time-range, --positions'),
    ('convert', ['--size-c', '2'], 'sizes are read from the file'),
    ('convert', ['--size-s', '3'], 'sizes are read from the file'),
    ('convert', ['--trust-metadata'], 'file metadata is always used'),
    ('convert', ['--no-trust-metadata'], 'file metadata is always used'),
    ('convert', ['--basename', 'x_'], 'output names follow the GUI'),
    ('convert', ['--format', 'tif'], 'use --image-format tif'),
    ('convert', ['--format', 'h5'], 'use --image-format h5'),
    ('restructure', ['--layout', 'multi-channel'], 'the dimension flags'),
    ('restructure', ['--action', 'move'], 'only read'),
    ('restructure', ['--segm-folder', 'masks'], 'segmentation masks'),
    ('restructure', ['--channels', 'A,B'], 'repeat --channel NAME[=ID]'),
])
def test_removed_flags_exit_2_with_hint(command, flag, hint, tmp_path,
                                        capsys):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    one_per_pos(src)
    before = snapshot(src, dst)
    if command == 'convert':
        argv = convert_args(src, dst, *flag)
    else:
        argv = ['restructure', '-i', src, '-o', dst, *flag]
    assert cli(*argv) == 2
    err = capsys.readouterr().err
    assert hint in err
    assert 'Differences from pull request #1109' in err
    assert snapshot(src, dst) == before


# ================================================================= metadata
def test_metadata_text(tmp_path, capsys):
    raw = tmp_path / 'raw'
    fake(raw, 'pos1.czi', channel_names=['GFP', 'mCh!'])
    assert cli('metadata', '-i', raw / 'pos1.czi') == 0
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[:2] == [
        f'file: {raw / "pos1.czi"}',
        'reader: tests.datastruct.fake_bioio.FakeBioImage',
    ]
    for line in ('LensNA: 1.4', 'SizeT: 3', 'TimeIncrement_s: 1.0',
                 'channel_names: GFP, mCh!', 'channel_names_sanitized: GFP, '
                 'mCh', 'em_wavelens: 0.0, 0.0', 'ImageName:'):
        assert line in lines
    assert lines[-1].startswith('defaulted: LensNA')
    # Every defaulted field is a warning with its convert flag,
    # including the emission wavelength that has no legacy warning line
    assert (
        'WARNING: LensNA not found in the metadata of pos1.czi; using the '
        'default (set it with acdc-data convert --lens-na).'
    ) in captured.err
    assert 'acdc-data convert --em-wavelens' in captured.err


@pytest.mark.parametrize('include_xml', [False, True])
def test_metadata_json(include_xml, tmp_path, capsys):
    raw = tmp_path / 'raw'
    fake(raw, 'pos1.czi', metadata_text='<meta x="1"/>', time_interval_s=0.5)
    argv = ['metadata', '--input', raw / 'pos1.czi', '--format', 'json']
    if include_xml:
        argv.append('--include-xml')
    assert cli(*argv) == 0
    data = json.loads(capsys.readouterr().out)
    keys = [
        'schema', 'file', 'reader', 'metadata', 'defaulted', 'warnings',
        'raw', 'metadata_xml_length',
    ]
    assert list(data) == keys + (['metadata_xml'] if include_xml else [])
    assert data['schema'] == 'acdc-data/metadata/1'
    assert data['file'] == str(raw / 'pos1.czi')
    assert list(data['metadata']) == METADATA_KEYS
    assert data['metadata']['TimeIncrement'] == 0.5
    assert data['metadata']['TimeIncrement_s'] == 0.5
    assert data['metadata']['PhysicalSizeX_um'] == 1.0
    assert data['warnings'][0] == {
        'field': 'LensNA', 'file': str(raw / 'pos1.czi'),
        'message': 'WARNING: LensNA not found in metadata.',
    }
    assert {'LensNA', 'em_wavelens'} <= set(data['defaulted'])
    assert 'TimeIncrement' not in data['defaulted']
    assert 'metadata_xml' not in data['raw']
    assert data['raw']['lens_na'] is None
    assert data['metadata_xml_length'] == len('<meta x="1"/>')
    if include_xml:
        assert data['metadata_xml'] == '<meta x="1"/>'


def test_json_fields_are_metadata_keys(tmp_path, capsys):
    """'defaulted' and 'warnings[].field' of metadata and convert name keys
    of the 'metadata' object, not the GUI labels ('Image Name', 'chNames',
    'EmissionWavelength'); 'message' keeps the GUI text."""
    raw, dst = tmp_path / 'raw', tmp_path / 'exp'
    fake(raw, 'pos1.czi', channel_names=['GFP', ''])
    assert cli('metadata', '-i', raw / 'pos1.czi', '--format', 'json') == 0
    data = json.loads(capsys.readouterr().out)
    assert {'ImageName', 'channel_names', 'em_wavelens'} <= set(
        data['defaulted']
    )
    assert set(data['defaulted']) <= set(data['metadata'])
    assert all(w['field'] in data['metadata'] for w in data['warnings'])
    assert {
        'field': 'ImageName', 'file': str(raw / 'pos1.czi'),
        'message': 'WARNING: Image Name not found in metadata.',
    } in data['warnings']
    assert cli(*convert_args(raw, dst, '--dry-run', '--format', 'json')) == 0
    data = json.loads(capsys.readouterr().out)
    fields = [w['field'] for w in data['warnings']]
    assert 'ImageName' in fields
    assert set(fields) <= set(METADATA_KEYS)


@pytest.mark.parametrize('fmt', ['text', 'json', 'csv'])
def test_metadata_output_folder(fmt, tmp_path, capsys):
    raw, out_dir = tmp_path / 'raw', tmp_path / 'out dir'
    fake(raw, 'a.b.czi', metadata_text='<xml/>')
    assert cli('metadata', '-i', raw / 'a.b.czi', '--format', fmt,
               '-o', out_dir) == 0
    out = capsys.readouterr().out
    if fmt == 'json':
        assert json.loads(out)['schema'] == 'acdc-data/metadata/1'
    elif fmt == 'csv':
        assert out.startswith('Description,values\n')
    else:
        assert out.startswith('file: ')
    assert writers.read_metadata_csv(
        str(out_dir / 'a_b_metadata.csv')
    )['basename'] == 'a_b_'
    xml = out_dir / 'a_b_metadataXML.txt'
    assert xml.read_text(encoding='utf-8') == '<xml/>'
    assert cli('metadata', '-i', raw / 'a.b.czi', '--format', fmt,
               '--output', out_dir, '--basename', 'my_') == 0
    capsys.readouterr()
    assert writers.read_metadata_csv(
        str(out_dir / 'my_metadata.csv')
    )['basename'] == 'my_'
    assert (out_dir / 'my_metadataXML.txt').is_file()


def test_metadata_csv_round_trip_into_convert(tmp_path, capsys):
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    one_per_pos(src)
    assert cli('metadata', '-i', src / 'pos1.czi', '--format', 'csv') == 0
    text = capsys.readouterr().out
    assert 'basename,pos1_\n' in text and 'channel_1_name,mCherry\n' in text
    csv_path = tmp_path / 'edited.csv'
    edited = text.replace('LensNA,1.4', 'LensNA,1.2').replace(
        'PhysicalSizeX,1.0', 'PhysicalSizeX,0.3'
    )
    csv_path.write_text(edited, encoding='utf-8')
    # flags > CSV > file
    assert cli(*convert_args(src, dst, '--metadata-csv', csv_path,
                             '--physical-size-x', '0.25')) == 0
    err = capsys.readouterr().err
    assert 'WARNING: The metadata CSV row "basename" is ignored' in err
    values = writers.read_metadata_csv(
        str(dst / 'Position_1' / 'Images' / 'pos1_s1_metadata.csv')
    )
    assert float(values['LensNA']) == 1.2
    assert float(values['PhysicalSizeX']) == 0.25
    assert float(values['PhysicalSizeY']) == 1.0


def test_metadata_input_checks(tmp_path, capsys):
    raw = tmp_path / 'raw'
    fake(raw, 'pos1.czi')
    (raw / 'notes.md').write_text('x', encoding='utf-8')
    assert cli('metadata', '-i', raw, '--format', 'json') == 0
    assert json.loads(capsys.readouterr().out)['file'] == str(
        raw / 'pos1.czi'
    )
    fake(raw, 'pos2.czi')
    empty = tmp_path / 'empty'
    empty.mkdir()
    before = snapshot(raw, empty)
    for argv, listed in (
            (['-i', raw], ['pos1.czi', 'pos2.czi']),
            (['-i', empty], ['(none)']),
            (['-i', tmp_path / 'missing.czi'], ['missing.czi']),
            (['-i', raw / 'notes.md'], ['notes.md']),
            (['-i', raw / 'pos1.czi', '--include-xml'], ['--format json']),
            (['-i', raw / 'pos1.czi', '--basename', 'x_'], ['--output'])):
        assert cli('metadata', *argv) == 2
        captured = capsys.readouterr()
        assert captured.out == ''
        for text in listed:
            assert text in captured.err
    assert snapshot(raw, empty) == before


# ================================================================== convert
def test_convert_json_result_and_log_file(tmp_path, capsys):
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    one_per_pos(src)
    assert cli(*convert_args(src, dst, '--format', 'json')) == 0
    data = json.loads(capsys.readouterr().out)
    assert list(data) == CONVERT_KEYS
    assert (data['schema'], data['command'], data['status']) == (
        'acdc-data/convert/1', 'convert', 'ok'
    )
    assert (data['exit_code'], data['dry_run'], data['error']) == (
        0, False, None
    )
    assert (data['source'], data['destination'], data['layout']) == (
        str(src), str(dst), 'one-per-pos'
    )
    assert [list(p) for p in data['positions']] == [POSITION_KEYS] * 2
    first = data['positions'][0]
    assert (first['output'], first['source_file'], first['channels']) == (
        'Position_1', 'pos1.czi', ['GFP', 'mCherry']
    )
    assert all(os.path.isfile(path) for path in first['files'])
    assert set(data['warnings'][0]) == {'field', 'file', 'message'}
    # default log file: <logs>/<stamp>_acdc-data-convert_<id>.log
    log_file = Path(data['log_file'])
    assert log_file.parent == tmp_path / 'logs'
    assert re.fullmatch(LOG_NAME % 'convert', log_file.name)
    assert 'acdc-data convert (Cell-ACDC' in log_file.read_text('utf-8')


def test_convert_dry_run_changes_nothing(tmp_path, capsys):
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    one_per_pos(src)
    before = snapshot(src, dst)
    assert cli(*convert_args(src, dst, '--dry-run', '--move-raw')) == 0
    out = capsys.readouterr().out
    assert out.startswith('Dry run: nothing was changed.\n')
    assert 'pos2.czi (input 2) -> Position_2' in out
    assert 'Raw files to move:' in out
    assert cli(*convert_args(src, dst, '--dry-run', '--format', 'json')) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['dry_run'] is True
    assert [p['files'] for p in data['positions']] == [[], []]
    assert snapshot(src, dst) == before
    assert cli(*convert_args(src, dst)) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[:4] == [
        f'Source: {src}', f'Destination: {dst}', 'Layout: one-per-pos',
        'Written positions: 2',
    ]


def short_second_scene(src, name: str = 'exp (1).czi') -> str:
    """Single-multi-pos file whose scene 1 has 2 frames, scene 0 (where
    the plan comes from) 4."""
    os.makedirs(src, exist_ok=True)
    scene = {'C': 1, 'Z': 1, 'Y': 8, 'X': 8}
    fake_bioio.write_fake_raw(os.path.join(src, name), {
        'scenes': [{'T': 4, **scene}, {'T': 2, **scene}],
        'channel_names': ['GFP'],
    })
    return name


@pytest.mark.parametrize('dry_run', [False, True])
@pytest.mark.parametrize('case', ['one-per-channel', 'single-multi-pos'])
def test_convert_checks_every_raw_file_before_any_change(tmp_path, capsys,
                                                         case, dry_run):
    """A file (one-per-channel: ASY015_2_GFP.czi has 2 frames, its plan 3)
    or scene (single-multi-pos: scene 1 has 2 frames, scene 0 4) that does
    not fit the plan: exit 2 before anything is renamed or moved and before
    DST is created, also with --dry-run; the message names the file in
    SRC."""
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    if case == 'one-per-channel':
        one_per_channel(src)
        fake(src, 'ASY015_2_GFP.czi', C=1, T=2, s_offset=2,
             channel_names=['GFP'])
        extra, short, where = ['--move-raw'], 'ASY015_2_GFP.czi', (
            'Time range 1-3', 'series 0'
        )
    else:
        short = short_second_scene(src)
        extra, where = ['--rename-invalid', 'underscore', '--move-raw'], (
            'Time range 1-4', 'series 1'
        )
    if dry_run:
        extra.append('--dry-run')
    before = snapshot(src)
    assert cli(*convert_args(src, dst, *extra, '--format', 'json',
                             layout=case)) == 2
    data = json.loads(capsys.readouterr().out)
    assert (data['status'], data['dry_run']) == ('error', dry_run)
    assert data['error']['type'] == 'SettingsError'
    message = data['error']['message']
    assert message.startswith(f'{where[0]} (1-based, inclusive) does not '
                              'fit the 2 frame(s)')
    assert f'"{src / short}" ({where[1]})' in message
    assert '--time-range' in data['error']['hint']
    assert (data['moved'], data['renamed'],
            data['deleted_images_folders']) == ([], [], [])
    assert snapshot(src) == before
    assert not dst.exists()


def test_convert_metadata_flags(tmp_path, capsys):
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    one_per_pos(src, count=3, T=4)
    assert cli(*convert_args(
        src, dst, '--lens-na', '1.2', '--time-increment', '2',
        '--time-unit', 'min', '--physical-size-x', '500',
        '--physical-size-y', '250', '--physical-size-z', '2',
        '--physical-size-unit', 'nm', '--channels', 'phase, GFP!',
        '--em-wavelens', '510,610', '--save-channels', 'GFP',
        '--positions', 'Position_2-3', '--time-range', '2:',
        '--image-format', 'h5', '--format', 'json'
    )) == 0
    data = json.loads(capsys.readouterr().out)
    assert [p['output'] for p in data['positions']] == [
        'Position_2', 'Position_3'
    ]
    assert data['skipped'] == [1]
    assert data['positions'][0]['channels'] == ['GFP']
    assert data['positions'][0]['time_range'] == [2, 4]
    images = dst / 'Position_2' / 'Images'
    values = writers.read_metadata_csv(str(images / 'pos2_s2_metadata.csv'))
    assert float(values['LensNA']) == 1.2
    assert float(values['TimeIncrement']) == 120.0
    assert [float(values[f'PhysicalSize{axis}']) for axis in 'XYZ'] == [
        0.5, 0.25, 0.002
    ]
    assert values['SizeT'] == '3'
    assert values['channel_1_name'] == 'GFP'
    assert float(values['channel_1_emWavelen']) == 610.0
    assert 'channel_0_name' not in values
    assert (images / 'pos2_s2_GFP.h5').is_file()
    assert not (dst / 'Position_1').exists()


@pytest.mark.parametrize('dry_run', [False, True])
def test_convert_overrides_ignore_unselected_files(dry_run, tmp_path,
                                                   capsys):
    """One file per position: overrides that do not fit a file that
    --positions does not select (here its frames and number of channels)
    do not reject the run, since nothing of that file is written."""
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    fake(src, 'pos1.czi', T=3, C=1, channel_names=[''])
    fake(src, 'pos2.czi', T=5, s_offset=1)
    extra = ['--dry-run'] if dry_run else []
    assert cli(*convert_args(
        src, dst, '--positions', '2', '--time-range', '4:5', '--channels',
        'A,B', *extra, '--format', 'json'
    )) == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert [(p['output'], p['time_range'], p['channels'])
            for p in data['positions']] == [('Position_2', [4, 5], ['A', 'B'])]
    assert data['skipped'] == [1]
    assert not (dst / 'Position_1').exists()
    # no warning about the channel names of the unselected file
    assert 'has no name' not in captured.err
    # a selected file that does not fit is still an error, naming the file
    assert cli(*convert_args(
        src, dst, '--positions', '1', '--time-range', '4:5', '--dry-run'
    )) == 2
    assert 'pos1.czi: Invalid --time-range 4:5' in capsys.readouterr().err


def test_convert_file_selection_and_renames(tmp_path, capsys):
    src = tmp_path / 'raw'
    one_per_pos(src)
    fake(src, 'pos#3.czi')
    fake(src, 'extra.nd2')
    # --files: input ids are the order of the selected files
    assert cli(*convert_args(src, tmp_path / 'a', '--files', 'pos2.czi',
                             '--format', 'json')) == 0
    data = json.loads(capsys.readouterr().out)
    assert [(p['input_id'], p['output'], p['source_file'])
            for p in data['positions']] == [(1, 'Position_1', 'pos2.czi')]
    # an invalid name is an error without --rename-invalid
    before = snapshot(src)
    assert cli(*convert_args(src, tmp_path / 'b', '--ext', '.czi')) == 2
    assert '--rename-invalid' in capsys.readouterr().err
    assert snapshot(src) == before
    # --rename-invalid renames right before writing
    assert cli(*convert_args(src, tmp_path / 'b', '--ext', 'CZI',
                             '--rename-invalid', 'dash', '--format',
                             'json')) == 0
    data = json.loads(capsys.readouterr().out)
    assert data['renamed'] == [
        [str(src / 'pos#3.czi'), str(src / 'pos-3.czi')]
    ]
    assert (src / 'pos-3.czi').is_file() and (src / 'extra.nd2').is_file()
    assert data['positions'][2]['files'][0].endswith(
        'pos-3_s3_metadataXML.txt'
    )


def test_convert_one_per_channel_other_files(tmp_path, capsys):
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    one_per_channel(src)
    (src / 'ASY015_1_notes.md').write_text('notes', encoding='utf-8')
    assert cli(*convert_args(src, dst, '--ext', '.czi', '--other-files',
                             'copy', '--format', 'json',
                             layout='one-per-channel')) == 0
    data = json.loads(capsys.readouterr().out)
    assert [p['channels'] for p in data['positions']] == [['GFP', 'mCh']] * 2
    actions = data['positions'][0]['other_files']
    assert [(a['mode'], os.path.basename(a['src'])) for a in actions] == [
        ('copy', 'ASY015_1_notes.md')
    ]
    assert os.path.isfile(actions[0]['dst'])
    assert (src / 'ASY015_1_notes.md').is_file()
    assert data['positions'][1]['other_files'] == []


def test_convert_one_per_channel_rename_invalid(tmp_path, capsys):
    """One file per channel with --rename-invalid: the outputs are
    named after the renamed files, exactly as when the files are renamed
    first (GUI rename button)."""
    outputs = []
    for case in ('rename-invalid', 'renamed first'):
        src, dst = tmp_path / case / 'raw', tmp_path / case / 'exp'
        for position in (1, 2):
            for c, channel in enumerate(('GFP', 'mCh')):
                name = f'exp(1)_{position}_{channel}.czi'
                if case == 'renamed first':
                    name = sources.planned_valid_filename(name, '_')
                fake(src, name, C=1, s_offset=position, c_offset=c,
                     channel_names=[channel])
        extra = ['--rename-invalid', 'underscore']
        if case == 'renamed first':
            extra = []
        assert cli(*convert_args(src, dst, *extra, '--format', 'json',
                                 layout='one-per-channel')) == 0
        data = json.loads(capsys.readouterr().out)
        outputs.append(sorted(
            os.path.relpath(path, dst)
            for position in data['positions'] for path in position['files']
        ))
    assert os.path.join('Position_1', 'Images', 'exp_1___s1_GFP.tif') in (
        outputs[0]
    )
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('flag, expected', [
    ('--lazy-load', 'True'), ('--no-lazy-load', 'False'),
])
def test_convert_symlinks_and_lazy_load(flag, expected, tmp_path, capsys):
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    one_per_pos(src, count=1)
    assert cli(*convert_args(src, dst, '--use-symlinks', flag)) == 0
    ini = configparser.ConfigParser()
    ini.read(dst / 'Position_1' / 'Images' / 'pos1_s1_symlink.ini')
    assert ini.sections() == ['channel_name.GFP', 'channel_name.mCherry']
    for section in ini.sections():
        assert ini[section]['lazy_load'] == expected
        assert ini[section]['source_filepath'] == str(src / 'pos1.czi')


def test_raw_files_stay_unless_move_raw(tmp_path, capsys):
    src = tmp_path / 'raw'
    one_per_pos(src)
    before = snapshot(src)
    assert cli(*convert_args(src, tmp_path / 'a')) == 0
    assert cli(*convert_args(src, tmp_path / 'b', '--no-move-raw')) == 0
    assert snapshot(src) == before
    capsys.readouterr()
    dst = tmp_path / 'c'
    assert cli(*convert_args(src, dst, '--move-raw', '--format', 'json')) == 0
    data = json.loads(capsys.readouterr().out)
    moved_to = dst / 'raw_microscopy_files'
    assert data['moved'] == [
        [str(src / name), str(moved_to / name)]
        for name in ('pos1.czi', 'pos2.czi')
    ]
    assert os.listdir(src) == []
    assert sorted(os.listdir(moved_to)) == ['pos1.czi', 'pos2.czi']


def test_on_existing(tmp_path, capsys):
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    one_per_pos(src)
    assert cli(*convert_args(src, dst)) == 0
    extra_file = dst / 'Position_1' / 'Images' / 'acdc_output.csv'
    extra_file.write_text('x', encoding='utf-8')
    capsys.readouterr()
    before = snapshot(src, dst)
    # default: error, listing the folders and the choices
    assert cli(*convert_args(src, dst, '--format', 'json')) == 2
    data = json.loads(capsys.readouterr().out)
    assert (data['status'], data['exit_code']) == ('error', 2)
    assert data['error']['type'] == 'ExistingPositionsError'
    assert 'Position_1, Position_2' in data['error']['message']
    for choice in ('overwrite', 'add', 'create-new'):
        assert choice in data['error']['hint']
    assert snapshot(src, dst) == before
    # overwrite: dry run lists what would be deleted
    assert cli(*convert_args(src, dst, '--on-existing', 'overwrite',
                             '--dry-run', '--format', 'json')) == 0
    images = [str(dst / f'Position_{n}' / 'Images') for n in (1, 2)]
    data = json.loads(capsys.readouterr().out)
    assert data['deleted_images_folders'] == images
    assert snapshot(src, dst) == before
    assert cli(*convert_args(src, dst, '--on-existing', 'overwrite',
                             '--format', 'json')) == 0
    assert json.loads(
        capsys.readouterr().out
    )['deleted_images_folders'] == images
    assert not extra_file.exists()
    assert cli(*convert_args(src, dst, '--on-existing', 'add')) == 0
    assert 'Written positions: 2' in capsys.readouterr().out
    assert cli(*convert_args(src, dst, '--on-existing', 'create-new',
                             '--format', 'json')) == 0
    data = json.loads(capsys.readouterr().out)
    assert [p['output'] for p in data['positions']] == [
        'Position_3', 'Position_4'
    ]


def test_overwrite_reports_a_partly_deleted_images_folder(tmp_path, capsys,
                                                          monkeypatch):
    """Deleting an existing Images folder stops part way (on Windows: a
    file open in another program): exit 1 and the folder is listed in
    deleted_images_folders, in text as deleted or partly deleted."""
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    one_per_pos(src, count=1)
    images = dst / 'Position_1' / 'Images'
    images.mkdir(parents=True)
    for name in ('a_segm.npz', 'b_acdc_output.csv', 'z_open.tif'):
        (images / name).write_text('old')

    rmtree = writers.shutil.rmtree

    def stop_at_the_open_file(path, *args, **kwargs):
        if os.path.normpath(path) != os.path.normpath(images):
            return rmtree(path, *args, **kwargs)
        for name in ('a_segm.npz', 'b_acdc_output.csv'):
            os.remove(os.path.join(path, name))
        raise PermissionError(f'{path}: z_open.tif is used by another '
                              'process')

    monkeypatch.setattr(writers.shutil, 'rmtree', stop_at_the_open_file)
    assert cli(*convert_args(src, dst, '--on-existing', 'overwrite',
                             '--format', 'json')) == 1
    data = json.loads(capsys.readouterr().out)
    assert data['error']['type'] == 'PermissionError'
    assert data['deleted_images_folders'] == [str(images)]
    assert data['positions'] == [] and os.listdir(images) == ['z_open.tif']
    # Text
    for name in ('a_segm.npz', 'b_acdc_output.csv'):
        (images / name).write_text('old')
    assert cli(*convert_args(src, dst, '--on-existing', 'overwrite')) == 1
    out = capsys.readouterr().out
    assert out.startswith('The conversion failed after these changes:\n')
    assert out.endswith(
        f'Deleted or partly deleted Images folders:\n  {images}\n'
    )
    monkeypatch.setattr(writers.shutil, 'rmtree', rmtree)
    assert cli(*convert_args(src, dst, '--on-existing', 'overwrite')) == 0
    assert capsys.readouterr().out.endswith(
        f'Deleted Images folders:\n  {images}\n'
    )


USAGE_CASES = {
    'missing input folder': ('missing', [], 'SourceError'),
    'mixed extensions': ('mixed', [], 'SourceError'),
    'no file with --ext': ('pos', ['--ext', '.nd2'], 'SourceError'),
    'unknown --files': ('pos', ['--files', 'nope.czi'], 'SourceError'),
    'invalid file names': ('invalid', [], 'SourceError'),
    'single-multi-pos with two files': (
        'pos', ['--layout', 'single-multi-pos'], 'SourceError'
    ),
    'unknown channel to save': (
        'pos', ['--save-channels', 'Cy5'], 'SettingsError'
    ),
    'wrong number of names': ('pos', ['--channels', 'A'], 'SettingsError'),
    'unknown position': ('pos', ['--positions', '7'], 'PlanError'),
    'frames beyond the file': (
        'pos', ['--time-range', '2:9'], 'SettingsError'
    ),
    'no image name': ('pos', ['--add-image-name'], 'SettingsError'),
    'existing positions': ('existing', [], 'ExistingPositionsError'),
    'moved raw file exists': ('raw_taken', [], 'PlanError'),
    'destination is a file': ('dst_file', [], 'PlanError'),
    'bad one-per-channel names': (
        'channel_bad', ['--layout', 'one-per-channel'], 'FilenamePatternError'
    ),
    'missing channel file': (
        'channel_missing', ['--layout', 'one-per-channel'],
        'MissingChannelFilesError'
    ),
    'other files with one-per-pos': (
        'pos', ['--other-files', 'copy'], 'UsageError'
    ),
    'symlinks with h5': (
        'pos', ['--use-symlinks', '--image-format', 'h5'], 'UsageError'
    ),
    'time unit without value': ('pos', ['--time-unit', 'min'], 'UsageError'),
    'size unit without size': (
        'pos', ['--physical-size-unit', 'nm'], 'UsageError'
    ),
    'bad wavelengths': ('pos', ['--em-wavelens', 'a,b'], 'UsageError'),
    'NaN and negative wavelengths': (
        'pos', ['--em-wavelens', 'nan,-5'], 'UsageError'
    ),
    'negative pixel size': (
        'pos', ['--physical-size-x', '-0.5'], 'UsageError'
    ),
    'infinite pixel size': ('pos', ['--physical-size-z', 'inf'], 'UsageError'),
    'NaN frame interval': ('pos', ['--time-increment', 'nan'], 'UsageError'),
    'zero numerical aperture': ('pos', ['--lens-na', '0'], 'UsageError'),
    'empty channel name': ('pos', ['--channels', 'A,,B'], 'UsageError'),
    'bad positions': ('pos', ['--positions', 'x'], 'UsageError'),
    'bad time range': ('pos', ['--time-range', '3:2'], 'UsageError'),
    'bad metadata csv': ('csv', [], 'SettingsError'),
}


@pytest.mark.parametrize('case', list(USAGE_CASES))
def test_usage_errors_change_nothing(case, tmp_path, capsys):
    """Exit 2 for every UsageError class, with --move-raw requested and the
    source and destination unchanged."""
    setup, extra, error_type = USAGE_CASES[case]
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    extra = list(extra)
    if setup in ('pos', 'mixed', 'invalid', 'existing', 'raw_taken',
                 'dst_file', 'csv'):
        one_per_pos(src)
    if setup == 'mixed':
        fake(src, 'pos3.nd2')
    elif setup == 'invalid':
        fake(src, 'pos#3.czi')
    elif setup == 'existing':
        assert cli(*convert_args(src, dst)) == 0
    elif setup == 'raw_taken':
        fake(dst / 'raw_microscopy_files', 'pos1.czi')
    elif setup == 'dst_file':
        dst.write_text('not a folder', encoding='utf-8')
    elif setup == 'csv':
        csv_path = tmp_path / 'bad.csv'
        csv_path.write_text('Description,values\nFoo,1\n', encoding='utf-8')
        extra += ['--metadata-csv', csv_path]
    elif setup == 'channel_bad':
        fake(src, 'nopattern.czi')
        fake(src, 'other.czi')
    elif setup == 'channel_missing':
        one_per_channel(src, skip={(2, 'mCh')})
    capsys.readouterr()
    before = snapshot(src, dst)
    assert cli(*convert_args(src, dst, *extra, '--move-raw', '--format',
                             'json')) == 2
    data = json.loads(capsys.readouterr().out)
    assert (data['status'], data['exit_code']) == ('error', 2)
    assert data['error']['type'] == error_type
    assert data['positions'] == [] and data['moved'] == []
    if case == 'negative pixel size':
        assert data['error']['message'] == (
            'Invalid --physical-size-x -0.5: it must be a finite number '
            'greater than 0.'
        )
    assert snapshot(src, dst) == before


def test_runtime_errors_exit_1(tmp_path, capsys):
    src = tmp_path / 'raw'
    one_per_pos(src)
    # a worker failure while writing
    failure = runner.WorkerError(
        'save_positions', 'disk full', returncode=1, error_type='OSError',
        child_traceback='Traceback ...\nOSError: disk full'
    )
    assert cli(*convert_args(src, tmp_path / 'a', '--format', 'json'),
               backend_factory=lambda: SpyBackend(save_positions=failure)
               ) == 1
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert (data['status'], data['exit_code']) == ('error', 1)
    assert data['error'] == {
        'type': 'WorkerError', 'hint': None,
        'message': 'Worker task save_positions failed (exit 1): OSError: '
                   'disk full',
    }
    last_line = captured.err.splitlines()[-1]
    assert last_line == f'See the log file: {data["log_file"]}'
    assert 'Traceback ...' in Path(data['log_file']).read_text('utf-8')
    # nothing written: exit 1
    assert cli(*convert_args(src, tmp_path / 'b', '--format', 'json'),
               backend_factory=SilentSaveBackend) == 1
    data = json.loads(capsys.readouterr().out)
    assert data['error']['type'] == 'NoPositionWrittenError'
    # an unreadable file
    fake(tmp_path / 'bad', 'bad.czi', fail={'stage': 'open'})
    assert cli(*convert_args(tmp_path / 'bad', tmp_path / 'c',
                             '--format', 'json')) == 1
    data = json.loads(capsys.readouterr().out)
    assert data['error']['type'] == 'MetadataReadError'
    assert not (tmp_path / 'c').exists()


def test_partial_result_after_a_failure(tmp_path, capsys):
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    fake(src, 'pos1.czi')
    fake(src, 'pos2.czi', s_offset=1, fail={'stage': 'read'})
    assert cli(*convert_args(src, dst, '--format', 'json')) == 1
    data = json.loads(capsys.readouterr().out)
    assert [p['output'] for p in data['positions']] == ['Position_1']
    assert 'injected read failure' in data['error']['message']
    assert cli(*convert_args(src, tmp_path / 'exp2')) == 1
    out = capsys.readouterr().out
    assert out.startswith('The conversion failed after these changes:\n')
    assert 'Written positions: 1' in out


@pytest.mark.parametrize('failure', [
    KeyboardInterrupt(), errors.ConversionCancelled('cancelled'),
])
def test_interrupt_exit_130(failure, tmp_path, capsys):
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    one_per_pos(src)
    assert cli(*convert_args(src, dst, '--format', 'json'),
               backend_factory=lambda: SpyBackend(read_metadata=failure)
               ) == 130
    data = json.loads(capsys.readouterr().out)
    assert (data['status'], data['exit_code']) == ('error', 130)
    assert data['error']['type'] == type(failure).__name__


def test_stdout_is_only_the_payload(tmp_path, capfd):
    """Prints, C-level writes and reader noise in this process go to
    stderr; stdout holds exactly the JSON."""
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    one_per_pos(src, noise='NOISE')
    assert cli(*convert_args(src, dst, '--format', 'json'),
               backend_factory=NoisyBackend) == 0
    captured = capfd.readouterr()
    data = json.loads(captured.out)
    assert captured.out == json.dumps(data, indent=2) + '\n'
    for text in ('NOISE', 'PRINTED', 'C-LEVEL'):
        assert text in captured.err


@pytest.mark.parametrize('argv', [
    ['metadata', '--format', 'text'], ['metadata', '--format', 'json'],
    ['metadata', '--format', 'csv'], ['convert', '--format', 'text'],
])
def test_payload_bytes_use_lf_only(argv, tmp_path, capfdbinary):
    src = tmp_path / 'raw'
    one_per_pos(src, channel_names=['GFP\u00b5', 'mCherry'])
    if argv[0] == 'metadata':
        argv = ['metadata', '-i', src / 'pos1.czi', *argv[1:]]
    else:
        argv = convert_args(src, tmp_path / 'exp', *argv[1:])
    assert cli(*argv) == 0
    out = capfdbinary.readouterr().out
    assert out.endswith(b'\n')
    assert b'\r\r' not in out and b'\r\n' not in out


def test_payload_stream_is_lossless_on_a_cp1252_pipe(capfdbinary):
    guard = data_cli._StdoutGuard('cp1252')
    with guard:
        print('noise')
        os.write(1, b'c-level\n')
    guard.write('PhysicalSizeUnit: \u03bcm\nok\n')
    captured = capfdbinary.readouterr()
    assert captured.out == b'PhysicalSizeUnit: \\u03bcm\nok\n'
    assert b'noise' in captured.err and b'c-level' in captured.err


def test_csv_payload_is_utf8_whatever_the_stdout_encoding(monkeypatch):
    """`convert --metadata-csv` reads UTF-8, so `metadata --format csv >
    file` must not follow a cp1252 standard output (the round trip through
    a child: test_data_cli_subprocess)."""
    monkeypatch.setattr(
        sys, 'stdout', io.TextIOWrapper(io.BytesIO(), encoding='cp1252')
    )
    encodings = {
        name: data_cli._payload_encoding(name)
        for name in ('csv', 'text', 'json')
    }
    assert encodings == {'csv': 'utf-8', 'text': 'cp1252', 'json': 'ascii'}


def test_child_process_count_cli(tmp_path, capsys):
    """convert = one read_metadata + one check_positions + one
    save_positions task; a dry run stops after the check."""
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    one_per_pos(src, count=3)
    spy = SpyBackend()
    assert cli(*convert_args(src, dst, '--dry-run'),
               backend_factory=lambda: spy) == 0
    assert spy.calls == ['read_metadata', 'check_positions']
    spy = SpyBackend()
    assert cli(*convert_args(src, dst), backend_factory=lambda: spy) == 0
    assert spy.calls == ['read_metadata', 'check_positions', 'save_positions']
    spy = SpyBackend()
    assert cli('metadata', '-i', src / 'pos1.czi',
               backend_factory=lambda: spy) == 0
    assert spy.calls == ['read_metadata']


@pytest.mark.parametrize('keep', [False, True])
def test_keep_temp_sets_the_runner_options(keep, tmp_path, monkeypatch,
                                           capsys):
    seen = {}

    def default_backend(*, for_gui, on_output=None, options=None):
        seen.update(for_gui=for_gui, options=options, on_output=on_output)
        return fake_backend()

    monkeypatch.setattr(backends, 'default_backend', default_backend)
    src = tmp_path / 'raw'
    one_per_pos(src)
    argv = convert_args(src, tmp_path / 'exp')
    if keep:
        argv.append('--keep-temp')
    assert data_cli.main([str(arg) for arg in argv]) == 0
    assert seen['for_gui'] is False
    assert seen['options'].keep_run_dir is keep
    assert seen['on_output'] == data_cli.logger.debug


# ========================================================= reader packages
def test_missing_reader_package_without_a_terminal(tmp_path, monkeypatch,
                                                   capsys):
    monkeypatch.setattr(install, 'missing_packages', lambda specs: [
        spec for spec in specs if spec.pypi_name == 'bioio-czi'
    ])
    monkeypatch.setattr(data_cli, '_isatty', lambda stream: False)
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    one_per_pos(src)
    before = snapshot(src, dst)
    backend = ProductionLikeBackend()
    assert cli(*convert_args(src, dst, '--format', 'json'),
               backend_factory=lambda: backend) == 1
    data = json.loads(capsys.readouterr().out)
    assert data['error']['type'] == 'MissingDependencyError'
    assert 'pip install bioio-czi' in data['error']['message']
    assert '--yes' in data['error']['hint']
    assert backend.calls == []
    assert snapshot(src, dst) == before
    assert cli('metadata', '-i', src / 'pos1.czi',
               backend_factory=lambda: backend) == 1
    assert backend.calls == []


@pytest.mark.parametrize('tty', [False, True])
def test_reader_packages_installed_with_yes_or_in_a_terminal(
        tty, tmp_path, monkeypatch, capsys):
    """--yes installs without asking; in a terminal the GUI helpers ask
    (on stderr). Plugins go through install_reader_dependencies (GUI version
    pins), the others through check_install_package."""
    still_missing = {'bioio-czi', 'zarr'}
    calls = []

    def install_reader_dependencies(path, exception, qparent=None, *,
                                    auto_install=False, logger_func=print):
        calls.append(('plugin', os.path.basename(path), auto_install))
        still_missing.discard('bioio-czi')

    def check_install_package(name, **kwargs):
        calls.append(('package', name, kwargs['auto_install']))
        still_missing.discard(name)

    monkeypatch.setattr(install, 'missing_packages', lambda specs: [
        spec for spec in specs if spec.pypi_name in still_missing
    ])
    monkeypatch.setattr(
        install, 'install_reader_dependencies', install_reader_dependencies
    )
    monkeypatch.setattr(myutils, 'check_install_package',
                        check_install_package)
    monkeypatch.setattr(data_cli, '_isatty', lambda stream: tty)
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    one_per_pos(src)
    argv = convert_args(src, dst) + ([] if tty else ['--yes'])
    assert cli(*argv, backend_factory=ProductionLikeBackend) == 0
    assert calls == [
        ('package', 'zarr', not tty), ('plugin', 'pos1.czi', not tty)
    ]
    assert (dst / 'Position_2').is_dir()


# ============================================================ common options
def quick_invocation(command: str, tmp_path) -> list:
    if command == 'metadata':
        fake(tmp_path / 'raw', 'pos1.czi')
        return ['metadata', '-i', tmp_path / 'raw' / 'pos1.czi']
    if command == 'convert':
        one_per_pos(tmp_path / 'raw')
        return convert_args(tmp_path / 'raw', tmp_path / 'exp', '--dry-run')
    loose_tiffs(tmp_path / 'tiffs', channels=(None,), timepoints=(None,),
                pattern='pos{pos}.tif')
    return ['restructure', '-i', tmp_path / 'tiffs', '-o', tmp_path / 'exp',
            '--dry-run']


@pytest.mark.parametrize('command', ['metadata', 'convert', 'restructure'])
@pytest.mark.parametrize('option', [
    '-v', '--verbose', '-q', '--quiet', '--log-file', '--no-log-file',
    '--yes', '--debug',
])
def test_common_options(command, option, tmp_path, capsys):
    from cellacdc import config
    argv = quick_invocation(command, tmp_path)
    log_path = tmp_path / 'my logs' / 'run.log'
    extra = [option, log_path] if option == '--log-file' else [option]
    assert cli(*argv, *extra) == 0
    err = capsys.readouterr().err
    header = f'acdc-data {command} (Cell-ACDC'
    assert (header in err) is (option in ('-v', '--verbose'))
    if option in ('-q', '--quiet'):
        assert all(
            line.startswith(('WARNING: ', 'ERROR: '))
            for line in err.splitlines()
        ), err
    default_logs = tmp_path / 'logs'
    logs = sorted(os.listdir(default_logs)) if default_logs.exists() else []
    if option == '--log-file':
        assert header in log_path.read_text(encoding='utf-8')
        assert logs == []
    elif option == '--no-log-file':
        assert logs == []
    else:
        assert len(logs) == 1 and re.fullmatch(LOG_NAME % command, logs[0])
    if option == '--debug':
        assert config.parser_args['debug'] is True


# ============================================================== restructure
def test_restructure_dry_run_then_run(tmp_path, capsys):
    src, dst = tmp_path / 'tiffs', tmp_path / 'exp'
    loose_tiffs(src)
    before = snapshot(src, dst)
    argv = [
        'restructure', '-i', src, '-o', dst, *PER_FILE_CHANNELS,
        '--basename', 'exp1', '--physical-size-x', '0.2',
        '--physical-size-y', '0.3', '--physical-size-z', '0.5',
        '--time-increment', '2', '--time-unit', 'min',
    ]
    assert cli(*argv, '--dry-run') == 0
    out = capsys.readouterr().out
    assert out.startswith('Dry run: nothing was written.\n')
    assert '  pos1 -> Position_1 (T=3, Z=1, Y=4, X=5)' in out
    assert cli(*argv, '--dry-run', '--format', 'json') == 0
    plan = json.loads(capsys.readouterr().out)
    assert snapshot(src, dst) == before
    assert list(plan) == ['schema', *PLAN_KEYS, *RESTRUCTURE_RESULT_KEYS]
    assert plan['schema'] == 'acdc-data/restructure/1'
    assert (plan['position_folders'], plan['image_files']) == ({}, [])
    assert cli(*argv, '--format', 'json') == 0
    result = json.loads(capsys.readouterr().out)
    assert (result['status'], result['exit_code'], result['error']) == (
        'ok', 0, None
    )
    assert {key: result[key] for key in PLAN_KEYS} == {
        key: plan[key] for key in PLAN_KEYS
    }
    assert sorted(result['position_folders']) == ['pos1', 'pos2']
    images = dst / 'Position_1' / 'Images'
    assert sorted(result['image_files'])[:2] == [
        str(images / 'exp1_GFP.tif'), str(images / 'exp1_mCherry.tif')
    ]
    values = writers.read_metadata_csv(str(images / 'exp1_metadata.csv'))
    assert float(values['TimeIncrement']) == 120.0
    assert [float(values[f'PhysicalSize{axis}']) for axis in 'XYZ'] == [
        0.2, 0.3, 0.5
    ]
    assert [o[1] for o in plane_origins(images / 'exp1_mCherry.tif')] == [
        1, 2, 3
    ]


def test_restructure_recipes_and_flag_overrides(tmp_path, capsys):
    import cellacdc
    src = tmp_path / 'tiffs'
    loose_tiffs(src)
    recipes = cellacdc.restruct_files_recipes_path
    metadata = {
        'PhysicalSizeX': 0.5, 'PhysicalSizeY': 0.5, 'PhysicalSizeZ': 1.0,
        'TimeIncrement': 5.0, 'TimeIncrementUnit': 'seconds',
    }
    restructure.save_recipe(
        os.path.join(recipes, restructure.LAST_SETUP_FILENAME),
        DIALOG_RECIPE, metadata
    )

    def csv_of(dst, prefix=''):
        return writers.read_metadata_csv(str(
            dst / 'Position_1' / 'Images' / f'{prefix}metadata.csv'
        ))

    assert cli('restructure', '-i', src, '-o', tmp_path / 'a',
               '--recipe', 'last') == 0
    values = csv_of(tmp_path / 'a')
    assert (float(values['TimeIncrement']), values['channel_1_name']) == (
        5.0, 'mCherry'
    )
    # flags replace the matching recipe entries
    assert cli('restructure', '-i', src, '-o', tmp_path / 'b', '--recipe',
               'last', '--time-unit', 'min', '--channel', 'G=_GFP',
               '--channel', 'M=_mCh') == 0
    values = csv_of(tmp_path / 'b')
    assert (float(values['TimeIncrement']), values['channel_1_name']) == (
        300.0, 'M'
    )
    # a file path and a saved NAME
    path = tmp_path / 'my recipe.json'
    restructure.save_recipe(str(path), DIALOG_RECIPE, metadata)
    assert cli('restructure', '-i', src, '-o', tmp_path / 'c', '--recipe',
               path) == 0
    restructure.save_recipe(
        os.path.join(recipes, f'{restructure.RECIPE_PREFIX}mine.json'),
        DIALOG_RECIPE, metadata
    )
    assert cli('restructure', '-i', src, '-o', tmp_path / 'd', '--recipe',
               'mine') == 0
    capsys.readouterr()
    assert cli('restructure', '-i', src, '-o', tmp_path / 'e', '--recipe',
               'nope') == 2
    assert 'saved recipes: mine' in capsys.readouterr().err
    assert not (tmp_path / 'e').exists()


def test_restructure_save_recipe_uses_the_dialog_format(tmp_path, capsys):
    import cellacdc
    src, dst = tmp_path / 'tiffs', tmp_path / 'exp'
    synthetic.make_loose_tiff_set(
        str(src), pattern='pos{pos}.tif', positions=(1, 2), file_axes='CTYX',
        C=2, T=3
    )
    before = snapshot(src, dst)
    assert cli('restructure', '-i', src, '-o', dst, '--channel', 'A',
               '--channel', 'B', '--channel-axis', '0', '--time-axis', '1',
               '--time-increment', '1.5', '--time-unit', 'h',
               '--save-recipe', 'axes', '--dry-run') == 0
    assert snapshot(src, dst) == before
    path = os.path.join(
        cellacdc.restruct_files_recipes_path,
        'datastruct_restructure_recipe_axes.json'
    )
    expected = {
        'channels': {
            'mode': 'in_file', 'axis': '0', 'count': 2,
            'names': [{'name': 'A'}, {'name': 'B'}],
        },
        'timepoints': {'mode': 'in_file', 'axis': '1'},
        'metadata': {
            'PhysicalSizeX': 1.0, 'PhysicalSizeY': 1.0, 'PhysicalSizeZ': 1.0,
            'TimeIncrement': 1.5, 'TimeIncrementUnit': 'hours',
        },
    }
    with open(path, encoding='utf-8') as file:
        assert file.read() == json.dumps(expected, indent=2)
    assert restructure.load_recipe(path) == expected
    # the saved recipe reproduces the run
    assert cli('restructure', '-i', src, '-o', dst, '--recipe', 'axes') == 0
    images = dst / 'Position_2' / 'Images'
    values = writers.read_metadata_csv(str(images / 'metadata.csv'))
    assert float(values['TimeIncrement']) == 5400.0
    assert [o[1:3] for o in plane_origins(images / 'B.tif')] == [
        (0, 1), (1, 1), (2, 1)
    ]
    # --save-recipe with a path
    saved = tmp_path / 'saved' / 'r.json'
    assert cli('restructure', '-i', src, '-o', tmp_path / 'x', '--channel',
               'A', '--channel', 'B', '--channel-axis', '0', '--time-axis',
               '1', '--save-recipe', saved, '--dry-run') == 0
    assert restructure.load_recipe(str(saved))['channels']['axis'] == '0'


@pytest.mark.parametrize('flag, value, pattern, axes, size_z', [
    ('--z-marker', '_z', 'pos{pos}_z{z}.tif', 'YX', 2),
    ('--z-axis', '0', 'pos{pos}.tif', 'ZYX', 3),
])
def test_restructure_z_slices(flag, value, pattern, axes, size_z, tmp_path,
                              capsys):
    src = tmp_path / 'tiffs'
    synthetic.make_loose_tiff_set(
        str(src), pattern=pattern, positions=(1,), file_axes=axes, Z=3,
        z_slices=(1, 2) if flag == '--z-marker' else (None,)
    )
    assert cli('restructure', '-i', src, '-o', tmp_path / 'exp', flag, value,
               '--dry-run', '--format', 'json') == 0
    plan = json.loads(capsys.readouterr().out)
    assert [p['size_z'] for p in plan['positions']] == [size_z]


RESTRUCTURE_USAGE_CASES = {
    'bad basename': (['--basename', 'a b'], 'RestructureOptionsError'),
    'unknown --files': (
        ['--files', 'pos1_GFP_t1.tif', 'typo.tif'], 'SourceError'
    ),
    # an existing file on a case-insensitive file system (Windows, macOS),
    # but the core matches names exactly and would drop it silently
    '--files in another case': (
        ['--files', 'pos1_GFP_t1.tif', 'POS2_GFP_t1.tif'], 'SourceError'
    ),
    'negative pixel size': (['--physical-size-y', '-1'], 'UsageError'),
    'NaN frame interval': (['--time-increment', 'nan'], 'UsageError'),
    'missing source': (['-i', '{missing}'], 'SourceError'),
    'ids on some channels only': (
        ['--channel', 'GFP=_GFP', '--channel', 'mCherry'], 'UsageError'
    ),
    'channel axis without names': (['--channel-axis', '0'], 'UsageError'),
    'channel axis with ids': (
        ['--channel', 'A=_GFP', '--channel-axis', '0'], 'UsageError'
    ),
    'several names, channels not present': (
        ['--channel', 'A', '--channel', 'B'], 'RestructureOptionsError'
    ),
    'unknown recipe': (['--recipe', 'nope'], 'RestructureOptionsError'),
    'existing positions': ([], 'ExistingPositionsError'),
}


@pytest.mark.parametrize('case', list(RESTRUCTURE_USAGE_CASES))
def test_restructure_usage_errors_change_nothing(case, tmp_path, capsys):
    extra, error_type = RESTRUCTURE_USAGE_CASES[case]
    src, dst = tmp_path / 'tiffs', tmp_path / 'exp'
    loose_tiffs(src)
    if case == 'existing positions':
        assert cli('restructure', '-i', src, '-o', dst,
                   *PER_FILE_CHANNELS) == 0
    extra = [str(tmp_path / 'nope') if arg == '{missing}' else arg
             for arg in extra]
    capsys.readouterr()
    before = snapshot(src, dst)
    assert cli('restructure', '-i', src, '-o', dst, *extra, '--format',
               'json') == 2
    data = json.loads(capsys.readouterr().out)
    assert (data['status'], data['exit_code']) == ('error', 2)
    assert data['error']['type'] == error_type
    assert 'positions' not in data      # no plan
    assert (data['position_folders'], data['image_files']) == ({}, [])
    if case == 'existing positions':
        assert '--on-existing: overwrite or create-new' in (
            data['error']['hint']
        )
    elif case == '--files in another case':
        assert '- POS2_GFP_t1.tif (the file is named pos2_GFP_t1.tif)' in (
            data['error']['message']
        )
    assert snapshot(src, dst) == before


def test_restructure_exclusive_dimension_flags(tmp_path, capsys):
    src, dst = tmp_path / 'tiffs', tmp_path / 'exp'
    loose_tiffs(src)
    before = snapshot(src, dst)
    assert cli('restructure', '-i', src, '-o', dst, '--time-marker', '_t',
               '--time-axis', '0') == 2
    assert cli('restructure', '-i', src, '-o', dst, '--z-marker', '_z',
               '--z-axis', '0') == 2
    assert 'not allowed with argument' in capsys.readouterr().err
    assert snapshot(src, dst) == before


def test_restructure_on_existing(tmp_path, capsys):
    src, dst = tmp_path / 'tiffs', tmp_path / 'exp'
    loose_tiffs(src)
    argv = ['restructure', '-i', src, '-o', dst, *PER_FILE_CHANNELS,
            '--format', 'json']
    assert cli(*argv) == 0
    capsys.readouterr()
    assert cli(*argv, '--on-existing', 'overwrite') == 0
    data = json.loads(capsys.readouterr().out)
    assert data['deleted_images_folders'] == [
        str(dst / f'Position_{n}' / 'Images') for n in (1, 2)
    ]
    assert cli(*argv, '--on-existing', 'create-new') == 0
    data = json.loads(capsys.readouterr().out)
    assert sorted(data['position_folders'].values()) == [
        str(dst / 'Position_3'), str(dst / 'Position_4')
    ]


def test_restructure_create_new_collides_with_empty_position(tmp_path,
                                                             capsys):
    """create-new numbers after the last valid Position: a following
    Position folder without images collides. The hint names it and does not
    suggest overwrite, which would delete the Images of Position_1 and
    Position_2 instead."""
    src, dst = tmp_path / 'tiffs', tmp_path / 'exp'
    loose_tiffs(src)
    argv = ['restructure', '-i', src, '-o', dst, *PER_FILE_CHANNELS,
            '--format', 'json']
    assert cli(*argv) == 0
    os.makedirs(dst / 'Position_3' / 'Images')
    capsys.readouterr()
    before = snapshot(src, dst)
    assert cli(*argv, '--on-existing', 'create-new') == 2
    data = json.loads(capsys.readouterr().out)
    assert data['error']['type'] == 'ExistingPositionsError'
    hint = data['error']['hint']
    assert 'Position_3' in hint
    assert 'overwrite' not in hint and 'create-new or' not in hint
    assert snapshot(src, dst) == before


def test_restructure_write_error_exit_1(tmp_path, monkeypatch, capsys):
    def broken(plan, *, signals=None):
        raise OSError('disk full')

    monkeypatch.setattr(restructure, 'execute_restructure_plan', broken)
    src, dst = tmp_path / 'tiffs', tmp_path / 'exp'
    loose_tiffs(src)
    assert cli('restructure', '-i', src, '-o', dst, *PER_FILE_CHANNELS,
               '--format', 'json') == 1
    data = json.loads(capsys.readouterr().out)
    assert (data['status'], data['exit_code']) == ('error', 1)
    assert data['error']['type'] == 'OSError'
    assert len(data['positions']) == 2
    assert data['position_folders'] is None and data['image_files'] is None


# ==================================================== real worker children
@pytest.fixture
def worker_hooks(monkeypatch):
    """The production backend with the fake reader in the children."""
    monkeypatch.setenv(runner.TEST_HOOKS_ENV, '1')
    monkeypatch.setenv(
        backends.READER_FACTORY_ENV, fake_bioio.FAKE_BIOIMAGE_PATH
    )


@conditions.subprocess_heavy
def test_parent_never_opens_raw_cli(tmp_path, monkeypatch, worker_hooks,
                                    capsys):
    def refuse(*args, **kwargs):
        raise AssertionError('the parent process opened a raw file')

    monkeypatch.setattr(reader.ImageReader, '__init__', refuse)
    monkeypatch.setattr(fake_bioio.FakeBioImage, '__init__', refuse)
    try:
        import bioio
    except ImportError:
        pass
    else:
        monkeypatch.setattr(bioio.BioImage, '__init__', refuse)
    tasks = []
    run_task = runner.run_task

    def counting_run_task(task, *args, **kwargs):
        tasks.append(task)
        return run_task(task, *args, **kwargs)

    monkeypatch.setattr(runner, 'run_task', counting_run_task)
    src, dst = tmp_path / 'raw', tmp_path / 'exp'
    one_per_pos(src)
    assert data_cli.main([
        'convert', '-i', str(src), '-o', str(dst), '--layout', 'one-per-pos',
        '--format', 'json',
    ]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [p['output'] for p in data['positions']] == [
        'Position_1', 'Position_2'
    ]
    assert tasks == ['read_metadata', 'check_positions', 'save_positions']


@conditions.subprocess_heavy
def test_relative_paths(tmp_path, monkeypatch, worker_hooks, capsys):
    monkeypatch.chdir(tmp_path)
    one_per_pos(tmp_path / 'raw', count=1)
    assert data_cli.main([
        'convert', '--input', 'raw', '--output', 'exp', '--layout',
        'one-per-pos', '--move-raw', '--use-symlinks', '--format', 'json',
    ]) == 0
    data = json.loads(capsys.readouterr().out)
    assert os.path.samefile(data['source'], tmp_path / 'raw')
    assert os.path.isabs(data['destination'])
    ini = configparser.ConfigParser()
    images = tmp_path / 'exp' / 'Position_1' / 'Images'
    ini.read(images / 'pos1_s1_symlink.ini')
    moved = tmp_path / 'exp' / 'raw_microscopy_files' / 'pos1.czi'
    for section in ini.sections():
        path = ini[section]['source_filepath']
        assert os.path.isabs(path) and os.path.samefile(path, moved)
    assert not (tmp_path / 'raw' / 'pos1.czi').exists()


# ============================================================= registration
def test_console_script_is_registered():
    tomllib = pytest.importorskip('tomllib')
    data = tomllib.loads(
        (REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8')
    )
    assert data['project']['scripts']['acdc-data'] == (
        'cellacdc.data_cli:main'
    )


COMMON = 'test_common_options'
FLAG_TESTS = {
    ('acdc-data', '--version'): 'test_version_and_usage_errors',
    # metadata
    ('metadata', '-i'): 'test_metadata_text',
    ('metadata', '--input'): 'test_metadata_json',
    ('metadata', '--format'): 'test_metadata_json',
    ('metadata', '-o'): 'test_metadata_output_folder',
    ('metadata', '--output'): 'test_metadata_output_folder',
    ('metadata', '--basename'): 'test_metadata_output_folder',
    ('metadata', '--include-xml'): 'test_metadata_json',
    # convert
    ('convert', '-i'): 'convert_args',
    ('convert', '--input'): 'test_relative_paths',
    ('convert', '-o'): 'convert_args',
    ('convert', '--output'): 'test_relative_paths',
    ('convert', '--layout'): 'convert_args',
    ('convert', '--files'): 'test_convert_file_selection_and_renames',
    ('convert', '--ext'): 'test_convert_file_selection_and_renames',
    ('convert', '--rename-invalid'): 'test_convert_file_selection_and_renames',
    ('convert', '--other-files'): 'test_convert_one_per_channel_other_files',
    ('convert', '--on-existing'): 'test_on_existing',
    ('convert', '--move-raw'): 'test_raw_files_stay_unless_move_raw',
    ('convert', '--no-move-raw'): 'test_raw_files_stay_unless_move_raw',
    ('convert', '--use-symlinks'): 'test_convert_symlinks_and_lazy_load',
    ('convert', '--image-format'): 'test_convert_metadata_flags',
    ('convert', '--lazy-load'): 'test_convert_symlinks_and_lazy_load',
    ('convert', '--no-lazy-load'): 'test_convert_symlinks_and_lazy_load',
    ('convert', '--lens-na'): 'test_convert_metadata_flags',
    ('convert', '--time-increment'): 'test_convert_metadata_flags',
    ('convert', '--time-unit'): 'test_convert_metadata_flags',
    ('convert', '--physical-size-x'): 'test_convert_metadata_flags',
    ('convert', '--physical-size-y'): 'test_convert_metadata_flags',
    ('convert', '--physical-size-z'): 'test_convert_metadata_flags',
    ('convert', '--physical-size-unit'): 'test_convert_metadata_flags',
    ('convert', '--channels'): 'test_convert_metadata_flags',
    ('convert', '--em-wavelens'): 'test_convert_metadata_flags',
    ('convert', '--save-channels'): 'test_convert_metadata_flags',
    ('convert', '--positions'): 'test_convert_metadata_flags',
    ('convert', '--time-range'): 'test_convert_metadata_flags',
    ('convert', '--add-image-name'): 'USAGE_CASES',
    ('convert', '--metadata-csv'): 'test_metadata_csv_round_trip_into_convert',
    ('convert', '--dry-run'): 'test_convert_dry_run_changes_nothing',
    ('convert', '--format'): 'test_convert_json_result_and_log_file',
    ('convert', '--keep-temp'): 'test_keep_temp_sets_the_runner_options',
    ('convert', '--size-t'): 'test_removed_flags_exit_2_with_hint',
    ('convert', '--size-z'): 'test_removed_flags_exit_2_with_hint',
    ('convert', '--size-c'): 'test_removed_flags_exit_2_with_hint',
    ('convert', '--size-s'): 'test_removed_flags_exit_2_with_hint',
    ('convert', '--trust-metadata'): 'test_removed_flags_exit_2_with_hint',
    ('convert', '--no-trust-metadata'): 'test_removed_flags_exit_2_with_hint',
    ('convert', '--basename'): 'test_removed_flags_exit_2_with_hint',
    # restructure
    ('restructure', '-i'): 'test_restructure_dry_run_then_run',
    ('restructure', '--input'): 'test_restructure_input_and_output_long',
    ('restructure', '-o'): 'test_restructure_dry_run_then_run',
    ('restructure', '--output'): 'test_restructure_input_and_output_long',
    ('restructure', '--recipe'): 'test_restructure_recipes_and_flag_overrides',
    ('restructure', '--channel'): 'PER_FILE_CHANNELS',
    ('restructure', '--channel-axis'):
        'test_restructure_save_recipe_uses_the_dialog_format',
    ('restructure', '--time-marker'): 'PER_FILE_CHANNELS',
    ('restructure', '--time-axis'):
        'test_restructure_save_recipe_uses_the_dialog_format',
    ('restructure', '--z-marker'): 'test_restructure_z_slices',
    ('restructure', '--z-axis'): 'test_restructure_z_slices',
    ('restructure', '--basename'): 'test_restructure_dry_run_then_run',
    ('restructure', '--files'): 'RESTRUCTURE_USAGE_CASES',
    ('restructure', '--on-existing'): 'test_restructure_on_existing',
    ('restructure', '--physical-size-x'): 'test_restructure_dry_run_then_run',
    ('restructure', '--physical-size-y'): 'test_restructure_dry_run_then_run',
    ('restructure', '--physical-size-z'): 'test_restructure_dry_run_then_run',
    ('restructure', '--time-increment'): 'test_restructure_dry_run_then_run',
    ('restructure', '--time-unit'): 'test_restructure_dry_run_then_run',
    ('restructure', '--dry-run'): 'test_restructure_dry_run_then_run',
    ('restructure', '--save-recipe'):
        'test_restructure_save_recipe_uses_the_dialog_format',
    ('restructure', '--format'): 'test_restructure_dry_run_then_run',
    ('restructure', '--layout'): 'test_removed_flags_exit_2_with_hint',
    ('restructure', '--action'): 'test_removed_flags_exit_2_with_hint',
    ('restructure', '--segm-folder'): 'test_removed_flags_exit_2_with_hint',
    ('restructure', '--channels'): 'test_removed_flags_exit_2_with_hint',
}
for _command in ('metadata', 'convert', 'restructure'):
    for _flag in ('-v', '--verbose', '-q', '--quiet', '--log-file',
                  '--no-log-file', '--yes', '--debug'):
        FLAG_TESTS[(_command, _flag)] = COMMON


def test_restructure_input_and_output_long(tmp_path, capsys):
    src = tmp_path / 'tiffs'
    loose_tiffs(src, channels=(None,), timepoints=(None,),
                pattern='pos{pos}.tif')
    assert cli('restructure', '--input', src, '--output', tmp_path / 'exp',
               '--format', 'json') == 0
    data = json.loads(capsys.readouterr().out)
    assert data['source_folder'] == str(src)
    assert len(data['image_files']) == 2


def test_every_flag_is_exercised():
    """Every option of the parser (hidden ones included) is used by the test
    (or test data) named in FLAG_TESTS, whose source contains it."""
    parser = data_cli.build_parser()
    helps = {'-h', '--help'}
    flags = {
        ('acdc-data', option)
        for action in parser._actions for option in action.option_strings
        if option not in helps
    }
    for name, subparser in subparsers(parser).items():
        flags |= {
            (name, option)
            for action in subparser._actions
            for option in action.option_strings if option not in helps
        }
    assert flags == set(FLAG_TESTS), sorted(flags ^ set(FLAG_TESTS))
    for (command, flag), name in FLAG_TESTS.items():
        target = globals()[name]
        source = (
            inspect.getsource(target) if callable(target) else repr(target)
        )
        assert repr(flag) in source, (command, flag, name)
