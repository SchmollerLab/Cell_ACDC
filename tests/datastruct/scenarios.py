"""Characterization scenarios and golden comparison.

Three golden families, all captured on main @ 7c0f7532 by
`tools/make_goldens.py` and stored as JSON under `golden/`:

- Module 0 scenarios S01-S18 (`MODULE0`), run by
  `harness_module0.run_module0_scenario` -> `golden/module0/<id>.json`.
- Re-structure cases RA-RO, all 27 mode combinations
  (`combo_<c>-<t>-<z>`) and edge cases (`extra_*`) (`RESTRUCTURE`), run by
  `harness_restructure.run_restructure_case` -> `golden/restructure/<id>.json`.
- The legacy metadata cascade (`CASCADE_*`), captured by
  `harness_module0.capture_cascade` -> `golden/cascade.json`.

Inputs are built at run time with `synthetic` (goldens hold manifests only).

`ExpectedChange` and `assert_golden` implement the expected-change
registries (`expected_module0_changes.CHANGES`,
`expected_restructure_changes.CHANGES`): every diff between a golden and the
actual result must be claimed by an entry listed for that scenario, and every
entry listed for that scenario must claim at least one diff.
"""
from __future__ import annotations

import difflib
import fnmatch
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Sequence

from . import fake_omexml, manifest, synthetic

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden')
MAIN_COMMIT = '7c0f7532'


# ------------------------------------------------------- golden comparison
@dataclass(frozen=True)
class ExpectedChange:
    """One registry entry of an intentional behaviour change.

    gv           change id, e.g. 'GV-03' or 'GV-R1'.
    scenarios    scenario/case ids (fnmatch patterns) whose diffs it claims.
    key_glob     fnmatch pattern(s) matched against `manifest.Diff.key`
                 (e.g. 'dst/Position_1/Images/*_metadata.csv/rows/*').
    predicate    optional `predicate(diff) -> bool` narrowing the claim.
    description  one line for the PR description.
    """
    gv: str
    scenarios: tuple[str, ...]
    key_glob: str | tuple[str, ...]
    predicate: Callable[[manifest.Diff], bool] | None = None
    description: str = ''

    def applies_to(self, scenario_id: str) -> bool:
        return any(fnmatch.fnmatchcase(scenario_id, pattern)
                   for pattern in self.scenarios)

    def claims(self, diff: manifest.Diff) -> bool:
        globs = (self.key_glob,) if isinstance(self.key_glob, str) \
            else tuple(self.key_glob)
        if not any(fnmatch.fnmatchcase(diff.key, glob) for glob in globs):
            return False
        return self.predicate is None or bool(self.predicate(diff))


def check_registry(changes: Sequence[ExpectedChange],
                   known_ids: Iterable[str]) -> list[str]:
    """Problems of a registry: empty fields, or scenario patterns that match
    no known scenario id ([] = valid)."""
    known_ids = list(known_ids)
    problems = []
    for change in changes:
        if not change.gv or not change.scenarios or not change.key_glob:
            problems.append(f'{change!r}: gv, scenarios and key_glob are '
                            'required')
        for pattern in change.scenarios:
            if not fnmatch.filter(known_ids, pattern):
                problems.append(
                    f'{change.gv}: scenario pattern {pattern!r} matches no '
                    'scenario'
                )
    return problems


def _short(value, limit: int = 300) -> str:
    text = json.dumps(manifest._jsonable(value), ensure_ascii=False,
                      sort_keys=True, default=repr) \
        if value is not manifest.ABSENT else '<absent>'
    return text if len(text) <= limit else text[:limit] + '...'


def describe_diff(diff: manifest.Diff, max_lines: int = 40) -> str:
    """Readable description of one diff (unified diff for lists)."""
    if isinstance(diff.old, list) and isinstance(diff.new, list):
        old = [_short(item, 200) for item in diff.old]
        new = [_short(item, 200) for item in diff.new]
        lines = list(difflib.unified_diff(
            old, new, 'golden', 'actual', lineterm='', n=1
        ))[2:]
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f'... ({len(lines)} lines)']
        return f'{diff.key}: list changed\n    ' + '\n    '.join(lines)
    return f'{diff.key}: {_short(diff.old)} -> {_short(diff.new)}'


def assert_golden(scenario_id: str, golden: Mapping, actual: Mapping,
                  changes: Sequence[ExpectedChange]) -> list[manifest.Diff]:
    """Compare `actual` with `golden` (`manifest.diff_manifests`, which
    ignores '_info' and '@env' keys) under the registry `changes`.

    Raises AssertionError listing every diff not claimed by an entry that
    applies to `scenario_id`, and every applicable entry that claims no diff.
    Returns the (all claimed) diffs."""
    diffs = manifest.diff_manifests(golden, actual)
    applicable = [c for c in changes if c.applies_to(scenario_id)]
    unclaimed = [d for d in diffs if not any(c.claims(d) for c in applicable)]
    unused = [c for c in applicable if not any(c.claims(d) for d in diffs)]
    if unclaimed or unused:
        lines = [f'{scenario_id}: golden mismatch '
                 f'({len(diffs)} diffs, {len(unclaimed)} unclaimed, '
                 f'{len(unused)} unused registry entries)']
        for diff in unclaimed[:30]:
            lines.append('  UNCLAIMED ' + describe_diff(diff))
        if len(unclaimed) > 30:
            lines.append(f'  ... and {len(unclaimed) - 30} more')
        for change in unused:
            lines.append(f'  UNUSED {change.gv} {change.key_glob!r}: '
                         f'{change.description}')
        env = (golden.get(manifest.ENV_KEY), actual.get(manifest.ENV_KEY))
        lines.append(f'  golden env: {_short(env[0], 400)}')
        lines.append(f'  actual env: {_short(env[1], 400)}')
        raise AssertionError('\n'.join(lines))
    return diffs


def golden_path(kind: str, case_id: str | None = None) -> str:
    """'module0'/'restructure' + id -> golden/<kind>/<id>.json;
    'cascade' -> golden/cascade.json."""
    if kind == 'cascade':
        return os.path.join(GOLDEN_DIR, 'cascade.json')
    return os.path.join(GOLDEN_DIR, kind, f'{case_id}.json')


def load_golden(kind: str, case_id: str | None = None) -> dict:
    return manifest.load_json(golden_path(kind, case_id))


# ---------------------------------------------------------- Module 0 model
@dataclass(frozen=True)
class DialogAnswer:
    """Scripted answer to one `QDialogMetadataXML`.

    `edits` are applied in order as (key, value) pairs; keys: 'SizeT',
    'SizeZ', 'SizeC', 'SizeS', 'LensNA', 'TimeIncrement',
    'TimeIncrementUnit', 'PhysicalSizeX/Y/Z', 'PhysicalSizeUnit', 'to_h5',
    'timeRange' ((t0, t1)), 'positions' (labels to select), 'addImageName'
    (the visible checkbox), 'chName:<c>', 'save:<c>', 'emWavelen:<c>'.
    `button`: 'ok', 'trust', 'reuse', 'cancel' or 'eye:<c>' (the preview
    button of channel c; it must trigger the re-read restart).
    """
    button: str = 'ok'
    edits: tuple[tuple[str, object], ...] = ()


OK = DialogAnswer()


@dataclass(frozen=True)
class Module0Scenario:
    """One Module 0 conversion as the GUI would start it.

    `make_inputs(src, dst)` creates the raw files in `src` (and any
    pre-existing destination content in `dst`, both folders exist) and
    returns the raw file names selected in the setup dialog. `answers` maps
    the file name shown in the dialog title ('Metadata for <name>'; layout 2:
    the common basename) to one answer or a sequence used for its 1st, 2nd,
    ... dialog; dialogs without an entry use `default_answer` (None: the run
    fails). The flags are the setup-dialog options of
    `createDataStructWin.main`.
    """
    id: str
    description: str
    layout: int
    make_inputs: Callable[[str, str], list[str]]
    answers: Mapping[str, DialogAnswer | tuple[DialogAnswer, ...]] = \
        field(default_factory=dict)
    default_answer: DialogAnswer | None = OK
    lazy_load: bool = True
    move_raw: bool = False
    use_symlinks: bool = False
    overwrite: bool = False
    add_files: bool = False
    create_new: bool = False
    start_pos_n: int = 1
    other_files: str = 'leave'          # layout 2: 'leave' | 'move' | 'copy'

    def answer_for(self, name: str, occurrence: int) -> DialogAnswer | None:
        """Answer for the `occurrence`-th (0-based) dialog titled `name`."""
        entry = self.answers.get(name)
        if entry is not None:
            sequence = entry if isinstance(entry, tuple) else (entry,)
            if occurrence < len(sequence):
                return sequence[occurrence]
        return self.default_answer


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as file:
        file.write(text)


def _one_per_pos(filenames=('pos1.tif', 'pos2.tif', 'pos3.tif'),
                 overrides=None, **kwargs):
    def make_inputs(src: str, dst: str) -> list[str]:
        return synthetic.make_one_per_pos_set(
            src, filenames=filenames, overrides=overrides, **kwargs
        )
    return make_inputs


def _one_per_channel(positions=(1, 2), channels=('GFP', 'mCherry'),
                     other_files=(), existing_positions=()):
    def make_inputs(src: str, dst: str) -> list[str]:
        files = synthetic.make_one_per_channel_set(
            src, basename='ASY015_', positions=positions, channels=channels
        )
        for name in other_files:
            _write_text(os.path.join(src, name), f'other file {name}\n')
        _make_existing_positions(dst, existing_positions)
        return files
    return make_inputs


def _make_existing_positions(dst: str, numbers: Iterable[int]) -> None:
    """Valid (non-empty Images) Position folders, as a previous run left."""
    for n in numbers:
        _write_text(
            os.path.join(dst, f'Position_{n}', 'Images', 'previous_run.txt'),
            f'Position_{n}\n'
        )


def _single_ome(filename='exp.ome.tif', **kwargs):
    def make_inputs(src: str, dst: str) -> list[str]:
        synthetic.make_ome_tif(os.path.join(src, filename), **kwargs)
        return [filename]
    return make_inputs


_EXISTING_CSV = (
    'Description,values\n'
    'LensNA,1.4\nSizeT,3\nSizeZ,2\nTimeIncrement,1.0\nPhysicalSizeZ,1.0\n'
    'PhysicalSizeY,1.0\nPhysicalSizeX,1.0\nbasename,pos1_s1_\n'
    'channel_0_name,oldGFP\nchannel_1_name,mCherry\n'
    'channel_0_emWavelen,500.0\nchannel_1_emWavelen,610.0\n'
)


def _add_into_existing(src: str, dst: str) -> list[str]:
    files = synthetic.make_one_per_pos_set(src, filenames=('pos1.tif',))
    images = os.path.join(dst, 'Position_1', 'Images')
    _write_text(os.path.join(images, 'pos1_s1_metadata.csv'), _EXISTING_CSV)
    _write_text(os.path.join(images, 'pos1_s1_acdc_output.csv'),
                'frame_i,Cell_ID\n0,1\n')
    return files


def _overwrite_existing(src: str, dst: str) -> list[str]:
    files = synthetic.make_one_per_pos_set(src, filenames=('pos1.tif',))
    _write_text(os.path.join(dst, 'Position_1', 'Images', 'old_segm.txt'),
                'old segmentation\n')
    _write_text(os.path.join(dst, 'Position_1', 'keep.txt'), 'keep\n')
    _write_text(os.path.join(dst, 'Position_2', 'Images', 'other.txt'),
                'other position\n')
    return files


def _create_new_layout1(src: str, dst: str) -> list[str]:
    files = synthetic.make_one_per_pos_set(
        src, filenames=('pos1.tif', 'pos2.tif')
    )
    _make_existing_positions(dst, (1, 2))
    return files


def _truncated(src: str, dst: str) -> list[str]:
    synthetic.make_truncated_tif(os.path.join(src, 'pos1.tif'),
                                 keep_bytes=64)
    return ['pos1.tif']


_POS_1_3 = ('positions', ('Position_1', 'Position_3'))

MODULE0_SCENARIOS = (
    Module0Scenario(
        'S01', 'layout 1, 3 ImageJ files, OK in each dialog', 1,
        _one_per_pos(),
    ),
    Module0Scenario(
        'S02', 'layout 1, Trust on file 1 (file 2 SizeC=3, file 3 SizeT=5)',
        1, _one_per_pos(overrides={'pos2.tif': {'C': 3},
                                   'pos3.tif': {'T': 5}}),
        answers={'pos1.tif': DialogAnswer('trust'),
                 'pos2.tif': DialogAnswer('trust')},
    ),
    Module0Scenario(
        'S03', 'layout 1, "Use the above metadata" on file 1 with renamed '
        'channels', 1, _one_per_pos(),
        answers={'pos1.tif': DialogAnswer('reuse', (
            ('chName:0', 'GFP'), ('chName:1', 'mCherry')))},
    ),
    Module0Scenario(
        'S04', 'layout 1, save channel 0 only, time range (2, 3), h5', 1,
        _one_per_pos(filenames=('pos1.tif',)),
        answers={'pos1.tif': DialogAnswer('ok', (
            ('save:1', False), ('timeRange', (2, 3)), ('to_h5', True)))},
    ),
    Module0Scenario(
        'S05', 'layout 1, symbolic links', 1,
        _one_per_pos(filenames=('pos1.tif', 'pos2.tif')), use_symlinks=True,
    ),
    Module0Scenario(
        'S06', 'layout 0, .ome.tif with 3 scenes, positions 1 and 3', 0,
        _single_ome(series=3),
        answers={'exp.ome.tif': DialogAnswer('ok', (_POS_1_3,))},
    ),
    Module0Scenario(
        'S07', 'S06 + "Include image name" (SizeC=2: main reads the hidden '
        'dummy checkbox of channel 1)', 0, _single_ome(series=3),
        answers={'exp.ome.tif': DialogAnswer(
            'ok', (_POS_1_3, ('addImageName', True)))},
    ),
    Module0Scenario(
        'S07b', 'S07 with one channel (main reads the checkbox)', 0,
        _single_ome(series=3, C=1),
        answers={'exp.ome.tif': DialogAnswer(
            'ok', (_POS_1_3, ('addImageName', True)))},
    ),
    Module0Scenario(
        'S08', 'layout 2 + copy other files (ASY015_1_notes.txt, '
        'ASY015_10_x.txt), whole field of view in RAM', 2,
        _one_per_channel(other_files=('ASY015_1_notes.txt',
                                      'ASY015_10_x.txt', 'unrelated.txt')),
        other_files='copy', lazy_load=False,
    ),
    Module0Scenario(
        'S09', 'layout 2 with channel mNeon-Green', 2,
        _one_per_channel(channels=('GFP', 'mNeon-Green')),
    ),
    Module0Scenario(
        'S10', 'layout 1, add files into an existing Position (csv merge), '
        'channel 1 not saved', 1, _add_into_existing, add_files=True,
        answers={'pos1.tif': DialogAnswer('ok', (
            ('chName:0', 'GFP'), ('save:1', False)))},
    ),
    Module0Scenario(
        'S11', 'layout 1, overwrite (extra files inside and outside Images)',
        1, _overwrite_existing, overwrite=True,
    ),
    Module0Scenario(
        'S12a', 'layout 1, create new Positions after Position_2', 1,
        _create_new_layout1, create_new=True, start_pos_n=3,
    ),
    Module0Scenario(
        'S12b', 'layout 2, create new Positions after Position_2', 2,
        _one_per_channel(existing_positions=(1, 2)), create_new=True,
        start_pos_n=3,
    ),
    Module0Scenario(
        'S13', 'layout 1, move raw files + symbolic links', 1,
        _one_per_pos(filenames=('pos1.tif', 'pos2.tif')), move_raw=True,
        use_symlinks=True,
    ),
    Module0Scenario(
        'S14', 'layout 1, SizeT 3 -> 2 and SizeZ 2 -> 1, eye button '
        '(re-read), then OK', 1, _one_per_pos(filenames=('pos1.tif',)),
        answers={'pos1.tif': (
            DialogAnswer('eye:0', (('SizeT', 2), ('SizeZ', 1))), OK)},
    ),
    Module0Scenario(
        'S15', 'layout 1, Cancel in the first dialog', 1,
        _one_per_pos(filenames=('pos1.tif', 'pos2.tif')),
        answers={'pos1.tif': DialogAnswer('cancel')},
    ),
    Module0Scenario(
        'S16', 'layout 1, truncated (unreadable) TIFF', 1, _truncated,
    ),
    Module0Scenario(
        'S17', 'layout 1, .ome.tif with physical sizes in nm', 1,
        _one_per_pos(filenames=('pos1.ome.tif',), physical_size_unit='nm',
                     physical_sizes=(500.0, 100.0, 100.0)),
    ),
    Module0Scenario(
        'S18a', 'layout 1, T=10 file, SizeT edited 10 -> 5', 1,
        _one_per_pos(filenames=('pos1.tif',), T=10),
        answers={'pos1.tif': DialogAnswer('ok', (('SizeT', 5),))},
    ),
    Module0Scenario(
        'S18b', 'layout 1, T=5 file, SizeT edited 5 -> 1 (range widget '
        'hidden, stale (1, 5))', 1,
        _one_per_pos(filenames=('pos1.tif',), T=5),
        answers={'pos1.tif': DialogAnswer('ok', (('SizeT', 1),))},
    ),
)
MODULE0 = {scenario.id: scenario for scenario in MODULE0_SCENARIOS}
MODULE0_IDS = tuple(MODULE0)


# --------------------------------------------------------------- cascade
CASCADE_LAYOUTS = (1, 2)
CASCADE_FAKE_FILENAME = 'cascade.czi'
CASCADE_LAYOUT2_CHANNELS = ('GFP', 'mNeon-Green')
CASCADE_LAYOUT2_SIZE_S = 2
OME_CASCADE_VARIANTS = {
    'ome:default': {},
    'ome:scenes3': {'series': 3},
    'ome:no_image_name': {'image_names': None},
    'ome:no_physical_sizes': {'physical_sizes': None},
    'ome:no_time_increment': {'time_increment': None},
    'ome:no_channel_names': {'names': None},
    'ome:no_emission': {'em_wavelens': None},
    'ome:lens_na': {'lens_na': 0.75},
    'ome:unit_nm': {'physical_size_unit': 'nm'},
    'ome:time_ms': {'time_increment_unit': 'ms'},
    'ome:no_ome_xml': None,              # plain TIFF named *.ome.tif
}
"""`.ome.tif` cascade inputs: make_ome_tif(series=1, **kwargs), None = a
plain tifffile TIFF without OME-XML."""


def make_ome_cascade_file(folder: str, name: str) -> str:
    """Create the `.ome.tif` of OME_CASCADE_VARIANTS[name]; returns its
    file name."""
    filename = name.split(':', 1)[1] + '.ome.tif'
    path = os.path.join(folder, filename)
    kwargs = OME_CASCADE_VARIANTS[name]
    if kwargs is None:
        import tifffile
        tifffile.imwrite(path, synthetic.encoded_array('YX', Y=16, X=20))
    else:
        synthetic.make_ome_tif(path, **{'series': 1, **kwargs})
    return filename


def cascade_grid() -> dict[str, dict]:
    """The shared FakeOMEXML grid (fake_omexml.cascade_grid)."""
    return fake_omexml.cascade_grid()


# ------------------------------------------------------- restructure model
@dataclass(frozen=True)
class RestructureCase:
    """One call of create_datastruct_from_restruct_tiff_dialogue_options.

    `make_inputs(src, dst, work)` creates the input TIFFs in `src` (both src
    and dst exist) and any pre-existing destination content. The harness
    builds `selectedOptions` = source/destination folders, `dimensions`,
    overwrite/create-new False, then `options` (string values starting with
    '<WORK>', '<SRC>' or '<DST>' are resolved to paths; 'metadata_rows'
    becomes the dialog-style `metadata_df`), then removes `drop_keys`.
    """
    id: str
    description: str
    make_inputs: Callable[[str, str, str], object]
    dimensions: Mapping
    options: Mapping = field(default_factory=dict)
    drop_keys: tuple[str, ...] = ()


_NP = {'mode': 'not_present'}
_ALL_NOT_PRESENT = {'channels': _NP, 'timepoints': _NP, 'z_slices': _NP}


def _per_file_channels(*names: str) -> dict:
    return {'mode': 'per_file', 'names': [
        {'name': name, 'filename_identifier': name} for name in names
    ]}


def _loose(pattern, **kwargs):
    def make_inputs(src: str, dst: str, work: str) -> None:
        synthetic.make_loose_tiff_set(src, pattern=pattern, **kwargs)
    return make_inputs


def _with_existing(make_inputs, files: Mapping[str, str] = (),
                   empty_dirs: Sequence[str] = ()):
    """`make_inputs` plus pre-existing files/folders (relative to dst)."""
    def wrapped(src: str, dst: str, work: str) -> None:
        make_inputs(src, dst, work)
        for relpath, text in dict(files).items():
            _write_text(os.path.join(dst, *relpath.split('/')), text)
        for relpath in empty_dirs:
            os.makedirs(os.path.join(dst, *relpath.split('/')), exist_ok=True)
    return wrapped


def _sequence(*makers):
    def make_inputs(src: str, dst: str, work: str) -> None:
        for maker in makers:
            maker(src, dst, work)
    return make_inputs


def _raw_file(relpath: str, data: bytes):
    def make_inputs(src: str, dst: str, work: str) -> None:
        path = os.path.join(src, relpath)
        with open(path, 'wb') as file:
            file.write(data)
    return make_inputs


def _work_folder(name: str, files: Sequence[str] = ()):
    def make_inputs(src: str, dst: str, work: str) -> None:
        folder = os.path.join(work, name)
        os.makedirs(folder, exist_ok=True)
        for filename in files:
            _write_text(os.path.join(folder, filename), filename + '\n')
    return make_inputs


_IMG_1_2 = _loose('img{pos}.tif', positions=(1, 2))
_TIME_GAPS_SKIP = tuple(
    (pos, ch, t, None)
    for ch in ('GFP', 'mCh')
    for pos, frames in (('pos1', (6, 7)), ('pos2', (1, 3)))
    for t in frames
)

_LETTER_CASES = (
    RestructureCase(
        'RA', 'per-file channels + T marker "_", gaps and offsets, '
        'basename exp',
        _loose('{pos}_{ch}_{t}.tif', positions=('pos1', 'pos2'),
               channels=('GFP', 'mCh'), timepoints=(1, 3, 5, 6, 7),
               skip=_TIME_GAPS_SKIP),
        {'channels': _per_file_channels('GFP', 'mCh'),
         'timepoints': {'mode': 'per_file', 'filename_marker': '_'},
         'z_slices': _NP},
        {'basename': 'exp'},
    ),
    RestructureCase(
        'RB', '2D files, all not present, no basename', _IMG_1_2,
        _ALL_NOT_PRESENT,
    ),
    RestructureCase(
        'RC', 'one file (2, 3, Y, X): channels in file axis 0, T axis 1',
        _loose('well_A1.tif', file_axes='CTYX', C=2, T=3),
        {'channels': {'mode': 'in_file', 'axis': '0',
                      'names': [{'name': 'ph'}, {'name': 'fl'}]},
         'timepoints': {'mode': 'in_file', 'axis': '1'},
         'z_slices': _NP},
    ),
    RestructureCase(
        'RD', 'create new Positions into an empty destination', _IMG_1_2,
        _ALL_NOT_PRESENT, {'create_new_position_destination': True},
    ),
    RestructureCase(
        'RE', 'existing Position_1, no overwrite',
        _with_existing(_IMG_1_2, {'Position_1/Images/old.txt': 'old\n'}),
        _ALL_NOT_PRESENT,
    ),
    RestructureCase(
        'RF', 'overwrite: Images removed, Position_1/keep.txt kept',
        _with_existing(_IMG_1_2, {
            'Position_1/Images/old_segm.npz.txt': 'segm\n',
            'Position_1/keep.txt': 'keep\n',
            'Position_3/Images/stale.txt': 'stale\n',
        }),
        _ALL_NOT_PRESENT, {'overwrite_destination': True},
    ),
    RestructureCase(
        'RG', 'create new Positions after Position_1-2',
        _with_existing(_IMG_1_2, {
            'Position_1/Images/a.txt': 'a\n',
            'Position_2/Images/a.txt': 'a\n',
        }),
        _ALL_NOT_PRESENT, {'create_new_position_destination': True},
    ),
    RestructureCase(
        'RH', 'ambiguous T marker "_" in pos_01_GFP_1.tif',
        _loose('pos_01_GFP_{t}.tif', timepoints=(1,)),
        {'channels': {'mode': 'not_present', 'names': [{'name': 'GFP'}]},
         'timepoints': {'mode': 'per_file', 'filename_marker': '_'},
         'z_slices': _NP},
    ),
    RestructureCase(
        'RI', 'channel identifiers GFP and GFP2 clash in p1_GFP2.tif',
        _loose('p1_{ch}.tif', channels=('GFP', 'GFP2')),
        {'channels': _per_file_channels('GFP', 'GFP2'),
         'timepoints': _NP, 'z_slices': _NP},
    ),
    RestructureCase(
        'RJ', 'overwrite/create-new keys missing', _IMG_1_2,
        _ALL_NOT_PRESENT,
        drop_keys=('overwrite_destination',
                   'create_new_position_destination'),
    ),
    RestructureCase(
        'RK', 'metadata_df rows + basename "exp_"', _IMG_1_2,
        _ALL_NOT_PRESENT,
        {'basename': 'exp_', 'metadata_rows': (
            ('TimeIncrement', 60.0), ('PhysicalSizeZ', 1.0),
            ('PhysicalSizeY', 0.1), ('PhysicalSizeX', 0.1),
            ('channel_0_name', 'x'))},
    ),
    RestructureCase(
        'RL', 'only a .png in the source folder',
        _raw_file('a.png', b'not a tiff'), _ALL_NOT_PRESENT,
    ),
    RestructureCase(
        'RM', 'T in file, Y == 1 (squeeze)',
        _loose('line.tif', file_axes='TYX', T=3, Y=1, X=5),
        {'channels': _NP, 'timepoints': {'mode': 'in_file', 'axis': 0},
         'z_slices': _NP},
    ),
    RestructureCase(
        'RN', 'T, Z, C per file; position b lacks mCh (fails mid-run)',
        _loose('{pos}_t{t}_z{z}_{ch}.tif', positions=('a', 'b'),
               channels=('GFP', 'mCh'), timepoints=(0, 1), z_slices=(0, 1, 2),
               skip=tuple(('b', 'mCh', t, z) for t in (0, 1)
                          for z in (0, 1, 2))),
        {'channels': _per_file_channels('GFP', 'mCh'),
         'timepoints': {'mode': 'per_file', 'filename_marker': '_t'},
         'z_slices': {'mode': 'per_file', 'filename_marker': '_z'}},
    ),
    RestructureCase(
        'RO', "files=['img2.tif'] subset", _IMG_1_2, _ALL_NOT_PRESENT,
        {'files': ['img2.tif']},
    ),
)

_MODE_ABBR = {'not_present': 'np', 'per_file': 'pf', 'in_file': 'if'}
_AXIS_ORDERS = ('CTZ', 'TZC', 'ZCT')


def _combo_case(index: int, c_mode: str, t_mode: str, z_mode: str
                ) -> RestructureCase:
    """Two positions (p1, p2) for one combination of the three modes.

    per_file: channels GFP/mCh, timepoints 1 and 3 (gap), z-slices 0 and 1;
    in_file: C=2 (names ph/fl), T=3, Z=2, axis order rotating with `index`,
    axis given as str (dialog style) for even and int for odd indices;
    basename 'exp' for even indices.
    """
    modes = {'C': c_mode, 'T': t_mode, 'Z': z_mode}
    order = _AXIS_ORDERS[index % len(_AXIS_ORDERS)]
    file_axes = ''.join(a for a in order if modes[a] == 'in_file') + 'YX'

    def axis(letter: str):
        value = file_axes.index(letter)
        return str(value) if index % 2 == 0 else value

    pattern = '{pos}'
    pattern += '_t{t}' if t_mode == 'per_file' else ''
    pattern += '_z{z}' if z_mode == 'per_file' else ''
    pattern += '_{ch}' if c_mode == 'per_file' else ''
    make_inputs = _loose(
        pattern + '.tif', positions=('p1', 'p2'),
        channels=('GFP', 'mCh') if c_mode == 'per_file' else (None,),
        timepoints=(1, 3) if t_mode == 'per_file' else (None,),
        z_slices=(0, 1) if z_mode == 'per_file' else (None,),
        file_axes=file_axes,
        C=2 if c_mode == 'in_file' else 1,
        T=3 if t_mode == 'in_file' else 1,
        Z=2 if z_mode == 'in_file' else 1,
    )
    if c_mode == 'per_file':
        channels = _per_file_channels('GFP', 'mCh')
    elif c_mode == 'in_file':
        channels = {'mode': 'in_file', 'axis': axis('C'),
                    'names': [{'name': 'ph'}, {'name': 'fl'}]}
    else:
        channels = dict(_NP)
    dimensions = {'channels': channels}
    for key, letter, marker in (('timepoints', 'T', '_t'),
                                ('z_slices', 'Z', '_z')):
        mode = modes[letter]
        if mode == 'per_file':
            dimensions[key] = {'mode': mode, 'filename_marker': marker}
        elif mode == 'in_file':
            dimensions[key] = {'mode': mode, 'axis': axis(letter)}
        else:
            dimensions[key] = dict(_NP)
    case_id = 'combo_' + '-'.join(
        _MODE_ABBR[m] for m in (c_mode, t_mode, z_mode)
    )
    options = {'basename': 'exp'} if index % 2 == 0 else {}
    return RestructureCase(
        case_id, f'channels {c_mode}, timepoints {t_mode}, z-slices '
        f'{z_mode} (file axes {file_axes})', make_inputs, dimensions, options,
    )


_MODES = ('not_present', 'per_file', 'in_file')
_COMBO_CASES = tuple(
    _combo_case(i, c, t, z)
    for i, (c, t, z) in enumerate(
        (c, t, z) for c in _MODES for t in _MODES for z in _MODES
    )
)

_GFP_MCH_P1_P2 = _loose('{pos}_{ch}.tif', positions=('p1', 'p2'),
                        channels=('GFP', 'mCh'))

_EXTRA_CASES = (
    RestructureCase(
        'extra_yx_mismatch', 'p2_mCh has another Y than p2_GFP (fails at '
        'the second Position)',
        _sequence(
            _loose('{pos}_{ch}.tif', positions=('p1', 'p2'),
                   channels=('GFP', 'mCh'), skip=(('p2', 'mCh', None, None),)),
            _loose('p2_{ch}.tif', channels=('mCh',), Y=6),
        ),
        {'channels': _per_file_channels('GFP', 'mCh'),
         'timepoints': _NP, 'z_slices': _NP},
    ),
    RestructureCase(
        'extra_overlap', 'p2_GFP.tif and p2_GFP.tiff overlap (fails at the '
        'second Position)',
        _sequence(
            _loose('{pos}_GFP.tif', positions=('p1', 'p2')),
            _loose('p2_GFP.tiff'),
        ),
        {'channels': _per_file_channels('GFP'),
         'timepoints': _NP, 'z_slices': _NP},
    ),
    RestructureCase(
        'extra_existing_later_position', 'Position_2 exists, no overwrite '
        '(fails at the second Position)',
        _with_existing(_IMG_1_2, {'Position_2/Images/old.txt': 'old\n'}),
        _ALL_NOT_PRESENT,
    ),
    RestructureCase(
        'extra_create_new_collides_invalid_position', 'create new after '
        'Position_1 while Position_2 exists with an empty Images folder',
        _with_existing(_IMG_1_2, {'Position_1/Images/a.txt': 'a\n'},
                       empty_dirs=('Position_2/Images',)),
        _ALL_NOT_PRESENT, {'create_new_position_destination': True},
    ),
    RestructureCase(
        'extra_both_overwrite_and_create_new', 'overwrite and create new '
        'both set, Position_1 exists',
        _with_existing(_IMG_1_2, {'Position_1/Images/a.txt': 'a\n'}),
        _ALL_NOT_PRESENT,
        {'overwrite_destination': True,
         'create_new_position_destination': True},
    ),
    RestructureCase(
        'extra_segmentation_folder', 'a segmentation mask folder is selected '
        '(ignored)',
        _sequence(_IMG_1_2, _work_folder('segm', ('img1_segm.txt',))),
        _ALL_NOT_PRESENT, {'segmentation_mask_folder': '<WORK>/segm'},
    ),
    RestructureCase(
        'extra_source_missing', 'the source folder does not exist', _IMG_1_2,
        _ALL_NOT_PRESENT, {'source_folder': '<WORK>/missing'},
    ),
    RestructureCase(
        'extra_unknown_channel_mode', "channels mode 'separate' (typo) with "
        'two names', _IMG_1_2,
        {'channels': {'mode': 'separate',
                      'names': [{'name': 'A'}, {'name': 'B'}]},
         'timepoints': _NP, 'z_slices': _NP},
    ),
    RestructureCase(
        'extra_channel_name_with_space', "channel name 'my GFP'", _IMG_1_2,
        {'channels': {'mode': 'not_present', 'names': [{'name': 'my GFP'}]},
         'timepoints': _NP, 'z_slices': _NP},
    ),
    RestructureCase(
        'extra_duplicate_channel_names', "names 'GFP' and 'gfp'",
        _GFP_MCH_P1_P2,
        {'channels': {'mode': 'per_file', 'names': [
            {'name': 'GFP', 'filename_identifier': 'GFP'},
            {'name': 'gfp', 'filename_identifier': 'mCh'}]},
         'timepoints': _NP, 'z_slices': _NP},
    ),
    RestructureCase(
        'extra_missing_identifier', 'per-file channel without identifier',
        _GFP_MCH_P1_P2,
        {'channels': {'mode': 'per_file', 'names': [
            {'name': 'GFP', 'filename_identifier': 'GFP'}, {'name': 'mCh'}]},
         'timepoints': _NP, 'z_slices': _NP},
    ),
    RestructureCase(
        'extra_in_file_channel_count', 'in-file channels: 3 names, 2 '
        'channels in the file', _loose('a.tif', file_axes='CYX', C=2),
        {'channels': {'mode': 'in_file', 'axis': '0', 'names': [
            {'name': 'c0'}, {'name': 'c1'}, {'name': 'c2'}]},
         'timepoints': _NP, 'z_slices': _NP},
    ),
    RestructureCase(
        'extra_axis_out_of_range', 'T in file on a 2D TIFF', _IMG_1_2,
        {'channels': _NP, 'timepoints': {'mode': 'in_file', 'axis': '0'},
         'z_slices': _NP},
    ),
    RestructureCase(
        'extra_negative_axis', "T in file with axis '-3' on (T, Y, X)",
        _loose('a.tif', file_axes='TYX', T=3),
        {'channels': _NP, 'timepoints': {'mode': 'in_file', 'axis': '-3'},
         'z_slices': _NP},
    ),
    RestructureCase(
        'extra_x1_squeeze', 'T in file, X == 1 (squeeze)',
        _loose('col.tif', file_axes='TYX', T=3, Y=4, X=1),
        {'channels': _NP, 'timepoints': {'mode': 'in_file', 'axis': 0},
         'z_slices': _NP},
    ),
    RestructureCase(
        'extra_extensions', '.ome.tif, upper-case .TIFF, .tiff and an '
        'ignored .png',
        _sequence(_loose('a.ome.tif'), _loose('b.TIFF'), _loose('c.tiff'),
                  _raw_file('d.png', b'png')),
        _ALL_NOT_PRESENT,
    ),
    RestructureCase(
        'extra_z_in_file_t_per_file_metadata', 'Z in file, T per file '
        '(day marker), dialog metadata rows, basename with dash',
        _loose('{pos}_day{t}.tif', positions=('w1', 'w2'), timepoints=(2, 4),
               file_axes='ZYX', Z=3),
        {'channels': {'mode': 'not_present', 'names': [{'name': 'BF'}]},
         'timepoints': {'mode': 'per_file', 'filename_marker': '_day'},
         'z_slices': {'mode': 'in_file', 'axis': '0'}},
        {'basename': 'exp-1', 'metadata_rows': (
            ('TimeIncrement', 300.0), ('PhysicalSizeZ', 0.5),
            ('PhysicalSizeY', 0.2), ('PhysicalSizeX', 0.2),
            ('channel_0_name', 'BF'))},
    ),
)

RESTRUCTURE_CASES = _LETTER_CASES + _COMBO_CASES + _EXTRA_CASES
RESTRUCTURE = {case.id: case for case in RESTRUCTURE_CASES}
RESTRUCTURE_IDS = tuple(RESTRUCTURE)
