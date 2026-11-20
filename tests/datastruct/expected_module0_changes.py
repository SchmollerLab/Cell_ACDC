"""Expected GUI-visible changes of Module 0 compared with main.

Registry used by `test_golden_module0.py`: the `ExpectedChange` entries of
every change id (GV-01 ... GV-20) that the refactor makes visible in a
golden scenario. `assert_golden` requires every golden diff to be claimed by
an entry listed for that scenario and every listed entry to claim at least
one diff, so this list is the reviewed list of GUI changes of the PR (one
PR row per change id; a change id can have several entries).

The claims are exact: the predicates check the new values, against main's
output in the golden or, for files main never wrote, against main's file of
the same kind holding the planes of the synthetic input. Every output file
that a change adds, removes or renames has an entry of its own
(`_file_changes`), so a missing, left over or different file fails the
scenario as an unused entry or an unclaimed diff (registry section of
`test_module0_adapter.py`).

Created empty on main (the goldens were captured there); entries are added
together with the change that causes them.

The other GUI-visible changes cannot show in a golden scenario and are
pinned by tests instead: GV-02 (BioIO TimeIncrement; test_bioio_reader,
test_bioio_raw_metadata), GV-04, GV-05, GV-05b, GV-05c, GV-15 (dialog
checks; test_module0_dialogs_qt), GV-05 for files that do not fit and GV-09
(test_convert_inprocess), GV-07 for later files with more frames
(test_module0_adapter), GV-12 (output identical; S08 runs with "Load entire
field of view in RAM"), GV-16 (test_planning, test_convert_inprocess),
GV-17, GV-18, GV-19 (test_module0_adapter, test_module0_worker_qt,
test_runner), GV-20 (test_module0_adapter), GV-21 (the rename buttons of
the setup dialog plan with `sources.plan_invalid_name_renames` and rename
with `writers.execute_source_mutations`, as `acdc-data convert
--rename-invalid`: when a new name is taken or two files would get the
same name, ignoring case, a "Files not renamed" box lists the collisions
in its details, nothing is renamed and the dialog is left as after "Let me
rename files myself"; main renamed file by file, so 'a b.tif' -> 'a_b.tif'
raised FileExistsError on Windows (after renaming the files listed before
it, if any) and replaced the raw file a_b.tif on Linux/macOS; without a
collision the new names, the order and the moment of the renames are
main's; an OS error while renaming reaches the error dialog as
`errors.MoveRawFilesError` naming both paths; test_module0_dialogs_qt,
test_sources). GV-16 also covers raw files whose names differ only in case
('Pos1.czi' and 'pos1.czi' in a source folder on a case-sensitive file
system): with "Move raw microscopy files" the conversion stops before
anything is moved or written and an `errors.PlanError` naming both files
reaches the error dialog (`createDataStructWin.workerCritical`), also when
the destination would keep both files, because `raw_microscopy_files` can
be on a file system that ignores case (network share, exFAT, macOS), where
main moved the second file over the first and converted both Positions
from it (test_planning, test_module0_adapter).
"""
from __future__ import annotations

import copy
import functools
import math
import re
from dataclasses import dataclass
from typing import Callable, Mapping

from . import manifest, scenarios, synthetic
from .scenarios import ExpectedChange, assert_golden

__all__ = ['CHANGES', 'ExpectedChange', 'assert_golden']

_WARNING_SEPARATOR = '=' * 51
_MAIN_INDEX_ERROR = ['IndexError', 'list index out of range']
"""Main's crash in S02 and S09 (type and first line)."""
_SAMPLE_LOG = ['progress', 'Reading sample image data...']


def _none_text_became_metadata(diff: manifest.Diff) -> bool:
    return diff.old is True and diff.new is False


def _nm_value_now_in_um(diff: manifest.Diff) -> bool:
    try:
        return math.isclose(float(diff.new), float(diff.old) / 1000)
    except (TypeError, ValueError):
        return False


def _is_metadata_read_log(signal: list) -> bool:
    """'Reading OME metadata...' or a line of a 'not found' warning."""
    if signal[0] != 'progress':
        return False
    text = signal[1]
    return (
        text in ('Reading OME metadata...', _WARNING_SEPARATOR)
        or text.startswith('<SRC>/')
        or (text.startswith('WARNING: ')
            and text.endswith(' not found in metadata.'))
    )


def _only_metadata_read_logs_added(diff: manifest.Diff) -> bool:
    if not isinstance(diff.old, list) or not isinstance(diff.new, list):
        return False

    def others(signals):
        return [s for s in signals if not _is_metadata_read_log(s)]

    return (
        len(diff.new) > len(diff.old)
        and others(diff.new) == others(diff.old)
    )


def _removed(diff: manifest.Diff) -> bool:
    return diff.old is not manifest.ABSENT and diff.new is manifest.ABSENT


def _halved(diff: manifest.Diff) -> bool:
    return diff.new * 2 == diff.old


# ------------------------------------------------------- exact file claims
@functools.lru_cache(maxsize=None)
def _golden_dst(scenario_id: str) -> dict:
    """Destination manifest that main wrote in a scenario."""
    return scenarios.load_golden('module0', scenario_id)['dst']


def _same(expected, actual) -> bool:
    """Equal as the golden comparison sees it (`manifest.diff_manifests`)."""
    return not manifest.diff_manifests({'entry': expected},
                                       {'entry': actual})


def _name_prefix(relpath: str) -> str:
    """'<stem>_s<NN>_' of an output file (its metadata.csv basename)."""
    return re.match(r'.*_s\d+_', relpath.rsplit('/', 1)[-1]).group(0)


def _set_pixels(entry: dict, s: int, c: int, frames: int | None) -> None:
    """Make the TZYX tif `entry` hold channel `c` of synthetic source `s`
    (`frames` frames, default: as many as before)."""
    size_t, size_z, size_y, size_x = entry['shape']
    size_t = frames or size_t
    array = synthetic.encoded_array(
        'TZYX', T=size_t, Z=size_z, Y=size_y, X=size_x, s=s, c0=c,
        dtype=entry['dtype']
    )
    entry.update(
        shape=list(array.shape), sha256=manifest.array_sha256(array),
        origins=[[s, t, c, 0] for t in range(size_t)],
    )
    entry['imagej'].update(frames=size_t, images=size_t*size_z)


@dataclass(frozen=True)
class _Like:
    """Expected manifest entry of an output file: main's entry `template`
    (a destination relpath in the golden of the same scenario) with

    pixels              (s, c): the planes of channel c of synthetic source
                        s (`synthetic.encoded_array`, TZYX);
    frames              their number of frames (default: the template's);
    rows                ((name, value), ...) replaced in metadata.csv;
    channels            ((name, emWavelen), ...) replacing the channel rows
                        of metadata.csv;
    xml_holds_metadata  a metadataXML text holds the reader metadata (GV-01:
                        main wrote 'None' for non-.ome.tif input).

    The basename row of metadata.csv always follows the new file name."""
    template: str
    pixels: tuple[int, int] | None = None
    frames: int | None = None
    rows: tuple[tuple[str, str], ...] = ()
    channels: tuple[tuple[str, str], ...] | None = None
    xml_holds_metadata: bool = False

    def entry(self, scenario_id: str, relpath: str) -> dict:
        """The expected entry of `relpath` in `scenario_id`."""
        entry = copy.deepcopy(_golden_dst(scenario_id)[self.template])
        if self.pixels is not None:
            _set_pixels(entry, *self.pixels, self.frames)
        if entry['kind'] == 'csv':
            entry['rows'] = self._csv_rows(entry['rows'], relpath)
        if self.xml_holds_metadata and entry['kind'] == 'metadataXML':
            entry['is_literal_None'] = False
        return entry

    def _csv_rows(self, rows: list, relpath: str) -> list:
        values = {**dict(self.rows), 'basename': _name_prefix(relpath)}
        rows = [[name, values.get(name, value)] for name, value in rows]
        if self.channels is not None:
            rows = [row for row in rows if not row[0].startswith('channel_')]
            rows += [[f'channel_{c}_name', name]
                     for c, (name, _) in enumerate(self.channels)]
            rows += [[f'channel_{c}_emWavelen', wavelen]
                     for c, (_, wavelen) in enumerate(self.channels)]
        return rows


def _file_changes(change: Mapping, scenario_id: str, *,
                  added: Mapping[str, _Like] | None = None,
                  removed: tuple[str, ...] = ()) -> list[ExpectedChange]:
    """Entries of `change` (gv, description) for the output files it
    adds (`added`: {relpath: expected entry}) or removes (`removed`) in one
    scenario, one entry per file."""
    def added_as(relpath: str, like: _Like):
        def predicate(diff: manifest.Diff) -> bool:
            return diff.old is manifest.ABSENT and _same(
                like.entry(scenario_id, relpath), diff.new
            )
        return predicate

    entries = [
        ExpectedChange(**change, scenarios=(scenario_id,),
                       key_glob=f'dst/{relpath}',
                       predicate=added_as(relpath, like))
        for relpath, like in (added or {}).items()
    ]
    entries += [
        ExpectedChange(**change, scenarios=(scenario_id,),
                       key_glob=f'dst/{relpath}', predicate=_removed)
        for relpath in removed
    ]
    return entries


def _renamed(renames: Mapping[str, str], **like) -> dict:
    """`_file_changes` arguments for files main wrote under other names
    ({old relpath: new relpath}): same content, basename row renamed."""
    return {
        'added': {new: _Like(old, **like) for old, new in renames.items()},
        'removed': tuple(renames),
    }


def _run_changes(change: Mapping, scenario_id: str,
                 checks: Mapping[str, Callable[[object, object], bool]]
                 ) -> ExpectedChange:
    """Entry of `change` for the record keys `checks` ({key: check(old,
    new)}) of one scenario."""
    def predicate(diff: manifest.Diff) -> bool:
        return bool(checks[diff.key](diff.old, diff.new))

    return ExpectedChange(**change, scenarios=(scenario_id,),
                          key_glob=tuple(checks), predicate=predicate)


def _main_crash_fixed(old, new) -> bool:
    """`exceptions`: main's IndexError is gone."""
    return old == [_MAIN_INDEX_ERROR] and new == []


def _now_ends(old, new) -> bool:
    """`task_ended`: "Conversion task ended" (main showed nothing)."""
    return old == [] and new == ['ended']


def _saving_log(position: int, total: int) -> list:
    return ['progress', f'Position {position}/{total}: saving data to '
                        f'<DST>/Position_{position}/Images...']


# ------------------------------------------------- scenario expectations
# S02: Trust on file 1 (2 channels); file 2 has 3 channels (main crashed
# with IndexError writing it), file 3 has 2 channels and 5 frames.
_S02_P1 = 'Position_1/Images/pos1_s1_'
_S02_FILES = {
    **{
        f'Position_2/Images/pos2_s2_Channel_0_{c}.tif':
            _Like(_S02_P1 + 'Channel_0_0.tif', pixels=(1, c))
        for c in range(3)
    },
    'Position_2/Images/pos2_s2_metadata.csv': _Like(
        _S02_P1 + 'metadata.csv',
        channels=tuple((f'Channel_0_{c}', '0.0') for c in range(3))
    ),
    **{
        f'Position_3/Images/pos3_s3_Channel_0_{c}.tif':
            _Like(_S02_P1 + 'Channel_0_0.tif', pixels=(2, c), frames=5)
        for c in range(2)
    },
    'Position_3/Images/pos3_s3_metadata.csv': _Like(
        _S02_P1 + 'metadata.csv', rows=(('SizeT', '5'),)
    ),
    'Position_3/Images/pos3_s3_metadataXML.txt': _Like(
        _S02_P1 + 'metadataXML.txt', xml_holds_metadata=True
    ),
}
_S02_ASKED_AGAIN = [
    {'accepted': True, 'button': 'trust', 'title': 'Metadata for pos2.tif'},
    {'accepted': True, 'button': 'ok', 'title': 'Metadata for pos3.tif'},
]
_CONFIRM_AS_FILE_1 = (
    'LensNA', 'SizeZ', 'SizeS', 'TimeIncrement', 'TimeIncrementUnit',
    'PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeZ', 'PhysicalSizeUnit',
    'ImageName',
)


def _s02_confirm(old, new) -> bool:
    """The dialogs of files 2 and 3 show that file's sizes."""
    return new[:1] == old and [
        (args['filename'], args['SizeC'], args['SizeT']) for args in new[1:]
    ] == [('pos2.tif', 3, 3), ('pos3.tif', 2, 5)] and all(
        args[key] == old[0][key]
        for args in new[1:] for key in _CONFIRM_AS_FILE_1
    )


def _s02_log(old, new) -> bool:
    """Main's log up to its crash in file 2, then: the dialog of file 2
    before its saving line, file 3 logged as file 2 was, 'finished'."""
    end_1, end_2 = old.index(_saving_log(1, 3)), old.index(_saving_log(2, 3))
    file_3 = [[kind, text.replace('pos2.tif', 'pos3.tif')]
              for kind, text in old[end_1 + 1:end_2]]
    return old[end_2 + 1:] == [['critical', *_MAIN_INDEX_ERROR]] and new == (
        old[:end_2] + [_SAMPLE_LOG, ['confirmMetadata', 1], old[end_2]]
        + file_3 + [_SAMPLE_LOG, ['confirmMetadata', 2], _saving_log(3, 3),
                    ['finished']]
    )


_S02_RUN = {
    'confirm': _s02_confirm,
    'dialogs': lambda old, new: new == old + _S02_ASKED_AGAIN,
    'exceptions': _main_crash_fixed,
    'pbar/step_sum': lambda old, new: new == 2 + 3 + 2,
    'signals': _s02_log,
    'task_ended': _now_ends,
}

# S09: main wrote Position_1's GFP.tif, metadata.csv and metadataXML.txt,
# then looked up the sanitized channel 'mNeon_Green' (IndexError).
_S09_P1 = 'Position_1/Images/ASY015__s1_'
_S09_FILES = {
    _S09_P1 + 'mNeon_Green.tif': _Like(_S09_P1 + 'GFP.tif', pixels=(1, 1)),
    'Position_2/Images/ASY015__s2_GFP.tif':
        _Like(_S09_P1 + 'GFP.tif', pixels=(2, 0)),
    'Position_2/Images/ASY015__s2_mNeon_Green.tif':
        _Like(_S09_P1 + 'GFP.tif', pixels=(2, 1)),
    'Position_2/Images/ASY015__s2_metadata.csv':
        _Like(_S09_P1 + 'metadata.csv'),
    'Position_2/Images/ASY015__s2_metadataXML.txt':
        _Like(_S09_P1 + 'metadataXML.txt', xml_holds_metadata=True),
}
_S09_RUN = {
    'exceptions': _main_crash_fixed,
    'pbar/step_sum': lambda old, new: new == 2 * 2,
    'signals': lambda old, new: (
        old[-1] == ['critical', *_MAIN_INDEX_ERROR]
        and new == old[:-1] + [_saving_log(2, 2), ['finished']]
    ),
    'task_ended': _now_ends,
}

# S12b: main wrote input positions 1 and 2 into the existing Position_1/2.
_ONE_PER_CHANNEL_FILES = (
    'GFP.tif', 'mCherry.tif', 'metadata.csv', 'metadataXML.txt'
)
_S12B_RENAMES = {
    f'Position_{p}/Images/ASY015__s{p}_{name}':
        f'Position_{p + 2}/Images/ASY015__s{p + 2}_{name}'
    for p in (1, 2) for name in _ONE_PER_CHANNEL_FILES
}
_POSITION_NUMBER = re.compile(r'(Position[ _])(\d+)')


def _two_positions_later(diff: manifest.Diff) -> bool:
    """'Position 1/2: saving data to <DST>/Position_1/Images...' -> 'Position
    3/2: ... Position_3 ...' (main logs layout 1 the same way, S12a)."""
    return isinstance(diff.old, str) and diff.new == _POSITION_NUMBER.sub(
        lambda match: f'{match[1]}{int(match[2]) + 2}', diff.old
    )


# S07 (2 channels) / S07b (1 channel): Positions 1 and 3 of 'exp.ome.tif'.
_S07_RENAMES = {
    f'Position_{k}/Images/exp_ome_s{k}_{name}':
        f'Position_{k}/Images/exp_ome_Scene__{k}_s{k}_{name}'
    for k in (1, 3) for name in _ONE_PER_CHANNEL_FILES
}
_S07B_RENAMES = {
    f'Position_{k}/Images/exp_ome{sep}s{k}_{name}':
        f'Position_{k}/Images/exp_ome_Scene__{k}_s{k}_{name}'
    for k in (1, 3) for sep, name in (
        ('_', 'GFP.tif'), ('__', 'metadata.csv'), ('__', 'metadataXML.txt')
    )
}

# --------------------------------------------------------------- changes
_GV_06 = {
    'gv': 'GV-06',
    'description': (
        '"Trust metadata reader for all next positions": TimeIncrement '
        'and pixel sizes of later files are converted to s/um; a later '
        'file with a different channel count, or an explicit time range '
        'that does not fit, shows the metadata dialog for that file; a '
        'full time range (SizeT left at the file\'s value) means all '
        'frames of each file.'
    ),
}
_GV_08 = {
    'gv': 'GV-08',
    'description': (
        'One file per channel: files found by an exact (position, raw '
        'token) map, so names changed by sanitizing or renamed in the '
        'dialog work, GFP no longer picks GFP2/GFP-2, zero-padded '
        'numbers work, .ome is no longer part of channel names; channel '
        'order (dialog rows, channel_<c> rows) is natural-sorted instead '
        'of random per session; the "Non-compatible pattern" message '
        'lists the file names that do not match in its details.'
    ),
}
_GV_10 = {
    'gv': 'GV-10',
    'description': (
        'One file per channel + "Create new Position folders": new '
        'folders start after the last existing one (previously wrote '
        'into the existing Position_<n>). No-op re.sub on the path '
        'removed.'
    ),
}
_GV_11 = {
    'gv': 'GV-11',
    'description': (
        'One file per channel, move/copy "other files": found in the '
        'original source folder even after raw files were moved; '
        'matched by exact position number (_1_ no longer matches _10_); '
        'the name must start with <basename><N>_ (main matched the '
        'substring anywhere); errors logged as text.'
    ),
}
_GV_14 = {
    'gv': 'GV-14',
    'description': (
        '"Include image name": metadata and image files share '
        '<stem>_<ImageName>_s<NN>_ with the sanitized per-series name '
        '(previously stem__sNN_metadata.csv vs stem_sNN_GFP.tif), and '
        'the checkbox is honoured for files with more than one channel '
        '(previously ignored).'
    ),
}

CHANGES: list[ExpectedChange] = [
    # Comes with the reader fix (`Metadata.init_from_file` returns self), so
    # the legacy worker already shows it. Scenarios = every scenario that
    # rewrites a Position main wrote from a non-'.ome.tif' file (S12b: the
    # Positions are new, see GV-10).
    ExpectedChange(
        gv='GV-01',
        scenarios=(
            'S01', 'S02', 'S03', 'S04', 'S05', 'S08', 'S09', 'S10', 'S11',
            'S12a', 'S13', 'S14', 'S18a', 'S18b',
        ),
        key_glob='dst/*_metadataXML.txt/is_literal_None',
        predicate=_none_text_became_metadata,
        description=(
            '*_metadataXML.txt for non-.ome.tif input holds the reader '
            'metadata (str(bioimage.metadata)) instead of the text None.'
        ),
    ),
    # Metadata dialog (`convertUnits`); S17 shows nm pixel sizes.
    ExpectedChange(
        gv='GV-03',
        scenarios=('S17',),
        key_glob='dst/*_metadata.csv/rows/*/1',
        predicate=_nm_value_now_in_um,
        description=(
            'Pixel sizes shown/entered in nm, mm or cm are converted to um '
            'before metadata.csv and TIFF spacing (previously written '
            'unconverted).'
        ),
    ),
    _run_changes(_GV_06, 'S02', _S02_RUN),
    *_file_changes(_GV_06, 'S02', added=_S02_FILES),
    ExpectedChange(
        gv='GV-07',
        scenarios=('S03',),
        key_glob='signals',
        predicate=_only_metadata_read_logs_added,
        description=(
            '"Use the above metadata for all the next positions": each '
            'position\'s metadataXML holds its own file\'s metadata; a file '
            'with a different channel count or a non-fitting time range '
            'shows the dialog; a full time range (SizeT left at file 1\'s '
            'value) means all frames of each later file (main wrote file '
            '1\'s frame range, also for a later file with more frames). The '
            'log now shows "Reading OME metadata..." and the metadata '
            'warnings for every file (main skipped readMetadata for files '
            '2..N).'
        ),
    ),
    _run_changes(_GV_08, 'S09', _S09_RUN),
    *_file_changes(_GV_08, 'S09', added=_S09_FILES),
    ExpectedChange(**_GV_10, scenarios=('S12b',), key_glob='signals/*/1',
                   predicate=_two_positions_later),
    *_file_changes(_GV_10, 'S12b',
                   **_renamed(_S12B_RENAMES, xml_holds_metadata=True)),
    # S08: ASY015_10_x.txt was copied into Position_1 (substring '_1').
    *_file_changes(_GV_11, 'S08',
                   removed=('Position_1/Images/ASY015__s1_x.txt',)),
    ExpectedChange(
        gv='GV-13',
        scenarios=('S08', 'S12b'),
        key_glob='pbar/step_sum',
        predicate=_halved,
        description=(
            'One file per channel: the progress bar advances by SizeC once '
            'per position (like the other layouts) and reaches exactly 100% '
            '(main counted every channel twice).'
        ),
    ),
    # S07 (2 channels): main ignored the checkbox (overwritten by the hidden
    # dummy checkbox of the second channel row). S07b (1 channel): main
    # wrote `exp_ome__s1_metadata.csv` next to `exp_ome_s1_GFP.tif`.
    *_file_changes(_GV_14, 'S07', **_renamed(_S07_RENAMES)),
    *_file_changes(_GV_14, 'S07b', **_renamed(_S07B_RENAMES)),
]
