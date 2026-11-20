"""Expected behaviour changes of Re-structure compared with main.

Registry used by `test_golden_restructure.py`: one `ExpectedChange` per
change id (GV-R1 ... GV-R5) that the refactor makes visible in a golden
case. `assert_golden` requires every golden diff to be claimed by an entry
listed for that case and every listed entry to claim at least one diff.
The predicates check the new values themselves (GV-R2 against the golden
of the plain run RB, GV-R3 against the source shape), so a wrong output in
a listed case stays unclaimed (`test_restructure.py`, registry section).

Created empty on main (the goldens were captured there); the verbatim move
of the core keeps it empty, the fixes add one entry each. GV-R5 (after an
error the restructure worker thread quits) changes no output, so it has no
entry here: `test_restructure.py::test_worker_thread_quits_after_an_error`
tests it directly. GV-R6 (the dialog's recipe files go through the core,
as `acdc-data restructure --recipe`: "Load last parameters" loads
`_last_setup.json` (`restructure.last_setup_path`, the file of `--recipe
last`) where main loaded the first natsorted file whose name starts with
'_last_setup', e.g. '_last_setup - Copy.json', and without
`_last_setup.json` it says "No recipes available"; "Load recipe" lists
`restructure.recipe_filenames` and "Save recipe" names the file with
`restructure.RECIPE_PREFIX`, both as on main) changes no golden output
either:
`test_restructure_recipes.py::test_dialog_recipe_files_use_the_core`
tests it.
"""
from __future__ import annotations

from cellacdc.acdc_datastruct import errors

from . import manifest, scenarios
from .scenarios import ExpectedChange, assert_golden

__all__ = ['CHANGES', 'ExpectedChange', 'assert_golden']

PARITY_SIGNALS = [['initProgressBar', 0],
                  ['progress', 'Processing TIFF files...', 'INFO']]
"""What the core emits before the header scan (as on main)."""

PLAIN_RUN = scenarios.load_golden('restructure', 'RB')
"""Golden RB: the inputs of RD and RJ (img1.tif, img2.tif into an empty
destination) with overwrite and create new False. GV-R2 makes RD and RJ
give exactly this run (Position_1 and Position_2)."""

GV_R3_CASES = ('RM', 'extra_x1_squeeze')
"""One (T, Y, X) source file with Y or X of size 1, T in the file."""
GV_R3_FILE = 'dst/Position_1/Images/Channel_1.tif/'


def _same(old, new) -> bool:
    """Equal as the golden comparison sees it (`manifest.diff_manifests`)."""
    return not manifest.diff_manifests({'value': old}, {'value': new})


def _golden_value(golden, key: str):
    """The value of a diff key in a golden ('dst/<relpath>': destination
    entries are keyed by their relpath)."""
    if key.startswith('dst/'):
        return golden['dst'].get(key[len('dst/'):], manifest.ABSENT)
    return golden.get(key, manifest.ABSENT)


def _is_usage_error(name) -> bool:
    error = getattr(errors, name, None) if isinstance(name, str) else None
    return isinstance(error, type) and issubclass(error, errors.UsageError)


def _nothing_written(diff: manifest.Diff) -> bool:
    """A destination diff of a run that wrote nothing: the destination stays
    an empty folder, every other entry is absent."""
    if diff.key == 'dst/.':
        return diff.new == {'kind': 'empty_dir'}
    return diff.new is manifest.ABSENT


def _validated_before_writing(diff: manifest.Diff) -> bool:
    """GV-R1: the legacy error, as a UsageError subclass, raised before the
    first Position is touched (only the parity signals, no output)."""
    if diff.key == 'exception/0':
        return (diff.old in ('ValueError', 'FileExistsError')
                and _is_usage_error(diff.new))
    if diff.key == 'signals':
        return diff.new == PARITY_SIGNALS
    return _nothing_written(diff)


def _options_error(value) -> bool:
    return isinstance(value, list) and \
        value[0] == errors.RestructureOptionsError.__name__


def _options_normalized(diff: manifest.Diff) -> bool:
    """GV-R2: the crashes on missing keys (RJ) and on create new without
    Positions (RD) become exactly the plain run RB; invalid options become
    RestructureOptionsError before anything is emitted or written."""
    if diff.key == 'exception/0':
        return diff.old == 'ValueError' and \
            diff.new == errors.RestructureOptionsError.__name__
    if diff.key == 'exception':
        return (diff.new is None and diff.old[0] in ('IndexError', 'KeyError')
                or diff.old is None and _options_error(diff.new))
    plain_run = _same(diff.new, _golden_value(PLAIN_RUN, diff.key))
    if diff.key == 'result':
        return plain_run if diff.old is None else diff.new is None
    if diff.key == 'signals':
        return plain_run or diff.new == []
    return plain_run or _nothing_written(diff)


def _size1_axis_kept(case_id: str) -> dict:
    """{field: (old, new)}: the diffs GV-R3 makes to Channel_1.tif of a
    GV-R3 case. Main squeezed the (T, Y, X) source to 2D; now the output
    keeps the source shape, which tifffile writes with its default OME axes
    for 3D data, CYX (SizeC = T, as every (T, Y, X) output of the core)."""
    (source,) = scenarios.load_golden('restructure', case_id)['src'].values()
    size_t, size_y, size_x = source['shape']
    legacy = [size for size in source['shape'] if size != 1]
    expected = {
        'shape': (legacy, [size_t, size_y, size_x]),
        'axes': ('YX', 'CYX'),
        'ome/images/0/Channels': ([{}], [{}] * size_t),
        'ome/images/0/Pixels/SizeC': ('1', str(size_t)),
        'ome/images/0/Pixels/SizeY': (str(legacy[0]), str(size_y)),
        'ome/images/0/Pixels/SizeX': (str(legacy[1]), str(size_x)),
    }
    return {field: values for field, values in expected.items()
            if values[0] != values[1]}


GV_R3_DIFFS = [_size1_axis_kept(case_id) for case_id in GV_R3_CASES]


def _keeps_y_or_x(diff: manifest.Diff) -> bool:
    """GV-R3: the output keeps the source shape (T, 1, X) / (T, Y, 1); only
    the shape and the OME sizes that follow from it change."""
    field = diff.key[len(GV_R3_FILE):]
    return any(
        field in expected
        and _same(list(expected[field]), [diff.old, diff.new])
        for expected in GV_R3_DIFFS
    )


def _segmentation_warning_added(diff: manifest.Diff) -> bool:
    """GV-R4: one WARNING line, otherwise the same signals."""
    added = [s for s in diff.new if s not in diff.old]
    kept = [s for s in diff.new if s in diff.old]
    return (kept == diff.old and len(added) == 1
            and added[0][0] == 'progress' and added[0][2] == 'WARNING'
            and 'segmentation mask folder' in added[0][1])


CHANGES: list[ExpectedChange] = [
    ExpectedChange(
        gv='GV-R1',
        scenarios=(
            'RE', 'RH', 'RI', 'RL', 'RN', 'extra_yx_mismatch',
            'extra_overlap', 'extra_existing_later_position',
            'extra_create_new_collides_invalid_position',
            'extra_source_missing', 'extra_in_file_channel_count',
            'extra_axis_out_of_range',
        ),
        key_glob=('exception/0', 'signals', 'dst/*'),
        predicate=_validated_before_writing,
        description=(
            'Every validation error (channel missing for a later position, '
            'Y/X mismatch, overlap, out of bounds, existing Position folder, '
            'pixel type that OME-TIFF cannot store) is raised before any '
            'folder is created or Images deleted; same messages (new one for '
            'the pixel type), as UsageError subclasses '
            '(RestructureOptionsError, SourceError, ExistingPositionsError); '
            'no partial output.'
        ),
    ),
    ExpectedChange(
        gv='GV-R2',
        scenarios=(
            'RD', 'RJ', 'extra_both_overwrite_and_create_new',
            'extra_unknown_channel_mode', 'extra_channel_name_with_space',
            'extra_duplicate_channel_names', 'extra_missing_identifier',
        ),
        key_glob=('exception', 'exception/0', 'result', 'signals', 'dst/*'),
        predicate=_options_normalized,
        description=(
            'Core only (not reachable from the dialog): create new on a '
            'destination without valid Positions starts at Position_1 (was '
            'IndexError); missing overwrite/create-new keys default to '
            'False; both True is an error; invalid modes, names and '
            'basenames and missing identifiers/markers are rejected '
            '(RestructureOptionsError) before anything is written.'
        ),
    ),
    ExpectedChange(
        gv='GV-R3',
        scenarios=GV_R3_CASES,
        key_glob=GV_R3_FILE + '*',
        predicate=_keeps_y_or_x,
        description=(
            'Outputs whose Y or X has size 1 keep that axis (only singleton '
            'T/Z are squeezed).'
        ),
    ),
    ExpectedChange(
        gv='GV-R4',
        scenarios=('extra_segmentation_folder',),
        key_glob='signals',
        predicate=_segmentation_warning_added,
        description=(
            'A warning is logged when a segmentation mask folder is selected '
            '(it is ignored, as before).'
        ),
    ),
]
