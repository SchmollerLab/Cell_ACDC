"""Re-structure core `cellacdc.acdc_datastruct.restructure`: the move out
of `dataReStruct` (re-exports, Qt-free import), and the headless signal
adapters of `callbacks` driving the core (the golden cases themselves run
in `test_golden_restructure.py`).
"""
from __future__ import annotations

import json

import pytest

from cellacdc.acdc_datastruct import callbacks, restructure

from . import conditions, harness_restructure, manifest, scenarios

MOVED_NAMES = (
    'fail', 'mode_for', 'tiff_stem', 'natural_key', 'parse_index',
    'parse_channel', 'axis_index', 'get_layout', 'normalize_array',
    'create_datastruct_from_restruct_tiff_dialogue_options',
)
GUI_MODULES = ('cellacdc.apps', 'cellacdc.widgets', 'cellacdc.workers',
               'cellacdc._main', 'cellacdc.dataStruct',
               'cellacdc.dataReStruct')

run_core = restructure.create_datastruct_from_restruct_tiff_dialogue_options


def prepare_case(case_id: str, tmp_path) -> tuple[dict, dict]:
    """Inputs and `selectedOptions` of a golden case (as the harness builds
    them) -> (options, path placeholders)."""
    case = scenarios.RESTRUCTURE[case_id]
    work = tmp_path / 'work'
    src, dst = work / 'src', work / 'dst'
    src.mkdir(parents=True)
    dst.mkdir()
    placeholders = {'<SRC>': str(src), '<DST>': str(dst), '<WORK>': str(work)}
    case.make_inputs(str(src), str(dst), str(work))
    options = harness_restructure.build_options(
        case, str(src), str(dst), placeholders
    )
    return options, placeholders


# ------------------------------------------------------------------- move
@conditions.requires_qt
def test_datarestruct_reexports_the_moved_core():
    from cellacdc import dataReStruct

    for name in MOVED_NAMES:
        assert getattr(dataReStruct, name) is getattr(restructure, name), name


@conditions.requires_qt
def test_datarestruct_keeps_its_gui_names():
    from cellacdc import dataReStruct

    for name in ('run', '_log', 'saveTiff'):
        assert callable(getattr(dataReStruct, name)), name


def test_harness_uses_the_qt_free_core():
    assert harness_restructure.core_module_name() == \
        'cellacdc.acdc_datastruct.restructure'


_HEADLESS_RUN = '''
import json, os, sys
import numpy as np, tifffile
from cellacdc.acdc_datastruct import callbacks, restructure
src, dst = sys.argv[1], sys.argv[2]
for i in (1, 2):
    tifffile.imwrite(os.path.join(src, f'img{i}.tif'),
                     np.full((4, 5), i, dtype='uint16'))
lines = []
cb = callbacks.ProgressCallbacks(
    log=lambda text, level='INFO': lines.append([text, level]))
out = restructure.create_datastruct_from_restruct_tiff_dialogue_options(
    {'source_folder': src, 'destination_folder': dst,
     'overwrite_destination': False,
     'create_new_position_destination': False,
     'dimensions': {k: {'mode': 'not_present'}
                    for k in restructure.DIMENSION_KEYS}},
    signals=callbacks.SignalsAdapter(cb))
import cellacdc
blocked = _AcdcBlocker.blocked
loaded = sorted(m for m in sys.modules
                if m.split('.')[0] in blocked or m in GUI_MODULES)
print(json.dumps({'gui': cellacdc.GUI_INSTALLED, 'loaded': loaded,
                  'positions': sorted(out['position_folders']),
                  'log': lines}))
'''


@conditions.subprocess_heavy
def test_restructure_runs_without_qt(tmp_path):
    """Import and run the core in a child where every Qt binding is blocked:
    no GUI module is loaded."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    src.mkdir()
    code = (conditions.blocked_imports_prelude()
            + f'GUI_MODULES = {GUI_MODULES!r}\n' + _HEADLESS_RUN)
    proc = conditions.run_python(code, str(src), str(dst))
    assert proc.returncode == 0, proc.stderr[-3000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result == {
        'gui': False, 'loaded': [], 'positions': ['img1', 'img2'],
        'log': [['Processing TIFF files...', 'INFO'],
                ['Creating Position folder Position_1...', 'INFO'],
                ['Creating Position folder Position_2...', 'INFO']],
    }
    assert (dst / 'Position_2' / 'Images' / 'Channel_1.tif').is_file()


# ---------------------------------------------- callbacks <-> signals adapters
@pytest.mark.parametrize('case_id', ['RA', 'RK', 'combo_if-pf-if'])
def test_signals_adapter_drives_progress_callbacks(case_id, tmp_path):
    """`SignalsAdapter(ProgressCallbacks(...))` (what the command line
    passes) receives exactly the GUI's signal sequence, in order."""
    golden = scenarios.load_golden('restructure', case_id)
    options, placeholders = prepare_case(case_id, tmp_path)
    calls = []

    def log(text, level='INFO'):
        calls.append(['progress',
                      manifest.normalize_path_text(text, placeholders),
                      level])

    progress = callbacks.ProgressCallbacks(
        log=log,
        progress_init=lambda total: calls.append(['initProgressBar', total]),
        progress_step=lambda step: calls.append(['progressBar', step]),
    )
    run_core(options, signals=callbacks.SignalsAdapter(progress))
    assert calls == golden['signals']


def test_progress_callbacks_and_signals_adapter_defaults():
    progress = callbacks.ProgressCallbacks()
    progress.log('text', 'WARNING')
    progress.progress_init(3)
    progress.progress_step(1)
    assert progress.is_cancelled() is False

    calls = []
    adapter = callbacks.SignalsAdapter(callbacks.ProgressCallbacks(
        log=lambda text, level='INFO': calls.append((text, level)),
    ))
    adapter.progress.emit('default level')
    adapter.progress.emit('warn', 'WARNING')
    adapter.critical.emit((None, ValueError('x')))
    adapter.finished.emit({'position_folders': {}})
    adapter.initProgressBar.emit(2)
    adapter.progressBar.emit(1)
    assert calls == [('default level', 'INFO'), ('warn', 'WARNING')]
