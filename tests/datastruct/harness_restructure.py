"""Re-structure characterization harness.

`run_restructure_case(case, workdir)` builds the inputs of a
`scenarios.RestructureCase` under `workdir` (`src/`, `dst/`), calls
`create_datastruct_from_restruct_tiff_dialogue_options(selectedOptions,
signals=...)` with a recording duck-typed `signals` object (the contract of
`workers.SimpleWorker.signals`) and returns a JSON-serializable record:

    dst, src     manifest.build_manifest of both folders after the call
    signals      every emit, in order: [name, *args] (paths normalized)
    result       the returned dict with normalized paths (None on error)
    exception    [type name, normalized message] (None on success)

Paths are normalized to '<SRC>', '<DST>' and '<WORK>'. The function is taken
from the Qt-free `cellacdc.acdc_datastruct.restructure` when it exists, else
from `cellacdc.dataReStruct` (main; needs Qt).
"""
from __future__ import annotations

import copy
import importlib
import importlib.util
import os

from . import conditions, manifest, scenarios

CORE_MODULES = (
    'cellacdc.acdc_datastruct.restructure', 'cellacdc.dataReStruct'
)
FUNCTION_NAME = 'create_datastruct_from_restruct_tiff_dialogue_options'
SIGNAL_NAMES = ('initProgressBar', 'progress', 'progressBar', 'critical',
                'finished')


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ImportError:
        return False


def core_module_name() -> str | None:
    """Module that will provide the restructure function, or None when
    neither the Qt-free core nor (without Qt) `dataReStruct` can load."""
    if _module_exists(CORE_MODULES[0]):
        return CORE_MODULES[0]
    if conditions.qt_available() and _module_exists(CORE_MODULES[1]):
        return CORE_MODULES[1]
    return None


def restructure_function():
    name = core_module_name()
    if name is None:
        raise RuntimeError(
            'No restructure core: cellacdc.acdc_datastruct.restructure is '
            'missing and cellacdc.dataReStruct needs Qt'
        )
    return getattr(importlib.import_module(name), FUNCTION_NAME)


class _Emitter:
    def __init__(self, name: str, events: list, placeholders: dict) -> None:
        self._name = name
        self._events = events
        self._placeholders = placeholders

    def emit(self, *args) -> None:
        self._events.append([self._name] + [
            manifest.normalize_path_text(arg, self._placeholders)
            if isinstance(arg, str) else manifest._jsonable(arg)
            for arg in args
        ])


class RecordingSignals:
    """Duck-typed `signals`: `<name>.emit(*args)` appends [name, *args] to
    `events` for every name in SIGNAL_NAMES."""

    def __init__(self, placeholders: dict) -> None:
        self.events: list[list] = []
        for name in SIGNAL_NAMES:
            setattr(self, name, _Emitter(name, self.events, placeholders))


def _resolve(value, placeholders: dict):
    if isinstance(value, str):
        for token, path in placeholders.items():
            if value == token or value.startswith(token + '/'):
                rest = value[len(token):].lstrip('/')
                return os.path.join(path, *rest.split('/')) if rest else path
    return value


def _metadata_df(rows):
    """Dialog-style metadata DataFrame (apps.py metadataDataFrame layout)."""
    import pandas as pd

    return pd.DataFrame({
        'Description': [row[0] for row in rows],
        'values': [row[1] for row in rows],
    }).set_index('Description')


def build_options(case: scenarios.RestructureCase, src: str, dst: str,
                  placeholders: dict) -> dict:
    """`selectedOptions` of `case` (see scenarios.RestructureCase)."""
    options = {
        'source_folder': src,
        'destination_folder': dst,
        'dimensions': copy.deepcopy(dict(case.dimensions)),
        'overwrite_destination': False,
        'create_new_position_destination': False,
    }
    for key, value in case.options.items():
        if key == 'metadata_rows':
            options['metadata_df'] = _metadata_df(value)
        else:
            options[key] = _resolve(copy.deepcopy(value), placeholders)
    for key in case.drop_keys:
        options.pop(key, None)
    return options


def _normalize_result(result, placeholders: dict):
    return manifest._normalize_nested(manifest._jsonable(result), placeholders)


def run_restructure_case(case: scenarios.RestructureCase | str,
                         workdir: str) -> dict:
    """Run one case in `workdir` (must be empty or missing) and return its
    record (module docstring)."""
    if isinstance(case, str):
        case = scenarios.RESTRUCTURE[case]
    function = restructure_function()
    work = os.path.abspath(workdir)
    src = os.path.join(work, 'src')
    dst = os.path.join(work, 'dst')
    os.makedirs(src)
    os.makedirs(dst)
    placeholders = {'<SRC>': src, '<DST>': dst, '<WORK>': work}
    case.make_inputs(src, dst, work)
    options = build_options(case, src, dst, placeholders)
    signals = RecordingSignals(placeholders)
    result = exception = None
    try:
        result = function(options, signals=signals)
    except Exception as err:
        exception = [type(err).__name__,
                     manifest.normalize_path_text(str(err), placeholders)]
    return {
        'case': case.id,
        manifest.INFO_KEY: {'description': case.description},
        'dst': manifest.build_manifest(dst, placeholders=placeholders,
                                       include_env=False),
        'src': manifest.build_manifest(src, placeholders=placeholders,
                                       include_env=False),
        'signals': signals.events,
        'result': None if result is None
        else _normalize_result(result, placeholders),
        'exception': exception,
    }
