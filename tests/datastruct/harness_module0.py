"""Module 0 characterization harness.

`run_module0_scenario(scenario, workdir, *, backend=None)` runs one
`scenarios.Module0Scenario` synchronously in this process: it builds the
inputs under `workdir` (`src/`, `dst/`), constructs the real
`dataStruct.bioFormatsWorker` as `createDataStructWin.main` does (layout 2:
after the real `createDataStructWin.attemptSeparateMultiChannel` on a stub),
connects every signal to recorders (direct calls, same thread), answers each
`confirmMetadata` with the real `apps.QDialogMetadataXML` (built with the
kwargs of `askConfirmMetadata`, shown offscreen, scripted widget edits, then
the scripted button) and calls `worker.run()`.

`backend` is passed to the worker only when its constructor accepts it
(`worker_accepts_backend()`; not on main, where the goldens are captured);
`None` keeps the worker's default (real child processes).

The returned record (JSON-serializable, paths normalized to '<SRC>'/'<DST>'):

    dst, src        manifest.build_manifest of both folders after the run
    signals         [name, ...] per emit of progress (text), confirmMetadata
                    (index into `confirm`), criticalError (action, file name,
                    traceback non-empty), critical (type, first line),
                    sigMoveRawMicrFilesFailed (details non-empty), finished
    confirm         the 17 confirmMetadata args by name (sample arrays as
                    shape/dtype/sha256)
    dialogs         per metadata dialog: title, button, accepted
    message_boxes   widgets.myMessageBox calls (method, title, plain text)
    task_ended      per `finished`: what createDataStructWin.taskEnded shows
                    ('ended', 'cancelled' or 'nothing')
    pbar            initPbar values and the progressPbar sum
    exceptions      [type, first line] of every `critical` error

`capture_cascade(workdir)` records the legacy metadata cascade
(`bioFormatsWorker.readMetadata`) on every FakeOMEXML grid entry and on real
`.ome.tif` variants (golden/cascade.json).
"""
from __future__ import annotations

import contextlib
import html
import inspect
import logging
import os
import re
import traceback
import types

from natsort import natsorted

from . import conditions, fake_omexml, manifest, scenarios

CONFIRM_ARG_NAMES = (
    'filename', 'LensNA', 'SizeT', 'SizeZ', 'SizeC', 'SizeS',
    'TimeIncrement', 'TimeIncrementUnit', 'PhysicalSizeX', 'PhysicalSizeY',
    'PhysicalSizeZ', 'PhysicalSizeUnit', 'chNames', 'emWavelens',
    'ImageName', 'rawFilePath', 'sampleImgData',
)
DIALOG_VALUE_NAMES = (
    'to_h5', 'LensNA', 'SizeT', 'SizeZ', 'SizeC', 'SizeS', 'TimeIncrement',
    'PhysicalSizeX', 'PhysicalSizeY', 'PhysicalSizeZ', 'selectedPos',
    'chNames', 'timeRangeToSave', 'saveChannels', 'emWavelens',
    'addImageName', 'trust', 'overWrite', 'cancel',
    'requestedReadingSampleImageDataAgain',
)
LOGGER_NAME = 'cellacdc-golden-harness'


class HarnessError(RuntimeError):
    """The scenario script cannot be applied (not a behaviour under test)."""


class FakeMutex:
    """QMutex stand-in for the synchronous harness."""

    def lock(self) -> None:
        pass

    def unlock(self) -> None:
        pass

    def tryLock(self, *args) -> bool:
        return True


class FakeWaitCondition:
    """QWaitCondition stand-in: the dialog was answered synchronously inside
    `confirmMetadata.emit`, so `wait` returns at once."""

    def wait(self, *args) -> bool:
        return True

    def wakeAll(self) -> None:
        pass

    def wakeOne(self) -> None:
        pass


def assert_imports_this_checkout() -> None:
    """Fail unless `cellacdc` is imported from the checkout that contains
    these tests (e.g. a capture from a worktree of main must not import
    `cellacdc` from the checkout that an editable install points to)."""
    import cellacdc

    path = os.path.normcase(os.path.realpath(cellacdc.__file__))
    root = os.path.normcase(os.path.realpath(conditions.REPO_ROOT))
    if not path.startswith(root + os.sep):
        raise HarnessError(
            f'cellacdc is imported from {cellacdc.__file__}, not from '
            f'{conditions.REPO_ROOT}: run with cwd and PYTHONPATH = the '
            'checkout that holds tests/datastruct'
        )


@contextlib.contextmanager
def children_import_this_checkout():
    """Put the repo root first on PYTHONPATH while the run spawns children
    (they inherit os.environ), restoring the previous value afterwards."""
    key = 'PYTHONPATH'
    old = os.environ.get(key)
    root = os.path.normcase(os.path.abspath(conditions.REPO_ROOT))
    rest = [p for p in (old or '').split(os.pathsep)
            if p and os.path.normcase(os.path.abspath(p)) != root]
    os.environ[key] = os.pathsep.join([conditions.REPO_ROOT] + rest)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


_QAPP = []


def _keep_qapp():
    """The offscreen QApplication, referenced for the whole session (a
    collected QApplication makes the next QWidget abort the process)."""
    if not _QAPP:
        _QAPP.append(conditions.qapp())
    return _QAPP[0]


def worker_accepts_backend() -> bool:
    """True when `bioFormatsWorker.__init__` has a `backend` parameter."""
    from cellacdc import dataStruct

    parameters = inspect.signature(
        dataStruct.bioFormatsWorker.__init__
    ).parameters
    return 'backend' in parameters


def plain_text(text) -> str:
    """HTML of a message box as plain text (line/paragraph tags -> space,
    other tags removed, entities decoded, spaces collapsed)."""
    text = re.sub(r'<(br|/?p|/?div|/?li)\b[^>]*>', ' ', str(text),
                  flags=re.IGNORECASE)
    text = html.unescape(re.sub(r'<[^>]+>', '', text))
    return ' '.join(text.split())


class _MessageBoxRecorder:
    """Non-blocking stand-in for `widgets.myMessageBox`: every method call
    (critical, warning, information, question, ...) is recorded."""

    def __init__(self, log: list, placeholders: dict) -> None:
        self._log = log
        self._placeholders = placeholders
        self.cancel = False
        self.clickedButton = None

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError(name)

        def record(*args, **kwargs):
            title = args[1] if len(args) > 1 else kwargs.get('title', '')
            text = args[2] if len(args) > 2 else kwargs.get('message', '')
            self._log.append({
                'method': name,
                'title': str(title),
                'text': manifest.normalize_path_text(
                    plain_text(text), self._placeholders
                ),
            })
        return record


@contextlib.contextmanager
def recorded_message_boxes(log: list, placeholders: dict):
    """Replace `widgets.myMessageBox` by a recorder during the block."""
    from cellacdc import widgets

    original = widgets.myMessageBox
    widgets.myMessageBox = lambda *a, **k: _MessageBoxRecorder(
        log, placeholders
    )
    try:
        yield
    finally:
        widgets.myMessageBox = original


def _array_summary(array) -> dict:
    return {
        'shape': list(array.shape),
        'dtype': str(array.dtype),
        'sha256': manifest.array_sha256(array),
    }


def sample_summary(sample):
    """Sample image data dict (6 dimension-order keys -> list of channel
    arrays) as shapes/dtypes/sha256; one list when all keys are equal."""
    if sample is None:
        return None
    if not isinstance(sample, dict):
        return {'unexpected_type': type(sample).__name__}
    keys = sorted(sample)
    per_key = {
        key: [_array_summary(array) for array in sample[key]] for key in keys
    }
    first = per_key[keys[0]] if keys else []
    if all(value == first for value in per_key.values()):
        return {'keys': keys, 'channels': first}
    return {'keys': keys, 'per_key': per_key}


def _first_line(text: str) -> str:
    lines = str(text).strip().splitlines()
    return lines[0] if lines else ''


_TRACEBACK_FILE = re.compile(r'File "[^"\n]*[\\/]([^"\\/\n]+)"')


def strip_traceback_paths(text: str) -> str:
    """`File "<path>"` lines of a traceback reduced to the file name, so a
    golden never holds a path of the machine that captured it."""
    return _TRACEBACK_FILE.sub(r'File "\1"', text)


# ------------------------------------------------------------ dialog script
def _set_combo(combo, text: str) -> None:
    index = combo.findText(text)
    if index < 0:
        raise HarnessError(f'{text!r} is not an item of the combo box')
    combo.setCurrentIndex(index)


def _select_positions(dlg, labels) -> None:
    selector = getattr(dlg, 'posSelector', None)
    if selector is None:
        raise HarnessError('the dialog has no "Positions to save" selector')
    list_box = selector.listW.listBox
    texts = [list_box.item(i).text() for i in range(list_box.count())]
    missing = [label for label in labels if label not in texts]
    if missing:
        raise HarnessError(f'positions {missing} not in {texts}')
    list_box.clearSelection()
    for i, text in enumerate(texts):
        list_box.item(i).setSelected(text in labels)


def _image_name_checkbox(dlg):
    """The visible 'Include image name' checkbox (channel 0 row). Not
    `dlg.addImageName_QCB`: on main that attribute is overwritten by the
    hidden dummy of the next channel row when SizeC > 1."""
    from qtpy.QtWidgets import QCheckBox

    boxes = [box for box in dlg.findChildren(QCheckBox)
             if box.text() == 'Include image name' and not box.isHidden()]
    if len(boxes) != 1:
        raise HarnessError(
            f'expected one visible "Include image name" checkbox, found '
            f'{len(boxes)}'
        )
    return boxes[0]


def apply_edit(dlg, key: str, value) -> None:
    """Apply one scripted widget edit (scenarios.DialogAnswer)."""
    name, _, index = key.partition(':')
    if name in ('SizeT', 'SizeZ', 'SizeC', 'SizeS'):
        getattr(dlg, f'{name}_SB').setValue(int(value))
    elif name in ('LensNA', 'PhysicalSizeX', 'PhysicalSizeY',
                  'PhysicalSizeZ', 'TimeIncrement'):
        getattr(dlg, f'{name}_DSB').setValue(float(value))
    elif name in ('TimeIncrementUnit', 'PhysicalSizeUnit'):
        _set_combo(getattr(dlg, f'{name}_CB'), value)
    elif name == 'to_h5':
        dlg.to_h5_radiobutton.setChecked(bool(value))
    elif name == 'timeRange':
        dlg.timeRangeToSaveWidget.setRange(*value)
    elif name == 'positions':
        _select_positions(dlg, list(value))
    elif name == 'addImageName':
        _image_name_checkbox(dlg).setChecked(bool(value))
    elif name == 'chName':
        dlg.chNames_QLEs[int(index)].setText(str(value))
    elif name == 'save':
        dlg.saveChannels_QCBs[int(index)].setChecked(bool(value))
    elif name == 'emWavelen':
        dlg.emWavelens_DSBs[int(index)].setValue(float(value))
    else:
        raise HarnessError(f'unknown dialog edit {key!r}')


def press_button(dlg, button: str) -> None:
    """Press the scripted button of an open metadata dialog."""
    name, _, index = button.partition(':')
    if name == 'ok':
        dlg.ok_cb(None)
    elif name in ('trust', 'reuse'):
        widget = dlg.trustButton if name == 'trust' else dlg.overWriteButton
        if widget is None:
            raise HarnessError(f'the dialog has no {name!r} button')
        widget.click()
    elif name == 'cancel':
        dlg.cancel_cb(None)
    elif name == 'eye':
        widget = dlg.showChannelDataButtons[int(index)]
        if not widget.isEnabled() or not dlg.readSampleImgDataAgain:
            raise HarnessError(
                'the preview button would not restart the dialog (disabled '
                'or no SizeT/SizeZ/SizeC edit)'
            )
        widget.click()
    else:
        raise HarnessError(f'unknown dialog button {button!r}')


def _watch_close(dlg) -> list:
    """List that becomes non-empty when the dialog closes itself
    (`self.close()` in ok_cb, cancel_cb or showChannelData). Visibility
    cannot tell: its closeEvent override leaves the event unaccepted."""
    closed = []
    close = dlg.close

    def recording_close():
        closed.append(True)
        return close()

    dlg.close = recording_close
    return closed


def _dialog_values(dlg) -> dict:
    return {
        name: manifest._jsonable(getattr(dlg, name))
        for name in DIALOG_VALUE_NAMES if hasattr(dlg, name)
    }


# ---------------------------------------------------------------- recorder
class _Recorder:
    """Slots for every worker signal plus the scripted metadata dialog."""

    def __init__(self, scenario: scenarios.Module0Scenario, placeholders,
                 layout_basename: str | None) -> None:
        self.scenario = scenario
        self.placeholders = placeholders
        self.layout_basename = layout_basename
        self.worker = None
        self.signals: list[list] = []
        self.confirm: list[dict] = []
        self.dialogs: list[dict] = []
        self.dialog_values: list[dict] = []
        self.message_boxes: list[dict] = []
        self.task_ended: list[str] = []
        self.pbar_init: list[int] = []
        self.pbar_sum = 0
        self.exceptions: list[list] = []
        self.tracebacks: list[str] = []
        self.harness_errors: list[str] = []
        self._occurrences: dict[str, int] = {}
        self._open_dialogs: list = []

    def norm(self, text) -> str:
        return manifest.normalize_path_text(str(text), self.placeholders)

    def traceback_text(self, text) -> str:
        return strip_traceback_paths(self.norm(text))

    def connect(self, worker) -> None:
        self.worker = worker
        worker.progress.connect(self.on_progress)
        worker.progressPbar.connect(self.on_progress_pbar)
        worker.initPbar.connect(self.on_init_pbar)
        worker.criticalError.connect(self.on_critical_error)
        worker.sigMoveRawMicrFilesFailed.connect(self.on_move_failed)
        worker.confirmMetadata.connect(self.on_confirm_metadata)
        worker.critical.connect(self.on_critical)
        worker.finished.connect(self.on_finished)

    def on_progress(self, text) -> None:
        self.signals.append(['progress', self.norm(text)])

    def on_progress_pbar(self, step) -> None:
        self.pbar_sum += int(step)

    def on_init_pbar(self, total) -> None:
        self.pbar_init.append(int(total))

    def on_critical_error(self, action, traceback_text, filename) -> None:
        self.signals.append([
            'criticalError', str(action), self.norm(filename),
            bool(str(traceback_text).strip()),
        ])
        self.tracebacks.append(self.traceback_text(traceback_text))

    def on_move_failed(self, traceback_text) -> None:
        # createDataStructWin.moveRawMicroscopyFilesFailed emits `finished`
        self.signals.append(
            ['sigMoveRawMicrFilesFailed', bool(str(traceback_text).strip())]
        )
        self.tracebacks.append(self.traceback_text(traceback_text))
        self.worker.finished.emit()

    def on_critical(self, error) -> None:
        entry = [type(error).__name__, self.norm(_first_line(error))]
        self.signals.append(['critical'] + entry)
        self.exceptions.append(entry)
        self.tracebacks.append(self.traceback_text(''.join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )))

    def on_finished(self) -> None:
        self.signals.append(['finished'])
        worker = self.worker
        cancelled = getattr(worker, 'cancelled', False)
        critical = getattr(worker, 'isCriticalError', False)
        if cancelled and not critical:
            self.task_ended.append('cancelled')
        elif not cancelled:
            self.task_ended.append('ended')
        else:
            self.task_ended.append('nothing')

    def on_confirm_metadata(self, *args) -> None:
        # An exception escaping a Qt slot aborts the process (PyQt >= 5.5):
        # record it, cancel the conversion and fail after `run` returns.
        try:
            self._confirm(*args)
        except Exception as err:
            self.harness_errors.append(''.join(traceback.format_exception(
                type(err), err, err.__traceback__
            )))
            self.worker.metadataWin = types.SimpleNamespace(
                cancel=True, requestedReadingSampleImageDataAgain=False
            )

    def _confirm(self, *args) -> None:
        from cellacdc import apps

        values = dict(zip(CONFIRM_ARG_NAMES, args))
        record = {
            name: manifest._jsonable(value) for name, value in values.items()
            if name != 'sampleImgData'
        }
        record['rawFilePath'] = self.norm(values['rawFilePath'])
        record['sampleImgData'] = sample_summary(values['sampleImgData'])
        self.signals.append(['confirmMetadata', len(self.confirm)])
        self.confirm.append(record)

        # createDataStructWin.askConfirmMetadata, with parent=None
        filename = values['filename']
        if self.scenario.layout == 2:
            filename = self.layout_basename
        occurrence = self._occurrences.get(filename, 0)
        self._occurrences[filename] = occurrence + 1
        answer = self.scenario.answer_for(filename, occurrence)
        if answer is None:
            raise HarnessError(
                f'no answer for dialog {occurrence + 1} of {filename!r}'
            )
        dlg = apps.QDialogMetadataXML(
            title=f'Metadata for {filename}', rawFilename=filename,
            LensNA=values['LensNA'], SizeT=values['SizeT'],
            SizeZ=values['SizeZ'], SizeC=values['SizeC'],
            SizeS=values['SizeS'], TimeIncrement=values['TimeIncrement'],
            TimeIncrementUnit=values['TimeIncrementUnit'],
            PhysicalSizeX=values['PhysicalSizeX'],
            PhysicalSizeY=values['PhysicalSizeY'],
            PhysicalSizeZ=values['PhysicalSizeZ'],
            PhysicalSizeUnit=values['PhysicalSizeUnit'],
            ImageName=values['ImageName'], chNames=values['chNames'],
            emWavelens=values['emWavelens'], parent=None,
            rawDataStruct=self.scenario.layout,
            sampleImgData=values['sampleImgData'],
            rawFilePath=values['rawFilePath'],
        )
        self._open_dialogs.append(dlg)
        closed = _watch_close(dlg)
        dlg.show()
        for key, value in answer.edits:
            apply_edit(dlg, key, value)
        press_button(dlg, answer.button)
        accepted = bool(closed)
        if not accepted:
            # The dialog refused the scripted values (it stays open); the
            # user gives up with Cancel.
            dlg.cancel_cb(None)
        self.dialogs.append({
            'title': f'Metadata for {filename}',
            'button': answer.button,
            'accepted': accepted,
        })
        self.dialog_values.append(_dialog_values(dlg))
        self.worker.metadataWin = dlg

    def close_dialogs(self) -> None:
        for dlg in self._open_dialogs:
            dlg.hide()
        self._open_dialogs.clear()


def _separate_multichannel(files: list[str], boxes: list[dict]):
    """createDataStructWin.attemptSeparateMultiChannel on a stub."""
    from cellacdc import dataStruct

    def critical_no_pattern(error=''):
        boxes.append({'method': 'criticalNoFilenamePattern',
                      'title': 'Non-compatible pattern',
                      'text': _first_line(error)})

    stub = types.SimpleNamespace(
        logger=logging.getLogger(LOGGER_NAME),
        criticalNoFilenamePattern=critical_no_pattern,
    )
    proceed = dataStruct.createDataStructWin.attemptSeparateMultiChannel(
        stub, files
    )
    return stub, proceed


def _inject_layout2(worker, stub) -> None:
    """Attributes createDataStructWin.main sets for layout 2.

    On main `chNames` is `list(set)`, so its order depends on the string
    hash seed; it is natsorted here (one of main's possible orders, the one
    the refactor makes deterministic) so goldens do not depend on
    PYTHONHASHSEED. A refactored stub (with `one_per_channel`) is copied
    unchanged."""
    worker.basename = stub.basename
    worker.SizeS = stub.SizeS
    worker.posNums = stub.posNums
    if hasattr(stub, 'one_per_channel'):
        worker.chNames = stub.chNames
        worker.one_per_channel = stub.one_per_channel
    else:
        worker.chNames = natsorted(stub.chNames)


def run_module0_scenario(scenario: scenarios.Module0Scenario | str,
                         workdir: str, *, backend=None) -> dict:
    """Run one scenario in `workdir` (empty or missing) and return its record
    (module docstring). Raises HarnessError when the scenario script cannot
    be applied."""
    if isinstance(scenario, str):
        scenario = scenarios.MODULE0[scenario]
    assert_imports_this_checkout()
    _keep_qapp()
    from cellacdc import dataStruct

    work = os.path.abspath(workdir)
    src = os.path.join(work, 'src')
    dst = os.path.join(work, 'dst')
    os.makedirs(src)
    os.makedirs(dst)
    placeholders = {'<SRC>': src, '<DST>': dst}
    files = natsorted(scenario.make_inputs(src, dst))

    worker_kwargs = {}
    if backend is not None:
        if not worker_accepts_backend():
            raise HarnessError('bioFormatsWorker does not accept backend=')
        worker_kwargs['backend'] = backend

    boxes: list[dict] = []
    stub = None
    if scenario.layout == 2:
        stub, proceed = _separate_multichannel(files, boxes)
        if not proceed:
            raise HarnessError(f'layout 2 names not accepted: {boxes}')
    recorder = _Recorder(
        scenario, placeholders, stub.basename if stub is not None else None
    )
    recorder.message_boxes = boxes

    with children_import_this_checkout(), \
            recorded_message_boxes(boxes, placeholders):
        worker = dataStruct.bioFormatsWorker(
            src, files, dst, FakeMutex(), FakeWaitCondition(),
            scenario.layout, 'bioio', lazy_load=scenario.lazy_load,
            move_raw_microscopy_files=scenario.move_raw,
            overwrite=scenario.overwrite, add_files=scenario.add_files,
            create_new=scenario.create_new,
            start_pos_n=scenario.start_pos_n, **worker_kwargs
        )
        worker.useSymLink = scenario.use_symlinks
        if scenario.layout == 2:
            _inject_layout2(worker, stub)
            worker.moveOtherFiles = scenario.other_files == 'move'
            worker.copyOtherFiles = scenario.other_files == 'copy'
        recorder.connect(worker)
        try:
            worker.run()
        finally:
            recorder.close_dialogs()
    if recorder.harness_errors:
        raise HarnessError('; '.join(recorder.harness_errors))

    return {
        'scenario': scenario.id,
        'files': files,
        'dst': manifest.build_manifest(dst, placeholders=placeholders,
                                       include_env=False),
        'src': manifest.build_manifest(src, placeholders=placeholders,
                                       include_env=False),
        'signals': recorder.signals,
        'confirm': recorder.confirm,
        'dialogs': recorder.dialogs,
        'message_boxes': recorder.message_boxes,
        'task_ended': recorder.task_ended,
        'pbar': {'init': recorder.pbar_init, 'step_sum': recorder.pbar_sum},
        'exceptions': recorder.exceptions,
        manifest.INFO_KEY: {
            'description': scenario.description,
            'channel_order': list(getattr(worker, 'chNames', []) or [])
            if scenario.layout == 2 else None,
            'dialog_values': recorder.dialog_values,
            'tracebacks': recorder.tracebacks,
        },
    }


# ----------------------------------------------------------------- cascade
def _cascade_worker(layout: int, placeholders: dict):
    """A legacy worker whose `readMetadata` runs without children: the
    dialog answers Cancel and sample data is not read."""
    from cellacdc import dataStruct

    worker = dataStruct.bioFormatsWorker(
        placeholders['<SRC>'], [], placeholders['<SRC>'], FakeMutex(),
        FakeWaitCondition(), layout, 'bioio'
    )
    if layout == 2:
        worker.SizeS = scenarios.CASCADE_LAYOUT2_SIZE_S
        worker.chNames = list(scenarios.CASCADE_LAYOUT2_CHANNELS)
    worker.readSampleData = lambda *args, **kwargs: None
    log: list[str] = []
    confirm: list[dict] = []
    critical: list[list] = []

    def on_confirm(*args):
        record = dict(zip(CONFIRM_ARG_NAMES, manifest._jsonable(list(args))))
        record['rawFilePath'] = manifest.normalize_path_text(
            record['rawFilePath'], placeholders
        )
        confirm.append(record)
        worker.metadataWin = types.SimpleNamespace(cancel=True)

    worker.progress.connect(lambda text: log.append(
        manifest.normalize_path_text(str(text), placeholders)
    ))
    worker.confirmMetadata.connect(on_confirm)
    worker.criticalError.connect(lambda action, tb, filename: critical.append(
        [str(action), str(filename), bool(str(tb).strip())]
    ))
    return worker, log, confirm, critical


def _cascade_entry(worker, log, confirm, critical, src, filename) -> dict:
    del log[:], confirm[:], critical[:]
    cancelled = worker.readMetadata(src, filename)
    return {
        'returned': cancelled,
        'confirm': confirm[0] if confirm else None,
        'log': list(log),
        'criticalError': critical[0] if critical else None,
    }


def capture_cascade(workdir: str) -> dict:
    """Legacy `readMetadata` cascade: every
    `fake_omexml.cascade_grid()` entry through a monkeypatched
    `_readMetadataBioIO` (layouts 1 and 2) and every real `.ome.tif`
    variant of `scenarios.OME_CASCADE_VARIANTS` (layout 1). Each entry holds
    the confirmMetadata args, the progress lines and readMetadata's return
    value."""
    assert_imports_this_checkout()
    _keep_qapp()
    src = os.path.join(os.path.abspath(workdir), 'src')
    os.makedirs(src)
    placeholders = {'<SRC>': src}
    grid = scenarios.cascade_grid()
    fake = {}
    for layout in scenarios.CASCADE_LAYOUTS:
        worker, log, confirm, critical = _cascade_worker(layout, placeholders)
        entries = {}
        for name, spec in grid.items():
            worker._readMetadataBioIO = (
                lambda path, spec=spec: (fake_omexml.FakeOMEXML(spec), 'xml')
            )
            entry = _cascade_entry(worker, log, confirm, critical, src,
                                   scenarios.CASCADE_FAKE_FILENAME)
            entry['spec'] = spec
            entries[name] = entry
        fake[f'layout{layout}'] = entries
    worker, log, confirm, critical = _cascade_worker(1, placeholders)
    ome = {}
    for name in scenarios.OME_CASCADE_VARIANTS:
        filename = scenarios.make_ome_cascade_file(src, name)
        ome[name] = _cascade_entry(worker, log, confirm, critical, src,
                                   filename)
        ome[name]['filename'] = filename
    return {
        'fake_filename': scenarios.CASCADE_FAKE_FILENAME,
        'layout2_channels': list(scenarios.CASCADE_LAYOUT2_CHANNELS),
        'layout2_size_s': scenarios.CASCADE_LAYOUT2_SIZE_S,
        'fake': fake,
        'ome_tif': {'layout1': ome},
    }
