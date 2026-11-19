"""`bioFormatsWorker` in a real QThread.

As in `createDataStructWin.main`: the worker runs `convert()` in its own
QThread with a real QMutex and QWaitCondition (`InProcessBackend` +
FakeBioImage). `confirmMetadata` reaches the real
`createDataStructWin.askConfirmMetadata` of a host widget in the GUI thread
(queued), which builds the real `QDialogMetadataXML`; a QTimer presses the
scripted button after `delay_ms` (0 = at once).

Checks: no deadlock or lost wake-up (50 files answered at once, files
answered after 300 ms), each file is written only after its dialog was
accepted, Cancel, the eye re-read loop, and an error: `critical`, then
`finished` and the thread finishes (GV-19).
"""
from __future__ import annotations

import os

import pytest

from . import conditions
from .test_module0_adapter import (
    SpyBackend, fake_file, one_per_pos_files, written_positions
)

pytestmark = conditions.requires_qt

TIMEOUT_S = 120


@pytest.fixture(scope='module')
def qt():
    """The QApplication and the Qt classes used here (imported lazily, so
    the module is collected without Qt)."""
    from qtpy import QtCore, QtWidgets

    return {'app': conditions.qapp(), 'QtCore': QtCore,
            'QtWidgets': QtWidgets}


@pytest.fixture
def scripted_dialogs(qt, monkeypatch):
    """Replace `apps.QDialogMetadataXML` by a subclass that presses the
    next scripted button `delay_ms` after `exec_` starts; message boxes do
    not block. Returns `configure(script, delay_ms, events)`."""
    from cellacdc import apps, widgets

    QTimer = qt['QtCore'].QTimer
    settings = {'script': [], 'delay_ms': 0, 'events': []}

    class MessageBox:
        def __init__(self, *args, **kwargs) -> None:
            self.cancel = False

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    class ScriptedDialog(apps.QDialogMetadataXML):
        def exec_(self):
            action = settings['script'].pop(0) if settings['script'] else 'ok'
            QTimer.singleShot(settings['delay_ms'], lambda: self._act(action))
            super().exec_()

        def _act(self, action: str) -> None:
            settings['events'].append(('dialog', self.windowTitle(), action))
            if action == 'ok':
                self.ok_cb(None)
            elif action == 'cancel':
                self.cancel_cb(None)
            elif action == 'eye':
                # Edited sizes: the preview button restarts the dialog
                self.SizeT_SB.setValue(2)
                self.SizeZ_SB.setValue(1)
                self.showChannelDataButtons[0].click()

    monkeypatch.setattr(apps, 'QDialogMetadataXML', ScriptedDialog)
    monkeypatch.setattr(widgets, 'myMessageBox', MessageBox)

    def configure(script, delay_ms, events):
        settings.update(script=list(script), delay_ms=delay_ms, events=events)
    return configure


def run_in_thread(qt, src, files, dst, *, backend, events):
    """Run the worker as `createDataStructWin.main` does and wait (in a
    local event loop) until its thread finished. Returns the worker and
    whether the thread finished in time."""
    from cellacdc import dataStruct

    QtCore, QtWidgets = qt['QtCore'], qt['QtWidgets']

    class Host(QtWidgets.QWidget):
        """Stands in for createDataStructWin (GUI thread)."""
        askConfirmMetadata = dataStruct.createDataStructWin.askConfirmMetadata

        def __init__(self, worker, mutex, waitCond) -> None:
            super().__init__()
            self.rawDataStruct = 1
            self.metadataDialogIsOpen = False
            self.worker = worker
            self.mutex = mutex
            self.waitCond = waitCond

        def on_critical(self, error) -> None:
            events.append(('critical', type(error).__name__, str(error)))

        def on_finished(self) -> None:
            events.append(('finished',))

    mutex, waitCond = QtCore.QMutex(), QtCore.QWaitCondition()
    thread = QtCore.QThread()
    worker = dataStruct.bioFormatsWorker(
        str(src), list(files), str(dst), mutex, waitCond, 1, 'bioio',
        move_raw_microscopy_files=False, backend=backend
    )
    worker.useSymLink = False
    host = Host(worker, mutex, waitCond)
    worker.moveToThread(thread)
    worker.finished.connect(thread.quit)
    worker.finished.connect(host.on_finished)
    worker.critical.connect(host.on_critical)
    worker.confirmMetadata.connect(host.askConfirmMetadata)
    thread.started.connect(worker.run)

    loop = QtCore.QEventLoop()
    thread.finished.connect(loop.quit)
    guard = QtCore.QTimer()
    guard.setSingleShot(True)
    guard.timeout.connect(loop.quit)
    guard.start(TIMEOUT_S * 1000)
    thread.start()
    loop.exec_()
    finished = thread.wait(5000)
    if not finished:
        # Deadlock: release the worker (Cancel) before failing
        worker.metadataWin = type('Cancelled', (), {'cancel': True})()
        waitCond.wakeAll()
        thread.wait(10000)
    host.deleteLater()
    return worker, finished


def dialog_index(events, filename: str, action: str = 'ok') -> int:
    return events.index(('dialog', f'Metadata for {filename}', action))


def save_index(events, input_id: int) -> int:
    return events.index(('save_positions', [input_id]))


@pytest.mark.parametrize('count, delay_ms', [(50, 0), (3, 300)])
def test_no_deadlock_and_each_file_written_after_its_dialog(
        qt, scripted_dialogs, tmp_path, count, delay_ms):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = one_per_pos_files(src, count=count, T=2, C=1, Y=2, X=2)
    events = []
    backend = SpyBackend()
    backend.calls = events          # one timeline for both threads
    scripted_dialogs([], delay_ms, events)
    worker, finished = run_in_thread(qt, src, files, dst, backend=backend,
                                     events=events)
    assert finished, 'the worker thread did not finish (deadlock?)'
    assert events[-1] == ('finished',) and events.count(('finished',)) == 1
    assert not any(event[0] == 'critical' for event in events)
    assert not worker.cancelled
    for i, filename in enumerate(files, start=1):
        assert dialog_index(events, filename) < save_index(events, i)
    assert len(written_positions(dst)) == count


def test_cancel_in_the_first_dialog(qt, scripted_dialogs, tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = one_per_pos_files(src, count=2)
    events = []
    backend = SpyBackend()
    backend.calls = events
    scripted_dialogs(['cancel'], 0, events)
    worker, finished = run_in_thread(qt, src, files, dst, backend=backend,
                                     events=events)
    assert finished
    assert [e for e in events if e[0] in ('dialog', 'finished')] == [
        ('dialog', 'Metadata for pos1.czi', 'cancel'), ('finished',)
    ]
    assert worker.cancelled and not worker.isCriticalError
    assert written_positions(dst) == []


def test_eye_reread_shows_the_dialog_again(qt, scripted_dialogs, tmp_path):
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = [fake_file(src, 'pos1.czi', Z=2)]
    events = []
    backend = SpyBackend()
    backend.calls = events
    scripted_dialogs(['eye', 'ok'], 0, events)
    worker, finished = run_in_thread(qt, src, files, dst, backend=backend,
                                     events=events)
    assert finished
    assert [e for e in events if e[0] in ('dialog', 'read_sample')] == [
        ('read_sample', (2, 3, 2)),
        ('dialog', 'Metadata for pos1.czi', 'eye'),
        ('read_sample', (2, 2, 1)),
        ('dialog', 'Metadata for pos1.czi', 'ok'),
    ]
    csv = os.path.join(str(dst), 'Position_1', 'Images',
                       'pos1_s1_metadata.csv')
    with open(csv) as file:
        rows = file.read().splitlines()
    assert 'SizeT,2' in rows and 'SizeZ,1' in rows


def test_error_reports_critical_then_finishes_the_thread(
        qt, scripted_dialogs, tmp_path):
    """GV-19: `critical` (error dialog), then `finished`: the thread quits
    and `taskEnded` shows nothing more."""
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    files = one_per_pos_files(src, count=1)
    events = []
    backend = SpyBackend(save_positions=RuntimeError('disk full'))
    backend.calls = events
    scripted_dialogs([], 0, events)
    worker, finished = run_in_thread(qt, src, files, dst, backend=backend,
                                     events=events)
    assert finished
    assert events[-2:] == [('critical', 'RuntimeError', 'disk full'),
                           ('finished',)]
    assert worker.isCriticalError and worker.cancelled
