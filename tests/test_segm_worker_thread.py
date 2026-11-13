import os
import subprocess
import sys
import textwrap
from collections import namedtuple

import pytest


@pytest.mark.parametrize("mode, delay", [
    ("success", "run"),
    ("success", "finish"),
    ("error", "run"),
    ("error", "finish"),
    ("orphan", None),
])
def test_segm_worker_threads_are_finished_before_release(mode, delay):
    """Destroying a QThread that is still running aborts the process with
    "QThread: Destroyed while thread is still running". This happened at
    random between two positions of the segmentation module (the worker
    thread had emitted `finished` but had not finished yet) and every time
    after a worker error (the thread was never quit). Run in a subprocess,
    because the abort cannot be caught.

    `delay` keeps the worker thread busy (and needing the GIL) either right
    after emitting `finished` from `SegmWorker.run` ("run") or inside
    `QThreadPrivate::finish` ("finish", when `isRunning` is already False).
    In "orphan" mode the wait loop is exited before the worker finished
    (like when the app quits) and the thread must be kept alive.
    """
    script = textwrap.dedent("""
        import gc
        import os
        import sys
        import threading
        import time
        import traceback
        import weakref

        from qtpy.QtCore import QEventLoop, QTimer, Qt
        from qtpy.QtWidgets import QApplication

        import cellacdc
        from cellacdc import segm

        app = QApplication([])

        # Do not let an escaped exception pass as a successful test run
        def excepthook(*args):
            traceback.print_exception(*args, file=sys.__stderr__)
            sys.__stderr__.flush()
            os._exit(97)

        sys.excepthook = excepthook
        mode, delay = sys.argv[1], sys.argv[2]

        def process_events(ms):
            loop = QEventLoop()
            QTimer.singleShot(ms, loop.quit)
            loop.exec_()

        errors = []
        def report_error(window, name):
            # Do not store the exception: its traceback would keep the
            # frames (and the threads) of the segmentation loop alive
            error = sys.exc_info()[1]
            errors.append((name, str(error)))
            # Mimic the modal error dialog (nested event loop)
            process_events(50)

        cellacdc._critical_exception_gui = report_error

        thread_refs = []
        class TestQThread(segm.QThread):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # Weak reference: a strong one would hide the bug
                thread_refs.append(weakref.ref(self))
                if delay == 'finish':
                    # Keep the thread inside `QThreadPrivate::finish`
                    # with a slot that needs the GIL
                    self.finished.connect(
                        lambda: time.sleep(0.05),
                        Qt.ConnectionType.DirectConnection
                    )

        segm.QThread = TestQThread

        class SlowFinishedSignal:
            def __init__(self, signal):
                self._signal = signal

            def connect(self, *args, **kwargs):
                return self._signal.connect(*args, **kwargs)

            def emit(self, *args):
                self._signal.emit(*args)
                time.sleep(0.03)

        class WorkerSignals:
            def __init__(self, signals):
                self._signals = signals
                self.finished = SlowFinishedSignal(signals.finished)

            def __getattr__(self, name):
                return getattr(self._signals, name)

        processed = []
        release_worker = threading.Event()
        def init_kernel(worker, mainWin):
            if delay == 'run':
                worker.signals = WorkerSignals(worker.signals)
            if mode == 'orphan':
                # Exit the wait loop before the worker finished
                QTimer.singleShot(100, lambda: worker.loop.exit())

        def run_kernels(worker):
            if mode == 'orphan':
                release_worker.wait(30)
            if mode == 'error' and worker.img_path == 'pos1':
                raise RuntimeError('simulated segmentation error')
            processed.append(worker.img_path)

        segm.SegmWorker.init_kernel = init_kernel
        segm.SegmWorker.run_kernels = run_kernels

        win = segm.segmWin()
        win.user_ch_file_paths = [f'pos{i}' for i in range(4)]
        win.stopFrames = [1]*4
        win.runSegmWorkers()

        if mode == 'orphan':
            # The still running thread is kept alive and the loop stopped
            assert processed == [], processed
            assert len(segm.SEGM_ORPHAN_THREADS) == 1
            release_worker.set()
            thread, worker = segm.SEGM_ORPHAN_THREADS.pop()
            assert thread.wait(10000)
            del thread, worker

        # Release everything like `segmWin.main` returning
        gc.collect()
        process_events(200)
        gc.collect()

        if mode == 'error':
            assert processed == ['pos0'], processed
            assert errors == [
                ('workerCritical', 'simulated segmentation error')
            ], errors
        elif mode == 'orphan':
            assert processed == ['pos0'], processed
            assert not errors, errors
        else:
            assert processed == ['pos0', 'pos1', 'pos2', 'pos3'], processed
            assert not errors, errors

        assert not segm.SEGM_ORPHAN_THREADS, segm.SEGM_ORPHAN_THREADS
        alive_threads = [ref() for ref in thread_refs if ref() is not None]
        assert all(thread.isFinished() for thread in alive_threads)
        print('Segmentation worker threads released safely', flush=True)
    """)
    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", script, mode, str(delay)],
            capture_output=True, text=True, timeout=60,
            env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
        )
    except subprocess.TimeoutExpired as err:
        pytest.fail(
            'Segmentation worker threads test hung (deadlock?).\n'
            f'{err.stdout}\n{err.stderr}'
        )
    output = result.stdout + result.stderr
    assert "QThread: Destroyed while thread is still running" not in output
    assert result.returncode == 0, output
    assert "Segmentation worker threads released safely" in result.stdout


ArgSpec = namedtuple('ArgSpec', ['name', 'type'])

def test_parse_model_params_keeps_explicit_none():
    from cellacdc import myutils

    argspecs = [
        ArgSpec('model_type', str),
        ArgSpec('model_path', os.PathLike),
        ArgSpec('batch_size', int),
        ArgSpec('augment', bool),
        ArgSpec('diameter', float),
    ]
    parsed = myutils.parse_model_params(argspecs, {
        'model_type': None, 'model_path': 'model.pt', 'batch_size': '36',
        'augment': None, 'unknown': 'value',
    })

    # Explicit None is passed to the model (not replaced by its default),
    # missing and unknown parameters are skipped
    assert parsed == {
        'model_type': None, 'model_path': 'model.pt', 'batch_size': 36,
        'augment': None,
    }


def test_workflow_config_roundtrip_keeps_model_type_none(tmp_path):
    """Running a workflow saved from the segmentation module with a custom
    cellpose `model_path` failed with "You cannot set both `model_type` and
    `model_path`", because `model_type = None` was dropped and the model
    default (`model_type='cpsam'`) was used instead.
    """
    from cellacdc import load, myutils

    images_path = tmp_path / 'Position_1' / 'Images'
    images_path.mkdir(parents=True)
    ini_path = tmp_path / 'workflow.ini'
    init_kwargs = {
        'model_type': None, 'model_path': 'C:/models/custom_model',
        'device_type': 'gpu', 'device': None, 'batch_size': 36,
    }
    load.save_workflow_to_config(
        str(ini_path), {'init_segmentation_model_params': init_kwargs},
        [str(images_path)], [1]
    )
    workflow_params = load.read_segm_workflow_from_config(str(ini_path))

    argspecs = [
        ArgSpec('model_type', str),
        ArgSpec('model_path', os.PathLike),
        ArgSpec('device_type', str),
        ArgSpec('device', str),
        ArgSpec('batch_size', int),
    ]
    parsed = myutils.parse_model_params(
        argspecs, workflow_params['init_segmentation_model_params']
    )
    assert parsed == init_kwargs
