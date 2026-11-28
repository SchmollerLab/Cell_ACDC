"""Private QObject workers of the measurements viewer: the thread task
worker used by runInThread (measurements_viewer/main_window/data_io.py)
and the one-shot measurement processing worker.
"""
import traceback
from qtpy.QtCore import Signal
from cellacdc import measurement_processing
from cellacdc.shared.workers import base as workers_base
from qtpy.QtCore import QThread, QObject, QCoreApplication


class MeasurementProcessingWorkerGUI(workers_base.CustomPreprocessWorkerGUI):
    """One-shot measurement job using the preprocessing worker's job API.

    ``max_frame_i`` (plain int or None, given to ``setupJob``) is passed to
    ``measurement_processing.apply_recipe``: only rows up to that frame are
    processed (the worker never reads GUI state).
    """
    sigDone = Signal(object)
    def applyRecipe(self, df, cols, recipe, new_colnames, max_frame_i=None):
        return measurement_processing.apply_recipe(
            df, cols, recipe, new_colnames, max_frame_i=max_frame_i
        )
            
    def setupJob(self, df, cols, recipe, new_colnames, max_frame_i=None):
        self._df = df
        self._cols = cols
        self._recipe = recipe
        self._new_colnames = new_colnames
        self._max_frame_i = max_frame_i
        
    def runJob(self):
        try:
            df = self.applyRecipe(
                self._df,
                self._cols,
                self._recipe,
                self._new_colnames,
                max_frame_i=self._max_frame_i
            )
            self.sigDone.emit(df)
            return
        except Exception as error:
            self.logger.exception(traceback.format_exc())
            self.signals.critical.emit((self, error))
        finally:
            self.signals.finished.emit(self)
        
    def run(self):
        # One-shot job (no preview queue): the result is emitted by runJob 
        # through `sigDone`, errors through `signals.critical`.
        self.runJob()
    
    def enqueue(self, *args, **kwargs):
        raise NotImplementedError(
            'MeasurementProcessingWorkerGUI runs a single job, use setupJob'
        )


# Old names of cellacdc.workers that code reached through this module
# path before it existed (cellacdc/_compat.py).
from cellacdc import _compat as _acdc_compat  # noqa: E402

__getattr__, __dir__ = _acdc_compat.forward_module(
    __name__, 'cellacdc.workers')


class _ThreadTaskWorker(QObject):
    """Run `func(*args, **kwargs)` in a worker `QThread` (see
    `MeasurementsViewerWin.runInThread`). The return value or the raised
    exception is stored in `result` / `error`, to be read once the thread
    finished. With `reportProgress`, `func` also gets the keyword argument
    ``progress``, a callable that emits `sigProgress` with a text."""
    sigProgress = Signal(str)

    def __init__(self, func, args, kwargs, reportProgress=False):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = dict(kwargs)
        if reportProgress:
            self.kwargs['progress'] = self.sigProgress.emit
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self.func(*self.args, **self.kwargs)
        except BaseException as err:
            self.error = err
        finally:
            # Back to the GUI thread, so that `deleteLater` works after this
            # thread stopped
            app = QCoreApplication.instance()
            if app is not None:
                self.moveToThread(app.thread())
            QThread.currentThread().quit()
