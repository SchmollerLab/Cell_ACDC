"""Private QObject workers of the measurements viewer: the thread task
worker used by runInThread (measurements_viewer/main_window/data_io.py)
and the one-shot measurement processing worker.
"""
import traceback
from qtpy.QtCore import Signal
from cellacdc import measurement_processing
from cellacdc.shared.workers import base as workers_base


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
