"""Shared worker infrastructure and generic workers.

worker_exception_handler decorates the run method of every worker.
workerLogger and the signals QObject are the logging/progress signals of
QObject workers, also used by utility workers and the CellACDC_2steps
tracker. SimpleWorker runs a function in a QThread.
CustomPreprocessWorkerGUI is the queued preprocessing job base
subclassed by gui/workers/image_processing.py and
measurements_viewer/workers.py. ApplyImageFilterWorker,
ScreenRecorderWorker (record.py) and AutoPilotWorker are generic workers
without static users.

Foundation leaf: this module imports no dialog, widget, gui or utils
module, so any module can import it at module level without an import
cycle.
"""
import os
import time
from collections import deque
from typing import Union, List, Dict, Callable, Any, Tuple
from functools import wraps
import numpy as np
import traceback
from qtpy.QtCore import Signal, QObject
from cellacdc import printl
from cellacdc.myutils import image_utils


DEBUG = False

def worker_exception_handler(func):
    @wraps(func)
    def run(self):
        try:
            func(self)
        except Exception as error:
            printl(traceback.format_exc())
            try:
                self.dataQ.clear()
            except Exception as err:
                pass
            
            # Some workers have both self.critical and self.signals.critical 
            # errors but only one of them is connected --> emit both just 
            # in case, but each of them only once (emitting the same signal
            # twice shows the error dialog twice)
            for owner in (self, getattr(self, 'signals', None)):
                critical = getattr(owner, 'critical', None)
                if critical is None:
                    continue
                try:
                    critical.emit((self, error))
                except Exception as err:
                    pass
            
            # Release the mutex in case the worker raised while holding it.
            # Unlocking a mutex that is not locked crashes on Qt5, so first
            # lock it if it is free (`tryLock` does nothing and returns
            # False if the mutex is already locked)
            try:
                self.mutex.tryLock()
                self.mutex.unlock()
            except Exception as err:
                pass
    return run

class workerLogger:
    def __init__(self, sigProcess):
        self.sigProcess = sigProcess
    
    def log(self, message, level='INFO'):
        try:
            self.sigProcess.emit(str(message), level)
        except Exception as err:
            print(message, level)
            try:
                traceback_format = traceback.format_exc()
                print(traceback_format)
            except Exception as err:
                pass
            printl(err)
        finally:
            pass
    
    def info(self, message):
        self.log(message, level='INFO')

    def warning(self, message):
        self.log(message, level='WARNING')
    
    def exception(self, message):
        self.log(message, level='EXCEPTION')

class signals(QObject):
    progress = Signal(str, object)
    finished = Signal(object)
    initProgressBar = Signal(int)
    progressBar = Signal(int)
    critical = Signal(object)
    dataIntegrityWarning = Signal(str)
    dataIntegrityCritical = Signal()
    sigLoadingFinished = Signal()
    sigLoadingNewChunk = Signal(object)
    resetInnerPbar = Signal(int)
    progress_tqdm = Signal(int)
    signal_close_tqdm = Signal()
    create_tqdm = Signal(int)
    innerProgressBar = Signal(int)
    sigPermissionError = Signal(str, object)
    sigSelectSegmFiles = Signal(object, object)
    sigSelectAcdcOutputFiles = Signal(object, object, str, bool, bool)
    sigSelectSpotmaxRun = Signal(object, object, object, str, bool, bool)
    sigSetMeasurements = Signal(object)
    sigInitAddMetrics = Signal(object, object)
    sigUpdatePbarDesc = Signal(str)
    sigComputeVolume = Signal(int, object)
    sigAskStopFrame = Signal(object)
    sigWarnMismatchSegmDataShape = Signal(object)
    sigErrorsReport = Signal(dict, dict, dict)
    sigMissingAcdcAnnot = Signal(dict)
    sigRecovery = Signal(object)
    sigInitInnerPbar = Signal(int)
    sigUpdateInnerPbar = Signal(int)
    sigSelectFile = Signal(str, str, str)
    sigAskCopyCca = Signal(str)
    sigSelectFilesWithText = Signal(str, object, str, object)
    sigAskRunNow = Signal(object)
    sigSelectVideoFile = Signal(object, object, bool)

class AutoPilotWorker(QObject):
    finished = Signal()
    critical = Signal(object)
    progress = Signal(str, object)
    sigStarted = Signal()
    sigStopTimer = Signal()

    def __init__(self, guiWin):
        QObject.__init__(self)
        self.logger = workerLogger(self.progress)
        self.guiWin = guiWin
        self.app = guiWin.app
        # self.timer = timer
    
    def timerCallback(self):
        pass
    
    def stop(self):
        self.sigStopTimer.emit()
        self.finished.emit()        
    
    def run(self):
        self.sigStarted.emit()


class ScreenRecorderWorker(QObject):
    sigGrabScreen = Signal()
    finished = Signal()

    def __init__(self, screenRecorderWin, folder_path):
        QObject.__init__(self)
        self.screenRecorderWin = screenRecorderWin
        self.folder_path = folder_path
    
    def run(self):
        for i in range(4):
            fn = f'shot_{i:03}.jpg'
            grab_path = os.path.join(self.folder_path, fn)
            screen = self.screenRecorderWin.screen()
            screenshot = screen.grabWindow(self.screenRecorderWin.winId())
            screenshot.save(grab_path, 'jpg')
            print(grab_path)
            time.sleep(0.2)

        self.finished.emit()


class ApplyImageFilterWorker(QObject):
    finished = Signal(object)
    critical = Signal(object)
    progress = Signal(str)
    
    def __init__(self, filter_func, input_data):
        QObject.__init__(self)
        self.filter_func = filter_func
        self.input_data = input_data
    
    @worker_exception_handler
    def run(self):
        self.progress.emit('Filtering image...')
        filtered_data = self.filter_func(self.input_data)
        self.finished.emit(filtered_data)


class SimpleWorker(QObject):
    def __init__(self, posData, func, func_args=None, func_kwargs=None):
        QObject.__init__(self)
        self.posData = posData
        self.signals = signals()
        self.output = {}
        
        if func_args is None:
            func_args = []
        
        if func_kwargs is None:
            func_kwargs = {}
        
        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs
        self.posData = posData
    
    def set_func_kwargs(self, func_kwargs: dict[str, Any]):
        self.func_kwargs = func_kwargs
    
    @worker_exception_handler
    def run(self):
        if self.posData is None:
            self.output = self.func(*self.func_args, **self.func_kwargs)
        else:
            self.output = self.func(
                self.posData, *self.func_args, **self.func_kwargs
            )
        self.signals.finished.emit(self.output)


class CustomPreprocessWorkerGUI(QObject):
    sigDone = Signal(object, str)
    sigPreviewDone = Signal(object, tuple)
    sigIsQueueEmpty = Signal(bool)
    
    def __init__(self, mutex, waitCond):
        QObject.__init__(self)
        self.signals = signals()
        self.mutex = mutex
        self.waitCond = waitCond
        self.logger = workerLogger(self.signals.progress)
        self.dataQ = deque(maxlen=2)
        self.exit = False
        self.wait = True
        self._abort = False
    
    def enqueue(
            self, 
            func: Callable,
            image: np.ndarray, 
            recipe: Dict[str, Any],
            key: Tuple[int, int, Union[int, str]]
        ):
        self.dataQ.append((func, image, recipe, key))
        if len(self.dataQ) == 1:
            self.sigIsQueueEmpty.emit(False)
            # Wake up worker upon inserting first element
            self.wakeUp()
    
    def wakeUp(self):
        self.wait = False
        self.waitCond.wakeAll()
    
    def pause(self):
        self.wait = True
        self.mutex.lock()
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
    
    def abort(self):
        self._abort = True
    
    def stop(self):
        self.abort()
        self.exit = True
        self.waitCond.wakeAll()
        self.signals.finished.emit(self)
    
    def setupJob(
            self, 
            func: Callable, 
            image_data: np.ndarray, 
            recipe: Dict[str, Any],
            how: str
        ):
        self._func = func
        self._image_data = image_data
        self._recipe = recipe
        self._how = how
    
    def runJob(self, image=None, recipe=None):
        if image is None:
            image = self._image_data.copy()
        if recipe is None:
            recipe = self._recipe
        
        return self.applyRecipe(self._func, image, recipe)
        
    def applyRecipe(
            self, 
            func: Callable, 
            image: np.ndarray, 
            recipe: List[Dict[str, Any]]
        ):
        preprocessed_data = func(image, recipe)

        keep_input_data_type = recipe[0].get('keep_input_data_type', True)
        if not keep_input_data_type:
            return preprocessed_data

        try:
            preprocessed_data = image_utils.convert_to_dtype(
                preprocessed_data, image.dtype
            )
        except Exception as err:
            preprocessed_data = preprocessed_data.astype(image.dtype)
        return preprocessed_data
        
    @worker_exception_handler
    def run(self):
        while True:
            if self.exit:
                self.logger.log('Closing pre-processing worker...')
                break
            elif self.wait:
                self.logger.log('Pre-processing worker paused.')
                self.pause()
            elif len(self.dataQ) > 0:
                func, image, recipe, key = self.dataQ.pop()
                processed_data = self.applyRecipe(func, image, recipe)
                self.sigPreviewDone.emit(processed_data, key)
                if len(self.dataQ) == 0:
                    self.wait = True
                    self.sigIsQueueEmpty.emit(True)
            else:
                self.logger.log('Pre-processing worker resumed.')
                processed_data = self.runJob()
                self.sigDone.emit(processed_data, self._how)
                self.wait = True

        self.signals.finished.emit(self)
