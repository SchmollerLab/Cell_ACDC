from typing import TYPE_CHECKING

from qtpy.QtCore import QThread

from . import apps, myutils, workers
from . import exception_handler

# The Qt-free core moved to acdc_datastruct.restructure (re-exported here
# under the same names).
from .acdc_datastruct.restructure import (  # noqa: F401
    fail, mode_for, tiff_stem, natural_key, parse_index, parse_channel,
    axis_index, get_layout, normalize_array,
    create_datastruct_from_restruct_tiff_dialogue_options,
)

if TYPE_CHECKING:
    from . import _main

def _log(mainWin, text):
    mainWin.log(text)

@exception_handler
def run(mainWin: '_main.mainWin'):
    setupWindow = apps.ReStructTiffDataStructureSetupDialogue(parent=mainWin)
    setupWindow.exec_()
    if setupWindow.cancel:
        return False

    mainWin._restructure_thread = QThread()
    mainWin._restructure_worker = workers.SimpleWorker(
        None, create_datastruct_from_restruct_tiff_dialogue_options,
        func_args=(setupWindow.selectedOptions,),
    )
    mainWin._restructure_worker.set_func_kwargs(
        {'signals': mainWin._restructure_worker.signals}
    )
    mainWin._restructure_worker.moveToThread(mainWin._restructure_thread)
    mainWin._restructure_worker.signals.finished.connect(
        mainWin._restructure_thread.quit
    )
    mainWin._restructure_worker.signals.finished.connect(
        mainWin._restructure_worker.deleteLater
    )
    mainWin._restructure_thread.finished.connect(
        mainWin._restructure_thread.deleteLater
    )

    mainWin._restructure_worker.signals.critical.connect(
        mainWin.workerCritical
    )
    mainWin._restructure_worker.signals.critical.connect(
        mainWin._restructure_thread.quit
    )
    mainWin._restructure_worker.signals.initProgressBar.connect(
        mainWin.workerInitProgressbar
    )
    mainWin._restructure_worker.signals.progressBar.connect(
        mainWin.workerUpdateProgressbar
    )
    mainWin._restructure_worker.signals.progress.connect(
        mainWin.workerProgress
    )
    mainWin._restructure_worker.signals.finished.connect(
        mainWin.workerFinished
    )

    mainWin._restructure_thread.started.connect(mainWin._restructure_worker.run)
    mainWin._restructure_thread.start()

    return True

def saveTiff(filePath, data, waitCond):
    myutils.to_tiff(filePath, data)
    waitCond.wakeAll()
    del data