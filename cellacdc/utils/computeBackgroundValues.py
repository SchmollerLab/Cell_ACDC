import os

import numpy as np
import pandas as pd
from qtpy.QtCore import QThread

from .. import (
    apps, myutils, workers, widgets, html_utils, background_values, load
)

from .base import NewThreadMultipleExpBaseUtil


def select_positions(parent, experimentPositions):
    """Ask which Positions of the given experiments to process (the
    experiments are not asked).

    `experimentPositions` is `{exp_path: [pos_foldername, ...]}` with the
    Positions preselected in the prompt (all when empty). Experiments with
    a single Position are not asked. Returns `{exp_path: [pos_foldername,
    ...]}` or None if cancelled or nothing was selected.
    """
    selection = {}
    for exp_path, preselected in experimentPositions.items():
        pos_foldernames = myutils.get_pos_foldernames(exp_path)
        if len(pos_foldernames) <= 1:
            if pos_foldernames:
                selection[exp_path] = pos_foldernames
            continue
        select = load.select_exp_folder()
        values = select.get_values_acdc_and_spotmax(exp_path)
        preselected = set(preselected or pos_foldernames)
        selectedValues = [
            value for pos, value in zip(pos_foldernames, values)
            if pos in preselected
        ]
        select.QtPrompt(
            parent, values, toggleMulti=True,
            informativeText=html_utils.paragraph(
                'Select the Position folder(s) of the experiment<br><br>'
                f'<code>{exp_path}</code><br><br>'
                'to compute the background values from:'
            ),
            selectedValues=selectedValues or values
        )
        if select.cancel:
            return None
        if select.selected_pos:
            selection[exp_path] = list(select.selected_pos)
    return selection or None


def _format_value(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'n.a.'
    return f'{value:.4g}'


def summary_html(summary_df: pd.DataFrame, num_positions: int) -> str:
    """HTML table of the pooled background and normalization constants."""
    if summary_df.empty or num_positions == 0:
        return 'No Position was processed (see the log for details).'
    low = summary_df['low_percentile'].iloc[0]
    high = summary_df['high_percentile'].iloc[0]
    statistic = summary_df['background_statistic'].iloc[0]
    header = (
        '<tr><th>Channel</th>'
        f'<th>Constant background<br>(median of per-frame {statistic})</th>'
        f'<th>Constant: min<br>({low:g}th perc.)</th>'
        f'<th>Constant: max<br>({high:g}th perc.)</th></tr>'
    )
    rows = []
    for row in summary_df.itertuples():
        values = (row.constant_background, row.constant_min, row.constant_max)
        cells = ''.join(
            f'<td align="center">{_format_value(value)}</td>'
            for value in values
        )
        rows.append(f'<tr><td>{row.channel}</td>{cells}</tr>')
    table = (
        '<table border="1" cellpadding="4" cellspacing="0">'
        f'{header}{"".join(rows)}</table>'
    )
    return (
        f'Background computed for <b>{num_positions} Position(s)</b>.<br><br>'
        f'{table}<br>'
        'Use them with <code>&lt;channel&gt;_mean</code> in the "Min-max '
        'normalization"<br>'
        'measurement processing (e.g., premade FUCCI configuration of the '
        'Measurements viewer).<br><br>'
        'Min/max values are empty when no acdc_output table with '
        '<code>&lt;channel&gt;_mean</code> was selected.'
    )


class ComputeBackgroundValuesUtil(NewThreadMultipleExpBaseUtil):
    """Compute the background of the selected experiments.

    `presetParams` (optional, same keys as `BackgroundValuesParamsDialog.params`,
    any subset) prefill the parameters dialog, which is still shown, e.g.
    when launched from the setup of a premade measurements viewer
    configuration. `presetNote` is shown on top of that dialog. After the
    run, `summaryDf` is the summary table (None if cancelled or failed,
    empty if no Position was processed).

    The window closes itself only once the worker has stopped (finished,
    cancelled or failed), so `exec_` returns after the run: closing the
    progress window cancels the run at the next Position, and closing this
    window while the worker runs is refused.
    """
    def __init__(
            self, expPaths, app, title: str, infoText: str,
            progressDialogueTitle: str, parent=None, presetParams=None,
            presetNote=''
        ):
        module = myutils.get_module_name(__file__)
        super().__init__(
            expPaths, app, title, module, infoText, progressDialogueTitle,
            parent=parent
        )
        self.expPaths = expPaths
        self.presetParams = dict(presetParams or {})
        self.presetNote = presetNote
        self.summaryDf = None
        self.summaryInfo = {}
        self._workerDone = False

    def runWorker(self):
        self.worker = workers.ComputeBackgroundValuesWorker(self)
        self.worker.sigAskParams.connect(self.askParams)
        self.worker.sigSummary.connect(self.storeSummary)
        self.worker.sigAborted.connect(self.workerAborted)
        super().runWorker(self.worker)

    def showEvent(self, event):
        self.runWorker()

    def channelNames(self, expPaths):
        channelNames = {}
        for exp_path, pos_foldernames in expPaths.items():
            for pos in pos_foldernames:
                images_path = os.path.join(exp_path, pos, 'Images')
                basename, chNames = myutils.getBasenameAndChNames(images_path)
                channelNames.update(dict.fromkeys(chNames))
        return list(channelNames)

    def findExistingFile(self, outputName):
        """Existing file named `<basename><outputName>` (any extension) in
        the selected Positions, ignoring background tables of previous runs.
        """
        for exp_path, pos_foldernames in self.expPaths.items():
            for pos in pos_foldernames:
                images_path = os.path.join(exp_path, pos, 'Images')
                basename, _ = myutils.getBasenameAndChNames(images_path)
                stem = f'{basename}{outputName}'
                for file in myutils.listdir(images_path):
                    if os.path.splitext(file)[0] != stem:
                        continue
                    filepath = os.path.join(images_path, file)
                    if background_values.is_background_table(filepath):
                        continue
                    return file
        return None

    def askParams(self, expPaths):
        channelNames = self.channelNames(expPaths)
        if not channelNames:
            txt = 'No channel images found in the selected Positions.'
            self.logger.error(txt)
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            msg.critical(self, 'Error', html_utils.paragraph(txt))
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return

        win = apps.BackgroundValuesParamsDialog(
            channelNames, findExistingFile=self.findExistingFile, parent=self,
            presetParams=self.presetParams, presetNote=self.presetNote
        )
        win.exec_()
        self.worker.params = win.params
        self.worker.abort = win.cancel
        self.worker.waitCond.wakeAll()

    def storeSummary(self, summaryDf, info):
        self.summaryDf = summaryDf
        self.summaryInfo = info

    def _workerThread(self):
        # `self.thread` is the QThread set by `runWorker` (it shadows the
        # QObject.thread() method)
        thread = getattr(self, 'thread', None)
        return thread if isinstance(thread, QThread) else None

    def _isWorkerRunning(self):
        if self._workerDone:
            return False
        thread = self._workerThread()
        try:
            return thread is not None and thread.isRunning()
        except RuntimeError:
            # Already deleted (deleteLater after finishing)
            return False

    def progressWinClosed(self, aborted):
        # Cancel at the next Position, but keep this window open until the
        # worker has stopped (`workerAborted` closes it)
        self.abort = aborted
        if aborted and self.worker is not None:
            self.worker.abort = True

    def closeEvent(self, event):
        if self._isWorkerRunning():
            msg = widgets.myMessageBox(wrapText=False)
            msg.warning(self, 'Process still running', html_utils.paragraph(
                'The process is still running.<br><br>'
                'Wait for it to finish or cancel it by closing the progress '
                'window.'
            ))
            event.ignore()
            return
        super().closeEvent(event)

    def workerCritical(self, error):
        # `worker_exception_handler` emits the error twice: handle it once.
        # signals.finished is not emitted: stop the thread and close here so
        # that `exec_` returns (summaryDf stays None)
        if self._workerDone:
            return
        super().workerCritical(error)
        self._workerDone = True
        thread = self._workerThread()
        if thread is not None:
            thread.quit()
        self.worker = None
        self.progressWin = None
        self.summaryDf = None
        self.close()

    def workerAborted(self):
        # signals.finished (connected to thread.quit and worker.deleteLater)
        # is not emitted on abort: stop the thread and delete the worker here
        worker = self.worker
        thread = self._workerThread()
        if thread is not None:
            thread.quit()
        if worker is not None:
            worker.deleteLater()
        self.summaryDf = None
        self.workerFinished(None, aborted=True)

    def workerFinished(self, worker, aborted=False):
        if aborted:
            txt = f'"{self._title}" process cancelled.'
        else:
            txt = f'"{self._title}" process completed.'
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process cancelled', html_utils.paragraph(txt))
        else:
            if self.summaryDf is not None:
                self.logger.info(
                    'Background and normalization constants:\n'
                    f'{self.summaryDf.to_string()}'
                )
                num_positions = self.summaryInfo.get('num_positions', 0)
                txt = f'{txt}<br><br>{summary_html(self.summaryDf, num_positions)}'
            msg.information(
                self, 'Process completed', html_utils.paragraph(txt)
            )
        super().workerFinished(worker)
        self._workerDone = True
        self.close()
