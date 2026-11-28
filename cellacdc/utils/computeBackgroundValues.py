import os

import numpy as np
import pandas as pd
from qtpy.QtCore import QThread

from .. import (
    apps, myutils, workers, widgets, html_utils, background_values, load
)

from .base import NewThreadMultipleExpBaseUtil
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel, QVBoxLayout, QListWidget, QCheckBox, QComboBox, QGridLayout,
    QGroupBox, QListWidgetItem
)
from cellacdc.shared.dialogs import base as dialogs_base


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

        Names are compared case-insensitively: on the case-insensitive file
        systems of Windows and macOS, writing `<basename>ACDC_output.csv`
        would replace `<basename>acdc_output.csv`.
        """
        for exp_path, pos_foldernames in self.expPaths.items():
            for pos in pos_foldernames:
                images_path = os.path.join(exp_path, pos, 'Images')
                basename, _ = myutils.getBasenameAndChNames(images_path)
                stem = f'{basename}{outputName}'.casefold()
                for file in myutils.listdir(images_path):
                    if os.path.splitext(file)[0].casefold() != stem:
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
        # `worker_exception_handler` emits both `critical` signals of a
        # worker that has two (`critical` and `signals.critical`): handle
        # the error once.
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


class BackgroundValuesParamsDialog(dialogs_base.QBaseDialog):
    """Parameters of the "Compute background intensity" utility.

    After `exec_`, `cancel` is False when the user pressed Ok and `params`
    holds the keys documented in `workers.ComputeBackgroundValuesWorker`.

    `findExistingFile(name)` returns the name of an existing file that the
    background table `<basename><name>.csv` would collide with, or None.

    `presetParams` (any subset of the `params` keys) prefill the widgets:
    with `channels`, only these channels are checked. `presetNote` is shown
    on top (e.g. why the values are preset).
    """
    def __init__(
            self, channelNames, findExistingFile=None, parent=None,
            presetParams=None, presetNote=''
        ):
        super().__init__(parent=parent)
        self.setWindowTitle('Compute background intensity')
        self.cancel = True
        self.params = None
        self.findExistingFile = findExistingFile

        mainLayout = QVBoxLayout()
        if presetNote:
            noteLabel = QLabel(html_utils.paragraph(presetNote))
            noteLabel.setWordWrap(True)
            mainLayout.addWidget(noteLabel)
        mainLayout.addWidget(QLabel(html_utils.paragraph(
            'For every frame, the background of each selected channel is the '
            '<b>intensity of the pixels outside of the segmented '
            'objects</b>.<br>'
            'You will select the segmentation file next: use one where '
            '<b>(nearly) all cells are segmented</b>.'
        )))

        mainLayout.addWidget(QLabel(html_utils.paragraph(
            '<b>Channels</b>:'
        )))
        self.channelsListWidget = QListWidget()
        for channel in channelNames:
            item = QListWidgetItem(channel)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.channelsListWidget.addItem(item)
        self.channelsListWidget.setMaximumHeight(
            max(60, min(160, 24*len(channelNames) + 8))
        )
        mainLayout.addWidget(self.channelsListWidget)

        gridLayout = QGridLayout()
        row = 0
        gridLayout.addWidget(
            QLabel('Background statistic subtracted from cells:'), row, 0
        )
        self.statisticCombobox = QComboBox()
        self.statisticCombobox.addItems(['mean', 'median'])
        gridLayout.addWidget(self.statisticCombobox, row, 1)
        gridLayout.addWidget(widgets.infoPushButton(
            info_title='Background statistic',
            info_text=html_utils.paragraph(
                'Both the mean and the median of the background pixels are '
                'saved for every frame. This statistic is the one subtracted '
                'from the mean intensity of the cells and used for the '
                'normalization constants.'
            )
        ), row, 2)

        row += 1
        self.excludeZeroCheckbox = QCheckBox(
            'Ignore pixels with intensity 0 (e.g., padding from alignment)'
        )
        self.excludeZeroCheckbox.setChecked(True)
        gridLayout.addWidget(self.excludeZeroCheckbox, row, 0, 1, 3)

        row += 1
        gridLayout.addWidget(QLabel('Name of the background table:'), row, 0)
        self.outputNameLineEdit = widgets.alphaNumericLineEdit()
        self.outputNameLineEdit.setText('background_values')
        gridLayout.addWidget(self.outputNameLineEdit, row, 1)
        gridLayout.addWidget(widgets.infoPushButton(
            info_title='Background table',
            info_text=html_utils.paragraph(
                'The per-frame background values of each Position are saved '
                'in the Images folder as '
                '<code>&lt;basename&gt;&lt;name&gt;.csv</code>, the pooled '
                'summary in each experiment folder as '
                '<code>&lt;name&gt;_summary.csv</code>. Existing files are '
                'overwritten.'
            )
        ), row, 2)
        gridLayout.setColumnStretch(1, 1)
        mainLayout.addLayout(gridLayout)

        self.normalizationGroupbox = QGroupBox(
            'Compute normalization constants from an acdc_output table'
        )
        self.normalizationGroupbox.setCheckable(True)
        self.normalizationGroupbox.setChecked(True)
        groupLayout = QGridLayout()
        groupLayout.addWidget(QLabel(html_utils.paragraph(
            'Reads <code>&lt;channel&gt;_mean</code> of the cells from the '
            'selected acdc_output table (the table is not modified).<br>'
            'The percentiles below of these values minus the constant '
            'background, pooled over all cells,<br>'
            'frames and Positions, are the min/max values of the "Min-max '
            'normalization" measurement processing.'
        )), 0, 0, 1, 2)
        groupLayout.addWidget(QLabel('Percentile mapped to 0:'), 1, 0)
        self.lowPercentileSpinbox = widgets.DoubleSpinBox()
        self.lowPercentileSpinbox.setMinimum(0)
        self.lowPercentileSpinbox.setMaximum(100)
        self.lowPercentileSpinbox.setValue(5)
        groupLayout.addWidget(self.lowPercentileSpinbox, 1, 1)
        groupLayout.addWidget(QLabel('Percentile mapped to 1:'), 2, 0)
        self.highPercentileSpinbox = widgets.DoubleSpinBox()
        self.highPercentileSpinbox.setMinimum(0)
        self.highPercentileSpinbox.setMaximum(100)
        self.highPercentileSpinbox.setValue(95)
        groupLayout.addWidget(self.highPercentileSpinbox, 2, 1)
        self.normalizationGroupbox.setLayout(groupLayout)
        mainLayout.addWidget(self.normalizationGroupbox)

        buttonsLayout = widgets.CancelOkButtonsLayout()
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        mainLayout.addSpacing(10)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)
        if presetParams:
            self.setParams(presetParams)

    def setParams(self, params):
        """Prefill the widgets with `params` (any subset of the keys of
        `self.params`)."""
        if 'channels' in params:
            selected = set(params['channels'])
            for i in range(self.channelsListWidget.count()):
                item = self.channelsListWidget.item(i)
                checked = item.text() in selected
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        if params.get('statistic') in ('mean', 'median'):
            self.statisticCombobox.setCurrentText(params['statistic'])
        if 'exclude_zero_pixels' in params:
            self.excludeZeroCheckbox.setChecked(
                bool(params['exclude_zero_pixels'])
            )
        if params.get('output_name'):
            self.outputNameLineEdit.setText(params['output_name'])
        if 'use_acdc_output' in params:
            self.normalizationGroupbox.setChecked(
                bool(params['use_acdc_output'])
            )
        if 'low_percentile' in params:
            self.lowPercentileSpinbox.setValue(params['low_percentile'])
        if 'high_percentile' in params:
            self.highPercentileSpinbox.setValue(params['high_percentile'])

    def selectedChannels(self):
        channels = []
        for i in range(self.channelsListWidget.count()):
            item = self.channelsListWidget.item(i)
            if item.checkState() == Qt.Checked:
                channels.append(item.text())
        return channels

    def _warn(self, text):
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Invalid parameters', html_utils.paragraph(text))

    def ok_cb(self):
        channels = self.selectedChannels()
        if not channels:
            self._warn('Select at least one channel.')
            return
        outputName = self.outputNameLineEdit.text().strip()
        if not outputName:
            self._warn('The name of the background table cannot be empty.')
            return
        existingFile = None
        if self.findExistingFile is not None:
            existingFile = self.findExistingFile(outputName)
        if existingFile is not None:
            self._warn(
                f'The name "{outputName}" is already used by the file '
                f'<code>{existingFile}</code>.<br><br>'
                'Please choose a different name for the background table.'
            )
            return
        low = self.lowPercentileSpinbox.value()
        high = self.highPercentileSpinbox.value()
        useAcdcOutput = self.normalizationGroupbox.isChecked()
        if low >= high:
            self._warn(
                'The percentile mapped to 0 must be lower than the percentile '
                'mapped to 1.'
            )
            return
        self.params = {
            'channels': channels,
            'statistic': self.statisticCombobox.currentText(),
            'exclude_zero_pixels': self.excludeZeroCheckbox.isChecked(),
            'output_name': outputName,
            'use_acdc_output': useAcdcOutput,
            'low_percentile': low,
            'high_percentile': high,
        }
        self.cancel = False
        self.close()
