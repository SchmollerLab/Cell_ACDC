import os

import numpy as np
import pandas as pd
from qtpy.QtCore import QThread

from .. import (
    html_utils, background_values, load
)
from cellacdc.myutils import filesystem
from cellacdc.shared.widgets import (
    line_edits as widgets_line_edits,
    message_box as widgets_message_box,
    numeric_inputs as widgets_numeric_inputs,
    push_buttons as widgets_push_buttons,
)

from .base import NewThreadMultipleExpBaseUtil
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel, QVBoxLayout, QListWidget, QCheckBox, QComboBox, QGridLayout,
    QGroupBox, QListWidgetItem
)
from cellacdc.shared.dialogs import base as dialogs_base
from collections import defaultdict
import traceback
from qtpy.QtCore import Signal
from . import base as utils_base
from cellacdc.shared.workers import base as workers_base


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
        pos_foldernames = filesystem.get_pos_foldernames(exp_path)
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
        module = filesystem.get_module_name(__file__)
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
        self.worker = ComputeBackgroundValuesWorker(self)
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
                basename, chNames = filesystem.getBasenameAndChNames(images_path)
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
                basename, _ = filesystem.getBasenameAndChNames(images_path)
                stem = f'{basename}{outputName}'.casefold()
                for file in filesystem.listdir(images_path):
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
            msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
            msg.critical(self, 'Error', html_utils.paragraph(txt))
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return

        win = BackgroundValuesParamsDialog(
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
            msg = widgets_message_box.myMessageBox(wrapText=False)
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
        msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
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
        gridLayout.addWidget(widgets_push_buttons.infoPushButton(
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
        self.outputNameLineEdit = widgets_line_edits.alphaNumericLineEdit()
        self.outputNameLineEdit.setText('background_values')
        gridLayout.addWidget(self.outputNameLineEdit, row, 1)
        gridLayout.addWidget(widgets_push_buttons.infoPushButton(
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
        self.lowPercentileSpinbox = widgets_numeric_inputs.DoubleSpinBox()
        self.lowPercentileSpinbox.setMinimum(0)
        self.lowPercentileSpinbox.setMaximum(100)
        self.lowPercentileSpinbox.setValue(5)
        groupLayout.addWidget(self.lowPercentileSpinbox, 1, 1)
        groupLayout.addWidget(QLabel('Percentile mapped to 1:'), 2, 0)
        self.highPercentileSpinbox = widgets_numeric_inputs.DoubleSpinBox()
        self.highPercentileSpinbox.setMinimum(0)
        self.highPercentileSpinbox.setMaximum(100)
        self.highPercentileSpinbox.setValue(95)
        groupLayout.addWidget(self.highPercentileSpinbox, 2, 1)
        self.normalizationGroupbox.setLayout(groupLayout)
        mainLayout.addWidget(self.normalizationGroupbox)

        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()
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
        msg = widgets_message_box.myMessageBox(wrapText=False)
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


class ComputeBackgroundValuesWorker(utils_base.BaseWorkerUtil):
    """Worker of `utils.computeBackgroundValues.ComputeBackgroundValuesUtil`.

    See `cellacdc.background_values` for the method. The GUI answers
    `sigAskParams` by setting `self.params` (dict with the keys `channels`,
    `statistic`, `exclude_zero_pixels`, `output_name`, `use_acdc_output`,
    `low_percentile`, `high_percentile`) or `self.abort`. With
    `use_acdc_output`, the mean intensities of the cells are read from a
    selected acdc_output table (not modified) for the normalization
    constants.
    """
    sigAskParams = Signal(object)
    sigAborted = Signal()
    sigSummary = Signal(object, object)

    def __init__(self, mainWin):
        super().__init__(mainWin)
        self.params = None

    def emitAskParams(self, expPaths):
        self.mutex.lock()
        self.sigAskParams.emit(expPaths)
        self.waitCond.wait(self.mutex)
        self.mutex.unlock()
        return self.abort

    def _saveCsv(self, df, csv_path, index=False):
        while True:
            try:
                df.to_csv(csv_path, index=index)
                return
            except PermissionError:
                traceback_str = traceback.format_exc()
                self.mutex.lock()
                self.signals.sigPermissionError.emit(traceback_str, csv_path)
                self.waitCond.wait(self.mutex)
                self.mutex.unlock()

    def _selectFiles(self, expPaths, use_acdc_output):
        """Ask the segmentation file (and acdc_output table) per experiment.

        Returns None when the user cancelled, otherwise
        {exp_path: (segm_endname, acdc_output_endname or None)}. Experiments
        without segmentation files are skipped.
        """
        selections = {}
        for exp_path, pos_foldernames in expPaths.items():
            has_segm = has_acdc_output = False
            for pos in pos_foldernames:
                images_path = os.path.join(exp_path, pos, 'Images')
                has_segm = has_segm or bool(load.get_segm_files(images_path))
                has_acdc_output = has_acdc_output or bool(
                    load.get_acdc_output_files(images_path)
                )
            if not has_segm:
                self.logger.log(
                    f'[WARNING]: No segmentation file found in "{exp_path}". '
                    'Experiment skipped.'
                )
                continue

            self.mainWin.infoText = (
                'Select the segmentation file that defines the '
                '<b>background</b> (pixels outside of the segmented objects).'
                '<br>Ideally, <b>all cells are segmented</b> in this file.'
            )
            abort = self.emitSelectSegmFiles(exp_path, pos_foldernames)
            if abort:
                return
            segm_endname = self.mainWin.endFilenameSegm

            acdc_output_endname = None
            if use_acdc_output and not has_acdc_output:
                self.logger.log(
                    f'[WARNING]: No acdc_output table found in "{exp_path}". '
                    'Only the background values will be computed.'
                )
            elif use_acdc_output:
                abort = self.emitSelectAcdcOutputFiles(
                    exp_path, pos_foldernames,
                    infoText=(
                        ' with the mean intensities for the normalization '
                        'constants'
                    ),
                    allowSingleSelection=True, multiSelection=False
                )
                if abort or not self.mainWin.selectedAcdcOutputEndnames:
                    return
                acdc_output_endname = (
                    self.mainWin.selectedAcdcOutputEndnames[0]
                )
            selections[exp_path] = (segm_endname, acdc_output_endname)
        return selections

    def _loadPositionData(self, images_path, segm_endname, channels):
        """Load the segmentation and channel images of one Position.

        Returns (images, segm, basename) with data shaped (T, [Z,] Y, X).
        Raises FileNotFoundError with a message for the log when the Position
        cannot be processed.
        """
        basename, _ = filesystem.getBasenameAndChNames(images_path)
        segm_filenames = {
            f'{basename}{segm_endname}.npz', f'{basename}{segm_endname}.npy'
        }
        segm_files = [
            file for file in load.get_segm_files(images_path)
            if file in segm_filenames
        ]
        if not segm_files:
            raise FileNotFoundError(
                f'Segmentation file "{basename}{segm_endname}.npz" not found.'
            )
        segm_filepath = os.path.join(images_path, segm_files[0])

        channel_filepaths = {}
        for channel in channels:
            # Match '<basename><channel>' so that e.g. 'GFP' is not
            # loaded from 'mGFP'
            filepath = load.get_filename_from_channel(
                images_path, channel, basename=basename
            )
            if filepath:
                channel_filepaths[channel] = filepath
        if not channel_filepaths:
            raise FileNotFoundError(
                f'None of the channels {list(channels)} found.'
            )

        first_channel, first_filepath = next(iter(channel_filepaths.items()))
        posData = load.loadData(first_filepath, first_channel)
        posData.getBasenameAndChNames()
        posData.buildPaths()
        posData.loadOtherFiles(load_segm_data=False, load_metadata=True)
        size_t = int(posData.SizeT)

        if segm_filepath.endswith('.npy'):
            segm = np.load(segm_filepath)
        else:
            with np.load(segm_filepath) as archive:
                segm = archive[archive.files[0]]
        segm = background_values.as_frames(segm, size_t)

        images = {}
        for channel, filepath in channel_filepaths.items():
            img = load.load_image_file(filepath)
            images[channel] = background_values.as_frames(img, size_t)
        return images, segm, basename

    def _processPosition(
            self, images_path, segm_endname, acdc_endname, params,
            raw_means, backgrounds
        ):
        """Compute, save and pool the background of one Position.

        Returns True if the Position was processed.
        """
        statistic = params['statistic']
        try:
            images, segm, basename = self._loadPositionData(
                images_path, segm_endname, params['channels']
            )
        except FileNotFoundError as err:
            self.logger.log(
                f'[WARNING]: {err} Position "{images_path}" skipped.'
            )
            return False

        missing = [ch for ch in params['channels'] if ch not in images]
        if missing:
            self.logger.log(
                f'[WARNING]: Channels {missing} not found in "{images_path}".'
            )
        num_frames = min([len(segm), *[len(img) for img in images.values()]])
        self.signals.sigInitInnerPbar.emit(num_frames)
        try:
            bkgr_df = background_values.compute_background_values(
                images, segm,
                exclude_zero_pixels=params['exclude_zero_pixels'],
                progress_callback=(
                    lambda: self.signals.sigUpdateInnerPbar.emit(1)
                )
            )
        except ValueError as err:
            self.logger.log(
                f'[WARNING]: {err} Position "{images_path}" skipped.'
            )
            return False

        num_empty = int((bkgr_df['num_objects'] == 0).sum())
        if num_empty:
            self.logger.log(
                f'[WARNING]: {num_empty} frame(s) without segmented objects: '
                'their background is left empty (NaN).'
            )
        bkgr_csv_path = os.path.join(
            images_path, f'{basename}{params["output_name"]}.csv'
        )
        self._saveCsv(bkgr_df, bkgr_csv_path)
        self.logger.log(f'Background values saved to "{bkgr_csv_path}"')
        for channel in images:
            col = background_values.background_colname(channel, statistic)
            backgrounds[channel].append(bkgr_df[col].to_numpy())

        if acdc_endname is not None:
            acdc_csv_path = os.path.join(
                images_path, f'{basename}{acdc_endname}.csv'
            )
            self._collectCellMeans(
                acdc_csv_path, list(images), segm.ndim == 4, raw_means
            )
        return True

    @workers_base.worker_exception_handler
    def run(self):
        expPaths = self.mainWin.expPaths
        self.signals.initProgressBar.emit(0)

        abort = self.emitAskParams(expPaths)
        if abort or self.params is None:
            self.sigAborted.emit()
            return
        params = self.params

        selections = self._selectFiles(
            expPaths, params['use_acdc_output']
        )
        if selections is None:
            self.sigAborted.emit()
            return

        raw_means = defaultdict(list)
        backgrounds = defaultdict(list)
        num_positions = 0
        tot_pos = sum(len(expPaths[exp_path]) for exp_path in selections)
        self.signals.initProgressBar.emit(tot_pos)
        for i, (exp_path, (segm_endname, acdc_endname)) in enumerate(
                selections.items()
            ):
            pos_foldernames = expPaths[exp_path]
            for p, pos in enumerate(pos_foldernames):
                if self.abort:
                    self.sigAborted.emit()
                    return

                self.logger.log(
                    f'Processing experiment n. {i+1}/{len(selections)}, '
                    f'{pos} ({p+1}/{len(pos_foldernames)})'
                )
                images_path = os.path.join(exp_path, pos, 'Images')
                self.signals.sigUpdatePbarDesc.emit(
                    f'Processing {os.path.join(exp_path, pos)}'
                )
                processed = self._processPosition(
                    images_path, segm_endname, acdc_endname, params,
                    raw_means, backgrounds
                )
                num_positions += int(processed)
                self.signals.progressBar.emit(1)

        if self.abort:
            # Cancelled during the last Position: no summary
            self.sigAborted.emit()
            return

        if num_positions == 0:
            self.sigSummary.emit(pd.DataFrame(), {'num_positions': 0})
            self.signals.finished.emit(self)
            return

        summary_df = background_values.normalization_constants(
            raw_means, backgrounds,
            low_percentile=params['low_percentile'],
            high_percentile=params['high_percentile']
        )
        summary_df.insert(1, 'background_statistic', params['statistic'])
        summary_df['exclude_zero_pixels'] = params['exclude_zero_pixels']
        summary_df['num_positions'] = num_positions
        summary_df['experiment_folders'] = ';'.join(selections)
        for exp_path in selections:
            summary_csv_path = os.path.join(
                exp_path, f'{params["output_name"]}_summary.csv'
            )
            self._saveCsv(summary_df, summary_csv_path)
            self.logger.log(f'Summary saved to "{summary_csv_path}"')
        self.sigSummary.emit(summary_df, {'num_positions': num_positions})
        self.signals.finished.emit(self)

    def _collectCellMeans(
            self, acdc_csv_path, channels, is_segm_3D, raw_means
        ):
        """Pool the mean intensities of the included cells of `channels`
        from an acdc_output table (read only) into `raw_means`."""
        if not os.path.exists(acdc_csv_path):
            self.logger.log(
                f'[WARNING]: Table "{acdc_csv_path}" not found. '
                'Its cells are not used for the normalization constants.'
            )
            return
        acdc_df = load.read_acdc_df_csv(acdc_csv_path)
        missing = []
        for channel in channels:
            source_col, values = background_values.cell_mean_values(
                acdc_df, channel, is_segm_3D=is_segm_3D
            )
            if source_col is None:
                missing.append(channel)
                continue
            raw_means[channel].append(values)
        if missing:
            expected = [
                ' or '.join(
                    background_values.mean_source_columns(ch, is_segm_3D)
                )
                for ch in missing
            ]
            self.logger.log(
                f'[WARNING]: Columns {expected} not found in '
                f'"{acdc_csv_path}". Compute the mean intensity of these '
                'channels first (e.g., with the "Compute measurements" '
                'utility).'
            )

    # Disabled for now (not used): added per-frame background-corrected
    # columns to the acdc_output table. See `background_values`.
    # def _correctAcdcOutput(
    #         self, acdc_csv_path, bkgr_df, channels, statistic, is_segm_3D,
    #         raw_means, corrected_means
    #     ):
    #     if not os.path.exists(acdc_csv_path):
    #         self.logger.log(
    #             f'[WARNING]: Table "{acdc_csv_path}" not found. '
    #             'Background correction skipped.'
    #         )
    #         return
    #     acdc_df = load.read_acdc_df_csv(acdc_csv_path)
    #     acdc_df, corrected, skipped = (
    #         background_values.add_background_corrected_columns(
    #             acdc_df, bkgr_df, channels, statistic, is_segm_3D=is_segm_3D
    #         )
    #     )
    #     if skipped:
    #         expected = [
    #             ' or '.join(
    #                 background_values.mean_source_columns(ch, is_segm_3D)
    #             )
    #             for ch in skipped
    #         ]
    #         self.logger.log(
    #             f'[WARNING]: Columns {expected} not found in '
    #             f'"{acdc_csv_path}". Compute the mean intensity of these '
    #             'channels first (e.g., with the "Compute measurements" '
    #             'utility).'
    #         )
    #     if not corrected:
    #         return
    #     self._saveCsv(acdc_df, acdc_csv_path)
    #     included = background_values.included_cells_mask(acdc_df)
    #     for channel, source_col in corrected.items():
    #         corrected_col = background_values.corrected_colname(source_col)
    #         raw = pd.to_numeric(acdc_df[source_col], errors='coerce')
    #         num_nan = int((raw.notna() & acdc_df[corrected_col].isna()).sum())
    #         self.logger.log(
    #             f'Added column "{corrected_col}" to "{acdc_csv_path}"'
    #         )
    #         if num_nan:
    #             self.logger.log(
    #                 f'[WARNING]: "{corrected_col}" is empty for {num_nan} '
    #                 'row(s) of frames without a background value. Cell-ACDC '
    #                 'reads these as 0 when the table has cell cycle '
    #                 'annotations.'
    #             )
    #         raw_means[channel].append(raw[included].to_numpy())
    #         corrected_means[channel].append(
    #             acdc_df.loc[included, corrected_col].to_numpy()
    #         )
