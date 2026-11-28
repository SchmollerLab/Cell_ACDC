"""Main-GUI-only loading, saving, restore and export dialogs (from
apps.py).

QDialogZsliceAbsent, the save progress bar QDialogPbar, the acdc_output
version restore SelectAcdcDfVersionToRestore, AutoSaveIntervalDialog and
the export to video/image parameters (ExportToVideoParametersDialog,
ExportToImageParametersDialog), plus the widgets only they use.
"""
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QHBoxLayout, QWidget
from cellacdc.shared.widgets import numeric_inputs as widgets_numeric_inputs
from cellacdc.shared.widgets import combo_boxes as widgets_combo_boxes
import os
import datetime
import zipfile
from natsort import natsorted
from functools import partial
from qtpy.QtCore import Qt, QSize, QEventLoop
from qtpy.QtWidgets import (
    QLabel, QVBoxLayout, QPushButton, QDialog, QComboBox, QGridLayout
)
from cellacdc import _window_stacking
from cellacdc import load, html_utils
from cellacdc import is_mac
from cellacdc.myutils import filesystem
from cellacdc.shared.widgets import (
    basic_widgets as widgets_basic_widgets,
    item_views as widgets_item_views,
    line_edits as widgets_line_edits,
    message_box as widgets_message_box,
    push_buttons as widgets_push_buttons,
)
from cellacdc import fonts
from cellacdc.shared.dialogs import base as dialogs_base


class AutoSaveIntervalWidget(QWidget):
    sigValueChanged = Signal(float, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout()
        
        autoSaveIntervalTooltip = (
            'Autosave every minutes or frames specified here.'
        )
        
        self.setToolTip(autoSaveIntervalTooltip)
        
        self.spinbox = widgets_numeric_inputs.DoubleSpinBox()
        self.spinbox.setMinimum(0)
        self.spinbox.setValue(2)
        self.spinbox.setDecimals(2)
        self.spinbox.setSingleStep(1.0)
        
        layout.addWidget(self.spinbox)
        
        self.unitCombobox = widgets_combo_boxes.ComboBox()
        self.unitCombobox.addItems(['minutes', 'frames'])
        layout.addWidget(self.unitCombobox)
        
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)
        layout.setContentsMargins(5, 0, 5, 0)
    
        self.setLayout(layout)
        
        self.spinbox.sigValueChanged.connect(self.emitSigValueChanged)
        self.unitCombobox.sigTextChanged.connect(self.emitSigValueChanged)
    
    def emitSigValueChanged(self, *args, **kwargs):
        self.sigValueChanged.emit(
            self.spinbox.value(), 
            self.unitCombobox.currentText()
        )


class QDialogZsliceAbsent(QDialog):
    def __init__(self, filename, SizeZ, filenamesWithInfo, parent=None):
        self.runDataPrep = False
        self.useMiddleSlice = False
        self.useSameAsCh = False

        self.cancel = True

        super().__init__(parent)
        self.setWindowTitle('Reference z-slice info absent')

        mainLayout = QVBoxLayout()
        buttonsLayout = QGridLayout()

        txt = html_utils.paragraph(f"""
            You loaded the fluorescent file called<br><br>
            <code>{filename}</code><br><br>
            however you <b>never selected which z-slice</b><br> you want to use
            when calculating metrics<br> (e.g., mean, median, amount...etc.)<br><br>
            Choose one of following options:
        """, center=True
        )
        infoLabel = QLabel(txt)
        mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        runDataPrepButton = QPushButton(
            '   Visualize the data now and select a z-slice    '
        )
        buttonsLayout.addWidget(runDataPrepButton, 0, 1, 1, 2)
        runDataPrepButton.clicked.connect(self.runDataPrep_cb)

        useMiddleSliceButton = QPushButton(
            f'  Use the middle z-slice ({int(SizeZ/2)+1})  '
        )
        buttonsLayout.addWidget(useMiddleSliceButton, 1, 1, 1, 2)
        useMiddleSliceButton.clicked.connect(self.useMiddleSlice_cb)

        useSameAsChButton = QPushButton(
            '  Use the same z-slice used for the channel: '
        )
        useSameAsChButton.clicked.connect(self.useSameAsCh_cb)

        chNameComboBox = QComboBox()
        chNameComboBox.addItems(filenamesWithInfo)
        # chNameComboBox.setEditable(True)
        # chNameComboBox.lineEdit().setAlignment(Qt.AlignCenter)
        # chNameComboBox.lineEdit().setReadOnly(True)
        self.chNameComboBox = chNameComboBox
        buttonsLayout.addWidget(useSameAsChButton, 2, 1)
        buttonsLayout.addWidget(chNameComboBox, 2, 2)

        

        buttonsLayout.setColumnStretch(0, 1)
        buttonsLayout.setColumnStretch(3, 1)
        buttonsLayout.setContentsMargins(10, 0, 10, 0)

        

        cancelButtonLayout = QHBoxLayout()
        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')
        cancelButtonLayout.addStretch(1)
        cancelButtonLayout.addWidget(cancelButton)
        cancelButtonLayout.addStretch(1)
        cancelButtonLayout.setStretch(1,1)
        cancelButton.clicked.connect(self.close)

        mainLayout.addLayout(buttonsLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(cancelButtonLayout)
        mainLayout.addStretch(1)

        self.setLayout(mainLayout)

        self.setFont(fonts.font)

        # self.setModal(True)
    
    def ok_cb(self, checked=True):
        self.cancel = False
        self.close()

    def useSameAsCh_cb(self, checked):
        self.useSameAsCh = True
        self.selectedChannel = self.chNameComboBox.currentText()
        self.ok_cb()

    def useMiddleSlice_cb(self, checked):
        self.useMiddleSlice = True
        self.ok_cb()

    def runDataPrep_cb(self, checked):
        self.runDataPrep = True
        self.ok_cb()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()


class QDialogPbar(QDialog):
    def __init__(self, title='Progress', infoTxt='', parent=None):
        self.workerFinished = False
        self.aborted = False
        self.clickCount = 0
        super().__init__(parent)

        abort_text = 'Option+Command+C' if is_mac else 'Ctrl+Alt+C'
        self.abort_text = abort_text

        self.setWindowTitle(f'{title} ({abort_text} to cancel process)')
        self.setWindowFlags(Qt.Window)

        mainLayout = QVBoxLayout()
        pBarLayout = QGridLayout()

        if infoTxt:
            infoLabel = QLabel(infoTxt)
            mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        self.progressLabel = QLabel()

        self.QPbar = widgets_basic_widgets.ProgressBar(self)
        pBarLayout.addWidget(self.QPbar, 0, 0)
        self.ETA_label = QLabel('NDh:NDm:NDs')
        pBarLayout.addWidget(self.ETA_label, 0, 1)

        self.metricsQPbar = widgets_basic_widgets.ProgressBar(self)
        self.metricsQPbar.setValue(0)
        pBarLayout.addWidget(self.metricsQPbar, 1, 0)

        #pBarLayout.setColumnStretch(2, 1)

        mainLayout.addWidget(self.progressLabel)
        mainLayout.addLayout(pBarLayout)

        self.setLayout(mainLayout)
        # self.setModal(True)

    def keyPressEvent(self, event):
        isCtrlAlt = event.modifiers() == (Qt.ControlModifier | Qt.AltModifier)
        if isCtrlAlt and event.key() == Qt.Key_C:
            doAbort = self.askAbort()
            if doAbort:
                self.aborted = True
                self.workerFinished = True
                self.close()

    def askAbort(self):
        msg = widgets_message_box.myMessageBox()
        txt = html_utils.paragraph(f"""
            Aborting with <code>{self.abort_text}</code> to abort 
            is <b>not safe</b>.<br><br>
            The system status cannot be predicted and
            it will <b>require a restart</b>.<br><br>
            Are you sure you want to abort?
        """)
        yesButton, noButton = msg.critical(
            self, 'Are you sure you want to abort?', txt,
            buttonsTexts=('Yes', 'No')
        )
        return msg.clickedButton == yesButton


    def abort(self):
        self.clickCount += 1
        self.aborted = True
        if self.clickCount > 3:
            self.workerFinished = True
            self.close()

    def closeEvent(self, event):
        if not self.workerFinished:
            event.ignore()


class SelectAcdcDfVersionToRestore(dialogs_base.QBaseDialog):
    def __init__(self, posData, parent=None):
        super().__init__(parent=parent)
        
        self.cancel = True
        
        self.setWindowTitle('Select annotations table to restore')
        
        mainLayout = QVBoxLayout()
        
        acdc_df_filename = os.path.basename(posData.acdc_output_csv_path)
        instructionsLabel = html_utils.paragraph(
            f'Select an <b>older version</b> of the <code>{acdc_df_filename}</code> '
            'annotations table to load.<br><br>'
            'The datetime refers to the time you replaced the old version with '
            'a newer one.<br><br>'
        )
        mainLayout.addWidget(QLabel(instructionsLabel))
        
        self.savedListBox = None
        if os.path.exists(posData.acdc_output_backup_zip_path):
            zip_path = posData.acdc_output_backup_zip_path
            self.savedArchivefilepath = zip_path
            with zipfile.ZipFile(zip_path, mode='r') as zip:
                csv_names = natsorted(zip.namelist(), reverse=True)
            
            keys = [csv_name[:-4] for csv_name in csv_names]

            self.savedKeys = keys
            f = load.ISO_TIMESTAMP_FORMAT
            timestamps = [datetime.datetime.strptime(key, f) for key in keys]
            items = [date.strftime(r'%d %b %Y, %H:%M:%S') for date in timestamps]
            mainLayout.addWidget(QLabel('Saved annotations:'))
            self.savedListBox = widgets_item_views.listWidget()
            self.savedListBox.addItems(items)
            mainLayout.addWidget(self.savedListBox)
            self.savedListBox.itemSelectionChanged.connect(
                self.onItemSelectionChanged
            )
        
        recovery_folderpath = posData.recoveryFolderpath()
        unsaved_recovery_folderpath = os.path.join(
            recovery_folderpath, 'never_saved'
        )
        self.neverSavedFolderpath = unsaved_recovery_folderpath
        files = filesystem.listdir(unsaved_recovery_folderpath)
        csv_files = [file for file in files if file.endswith('.csv')]
        self.neverSavedListBox = None
        if csv_files:
            csv_names = natsorted(csv_files, reverse=True)
            keys = [csv_name[:-4] for csv_name in csv_names]
            self.neverSavedKeys = keys
            f = load.ISO_TIMESTAMP_FORMAT
            timestamps = [
                datetime.datetime.strptime(key, f) for key in keys
            ]
            items = [date.strftime(r'%d %b %Y, %H:%M:%S') for date in timestamps]
            mainLayout.addWidget(QLabel('Never saved annotations:'))
            self.neverSavedListBox = widgets_item_views.listWidget()
            self.neverSavedListBox.addItems(items)
            mainLayout.addWidget(self.neverSavedListBox)
            self.neverSavedListBox.itemSelectionChanged.connect(
                self.onItemSelectionChanged
            )
                
        cancelOkLayout = widgets_push_buttons.CancelOkButtonsLayout()
        
        cancelOkLayout.okButton.clicked.connect(self.ok_cb)
        cancelOkLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addSpacing(20)
        mainLayout.addLayout(cancelOkLayout)
        
        self.setLayout(mainLayout)
        
        self.setFont(fonts.font)
    
    def ok_cb(self):
        self.cancel = False
        try:
            for i in range(self.savedListBox.count()):
                item = self.savedListBox.item(i)
                if item.isSelected():
                    self.selectedTimestamp = item.text()
                    self.selectedKey = self.savedKeys[i]
                    self.archiveFilePath = self.savedArchivefilepath
                    break
        except Exception as e:
            pass
        
        try:
            for i in range(self.neverSavedListBox.count()):
                item = self.neverSavedListBox.item(i)
                if item.isSelected():
                    self.selectedTimestamp = item.text()
                    self.selectedKey = self.neverSavedKeys[i]
                    self.archiveFilePath = self.neverSavedFolderpath
                    break
        except Exception as e:
            pass
        self.close()
    
    def onItemSelectionChanged(self):
        otherListBox = (
            self.savedListBox if self.sender() == self.neverSavedListBox
            else self.neverSavedListBox
        )
        if otherListBox is None:
            return
        for i in range(otherListBox.count()):
            item = otherListBox.item(i)
            item.setSelected(False)


class ExportToVideoParametersDialog(dialogs_base.QBaseDialog):
    sigOk = Signal(dict)
    sigAddScaleBar = Signal(bool)
    sigAddTimestamp = Signal(bool)
    sigRescaleIntensLut = Signal(str, str)
    sigChangeStartTime = Signal(str)
    
    def __init__(
            self, channels, parent=None, startFolderpath='', startFilename='', 
            startFrameNum=1, SizeT=1, SizeZ=1, isTimelapseVideo=True, 
            isScaleBarPresent=False, isTimestampPresent=False, 
            rescaleIntensChannelHowMapper=None, 
            startTime=None
        ):
        self.cancel = True
        
        if rescaleIntensChannelHowMapper is None:
            rescaleIntensChannelHowMapper = {}
        
        super().__init__(parent=parent)
        
        self.setWindowTitle('Preferences for output video')
        
        mainLayout = QVBoxLayout()
        
        gridLayout = QGridLayout()
        
        navVar = 'frame number' if isTimelapseVideo else 'z-slice'
        maxNavVar = SizeT if isTimelapseVideo else SizeZ
        
        self.isTimelapseVideo = isTimelapseVideo
        
        row = 0
        gridLayout.addWidget(QLabel(f'Start {navVar}:'), row, 0)
        self.startNavVarNumberEntry = widgets_numeric_inputs.SpinBox()
        self.startNavVarNumberEntry.setMinimum(1)
        self.startNavVarNumberEntry.setMaximum(maxNavVar-1)
        self.startNavVarNumberEntry.setValue(startFrameNum)
        gridLayout.addWidget(self.startNavVarNumberEntry, row, 1)
        
        row += 1
        gridLayout.addWidget(QLabel(f'Stop {navVar}:'), row, 0)
        self.stopNavVarNumberEntry = widgets_numeric_inputs.SpinBox()
        self.stopNavVarNumberEntry.setMinimum(2)
        self.stopNavVarNumberEntry.setMaximum(maxNavVar)
        self.stopNavVarNumberEntry.setValue(maxNavVar)
        gridLayout.addWidget(self.stopNavVarNumberEntry, row, 1)
        
        row += 1
        gridLayout.addWidget(QLabel('File format:'), row, 0)
        self.fileFormatCombobox = QComboBox()
        self.fileFormatCombobox.addItems(['MP4', 'AVI'])
        gridLayout.addWidget(self.fileFormatCombobox, row, 1)
        
        row += 1
        gridLayout.addWidget(QLabel('Frame rate (FPS):'), row, 0)
        self.fpsWidget = widgets_line_edits.FloatLineEdit(allowNegative=False)
        self.fpsWidget.setValue(10.0)
        gridLayout.addWidget(self.fpsWidget, row, 1)
        
        row += 1
        self.dpiWidget = widgets_line_edits.IntLineEdit(allowNegative=False)
        self.dpiWidget.setValue(300)
        self.dpiWidget.label = QLabel('DPI')
        gridLayout.addWidget(self.dpiWidget.label, row, 0)
        gridLayout.addWidget(self.dpiWidget, row, 1)
        
        row += 1
        gridLayout.addWidget(QLabel('Folder path:'), row, 0)
        self.folderPathLineEdit = widgets_line_edits.ElidingLineEdit(minWidth=240)
        self.folderPathLineEdit.setText(startFolderpath)
        gridLayout.addWidget(self.folderPathLineEdit, row, 1)
        self.browseButton = widgets_push_buttons.browseFileButton(
            start_dir=startFolderpath, openFolder=True
        )
        gridLayout.addWidget(self.browseButton, row, 2)
        
        row += 1
        gridLayout.addWidget(QLabel('Filename:'), row, 0)
        self.filenameLineEdit = widgets_line_edits.alphaNumericLineEdit()
        self.filenameLineEdit.setAlignment(Qt.AlignCenter)
        self.filenameLineEdit.setText(startFilename)
        gridLayout.addWidget(self.filenameLineEdit, row, 1)
        self.fileFormatLabel = QLabel('.mp4')
        gridLayout.addWidget(self.fileFormatLabel, row, 2)
        
        row += 1
        gridLayout.addWidget(QLabel('Add Scale Bar:'), row, 0)
        self.addScaleBarToggle = widgets_push_buttons.Toggle()
        gridLayout.addWidget(
            self.addScaleBarToggle, row, 1, alignment=Qt.AlignCenter
        )
        self.addScaleBarToggle.setChecked(isScaleBarPresent)
        
        if isTimelapseVideo:
            row += 1
            gridLayout.addWidget(QLabel('Add timestamp:'), row, 0)
            self.addTimestampToggle = widgets_push_buttons.Toggle()
            gridLayout.addWidget(
                self.addTimestampToggle, row, 1, alignment=Qt.AlignCenter
            )
            self.addTimestampToggle.setChecked(isTimestampPresent)
        
        for channel in channels:
            row += 1
            labelText = f'Rescale intensities (LUT) <i>{channel}</i>:'
            gridLayout.addWidget(QLabel(labelText), row, 0)
            rescaleItems = ['Rescale each 2D image']
            if SizeZ > 1:
                rescaleItems.append('Rescale across z-stack')
            if isTimelapseVideo:
                rescaleItems.append('Rescale across time frames')
            rescaleItems.append('Choose custom levels...')
            rescaleItems.append('Do no rescale, display raw image')
            rescaleIntensCombobox = QComboBox()
            rescaleIntensCombobox.addItems(rescaleItems)
            rescaleIntensHow = rescaleIntensChannelHowMapper.get(channel)
            if rescaleIntensHow is not None:
                rescaleIntensCombobox.setCurrentText(rescaleIntensHow)
            gridLayout.addWidget(rescaleIntensCombobox, row, 1)
            rescaleIntensCombobox.textActivated.connect(
                partial(self.emitRescaleIntens, channel=channel)
            )
        
        row += 1
        gridLayout.addWidget(QLabel('Save a PNG for each frame:'), row, 0)
        self.saveFramesToggle = widgets_push_buttons.Toggle()
        gridLayout.addWidget(
            self.saveFramesToggle, row, 1, alignment=Qt.AlignCenter
        )
        
        if isTimelapseVideo:
            row += 1
            gridLayout.addWidget(QLabel('Video compression quality preset (crf)'), row, 0)
            self.crf_widget = widgets_line_edits.IntLineEdit(allowNegative=False, initial=23, 
                                                minimum=0, maximum=51)
            gridLayout.addWidget(self.crf_widget, row, 1)
            info_txt = """
            The range of the Constant Rate Factor (CRF) scale is 0-51, where 0 is lossless, 23 is the default, and 51 is worst quality possible.
            <br><br> 
            A <b>lower value</b> generally leads to a <b>higher quality</b>, 
            and a subjectively sane range is 17-28.
            <br><br>
            Consider 17 or 18 to be 
            visually lossless or nearly so; it should look the same or nearly 
            the same as the input but it isn't technically lossless.
            <br><br>
            The range is exponential, so increasing the CRF value 
            +6 results in roughly half the bitrate / file size, while -6 leads 
            to roughly twice the bitrate.
            <br><br>
            Choose the highest CRF value that still provides an acceptable 
            quality. If the output looks good, but the video file size is too big for you, try a higher value. If it looks bad, choose a lower value.
            <br><br>
            For more information, see 
            <a href="https://trac.ffmpeg.org/wiki/Encode/H.264#crf">
            this page</a>.
            """
            infobutton = widgets_push_buttons.infoPushButton(
                info_text=info_txt, 
                info_title='Constant Rate Factor (CRF) information'
                )
            gridLayout.addWidget(infobutton, row, 2)
        else:
            self.crf_widget = None
        
        gridLayout.setColumnStretch(0, 0)
        gridLayout.setColumnStretch(1, 1)
        gridLayout.setColumnStretch(2, 0)
        
        self.fileFormatCombobox.currentTextChanged.connect(
            self.updateFileFormat
        )
        self.browseButton.sigPathSelected.connect(self.updateFolderPath)
        self.addScaleBarToggle.toggled.connect(self.addScaleBarToggled)
        if isTimelapseVideo:
            self.addTimestampToggle.toggled.connect(self.addTimestampToggled)
        
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()
        buttonsLayout.okButton.setText('Export')
        
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addLayout(gridLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
    
    def emitRescaleIntens(self, how, channel=''):
        self.sigRescaleIntensLut.emit(how, channel)
    
    def addScaleBarToggled(self, checked):
        self.sigAddScaleBar.emit(checked)
    
    def addTimestampToggled(self, checked):
        self.sigAddTimestamp.emit(checked)
    
    def updateFolderPath(self, folderPath):
        self.folderPathLineEdit.setText(folderPath)
        self.browseButton.setStartPath(folderPath)
    
    def updateFileFormat(self, fileFormat):
        self.fileFormatLabel.setText(f'.{fileFormat.lower()}')
    
    def validateFolderPath(self):
        folderPath = self.folderPathLineEdit.text()
        if os.path.exists(folderPath) and os.path.isdir(folderPath):
            return True
        
        text = html_utils.paragraph(
            'The selected folder path is not a valid folder or does not exist'
        )
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.warning(self, 'Not a valid folder', text)
        return False
    
    def validateFilename(self):
        filename = self.filenameLineEdit.text()
        if filename:
            return True
        
        text = html_utils.paragraph(
            'The filename cannot be empty!'
        )
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.warning(self, 'Not a valid folder', text)
        return False
    
    def validate(self):
        proceed = self.validateFolderPath()
        if not proceed:
            return False
        
        proceed = self.validateFilename()
        if not proceed:
            return False
        
        return True
    
    def preferences(self, makedirs=True):
        filename = f'{self.filenameLineEdit.text()}{self.fileFormatLabel.text()}'
        avi_filename = f'{self.filenameLineEdit.text()}.avi'
        avi_filepath = os.path.join(self.folderPathLineEdit.text(), avi_filename)
        png_foldername = (
            f'{self.filenameLineEdit.text()}_frames_PNG'
        )
        pngs_folderpath = os.path.join(
            self.folderPathLineEdit.text(), png_foldername
        )
        if makedirs:
            os.makedirs(pngs_folderpath, exist_ok=True)
        
        preferences = {
            'start_nav_var_num': self.startNavVarNumberEntry.value(),
            'stop_nav_var_num': self.stopNavVarNumberEntry.value(),
            'filepath': os.path.join(self.folderPathLineEdit.text(), filename), 
            'filename': self.filenameLineEdit.text(),
            'avi_filepath': avi_filepath,
            'pngs_folderpath': pngs_folderpath,
            'num_digits': len(str(self.stopNavVarNumberEntry.value())),
            'fps': self.fpsWidget.value(),
            'save_pngs':  self.saveFramesToggle.isChecked(),
            'is_timelapse': self.isTimelapseVideo,
            'dpi': self.dpiWidget.value(),
            'crf': self.crf_widget.value() if self.crf_widget is not None else None
        }
        return preferences
    
    def ok_cb(self):
        proceed = self.validate()
        if not proceed:
            return
        self.cancel = False
        self.sigOk.emit(self.preferences())
        self.selected_preferences = self.preferences()
        self.close()


class ExportToImageParametersDialog(dialogs_base.QBaseDialog):
    sigOk = Signal(dict)
    sigAddScaleBar = Signal(bool)
    sigRangeChanged = Signal(object)
    
    def __init__(
            self, parent=None, startFolderpath='', startFilename='', 
            startViewRange=None, isScaleBarPresent=False
        ):
        self.cancel = True
        
        super().__init__(parent=parent)
        
        self.setWindowTitle('Preferences for output image')
        
        mainLayout = QVBoxLayout()
        
        gridLayout = QGridLayout()
        
        row = 0
        gridLayout.addWidget(QLabel('View range X axis:'), row, 0)
        self.xRangeSelector = widgets_numeric_inputs.RangeSelector(integers=True)
        if startViewRange is not None:
            xRange, yRange = startViewRange
            self.xRangeSelector.setRange(*xRange)
        gridLayout.addWidget(self.xRangeSelector, row, 1)
        
        row += 1
        gridLayout.addWidget(QLabel('View range Y axis:'), row, 0)
        self.yRangeSelector = widgets_numeric_inputs.RangeSelector(integers=True)
        if startViewRange is not None:
            xRange, yRange = startViewRange
            self.yRangeSelector.setRange(*yRange)
        gridLayout.addWidget(self.yRangeSelector, row, 1)
        
        row += 1
        gridLayout.addWidget(QLabel('Width and Height:'), row, 0)
        self.widthHeightSelector = widgets_numeric_inputs.RangeSelector(
            integers=True, ordered=False
        )
        if startViewRange is not None:
            xRange, yRange = startViewRange
            width = int(xRange[1] - xRange[0])
            height = int(yRange[1] - yRange[0])
            self.widthHeightSelector.setRange(width, height)
        gridLayout.addWidget(self.widthHeightSelector, row, 1)
        self.lockSizeButton = widgets_push_buttons.LockPushButton()
        self.lockSizeButton.setCheckable(True)
        self.lockSizeButton.setToolTip(
            'Lock width and height'
        )
        gridLayout.addWidget(self.lockSizeButton, row, 2)
        
        row += 1
        gridLayout.addWidget(QLabel('File format:'), row, 0)
        self.fileFormatCombobox = QComboBox()
        self.fileFormatCombobox.addItems(['SVG', 'PNG', 'TIFF', 'JPEG'])
        gridLayout.addWidget(self.fileFormatCombobox, row, 1)
        
        row += 1
        self.dpiWidget = widgets_line_edits.IntLineEdit(allowNegative=False)
        self.dpiWidget.setValue(300)
        self.dpiWidget.label = QLabel('DPI')
        gridLayout.addWidget(self.dpiWidget.label, row, 0)
        gridLayout.addWidget(self.dpiWidget, row, 1)
        self.dpiWidget.hide()
        self.dpiWidget.label.hide()
        
        row += 1
        gridLayout.addWidget(QLabel('Folder path:'), row, 0)
        self.folderPathLineEdit = widgets_line_edits.ElidingLineEdit(minWidth=240)
        self.folderPathLineEdit.setText(startFolderpath)
        gridLayout.addWidget(self.folderPathLineEdit, row, 1)
        self.browseButton = widgets_push_buttons.browseFileButton(
            start_dir=startFolderpath, openFolder=True
        )
        gridLayout.addWidget(self.browseButton, row, 2)
        
        row += 1
        gridLayout.addWidget(QLabel('Filename:'), row, 0)
        self.filenameLineEdit = widgets_line_edits.alphaNumericLineEdit()
        self.filenameLineEdit.setAlignment(Qt.AlignCenter)
        self.filenameLineEdit.setText(startFilename)
        gridLayout.addWidget(self.filenameLineEdit, row, 1)
        self.fileFormatLabel = QLabel(
            f'.{self.fileFormatCombobox.currentText().lower()}'
        )
        gridLayout.addWidget(self.fileFormatLabel, row, 2)
        
        row += 1
        gridLayout.addWidget(QLabel('Add Scale Bar:'), row, 0)
        self.addScaleBarToggle = widgets_push_buttons.Toggle()
        gridLayout.addWidget(
            self.addScaleBarToggle, row, 1, alignment=Qt.AlignCenter
        )
        self.addScaleBarToggle.setChecked(isScaleBarPresent)
        
        self.fileFormatCombobox.currentTextChanged.connect(
            self.updateFileFormat
        )
        self.browseButton.sigPathSelected.connect(self.updateFolderPath)
        self.addScaleBarToggle.toggled.connect(self.addScaleBarToggled)
        self.xRangeSelector.sigLowValueChanged.connect(self.x0Changed)
        self.xRangeSelector.sigHighValueChanged.connect(self.x1Changed)
        self.yRangeSelector.sigLowValueChanged.connect(self.y0Changed)
        self.yRangeSelector.sigHighValueChanged.connect(self.y1Changed)
        self.widthHeightSelector.sigLowValueChanged.connect(self.widthChanged)
        self.widthHeightSelector.sigHighValueChanged.connect(self.heightChanged)
        self.widthHeightSelector.sigRangeManuallyChanged.connect(
            self.widthHeightManuallyChanged
        )
        
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()
        buttonsLayout.okButton.setText('Export')
        
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        gridLayout.setColumnStretch(2, 0)
        
        mainLayout.addLayout(gridLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
    
    def widthHeightManuallyChanged(self, *args):
        self.lockSizeButton.setChecked(True)
    
    def x0Changed(self, *args):
        if self.lockSizeButton.isChecked():
            x0, _ = self.xRangeSelector.range()
            yRange = self.yRangeSelector.range()
            width, height = self.widthHeightSelector.range()
            x1 = x0 + width
            xRange = (x0, x1)
        else:
            xRange = self.xRangeSelector.range()
            yRange = self.yRangeSelector.range()
            _, height = self.widthHeightSelector.range()
            width = int(xRange[1] - xRange[0])
            
        self.xRangeSelector.setRangeNoEmit(*xRange)
        self.yRangeSelector.setRangeNoEmit(*yRange)
        self.widthHeightSelector.setRangeNoEmit(width, height)
        self.rangeChanged()
        
    def x1Changed(self, *args):
        if self.lockSizeButton.isChecked():
            _, x1 = self.xRangeSelector.range()
            yRange = self.yRangeSelector.range()
            width, height = self.widthHeightSelector.range()
            x0 = x1 - width
            xRange = (x0, x1)
        else:
            xRange = self.xRangeSelector.range()
            yRange = self.yRangeSelector.range()
            _, height = self.widthHeightSelector.range()
            width = int(xRange[1] - xRange[0])
            
        self.xRangeSelector.setRangeNoEmit(*xRange)
        self.yRangeSelector.setRangeNoEmit(*yRange)
        self.widthHeightSelector.setRangeNoEmit(width, height)
        
        self.rangeChanged()
        
    def y0Changed(self, *args):
        if self.lockSizeButton.isChecked():
            xRange = self.xRangeSelector.range()
            y0, _ = self.yRangeSelector.range()
            width, height = self.widthHeightSelector.range()
            y1 = y0 + height
            yRange = (y0, y1)
        else:
            xRange = self.xRangeSelector.range()
            yRange = self.yRangeSelector.range()
            width, _ = self.widthHeightSelector.range()
            height = int(yRange[1] - yRange[0])
            
        self.xRangeSelector.setRangeNoEmit(*xRange)
        self.yRangeSelector.setRangeNoEmit(*yRange)
        self.widthHeightSelector.setRangeNoEmit(width, height)
        
        self.rangeChanged()
    
    def y1Changed(self, *args):
        if self.lockSizeButton.isChecked():
            xRange = self.xRangeSelector.range()
            _, y1 = self.yRangeSelector.range()
            width, height = self.widthHeightSelector.range()
            y0 = y1 - height
            yRange = (y0, y1)
        else:
            xRange = self.xRangeSelector.range()
            yRange = self.yRangeSelector.range()
            width, _ = self.widthHeightSelector.range()
            height = int(yRange[1] - yRange[0])
            
        self.xRangeSelector.setRangeNoEmit(*xRange)
        self.yRangeSelector.setRangeNoEmit(*yRange)
        self.widthHeightSelector.setRangeNoEmit(width, height)
        
        self.rangeChanged()
        
    def widthChanged(self, *args):
        self.widthHeightChanged()
        self.rangeChanged()
        
    def heightChanged(self, *args):
        self.widthHeightChanged()
        self.rangeChanged()
    
    def updateViewRangeExportToImageDialog(self, viewBox, viewRange, changed):
        xRange, yRange = viewRange
        self.xRangeSelector.setRangeNoEmit(*xRange)
        self.yRangeSelector.setRangeNoEmit(*yRange)
    
    def widthHeightChanged(self, *args):
        x0, _ = self.xRangeSelector.range()
        y0, _ = self.yRangeSelector.range()
        width, height = self.widthHeightSelector.range()
        x1 = x0 + width
        y1 = y0 + height
        self.xRangeSelector.setRangeNoEmit(x0, x1)
        self.yRangeSelector.setRangeNoEmit(y0, y1)
        self.rangeChanged()
    
    def rangeChanged(self, *args):
        xRange = self.xRangeSelector.range()
        yRange = self.yRangeSelector.range()
        self.sigRangeChanged.emit((xRange, yRange))
    
    def addScaleBarToggled(self, checked):
        self.sigAddScaleBar.emit(checked)
    
    def updateFolderPath(self, folderPath):
        self.folderPathLineEdit.setText(folderPath)
        self.browseButton.setStartPath(folderPath)
    
    def updateFileFormat(self, fileFormat):
        if fileFormat == 'SVG':
            self.dpiWidget.hide()
            self.dpiWidget.label.hide()
        else:
            self.dpiWidget.show()
            self.dpiWidget.label.show()
            
        self.fileFormatLabel.setText(f'.{fileFormat.lower()}')
    
    def validateFolderPath(self):
        folderPath = self.folderPathLineEdit.text()
        if os.path.exists(folderPath) and os.path.isdir(folderPath):
            return True
        
        text = html_utils.paragraph(
            'The selected folder path is not a valid folder or does not exist'
        )
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.warning(self, 'Not a valid folder', text)
        return False
    
    def validateFilename(self):
        filename = self.filenameLineEdit.text()
        if filename:
            return True
        
        text = html_utils.paragraph(
            'The filename cannot be empty!'
        )
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.warning(self, 'Not a valid folder', text)
        return False
    
    def validate(self):
        proceed = self.validateFolderPath()
        if not proceed:
            return False
        
        proceed = self.validateFilename()
        if not proceed:
            return False
        
        return True
    
    def setViewRange(self, xRange, yRange, emitSignal=True):
        if self.lockSizeButton.isChecked():
            x0, _ = xRange
            y0, _ = yRange
            width, height = self.widthHeightSelector.range()
            x1 = x0 + width
            y1 = y0 + height
            xRange = (x0, x1)
            yRange = (y0, y1)
        else:
            width = int(xRange[1] - xRange[0])
            height = int(yRange[1] - yRange[0])
            
        self.xRangeSelector.setRangeNoEmit(*xRange)
        self.yRangeSelector.setRangeNoEmit(*yRange)
        self.widthHeightSelector.setRangeNoEmit(width, height)
        if not emitSignal:
            return
        
        self.rangeChanged()
    
    def viewRange(self):
        xRange = self.xRangeSelector.range()
        yRange = self.yRangeSelector.range()
        return (xRange, yRange)
    
    def preferences(self):
        filename = f'{self.filenameLineEdit.text()}{self.fileFormatLabel.text()}'
        preferences = {
            'view_range_x': self.xRangeSelector.range(),
            'view_range_y': self.yRangeSelector.range(),
            'filepath': os.path.join(self.folderPathLineEdit.text(), filename), 
            'filename': self.filenameLineEdit.text(),
            'dpi': self.dpiWidget.value(),
        }
        return preferences
    
    def ok_cb(self):
        proceed = self.validate()
        if not proceed:
            return
        self.cancel = False
        self.sigOk.emit(self.preferences())
        self.selected_preferences = self.preferences()
        self.close()


class AutoSaveIntervalDialog(dialogs_base.QBaseDialog):
    sigValueChanged = Signal(float, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.cancel = True
        
        self.setWindowTitle('Change autosave interval')

        mainLayout = QVBoxLayout()
        
        self.autoSaveIntervalWidget = (
            AutoSaveIntervalWidget(parent=self)
        )
        
        mainLayout.addWidget(QLabel('Autosave interval:'))
        mainLayout.addWidget(self.autoSaveIntervalWidget)
        
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()
        
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
    
    def setValues(self, autoSaveIntevalValue, autoSaveIntervalUnit):
        self.autoSaveIntervalWidget.spinbox.setValue(autoSaveIntevalValue)
        self.autoSaveIntervalWidget.unitCombobox.setCurrentText(
            autoSaveIntervalUnit
        )
    
    def sizeHint(self):
        defaultWidth = super().sizeHint().width()
        defaultHeight = super().sizeHint().height()
        return QSize(defaultWidth*2, defaultHeight)
    
    def ok_cb(self):
        self.cancel = False
        self.sigValueChanged.emit(
            self.autoSaveIntervalWidget.spinbox.value(), 
            self.autoSaveIntervalWidget.unitCombobox.currentText()
        )
        self.close()
