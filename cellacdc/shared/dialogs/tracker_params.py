"""Tracker-specific parameter windows (Cell-ACDC, BayesianTracker, Delta)
opened by the tracker initialisation in myutils/segm_models.py (from
apps.py).

Paths to the tracker folders are built from cellacdc_path, not from this
file location.
"""
import os
import re
from qtpy.QtCore import Qt, QEventLoop
from qtpy.QtWidgets import (
    QLabel, QVBoxLayout, QLineEdit, QHBoxLayout, QDialog, QComboBox,
    QGridLayout, QSpinBox, QDoubleSpinBox, QGroupBox
)
from cellacdc import _window_stacking
from cellacdc import measurements, html_utils
from ..widgets import (
    combo_boxes as widgets_combo_boxes,
    item_views as widgets_item_views,
    message_box as widgets_message_box,
    push_buttons as widgets_push_buttons,
)
from cellacdc import cellacdc_path
from cellacdc import fonts


class CellACDCTrackerParamsWin(QDialog):
    def __init__(self, parent=None):
        self.cancel = True
        super().__init__(parent)

        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        self.setWindowTitle('Cell-ACDC tracker parameters')

        paramsLayout = QGridLayout()
        paramsBox = QGroupBox()

        row = 0
        label = QLabel(html_utils.paragraph(
            'Minimum overlap between objects'
        ))
        paramsLayout.addWidget(label, row, 0)
        maxOverlapSpinbox = QDoubleSpinBox()
        maxOverlapSpinbox.setAlignment(Qt.AlignCenter)
        maxOverlapSpinbox.setMinimum(0)
        maxOverlapSpinbox.setMaximum(1)
        maxOverlapSpinbox.setSingleStep(0.1)
        maxOverlapSpinbox.setValue(0.4)
        self.maxOverlapSpinbox = maxOverlapSpinbox
        paramsLayout.addWidget(maxOverlapSpinbox, row, 1)
        infoButton = widgets_push_buttons.infoPushButton()
        infoButton.clicked.connect(self.showInfo)
        paramsLayout.addWidget(infoButton, row, 2)
        paramsLayout.setColumnStretch(0, 0)
        paramsLayout.setColumnStretch(1, 1)
        paramsLayout.setColumnStretch(2, 0)

        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')
        okButton = widgets_push_buttons.okPushButton(' Ok ')
        cancelButton.clicked.connect(self.cancel_cb)
        okButton.clicked.connect(self.ok_cb)

        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        layout = QVBoxLayout()
        infoText = html_utils.paragraph('<b>Cell-ACDC tracker parameters</b>')
        infoLabel = QLabel(infoText)
        layout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        layout.addSpacing(10)
        paramsBox.setLayout(paramsLayout)
        layout.addWidget(paramsBox)
        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)
        layout.addStretch(1)
        self.setLayout(layout)
        self.setFont(fonts.font)

    def showInfo(self):
        msg = widgets_message_box.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            'Cell-ACDC tracker computes the percentage of overlap between '
            'all the objects<br> at frame <code>n</code> and all the '
            'objects in previous frame <code>n-1</code>.<br><br>'
            'All objects with <b>overlap less than</b> '
            '<code>Minimum overlap between objects</code><br>are considered '
            '<b>new objects</b>.<br><br>'
            'Set this value to 0 if you want to force tracking of ALL the '
            'objects<br> in the previous frame (e.g., if cells move a lot '
            'between frames)'
        )
        msg.information(self, 'Cell-ACDC tracker info', txt)

    def ok_cb(self, checked=False):
        self.cancel = False
        self.params = {'IoA_thresh': self.maxOverlapSpinbox.value()}
        self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        super().show()
        self.resize(int(self.width()*1.3), self.height())
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class BayesianTrackerParamsWin(QDialog):
    def __init__(
            self, segmShape, parent=None, channels=None, 
            currentChannelName=None
        ):
        self.cancel = True
        super().__init__(parent)

        self.channels = channels
        self.currentChannelName = currentChannelName

        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        self.setWindowTitle('Bayesian tracker parameters')

        paramsLayout = QGridLayout()
        paramsBox = QGroupBox()

        row = 0
        this_path = cellacdc_path
        default_model_path = os.path.join(
            this_path, 'trackers', 'BayesianTracker',
            'model', 'cell_config.json'
        )
        label = QLabel(html_utils.paragraph('Model path'))
        paramsLayout.addWidget(label, row, 0)
        modelPathLineEdit = QLineEdit()
        start_dir = ''
        if os.path.exists(default_model_path):
            start_dir = os.path.dirname(default_model_path)
            modelPathLineEdit.setText(default_model_path)
        self.modelPathLineEdit = modelPathLineEdit
        paramsLayout.addWidget(modelPathLineEdit, row, 1)
        browseButton = widgets_push_buttons.browseFileButton(
            title='Select Bayesian Tracker model file',
            ext={'JSON Config': ('.json',)},
            start_dir=start_dir
        )
        browseButton.sigPathSelected.connect(self.onPathSelected)
        paramsLayout.addWidget(browseButton, row, 2, alignment=Qt.AlignLeft)

        if self.channels is not None:
            row += 1
            label = QLabel(html_utils.paragraph('Intensity image channel:  '))
            paramsLayout.addWidget(label, row, 0)
            items = ['None', *self.channels]
            self.channelCombobox = widgets_combo_boxes.QCenteredComboBox()
            self.channelCombobox.addItems(items)
            paramsLayout.addWidget(self.channelCombobox, row, 1)
            if self.currentChannelName is not None:
                self.channelCombobox.setCurrentText(self.currentChannelName)

        row += 1
        label = QLabel(html_utils.paragraph('Features'))
        paramsLayout.addWidget(label, row, 0)
        selectFeaturesButton = widgets_push_buttons.setPushButton('Select features')
        paramsLayout.addWidget(selectFeaturesButton, row, 1)
        self.features = []
        selectFeaturesButton.clicked.connect(self.selectFeatures)

        row += 1
        label = QLabel(html_utils.paragraph('Verbose'))
        paramsLayout.addWidget(label, row, 0)
        verboseToggle = widgets_push_buttons.Toggle()
        verboseToggle.setChecked(True)
        self.verboseToggle = verboseToggle
        paramsLayout.addWidget(verboseToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph('Run optimizer'))
        paramsLayout.addWidget(label, row, 0)
        optimizeToggle = widgets_push_buttons.Toggle()
        optimizeToggle.setChecked(True)
        self.optimizeToggle = optimizeToggle
        paramsLayout.addWidget(optimizeToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph('Max search radius'))
        paramsLayout.addWidget(label, row, 0)
        maxSearchRadiusSpinbox = QSpinBox()
        maxSearchRadiusSpinbox.setAlignment(Qt.AlignCenter)
        maxSearchRadiusSpinbox.setMinimum(1)
        maxSearchRadiusSpinbox.setMaximum(2147483647)
        maxSearchRadiusSpinbox.setValue(50)
        self.maxSearchRadiusSpinbox = maxSearchRadiusSpinbox
        self.maxSearchRadiusSpinbox.setDisabled(True)
        paramsLayout.addWidget(maxSearchRadiusSpinbox, row, 1)

        row += 1
        Z, Y, X = segmShape
        label = QLabel(html_utils.paragraph('Tracking volume'))
        paramsLayout.addWidget(label, row, 0)
        volumeLineEdit = QLineEdit()
        defaultVol = f'  (0, {X}), (0, {Y})  '
        if Z > 1:
            defaultVol = f'{defaultVol}, (0, {Z})  '
        volumeLineEdit.setText(defaultVol)
        volumeLineEdit.setAlignment(Qt.AlignCenter)
        self.volumeLineEdit = volumeLineEdit
        paramsLayout.addWidget(volumeLineEdit, row, 1)

        row += 1
        label = QLabel(html_utils.paragraph('Interactive mode step size'))
        paramsLayout.addWidget(label, row, 0)
        stepSizeSpinbox = QSpinBox()
        stepSizeSpinbox.setAlignment(Qt.AlignCenter)
        stepSizeSpinbox.setMinimum(1)
        stepSizeSpinbox.setMaximum(2147483647)
        stepSizeSpinbox.setValue(100)
        self.stepSizeSpinbox = stepSizeSpinbox
        paramsLayout.addWidget(stepSizeSpinbox, row, 1)

        row += 1
        label = QLabel(html_utils.paragraph('Update method'))
        paramsLayout.addWidget(label, row, 0)
        updateMethodCombobox = QComboBox()
        updateMethodCombobox.addItems(['EXACT', 'APPROXIMATE'])
        self.updateMethodCombobox = updateMethodCombobox
        self.updateMethodCombobox.currentTextChanged.connect(self.methodChanged)
        paramsLayout.addWidget(updateMethodCombobox, row, 1)

        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')
        okButton = widgets_push_buttons.okPushButton(' Ok ')
        cancelButton.clicked.connect(self.cancel_cb)
        okButton.clicked.connect(self.ok_cb)

        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        layout = QVBoxLayout()
        infoText = html_utils.paragraph('<b>Bayesian Tracker parameters</b>')
        infoLabel = QLabel(infoText)
        layout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        layout.addSpacing(10)
        paramsBox.setLayout(paramsLayout)
        layout.addWidget(paramsBox)

        url = 'https://btrack.readthedocs.io/en/latest/index.html'
        moreInfoText = html_utils.paragraph(
            '<i>Find more info on the Bayesian Tracker\'s '
            f'<a href="{url}">home page</a></i>'
        )
        moreInfoLabel = QLabel(moreInfoText)
        moreInfoLabel.setOpenExternalLinks(True)
        layout.addWidget(moreInfoLabel, alignment=Qt.AlignCenter)

        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)
        layout.addStretch(1)
        self.setLayout(layout)
        self.setFont(fonts.font)
    
    def selectFeatures(self):
        features = measurements.get_btrack_features()
        selectWin = widgets_item_views.QDialogListbox(
            'Select features',
            'Select features to use for tracking:\n',
            features, multiSelection=True, parent=self, 
            includeSelectionHelp=True
        )
        for i in range(selectWin.listBox.count()):
            item = selectWin.listBox.item(i)
            if item.text() in self.features:
                item.setSelected(True)
        selectWin.exec_()
        if selectWin.cancel:
            return
        self.features = selectWin.selectedItemsText

    def methodChanged(self, method):
        if method == 'APPROXIMATE':
            self.maxSearchRadiusSpinbox.setDisabled(False)
        else:
            self.maxSearchRadiusSpinbox.setDisabled(True)

    def onPathSelected(self, path):
        self.modelPathLineEdit.setText(path)

    def ok_cb(self, checked=False):
        self.cancel = False
        try:
            m = re.findall(r'\((\d+), *(\d+)\)', self.volumeLineEdit.text())
            if len(m) < 2:
                raise
            self.volume = tuple([(int(start), int(end)) for start, end in m])
            if len(self.volume) == 2:
                self.volume = (self.volume[0], self.volume[1], (-1e5, 1e5))
        except Exception as e:
            self.warnNotAcceptedVolume()
            return

        if not os.path.exists(self.modelPathLineEdit.text()):
            self.warnNotValidPath()
            return

        self.intensityImageChannel = None
        self.verbose = self.verboseToggle.isChecked()
        self.max_search_radius = self.maxSearchRadiusSpinbox.value()
        self.update_method = self.updateMethodCombobox.currentText()
        self.model_path = os.path.normpath(self.modelPathLineEdit.text())
        self.params = {
            'model_path': self.model_path,
            'verbose': self.verbose,
            'volume': self.volume,
            'max_search_radius': self.max_search_radius,
            'update_method': self.update_method,
            'step_size': self.stepSizeSpinbox.value(),
            'optimize': self.optimizeToggle.isChecked(),
            'features': self.features
        }
        if self.channels is not None:
            if self.channelCombobox.currentText() != 'None':
                self.intensityImageChannel = self.channelCombobox.currentText()
        self.close()

    def warnNotValidPath(self):
        url = 'https://github.com/lowe-lab-ucl/segment-classify-track/tree/main/models'
        msg = widgets_message_box.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            'The model configuration file path<br><br>'
            f'{self.modelPathLineEdit.text()}<br><br> '
            'does <b>not exist.</b><br><br>'
            'You can find some <b>pre-configured models</b> '
            f'<a href="{url}">here</a>.'
        )
        msg.critical(
            self, 'Invalid volume', txt
        )

    def warnNotAcceptedVolume(self):
        msg = widgets_message_box.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            f'{self.volumeLineEdit.text()} is <b>not a valid volume!</b><br><br>'
            'Valid volume is for example (0, 2048), (0, 2048)<br>'
            'for 2D segmentation or (0, 2048), (0, 2048), (0, 2048)<br>'
            'for 3D segmentation.'
        )
        msg.critical(
            self, 'Invalid volume', txt
        )

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        super().show()
        self.resize(int(self.width()*1.3), self.height())
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class DeltaTrackerParamsWin(QDialog):

    def __init__(self, posData=None, parent=None):
        self.cancel = True
        super().__init__(parent)

        _window_stacking.set_window_flags_on_top(self, Qt.Dialog)
        self.setWindowTitle('Delta tracker parameters')

        paramsLayout = QGridLayout()
        paramsBox = QGroupBox()

        row = 0
        this_path = cellacdc_path
        default_model_path = this_path

        label = QLabel(html_utils.paragraph('Original Images path'))
        paramsLayout.addWidget(label, row, 0)
        modelPathLineEdit = QLineEdit()
        start_dir = ''
        if os.path.exists(default_model_path):
            start_dir = os.path.dirname(default_model_path)
            modelPathLineEdit.setText(default_model_path)
        self.modelPathLineEdit = modelPathLineEdit
        paramsLayout.addWidget(modelPathLineEdit, row, 1)
        browseButton = widgets_push_buttons.browseFileButton(
            title='Select Original Images',
            ext={'TIFF': ('.tif',)},
            start_dir=start_dir
        )
        if posData is not None:
            modelPathLineEdit.setText(posData.imgPath)
        browseButton.sigPathSelected.connect(self.onPathSelected)
        paramsLayout.addWidget(browseButton, row, 2, alignment=Qt.AlignLeft)

        row += 1
        label = QLabel(html_utils.paragraph('Model Type'))
        paramsLayout.addWidget(label, row, 0)
        updateMethodCombobox = QComboBox()
        updateMethodCombobox.addItems(['2D', 'mothermachine'])
        self.model_type = '2D'
        self.updateMethodCombobox = updateMethodCombobox
        self.updateMethodCombobox.currentTextChanged.connect(self.methodChanged)
        paramsLayout.addWidget(updateMethodCombobox, row, 1)

        row += 1
        label = QLabel(html_utils.paragraph('Single Mother Machine Chamber?'))
        paramsLayout.addWidget(label, row, 0)
        chamberToggle = widgets_push_buttons.Toggle()
        chamberToggle.setChecked(True)
        self.chamberToggle = chamberToggle
        paramsLayout.addWidget(chamberToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph('Verbose'))
        paramsLayout.addWidget(label, row, 0)
        verboseToggle = widgets_push_buttons.Toggle()
        verboseToggle.setChecked(True)
        self.verboseToggle = verboseToggle
        paramsLayout.addWidget(verboseToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph('Legacy Save (.mat)'))
        paramsLayout.addWidget(label, row, 0)
        legacyToggle = widgets_push_buttons.Toggle()
        legacyToggle.setChecked(False)
        self.legacyToggle = legacyToggle
        paramsLayout.addWidget(legacyToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph('Pickle (.pkl)'))
        paramsLayout.addWidget(label, row, 0)
        pickleToggle = widgets_push_buttons.Toggle()
        pickleToggle.setChecked(False)
        self.pickleToggle = pickleToggle
        paramsLayout.addWidget(pickleToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph('Movie (.mp4) *only for 2D images'))
        paramsLayout.addWidget(label, row, 0)
        movieToggle = widgets_push_buttons.Toggle()
        movieToggle.setChecked(False)
        self.movieToggle = movieToggle
        paramsLayout.addWidget(movieToggle, row, 1, alignment=Qt.AlignCenter)

        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')
        okButton = widgets_push_buttons.okPushButton(' Ok ')
        cancelButton.clicked.connect(self.cancel_cb)
        okButton.clicked.connect(self.ok_cb)

        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        layout = QVBoxLayout()
        infoText = html_utils.paragraph('<b>Delta Tracker parameters</b>')
        infoLabel = QLabel(infoText)
        layout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        layout.addSpacing(10)
        paramsBox.setLayout(paramsLayout)
        layout.addWidget(paramsBox)

        url = 'https://delta.readthedocs.io/en/latest/'
        moreInfoText = html_utils.paragraph(
            '<i>Find more info on Delta Tracker\'s '
            f'<a href="{url}">home page</a></i>'
        )
        moreInfoLabel = QLabel(moreInfoText)
        moreInfoLabel.setOpenExternalLinks(True)
        layout.addWidget(moreInfoLabel, alignment=Qt.AlignCenter)

        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)
        layout.addStretch(1)
        self.setLayout(layout)
        self.setFont(fonts.font)

    def methodChanged(self, method):
        if method == 'mothermachine':
            self.model_type = 'mothermachine'

    def onPathSelected(self, path):
        self.modelPathLineEdit.setText(path)

    def ok_cb(self, checked=False):
        self.cancel = False

        if not os.path.exists(self.modelPathLineEdit.text()):
            self.warnNotValidPath()
            return

        self.verbose = self.verboseToggle.isChecked()
        self.legacy = self.legacyToggle.isChecked()
        self.pickle = self.pickleToggle.isChecked()
        self.movie = self.movieToggle.isChecked()
        self.chamber = self.chamberToggle.isChecked()
        self.model_path = os.path.normpath(self.modelPathLineEdit.text())
        self.params = {
            'original_images_path': self.model_path,
            'verbose': self.verbose,
            'legacy': self.legacy,
            'pickle': self.pickle,
            'movie': self.movie,
            'model_type': self.model_type,
            'single mothermachine chamber': self.chamber
        }
        self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        super().show()
        self.resize(int(self.width()*1.3), self.height())
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()
