"""Main-GUI-only segmentation dialogs (from apps.py).

Promptable model selection and loading (SelectPromptableModelDialog,
LoadSavedPromptableModelDialog, addCustomPromptModelMessages), the model
info viewer ViewTextDialog, startStopFramesDialog, the post-processing
dialog PostProcessSegmDialog, the combine-channels setup
CombineChannelsSetupDialogGUI and the unused randomWalkerDialog, plus
the widgets only they use.
"""
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QSpinBox, QGridLayout, QGroupBox
from cellacdc import myutils
from cellacdc.shared.widgets import item_views as widgets_item_views
import os
from typing import Iterable
import pathlib
import numpy as np
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.exposure
import skimage.draw
import skimage.registration
import skimage.color
import skimage.segmentation
import time
import html
from qtpy.QtGui import QColor
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFileDialog, QVBoxLayout, QPushButton, QHBoxLayout, QDialog,
    QAbstractItemView, QSlider, QTextEdit, QListWidgetItem
)
from cellacdc import _window_stacking
from cellacdc import core, html_utils
from cellacdc import qutils
from cellacdc import widgets
from cellacdc import promptable_models_path
from cellacdc import features
from cellacdc import fonts
from cellacdc.shared.dialogs import base as dialogs_base
from cellacdc.shared.dialogs import model_params as dialogs_model_params
from cellacdc.shared.dialogs import preprocessing as dialogs_preprocessing


class selectStartStopFrames(QGroupBox):
    def __init__(self, SizeT, currentFrameNum=0, parent=None):
        super().__init__(parent)
        selectFramesLayout = QGridLayout()

        self.startFrame_SB = QSpinBox()
        self.startFrame_SB.setAlignment(Qt.AlignCenter)
        self.startFrame_SB.setMinimum(1)
        self.startFrame_SB.setMaximum(SizeT-1)
        self.startFrame_SB.setValue(currentFrameNum)

        self.stopFrame_SB = QSpinBox()
        self.stopFrame_SB.setAlignment(Qt.AlignCenter)
        self.stopFrame_SB.setMinimum(1)
        self.stopFrame_SB.setMaximum(SizeT)
        self.stopFrame_SB.setValue(SizeT)

        selectFramesLayout.addWidget(QLabel('Start frame n.'), 0, 0)
        selectFramesLayout.addWidget(self.startFrame_SB, 1, 0)

        selectFramesLayout.addWidget(QLabel('Stop frame n.'), 0, 1)
        selectFramesLayout.addWidget(self.stopFrame_SB, 1, 1)

        self.warningLabel = QLabel()
        palette = self.warningLabel.palette()
        palette.setColor(self.warningLabel.backgroundRole(), Qt.red)
        palette.setColor(self.warningLabel.foregroundRole(), Qt.red)
        self.warningLabel.setPalette(palette)
        selectFramesLayout.addWidget(
            self.warningLabel, 2, 0, 1, 2, alignment=Qt.AlignCenter
        )

        self.setLayout(selectFramesLayout)

        self.stopFrame_SB.valueChanged.connect(self._checkRange)

    def _checkRange(self):
        start = self.startFrame_SB.value()
        stop = self.stopFrame_SB.value()
        if stop <= start:
            self.warningLabel.setText(
                'stop frame smaller than start frame'
            )
        else:
            self.warningLabel.setText('')


class selectTrackerGUI(widgets_item_views.QDialogListbox):
    def __init__(
            self, SizeT, currentFrameNo=1, parent=None
        ):
        trackers = myutils.get_list_of_trackers()
        super().__init__(
            'Select tracker', 'Select one of the following trackers',
            trackers, multiSelection=False, parent=parent,
            allowEmptySelection=False
        )
        self.setWindowTitle('Select tracker')

        self.selectFramesGroupbox = selectStartStopFrames(
            SizeT, currentFrameNum=currentFrameNo, parent=parent
        )

        self.mainLayout.insertWidget(1, self.selectFramesGroupbox)

    def ok_cb(self, checked=False, button=None):
        if self.selectFramesGroupbox.warningLabel.text():
            return
        else:
            self.startFrame = self.selectFramesGroupbox.startFrame_SB.value()
            self.stopFrame = self.selectFramesGroupbox.stopFrame_SB.value()
            super().ok_cb(checked, button=button)


def addCustomPromptModelMessages(QParent=None):
    modelFilePath = None
    msg = widgets.myMessageBox(showCentered=False, wrapText=False)
    txt = html_utils.paragraph("""
    Do you <b>already have</b> the <code>acdcPromptSegment.py</code> file for your code 
    or do you <b>need instructions</b> on how to set-up your custom model?<br>
    """)
    infoButton = widgets.infoPushButton(' I need instructions')
    browseButton = widgets.browseFileButton(' I have the model, let me select it')
    msg.information(
        QParent, 'Add custom promptable model', txt, 
        buttonsTexts=('Cancel', infoButton, browseButton),
        showDialog=False
    )
    browseButton.clicked.disconnect()
    browseButton.clicked.connect(msg.buttonCallBack)
    msg.exec_()
    if msg.cancel:
        return
    if msg.clickedButton == infoButton:           
        txt = myutils.get_add_custom_prompt_model_instructions()
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.information(
            QParent, 'Custom promptable model instructions', 
            txt, buttonsTexts=('Ok',),
            path_to_browse=promptable_models_path,
            browse_button_text='Open promptable models folder...'
        )
    else:
        homePath = pathlib.Path.home()
        modelFilePath = QFileDialog.getOpenFileName(
            QParent, 'Select the acdcPromptSegment.py file of your model',
            str(homePath), 'acdcPromptSegment.py file (*.py);;All files (*)'
        )[0]
        if not modelFilePath:
            return
    
    return modelFilePath


class SelectPromptableModelDialog(dialogs_base.QBaseDialog):
    def __init__(self, parent=None):
        self.cancel = True
        super().__init__(parent)
        
        self.setWindowTitle('Select model for segmentation')
        
        mainLayout = QVBoxLayout()
        
        label = QLabel(html_utils.paragraph(
            'Select model to use for segmentation: '
        ))
        mainLayout.addWidget(label, alignment=Qt.AlignCenter)
        
        listBox = widgets.listWidget()
        models = myutils.get_list_of_promptable_models()
        listBox.addItems(models)
        listBox.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        listBox.setCurrentRow(0)
        listBox.itemDoubleClicked.connect(self.ok_cb)
        
        self.listBox = listBox
        
        mainLayout.addWidget(listBox)
        
        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)
    
    def ok_cb(self):
        self.cancel = False
        self.model_name = self.listBox.currentItem().text()
        self.close()

class LoadSavedPromptableModelDialog(dialogs_base.QBaseDialog):
    """Select a trained promptable model (e.g., ConvPaint) to load.

    Lists the models saved in the Cell-ACDC settings folder (see
    `promptable_models.saved_models`). "Browse..." selects a model file in
    a custom location to be loaded.

    After `exec_`, `cancel` is False if a model was selected and
    `selectedModel` is the selected `saved_models.SavedModel`. If
    `isNewLink` is True, the model file was selected with "Browse..." and
    it is not in the list yet: the caller adds it as a link (the file is not
    copied) with `saved_models.add_link` once the model loaded successfully.
    """
    # Folder of the last model file selected with "Browse..."
    _lastBrowseFolderpath = ''

    def __init__(self, model_name, ext, start_dir='', parent=None):
        from cellacdc.promptable_models import saved_models

        self.cancel = True
        self.selectedModel = None
        self.isNewLink = False
        super().__init__(parent)

        self._saved_models = saved_models
        self.model_name = model_name
        self.ext = ext
        self.start_dir = start_dir
        self._models = {}

        self.setWindowTitle(f'Load trained {model_name} model')

        mainLayout = QVBoxLayout()

        label = QLabel(html_utils.paragraph(
            f'Select the trained <b>{model_name} model</b> to load:'
        ))
        mainLayout.addWidget(label)

        self.listBox = widgets.listWidget()
        self.listBox.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.listBox.itemDoubleClicked.connect(self.ok_cb)
        self.listBox.currentItemChanged.connect(self.updateSelection)
        mainLayout.addWidget(self.listBox)

        self.pathLabel = QLabel()
        self.pathLabel.setWordWrap(True)
        self.pathLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        mainLayout.addWidget(self.pathLabel)

        folderpath = saved_models.get_folderpath(model_name)
        noteLabel = QLabel(html_utils.paragraph(
            'Models saved from Cell-ACDC are stored in the folder '
            f'<code>{html.escape(folderpath)}</code><br><br>'
            'Click on "<b>Browse...</b>" to load a model file saved somewhere '
            f'else (e.g., trained in napari).<br>'
            'The file is <b>not copied</b>: it is added to the list as a '
            'link to its location.',
            font_size='11px'
        ))
        noteLabel.setWordWrap(True)
        noteLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        mainLayout.addSpacing(5)
        mainLayout.addWidget(noteLabel)

        buttonsLayout = QHBoxLayout()
        browseButton = widgets.OpenFilePushButton('Browse...')
        self.removeButton = widgets.delPushButton('Remove from list')
        cancelButton = widgets.cancelPushButton('Cancel')
        self.okButton = widgets.okPushButton(' Load ')

        buttonsLayout.addWidget(browseButton)
        buttonsLayout.addWidget(self.removeButton)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(self.okButton)

        browseButton.clicked.connect(self.browse_cb)
        self.removeButton.clicked.connect(self.remove_cb)
        cancelButton.clicked.connect(self.close)
        self.okButton.clicked.connect(self.ok_cb)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)
        self.setFont(fonts.font)

        self.populate()

    def populate(self, selectName=None):
        self._models = {
            model.name: model
            for model in self._saved_models.list_saved_models(
                self.model_name, self.ext
            )
        }
        self.listBox.clear()
        if not self._models:
            item = QListWidgetItem(
                'No saved models yet. Click on "Browse..." to load a model '
                'file.'
            )
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.listBox.addItem(item)
            self.updateSelection()
            return

        selectRow = 0
        for row, (name, model) in enumerate(self._models.items()):
            text = name
            if model.is_link:
                text = f'{text}  (linked)'
            if not model.exists:
                text = f'{text}  (file not found)'
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setToolTip(model.filepath)
            if not model.exists:
                item.setForeground(QColor(128, 128, 128))
            self.listBox.addItem(item)
            if selectName is not None and name.casefold() == (
                    selectName.casefold()
                ):
                selectRow = row
        self.listBox.setCurrentRow(selectRow)
        self.updateSelection()

    def currentModel(self):
        item = self.listBox.currentItem()
        if item is None:
            return None
        name = item.data(Qt.ItemDataRole.UserRole)
        return self._models.get(name)

    def updateSelection(self, *args):
        model = self.currentModel()
        self.okButton.setDisabled(model is None)
        self.removeButton.setDisabled(model is None)
        if model is None:
            self.pathLabel.setText('')
            return

        kind = 'Linked file' if model.is_link else 'File'
        self.pathLabel.setText(html_utils.paragraph(
            f'{kind}: <code>{html.escape(model.filepath)}</code>',
            font_size='11px'
        ))

    def browse_cb(self):
        start_dir = (
            LoadSavedPromptableModelDialog._lastBrowseFolderpath
            or self.start_dir
        )
        filepath = QFileDialog.getOpenFileName(
            self, f'Select {self.model_name} model file', start_dir,
            f'{self.model_name} model (*{self.ext});;All Files (*)'
        )[0]
        if not filepath:
            return

        LoadSavedPromptableModelDialog._lastBrowseFolderpath = (
            os.path.dirname(filepath)
        )
        if not filepath.lower().endswith(self.ext.lower()):
            msg = widgets.myMessageBox(wrapText=False)
            txt = html_utils.paragraph(
                f'{self.model_name} models are loaded from '
                f'<code>{self.ext}</code> files, while the selected file '
                f'is<br><br><code>{html.escape(filepath)}</code>'
            )
            msg.critical(self, 'Invalid model file', txt)
            return

        # Already in the list (saved or linked) --> load that entry
        existing = self._saved_models.find_by_filepath(
            self.model_name, filepath, self.ext
        )
        if existing is not None:
            self.populate(selectName=existing.name)
            if self.currentModel() == existing:
                self.ok_cb()
                return

        # The link is added by the caller only if loading the model
        # succeeds (see `isNewLink`), so that files that cannot be loaded
        # are not added to the list
        self.selectedModel = self._saved_models.SavedModel(
            self._saved_models.get_unique_name(
                self.model_name,
                self._saved_models.get_link_name(filepath, self.ext),
                self.ext
            ),
            os.path.abspath(filepath),
            True
        )
        self.isNewLink = True
        self.cancel = False
        self.close()

    def remove_cb(self):
        model = self.currentModel()
        if model is None:
            return

        msg = widgets.myMessageBox(wrapText=False)
        name = html.escape(model.name)
        path = html.escape(model.filepath)
        if model.is_link:
            txt = html_utils.paragraph(
                f'Do you want to remove the model <b>{name}</b> from the '
                'list?<br><br>'
                f'The linked model file<br><br><code>{path}</code><br><br>'
                'will <b>not</b> be deleted.'
            )
            buttonsTexts = ('Cancel', 'Remove from list')
        else:
            txt = html_utils.paragraph(
                f'Do you want to <b>delete</b> the saved model '
                f'<b>{name}</b>?<br><br>'
                f'The model file<br><br><code>{path}</code><br><br>'
                '(and the other files of this model in the same folder) '
                'will be <b>permanently deleted</b>.'
            )
            buttonsTexts = ('Cancel', 'Delete model')
        _, removeButton = msg.warning(
            self, 'Remove saved model?', txt, buttonsTexts=buttonsTexts
        )
        if msg.clickedButton != removeButton:
            return

        try:
            self._saved_models.remove_saved_model(self.model_name, model.name)
        except Exception as err:
            msg = widgets.myMessageBox(wrapText=False)
            txt = html_utils.paragraph(
                f'The model <b>{name}</b> could not be removed.<br><br>'
                f'Error: <code>{html.escape(str(err))}</code>'
            )
            msg.critical(self, 'Error removing model', txt)
        self.populate()

    def ok_cb(self, *args):
        model = self.currentModel()
        if model is None:
            return

        if not model.exists:
            msg = widgets.myMessageBox(wrapText=False)
            txt = html_utils.paragraph(
                f'The file of the model <b>{html.escape(model.name)}</b> '
                'does not exist anymore:<br><br>'
                f'<code>{html.escape(model.filepath)}</code><br><br>'
                'If the file was moved, remove the model from the list and '
                'add it again with "Browse...".'
            )
            msg.critical(self, 'Model file not found', txt)
            return

        self.selectedModel = model
        self.cancel = False
        self.close()


class ViewTextDialog(dialogs_base.QBaseDialog):
    def __init__(self, text, parent=None):
        super().__init__(parent)

        mainLayout = QVBoxLayout()

        textViewWidget = QTextEdit()
        textViewWidget.setReadOnly(True)

        textViewWidget.setText(text)

        buttonsLayout = QHBoxLayout()
        okButton = widgets.okPushButton('Ok')

        okButton.clicked.connect(self.close)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(okButton)

        mainLayout.addWidget(textViewWidget)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)
        self.setFont(fonts.font)

class startStopFramesDialog(dialogs_base.QBaseDialog):
    def __init__(
            self, SizeT, currentFrameNum=0, parent=None,
            windowTitle='Select frame range to segment'
        ):
        super().__init__(parent=parent)

        self.setWindowTitle(windowTitle)

        self.cancel = True

        layout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()

        self.selectFramesGroupbox = widgets.selectStartStopFrames(
            SizeT, currentFrameNum=currentFrameNum, parent=parent
        )

        okButton = widgets.okPushButton('Ok')
        cancelButton = widgets.cancelPushButton('Cancel')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        layout.addWidget(self.selectFramesGroupbox)
        layout.addLayout(buttonsLayout)
        self.setLayout(layout)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)

        self.setFont(fonts.font)

    def ok_cb(self):
        if self.selectFramesGroupbox.warningLabel.text():
            return
        else:
            self.startFrame = self.selectFramesGroupbox.startFrame_SB.value()
            self.stopFrame = self.selectFramesGroupbox.stopFrame_SB.value()
            self.cancel = False
            self.close()
    
    def show(self, block=False):
        super().show(block=False)

        self.resize(int(self.width()*1.5), self.height())

        if block:
            super().show(block=True)


class randomWalkerDialog(QDialog):
    def __init__(self, mainWindow):
        super().__init__(mainWindow)
        self.cancel = True
        self.mainWindow = mainWindow

        if mainWindow is not None:
            posData = self.mainWindow.data[self.mainWindow.pos_i]
            items = [posData.filename]
        else:
            items = ['test']
        try:
            posData = self.mainWindow.data[self.mainWindow.pos_i]
            items.extend(list(posData.ol_data_dict.keys()))
        except Exception as e:
            pass

        self.keys = items

        self.setWindowTitle('Random walker segmentation')

        self.colors = [self.mainWindow.RWbkgrColor,
                       self.mainWindow.RWforegrColor]

        mainLayout = QVBoxLayout()
        paramsLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        self.mainWindow.clearAllItems()

        row = 0
        paramsLayout.addWidget(QLabel('Background threshold:'), row, 0)
        row += 1
        self.bkgrThreshValLabel = QLabel('0.05')
        paramsLayout.addWidget(self.bkgrThreshValLabel, row, 1)
        self.bkgrThreshSlider = QSlider(Qt.Horizontal)
        self.bkgrThreshSlider.setMinimum(1)
        self.bkgrThreshSlider.setMaximum(100)
        self.bkgrThreshSlider.setValue(5)
        self.bkgrThreshSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.bkgrThreshSlider.setTickInterval(10)
        paramsLayout.addWidget(self.bkgrThreshSlider, row, 0)

        row += 1
        foregrQSLabel = QLabel('Foreground threshold:')
        # padding: top, left, bottom, right
        foregrQSLabel.setStyleSheet("font-size:13px; padding:5px 0px 0px 0px;")
        paramsLayout.addWidget(foregrQSLabel, row, 0)
        row += 1
        self.foregrThreshValLabel = QLabel('0.95')
        paramsLayout.addWidget(self.foregrThreshValLabel, row, 1)
        self.foregrThreshSlider = QSlider(Qt.Horizontal)
        self.foregrThreshSlider.setMinimum(1)
        self.foregrThreshSlider.setMaximum(100)
        self.foregrThreshSlider.setValue(95)
        self.foregrThreshSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.foregrThreshSlider.setTickInterval(10)
        paramsLayout.addWidget(self.foregrThreshSlider, row, 0)

        # Parameters link label
        row += 1
        url1 = 'https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_random_walker_segmentation.html'
        url2 = 'https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.random_walker'
        htmlTxt1 = f'<a href=\"{url1}">here</a>'
        htmlTxt2 = f'<a href=\"{url2}">here</a>'
        seeHereLabel = QLabel()
        seeHereLabel.setText(f'See {htmlTxt1} and {htmlTxt2} for details '
                              'about Random walker segmentation.')
        seeHereLabel.setTextFormat(Qt.RichText)
        seeHereLabel.setTextInteractionFlags(Qt.TextBrowserInteraction)
        seeHereLabel.setOpenExternalLinks(True)
        seeHereLabel.setFont(fonts.font)
        seeHereLabel.setStyleSheet("padding:12px 0px 0px 0px;")
        paramsLayout.addWidget(seeHereLabel, row, 0, 1, 2)

        computeButton = QPushButton('Compute segmentation')
        closeButton = QPushButton('Close')

        buttonsLayout.addWidget(computeButton, alignment=Qt.AlignRight)
        buttonsLayout.addWidget(closeButton, alignment=Qt.AlignLeft)

        paramsLayout.setContentsMargins(0, 10, 0, 0)
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(paramsLayout)
        mainLayout.addLayout(buttonsLayout)

        self.bkgrThreshSlider.sliderMoved.connect(self.bkgrSliderMoved)
        self.foregrThreshSlider.sliderMoved.connect(self.foregrSliderMoved)
        computeButton.clicked.connect(self.computeSegmAndPlot)
        closeButton.clicked.connect(self.close)

        self.setLayout(mainLayout)

        self.getImage()
        self.plotMarkers()

    def getImage(self):
        img = self.mainWindow.getDisplayedImg1()
        self.img = img/img.max()
        self.imgRGB = (skimage.color.gray2rgb(self.img)*255).astype(np.uint8)

    def setSize(self):
        x = self.pos().x()
        y = self.pos().y()
        h = self.size().height()
        w = self.size().width()
        if w < 400:
            w = 400
        self.setGeometry(x, y, w, h)

    def plotMarkers(self):
        imgMin, imgMax = self.computeMarkers()

        img = self.img

        imgRGB = self.imgRGB.copy()
        R, G, B = self.colors[0]
        imgRGB[:, :, 0][img < imgMin] = R
        imgRGB[:, :, 1][img < imgMin] = G
        imgRGB[:, :, 2][img < imgMin] = B
        R, G, B = self.colors[1]
        imgRGB[:, :, 0][img > imgMax] = R
        imgRGB[:, :, 1][img > imgMax] = G
        imgRGB[:, :, 2][img > imgMax] = B

        self.mainWindow.img1.setImage(imgRGB)

    def computeMarkers(self):
        bkgrThresh = self.bkgrThreshSlider.sliderPosition()/100
        foregrThresh = self.foregrThreshSlider.sliderPosition()/100
        img = self.img
        self.markers = np.zeros(img.shape, np.uint8)
        imgRange = img.max() - img.min()
        imgMin = img.min() + imgRange*bkgrThresh
        imgMax = img.min() + imgRange*foregrThresh
        self.markers[img < imgMin] = 1
        self.markers[img > imgMax] = 2
        return imgMin, imgMax

    def computeSegm(self, checked=True):
        self.mainWindow.storeUndoRedoStates(False)
        self.mainWindow.titleLabel.setText(
            'Randomly walking around... ', color='w')
        img = self.img
        img = skimage.exposure.rescale_intensity(img)
        t0 = time.time()
        lab = skimage.segmentation.random_walker(img, self.markers, mode='bf')
        lab = skimage.measure.label(lab>1)
        t1 = time.time()
        if len(np.unique(lab)) > 2:
            lab = skimage.morphology.remove_small_objects(lab, min_size=5)
        posData = self.mainWindow.data[self.mainWindow.pos_i]
        posData.lab = lab
        return t1-t0

    def computeSegmAndPlot(self):
        deltaT = self.computeSegm()

        posData = self.mainWindow.data[self.mainWindow.pos_i]

        self.mainWindow.update_rp()
        self.mainWindow.tracking(enforce=True)
        self.mainWindow.updateAllImages()
        self.mainWindow.warnEditingWithCca_df('Random Walker segmentation')
        txt = f'Random Walker segmentation computed in {deltaT:.3f} s'
        print('-----------------')
        print(txt)
        print('=================')
        # self.mainWindow.titleLabel.setText(txt, color='g')

    def bkgrSliderMoved(self, intVal):
        self.bkgrThreshValLabel.setText(f'{intVal/100:.2f}')
        self.plotMarkers()

    def foregrSliderMoved(self, intVal):
        self.foregrThreshValLabel.setText(f'{intVal/100:.2f}')
        self.plotMarkers()

    def closeEvent(self, event):
        self.mainWindow.segmModel = ''
        self.mainWindow.updateAllImages()


class PostProcessSegmDialog(dialogs_base.QBaseDialog):
    sigClosed = Signal()
    sigValueChanged = Signal(object, object)
    sigEditingFinished = Signal()
    sigApplyToAllFutureFrames = Signal(object, object, object)

    def __init__(
            self, posData, 
            mainWin=None, 
            useSliders=True, 
            maxSize=None
        ):
        super().__init__(mainWin)
        self.cancel = True
        self.mainWin = mainWin
        self.isTimelapse = False
        self.isMultiPos = False
        if mainWin is not None:
            self.isMultiPos = len(self.mainWin.data) > 1
            self.isTimelapse = self.mainWin.data[self.mainWin.pos_i].SizeT > 1

        self.setWindowTitle('Post-processing segmentation parameters')
        _window_stacking.set_window_flags_on_top(self, Qt.Tool)

        mainLayout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()

        self.postProcessGroupbox = dialogs_model_params.PostProcessSegmParams(
            'Post-processing parameters', posData,
            useSliders=useSliders, 
            maxSize=maxSize,
            parent=mainWin
        )

        self.postProcessGroupbox.valueChanged.connect(self.valueChanged)
        self.postProcessGroupbox.editingFinished.connect(self.onEditingFinished)

        if self.isTimelapse:
            applyAllButton = widgets.futurePushButton(
                'Apply to all frames...'
            )
            applyAllButton.clicked.connect(self.applyAll_cb)
            applyButton = widgets.okPushButton(
                'Apply', isDefault=False
            )
            applyButton.clicked.connect(self.apply_cb)
        elif self.isMultiPos:
            applyAllButton = widgets.futurePushButton(
                'Apply to all Positions...'
            )
            applyAllButton.clicked.connect(self.applyAll_cb)
            applyButton = widgets.okPushButton('Apply', isDefault=False)
            applyButton.clicked.connect(self.apply_cb)
        else:
            applyAllButton = widgets.okPushButton('Apply', isDefault=False)
            applyAllButton.clicked.connect(self.ok_cb)
            applyButton = None

        cancelButton = widgets.cancelPushButton('Cancel')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        if applyButton is not None:
            buttonsLayout.addWidget(applyButton)
        buttonsLayout.addWidget(applyAllButton)
        
        emitEditingFinishedButton = widgets.okPushButton()
        buttonsLayout.addWidget(emitEditingFinishedButton)
        emitEditingFinishedButton.hide()
        buttonsLayout.setContentsMargins(0,10,0,0)

        mainLayout.addWidget(self.postProcessGroupbox)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        cancelButton.clicked.connect(self.cancel_cb)

        if mainWin is not None:
            self.setPosData()
    
    def keyPressEvent(self, event) -> None:
        return super().keyPressEvent(event)

    def setPosData(self):
        if self.mainWin is None:
            return

        self.mainWin.storeUndoRedoStates(False)
        self.posData = self.mainWin.data[self.mainWin.pos_i]
        # self.img.setCurrentPosIndex(self.pos_i)
        # self.img.minMaxValuesMapper = self.mainWin.img1.minMaxValuesMapper
        self.origLab = self.posData.lab.copy()
        self.origRp = skimage.measure.regionprops(self.origLab) # why seperate rp here?
        self.origObjs = {obj.label:obj for obj in self.origRp}

    def valueChanged(self, value):
        lab, delObjs = self.apply()
        self.sigValueChanged.emit(lab, delObjs)
        
    def apply(self, origLab=None):
        self.mainWin.warnEditingWithCca_df(
            'post-processing segmentation mask', update_images=False
        )
        ccaAnnotRemoved = self.mainWin.removeCcaAnnotationsCurrentFrame()
        if ccaAnnotRemoved:
            self.mainWin.updateAllImages()

        if origLab is None:
            origLab = self.origLab.copy()

        lab, delIDs = core.post_process_segm(
            origLab, return_delIDs=True, **self.postProcessGroupbox.kwargs()
        )
        
        if self.postProcessGroupbox.selectedFeaturesRange():
            lab, custom_delIDs = features.custom_post_process_segm(
                self.posData, 
                self.postProcessGroupbox.groupedFeatures(), 
                lab, 
                self.posData.img_data[self.posData.frame_i], 
                self.posData.frame_i, 
                self.posData.filename, 
                self.posData.user_ch_name,
                self.postProcessGroupbox.selectedFeaturesRange(), 
                return_delIDs=True
            )
            delIDs.extend(custom_delIDs)
        
        delObjs = {delID:self.origObjs[delID] for delID in delIDs}
        return lab, delObjs

    def onEditingFinished(self):
        self.sigEditingFinished.emit()

    def ok_cb(self):
        self.cancel = False
        self.apply()
        self.onEditingFinished()
        self.close()

    def apply_cb(self):
        self.cancel = False
        self.apply()
        self.onEditingFinished()

    def applyAll_cb(self):
        self.cancel = False
        self.sigApplyToAllFutureFrames.emit(
            self.postProcessGroupbox.kwargs(),
            self.postProcessGroupbox.groupedFeatures(),
            self.postProcessGroupbox.selectedFeaturesRange()
        )
        self.close()

    def cancel_cb(self):
        self.cancel = True
        self.close()
    
    def undoChanges(self):
        if self.mainWin is not None:
            self.posData.lab = self.origLab
            self.mainWin.update_rp()
            self.mainWin.updateAllImages()

        # Undo if changes were applied to all future frames
        if hasattr(self, 'origSegmData'):
            if self.isTimelapse:
                current_frame_i = self.posData.frame_i
                for frame_i in range(self.posData.segmSizeT):
                    self.posData.frame_i = frame_i
                    origLab = self.origSegmData[frame_i]
                    lab = self.posData.allData_li[frame_i]['labels']
                    if lab is None:
                        # Non-visited frame modify segm_data
                        self.posData.segm_data[frame_i] = origLab
                    else:
                        self.posData.allData_li[frame_i]['labels'] = origLab.copy()
                        self.posData.lab = origLab.copy()
                        self.mainWin.update_rp()
                        # Get the rest of the stored metadata based on the new lab
                        self.mainWin.get_data()
                        self.mainWin.store_data()
                # Back to current frame
                self.posData.frame_i = current_frame_i
                self.mainWin.get_data()
                self.mainWin.updateAllImages()
            elif self.isMultiPos:
                current_pos_i = self.mainWin.pos_i
                # Apply to all future frames or future positions
                for pos_i, posData in enumerate(self.mainWin.data):
                    self.mainWin.pos_i = pos_i
                    origLab = self.origSegmData[pos_i]
                    self.posData.allData_li[0]['labels'] = lab.copy()
                    # Get the rest of the stored metadata based on the new lab
                    self.mainWin.get_data()
                    self.mainWin.store_data()
                # Back to current pos and current frame
                self.mainWin.pos_i = current_pos_i
                self.mainWin.get_data()
                self.mainWin.updateAllImages()

    def show(self, block=False):
        # self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show(block=False)
        self.resize(int(self.width()*1.5), self.height())
        super().show(block=block)

    def closeEvent(self, event):
        self.sigClosed.emit()
        if self.cancel:
            self.undoChanges()
        super().closeEvent(event)


class CombineChannelsSetupDialogGUI(dialogs_preprocessing.CombineChannelsSetupDialog):
    def __init__(
            self,
            channel_names: Iterable[str],
            df_metadata=None,
            isTimelapse=False,
            isZstack=False,
            isMultiPos=False,
            parent=None,
            hideOnClosing=False
        ):        
        super().__init__(
            channel_names,
            df_metadata=df_metadata,
            isTimelapse=isTimelapse, 
            isZstack=isZstack, 
            isMultiPos=isMultiPos, 
            parent=parent,
            hideOnClosing=hideOnClosing,
        )

        # remove the preprocess buttons, we use the comb version of them
        qutils.delete_widget(self.loadLastRecipeButton)
        qutils.delete_widget(self.saveRecipeButton)
        qutils.delete_widget(self.loadRecipeButton)

        # self.allButtons.remove(self.loadLastRecipeButton)
        self.allButtons.remove(self.saveRecipeButton)
        self.allButtons.remove(self.loadRecipeButton)

        self.previewCheckbox.setChecked(True)
        self.saveAsSegmlabel.setText('Save and view as segmentation')
    
    def steps(self, return_keepInputDataType=False):
        steps = self.combineChannelsWidget.steps()
        formula = self.formulaEditWidget.text()
        # if not return_keepInputDataType:
        #     return steps, formula
        
        keep_input_dtype = self.keepInputDataTypeToggle.isChecked()
        return steps, keep_input_dtype, formula
