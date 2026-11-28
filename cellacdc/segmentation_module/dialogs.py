"""Prompts used only by the segmentation and tracking module
(segmentation_module/main_window.py; from apps.py): the stop-frame
selection askStopFrameSegm and the z-plane view askZPlaneView, which
imports launchDataPrepSelectZPlane locally from
data_prep/main_window.py.
"""
import os
import traceback
from functools import partial
import pandas as pd
from qtpy.QtGui import QFontMetrics
from qtpy.QtCore import Qt, QEventLoop
from qtpy.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QHBoxLayout, QDialog, QFormLayout, QComboBox,
    QGroupBox
)
from cellacdc import _window_stacking
from cellacdc import load, html_utils
from cellacdc import printl
from cellacdc.myutils import table_utils
from cellacdc.shared.widgets import (
    basic_widgets as widgets_basic_widgets,
    numeric_inputs as widgets_numeric_inputs,
    push_buttons as widgets_push_buttons,
)
from cellacdc import plot
from cellacdc import path
from cellacdc import disableWindow


class askStopFrameSegm(QDialog):
    def __init__(
            self, user_ch_file_paths, user_ch_name, parent=None
        ):
        self.parent = parent
        self.cancel = True

        super().__init__(parent)
        self.setWindowTitle('Enter stop frame')

        self.visualizeWindows = []

        mainLayout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()

        # Message
        infoTxt = html_utils.paragraph("""
            Enter a <b>stop frame number</b> when to stop 
            segmentation for each Position loaded:
        """)
        infoLabel = QLabel(infoTxt, self)
        infoLabel.setAlignment(Qt.AlignCenter)
        # padding: top, left, bottom, right
        infoLabel.setStyleSheet("padding:0px 0px 8px 0px;")

        self.dataDict = {}

        exp_path_pos_mapper = path.get_exp_path_pos_foldernames_mapper(
            user_ch_file_paths
        )

        columnsLayout = QHBoxLayout()
        mainScrollArea = widgets_basic_widgets.ScrollArea()
        mainScrollAreaWidget = QWidget()
        mainScrollAreaWidget.setLayout(columnsLayout)
        mainScrollArea.setWidget(mainScrollAreaWidget)
        self.mainScrollArea = mainScrollArea

        # Form layout widget
        self.spinBoxes = []
        self.tab_idx = 0
        iter_items = exp_path_pos_mapper.items()
        self.groupboxScrollAreas = []
        
        for col, (exp_path, pos_folders_files) in enumerate(iter_items):
            groupboxScrollArea = widgets_basic_widgets.ScrollArea()
            self.groupboxScrollAreas.append(groupboxScrollArea)
            groupbox = QGroupBox()
            groupbox.setCheckable(False)
            groupbox.setToolTip(exp_path)
            groupboxLayout = QFormLayout()
            groupbox.setLayout(groupboxLayout)
            groupboxScrollArea.setWidget(groupbox)
            columnsLayout.addWidget(groupboxScrollArea)
            pos_folders = pos_folders_files['pos_foldernames']
            filenames = pos_folders_files['filenames']
            for i, pos_foldername in enumerate(pos_folders):
                img_filename = filenames[i]
                images_path = os.path.join(exp_path, pos_foldername, 'Images')
                img_path = os.path.join(images_path, img_filename)
                spinBox = widgets_numeric_inputs.SpinBox()
                spinBox.sigTabEvent.connect(self.keyTabEventSpinbox)
                posData = load.loadData(img_path, user_ch_name, QParent=parent)
                posData.getBasenameAndChNames(qparent=self)
                posData.buildPaths()
                posData.loadOtherFiles(
                    load_segm_data=False,
                    load_metadata=True,
                    loadSegmInfo=True,
                )
                spinBox.setMaximum(posData.SizeT)
                stopFrameNum = posData.readLastUsedStopFrameNumber()
                if stopFrameNum is None:
                    spinBox.setValue(posData.SizeT)
                else:
                    spinBox.setValue(stopFrameNum)
                spinBox.setAlignment(Qt.AlignCenter)
                visualizeButton = widgets_push_buttons.viewPushButton('Visualize')
                visualizeButton.clicked.connect(self.visualize_cb)
                formLabel = QLabel(html_utils.paragraph(f'{pos_foldername}  '))
                layout = QHBoxLayout()
                layout.addWidget(formLabel, alignment=Qt.AlignRight)
                layout.addWidget(spinBox)
                layout.addWidget(visualizeButton)
                self.dataDict[visualizeButton] = (spinBox, posData)
                groupboxLayout.addRow(layout)
                spinBox.idx = i
                self.spinBoxes.append(spinBox)
            
            fm = QFontMetrics(self.font())
            elidedTitle = fm.elidedText(
                exp_path, Qt.ElideLeft, groupbox.sizeHint().width()
            )
            groupbox.setTitle(elidedTitle)

        mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        mainLayout.addWidget(mainScrollArea)

        okButton = widgets_push_buttons.okPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')

        buttonsLayout.addStretch(1)    
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        # # self.setModal(True)

    def keyTabEventSpinbox(self, event, sender):
        self.tab_idx += 1      
        if self.tab_idx >= len(self.spinBoxes):
            self.tab_idx = 0
        focusSpinbox = self.spinBoxes[self.tab_idx]
        focusSpinbox.setFocus()

    def saveStopFrameNumbers(self):
        for spinBox, posData in self.dataDict.values():
            posData.metadata_df.at['stop_frame_num', 'values'] = spinBox.value()
            posData.metadataToCsv()

    def ok_cb(self, event):
        self.cancel = False
        try:
            self.saveStopFrameNumbers()
        except Exception as err:
            printl(traceback.format_exc())
        self.stopFrames = [
            spinBox.value() for spinBox, posData in self.dataDict.values()
        ]
        self.close()
    
    def closeEvent(self, event):
        for window in self.visualizeWindows:
            window.close()

    def visualize_cb(self, checked=True):
        self.setDisabled(True)
        spinBox, posData = self.dataDict[self.sender()]
        print('Loading image data...')
        posData.loadImgData()
        posData.frame_i = spinBox.value()-1
        win = plot.imshow(
            posData.img_data, 
            lut='gray',
            figure_title=posData.relPath,
            block=False
        )
        self.visualizeWindows.append(win)
        self.setDisabled(False)

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        screenSize = self.screen().size()
        maxWidth = screenSize.width() - 50
        maxHeight = screenSize.height() - 100
        width, height = 0, 0
        for scrollArea in self.groupboxScrollAreas:
            width += scrollArea.minimumWidthNoScrollbar()
            scrollAreaHeight = scrollArea.minimumHeightNoScrollbar()
            if scrollAreaHeight > height:
                height = scrollAreaHeight
        
        width += 70
        height += (
            self.sizeHint().height() 
            - self.mainScrollArea.sizeHint().height()
        )

        if width > maxWidth:
            width = maxWidth
        
        if height > maxHeight:
            height = maxHeight

        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        super().show()
        self.resize(width, height)
        self.move(25, 50)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()


class askZPlaneView(QDialog):
    def __init__(
            self, user_ch_file_paths, user_ch_name, 
            mainWin, version, parent=None, logger_func=print
        ):
        self._parent = parent
        self.cancel = True
        self._user_ch_name = user_ch_name
        self._mainWin = mainWin
        self._version = version
        self._logger_func = logger_func

        super().__init__(parent)
        self.setWindowTitle('Select z-plane view')

        self.visualizeWindows = []

        mainLayout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()

        # Message
        infoTxt = html_utils.paragraph("""
            Enter a <b>z-slice</b> or <b>projection mode</b> to segment for each Position loaded.<br><br>
            <i>NOTE: for more complex selection launch the data prep module</i>
        """)
        infoLabel = QLabel(infoTxt, self)
        infoLabel.setAlignment(Qt.AlignCenter)
        # padding: top, left, bottom, right
        infoLabel.setStyleSheet("padding:0px 0px 8px 0px;")

        self.dataDict = {}

        exp_path_pos_mapper = path.get_exp_path_pos_foldernames_mapper(
            user_ch_file_paths
        )

        columnsLayout = QHBoxLayout()
        mainScrollArea = widgets_basic_widgets.ScrollArea()
        mainScrollAreaWidget = QWidget()
        mainScrollAreaWidget.setLayout(columnsLayout)
        mainScrollArea.setWidget(mainScrollAreaWidget)
        self.mainScrollArea = mainScrollArea

        # Form layout widget
        self.spinBoxes = []
        self.tab_idx = 0
        iter_items = exp_path_pos_mapper.items()
        self.groupboxScrollAreas = []
        
        for col, (exp_path, pos_folders_files) in enumerate(iter_items):
            groupboxScrollArea = widgets_basic_widgets.ScrollArea()
            self.groupboxScrollAreas.append(groupboxScrollArea)
            groupbox = QGroupBox()
            groupbox.setCheckable(False)
            groupbox.setToolTip(exp_path)
            groupboxLayout = QFormLayout()
            groupbox.setLayout(groupboxLayout)
            groupboxScrollArea.setWidget(groupbox)
            columnsLayout.addWidget(groupboxScrollArea)
            pos_folders = pos_folders_files['pos_foldernames']
            filenames = pos_folders_files['filenames']
            for i, pos_foldername in enumerate(pos_folders):
                img_filename = filenames[i]
                images_path = os.path.join(exp_path, pos_foldername, 'Images')
                img_path = os.path.join(images_path, img_filename)
                spinBox = widgets_numeric_inputs.SpinBox()
                spinBox.sigTabEvent.connect(self.keyTabEventSpinbox)
                posData = load.loadData(img_path, user_ch_name, QParent=parent)
                posData.getBasenameAndChNames(qparent=self)
                posData.buildPaths()
                posData.loadOtherFiles(
                    load_segm_data=False,
                    load_metadata=True,
                    loadSegmInfo=True,
                )
                spinBox.setMaximum(posData.SizeZ)
                zProjComboBox = QComboBox()
                zProjComboBox.addItems([
                    'single z-slice',
                    'max z-projection',
                    'mean z-projection',
                    'median z-proj.'
                ])

                spinBox.setValue(round(posData.SizeZ/2))
                
                visualizeButton = widgets_push_buttons.viewPushButton(' View')
                visualizeButton.clicked.connect(
                    partial(self.visualize_cb, images_path=images_path)
                )

                dataPrepButton = widgets_push_buttons.dataPrepButton('Launch data prep...')
                dataPrepButton.clicked.connect(
                    partial(self.launch_dataprep, images_path=images_path)
                )

                formLabel = QLabel(html_utils.paragraph(f'{pos_foldername}  '))
                layout = QHBoxLayout()
                layout.addWidget(formLabel, alignment=Qt.AlignRight)
                layout.addWidget(spinBox)
                layout.addWidget(zProjComboBox)
                layout.addWidget(visualizeButton)
                layout.addWidget(dataPrepButton)
                self.dataDict[images_path] = (
                    spinBox, zProjComboBox, posData
                )
                groupboxLayout.addRow(layout)
                self.spinBoxes.append(spinBox)
            
            fm = QFontMetrics(self.font())
            elidedTitle = fm.elidedText(
                exp_path, Qt.ElideLeft, groupbox.sizeHint().width()
            )
            groupbox.setTitle(elidedTitle)

        self.disableControlsDataPreppedPos()

        mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        mainLayout.addWidget(mainScrollArea)

        okButton = widgets_push_buttons.okPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')

        buttonsLayout.addStretch(1)    
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)

        if posData.SizeT == 1:
            launchDataPrepButton = widgets_push_buttons.dataPrepButton(
                'Launch data prep with all Positions...'
            )
            buttonsLayout.addWidget(launchDataPrepButton)
            launchDataPrepButton.clicked.connect(self.launch_dataprep)

        buttonsLayout.addWidget(okButton)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

    def disableControlsDataPreppedPos(self):
        for images_path, values in self.dataDict.items():
            posData = values[-1]
            posData.loadOtherFiles(
                load_segm_data=False,
                loadSegmInfo=True,
            )
            disabled = False
            if posData.segmInfo_df is not None:
                segmInfofilenames = (
                    posData.segmInfo_df.index.get_level_values(0).unique()
                )
                for _filename in segmInfofilenames:
                    if _filename.endswith(self._user_ch_name):
                        disabled = True
                        break
            spinBox, zProjComboBox = values[:2]
            if disabled:
                tooltip = (
                    'Disabled because already selected in data prep.\n\n'
                    'Launch data prep if you need to modify it.'
                )
                spinBox.setToolTip(tooltip)
                zProjComboBox.setToolTip(tooltip)
            
            spinBox.setDisabled(disabled)
            zProjComboBox.setDisabled(disabled)

    def keyTabEventSpinbox(self, event, sender):
        self.tab_idx += 1      
        if self.tab_idx >= len(self.spinBoxes):
            self.tab_idx = 0
        focusSpinbox = self.spinBoxes[self.tab_idx]
        focusSpinbox.setFocus()

    def ok_cb(self, event):
        self.cancel = False
        for spinBox, zProjComboBox, posData in self.dataDict.values():
            if not spinBox.isEnabled():
                continue
            
            filename = posData.filename
            if posData.segmInfo_df is None:
                df = table_utils.getDefault_SegmInfo_df(posData, filename)
                posData.segmInfo_df = df
            
            if filename not in posData.segmInfo_df.index:
                df = table_utils.getDefault_SegmInfo_df(posData, filename)
                posData.segmInfo_df = pd.concat([df, posData.segmInfo_df])
            
            posData.segmInfo_df.loc[filename, 'z_slice_used_dataPrep'] = (
                spinBox.value()
            )
            posData.segmInfo_df.loc[filename, 'which_z_proj'] = (
                zProjComboBox.currentText()
            )
            posData.segmInfo_df.loc[filename, 'is_from_dataPrep'] = (
                False
            )
            posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)

        self.close()
    
    def closeEvent(self, event):
        for window in self.visualizeWindows:
            window.close()

    @disableWindow
    def visualize_cb(self, checked=True, images_path=None):
        if images_path is None:
            return

        spinBox, zProjComboBox, posData = self.dataDict[images_path]
        print('Loading image data...')
        posData.loadImgData()
        posData.frame_i = spinBox.value()-1
        win = plot.imshow(
            posData.img_data, 
            lut='gray',
            figure_title=posData.relPath,
            block=False,
            parent=self
        )
        self.visualizeWindows.append(win)

    def launch_dataprep(self, checked=True, images_path=None):
        if images_path is not None:
            exp_path = os.path.dirname(images_path)
            posData = self.dataDict[images_path][-1]
            user_ch_file_paths = [posData.imgPath]
        else:
            user_ch_file_paths = []
            for images_path, values in self.dataDict.items():
                posData = values[-1]
                user_ch_file_paths.append(posData.imgPath)
        
        self._launch_dataprep(exp_path, user_ch_file_paths)

    @disableWindow
    def _launch_dataprep(self, exp_path, user_ch_file_paths):
        from cellacdc.data_prep.main_window import launchDataPrepSelectZPlane

        cancel = launchDataPrepSelectZPlane(
            exp_path, 
            user_ch_file_paths, 
            self._user_ch_name, 
            mainWin=self._mainWin, 
            version=self._version,
            logger_func=self._logger_func,
            parent=self._parent
        )
        if not cancel:
            self.disableControlsDataPreppedPos()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        screenSize = self.screen().size()
        maxWidth = screenSize.width() - 50
        maxHeight = screenSize.height() - 100
        width, height = 0, 0
        for scrollArea in self.groupboxScrollAreas:
            width += scrollArea.minimumWidthNoScrollbar()
            scrollAreaHeight = scrollArea.minimumHeightNoScrollbar()
            if scrollAreaHeight > height:
                height = scrollAreaHeight
        
        width += 70
        height += (
            self.sizeHint().height() 
            - self.mainScrollArea.sizeHint().height()
        )

        if width > maxWidth:
            width = maxWidth
        
        if height > maxHeight:
            height = maxHeight

        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        super().show()
        self.resize(width, height)
        self.move(25, 50)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()
