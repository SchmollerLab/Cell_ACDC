"""Image metadata dialog QDialogMetadata (from apps.py).

load.py (loadData) asks for it when the metadata is missing, and the
Edit metadata utility (utils/editMetadata.py) opens it.
"""
import os
from functools import partial
import pandas as pd
from qtpy.QtGui import QIcon
from qtpy.QtCore import Qt, QEventLoop
from qtpy.QtWidgets import (
    QLabel, QVBoxLayout, QLineEdit, QPushButton, QHBoxLayout, QDialog,
    QCheckBox, QGridLayout, QSpinBox
)
from cellacdc import _window_stacking
from cellacdc import load, html_utils
from cellacdc import myutils
from cellacdc import widgets


class QDialogMetadata(QDialog):
    def __init__(
            self, SizeT, SizeZ, TimeIncrement,
            PhysicalSizeZ, PhysicalSizeY, PhysicalSizeX,
            ask_SizeT, ask_TimeIncrement, ask_PhysicalSizes,
            parent=None, font=None, imgDataShape=None, posData=None,
            singlePos=False, askSegm3D=True, additionalValues=None,
            forceEnableAskSegm3D=False, SizeT_metadata=None, 
            SizeZ_metadata=None, basename=''
        ):
        self.cancel = True
        self.ask_TimeIncrement = ask_TimeIncrement
        self.ask_PhysicalSizes = ask_PhysicalSizes
        self.askSegm3D = askSegm3D
        self.imgDataShape = imgDataShape
        self.posData = posData
        self._additionalValues = additionalValues
        self.SizeT_metadata = SizeT_metadata
        self.SizeZ_metadata = SizeZ_metadata
        super().__init__(parent)
        self.setWindowTitle('Image properties')

        mainLayout = QVBoxLayout()
        gridLayout = QGridLayout()
        # formLayout = QFormLayout()
        buttonsLayout = QGridLayout()

        if imgDataShape is not None:
            label = QLabel(
                html_utils.paragraph(
                    f'<i>Image data shape</i> = <b>{imgDataShape}</b><br>'
                )
            )
            mainLayout.addWidget(label, alignment=Qt.AlignCenter)

        row = 0
        self.basenameLineEdit = None
        if basename:
            gridLayout.addWidget(
                QLabel('Basename (read-only)'), row, 0, alignment=Qt.AlignRight
            )
            self.basenameLineEdit = QLineEdit()
            self.basenameLineEdit.setReadOnly(True)
            self.basenameLineEdit.setText(basename)
            minWidth = (
                self.basenameLineEdit.fontMetrics()
                .boundingRect(basename).width() + 10
            )
            self.basenameLineEdit.setMinimumWidth(minWidth)
            self.basenameLineEdit.setAlignment(Qt.AlignCenter)
            gridLayout.addWidget(self.basenameLineEdit, row, 1)
            row += 1
        
        gridLayout.addWidget(
            QLabel('Number of frames (SizeT)'), row, 0, alignment=Qt.AlignRight
        )
        self.SizeT_SpinBox = QSpinBox()
        self.SizeT_SpinBox.setMinimum(1)
        self.SizeT_SpinBox.setMaximum(2147483647)
        SizeTinfoButton = widgets.infoPushButton()
        self.allowEditSizeTcheckbox = QCheckBox('Let me edit it')
        if ask_SizeT:
            self.SizeT_SpinBox.setValue(SizeT)
            SizeTinfoButton.hide()
            self.allowEditSizeTcheckbox.hide()
        else:
            self.SizeT_SpinBox.setValue(1)
            self.SizeT_SpinBox.setDisabled(True)
            SizeTinfoButton.show()
            SizeTinfoButton.clicked.connect(self.showWhySizeTisGrayed)
            self.allowEditSizeTcheckbox.show()
            self.allowEditSizeTcheckbox.toggled.connect(self.allowEditSizeT)
        self.SizeT_SpinBox.setAlignment(Qt.AlignCenter)
        self.SizeT_SpinBox.valueChanged.connect(self.TimeIncrementShowHide)
        gridLayout.addWidget(self.SizeT_SpinBox, row, 1)
        gridLayout.addWidget(SizeTinfoButton, row, 2)
        gridLayout.setColumnStretch(2,0)
        gridLayout.addWidget(self.allowEditSizeTcheckbox, row, 3)
        gridLayout.setColumnStretch(3,0)

        row += 1
        gridLayout.addWidget(
            QLabel('Number of z-slices (SizeZ)'), row, 0, alignment=Qt.AlignRight
        )
        self.SizeZ_SpinBox = QSpinBox()
        self.SizeZ_SpinBox.setMinimum(1)
        self.SizeZ_SpinBox.setMaximum(2147483647)
        self.SizeZ_SpinBox.setValue(SizeZ)
        self.SizeZ_SpinBox.setAlignment(Qt.AlignCenter)
        self.SizeZ_SpinBox.valueChanged.connect(self.SizeZvalueChanged)
        gridLayout.addWidget(self.SizeZ_SpinBox, row, 1)

        row += 1
        self.TimeIncrementLabel = QLabel('Time interval (s)')
        gridLayout.addWidget(
            self.TimeIncrementLabel, row, 0, alignment=Qt.AlignRight
        )
        self.TimeIncrementSpinBox = widgets.FloatLineEdit()
        self.TimeIncrementSpinBox.setValue(TimeIncrement)
        gridLayout.addWidget(self.TimeIncrementSpinBox, row, 1)

        if SizeT == 1 or not ask_TimeIncrement:
            self.TimeIncrementSpinBox.hide()
            self.TimeIncrementLabel.hide()

        row += 1
        self.PhysicalSizeZLabel = QLabel('Physical Size Z (um/pixel)')
        gridLayout.addWidget(
            self.PhysicalSizeZLabel, row, 0, alignment=Qt.AlignRight
        )
        self.PhysicalSizeZSpinBox = widgets.FloatLineEdit()
        self.PhysicalSizeZSpinBox.setValue(PhysicalSizeZ)
        gridLayout.addWidget(self.PhysicalSizeZSpinBox, row, 1)

        if SizeZ==1 or not ask_PhysicalSizes:
            self.PhysicalSizeZSpinBox.hide()
            self.PhysicalSizeZLabel.hide()

        row += 1
        self.PhysicalSizeYLabel = QLabel('Physical Size Y (um/pixel)')
        gridLayout.addWidget(
            self.PhysicalSizeYLabel, row, 0, alignment=Qt.AlignRight
        )
        self.PhysicalSizeYSpinBox = widgets.FloatLineEdit()
        self.PhysicalSizeYSpinBox.setValue(PhysicalSizeY)
        gridLayout.addWidget(self.PhysicalSizeYSpinBox, row, 1)

        if not ask_PhysicalSizes:
            self.PhysicalSizeYSpinBox.hide()
            self.PhysicalSizeYLabel.hide()

        row += 1
        self.PhysicalSizeXLabel = QLabel('Physical Size X (um/pixel)')
        gridLayout.addWidget(
            self.PhysicalSizeXLabel, row, 0, alignment=Qt.AlignRight
        )
        self.PhysicalSizeXSpinBox = widgets.FloatLineEdit()
        self.PhysicalSizeXSpinBox.setValue(PhysicalSizeX)
        gridLayout.addWidget(self.PhysicalSizeXSpinBox, row, 1)

        if not ask_PhysicalSizes:
            self.PhysicalSizeXSpinBox.hide()
            self.PhysicalSizeXLabel.hide()

        row += 1
        self.isSegm3Dtoggle = widgets.Toggle()
        if posData is not None:
            self.isSegm3Dtoggle.setChecked(posData.getIsSegm3D())
            disableToggle = (
                # Disable toggle if not force enable and if
                # segm data was found (we cannot change the shape of
                # loaded segmentation in the GUI)
                posData.segmFound is not None
                and posData.segmFound
                and not forceEnableAskSegm3D
            )
            if disableToggle:
                self.isSegm3Dtoggle.setDisabled(True)
        self.isSegm3DLabel = QLabel('Work with 3D segmentation masks (z-stack)')
        gridLayout.addWidget(
            self.isSegm3DLabel, row, 0, alignment=Qt.AlignRight
        )
        gridLayout.addWidget(
            self.isSegm3Dtoggle, row, 1, alignment=Qt.AlignCenter
        )
        self.infoButtonSegm3D = QPushButton(self)
        self.infoButtonSegm3D.setCursor(Qt.WhatsThisCursor)
        self.infoButtonSegm3D.setIcon(QIcon(":info.svg"))
        gridLayout.addWidget(
            self.infoButtonSegm3D, row, 2, alignment=Qt.AlignLeft
        )
        self.infoButtonSegm3D.clicked.connect(self.infoSegm3D)
        if SizeZ == 1 or not askSegm3D:
            self.isSegm3DLabel.hide()
            self.isSegm3Dtoggle.hide()
            self.infoButtonSegm3D.hide()

        self.SizeZvalueChanged(SizeZ)

        self.additionalFieldsWidgets = []
        addFieldButton = widgets.addPushButton('Add custom field')
        addFieldInfoButton = widgets.infoPushButton()
        addFieldInfoButton.clicked.connect(self.showAddFieldInfo)
        addFieldButton.clicked.connect(self.addField)
        addFieldLayout = QHBoxLayout()
        addFieldLayout.addStretch(1)
        addFieldLayout.addWidget(addFieldButton)
        addFieldLayout.addWidget(addFieldInfoButton)
        addFieldLayout.addStretch(1)

        if singlePos:
            okTxt = 'Apply only to this Position'
        else:
            okTxt = 'Ok for loaded Positions'
        okButton = widgets.okPushButton(okTxt)
        okButton.setToolTip(
            'Save metadata only for current positionh'
        )
        okButton.setShortcut(Qt.Key_Enter)
        self.okButton = okButton

        if ask_TimeIncrement or ask_PhysicalSizes:
            okAllButton = QPushButton('Apply to ALL Positions')
            okAllButton.setToolTip(
                'Update existing Physical Sizes, Time '
                'interval, cell volume (fl), cell\n'
                'area (um^2), and time (s) for all the '
                'positions in the experiment folder.'
            )
            self.okAllButton = okAllButton

            selectButton = QPushButton('Select the Positions to be updated')
            selectButton.setToolTip(
                'Ask to select positions then update existing Physical Sizes, '
                'Time interval,\n'
                'cell volume (fl), cell area (um^2), and time (s) for selected '
                'positions.'
            )
            self.selectButton = selectButton
        else:
            self.okAllButton = None
            self.selectButton = None
            okButton.setText('Ok')

        cancelButton = widgets.cancelPushButton('Cancel')

        buttonsLayout.setColumnStretch(0, 1)
        buttonsLayout.addWidget(okButton, 0, 1)
        if ask_TimeIncrement or ask_PhysicalSizes:
            buttonsLayout.addWidget(okAllButton, 0, 2)
            buttonsLayout.addWidget(selectButton, 1, 1)
            buttonsLayout.addWidget(cancelButton, 1, 2)
        else:
            buttonsLayout.addWidget(cancelButton, 0, 2)
        buttonsLayout.setColumnStretch(3, 1)

        gridLayout.setColumnMinimumWidth(1, 100)
        mainLayout.addLayout(gridLayout)
        mainLayout.addSpacing(10)
        mainLayout.addLayout(addFieldLayout)
        # mainLayout.addLayout(formLayout)
        mainLayout.addSpacing(20)
        mainLayout.addStretch(1)
        mainLayout.addLayout(buttonsLayout)
        self.mainLayout = mainLayout

        okButton.clicked.connect(self.ok_cb)
        if ask_TimeIncrement or ask_PhysicalSizes:
            okAllButton.clicked.connect(self.ok_cb)
            selectButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        self.addAdditionalValues(additionalValues)

        self.setLayout(mainLayout)
        if font is not None:
            self.setFont(font)
        # self.setModal(True)
    
    def showWhySizeTisGrayed(self):
        txt = html_utils.paragraph(f"""
            The "Number of frames" field is grayed-out because you loaded multiple Positions.<br><br>
            Cell-ACDC <b>cannot load multiple time-lapse Positions</b>, 
            so it is assuming you are loading NON time-lapse data.<br><br>
            To load time-lapse data, load <b>one Position at a time</b>.<br><br>
            Note that you can still edit the number of frames if you need to correct it.<br>
            However, <b>you can only edit the metadata</b>, then the loading process will be stopped.
        """)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        msg.information(
            self, 'Why is the number of frames grayed out?', txt
        )

    def addAdditionalValues(self, values):
        if values is None:
            return

        for i, (name, value) in enumerate(values.items()):
            self.addField()
            nameWidget = self.additionalFieldsWidgets[i]['nameWidget']
            valueWidget = self.additionalFieldsWidgets[i]['valueWidget']
            nameWidget.setText(str(name).strip('__'))
            valueWidget.setText(str(value))

    def addField(self):
        nameWidget = QLineEdit()
        nameWidget.setAlignment(Qt.AlignCenter)
        valueWidget = QLineEdit()
        valueWidget.setAlignment(Qt.AlignCenter)
        removeButton = widgets.delPushButton()

        fieldLayout = QGridLayout()
        fieldLayout.addWidget(QLabel('Name'), 0, 0)
        fieldLayout.addWidget(nameWidget, 1, 0)
        fieldLayout.addWidget(QLabel('Value'), 0, 1)
        fieldLayout.addWidget(valueWidget, 1, 1)
        fieldLayout.addWidget(removeButton, 1, 2)

        self.additionalFieldsWidgets.append({
            'nameWidget': nameWidget,
            'valueWidget': valueWidget,
            'removeButton': removeButton,
            'layout': fieldLayout
        })

        idx = len(self.additionalFieldsWidgets)-1
        removeButton.clicked.connect(partial(self.removeField, idx))

        row = self.mainLayout.count()-3
        self.mainLayout.insertLayout(row, fieldLayout)

    def removeField(self, idx):
        widgets = self.additionalFieldsWidgets[idx]

        layoutToRemove = widgets['layout']
        for row in range(layoutToRemove.rowCount()):
            for col in range(layoutToRemove.columnCount()):
                item = layoutToRemove.itemAtPosition(row, col)
                if item is not None:
                    widget = item.widget()
                    layoutToRemove.removeWidget(widget)

        self.additionalFieldsWidgets.pop(idx)

        self.mainLayout.removeItem(layoutToRemove)

    def showAddFieldInfo(self):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph("""
            Add a <b>field (name and value)</b> that will be saved to the
            <code>metadata.csv</code> file and as a column in the
            <code>acdc_output.csv</code> table.<br><br>
            Example: a strain name or the replicate number.
        """)
        msg.information(self, 'Add field info', txt)

    def infoSegm3D(self):
        txt = (
            'Cell-ACDC supports both <b>2D and 3D segmentation</b>. If your data '
            'also have a time dimension, then you can choose to segment '
            'a specific z-slice (2D segmentation mask per frame) or all of them '
            '(3D segmentation mask per frame)<br><br>'
            'In any case, if you choose to activate <b>3D segmentation</b> then the '
            'segmentation mask will have the <b>same number of z-slices '
            'of the image data</b>.<br><br>'
            'Additionally, in the model parameters window, you will be able '
            'to choose if you want to segment the <b>entire 3D volume at once</b> '
            'or use the <b>2D model on each z-slice</b>, one by one.<br><br>'
            '<i>NOTE: if the toggle is disabled it means you already '
            'loaded segmentation data and the shape cannot be changed now.<br>'
            'if you need to start with a blank segmentation, '
            'use the "Create a new segmentation file" button instead of the '
            '"Load folder" button.'
            '</i>'
        )
        msg = widgets.myMessageBox()
        msg.setIcon()
        msg.setWindowTitle(f'3D segmentation info')
        msg.addText(html_utils.paragraph(txt))
        msg.addButton('   Ok   ')
        msg.exec_()

    def SizeZvalueChanged(self, val):
        if self.imgDataShape is not None and len(self.imgDataShape) < 3:
            return
        
        if val > 1 and self.imgDataShape is not None:
            maxSizeZ = self.imgDataShape[-3]
            self.SizeZ_SpinBox.setMaximum(maxSizeZ)
        else:
            self.SizeZ_SpinBox.setMaximum(2147483647)

        if val > 1:
            if self.ask_PhysicalSizes:
                self.PhysicalSizeZSpinBox.show()
                self.PhysicalSizeZLabel.show()
            if self.askSegm3D:
                self.isSegm3DLabel.show()
                self.isSegm3Dtoggle.show()
                self.infoButtonSegm3D.show()
        else:
            self.PhysicalSizeZSpinBox.hide()
            self.PhysicalSizeZLabel.hide()
            self.isSegm3DLabel.hide()
            self.isSegm3Dtoggle.hide()
            self.infoButtonSegm3D.hide()
        
        self.checkSegmDataShape()
    
    def checkSegmDataShape(self):
        if self.posData is None:
            return
        
        if self.isSegm3Dtoggle.isEnabled():
            return
        
        SizeT = self.SizeT_SpinBox.value()
        SizeZ = self.SizeZ_SpinBox.value()
        segm_data_ndim = self.posData.segm_data.ndim
        isSegm3D = False
        if segm_data_ndim == 4:
            # Segm data is 4D so it must be 3D over time
            isSegm3D = True
        elif segm_data_ndim == 3 and SizeZ > 1 and SizeT == 1:
            # Segm data is 3D while SizeT == 1 and SizeZ > 1 
            # --> also segm is 3D z-stack
            isSegm3D = True
        
        self.isSegm3Dtoggle.setDisabled(False)
        self.isSegm3Dtoggle.setChecked(isSegm3D)
        self.isSegm3Dtoggle.setDisabled(True)

    def TimeIncrementShowHide(self, val):
        self.checkSegmDataShape()
        if not self.ask_TimeIncrement:
            return
        
        if val > 1:
            self.TimeIncrementSpinBox.show()
            self.TimeIncrementLabel.show()
        else:
            self.TimeIncrementSpinBox.hide()
            self.TimeIncrementLabel.hide()
    
    def allowEditSizeT(self, checked):
        if checked:
            self.SizeT_SpinBox.setDisabled(False)
            if self.SizeT_metadata is not None:
                self.SizeT_SpinBox.setValue(self.SizeT_metadata)
        else:
            self.SizeT_SpinBox.setDisabled(True)
            self.SizeT_SpinBox.setValue(1)
    
    def warnEditingMetadata(self, Size, Size_metadata, which_dim):
        txt = html_utils.paragraph(f"""
            The <b>number of {which_dim} in the saved metadata is {Size_metadata}</b>,  
            but you are requesting to <b>change it to {Size}</b>.<br><br>
            Are you <b>sure you want to proceed</b>?
        """)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        _, noButton, yesButton = msg.warning(
            self, 'WARNING: Edinting saved metadata', txt, 
            buttonsTexts=('Cancel', 'No', 'Yes, edit the metadata')
        )
        return msg.clickedButton == yesButton

    def ok_cb(self, checked=False):
        self.cancel = False
        self.SizeT = self.SizeT_SpinBox.value()
        self.SizeZ = self.SizeZ_SpinBox.value()

        if self.SizeT_metadata is not None:
            if self.SizeT != self.SizeT_metadata:
                proceed = self.warnEditingMetadata(
                    self.SizeT, self.SizeT_metadata, 'frames'
                )
                if not proceed:
                    return
        
        if self.SizeZ_metadata is not None:
            if self.SizeZ != self.SizeZ_metadata:
                proceed = self.warnEditingMetadata(
                    self.SizeZ, self.SizeZ_metadata, 'z-slices'
                )
                if not proceed:
                    return


        self.isSegm3D = self.isSegm3Dtoggle.isChecked()

        self.TimeIncrement = self.TimeIncrementSpinBox.value()
        self.PhysicalSizeX = self.PhysicalSizeXSpinBox.value()
        self.PhysicalSizeY = self.PhysicalSizeYSpinBox.value()
        self.PhysicalSizeZ = self.PhysicalSizeZSpinBox.value()
        self._additionalValues = {
            f"__{field['nameWidget'].text()}":field['valueWidget'].text()
            for field in self.additionalFieldsWidgets
        }
        proceed = self.checkShapeMismatchMetadata()
        if not proceed:
            return 

        if self.posData is not None and self.sender() != self.okButton:
            exp_path = self.posData.exp_path
            pos_foldernames = myutils.get_pos_foldernames(exp_path)
            if self.sender() == self.selectButton:
                select_folder = load.select_exp_folder()
                select_folder.pos_foldernames = pos_foldernames
                select_folder.QtPrompt(
                    self, pos_foldernames, allow_cancel=False, toggleMulti=True
                )
                pos_foldernames = select_folder.selected_pos
            for pos in pos_foldernames:
                images_path = os.path.join(exp_path, pos, 'Images')
                ls = myutils.listdir(images_path)
                search = [file for file in ls if file.find('metadata.csv')!=-1]
                metadata_df = None
                if search:
                    fileName = search[0]
                    metadata_csv_path = os.path.join(images_path, fileName)
                    metadata_df = pd.read_csv(
                        metadata_csv_path
                        ).set_index('Description')
                if metadata_df is not None:
                    metadata_df.at['TimeIncrement', 'values'] = self.TimeIncrement
                    metadata_df.at['PhysicalSizeZ', 'values'] = self.PhysicalSizeZ
                    metadata_df.at['PhysicalSizeY', 'values'] = self.PhysicalSizeY
                    metadata_df.at['PhysicalSizeX', 'values'] = self.PhysicalSizeX
                    metadata_df.to_csv(metadata_csv_path)

                search = [file for file in ls if file.find('acdc_output.csv')!=-1]
                acdc_df = None
                if search:
                    fileName = search[0]
                    acdc_df_path = os.path.join(images_path, fileName)
                    acdc_df = pd.read_csv(acdc_df_path)
                    yx_pxl_to_um2 = self.PhysicalSizeY*self.PhysicalSizeX
                    vox_to_fl = self.PhysicalSizeY*(self.PhysicalSizeX**2)
                    if 'cell_vol_fl' not in acdc_df.columns:
                        continue
                    acdc_df['cell_vol_fl'] = acdc_df['cell_vol_vox']*vox_to_fl
                    acdc_df['cell_area_um2'] = acdc_df['cell_area_pxl']*yx_pxl_to_um2
                    acdc_df['time_seconds'] = acdc_df['frame_i']*self.TimeIncrement
                    try:
                        acdc_df.to_csv(acdc_df_path, index=False)
                    except PermissionError:
                        err_msg = html_utils.paragraph(
                            'The below file is open in another app '
                            '(Excel maybe?).<br><br>'
                            f'<code>{acdc_df_path}</code><br><br>'
                            'Close file and then press "Ok".'
                        )
                        msg = widgets.myMessageBox()
                        msg.critical(self, 'Permission denied', err_msg)
                        acdc_df.to_csv(acdc_df_path, index=False)

        elif self.sender() == self.selectButton:
            pass

        self.close()

    def checkShapeMismatchMetadata(self):
        valid4D = True
        valid3D = True
        valid2D = True
        if self.imgDataShape is None:
            self.close()
        elif len(self.imgDataShape) == 4:
            T, Z, Y, X = self.imgDataShape
            valid4D = self.SizeT == T and self.SizeZ == Z
        elif len(self.imgDataShape) == 3:
            TorZ, Y, X = self.imgDataShape
            valid3D = self.SizeT == TorZ or self.SizeZ == TorZ
        elif len(self.imgDataShape) == 2:
            valid2D = self.SizeT == 1 and self.SizeZ == 1
            
        valid = all([valid4D, valid3D, valid2D])
        if valid:
            return True
        
        if not valid4D:
            txt = (f"""
                You loaded <b>4D data</b>, hence the number of frames MUST be
                <b>{T}</b><br> and the number of z-slices MUST be <b>{Z}</b>.<br><br>
                What do you want to do?
            """)
        if not valid3D:
            txt = (f"""
                You loaded <b>3D data</b>, hence either the number of frames or 
                the number of z-slices is <b>{TorZ}</b>.<br><br>
                However, if the number of frames is greater than 1 then the<br>
                number of z-slices MUST be 1, and vice-versa.<br><br>
                What do you want to do?
            """)

        if not valid2D:
            txt = (f"""
                You loaded <b>2D data</b>, hence the number of frames MUST be <b>1</b>
                and the number of z-slices MUST be <b>1</b>.<br><br>
                What do you want to do?
            """)
        
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(txt)
        
        continueButton = widgets.okPushButton('Continue anyway')
        correctButton = widgets.editPushButton('Let me correct')
        
        msg.warning(
            self, 'Shape-metadata mismatch', txt,
            buttonsTexts=(continueButton, correctButton)
        )
        if msg.cancel or msg.clickedButton == correctButton:
            return False
        
        return True
    
    def cancel_cb(self, event):
        self.cancel = True
        self.close()

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
