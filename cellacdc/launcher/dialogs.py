"""Launcher-only dialogs (from apps.py): the Track sub-cellular objects
setup TrackSubCellObjectsDialog, LogoDialog (only a commented use) and
the unused ChangeUserProfileFolderPathDialog, plus the widgets only they
use.
"""
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDoubleSpinBox
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QLabel, QVBoxLayout, QDialog
from cellacdc import load, html_utils
from cellacdc import widgets
from cellacdc import user_profile_path
from cellacdc import fonts
from cellacdc.shared.dialogs import base as dialogs_base


class CenteredDoubleSpinbox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31-1)


class TrackSubCellObjectsDialog(dialogs_base.QBaseDialog):
    def __init__(self, basename='', parent=None):
        self.cancel = True
        super().__init__(parent=parent)
    
        self.setWindowTitle('Track sub-cellular objects parameters')

        mainLayout = QVBoxLayout()
        entriesLayout = widgets.FormLayout()

        row = 0
        infoTxt = html_utils.paragraph("""
            Select <b>behaviour with untracked objects</b>:<br><br>
            NOTE: this utility <b>always create new files</b>.
            Original segmentation masks <br>are not modified</b>.
        """)
        options = (
            'Delete sub-cellular objects that do not belong to any cell',
            'Delete cells that do not have any sub-cellular object',
            'Delete both cells and sub-cellular objects without an assignment',
            'Only track the objects and keep all the non-tracked objects'
        )
        combobox = widgets.QCenteredComboBox()
        combobox.addItems(options)
        self.optionsWidget = widgets.formWidget(
            combobox, addInfoButton=True, labelTextLeft='Tracking mode: ',
            infoTxt=infoTxt
        )
        entriesLayout.addFormWidget(self.optionsWidget, row=row)
        
        row += 1
        infoTxt = html_utils.paragraph("""
            Re-label sub-cellular objects before assigning them to the cell.<br><br> 
            Activate this option if you have <b>merged sub-cellular objects</b>   
            that must be separated, or the segmentation is a <b>boolean mask</b> 
            (i.e., semantic segmentation).
        """)
        self.relabelSubObjLab = widgets.formWidget(
            widgets.Toggle(), addInfoButton=True, stretchWidget=False,
            labelTextLeft='Re-label sub-cellular objects before tracking: ', 
            infoTxt=infoTxt
        )
        entriesLayout.addFormWidget(self.relabelSubObjLab, row=row)

        row += 1
        IoAtext = html_utils.paragraph("""
            Enter a <b>minimum percentage (0-1) of the sub-cellular object's area</b><br>
            that MUST overlap with the parent cell to be considered belonging to a cell:
        """)
        spinbox = widgets.CenteredDoubleSpinbox()
        spinbox.setMaximum(1)
        spinbox.setValue(0.5)
        spinbox.setSingleStep(0.1)
        self.IoAwidget = widgets.formWidget(
            spinbox, addInfoButton=True, labelTextLeft='IoA threshold: ',
            infoTxt=IoAtext
        )
        entriesLayout.addFormWidget(self.IoAwidget, row=row)

        row += 1
        infoTxt = html_utils.paragraph("""
            The third segmentation file is the result of <b>subtracting the 
            sub-cellular objects from the parent objects</b><br><br>
            This is useful if, for example, you need to compute measurements 
            only from the cytoplasm (i.e., the sub-cellular object is the nucleus).
        """)
        self.createThirdSegmWidget = widgets.formWidget(
            widgets.Toggle(), addInfoButton=True, stretchWidget=False,
            labelTextLeft='Create third segmentation: ', infoTxt=infoTxt
        )
        entriesLayout.addFormWidget(self.createThirdSegmWidget, row=row)

        row += 1
        infoTxt = html_utils.paragraph("""
            Text to append at the end of the third segmentation file.<br><br>
            The third segmentation file is the result of <b>subtracting the 
            sub-cellular objects from the parent objects</b><br><br>
            This is useful if, for example, you need to compute measurements 
            only from the cytoplasm (i.e., the sub-cellular object is the nucleus).
        """)
        lineEdit = widgets.alphaNumericLineEdit()
        lineEdit.setText('difference')
        lineEdit.setAlignment(Qt.AlignCenter)
        self.appendTextWidget = widgets.formWidget(
            lineEdit, addInfoButton=True, labelTextLeft='Text to append: ', 
            infoTxt=infoTxt
        )
        entriesLayout.addFormWidget(self.appendTextWidget, row=row)
        self.appendTextWidget.setDisabled(True)
        

        self.createThirdSegmWidget.widget.toggled.connect(
            self.createThirdSegmToggled
        )

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addLayout(entriesLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)
        self.setFont(fonts.font)
    
    def createThirdSegmToggled(self, checked):
        self.appendTextWidget.setDisabled(not checked)
    
    def ok_cb(self):
        self.cancel = False
        if self.createThirdSegmWidget.widget.isChecked():
            if not self.appendTextWidget.widget.text():
                msg = widgets.myMessageBox(showCentered=False, wrapText=False)
                txt = html_utils.paragraph(
                    'When creating the third segmentation file, '
                    '<b>the name to append cannot be empty!</b>'
                )
                msg.critical(self, 'Empty name', txt)
                return
                
        self.trackSubCellObjParams = {
            'how': self.optionsWidget.widget.currentText(),
            'IoA': self.IoAwidget.widget.value(),
            'createThirdSegm': self.createThirdSegmWidget.widget.isChecked(),
            'relabelSubObjLab': self.relabelSubObjLab.widget.isChecked(),
            'thirdSegmAppendedText': self.appendTextWidget.widget.text()
        }
        self.close()


class ChangeUserProfileFolderPathDialog(dialogs_base.QBaseDialog):
    def __init__(self, posData, parent=None):
        super().__init__(parent=parent)
        
        self.cancel = True
        
        self.setWindowTitle('Change user profile folder path')
        
        mainLayout = QVBoxLayout()
        
        acdc_folders = load.get_all_acdc_folders(user_profile_path)
        acdc_folders_format = [f'  - {folder}' for folder in acdc_folders]
        acdc_folders_format = '<br>'.join(acdc_folders_format)
        
        txt = (f"""
            Current user profile path:<br><br>
            <code>{user_profile_path}</code><br><br>
            The user profile contains the following Cell-ACDC folders:<br><br>
            {acdc_folders_format}<br><br>
            After clicking "Ok" you will be <b>asked to select the folder</b> where 
            you want to <b>migrate</b> the user profile data. 
        """)
        
        txt = html_utils.paragraph(txt)
        label = QLabel(txt)
        
        mainLayout.addWidget(label)
        
        buttonsLayout = widgets.CancelOkButtonsLayout()
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch()
        
        self.setLayout(mainLayout)
    
    def ok_cb(self):
        self.cancel = False
        self.close()


class LogoDialog(QDialog):
    def __init__(self, logo_path, icon_path, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout()
        
        self.setWindowFlags(Qt.FramelessWindowHint)
        # self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        # self.setAttribute(Qt.WA_TranslucentBackground)
        # self.setWindowIcon(QIcon(icon_path))
        
        labelLogo = QLabel()
        pixmapLogo = QPixmap(logo_path)
        labelLogo.setPixmap(pixmapLogo)
        
        layout.addWidget(labelLogo)
    
        self.setLayout(layout)
