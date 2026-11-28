from .. import html_utils
from cellacdc.myutils import filesystem
from cellacdc.shared.dialogs import file_selection as dialogs_file_selection
from cellacdc.shared.widgets import (
    combo_boxes as widgets_combo_boxes,
    message_box as widgets_message_box,
    push_buttons as widgets_push_buttons,
)
from . import segm_workers

from .base import NewThreadMultipleExpBaseUtil
from qtpy.QtWidgets import QLabel, QVBoxLayout, QFormLayout
from cellacdc import fonts
from cellacdc.shared.dialogs import base as dialogs_base

class FilterObjsFromCoordsTable(NewThreadMultipleExpBaseUtil):
    def __init__(
            self, expPaths, app, title: str, infoText: str, 
            progressDialogueTitle: str, parent=None
        ):
        module = filesystem.get_module_name(__file__)
        super().__init__(
            expPaths, app, title, module, infoText, progressDialogueTitle, 
            parent=parent
        )
        self.expPaths = expPaths
    
    def runWorker(self):
        self.worker = segm_workers.FilterObjsFromCoordsTable(self)
        self.worker.sigAskAppendName.connect(self.askAppendName)
        self.worker.sigSetColumnsNames.connect(self.setColumnsNames)
        self.worker.sigAborted.connect(self.workerAborted)
        super().runWorker(self.worker)
    
    def showEvent(self, event):
        self.runWorker()
    
    def askAppendName(self, basename, existingEndnames):
        helpText = (
            """
            You can choose to save a new file for the filtered segmentation 
            or overwrite the existing one.
            """
        )
        win = dialogs_file_selection.filenameDialog(
            basename=basename,
            hintText='Insert a name for the <b>filtered segmentation</b> file:',
            existingNames=existingEndnames, 
            helpText=helpText, 
            allowEmpty=False,
            parent=self
        )
        win.exec_()
        if win.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return
        
        self.worker.appendedName = win.entryText
        self.worker.waitCond.wakeAll()
    
    def setColumnsNames(self, columns, categories, optionalCategories):
        win = SetColumnNamesDialog(
            columns, categories, optionalCategories=optionalCategories, 
            parent=self
        )
        win.exec_()
        if win.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return 
        
        self.selectedColumnsPerCategory = win.selectedColumns
        self.worker.waitCond.wakeAll()
    
    def workerAborted(self):
        self.workerFinished(None, aborted=True)
    
    def workerFinished(self, worker, aborted=False):
        if aborted:
            txt = 'Filter segmented objects from coordinates table process aborted.'
        else:
            txt = 'Filter segmented objects from coordinates table process completed.'
        self.logger.info(txt)
        msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()


class SetColumnNamesDialog(dialogs_base.QBaseDialog):
    def __init__(
            self, columnNames, categories, 
            optionalCategories=None, parent=None
        ):
        super().__init__(parent)
        
        if not optionalCategories:
            optionalCategories = None
        
        self.cancel = True
        
        mainLayout = QVBoxLayout()
        
        mainLayout.addWidget(QLabel(html_utils.paragraph(
            'Assign a column to the following categories:<br>'
        )))
        
        self.categoriesWidgets = {}
        formLayout = QFormLayout()
        for row, category in enumerate(categories):
            combobox = widgets_combo_boxes.ComboBox()
            combobox.addItems(columnNames)
            if optionalCategories is not None:
                text = f'* {category}'
            else:
                text = category
            formLayout.addRow(text, combobox)
            self.categoriesWidgets[category] = combobox
        
        if optionalCategories is not None:
            optionalItems = ['None', *columnNames]
            for row, category in enumerate(optionalCategories):
                combobox = widgets_combo_boxes.ComboBox()
                combobox.addItems(optionalItems)
                formLayout.addRow(category, combobox)
                self.categoriesWidgets[category] = combobox
        
        mainLayout.addLayout(formLayout)
        if optionalCategories is not None:
            mainLayout.addSpacing(10)
            mainLayout.addWidget(QLabel(html_utils.paragraph(
                '* mandatory', font_size='11px'
            )))
        
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)

        self.setFont(fonts.font)
    
    def _warnNonUniqueCategories(self, category_1, category_2):
        txt = html_utils.paragraph(f"""
            The following categories have the same column assigned to it.<br><br>
            Columns assigned to categories <b>must be unique</b>.<br><br>
            Categories with the same column:
            {html_utils.to_list((category_1, category_2))}
        """)
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.warning(self, 'Non-unique columns', txt)
    
    def _checkUniqueNames(self):
        self.textToCategoryMapper = {}
        for category, combobox in self.categoriesWidgets.items():
            if combobox.text() == 'None':
                continue
            
            if combobox.text() not in self.textToCategoryMapper:
                self.textToCategoryMapper[combobox.text()] = category
                continue
            
            sameCategory = self.textToCategoryMapper[combobox.text()]
            self._warnNonUniqueCategories(category, sameCategory)
            return False
        
        return True
    
    def ok_cb(self):
        proceed = self._checkUniqueNames()
        if not proceed:
            return
        
        self.selectedColumns = {
            category:combobox.text() 
            for category, combobox in self.categoriesWidgets.items() 
        }
        self.cancel = False
        self.close()
