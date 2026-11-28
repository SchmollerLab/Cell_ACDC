import os
from functools import partial

import pandas as pd

from .. import exception_handler
from .. import myutils, apps, widgets, html_utils, printl, workers
from ..utils import base

from qtpy.QtWidgets import QFileDialog
from qtpy.QtGui import QFontMetrics
from qtpy.QtWidgets import QListWidget
from natsort import natsorted
from qtpy.QtCore import Qt, QSize, QTimer
from qtpy.QtWidgets import QLabel, QWidget, QVBoxLayout, QGridLayout
from cellacdc import core
from cellacdc import io
from cellacdc import cca_functions
from cellacdc import fonts
from cellacdc.shared.dialogs import base as dialogs_base
from cellacdc.shared.dialogs import file_selection as dialogs_file_selection

class GenerateMothBudTotalUtil(base.MainThreadSinglePosUtilBase):
    def __init__(
            self, app, title: str, infoText: str, parent=None,
            callbackOnFinished=None
        ):
        module = myutils.get_module_name(__file__)
        super().__init__(
            app, title, module, infoText, parent
        )

        self.sigClose.connect(self.close)
        self.callbackOnFinished = callbackOnFinished

    @exception_handler
    def run(self):        
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        txt = html_utils.paragraph(
            'After clicking "Ok" you will be asked to <b>select the input table '
            'file</b> (.csv) containing <b>pedigree information</b>.'
        )
        msg.information(self, 'Instructions', txt)
        if msg.cancel:
            return False
        
        import qtpy.compat
        input_csv_filepath = qtpy.compat.getopenfilename(
            parent=self, 
            caption='Select CSV file to load', 
            filters='CSV (*.csv);;All Files (*)',
            basedir=myutils.getMostRecentPath()
        )[0]
        if input_csv_filepath is None or not input_csv_filepath:
            return False

        myutils.addToRecentPaths(os.path.dirname(input_csv_filepath))
        
        self.logger.info(f'Reading column names in table "{input_csv_filepath}"...')

        df = pd.read_csv(input_csv_filepath, nrows=2)

        win = apps.GenerateMotherBudTotalTableSelectColumnsDialog(
            df, parent=self
        )
        win.exec_()
        if win.cancel:
            return False
        
        selected_options = win.selected_options
        
        csv_filename = os.path.basename(input_csv_filepath)
        csv_filename_noext, ext = os.path.splitext(csv_filename)
        win = apps.filenameDialog(
            ext='.csv',
            basename=f'{csv_filename_noext}_',
            hintText='Insert a <b>filename for the output table</b> file:',
            allowEmpty=False, 
            defaultEntry='mother_bud_total', 
        )
        win.exec_()
        if win.cancel:
            return False

        out_csv_filename = win.filename
        out_csv_filepath = os.path.join(
            os.path.dirname(input_csv_filepath), out_csv_filename
        )
        
        self.worker = workers.GenerateMotherBudTotalTableWorker(
            self, input_csv_filepath, selected_options, out_csv_filepath
        )
        if self.callbackOnFinished is not None:
            self.worker.signals.finished.connect(self.callbackOnFinished)
        self.runWorker(self.worker)
        return True
        
    def overWriteClicked(self, win):
        win.cancel = False
        win.filename = ''
        win.close()
        


def get_min_width_for_no_scrollbar(list_widget: QListWidget) -> int:
    """
    Calculate the minimum width needed for the QListWidget
    so that the horizontal scrollbar will not be required.
    """
    font_metrics = QFontMetrics(list_widget.font())
    max_width = 0

    for i in range(list_widget.count()):
        item = list_widget.item(i)
        text_width = font_metrics.horizontalAdvance(item.text())
        max_width = max(max_width, text_width)

    # Add padding for icon, scrollbar margin, and frame
    padding = 30  # Adjust as needed (depends on style and icons)
    return max_width + padding


class GenerateMotherBudTotalTableSelectColumnsDialog(dialogs_base.QBaseDialog):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Select columns to combine into the output table')
        
        self.cancel = True
        
        self.columns = core.natsort_acdc_columns(df.columns)
        self.operations = (
            'Sum mother and bud', 
            'Copy column from mother',
        )
        
        self.mainLayout = QVBoxLayout()
        
        instructionsText = html_utils.paragraph("""
            <b>Select which columns</b> and <b>how</b> you want to combine them 
            into the output table.<br>
        """)
        self.mainLayout.addWidget(QLabel(instructionsText))
        
        settingsLayout = QGridLayout()    
        
        row = 0
        settingsLayout.addWidget(widgets.QHLine(), row, 0, 1, 2)  
        
        row += 1
        settingsLayout.addWidget(
            QLabel('Copy all non-selected columns from mother cell'), row, 0
        )
        self.copyAllColsToggle = widgets.Toggle()
        settingsLayout.addWidget(
            self.copyAllColsToggle, row, 1, alignment=Qt.AlignLeft
        )
        
        row += 1
        settingsLayout.addWidget(widgets.QHLine(), row, 0, 1, 2) 
        
        self.mainLayout.addLayout(settingsLayout)
        
        scrollArea = widgets.ScrollArea()
        scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scrollWidget = QWidget()
        scrollArea.setWidget(scrollWidget)
        self.centralLayout = QGridLayout()
        scrollWidget.setLayout(self.centralLayout)
        
        self.centralLayout.addWidget(QLabel('Grouping columns'), 0, 0)
        self.centralLayout.addWidget(QLabel('Column'), 0, 1)
        self.centralLayout.addWidget(QLabel('Operation'), 0, 2)
        self.centralLayout.setRowStretch(0, 0)
        
        self.groupingColsListWidget = widgets.listWidget(
            isMultipleSelection=True, 
        )
        self.groupingColsListWidget.addItems(self.columns)
        self.centralLayout.addWidget(self.groupingColsListWidget, 1, 0, 2, 1)
        
        selector = widgets.ComboBox(self)
        selector.addItems(self.columns)
        operationCombobox = widgets.ComboBox(self)
        operationCombobox.addItems(self.operations)
        self.addSelectorButton = widgets.addPushButton()
        
        dummyButton = widgets.delPushButton()
        dummyButton.setRetainSizeWhenHidden(True)
        dummyButton.hide()
        self.centralLayout.addWidget(dummyButton, 1, 4)
        
        self.centralLayout.addWidget(selector, 1, 1)
        self.centralLayout.addWidget(operationCombobox, 1, 2)
        self.centralLayout.addWidget(self.addSelectorButton, 1, 3)
        
        self.centralLayout.setRowStretch(1, 1)
        self.centralLayout.setRowStretch(2, 1)
        
        self.selectors = {1: (selector, operationCombobox)}
        
        buttonsLayout = widgets.CancelOkButtonsLayout()

        saveSelectionButton = widgets.savePushButton(
            'Save current selection'
        )
        buttonsLayout.insertWidget(3, saveSelectionButton)
        
        loadDefaultColsButton = widgets.reloadPushButton(
            'Load default summable columns'
        )
        buttonsLayout.insertWidget(4, loadDefaultColsButton)
        
        loadPreviousSelButton = widgets.OpenFilePushButton(
            'Load previous selection'
        )
        buttonsLayout.insertWidget(5, loadPreviousSelButton)
        
        saveSelectionButton.clicked.connect(self.saveSelection)
        loadDefaultColsButton.clicked.connect(self.loadDefaultCols)
        loadPreviousSelButton.clicked.connect(self.loadPreviousSelection)
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        self.mainLayout.addWidget(scrollArea)
        self.mainLayout.addSpacing(20)
        self.mainLayout.addLayout(buttonsLayout)
        
        self.addSelectorButton.clicked.connect(self.addSelector)
        selector.currentTextChanged.connect(
            self.selectorTextChanged
        )

        self.setLayout(self.mainLayout)
        self.setFont(fonts.font)
    
    def saveSelection(self):
        saved_selections = io.get_saved_moth_bud_tot_selections()
        existing_names = set(saved_selections.keys())
        win = dialogs_file_selection.filenameDialog(
            basename='',
            ext='',
            hintText='Insert a <b>name</b> for the current selection:',
            existingNames=existing_names,
            allowEmpty=False,
            defaultEntry='mother_bud_total_columns_selection'
        )
        win.exec_()
        if win.cancel:
            return
        
        name = win.filename
        saved_selections[name] = self.selectedOptions()
        io.save_moth_bud_tot_selected_options(saved_selections)
        
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        txt = html_utils.paragraph(f"""
            Current selection saved with name <code>{name}</code>.
        """)
        msg.information(self, 'Selection saved', txt)

    def loadDefaultCols(self):
        from cellacdc import single_pos_index_cols
        
        grouping_cols = [
            col for col in single_pos_index_cols if col in self.columns
        ]
        self.groupingColsListWidget.setSelectedItems(grouping_cols)
        
        column_operation_mapper = {
            col: 'Sum mother and bud' 
            for col in cca_functions.default_summable_columns
        }
        column_operation_mapper = {
            col: op for col, op in column_operation_mapper.items()
            if col in self.columns and op in self.operations
        }
        self.addSelectors(
            len(column_operation_mapper), 
            callback_on_finished=partial(
                self.setSelectorValues, column_operation_mapper
            )
        )

    def loadPreviousSelection(self):
        saved_selections = io.get_saved_moth_bud_tot_selections()
        if not saved_selections:
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            txt = html_utils.paragraph("""
                There are no saved selections.
            """)
            msg.warning(self, 'No saved selections', txt)
            return
        
        existing_names = natsorted(saved_selections.keys(), key=str.casefold)
        
        selectNameWin = widgets.QDialogListbox(
            'Choose selection to load',
            'Choose selection to load:\n',
            existing_names,
            multiSelection=False,
            parent=self,
            allowEmptySelection=False
        )
        selectNameWin.exec_()
        if selectNameWin.cancel:
            return
        
        self.loadOptions(saved_selections[selectNameWin.selectedItemsText[0]])
    
    def resetSelectors(self, callback_on_finished=None):
        self.callback_on_finished = callback_on_finished
        QTimer.singleShot(1, self._removeLastSelector)
    
    def _removeLastSelector(self):
        if len(self.selectors) == 1:
            if self.callback_on_finished is not None:
                self.callback_on_finished()
            return
        
        lastRow = max(self.selectors.keys())
        lastSelector, _ = self.selectors[lastRow]
        self.removeSelector(sender=lastSelector.delButton)
        QTimer.singleShot(1, self._removeLastSelector)
    
    def addSelectors(self, number, callback_on_finished=None):
        self.callback_on_finished = callback_on_finished
        QTimer.singleShot(1, partial(self._addSelectorRecursive, number))
    
    def _addSelectorRecursive(self, number):
        if len(self.selectors) == number:
            if self.callback_on_finished is not None:
                self.callback_on_finished()
            return

        self.addSelector()
        QTimer.singleShot(1, partial(self._addSelectorRecursive, number))
    
    def loadOptions(self, options: dict):
        if len(self.selectors) > 1:
            self.resetSelectors(
                callback_on_finished=partial(self.loadOptions, options)
            )
            return
        
        self.copyAllColsToggle.setChecked(
            options.get('do_copy_all_nonselected_columns', False)
        )
        self.groupingColsListWidget.setSelectedItems(
            options.get('grouping_columns', [])
        )
        column_operation_mapper = options.get('column_operation_mapper', {})
        column_operation_mapper = {
            col: op for col, op in column_operation_mapper.items()
            if col in self.columns and op in self.operations
        }
        if len(column_operation_mapper) > 1:
            self.addSelectors(
                len(column_operation_mapper), 
                callback_on_finished=partial(
                    self.setSelectorValues, column_operation_mapper
                )
            )
            return
        
        self.setSelectorValues(column_operation_mapper)
    
    def setSelectorValues(self, column_operation_mapper):
        for i, (col, op) in enumerate(column_operation_mapper.items()):
            selector, operationCombobox = self.selectors[i+1]
            selector.setCurrentText(col)
            operationCombobox.setCurrentText(op)                
    
    def resetSelectorsStyles(self):
        for selector, _ in self.selectors.values():
            selector.setStyleSheet('')
    
    def selectorTextChanged(self, text):
        self.resetSelectorsStyles()
        selector = self.sender()
        for other_selector, _ in self.selectors.values():
            if other_selector == selector:
                continue
            
            if selector.currentText() != other_selector.currentText():
                continue
            
            self.setWarningStyleSelector(selector)
            self.setWarningStyleSelector(other_selector)
            
    def addSelector(self):
        row = len(self.selectors) + 1
        
        selector = widgets.ComboBox(self)
        selector.addItems(self.columns)
        selector.setCurrentIndex(len(self.selectors))
        operationCombobox = widgets.ComboBox(self)
        operationCombobox.addItems(self.operations)
        delButton = widgets.delPushButton()
        selector.delButton = delButton
        delButton._row = row
        
        self.selectors[row] = (selector, operationCombobox)
        
        self.centralLayout.addWidget(selector, row, 1)
        self.centralLayout.addWidget(operationCombobox, row, 2)
        self.centralLayout.addWidget(delButton, row, 3)
        
        self.centralLayout.removeWidget(self.addSelectorButton)
        self.centralLayout.addWidget(self.addSelectorButton, row, 4)
        
        delButton.clicked.connect(self.removeSelector)
        
        self.centralLayout.removeWidget(self.groupingColsListWidget)
        rowSpan = self.centralLayout.rowCount()
        self.centralLayout.addWidget(
            self.groupingColsListWidget, 1, 0, rowSpan, 1
        )
        self.centralLayout.setRowStretch(rowSpan, 1)
        
        selector.currentTextChanged.connect(
            self.selectorTextChanged
        )
    
    def removeSelector(self, checked=False, sender=None):
        if sender is None:
            delButton = self.sender()
        else:
            delButton = sender
            
        selector, operationCombobox = self.selectors.pop(delButton._row)
        
        self.centralLayout.removeWidget(selector)
        self.centralLayout.removeWidget(operationCombobox)
        self.centralLayout.removeWidget(delButton)
        
        resorted_selectors = {}
        for i, (row, (sel, op)) in enumerate(self.selectors.items()):
            if i == 0:
                resorted_selectors[i+1] = (sel, op)
                continue
            
            delButton = sel.delButton
            delButton._row = i+1
            self.centralLayout.removeWidget(sel)
            self.centralLayout.removeWidget(op)
            self.centralLayout.removeWidget(delButton)
            self.centralLayout.addWidget(sel, i+1, 1)
            self.centralLayout.addWidget(op, i+1, 2)
            self.centralLayout.addWidget(delButton, i+1, 3)
            
            resorted_selectors[i+1] = (sel, op)
        
        last_row = i+1
        col = 4 if last_row > 1 else 3
        self.centralLayout.removeWidget(self.addSelectorButton)
        self.centralLayout.addWidget(self.addSelectorButton, i+1, col)
        
        self.selectors = resorted_selectors
    
    def sizeHint(self):
        width = super().sizeHint().width()
        height = super().sizeHint().height()
        groupingColsWidth = widgets.get_min_width_for_no_scrollbar(
            self.groupingColsListWidget
        )
        width += groupingColsWidth
        return QSize(width, height)
    
    def checkDuplicatedSelectedColumns(self):
        for selector, _ in self.selectors.values():
            selector.setStyleSheet('background-color: none')
            for other_selector, _ in self.selectors.values():
                if other_selector == selector:
                    continue
                
                if other_selector.currentText() != selector.currentText():
                    continue
                
                self.warnDuplicatedSelectedColumns(selector, other_selector)
                return False
        
        return True
    
    def setWarningStyleSelector(self, selector):
        popup = selector.view()
        palette = popup.palette()
        text_color = palette.color(palette.ColorRole.Text)
        warningStyleSheet = (f"""
            QComboBox {{
                color: black;
                background-color: orange;  /* main area */
            }}
            QComboBox QAbstractItemView {{
                background-color: {text_color.name()};
            }}
        """)
        selector.setStyleSheet(warningStyleSheet)
    
    def warnDuplicatedSelectedColumns(self, selector1, selector2):
        self.setWarningStyleSelector(selector1)
        self.setWarningStyleSelector(selector2)
        
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        txt = html_utils.paragraph(f"""
            The following column has been selected more than once 
            (highlighted in orange).<br><br>
            <code>{selector1.currentText()}</code><br><br>
            Please, select each column only once.<br><br>
            Thank you for your patience!
        """)
        msg.warning(self, 'Duplicated selection', txt)
    
    
    def checkGroupingColumnsNotSelected(self):
        if self.groupingColsListWidget.selectedItems():
            return True
        
        return self.warnGroupingColumnsNotSelected()
    
    def warnGroupingColumnsNotSelected(self):
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        txt = html_utils.paragraph(f"""
            Are you sure you do not want to select any grouping column?<br><br>
            Grouping columns are those needed to identify each unique 
            Position folder.
        """)
        _, noButton, yesButton = msg.question(
            self, 'No grouping columns selected?', txt,
            buttonsTexts=(
                'Cancel', 
                'No, let me select grouping columns', 
                'Yes, I do not need grouping columns'
            )
        )
        return msg.clickedButton == yesButton
    
    def selectedOptions(self):
        selected_options = {
            'grouping_columns': self.groupingColsListWidget.selectedItemsText(),
            'column_operation_mapper': {
                selector.currentText(): operationCombobox.currentText()
                for selector, operationCombobox in self.selectors.values()
            },
            'do_copy_all_nonselected_columns': self.copyAllColsToggle.isChecked()
        }
        return selected_options
    
    def ok_cb(self):
        proceed = self.checkDuplicatedSelectedColumns()
        if not proceed:
            return
        
        proceed = self.checkGroupingColumnsNotSelected()
        if not proceed:
            return
        
        self.selected_options = self.selectedOptions()
        
        self.cancel = False
        self.close()
