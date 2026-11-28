import pandas as pd

from .. import apps, myutils, workers, widgets, html_utils, load

from .base import NewThreadMultipleExpBaseUtil
import os
import re
from qtpy.QtGui import QColor
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QFileDialog, QLabel, QVBoxLayout, QHBoxLayout, QAbstractItemView,
    QGridLayout, QGroupBox
)
from cellacdc import config
from cellacdc import _palettes
from cellacdc import fonts
from cellacdc.shared.dialogs import base as dialogs_base
from cellacdc.shared.dialogs import combine_measurements as dialogs_combine_measurements

class ComputeMetricsMultiChannel(NewThreadMultipleExpBaseUtil):
    def __init__(
            self, expPaths, app, title: str, infoText: str, 
            progressDialogueTitle: str, parent=None
        ):
        module = myutils.get_module_name(__file__)
        super().__init__(
            expPaths, app, title, module, infoText, progressDialogueTitle, 
            parent=parent
        )
        self.expPaths = expPaths
    
    def runWorker(self):
        self.worker = workers.ComputeMetricsMultiChannelWorker(self)
        self.worker.sigAskAppendName.connect(self.askAppendName)
        self.worker.sigCriticalNotEnoughSegmFiles.connect(
            self.criticalNotEnoughSegmFiles
        )
        self.worker.sigHowCombineMetrics.connect(self.showHowCombineMetrics)
        self.worker.sigAborted.connect(self.workerAborted)
        super().runWorker(self.worker)
    
    def showEvent(self, event):
        self.runWorker()
    
    def showHowCombineMetrics(
            self, imagesPath, selectedAcdcOutputEndnames, 
            existingAcdcOutputEndnames, allChNames
        ):
        self.imagesPath = imagesPath
        self.existingAcdcOutputEndnames = existingAcdcOutputEndnames
        acdcDfsDict = {}
        for endname in selectedAcdcOutputEndnames:
            filePath, _ = load.get_path_from_endname(endname, imagesPath)
            acdc_df = pd.read_csv(filePath)
            acdcDfsDict[endname] = acdc_df

        self.combineWindow = apps.CombineMetricsMultiDfsSummaryDialog(
            acdcDfsDict, allChNames, parent=self
        )
        self.combineWindow.setLogger(
            self.logger, self.logs_path, self.log_path
        )
        self.combineWindow.sigLoadAdditionalAcdcDf.connect(
            self.loadAdditionalAcdcDf
        )
        self.combineWindow.exec_()
        if self.combineWindow.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return

        self.worker.equations = self.combineWindow.equations
        self.worker.acdcDfs = self.combineWindow.acdcDfs
        self.worker.waitCond.wakeAll()
    
    def loadAdditionalAcdcDf(self):
        selectWindow = widgets.QDialogListbox(
            'Select acdc_output files',
            f'Select acdc_output files to load\n',
            self.existingAcdcOutputEndnames, multiSelection=True, 
            parent=self, allowSingleSelection=True
        )
        selectWindow.exec_()
        if selectWindow.cancel or not selectWindow.selectedItemsText:
            self.logger.info('Loading additional tables cancelled.')
            return
        
        acdcDfsDict = {}
        for end in selectWindow.selectedItemsText:
            filePath, _ = load.get_path_from_endname(end, self.imagesPath)
            acdc_df = pd.read_csv(filePath)
            acdcDfsDict[end] = acdc_df
            
        self.combineWindow.addAcdcDfs(acdcDfsDict)

    def criticalNotEnoughSegmFiles(self, exp_path):
        text = html_utils.paragraph(f"""
            The following experiment folder<br><br>
            <code>{exp_path}</code><br><br>
            <b>does NOT contain AT LEAST TWO <code>acdc_output</code> 
            table files.</b>.
        """)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        msg.addShowInFileManagerButton(exp_path)
        msg.critical(
            self, 'Not enough segmentation files!', text
        )
        self.worker.abort = True
        self.worker.waitCond.wakeAll()
    
    def askAppendName(self, basename, existingEndnames, selectedEndnames):
        helpText = (
            """
            The CSV table file with the combined measurements 
            will be saved with a different file name.<br><br>
            Insert a name to append to the end of the new name. The rest of 
            the name will have the same basename as all other files.
            """
        )
        channels = [end.replace('acdc_output_', '') for end in selectedEndnames]
        channels = [end.replace('acdc_output', '') for end in channels]
        channels = [end if end else 'refCh' for end in channels]
        defaultEntry = f"{'_'.join(channels)}_combined_metrics"
        win = apps.filenameDialog(
            basename=basename,
            hintText='Insert a name for the <b>new, table</b> file:',
            existingNames=existingEndnames, 
            helpText=helpText, 
            allowEmpty=False,
            ext='.csv',
            defaultEntry=defaultEntry,
            resizeOnShow=True
        )
        win.exec_()
        if win.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return
        
        self.worker.appendedName = win.entryText
        self.worker.waitCond.wakeAll()
    
    def workerAborted(self):
        self.workerFinished(None, aborted=True)
    
    def workerCritical(self, error):
        super().workerCritical(error)
        self.worker.errors[error] = self.traceback_str
    
    def warnErrors(self, errors):
        win = apps.ComputeMetricsErrorsDialog(
            errors, self.logs_path, log_type='generic', parent=self
        )
        win.exec_()
    
    def workerFinished(self, worker, aborted=False):
        if aborted:
            txt = 'Combining multiple channels measurements aborted.'
            isWarning = True
        elif worker.errors:
            txt = 'Combining multiple channels measurements completed WITH ERRORS.'
            self.warnErrors(worker.errors)
            isWarning = True
        else:
            txt = html_utils.paragraph(
                'Combining multiple channels measurements completed.<br><br>'
                'Results were saved in the respective Position folder(s).'
            )
            isWarning = False
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        if isWarning:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()


BACKGROUND_RGBA = _palettes.get_disabled_colors()['Button']


class MultiListSelector(dialogs_base.QBaseDialog):
    def __init__(
            self, lists: dict, groupsDescr: dict=None, 
            title='Lists selector', infoTxt='', parent=None
        ):
        super().__init__(parent)

        self.setWindowTitle(title)
        
        self.cancel = True
        mainLayout = QVBoxLayout()

        if infoTxt:
            mainLayout.addWidget(QLabel(html_utils.paragraph(infoTxt)))

        self.listWidgets = {}
        createdGroupLayouts = {}
        for listName, listItems in lists.items():
            if groupsDescr is None:
                groupName = ''
            else:
                groupName = groupsDescr.get(listName, 'Group info missing')
            groupLayout = createdGroupLayouts.get(listName, None)
            if groupLayout is None:
                mainLayout.addWidget(QLabel(html_utils.paragraph(groupName)))
                groupBox = QGroupBox()
                mainLayout.addWidget(groupBox)
                groupLayout = QVBoxLayout()
                groupBox.setLayout(groupLayout)
                createdGroupLayouts[groupName] = groupLayout
            else:
                groupLayout.addSpacing(10)
            groupLayout.addWidget(QLabel(html_utils.paragraph(listName)))
            listWidget = widgets.listWidget()
            listWidget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            listWidget.addItems(listItems)
            groupLayout.addWidget(listWidget)
            mainLayout.addSpacing(20)
            self.listWidgets[listName] = listWidget
        
        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addSpacing(10)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)
    
    def ok_cb(self):
        self.cancel = False
        self.selectedItems = {}
        for listName, listWidget in self.listWidgets.items():
            if not listWidget.selectedItems():
                continue
            self.selectedItems[listName] = [
                item.text() for item in listWidget.selectedItems()
            ]
        self.close()


class CombineMetricsMultiDfsDialog(dialogs_combine_measurements._CombineMetricsDialog):
    sigOk = Signal(object, object)
    sigClose = Signal(bool)

    def __init__(self, acdcDfs, allChNames, parent=None, debug=False):
        super().__init__(
            allChNames, False, False, parent=parent, debug=debug,
            acdcDfs=acdcDfs
        )
    
    def setLogger(self, logger, logs_path, log_path):
        self.logger = logger
        self.logs_path = logs_path
        self.log_path = log_path
    
    def closeEvent(self, event):
        self.sigClose.emit(self.cancel)
        return super().closeEvent(event)
    
    def getCombinedDf(self):
        return self.getTestData()

    def ok_cb(self):
        if not self.newColNameLineEdit.text():
            self.warnEmptyEquationName()
            return
        if not self.okButton.isEnabled():
            self.test_cb()
            return

        self.expression = self.equationDisplay.toPlainText()
        self.newColname = self.newColNameLineEdit.text()
        self.cancel = False
        self.sigOk.emit(self.newColname, self.expression)
        self.close()

class CombineMetricsMultiDfsSummaryDialog(dialogs_base.QBaseDialog):
    sigLoadAdditionalAcdcDf = Signal()

    def __init__(
            self, acdcDfs, allChNames, parent=None, debug=False
        ):
        super().__init__(parent)

        self.editedIndex = None
        self.cancel = True
        self.acdcDfs = acdcDfs
        self.allChNames = allChNames

        self.setWindowTitle('Combine measurements summary')
        
        mainLayout = QVBoxLayout()
        viewLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        row = 0
        txt = html_utils.paragraph('Selected acdc_output tables:')
        viewLayout.addWidget(QLabel(txt), row, 0)

        row += 1
        items = [
            f'• <b>Table {i+1}</b>: <code>{e}</code>' 
            for i, e in enumerate(acdcDfs.keys())
        ]
        selectedAcdcDfsList = widgets.readOnlyQList()
        selectedAcdcDfsList.addItems(items)
        self.selectedAcdcDfsList = selectedAcdcDfsList

        tablesButtonsLayout = QVBoxLayout()
        loadAcdcDfButton = widgets.showInFileManagerButton('Load additional tables')
        tablesButtonsLayout.addWidget(loadAcdcDfButton)
        
        loadEquationsButton = widgets.reloadPushButton(
            'Load previously used equations'
        )
        tablesButtonsLayout.addWidget(loadEquationsButton)
        

        tablesButtonsLayout.addStretch(1)

        viewLayout.addWidget(selectedAcdcDfsList, row, 0)
        viewLayout.addLayout(tablesButtonsLayout, row, 1)
        viewLayout.setRowStretch(row, 1)

        row += 1
        txt = html_utils.paragraph('Equations:')
        viewLayout.addWidget(QLabel(txt), row, 0)

        row += 1
        self.equationsList = widgets.TreeWidget()
        self.equationsList.setFont(fonts.font)
        self.equationsList.setHeaderLabels(['Metric', 'Expression'])
        self.equationsList.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)

        equationsButtonsLayout = QVBoxLayout()
        addEquationButton = widgets.addPushButton('Add metric')
        removeEquationButton = widgets.subtractPushButton('Remove metric(s)')
        editEquationButton = widgets.editPushButton('Edit metric')
        removeEquationButton.setDisabled(True)
        editEquationButton.setDisabled(True)
        self.removeEquationButton = removeEquationButton
        self.editEquationButton = editEquationButton

        equationsButtonsLayout.addWidget(addEquationButton)
        equationsButtonsLayout.addWidget(removeEquationButton)
        equationsButtonsLayout.addWidget(editEquationButton)
        equationsButtonsLayout.addStretch(1)
        
        viewLayout.addWidget(self.equationsList, row, 0)
        viewLayout.addLayout(equationsButtonsLayout, row, 1)
        viewLayout.setRowStretch(row, 2)

        cancelButton = widgets.cancelPushButton('Cancel')
        okButton = widgets.okPushButton('Ok')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        viewLayout.setVerticalSpacing(10)
        mainLayout.addLayout(viewLayout)
        mainLayout.addSpacing(10)
        mainLayout.addLayout(buttonsLayout)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        addEquationButton.clicked.connect(self.addEquation_cb)
        loadAcdcDfButton.clicked.connect(self.loadButtonClicked)
        loadEquationsButton.clicked.connect(self.loadEquationsButtonClicked)
        removeEquationButton.clicked.connect(self.removeButtonClicked)
        editEquationButton.clicked.connect(self.editButtonClicked)
        self.equationsList.itemSelectionChanged.connect(
            self.onEquationItemSelectionChanged
        )

        self.setLayout(mainLayout)
    
    def setLogger(self, logger, logs_path, log_path):
        self.logger = logger
        self.logs_path = logs_path
        self.log_path = log_path
    
    def loadEquationsButtonClicked(self):
        MostRecentPath = myutils.getMostRecentPath()
        file_path = QFileDialog.getOpenFileName(
            self, 'Select equations file', MostRecentPath, "Config Files (*.ini)"
            ";;All Files (*)")[0]
        if file_path == '':
            return

        cp = config.ConfigParser()
        cp.read(file_path)
        sectionToMatch = [
            f'table{i+1}:{end}' for i, end in enumerate(self.acdcDfs)
        ]
        sectionToMatch = ';'.join(sectionToMatch)
        
        lists = {}
        nonMatchingLists = {}
        groupsDescr = {}
        
        for section in cp.sections():
            # Tag acdc_output names with <code> html and table(\d+) with html bold tag
            listName = ';'.join([
                re.sub(r'table(\d+):(.*)', r'<b>table\g<1></b>: <code>\g<2></code>', s) 
                for s in section.split(';')
            ])
            listName = listName.replace(';', ' ; ')
            children = [f'{opt} = {cp[section][opt]}' for opt in cp[section]]
            if section == sectionToMatch:
                groupsDescr[listName] = (
                    'Equations that were calculated from the <b>same '
                    'table names</b> you loaded'
                )
                lists[listName] = children
            else:
                groupsDescr[listName] = (
                    'Equations that were calculated from <b>table names that '
                    'you did not load</b> now'
                )
                nonMatchingLists[listName] = children
                # # Not implemented yet --> selecting from non matching table names
                # # would require an additional widget where the user sets 
                # # what df1 and df2 are.
                # trees[treeName] = children

        if not lists:
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            txt = html_utils.paragraph("""
                <b>None of the equations</b> in the selected file used the <b>same 
                table names</b> that you loaded.<br><br>
                See below which table names and equations are present in the loaded file.
            """)
            with open(file_path) as iniFile:
                detailedText = iniFile.read()
            
            msg.warning(self, 'Not the same tables', txt, showDialog=False)
            msg.setDetailedText(detailedText, visible=True)
            msg.addShowInFileManagerButton(os.path.dirname(file_path))
            msg.exec_()
            return

        selectWindow = MultiListSelector(
            lists, groupsDescr=groupsDescr, title='Select equations to load',
            infoTxt='Select equations you want to load'
        )
        selectWindow.exec_()
        if selectWindow.cancel or not selectWindow.selectedItems:
            return

        for listName, equations in selectWindow.selectedItems.items():
            for equation in equations:
                metricName, expression = equation.split(' = ')
                self.addEquation(metricName, expression)
        

    def ok_cb(self):
        self.cancel = False
        self.equations = {}
        for i in range(self.equationsList.topLevelItemCount()):
            item = self.equationsList.topLevelItem(i)
            self.equations[item.text(0)] = item.text(1)

        self.close()
    
    def loadButtonClicked(self):
        self.sigLoadAdditionalAcdcDf.emit()
    
    def removeButtonClicked(self):
        for item in self.equationsList.selectedItems():
            self.equationsList.invisibleRootItem().removeChild(item)
        
    def editButtonClicked(self):
        self.editedItem = self.equationsList.selectedItems()[0]
        self.editedIndex = self.equationsList.indexOfTopLevelItem(self.editedItem)
        self.addEquation_cb()
    
    def onEquationItemSelectionChanged(self):
        selectedItems = self.equationsList.selectedItems()
        if len(selectedItems) == 1:
            self.editEquationButton.setDisabled(False)
            self.removeEquationButton.setDisabled(False)
        elif len(selectedItems) > 1:
            self.removeEquationButton.setDisabled(False)
            self.editEquationButton.setDisabled(True)
        else:
            self.removeEquationButton.setDisabled(True)
            self.editEquationButton.setDisabled(True)
    
    def addAcdcDfs(self, acdcDfsDict):
        self.acdcDfs = {**self.acdcDfs, **acdcDfsDict}
        items = [
            f'• <b>Table {i+1}</b>: <code>{e}</code>' 
            for i, e in enumerate(self.acdcDfs.keys())
        ]
        self.selectedAcdcDfsList.clear()
        self.selectedAcdcDfsList.addItems(items)

    def addEquation(self, newColname, expression):
        if self.editedIndex is not None:
            self.equationsList.invisibleRootItem().removeChild(self.editedItem)
        bkgrColor = QColor(*BACKGROUND_RGBA[:3], 200)
        item = widgets.TreeWidgetItem(
            self.equationsList, columnColors=[None, bkgrColor]
        )
        item.setText(0, newColname)
        item.setText(1, expression)
        if self.editedIndex is not None:
            self.equationsList.insertTopLevelItem(self.editedIndex, item)
        else:
            self.equationsList.addTopLevelItem(item)
        self.equationsList.resizeColumnToContents(0)
        self.equationsList.resizeColumnToContents(1)
        self.editedIndex = None
    
    def addEquation_cb(self):
        self.addEquationWin = CombineMetricsMultiDfsDialog(
            self.acdcDfs, self.allChNames, parent=self
        )
        if hasattr(self, 'logger'):
            self.addEquationWin.setLogger(
                self.logger, self.logs_path, self.log_path
            )
        if self.editedIndex is not None:
            editedMetricName = self.editedItem.text(0)
            self.addEquationWin.newColNameLineEdit.setText(editedMetricName)
            editedExpression = self.editedItem.text(1)
            self.addEquationWin.equationDisplay.setPlainText(editedExpression)
        self.addEquationWin.show()
        self.addEquationWin.sigOk.connect(self.addEquation)
        self.addEquationWin.sigClose.connect(self.addEquationClosed)
    
    def addEquationClosed(self, cancelled):
        if cancelled:
            self.editedIndex = None
    
    def showEvent(self, event) -> None:
        self.resize(int(self.width()*2), self.height())
