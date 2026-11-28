"""Main-GUI-only Add points layer dialog AddPointsLayerDialog (from
apps.py). Its appearance group box comes from
shared/dialogs/image_tools.py (shared with the volume renderer). Keys
chosen here are checked against guiWin.reservedShortcuts.
"""
from qtpy.QtCore import QRegularExpression
from qtpy.QtGui import QRegularExpressionValidator
from qtpy.QtWidgets import QLineEdit
import os
import re
import numpy as np
import traceback
import pandas as pd
import html
from qtpy.QtGui import QKeySequence
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QLabel, QVBoxLayout, QHBoxLayout, QCheckBox, QGridLayout, QGroupBox,
    QRadioButton, QSpacerItem
)
from cellacdc import _window_stacking
from cellacdc import load, html_utils
from cellacdc import widgets
from cellacdc import fonts
from cellacdc.shared.dialogs import base as dialogs_base
from cellacdc.shared.dialogs import image_tools as dialogs_image_tools


class NumericCommaLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.validPattern = r'^[0-9,\.]+$'
        regExp = QRegularExpression(self.validPattern)
        self.setValidator(QRegularExpressionValidator(regExp))
    
    def values(self):
        try:
            vals = [float(c) for c in self.text().split(',')]
        except Exception as e:
            vals = []
        return vals


class AddPointsLayerDialog(dialogs_base.QBaseDialog):
    sigClosed = Signal()
    sigCriticalReadTable = Signal(str)
    sigLoadedTable = Signal(object, str)
    sigCheckClickEntryTableEndnameExists = Signal(str, bool)

    def __init__(
            self, 
            channelNames=None, 
            imagesPath='', 
            SizeT=1, 
            hideCentroidsSection=False,
            hideWeightedCentroidsSection=False,
            hideFromTableSection=False,
            hideManualEntrySection=False,
            hideWithMouseClicksSection=False,
            parent=None,
            reservedShortcuts=None,
        ):
        """`reservedShortcuts` : {key sequence text: used by} of shortcuts
        that cannot be used (refused on OK, the dialog stays open), or a
        callable returning it, called on OK (the dialog is not modal, so
        keys can be taken while it is open)."""
        self.cancel = True
        super().__init__(parent)

        self._parent = parent
        self._reservedShortcuts = reservedShortcuts

        self.imagesPath = imagesPath

        self.setWindowTitle('Add points layer')
        _window_stacking.set_window_flags_on_top(self, Qt.Window)

        mainLayout = QVBoxLayout()

        scrollArea = widgets.ScrollArea()
        typeGroupbox = QGroupBox('Points to draw')
        typeLayout = QGridLayout()
        typeGroupbox.setLayout(typeLayout)
        typeLayout.addItem(QSpacerItem(10,1), 0, 0)
        typeLayout.setColumnStretch(0, 0)
        typeLayout.setColumnStretch(2, 1)
        vSpacing = 15
        
        row = 0
        
        sections = (
            ('addCentroidsSection', hideCentroidsSection),
            ('addWeightedCentroidsSection', hideWeightedCentroidsSection),
            ('addFromTableSection', hideFromTableSection),
            ('addManualEntrySection', hideManualEntrySection),
            ('addWithMouseClicksSection', hideWithMouseClicksSection)
        )
        radioButtonChecked = False
        for section, hideSection in sections:
            addFunc = getattr(self, section)
            row, sectionWidgets = addFunc(
                row, typeLayout, 
                imagesPath=imagesPath, 
                SizeT=SizeT,
                channelNames=channelNames
            )
            if not hideSection:
                spacer = QSpacerItem(1, vSpacing)
                typeLayout.addItem(spacer, row, 0)
                row += 1
                if not radioButtonChecked:
                    sectionWidgets[0].setChecked(True)
                    radioButtonChecked = True
                continue
            
            for widget in sectionWidgets:
                widget.setVisible(False)

        self.scrollArea = scrollArea
        scrollArea.setWidget(typeGroupbox)

        self.appearanceGroupbox = dialogs_image_tools._PointsLayerAppearanceGroupbox()
        self.appearanceGroupbox.sizeSpinBox.setValue(3)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        self.buttonsLayout = buttonsLayout

        mainLayout.addWidget(scrollArea)
        mainLayout.addSpacing(20)
        _layout = QHBoxLayout()
        _layout.addWidget(self.appearanceGroupbox)
        _layout.addStretch(1)
        mainLayout.addLayout(_layout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        self.setFont(fonts.font)
    
    def addCentroidsSection(self, row, layout, **kwargs):
        sectionWidgets = []
        self.centroidsRadiobutton = QRadioButton('Centroids')
        layout.addWidget(self.centroidsRadiobutton, row, 0, 1, 2)
        sectionWidgets.append(self.centroidsRadiobutton)
        
        self.centroidsRadiobutton.setChecked(True)
        return row + 1, sectionWidgets
    
    def addWeightedCentroidsSection(
            self, row, layout, channelNames=None, **kwargs
        ):
        if channelNames is None:
            channelNames = []
            
        sectionWidgets = []
        
        self.weightedCentroidsRadiobutton = QRadioButton('Weighted centroids')
        layout.addWidget(self.weightedCentroidsRadiobutton, row, 0, 1, 2)
        sectionWidgets.append(self.weightedCentroidsRadiobutton)

        row += 1
        label = QLabel('Weighing channel: ')
        label.setEnabled(False)
        layout.addWidget(label, row, 1)
        sectionWidgets.append(label)
        
        self.channelNameForWeightedCentr = widgets.QCenteredComboBox()
        if channelNames:
            self.channelNameForWeightedCentr.addItems(channelNames)
        self.channelNameForWeightedCentr.setDisabled(True)
        layout.addWidget(self.channelNameForWeightedCentr, row, 2)
        sectionWidgets.append(self.channelNameForWeightedCentr)

        self.weightedCentroidsRadiobutton.toggled.connect(label.setEnabled)
        self.weightedCentroidsRadiobutton.toggled.connect(
            self.channelNameForWeightedCentr.setEnabled
        )
        
        return row + 1, sectionWidgets
    
    def addFromTableSection(
            self, row, layout, imagesPath='', SizeT=1, **kwargs
        ):
        sectionWidgets = []
        
        self.fromTableRadiobutton = QRadioButton('From table')
        layout.addWidget(self.fromTableRadiobutton, row, 0, 1, 2)
        sectionWidgets.append(self.fromTableRadiobutton)
        self.fromTableRadiobutton.widgets = []
        
        row += 1
        self.tablePath = widgets.ElidingLineEdit()
        self.tablePath.label = QLabel('Table file path: ')
        layout.addWidget(self.tablePath.label, row, 1)
        layout.addWidget(self.tablePath, row, 2)
        self.fromTableRadiobutton.widgets.append(self.tablePath)
        sectionWidgets.append(self.tablePath.label)
        sectionWidgets.append(self.tablePath)

        browseButton = widgets.browseFileButton(
            start_dir=imagesPath, ext={'Table': ['.csv', '.h5']}
        )
        layout.addWidget(browseButton, row, 3)
        browseButton.sigPathSelected.connect(self.tablePathSelected)
        self.browseTableButton = browseButton
        self.fromTableRadiobutton.widgets.append(browseButton)
        sectionWidgets.append(browseButton)

        row += 1
        self.xColName = widgets.QCenteredComboBox()
        self.xColName.addItem('None')
        self.xColName.label = QLabel('X coord. column: ')
        layout.addWidget(self.xColName.label, row, 1)
        layout.addWidget(self.xColName, row, 2)
        self.xColName.currentTextChanged.connect(self.checkColNameX)
        self.fromTableRadiobutton.widgets.append(self.xColName)
        sectionWidgets.append(self.xColName.label)
        sectionWidgets.append(self.xColName)

        row += 1
        self.yColName = widgets.QCenteredComboBox()
        self.yColName.addItem('None')
        self.yColName.label = QLabel('Y coord. column: ')
        layout.addWidget(self.yColName.label, row, 1)
        layout.addWidget(self.yColName, row, 2)
        self.yColName.currentTextChanged.connect(self.checkColNameY)
        self.fromTableRadiobutton.widgets.append(self.yColName)
        sectionWidgets.append(self.yColName.label)
        sectionWidgets.append(self.yColName)

        row += 1
        self.zColName = widgets.QCenteredComboBox()
        self.zColName.addItem('None')
        self.zColName.label = QLabel('Z coord. column: ')
        layout.addWidget(self.zColName.label, row, 1)
        layout.addWidget(self.zColName, row, 2)
        self.zColName.currentTextChanged.connect(self.checkColNameZ)
        self.fromTableRadiobutton.widgets.append(self.zColName)
        sectionWidgets.append(self.zColName.label)
        sectionWidgets.append(self.zColName)

        row += 1
        self.tColName = widgets.QCenteredComboBox()
        self.tColName.addItem('None')
        self.tColName.label = QLabel('Frame index column: ')
        self.tColRequiresGroupingCheckbox = QCheckBox('Group by unique values')
        layout.addWidget(self.tColName.label, row, 1)
        layout.addWidget(self.tColName, row, 2)
        layout.addWidget(self.tColRequiresGroupingCheckbox, row, 4)
        self.fromTableRadiobutton.widgets.append(self.tColName)
        self.fromTableRadiobutton.widgets.append(
            self.tColRequiresGroupingCheckbox
        )
        sectionWidgets.append(self.tColName.label)
        sectionWidgets.append(self.tColName)
        sectionWidgets.append(self.tColRequiresGroupingCheckbox)

        tColNameTooltip = (
'Select the column containing the frame index (starting from 0).\n\n'
'Alternatively, you can select any column whose values identify the rows '
'belonging to each frame.\n'
'Each unique value is treated as a separate frame and assigned a frame index according to its order in the table.\n\n'
'In the latter case, make sure to check the "Group by unique values" checkbox.'
        )
        self.tColName.label.setToolTip(tColNameTooltip)
        self.tColName.setToolTip(tColNameTooltip)

        tColGroupCheckboxTooltip = (
'Check this box if the selected column does not contain frame indices, '
'but rather contains values that identify the rows belonging to each frame.\n\n'
'Each unique value will be treated as a separate frame and assigned a frame index according to its order in the table.'
        )
        self.tColRequiresGroupingCheckbox.setToolTip(tColGroupCheckboxTooltip)

        if SizeT == 1:
            self.tColName.clear()
            self.tColName.addItem('None')
            self.tColName.label.setVisible(False)
            self.tColName.setVisible(False)
            self.tColRequiresGroupingCheckbox.setVisible(False)
        
        self.fromTableRadiobutton.toggled.connect(self.enableRadioButtonWidgets)
        self.enableRadioButtonWidgets(False, sender=self.fromTableRadiobutton)
        
        return row + 1, sectionWidgets
    
    def addManualEntrySection(self, row, layout, SizeT=1, **kwargs):
        sectionWidgets = []
        
        self.manualEntryRadiobutton = QRadioButton('Manual entry')
        layout.addWidget(self.manualEntryRadiobutton, row, 0, 1, 2)
        self.manualEntryRadiobutton.widgets = []
        sectionWidgets.append(self.manualEntryRadiobutton)
        
        row += 1
        self.manualXspinbox = widgets.NumericCommaLineEdit()
        self.manualXspinbox.label = QLabel('X coords: ')
        layout.addWidget(self.manualXspinbox.label, row, 1)
        layout.addWidget(self.manualXspinbox, row, 2)
        self.manualEntryRadiobutton.widgets.append(self.manualXspinbox)
        sectionWidgets.append(self.manualXspinbox.label)
        sectionWidgets.append(self.manualXspinbox)

        row += 1
        self.manualYspinbox = widgets.NumericCommaLineEdit()
        self.manualYspinbox.label = QLabel('Y coords: ')
        layout.addWidget(self.manualYspinbox.label, row, 1)
        layout.addWidget(self.manualYspinbox, row, 2)
        self.manualEntryRadiobutton.widgets.append(self.manualYspinbox)
        sectionWidgets.append(self.manualYspinbox.label)
        sectionWidgets.append(self.manualYspinbox)

        row += 1
        self.manualZspinbox = widgets.NumericCommaLineEdit()
        self.manualZspinbox.label = QLabel('Z coords: ')
        layout.addWidget(self.manualZspinbox.label, row, 1)
        layout.addWidget(self.manualZspinbox, row, 2)
        self.manualEntryRadiobutton.widgets.append(self.manualZspinbox)
        sectionWidgets.append(self.manualZspinbox.label)
        sectionWidgets.append(self.manualZspinbox)

        row += 1
        self.manualTspinbox = widgets.NumericCommaLineEdit()
        self.manualTspinbox.label = QLabel('Frame numbers: ')
        layout.addWidget(self.manualTspinbox.label, row, 1)
        layout.addWidget(self.manualTspinbox, row, 2)
        self.manualEntryRadiobutton.widgets.append(self.manualTspinbox)
        sectionWidgets.append(self.manualTspinbox.label)
        sectionWidgets.append(self.manualTspinbox)

        if SizeT == 1:
            self.manualTspinbox.setVisible(False)
            self.manualTspinbox.label.setVisible(False)
        
        self.manualEntryRadiobutton.toggled.connect(self.enableRadioButtonWidgets)
        self.enableRadioButtonWidgets(False, sender=self.manualEntryRadiobutton)
        
        return row + 1, sectionWidgets
    
    def addWithMouseClicksSection(self, row, layout, imagesPath='', **kwargs):
        sectionWidgets = []
        
        self.clickEntryIsLoadedDf = None
        
        self.clickEntryRadiobutton = QRadioButton('Add points with mouse clicks')
        layout.addWidget(self.clickEntryRadiobutton, row, 0, 1, 2) 
        self.clickEntryRadiobutton.widgets = [] 
        sectionWidgets.append(self.clickEntryRadiobutton)
        
        row += 1
        self.snapToMaxToggle = widgets.Toggle()
        self.snapToMaxToggle.label = QLabel('Snap to closest maximum: ')
        layout.addWidget(self.snapToMaxToggle.label, row, 1)
        layout.addWidget(
            self.snapToMaxToggle, row, 2, alignment=Qt.AlignCenter
        )
        sectionWidgets.append(self.snapToMaxToggle.label)
        sectionWidgets.append(self.snapToMaxToggle)
        
        self.snapToMaxInfoButton = widgets.infoPushButton()
        layout.addWidget(self.snapToMaxInfoButton, row, 3)
        sectionWidgets.append(self.snapToMaxInfoButton)
        
        self.snapToMaxInfoButton.clicked.connect(self.showSnapToMaxButton)
        self.clickEntryRadiobutton.widgets.append(self.snapToMaxToggle)
        self.clickEntryRadiobutton.widgets.append(self.snapToMaxInfoButton)
        
        row += 1
        self.autoPilotToggle = widgets.Toggle()
        self.autoPilotToggle.label = QLabel('Use auto-pilot: ')
        layout.addWidget(self.autoPilotToggle.label, row, 1)
        layout.addWidget(
            self.autoPilotToggle, row, 2, alignment=Qt.AlignCenter
        )
        sectionWidgets.append(self.autoPilotToggle.label)
        sectionWidgets.append(self.autoPilotToggle)
        self.autoPilotInfoButton = widgets.infoPushButton()
        layout.addWidget(self.autoPilotInfoButton, row, 3)
        sectionWidgets.append(self.autoPilotInfoButton)
        
        self.autoPilotInfoButton.clicked.connect(self.showAutoPilotInfo)
        self.clickEntryRadiobutton.widgets.append(self.autoPilotToggle)
        self.clickEntryRadiobutton.widgets.append(self.autoPilotInfoButton)
        
        row += 1
        self.clickEntryTableEndname = widgets.alphaNumericLineEdit()
        self.clickEntryTableEndname.setText('points_added_by_clicking')
        self.clickEntryTableEndname.setAlignment(Qt.AlignCenter)
        self.clickEntryTableEndname.label = QLabel('Table endname: ')
        loadButton = widgets.browseFileButton(
            start_dir=imagesPath, ext={'CSV': '.csv'}
        )
        layout.addWidget(loadButton, row, 3)
        sectionWidgets.append(loadButton)
        
        loadButton.sigPathSelected.connect(self.loadClickEntryTable)
        self.loadButton = loadButton
        self.clickEntryLoadTableButton = loadButton
        layout.addWidget(self.clickEntryTableEndname.label, row, 1)
        layout.addWidget(self.clickEntryTableEndname, row, 2)
        self.clickEntryRadiobutton.widgets.append(self.clickEntryTableEndname)
        self.clickEntryTableEndname.editingFinished.connect(
            self.emitCheckClickEntryTableEndnameExists
        )
        sectionWidgets.append(self.clickEntryTableEndname)
        sectionWidgets.append(self.clickEntryTableEndname.label)
        
        row += 1
        instructionsText = html_utils.paragraph(
            '<br><i>Left-click</i> to annotate a new point with a new id.<br><br>'
            '<i>Right-click</i> to annotate a point with the same id<br><br>'
            '<i>Same click used to delete objects</i> to annotate<br>'
            'a point with id = 0 (negative prompt)<br><br>'
            '<i>Click</i> on point to delete it',
            font_size='11px'
        )
        self.instructionsLabel = QLabel(instructionsText)
        self.instructionsLabel.label = QLabel('Instructions')
        layout.addWidget(self.instructionsLabel.label, row, 1)
        layout.addWidget(self.instructionsLabel, row, 2)
        self.clickEntryRadiobutton.widgets.append(self.instructionsLabel)
        sectionWidgets.append(self.instructionsLabel)
        sectionWidgets.append(self.instructionsLabel.label)
        
        self.clickEntryRadiobutton.toggled.connect(self.enableRadioButtonWidgets)
        self.clickEntryRadiobutton.toggled.connect(
            self.emitCheckClickEntryTableEndnameExists
        )
        self.enableRadioButtonWidgets(False, sender=self.clickEntryRadiobutton)
        
        return row + 1, sectionWidgets
    
    def emitCheckClickEntryTableEndnameExists(self, *args, **kwargs):
        if not self.clickEntryRadiobutton.isChecked():
            return
        self.clickEntryIsLoadedDf = None
        tableEndName = self.clickEntryTableEndname.text()
        self.sigCheckClickEntryTableEndnameExists.emit(
            tableEndName, False
        )
    
    def loadClickEntryTable(self, csv_path):
        self.clickEntryIsLoadedDf = None
        posData = load.loadData(csv_path, 'points')
        posData.getBasenameAndChNames(qparent=self)
        basename = posData.basename
        filename = os.path.basename(csv_path)
        filename, ext = os.path.splitext(filename)
        if not basename.endswith('_'):
            basename = f'{basename}_'
            
        endname = filename[len(basename):]
        self.clickEntryTableEndname.setText(endname)
        self.sigCheckClickEntryTableEndnameExists.emit(
            endname, True
        )
        
    def showAutoPilotInfo(self):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            With <b>Auto-pilot</b> mode active, Cell-ACDC will <b>automatically zoom</b> on 
            to an object<br>
            to allow you clicking on the points you want to add.<br><br>
            You can then go to the <b>next object</b> by pressing the 
            <code>Enter</code> key or go back to the<br>
            <b>previous object</b> by pressing <code>Backspace</code>.
        """)
        msg.information(self, 'Auto-pilot info', txt)
    
    def showSnapToMaxButton(self):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            With <bSnap to closest maximum</b> mode active, Cell-ACDC will 
            <b>automatically add the point</b><br>
            to the closest maximum within the point footprint (defined in 
            the appearance settings).
        """)
        msg.information(self, 'Snap to closest maximum info', txt)
    
    def closeEvent(self, event):
        self.sigClosed.emit()
    
    def enableRadioButtonWidgets(self, enabled, sender=None):
        if sender is None:
            sender = self.sender()
        for widget in sender.widgets:
            widget.setDisabled(not enabled)
            try:
                widget.label.setDisabled(not enabled)
            except:
                pass
    
    def _readTable(self, path):
        return load.load_df_points_layer(path)
    
    def tryAutoFillColNames(self, df):
        if 'x' in df.columns:
            self.xColName.setCurrentText('x')
        
        if 'y' in df.columns:
            self.yColName.setCurrentText('y')
        
        if 'z' in df.columns:
            self.zColName.setCurrentText('z')
        
        if 'frame_i' in df.columns:
            self.tColName.setCurrentText('frame_i')
    
    def tablePathSelected(self, path):
        self.tablePath.setText(path)
        try:
            df = self._readTable(path)
            self.xColName.addItems(df.columns)
            self.yColName.addItems(df.columns)
            self.zColName.addItems(df.columns)
            self.tColName.addItems(df.columns)
            self.tryAutoFillColNames(df)
            self.sigLoadedTable.emit(df, os.path.basename(path))
            self.browseTableButton.confirmAction()
        except Exception as e:
            traceback_format = traceback.format_exc()
            self.sigCriticalReadTable.emit(traceback_format)
            self.criticalReadTable(path, traceback_format)
            self.tablePath.setText('')
        
    
    def criticalLenMismatchManualEntry(self):
        txt = html_utils.paragraph(f"""
            X coords and Y coords must have the <b>same length</b>.
        """)
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.critical(self, f'X and Y have different length', txt)
        
    def criticalColNameIsNone(self, axis):        
        txt = html_utils.paragraph(f"""
            The "{axis.upper()} coord. column" <b>cannot be "None"</b>
        """)
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.critical(self, f'{axis.upper()} coord. is None', txt)
    
    def criticalReadTable(self, path, traceback_format):
        txt = html_utils.paragraph(f"""
            Something went <b>wrong when reading the table</b> from the 
            following path:<br><br>
            <code>{path}</code><br><br>
            See the <b>error message below</b>.
        """)
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        detailsText = traceback_format
        msg.critical(
            self, 'Error when reading table', txt, detailsText=detailsText)

    def criticalEmptyTablePath(self):
        txt = html_utils.paragraph(f"""
            The table file path <b>cannot be empty</b>.
        """)
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.critical(self, 'Table file path is empty', txt)   

    def state(self):
        _state = self.appearanceGroupbox.state() 
        return _state

    def _checkSelectedColName(self, colName, label):
        labelsToCheck = ['z', 'y', 'x']
        labelsToCheck.remove(label)
        for labelToCheck in labelsToCheck:
            if colName.find(labelToCheck) != -1:
                break
        else:
            return True

        txt = html_utils.paragraph(f"""
            Are you sure that the {label.upper()} coord. column should contain 
            the letter <code>{labelToCheck}<code>?
        """)
        
        msg = widgets.myMessageBox(wrapText=False)
        _, noButton, yesButton = msg.warning(
            self, 'Check column name', txt, 
            buttonsTexts=('Cancel', 'No, let me correct it', 'Yes, I am')
        )
        if msg.cancel or msg.clickedButton == noButton:
            return False
        return True
    
    def checkColNameX(self, text):
        accepted = self._checkSelectedColName(text, 'x')
        if accepted: 
            return
        self.xColName.setCurrentText('None')
    
    def checkColNameY(self, text):
        accepted = self._checkSelectedColName(text, 'y')
        if accepted: 
            return
        self.yColName.setCurrentText('None')
    
    def checkColNameZ(self, text):
        accepted = self._checkSelectedColName(text, 'z')
        if accepted: 
            return
        self.zColName.setCurrentText('None')

    def warnShortcutAlreadyUsed(self, shortcut, usedBy):
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Shortcut already used', html_utils.paragraph(
                f'The shortcut "{html.escape(shortcut)}" is already used by '
                f'"{html.escape(usedBy)}". Choose a different shortcut.'
            )
        )

    def reservedShortcuts(self):
        """Current {key sequence text: used by} (see `__init__`)."""
        reserved = self._reservedShortcuts
        return dict((reserved() if callable(reserved) else reserved) or {})

    def ok_cb(self):
        shortcut = self.appearanceGroupbox.shortcutWidget.widget.text().strip()
        usedBy = None
        if shortcut:
            usedBy = self.reservedShortcuts().get(
                QKeySequence(shortcut).toString()
            )
        if usedBy is not None:
            self.warnShortcutAlreadyUsed(shortcut, usedBy)
            return

        self.pointsData = {}
        self.loadedDfInfo = None
        self.loadedDf = None
        self.weighingChannel = ''
        if self.fromTableRadiobutton.isChecked():
            tablePath = self.tablePath.text()
            if not tablePath:
                self.criticalEmptyTablePath()
                return
            
            try:
                df = self._readTable(tablePath)
                tColName = self.tColName.currentText()
                xColName = self.xColName.currentText()
                yColName = self.yColName.currentText()
                zColName = self.zColName.currentText()

                tColRequiresGrouping = (
                    self.tColRequiresGroupingCheckbox.isChecked()
                )

                self.loadedDfInfo = {
                    'filepath': tablePath,
                    't': tColName, 
                    'z': zColName, 
                    'y': yColName, 
                    'x': xColName,
                    't_col_requires_grouping': tColRequiresGrouping
                }
                
                self._df_to_pointsData(
                    df, tColName, zColName, yColName, xColName, 
                    tColRequiresGrouping=tColRequiresGrouping
                )
                    
            except Exception as e:
                traceback_format = traceback.format_exc()
                self.sigCriticalReadTable.emit(traceback_format)
                self.criticalReadTable(tablePath, traceback_format)
                return
            
            if self.xColName.currentText() == 'None':
                self.criticalColNameIsNone('x')
                return
            if self.yColName.currentText() == 'None':
                self.criticalColNameIsNone('y')
                return
            
            self.layerType = os.path.basename(self.tablePath.text())
            self.layerTypeIdx = 2
        elif self.centroidsRadiobutton.isChecked():
            self.layerType = 'Centroids'
            self.layerTypeIdx = 0
        elif self.weightedCentroidsRadiobutton.isChecked():
            channel = self.channelNameForWeightedCentr.currentText()
            self.weighingChannel = channel
            self.layerType = f'Centroids weighted by channel {channel}'
            self.layerTypeIdx = 1
        elif self.manualEntryRadiobutton.isChecked():
            xx = self.manualXspinbox.values()
            yy = self.manualYspinbox.values()
            if len(xx) != len(yy):
                self.criticalLenMismatchManualEntry()
                return
            zz = self.manualZspinbox.values()
            tt = [t+1 for t in self.manualTspinbox.values()]
            df = pd.DataFrame({'x': xx, 'y': yy, 'id': np.arange(1, len(xx)+1)})
            if tt:
                df['t'] = tt
                tCol = 't'
            else:
                tCol = 'None'
            if zz:
                df['z'] = zz
                zCol = 'z'
            else:
                zCol = 'None'
            
            self._df_to_pointsData(df, tCol, zCol, 'y', 'x')
            
            self.layerType = 'Manual entry'
            self.layerTypeIdx = 3
        elif self.clickEntryRadiobutton.isChecked():
            self.layerType = ('Click to annotate point')
            self.description = (
                'Left-click to add a point, click on point to delete it.\n'
                'With auto-pilot you can navigate through object with Up/Down arrows.'
            )
            self.clickEntryTableEndnameText = self.clickEntryTableEndname.text()
            self.layerTypeIdx = 4
        
        self.cancel = False
        symbol = self.appearanceGroupbox.symbolWidget.widget.currentText()
        self.symbol = re.findall(r"\'(.+)\'", symbol)[0]
        self.symbolText = symbol
        self.color = self.appearanceGroupbox.colorButton.color()
        self.pointSize = self.appearanceGroupbox.sizeSpinBox.value()
        self.zHeight = self.appearanceGroupbox.zHeightSpinBox.value()
        shortcutWidget = self.appearanceGroupbox.shortcutWidget
        self.shortcut = shortcutWidget.widget.text()
        self.keySequence = shortcutWidget.widget.keySequence
        self.close()
    
    def _df_to_pointsData(
            self, df, tColName, zColName, yColName, xColName, 
            tColRequiresGrouping=False
        ):
        self.pointsData = load.loaded_df_to_points_data(
            df, tColName, zColName, yColName, xColName, t_col_requires_grouping=tColRequiresGrouping
        )
    
    def showEvent(self, event) -> None:
        if self._parent is None:
            screen = self.screen()
        else:
            screen = self._parent.screen()
        screenWidth = screen.size().width()
        screenHeight = screen.size().height()
        
        maxHeight = screenHeight - 100
        
        buttonHeight = self.buttonsLayout.okButton.minimumSizeHint().height()
        height = (
            self.scrollArea.minimumHeightNoScrollbar()
            + self.appearanceGroupbox.sizeHint().height()
            + buttonHeight + 70
        )
        width = self.scrollArea.minimumWidthNoScrollbar() + 50
        
        height = min(height, maxHeight)
        
        self.resize(width, height)
        
        screenLeft = screen.geometry().x()
        screenTop = screen.geometry().y()
        w, h = self.width(), self.height()
        left = int(screenLeft + screenWidth/2 - w/2)
        top = int(screenTop + screenHeight/2 - h/2 - 20)

        self.move(left, top)
