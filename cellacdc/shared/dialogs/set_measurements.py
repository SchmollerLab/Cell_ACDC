"""Set measurements dialog (from apps.py), used by the main GUI
(showSetMeasurements), the Compute measurements and Concatenate
utilities and the model parameters feature ranges.

SetMeasurementsDialog with its delete-unselected-columns prompt
askDeleteUncheckedExistingMeasurements, and the measurement errors
report ComputeMetricsErrorsDialog, plus the measurement group boxes that
only this dialog uses. Read the selection only via selectedMetrics() /
selectedMeasurements() (see CLAUDE.md).
"""
import re
from functools import partial
from qtpy.QtCore import Signal, Qt, QUrl
from qtpy.QtGui import QFont, QDesktopServices
from qtpy.QtWidgets import (
    QHBoxLayout, QCheckBox, QWidget, QVBoxLayout, QScrollArea, QPushButton,
    QGroupBox
)
from cellacdc import measurements, html_utils
from cellacdc import _palettes
from ..widgets import push_buttons as widgets_push_buttons
from ..widgets import basic_widgets as widgets_basic_widgets
from ..widgets import search_widgets as widgets_search_widgets
from ..widgets import message_box as widgets_message_box
import os
from typing import Generator, Tuple
from collections import defaultdict
import html
from types import SimpleNamespace
from qtpy.QtGui import QIcon, QKeyEvent
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QApplication, QLabel, QButtonGroup, QGridLayout, QFrame, QStyle
)
from cellacdc import _window_stacking
from cellacdc import load
from cellacdc import issues_url
from cellacdc.myutils import filesystem
from cellacdc import qutils
from ..widgets import item_views as widgets_item_views
from cellacdc import io
from cellacdc import fonts
from . import base as dialogs_base
from . import file_selection as dialogs_file_selection


class CheckboxesGroupBox(QGroupBox):
    def __init__(
            self, texts, title='', checkable=False, parent=None
        ):
        super().__init__(parent)
        
        self.setTitle(title)
        self.setCheckable(checkable)
        layout = QVBoxLayout()

        scrollLayout = QVBoxLayout()
        container = QWidget()
        scrollarea = QScrollArea()
        
        self.checkBoxes = []
        for text in texts:
            checkbox = QCheckBox(text)
            checkbox.setChecked(True)
            scrollLayout.addWidget(checkbox)
            self.checkBoxes.append(checkbox)
        
        container.setLayout(scrollLayout)
        scrollarea.setWidget(container)
        layout.addWidget(scrollarea)
        
        buttonsLayout = QHBoxLayout()
        selectAllButton = widgets_push_buttons.selectAllPushButton()
        selectAllButton.sigClicked.connect(self.checkAll)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(selectAllButton)
        layout.addLayout(buttonsLayout)
        
        self.setLayout(layout)
    
    def checkAll(self, button, checked):
        for checkBox in self.checkBoxes:
            checkBox.setChecked(checked)

class _metricsQGBox(QGroupBox):
    sigDelClicked = Signal(str, object)

    def __init__(
            self, desc_dict, title, favourite_funcs=None, isZstack=False,
            equations=None, addDelButton=False, delButtonMetricsDesc=None,
            parent=None, addCalcForEachZsliceToggle=False, 
            doesWarnOnUncheck=False,
            isSegm3D=False
        ):
        QGroupBox.__init__(self, parent)
        
        highlightRgba = _palettes._highlight_rgba()
        r, g, b, a = highlightRgba
        self._highlightStylesheetColor = f'rgb({r}, {g}, {b})'
        
        self._updatingCheckboxes = False
        self._parent = parent
        self.doesWarnOnUncheck = doesWarnOnUncheck
        self.scrollArea = QScrollArea()
        self.scrollAreaWidget = QWidget()
        self.favourite_funcs = favourite_funcs

        self.doNotWarn = False

        layout = QVBoxLayout()
        inner_layout = QVBoxLayout()
        self.inner_layout = inner_layout
        if delButtonMetricsDesc is None:
            delButtonMetricsDesc = []

        self.checkBoxes = []
        self.checkedState = {}
        self.metricDescriptions = desc_dict
        for metric_colname, metric_desc in desc_dict.items():
            rowLayout = QHBoxLayout()

            checkBox = QCheckBox(metric_colname)
            checkBox.setChecked(True)
            checkBox.scrollArea = self.scrollArea
            self.checkBoxes.append(checkBox)
            self.checkedState[checkBox] = True

            try:
                checkBox.equation = equations[metric_colname]
            except Exception as e:
                pass
            
            if addDelButton or metric_colname in delButtonMetricsDesc:
                delButton = widgets_push_buttons.delPushButton()
                delButton.setToolTip('Delete custom combined measurement')
                delButton.colname = metric_colname
                delButton.checkbox = checkBox
                delButton.clicked.connect(self.onDelClicked)
                delButton._layout = rowLayout
                rowLayout.addWidget(delButton)

            infoButton = widgets_push_buttons.infoPushButton()
            infoButton.setCursor(Qt.WhatsThisCursor)
            infoButton.info = metric_desc
            infoButton.colname = metric_colname
            infoButton.clicked.connect(self.showInfo)

            rowLayout.addWidget(infoButton)
            rowLayout.addWidget(checkBox)   
            rowLayout.addStretch(1)          

            inner_layout.addLayout(rowLayout)

        self.scrollAreaWidget.setLayout(inner_layout)
        self.scrollArea.setWidget(self.scrollAreaWidget)
        layout.addWidget(self.scrollArea)

        buttonsLayout = QHBoxLayout()    
            
        buttonsLayout.addStretch(1)

        if isSegm3D:
            selectAll3DButton = widgets_push_buttons.selectAllPushButton(
                suffix_text_select='only 3D metrics',
                suffix_text_deselect='all 3D metrics'
            )
            selectAll3DButton.setChecked(False)
            selectAll3DButton.sigClicked.connect(self.checkAll3D)
            self.selectAll3DButton = selectAll3DButton
            buttonsLayout.addWidget(selectAll3DButton)
        
        self.selectAllButton = widgets_push_buttons.selectAllPushButton()
        self.selectAllButton.sigClicked.connect(self.checkAll)
        
        buttonsLayout.addWidget(self.selectAllButton)

        if favourite_funcs is not None:
            self.loadFavouritesButton = widgets_push_buttons.reloadPushButton(
                '  Load last selection...  '
            )
            self.loadFavouritesButton.clicked.connect(self.checkFavouriteFuncs)
            # self.checkFavouriteFuncs()
            buttonsLayout.addWidget(self.loadFavouritesButton)

        layout.addLayout(buttonsLayout)
        
        self.calcForEachZsliceToggle = None
        if addCalcForEachZsliceToggle:
            buttonsLayout = QHBoxLayout()
            self.calcForEachZsliceToggle = widgets_push_buttons.Toggle()
            tooltip = (
                'Calculate `cell_area` for each z-slice.\n\n'
                'The measurements will be saved in the column with name\n'
                'ending with `_zsliceN` where N is the z-slice number\n'
                '(starting from 0).'
            )
            calcForEachZsliceLabel = widgets_basic_widgets.QClickableLabel(
                'Calculate for each z-slice'
            )
            calcForEachZsliceLabel.setToolTip(tooltip)
            self.calcForEachZsliceToggle.setToolTip(tooltip)
            buttonsLayout.addWidget(self.calcForEachZsliceToggle)
            buttonsLayout.addWidget(calcForEachZsliceLabel)
            buttonsLayout.addStretch(1) 
            layout.addLayout(buttonsLayout)
            calcForEachZsliceLabel.clicked.connect(
                partial(
                    self.toggleCalcForEachZslice, 
                    toggle=self.calcForEachZsliceToggle
                )
            )

        self.setTitle(title)
        self.setCheckable(True)
        self.setLayout(layout)
        _font = QFont()
        _font.setPixelSize(11)
        self.setFont(_font)

        self.toggled.connect(self.toggled_cb)
    
    def toggleCalcForEachZslice(self, label, toggle=None):
        if toggle is None:
            toggle = self.calcForEachZsliceToggle
        
        toggle.setChecked(not toggle.isChecked())
    
    def isCalcForEachZsliceRequested(self):
        if self.calcForEachZsliceToggle is None:
            return False
        
        return self.calcForEachZsliceToggle.isChecked()
    
    def highlightCheckboxesFromSearchText(self, text):
        for checkbox in self.checkBoxes:
            highlighted = widgets_search_widgets._matchesSearchText(text, checkbox.text())
            self.setCheckboxHighlighted(highlighted, checkbox)
    
    def setCheckboxHighlighted(self, highlighted, checkbox):
        if highlighted:
            checkbox.setStyleSheet(
                f'background: {self._highlightStylesheetColor}; color: black'
            )
            self.scrollArea.ensureWidgetVisible(checkbox)
        else:
            checkbox.setStyleSheet('')
    
    def onDelClicked(self):
        button = self.sender()
        button.checkbox.setChecked(False)
        self.sigDelClicked.emit(button.colname, button._layout)
    
    def toggled_cb(self, checked):
        alreadyWarnedRequired = False
        for checkbox in self.checkBoxes:
            if getattr(checkbox, 'forcedByCombined', False):
                # Input of a checked combined measurement --> stays checked
                # (see `SetMeasurementsDialog.setDisabledMetricsRequestedForCombined`)
                continue

            if self.doesWarnOnUncheck and alreadyWarnedRequired:
                checkbox.blockSignals(True)

            if not checked:
                self.checkedState[checkbox] = checkbox.isChecked()
                checkbox.setChecked(False)
            else:
                checkbox.setChecked(
                    self.checkedState[checkbox]
                )
            
            cannotBeUnchecked = (
                hasattr(checkbox, 'isRequired')
                and checkbox.isRequired
                and not checked
            )
            if not alreadyWarnedRequired:
                alreadyWarnedRequired = cannotBeUnchecked
            
            if cannotBeUnchecked and alreadyWarnedRequired:
                checkbox.setChecked(True)
            
            checkbox.blockSignals(False)

    def checkFavouriteFuncs(self, checked=True, isZstack=False):
        self.doNotWarn = True
        if self._parent is not None:
            self._parent.doNotWarn = True
        for checkBox in self.checkBoxes:
            checkBox.setChecked(False)
            for favourite_func in self.favourite_funcs:
                func_name = checkBox.text()
                if func_name.endswith(favourite_func):
                    checkBox.setChecked(True)
                    break
        self.doNotWarn = False
        if self._parent is not None:
            self._parent.doNotWarn = False

    def checkAll3D(self, button, checked):
        if self._parent is not None:
            self._parent.doNotWarn = True

        for checkBox in self.checkBoxes:
            if not checkBox.text().endswith('_3D'):
                if checked:
                    # Deselect non-3D 
                    checkBox.setChecked(False)
                continue

            checkBox.setChecked(checked)
        if self._parent is not None:
            self._parent.doNotWarn = False
        
        self.selectAllButton.sigClicked.disconnect()
        self.selectAllButton.setChecked(checked)
        self.selectAllButton.sigClicked.connect(self.checkAll)

    def checkAll(self, button, checked):
        if self._parent is not None:
            self._parent.doNotWarn = True
        for checkBox in self.checkBoxes:
            if checkBox.isHidden():
                # Deleted (e.g. combined measurement)
                continue
            checkBox.setChecked(checked)
            if not self.isChecked():
                # Restored by `toggled_cb` when the group is checked again
                self.checkedState[checkBox] = checkBox.isChecked()
        if self._parent is not None:
            self._parent.doNotWarn = False

    def selectedMetrics(self):
        """Names of the selected metrics, in display order.

        A metric is selected when its checkbox is checked and the group is
        checked. Checked metrics of an unchecked group are selected only
        when another selected measurement requires them (`isRequired`, e.g.
        the background median of an amount) or a checked combined
        measurement uses them (`forcedByCombined`).
        """
        selected = []
        for checkBox in self.checkBoxes:
            if not checkBox.isChecked() or checkBox.isHidden():
                continue
            isNeeded = (
                getattr(checkBox, 'isRequired', False)
                or getattr(checkBox, 'forcedByCombined', False)
            )
            if not self.isChecked() and not isNeeded:
                continue
            selected.append(checkBox.text())
        return selected

    def showInfo(self, checked=False):
        info_txt = self.sender().info
        msg = widgets_message_box.myMessageBox()
        msg.setWidth(600)
        msg.setIcon()
        msg.setWindowTitle(f'{self.sender().colname} info')
        msg.addText(info_txt)
        msg.addButton('   Ok   ')
        msg.exec_()

    def show(self):
        super().show()
        fw = self.inner_layout.contentsRect().width()
        sw = self.scrollArea.verticalScrollBar().sizeHint().width()
        self.minWidth = fw + sw

class channelMetricsQGBox(QGroupBox):
    sigDelClicked = Signal(str, object)
    sigCheckboxToggled = Signal(object)

    def __init__(
            self, isZstack, chName, isSegm3D, is_concat=False,
            posData=None, favourite_funcs=None
        ):
        QGroupBox.__init__(self)

        self.doNotWarn = False
        self.is_concat = is_concat
        isManualBackgrPresent = False
        if posData is not None:
            if posData.manualBackgroundLab is not None:
                isManualBackgrPresent = True

        layout = QVBoxLayout()
        metrics_desc, bkgr_val_desc = measurements.standard_metrics_desc(
            isZstack, chName, isSegm3D=isSegm3D, 
            isManualBackgrPresent=isManualBackgrPresent
        )

        metricsQGBox = _metricsQGBox(
            metrics_desc, 'Standard measurements',
            favourite_funcs=favourite_funcs, 
            parent=self, isZstack=isZstack,
            isSegm3D=isSegm3D
        )
        self.metricsQGBox = metricsQGBox
        
        bkgrValsQGBox = _metricsQGBox(
            bkgr_val_desc, 'Background values',
            favourite_funcs=favourite_funcs, 
            parent=self, isZstack=isZstack,
            doesWarnOnUncheck=True,
            isSegm3D=isSegm3D
        )
        self.bkgrValsQGBox = bkgrValsQGBox

        self.checkBoxes = metricsQGBox.checkBoxes.copy()
        self.checkBoxes.extend(bkgrValsQGBox.checkBoxes)

        self.uncheckAndDisableDataPrepIfPosNotPrepped(posData)
        self.uncheckAndDisableManualBkgrIfNotPresent(posData)

        self.groupboxes = [metricsQGBox, bkgrValsQGBox]

        for checkbox in metricsQGBox.checkBoxes:
            checkbox.toggled.connect(self.standardMetricToggled)
            self.standardMetricToggled(checkbox.isChecked(), checkbox=checkbox)
        
        for bkgrCheckbox in bkgrValsQGBox.checkBoxes:
            bkgrCheckbox.toggled.connect(self.backgroundMetricToggled)

        layout.addWidget(metricsQGBox)
        layout.addWidget(bkgrValsQGBox)

        items = measurements.custom_metrics_desc(
            isZstack, chName, posData=posData, isSegm3D=isSegm3D,
            return_combine=True
        )
        custom_metrics_desc, combine_metrics_desc = items

        if custom_metrics_desc:
            customMetricsQGBox = _metricsQGBox(
                custom_metrics_desc, 'Custom measurements', 
                delButtonMetricsDesc=combine_metrics_desc,
                favourite_funcs=favourite_funcs,
                isZstack=isZstack,
                isSegm3D=isSegm3D
            )
            layout.addWidget(customMetricsQGBox)
            self.checkBoxes.extend(customMetricsQGBox.checkBoxes)
            customMetricsQGBox.sigDelClicked.connect(self.onDelClicked)
            self.customMetricsQGBox = customMetricsQGBox

        self.calcForEachZsliceToggle = None
        if isZstack:
            buttonsLayout = QHBoxLayout()
            self.calcForEachZsliceToggle = widgets_push_buttons.Toggle()
            tooltip = (
                'Calculate the selected measurements for each z-slice.\n\n'
                'The measurements will be saved in the column with name\n'
                'ending with `_zsliceN` where N is the z-slice number\n'
                '(starting from 0).'
            )
            calcForEachZsliceLabel = widgets_basic_widgets.QClickableLabel(
                'Calculate for each z-slice'
            )
            calcForEachZsliceLabel.setToolTip(tooltip)
            self.calcForEachZsliceToggle.setToolTip(tooltip)
            buttonsLayout.addWidget(self.calcForEachZsliceToggle)
            buttonsLayout.addWidget(calcForEachZsliceLabel)   
            buttonsLayout.addStretch(1) 
            layout.addLayout(buttonsLayout)
            calcForEachZsliceLabel.clicked.connect(
                partial(
                    self.toggleCalcForEachZslice, 
                    toggle=self.calcForEachZsliceToggle
                )
            )
                
        
        self.setTitle(f'{chName} metrics')
        self.setCheckable(True)
        self.setLayout(layout)
    
    def toggleCalcForEachZslice(self, label, toggle=None):
        if toggle is None:
            toggle = self.calcForEachZsliceToggle

        toggle.setChecked(not toggle.isChecked())

    def isCalcForEachZsliceRequested(self):
        if self.calcForEachZsliceToggle is None:
            return False

        return self.calcForEachZsliceToggle.isChecked()

    def subGroupboxes(self):
        """Standard, background and (if present) custom measurements groups"""
        groupboxes = list(self.groupboxes)
        customMetricsQGBox = getattr(self, 'customMetricsQGBox', None)
        if customMetricsQGBox is not None:
            groupboxes.append(customMetricsQGBox)
        return groupboxes

    def selectedMetrics(self):
        """Names of the selected metrics of this channel, in display order.

        Nothing is selected when the channel group is unchecked. Otherwise
        see `_metricsQGBox.selectedMetrics` for each sub-group (checkboxes
        left checked in an unchecked sub-group are not selected).
        """
        if not self.isChecked():
            return []

        selected = []
        for groupbox in self.subGroupboxes():
            selected.extend(groupbox.selectedMetrics())
        return selected

    def uncheckAndDisableDataPrepIfPosNotPrepped(self, posData):
        # Uncheck and disable dataprep metrics if pos is not prepped
        if posData is None:
            return

        if posData.isBkgrROIpresent():
            return

        for checkbox in self.checkBoxes:
            if checkbox.text().find('dataPrep') == -1:
                continue

            checkbox.setChecked(False)
            checkbox.isDataPrepDisabled = True
    
    def uncheckAndDisableManualBkgrIfNotPresent(self, posData):
        # Uncheck and disable dataprep metrics if pos is not prepped
        if posData is None:
            return

        if posData.manualBackgroundLab is not None:
            return

        for checkbox in self.checkBoxes:
            if checkbox.text().find('manualBkgr') == -1:
                continue

            checkbox.setChecked(False)
            checkbox.isManualBkgrDisabled = True
    
    def _warnDataPrepCannotBeChecked(self):
        if self.doNotWarn:
            return
        txt = html_utils.paragraph("""
            <b>Data prep measurements cannot be saved</b> because you did 
            not select any background ROI at the data prep step.<br><br>

            You can read more details about data prep metrics by clicking 
            on the info button besides the measurement's name.<br><br>

            Thank you for you patience!
        """)
        msg = widgets_message_box.myMessageBox(showCentered=False)
        msg.warning(self, 'Metric cannot be saved', txt)
    
    def _warnManualBkgrCannotBeChecked(self):
        if self.doNotWarn:
            return
        txt = html_utils.paragraph("""
            <b>Measurements requiring manual background cannot be saved</b> because you did not setup any manual background ROI in the GUI.<br><br>

            You can read more details about manual background metrics by clicking 
            on the info button besides the measurement's name.<br><br>

            Thank you for you patience!
        """)
        msg = widgets_message_box.myMessageBox(showCentered=False)
        msg.warning(self, 'Metric cannot be saved', txt)

    def standardMetricToggled(self, checked, checkbox=None):
        """Method called when a check-box is toggled. It performs the following 
        actions:
            1. If the user try to check a data prep measurement, such as 
            dataPrep_amount, and this cannot be saved (checkbox has the attr 
            `isDataPrepDisabled`) then it warns and explains why it cannot be saved
            2. Make sure that background value median is checked if the user 
            requires amount or concentration metric.
            3. Do not allow unchecking background value median and explain why.

        Parameters
        ----------
        checked : bool
            State of the checkbox toggled
        checkbox : QtWidgets.QCheckBox, optional
            The checkbox that has been toggled. Default is None. If None 
            use `self.sender()`
        """        
        if self.is_concat:
            return
        
        if checkbox is None:
            checkbox = self.sender()

        if hasattr(checkbox, 'isDataPrepDisabled'):
            # Warn that user cannot check data prep metrics and uncheck it
            if not checkbox.isChecked():
                return
            checkbox.setChecked(False)
            self._warnDataPrepCannotBeChecked()
            return
        
        if hasattr(checkbox, 'isManualBkgrDisabled'):
            # Warn that user cannot check data prep metrics and uncheck it
            if not checkbox.isChecked():
                return
            checkbox.setChecked(False)
            self._warnManualBkgrCannotBeChecked()
            return

        self.sigCheckboxToggled.emit(checkbox)
        if checkbox.text().find('amount_') == -1:
            return
        pattern = r'amount_([A-Za-z]+)(_?[A-Za-z0-9]*)'
        repl = r'\g<1>_bkgrVal_median\g<2>'
        bkgrValMetric = s1 = re.sub(pattern, repl, checkbox.text())
        for bkgrCheckbox in self.groupboxes[1].checkBoxes:
            if bkgrCheckbox.text() == bkgrValMetric:
                break
        else:
            # Make sure to not check for similarly named custom metrics
            return
        
        if checked:
            if not bkgrCheckbox.isChecked():
                # Checked only because it is required --> uncheck it again
                # when it is not required anymore
                bkgrCheckbox.uncheckWhenNotRequired = True
            bkgrCheckbox.isRequired = True
            bkgrCheckbox.setChecked(True)
        else:
            bkgrCheckbox.setDisabled(False)
            bkgrCheckbox.isRequired = False
            bkgrValsQGBox = self.groupboxes[1]
            uncheck = (
                getattr(bkgrCheckbox, 'uncheckWhenNotRequired', False)
                or not bkgrValsQGBox.isChecked()
            )
            bkgrCheckbox.uncheckWhenNotRequired = False
            if uncheck:
                # The user unchecked it (or its group) while it was
                # required --> do not keep it checked
                bkgrCheckbox.setChecked(False)
                if not bkgrValsQGBox.isChecked():
                    bkgrValsQGBox.checkedState[bkgrCheckbox] = False
    
    def backgroundMetricToggled(self, checked):
        """Method called when a checkbox of a background metric is toggled.
        Check if the background value is required and explain why it cannot be 
        unchecked.

        Parameters
        ----------
        checked : bool
            State of the checkbox toggled
        """
        if self.is_concat:
            return
        
        checkbox = self.sender()
        if not getattr(checkbox, 'isRequired', False):
            # Toggled while not required --> the state is the user's choice
            checkbox.uncheckWhenNotRequired = False
            return

        if checkbox.isChecked():
            return

        checkbox.setChecked(True)
        # The user wanted it unchecked --> uncheck it when the measurement
        # requiring it is unchecked
        checkbox.uncheckWhenNotRequired = True
        if self.doNotWarn:
            return

        txt = html_utils.paragraph("""
            <b>This background value cannot be unchecked</b> because it is required 
            by the <code>_amount</code> and <code>_concentration</code> measurements 
            that you requested to save.<br><br>

            Thank you for you patience!
        """)
        msg = widgets_message_box.myMessageBox(showCentered=False)
        msg.warning(self, 'Background value required', txt)
    
    def onDelClicked(self, colname_to_del, hlayout):
        self.sigDelClicked.emit(colname_to_del, hlayout)
    
    def checkFavouriteFuncs(self):
        self.doNotWarn = True
        for groupbox in self.groupboxes:
            groupbox.checkFavouriteFuncs()
        self.doNotWarn = False


class SetMeasurementsGroupBox(QGroupBox):
    def __init__(
            self, title, itemsText, checkable=True, itemsInfo=None, 
            lastSelection=None, itemsInfoUrls=None, parent=None
        ):
        super().__init__(parent)
        
        if itemsInfo is None:
            itemsInfo = {}
        
        if itemsInfo is None:
            itemsInfoUrls = {}
        
        highlightRgba = _palettes._highlight_rgba()
        r, g, b, a = highlightRgba
        self._highlightStylesheetColor = f'rgb({r}, {g}, {b})'
        
        self.setTitle(title)
        self.setCheckable(checkable)
        
        mainLayout = QVBoxLayout()
        
        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)
        scrollAreaLayout = QVBoxLayout()
        scrollAreaWidget = QWidget()
        self.scrollAreaWidget = scrollAreaWidget
        self.scrollAreaLayout = scrollAreaLayout
        
        self.checkboxes = {}
        for text in itemsText:
            rowLayout = QHBoxLayout()
            infoText = itemsInfo.get(text)
            infoUrl = itemsInfoUrls.get(text)
            if infoText is not None or infoUrl is not None:
                infoButton = widgets_push_buttons.infoPushButton()
                infoButton.setCursor(Qt.WhatsThisCursor)
                rowLayout.addWidget(infoButton)
            
            if infoText is not None:
                infoButton.itemText = text
                infoButton.infoText = infoText
                infoButton.clicked.connect(self.showInfo)
            
            if infoUrl is not None:
                infoButton.itemText = text
                infoButton.infoUrl = infoUrl
                infoButton.clicked.connect(self.openInfoUrl)
                
            checkbox = QCheckBox(text)
            checkbox.setParent(self.scrollAreaWidget)
            checkbox.setChecked(True)
            rowLayout.addWidget(checkbox)
            rowLayout.addStretch(1) 
            
            self.checkboxes[text] = checkbox
            
            scrollAreaLayout.addLayout(rowLayout)

        scrollAreaLayout.addStretch(1)
        
        scrollAreaWidget.setLayout(scrollAreaLayout)
        scrollArea.setWidget(scrollAreaWidget)
        self.scrollArea = scrollArea
        
        buttonsLayout = QHBoxLayout()
        self.selectAllButton = widgets_push_buttons.selectAllPushButton()
        self.selectAllButton.sigClicked.connect(self.setCheckedAll)
        
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(self.selectAllButton)
        self.buttonsLayout = buttonsLayout
        
        if lastSelection is not None:
            self.lastSelection = lastSelection
            self.loadLastSelButton = widgets_push_buttons.reloadPushButton(
                '  Load last selection...  '
            )
            self.loadLastSelButton.clicked.connect(self.loadLastSelection)
            buttonsLayout.addWidget(self.loadLastSelButton)
        
        mainLayout.addWidget(scrollArea)
        mainLayout.addSpacing(10)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
    
    def openInfoUrl(self):
        url = self.sender().infoUrl
        QDesktopServices.openUrl(QUrl(url))
        # import webbrowser
        # url = self.sender().infoUrl
        # webbrowser.open(url)
    
    def getWidthNoScrollBarNeeded(self):
        width = (
            self.scrollArea.verticalScrollBar().sizeHint().width()
            # self.scrollAreaLayout.contentsRect().width()
            + self.scrollAreaWidget.sizeHint().width() 
            + 30
        )
        buttonsWidth = 0
        for i in range(self.buttonsLayout.count()):
            widget = self.buttonsLayout.itemAt(i).widget()
            if not isinstance(widget, QPushButton):
                continue
            buttonsWidth += widget.sizeHint().width() + 16
        largerWidth = max(width, buttonsWidth)
        return largerWidth
    
    def resizeWidthNoScrollBarNeeded(self):
        width = self.getWidthNoScrollBarNeeded()
        self.setMinimumWidth(width)
        # self.setFixedWidth(width)
    
    def loadLastSelection(self):
        for text, checkbox in self.checkboxes.items():
            checked = self.lastSelection.get(text, False)
            checkbox.setChecked(checked)
            
    def showInfo(self):
        infoText = self.sender().infoText
        itemText = self.sender().itemText
        
        title = f'{itemText} description'
        msg = widgets_message_box.myMessageBox()
        msg.setWidth(int(self.screen().size().width()/2))
        msg.information(self, title, infoText)
    
    def setCheckedAll(self, button, checked):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(checked)

    def highlightCheckboxesFromSearchText(self, text):
        for checkbox in self.checkboxes.values():
            highlighted = widgets_search_widgets._matchesSearchText(text, checkbox.text())
            self.setCheckboxHighlighted(highlighted, checkbox)
    
    def setCheckboxHighlighted(self, highlighted, checkbox):
        if highlighted:
            checkbox.setStyleSheet(
                f'background: {self._highlightStylesheetColor}; color: black'
            )
            self.scrollArea.ensureWidgetVisible(checkbox)
        else:
            checkbox.setStyleSheet('')


def askDeleteUncheckedExistingMeasurements(
        parent, unchecked_existing_colnames, acdc_df_path=None,
        locations=None, num_positions=None, exp_foldername=None
    ):
    """Ask whether to delete the existing measurement columns that the
    user did not select. Only the listed columns are deleted. Returns
    `(cancel, delete)`.

    `locations` (`{colname: [Position folder names]}`) and `num_positions`
    (number of Positions checked) show which Positions have a column, when
    not all of them. `exp_foldername` is the experiment folder of the tables.
    """
    msg = widgets_message_box.myMessageBox()
    msg.setWidth(500)
    if acdc_df_path:
        msg.addShowInFileManagerButton(acdc_df_path)
    folder_txt = ''
    if exp_foldername:
        folder_txt = (
            f'Experiment folder: <code>{html.escape(exp_foldername)}</code>'
            '<br><br>'
        )
    txt = html_utils.paragraph(
        f'{folder_txt}'
        'You chose to <b>not save</b> some measurements that are '
        '<b>already present</b> in the saved <code>acdc_output.csv</code> '
        'file.<br><br>'
        'Do you want to <b>delete</b> these measurements or '
        '<b>keep</b> them?<br><br>'
        'Only the listed measurements are deleted.<br><br>'
        'Existing measurements not selected:'
    )
    items = []
    for colname in unchecked_existing_colnames:
        positions = (locations or {}).get(colname) or []
        positions = [pos for pos in positions if pos]
        is_partial = (
            num_positions is not None and num_positions > 1
            and 0 < len(positions) < num_positions
        )
        if is_partial:
            colname = f'{colname}   (only in {", ".join(positions)})'
        items.append(colname)
    listView = widgets_item_views.readOnlyQList(msg)
    listView.addItems(items)
    _, delButton, keepButton = msg.warning(
        parent, 'Unchecked existing measurements', txt,
        widgets=listView, buttonsTexts=('Cancel', 'Delete', 'Keep')
    )
    return msg.cancel, msg.clickedButton == delButton

class SetMeasurementsDialog(dialogs_base.QBaseDialog):
    sigClosed = Signal()
    sigCancel = Signal()
    sigRestart = Signal()

    def __init__(
            self, loadedChNames, notLoadedChNames, isZstack, isSegm3D,
            favourite_funcs=None, parent=None, allPos_acdc_df_cols=None,
            acdc_df_path=None, posData=None, addCombineMetricCallback=None,
            allPosData=None, is_concat=False, isSingleSelection=False,
            state=None, addDoNotAskAgainCheckbox=False,
            allPos_acdc_df_cols_per_pos=None
        ):
        """`allPos_acdc_df_cols` are the columns of the existing tables: on
        OK the dialog asks whether to delete the ones that are measurements
        and are not selected. `allPos_acdc_df_cols_per_pos` (optional) are
        the columns of each table, in the order of `allPosData`: each table
        is then checked with the combined measurements of its Position."""
        super().__init__(parent=parent)
        
        self.checkBoxedGroup = QButtonGroup()
        self.checkBoxedGroup.setExclusive(isSingleSelection)

        _window_stacking.set_window_flags_on_top(self, Qt.Window)

        self.cancel = True

        self.delExistingCols = False
        # True when the user answered the Delete/Keep question about
        # existing measurements that are not selected (see `ok_cb`)
        self.uncheckedExistingAsked = False
        self.existingUncheckedColnames = []
        self.existingUncheckedRps = []
        self.okClicked = False
        self.is_concat = is_concat
        self.allPos_acdc_df_cols = allPos_acdc_df_cols
        self.allPos_acdc_df_cols_per_pos = allPos_acdc_df_cols_per_pos
        self.existingUncheckedLocations = {}
        self.acdc_df_path = acdc_df_path
        self.allPosData = allPosData
        self._posData = posData
        self._isSegm3D = isSegm3D
        self.doNotWarn = False
        self.doNotAskAgain = False

        self.setWindowTitle('Set measurements')
        # self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout()
        
        topLayout = QHBoxLayout()
        
        self.doNotAskAgainToggle = widgets_push_buttons.Toggle(
            label_text='Use the same selection for all following selected experiment folders'
        )
        self.doNotAskAgainToggle.setVisible(addDoNotAskAgainCheckbox)
        topLayout.addWidget(self.doNotAskAgainToggle)
        
        self.searchWidget = widgets_search_widgets.ButtonSearchWidget(loadingType=None)
        self.searchWidget.search_input.setPlaceholderText('Search measurements...')
        self.searchWidget.search_input.setToolTip(
            'Search measurement names and descriptions, including typos and '
            'synonyms. Choose a result with Up/Down and Enter, or click it, '
            'to reveal and highlight the measurement without checking it.'
        )
        self.searchWidget.search_input.setMinimumWidth(250)
        self.searchWidget.search_input.setMaximumWidth(16777215)
        topLayout.addStretch(5)
        topLayout.addWidget(self.searchWidget)
        topLayout.setStretch(0, 0)
        topLayout.setStretch(2, 3)
                
        mainScrollArea = widgets_basic_widgets.ScrollArea()
        self.mainScrollArea = mainScrollArea
        mainScrollAreaWidget = QWidget()
        mainScrollArea.setWidget(mainScrollAreaWidget)
        
        groupsLayout = QGridLayout()
        self.groupsLayout = groupsLayout
        
        mainScrollAreaWidget.setLayout(groupsLayout)
        
        buttonsLayout = QHBoxLayout()

        self.chNameGroupboxes = []
        self.all_metrics = []

        col = 0
        for col, chName in enumerate(loadedChNames):
            channelGBox = channelMetricsQGBox(
                isZstack, chName, isSegm3D, favourite_funcs=favourite_funcs,
                posData=posData, is_concat=is_concat
            )
            channelGBox.chName = chName
            groupsLayout.addWidget(channelGBox, 0, col, 3, 1)
            self.chNameGroupboxes.append(channelGBox)
            channelGBox.sigDelClicked.connect(self.delMixedChannelCombineMetric)
            channelGBox.sigCheckboxToggled.connect(self.channelCheckboxToggled)
            channelGBox.toggled.connect(self.channelGroupboxToggled)
            groupsLayout.setColumnStretch(col, 5)
            self.all_metrics.extend([c.text() for c in channelGBox.checkBoxes])

        current_col = col+1
        for col, chName in enumerate(notLoadedChNames):
            channelGBox = channelMetricsQGBox(
                isZstack, chName, isSegm3D, favourite_funcs=favourite_funcs,
                posData=posData, is_concat=is_concat,
            )
            channelGBox.setChecked(False)
            channelGBox.chName = chName
            groupsLayout.addWidget(channelGBox, 0, current_col, 3, 1)
            self.chNameGroupboxes.append(channelGBox)
            groupsLayout.setColumnStretch(current_col, 5)
            channelGBox.sigDelClicked.connect(self.delMixedChannelCombineMetric)
            channelGBox.sigCheckboxToggled.connect(self.channelCheckboxToggled)
            channelGBox.toggled.connect(self.channelGroupboxToggled)
            current_col += 1
            self.all_metrics.extend([c.text() for c in channelGBox.checkBoxes])

        current_col += 1

        if posData is None:
            isTimelapse = False
        else:
            isTimelapse = posData.SizeT>1
        size_metrics_desc = measurements.get_size_metrics_desc(
            isSegm3D, isTimelapse
        )
        if not isSegm3D:
            size_metrics_desc = {
                key:val for key,val in size_metrics_desc.items()
                if not key.endswith('_3D')
            }
        
        row = 0
        sizeMetricsQGBox = _metricsQGBox(
            size_metrics_desc, 'Physical measurements',
            favourite_funcs=favourite_funcs, isZstack=isZstack,
            addCalcForEachZsliceToggle=isSegm3D,
            doesWarnOnUncheck=True
        )
        self.all_metrics.extend([c.text() for c in sizeMetricsQGBox.checkBoxes])
        self.sizeMetricsQGBox = sizeMetricsQGBox
        for sizeCheckbox in sizeMetricsQGBox.checkBoxes:
            sizeCheckbox.toggled.connect(self.sizeMetricToggled)
        groupsLayout.addWidget(sizeMetricsQGBox, row, current_col)
        groupsLayout.setRowStretch(0, 1)
        groupsLayout.setColumnStretch(current_col, 3)
        row += 1

        props_info_txt_mapper = measurements.get_props_info_txt_mapper(
            isSegm3D=isSegm3D
        )
        rp_desc = props_info_txt_mapper
        regionPropsQGBox = _metricsQGBox(
            rp_desc, 'Morphological properties',
            favourite_funcs=favourite_funcs, isZstack=isZstack
        )
        self.regionPropsQGBox = regionPropsQGBox
        for rpCheckbox in regionPropsQGBox.checkBoxes:
            rpCheckbox.toggled.connect(self.rpMetricToggled)
        groupsLayout.addWidget(regionPropsQGBox, row, current_col)
        groupsLayout.setRowStretch(1, 2)
        self.all_metrics.extend([c.text() for c in regionPropsQGBox.checkBoxes])
        row += 1
        
        # Custom metrics that are channel indipendent
        self.chIndipendCustomeMetricsQGBox = None
        out = measurements.ch_indipend_custom_metrics_desc(
            isZstack, isSegm3D=isSegm3D,
        )
        ch_indipend_custom_metrics_desc = out
        if ch_indipend_custom_metrics_desc:
            self.chIndipendCustomeMetricsQGBox = _metricsQGBox(
                ch_indipend_custom_metrics_desc, 
                'Channel indipendent custom measurements',
                favourite_funcs=favourite_funcs, isZstack=isZstack,
                parent=self
            )
            groupsLayout.addWidget(
                self.chIndipendCustomeMetricsQGBox, row, current_col
            )
            groupsLayout.setRowStretch(1, 1)
            row += 1

        desc, equations = measurements.combine_mixed_channels_desc(
            isSegm3D=isSegm3D, posData=posData, available_cols=self.all_metrics
        )
        self.mixedChannelsCombineMetricsQGBox = None
        if desc:
            self.mixedChannelsCombineMetricsQGBox = _metricsQGBox(
                desc, 'Mixed channels combined measurements',
                favourite_funcs=favourite_funcs, isZstack=isZstack,
                equations=equations, addDelButton=True
            )
            self.mixedChannelsCombineMetricsQGBox.sigDelClicked.connect(
                self.delMixedChannelCombineMetric
            )
            groupsLayout.addWidget(
                self.mixedChannelsCombineMetricsQGBox, row, current_col
            )
            groupsLayout.setRowStretch(1, 1)
            if not self.is_concat:
                self.setDisabledMetricsRequestedForCombined(False)
                self.mixedChannelsCombineMetricsQGBox.toggled.connect(
                    self.setDisabledMetricsRequestedForCombined
                )
                for combCheckbox in self.mixedChannelsCombineMetricsQGBox.checkBoxes:
                    combCheckbox.toggled.connect(
                        self.setDisabledMetricsRequestedForCombined
                    )
            else:
                for combCheckbox in self.mixedChannelsCombineMetricsQGBox.checkBoxes:
                    combCheckbox.toggled.connect(
                        self.mixedChannelsMetricToggled
                    )
            row += 1

        self.last_row = row
        self.last_col = current_col

        okButton = widgets_push_buttons.okPushButton('   Ok   ')
        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')
        if addCombineMetricCallback is not None:
            addCombineMetricButton = widgets_push_buttons.addPushButton(
                'Add combined measurement...'
            )
            addCombineMetricButton.clicked.connect(addCombineMetricCallback)
        self.okButton = okButton

        loadLastSelButton = widgets_push_buttons.reloadPushButton('Load last selection...')
        self.selectAllButton = widgets_push_buttons.selectAllPushButton()

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        if isSegm3D:
            selectAll3DButton = widgets_push_buttons.selectAllPushButton(
                suffix_text_select='only 3D metrics',
                suffix_text_deselect='all 3D metrics'
            )
            selectAll3DButton.sigClicked.connect(self.checkAll3D)
            self.selectAll3DButton = selectAll3DButton
            buttonsLayout.addWidget(selectAll3DButton)
        buttonsLayout.addWidget(self.selectAllButton)
        buttonsLayout.addSpacing(20)
        
        if addCombineMetricCallback is not None:
            buttonsLayout.addWidget(addCombineMetricButton)
            buttonsLayout.addSpacing(20)
        
        saveCurrentSelectionButton = widgets_push_buttons.savePushButton(
            'Save current selection...'
        )
        saveCurrentSelectionButton.clicked.connect(
            self.saveCurrentSelectionClicked
        )
        
        buttonsLayout.addWidget(saveCurrentSelectionButton)
        
        loadSavedSelectionButton = widgets_push_buttons.OpenFilePushButton(
            'Load saved selection...'
        )
        loadSavedSelectionButton.clicked.connect(self.loadSavedSelectionClicked)
        buttonsLayout.addWidget(loadSavedSelectionButton)
        
        buttonsLayout.addWidget(loadLastSelButton)
        
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        self.okButton = okButton

        layout.addLayout(topLayout)
        layout.addSpacing(10)
        # layout.addLayout(groupsLayout)
        layout.addWidget(mainScrollArea)
        layout.addLayout(buttonsLayout)

        self.setLayout(layout)
        
        if state is not None:
            self.setState(state)

        self.searchWidget.search_input.textChanged.connect(self.searchAndHighlight)
        self.searchWidget.sigTriggerBlink.connect(self.highlightSearchMeasurement)
        self.updateMeasurementSearchRecords()
        self.selectAllButton.sigClicked.connect(self.selectAll)
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        loadLastSelButton.clicked.connect(self.loadLastSelection)
        
        self.addCheckboxesToGroup()

        self.updateRequiredSizeMetrics()
    
    def allMetricsDict(self):
        all_metrics = {
            'standard': {}, 
            'regionprop': [],
            'size': [],
            'mixed_channels': []
        }
        for chNameGroupbox in self.chNameGroupboxes:
            channel_name = chNameGroupbox.chName
            for checkBox in chNameGroupbox.checkBoxes:
                if channel_name not in all_metrics['standard']:
                    all_metrics['standard'][channel_name] = []
                all_metrics['standard'][channel_name].append(checkBox.text())
        
        for checkBox in self.regionPropsQGBox.checkBoxes:
            all_metrics['regionprop'].append(checkBox.text())
        
        for checkBox in self.sizeMetricsQGBox.checkBoxes:
            all_metrics['size'].append(checkBox.text())
        
        if self.chIndipendCustomeMetricsQGBox is not None:
            checkBoxes = self.chIndipendCustomeMetricsQGBox.checkBoxes
            for checkBox in checkBoxes:
                all_metrics['ch_indipend_custom_metric'].append(checkBox.text())
        
        if self.mixedChannelsCombineMetricsQGBox is None:
            return
        
        checkBoxes = self.mixedChannelsCombineMetricsQGBox.checkBoxes
        for checkBox in checkBoxes:
            all_metrics['mixed_channels'].append(checkBox.text())
        
        return all_metrics
    
    def measurementSearchGroups(
        self,
    ) -> 'Generator[Tuple[_metricsQGBox, str], None, None]':
        for chNameGroupbox in self.chNameGroupboxes:
            for groupbox in chNameGroupbox.groupboxes:
                yield groupbox, chNameGroupbox.chName
            customGroupbox = getattr(
                chNameGroupbox, 'customMetricsQGBox', None
            )
            if customGroupbox is not None:
                yield customGroupbox, chNameGroupbox.chName
        yield self.regionPropsQGBox, ''
        yield self.sizeMetricsQGBox, ''
        if self.chIndipendCustomeMetricsQGBox is not None:
            yield self.chIndipendCustomeMetricsQGBox, ''
        if self.mixedChannelsCombineMetricsQGBox is not None:
            yield self.mixedChannelsCombineMetricsQGBox, ''

    def updateMeasurementSearchRecords(self):
        records = {}
        for groupbox, channel in self.measurementSearchGroups():
            context = f'{channel}: {groupbox.title()}' if channel else groupbox.title()
            for checkbox in groupbox.checkBoxes:
                if checkbox.isHidden():
                    continue
                name = checkbox.text()
                display = f'{name} ({context})'
                records[display] = {
                    'display': display,
                    'search_name': display,
                    'synonyms': self.searchWidget._synonymsForName(name),
                    'tooltip': html_utils.to_plain_text(
                        groupbox.metricDescriptions.get(name, '')
                    ),
                    'target': checkbox,
                }
        self.searchWidget.setSearchRecords(records)

    def stopMeasurementSearchBlink(self):
        for blinker in self.findChildren(qutils.QControlBlink):
            blinker.stop()

    def searchAndHighlight(self, text):
        self.stopMeasurementSearchBlink()
        for groupbox, _ in self.measurementSearchGroups():
            groupbox.highlightCheckboxesFromSearchText(text)

    def highlightSearchMeasurement(self, checkbox):
        """Reveal and blink a popup result without changing its checked state."""
        self.stopMeasurementSearchBlink()
        checkbox.scrollArea.ensureWidgetVisible(checkbox)
        self.mainScrollArea.ensureWidgetVisible(checkbox.scrollArea)
        blinker = qutils.QControlBlink(checkbox, qparent=self)
        blinker.start()
    
    def selectedMetricNameAndGroup(self):
        for chNameGroupbox in self.chNameGroupboxes:
            for checkBox in chNameGroupbox.checkBoxes:
                if checkBox.isChecked():
                    return checkBox.text(), {'standard': chNameGroupbox.chName}
        
        for checkBox in self.regionPropsQGBox.checkBoxes:
            if checkBox.isChecked():
                return checkBox.text(), 'regionprop'
        
        for checkBox in self.sizeMetricsQGBox.checkBoxes:
            if checkBox.isChecked():
                return checkBox.text(), 'size'
        
        if self.chIndipendCustomeMetricsQGBox is not None:
            checkBoxes = self.chIndipendCustomeMetricsQGBox.checkBoxes
            for checkBox in checkBoxes:
                if checkBox.isChecked():
                    return checkBox.text(), 'ch_indipend_custom_metric'
        
        if self.mixedChannelsCombineMetricsQGBox is None:
            return
        
        checkBoxes = self.mixedChannelsCombineMetricsQGBox.checkBoxes
        for checkBox in checkBoxes:
            if checkBox.isChecked():
                return checkBox.text(), 'mixed_channels'
    
    def selectedMetricGroup(self):
        for chNameGroupbox in self.chNameGroupboxes:
             for checkBox in chNameGroupbox.checkBoxes:
                if checkBox.isChecked():
                    return checkBox.text()
        
        for checkBox in self.regionPropsQGBox.checkBoxes:
            if checkBox.isChecked():
                return checkBox.text()
        
        for checkBox in self.sizeMetricsQGBox.checkBoxes:
            if checkBox.isChecked():
                return checkBox.text()
        
        if self.chIndipendCustomeMetricsQGBox is not None:
            checkBoxes = self.chIndipendCustomeMetricsQGBox.checkBoxes
            for checkBox in checkBoxes:
                if checkBox.isChecked():
                    return checkBox.text()
        
        if self.mixedChannelsCombineMetricsQGBox is None:
            return
        
        checkBoxes = self.mixedChannelsCombineMetricsQGBox.checkBoxes
        for checkBox in checkBoxes:
            if checkBox.isChecked():
                return checkBox.text()
    
    def addCheckboxesToGroup(self):
        for chNameGroupbox in self.chNameGroupboxes:
            for checkBox in chNameGroupbox.checkBoxes:
                self.checkBoxedGroup.addButton(checkBox)
        
        for checkBox in self.regionPropsQGBox.checkBoxes:
            self.checkBoxedGroup.addButton(checkBox)
        
        for checkBox in self.sizeMetricsQGBox.checkBoxes:
            self.checkBoxedGroup.addButton(checkBox)
        
        if self.chIndipendCustomeMetricsQGBox is not None:
            checkBoxes = self.chIndipendCustomeMetricsQGBox.checkBoxes
            for checkBox in checkBoxes:
                self.checkBoxedGroup.addButton(checkBox)
        
        if self.mixedChannelsCombineMetricsQGBox is None:
            return
        
        checkBoxes = self.mixedChannelsCombineMetricsQGBox.checkBoxes
        for checkBox in checkBoxes:
            self.checkBoxedGroup.addButton(checkBox)
            
    def channelCheckboxToggled(self, checkbox):
        # Make sure to automatically check the requested cell_vol metric for
        # concentration metrics
        if checkbox.text().find('concentration_') == -1:
            return

        self.updateRequiredSizeMetrics()

    def channelGroupboxToggled(self, checked):
        # Concentration metrics of an unchecked channel do not require the
        # cell volume anymore
        self.updateRequiredSizeMetrics()

    def updateRequiredSizeMetrics(self):
        """Check the cell volume metrics required by the selected
        concentration metrics (they cannot be unchecked, see
        `sizeMetricToggled`) and release the ones not required anymore.
        """
        if self.is_concat:
            # When this dialogue is used in concatenate pos utility we do not
            # need to check that certain metrics are present
            return

        pattern = r'.+_from_vol_([a-z]+)(_3D)?(_?[A-Za-z0-9]*)'
        repl = r'cell_vol_\1\2'
        required = set()
        for channelGBox in self.chNameGroupboxes:
            for colname in channelGBox.selectedMetrics():
                if colname.find('concentration_') == -1:
                    continue
                required.add(re.sub(pattern, repl, colname))

        for sizeCheckbox in self.sizeMetricsQGBox.checkBoxes:
            isRequired = sizeCheckbox.text() in required
            sizeCheckbox.isRequired = isRequired
            if isRequired and not sizeCheckbox.isChecked():
                sizeCheckbox.setChecked(True)
    
    def rpMetricToggled(self, checked):
        pass
    
    def mixedChannelsMetricToggled(self, checked):
        pass
    
    def sizeMetricToggled(self, checked):
        """Method called when a checkbox of a size metric is toggled.
        Check if the size value is required and explain why it cannot be 
        unchecked.

        Parameters
        ----------
        checked : bool
            State of the checkbox toggled
        """        
        if self.is_concat:
            # When this dialogue is used in concatenate pos utility we do not 
            # need to check that certain metrics are present
            return

        checkbox = self.sender()
        if not hasattr(checkbox, 'isRequired'):
            return
        
        if not checkbox.isRequired:
            return
        
        if checkbox.isChecked():
            return
        
        checkbox.setChecked(True)
        if self.doNotWarn:
            return

        linked_autoBkgr_metric = checkbox.text().replace('cell', '_autoBkgr_from')
        linked_dataPrepBkgr_metric = checkbox.text().replace(
            'cell', '_dataPrepBkgr_from'
        )
        txt = html_utils.paragraph(f"""
            <b>This physical measurement cannot be unchecked</b> 
            because it is required 
            by the <code>{linked_autoBkgr_metric}</code> and 
            <code>{linked_dataPrepBkgr_metric}</code> measurements 
            that you requested to save.<br><br>

            Thank you for you patience!
        """)
        msg = widgets_message_box.myMessageBox(showCentered=False)
        msg.warning(self, 'Physical measurement required', txt)

    def checkAll3D(self, button, checked):
        for chNameGroupbox in self.chNameGroupboxes:
            for gb in chNameGroupbox.groupboxes:
                if not gb.isChecked():
                    continue
                gb.selectAll3DButton.setChecked(checked)
                gb.checkAll3D(None, checked)
            cgb = getattr(chNameGroupbox, 'customMetricsQGBox', None)
            if cgb is not None and cgb.isChecked():
                cgb.selectAll3DButton.setChecked(checked)
                cgb.checkAll3D(None, checked)

    def selectAll(self, button, checked):
        self.doNotWarn = True
        if self.mixedChannelsCombineMetricsQGBox is not None and not checked:
            # First release the measurements forced by combined measurements,
            # else they would be restored after being deselected
            self.mixedChannelsCombineMetricsQGBox.checkAll(None, checked)
        for chNameGroupbox in self.chNameGroupboxes:
            for gb in chNameGroupbox.groupboxes:
                gb.checkAll(None, checked)
            cgb = getattr(chNameGroupbox, 'customMetricsQGBox', None)
            if cgb is not None:
                cgb.checkAll(None, checked)
        
        self.sizeMetricsQGBox.checkAll(None, checked)
        self.regionPropsQGBox.checkAll(None, checked)
        if self.chIndipendCustomeMetricsQGBox is not None:
            self.chIndipendCustomeMetricsQGBox.checkAll(None, checked)
            
        if self.mixedChannelsCombineMetricsQGBox is not None:
            self.mixedChannelsCombineMetricsQGBox.checkAll(None, checked)
        self.doNotWarn = False
    
    def delMixedChannelCombineMetric(self, colname_to_del, hlayout):
        cp = measurements.read_saved_user_combine_config()
        for section in cp.sections():
            cp.remove_option(section, colname_to_del)
        measurements.save_common_combine_metrics(cp)

        for i in range(hlayout.count()):
            item = hlayout.itemAt(i)
            w = item.widget()
            if w is None:
                continue
            w.hide()

        self.updateMeasurementSearchRecords()
               
        if self.allPosData is not None:
            for posData in self.allPosData:
                _config = posData.combineMetricsConfig
                for section in _config.sections():
                    _config.remove_option(section, colname_to_del)
                posData.saveCombineMetrics()
    
    def _metricsGroupboxes(self):
        """Groups whose `selectedMetrics()` give the selected measurements, 
        except the channel groups."""
        groupboxes = [self.sizeMetricsQGBox, self.regionPropsQGBox]
        if self.chIndipendCustomeMetricsQGBox is not None:
            groupboxes.append(self.chIndipendCustomeMetricsQGBox)
        if self.mixedChannelsCombineMetricsQGBox is not None:
            groupboxes.append(self.mixedChannelsCombineMetricsQGBox)
        return groupboxes
    
    def _stateGroupboxes(self):
        """(group, state key, mapper key) of every group of measurements"""
        groups = [
            (gb, gb.title(), gb.chName) for gb in self.chNameGroupboxes
        ]
        groups.extend(
            (gb, gb.title(), gb.title()) for gb in self._metricsGroupboxes()
        )
        return groups
    
    def mixedChannelsSelectedMetrics(self):
        groupbox = self.mixedChannelsCombineMetricsQGBox
        if groupbox is None:
            return []
        
        return [
            checkBox.text() for checkBox in groupbox.checkBoxes
            if groupbox.isChecked() and checkBox.isChecked() 
            and not checkBox.isHidden()
        ]
    
    def _groupSelectedMetrics(self, groupbox):
        if groupbox is self.mixedChannelsCombineMetricsQGBox:
            return self.mixedChannelsSelectedMetrics()
        
        return groupbox.selectedMetrics()
    
    def selectedMeasurements(self):
        """Names of all the selected measurements (what the measurements 
        kernel will compute, see 
        `cli.ComputeMeasurementsKernel.set_metrics_from_set_measurements_dialog`)
        """
        selected = []
        for groupbox, _, _ in self._stateGroupboxes():
            selected.extend(self._groupSelectedMetrics(groupbox))
        return selected
    
    def offeredMeasurements(self):
        """Names of all the measurements that can be selected"""
        offered = []
        for groupbox, _, _ in self._stateGroupboxes():
            offered.extend(checkBox.text() for checkBox in groupbox.checkBoxes)
        return offered
    
    def _setSelection(self, selection, missing_unchecked=False):
        """Set the selection from `{key: {colname: checked}}` where `key` is
        the state key (`state`) or the mapper key
        (`currentSelectionMapper`) of each group.

        A group missing from `selection` keeps its current state (e.g. a
        group created after the selection was saved) and a group with an
        empty entry is unchecked. Measurements missing from a group entry
        are unchecked if `missing_unchecked` (mappers list only the selected
        ones), else they keep their current state (e.g. a combined
        measurement added after `state` was saved).
        """
        self.doNotWarn = True
        self._isSettingSelection = True
        for chNameGroupbox in self.chNameGroupboxes:
            chNameGroupbox.doNotWarn = True
        # Measurements forced by combined measurements get the state of the
        # selection (re-forced at the end if still required)
        for checkBox in self._forcedByCombinedCheckboxes():
            checkBox.forcedByCombined = False
            checkBox.setDisabled(False)
            checkBox.setToolTip('')
        try:
            for groupbox, key, alt_key in selection.pop('_groups'):
                if key not in selection:
                    continue
                
                groupSelection = selection[key]
                if not groupSelection:
                    groupbox.setChecked(False)
                    continue
                
                groupbox.setChecked(True)
                subGroupboxes = getattr(groupbox, 'subGroupboxes', None)
                if subGroupboxes is not None:
                    for subGroupbox in subGroupboxes():
                        subGroupbox.setChecked(True)
                for checkBox in groupbox.checkBoxes:
                    colname = checkBox.text()
                    if colname not in groupSelection:
                        if missing_unchecked:
                            checkBox.setChecked(False)
                        continue
                    checkBox.setChecked(bool(groupSelection[colname]))
        finally:
            self.doNotWarn = False
            self._isSettingSelection = False
            for chNameGroupbox in self.chNameGroupboxes:
                chNameGroupbox.doNotWarn = False
        
        self.updateRequiredSizeMetrics()
        self.setDisabledMetricsRequestedForCombined()
    
    def setState(self, state):
        """Restore a selection returned by `state`"""
        groups = self._stateGroupboxes()
        selection = {key: state[key] for _, key, _ in groups if key in state}
        selection['_groups'] = groups
        self._setSelection(selection)
            
    def state(self):
        """Current selection as `{group title: {colname: selected}}`. 

        Unchecked groups have an empty dictionary. Measurements left checked 
        in an unchecked sub-group are not selected (see `selectedMetrics`).
        """
        state = {}
        for groupbox, key, _ in self._stateGroupboxes():
            selected = set(self._groupSelectedMetrics(groupbox))
            if not selected and not groupbox.isChecked():
                state[key] = {}
                continue
            
            state[key] = {
                checkBox.text(): checkBox.text() in selected
                for checkBox in groupbox.checkBoxes
            }
        
        return state
    
    def restoreState(self, state):
        self.setState(state)
    
    def currentSelectionMapper(self):
        current_selected_meas = defaultdict(dict)
        for groupbox, _, key in self._stateGroupboxes():
            for colname in self._groupSelectedMetrics(groupbox):
                current_selected_meas[key][colname] = 'Yes'
        
        return current_selected_meas
    
    def saveCurrentSelectionClicked(self):
        current_selection_mapper = self.currentSelectionMapper()
        defaultEntry = '_and_'.join(current_selection_mapper.keys())
        defaultEntry = defaultEntry.replace(' ', '_').lower()
        saved_selections = io.get_saved_measurements_selections()
        win = dialogs_file_selection.filenameDialog(
            basename='',
            ext='',
            hintText='Insert a <b>name</b> for the current selection:',
            existingNames=saved_selections,
            allowEmpty=False,
            defaultEntry=defaultEntry
        )
        win.exec_()
        if win.cancel:
            return
        
        filename = win.filename
        ini_filepath = io.save_measurements_selections(
            filename, current_selection_mapper)
        
        msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
        txt = html_utils.paragraph(f"""
            Done!<br><br>
            Current selection saved with name <code>{filename}</code> at 
            the following path:
        """)
        msg.information(
            self, 'Selection saved', txt, 
            commands=(ini_filepath,),
            path_to_browse=os.path.dirname(ini_filepath),
        )
    
    def loadSavedSelectionClicked(self):
        self.doNotWarn = True
        
        saved_selections = io.get_saved_measurements_selections()
        
        selectNameWin = widgets_item_views.QDialogListbox(
            'Choose selection to load',
            'Choose selection to load:\n',
            saved_selections,
            multiSelection=False,
            parent=self,
            allowEmptySelection=False
        )
        selectNameWin.exec_()
        if selectNameWin.cancel:
            return
        
        selection_mapper = (
            io.read_measurements_selections(selectNameWin.selectedItemsText[0])
        )
        
        self.setCurrentSelectionFromMapper(selection_mapper)
        
        self.doNotWarn = False
    
    def saveLastSelection(self):
        last_selected_meas = self.currentSelectionMapper()        
        load.write_last_selected_set_measurements(last_selected_meas)
    
    def setCurrentSelectionFromMapper(self, selection_mapper):
        """Set the selection saved by `currentSelectionMapper` (groups 
        missing from the mapper are unchecked)."""
        groups = [
            (groupbox, mapper_key, state_key) 
            for groupbox, state_key, mapper_key in self._stateGroupboxes()
        ]
        selection = {
            mapper_key: selection_mapper.get(mapper_key) or {}
            for _, mapper_key, _ in groups
        }
        selection['_groups'] = groups
        self._setSelection(selection, missing_unchecked=True)

    def loadLastSelection(self):
        self.doNotWarn = True
        last_selected_meas = load.read_last_selected_set_measurements()
        last_selected_meas = dict(last_selected_meas)
        
        self.setCurrentSelectionFromMapper(last_selected_meas)
        
        self.doNotWarn = False
    
    def _forcedByCombinedCheckboxes(self):
        allCheckboxes = []
        for chNameGroupbox in self.chNameGroupboxes:
            allCheckboxes.extend(chNameGroupbox.checkBoxes)
        allCheckboxes.extend(self.sizeMetricsQGBox.checkBoxes)
        allCheckboxes.extend(self.regionPropsQGBox.checkBoxes)
        return [
            cb for cb in allCheckboxes
            if getattr(cb, 'forcedByCombined', False)
        ]

    def setDisabledMetricsRequestedForCombined(self, checked=None):
        """Check and disable the measurements used by the checked mixed 
        channels combined measurements and restore the previous state of 
        the ones not used anymore."""
        if self.is_concat:
            # When this dialogue is used in concatenate pos utility we do not 
            # need to check that certain metrics are present
            return
        
        if self.mixedChannelsCombineMetricsQGBox is None:
            return

        if getattr(self, '_isSettingSelection', False):
            # Called once at the end of `_setSelection`
            return

        requiredBy = {}
        for combCheckbox in self.mixedChannelsCombineMetricsQGBox.checkBoxes:
            if combCheckbox.isHidden():
                continue
            if combCheckbox.text() not in self.mixedChannelsSelectedMetrics():
                continue
            equation = getattr(combCheckbox, 'equation', '')
            for name, _, _ in measurements.equation_variable_spans(equation):
                requiredBy.setdefault(name, combCheckbox.text())
        
        allCheckboxes = []
        for chNameGroupbox in self.chNameGroupboxes:
            allCheckboxes.extend(chNameGroupbox.checkBoxes)
        allCheckboxes.extend(self.sizeMetricsQGBox.checkBoxes)
        allCheckboxes.extend(self.regionPropsQGBox.checkBoxes)
        
        for cb in allCheckboxes:
            combinedName = requiredBy.get(cb.text())
            wasForced = getattr(cb, 'forcedByCombined', False)
            if combinedName is not None:
                if not wasForced:
                    cb.checkedBeforeForced = cb.isChecked()
                cb.forcedByCombined = True
                cb.setChecked(True)
                cb.setDisabled(True)
                cb.setToolTip(
                    'This metric cannot be removed because it is required '
                    f'by the combined measurement "{combinedName}"'
                )
            elif wasForced:
                cb.forcedByCombined = False
                cb.setDisabled(False)
                cb.setToolTip('')
                cb.setChecked(getattr(cb, 'checkedBeforeForced', True))

    def keyPressEvent(self, a0: QKeyEvent) -> None:
        state = self.state()
        return super().keyPressEvent(a0)
    
    def closeEvent(self, event):
        self._closedWhileAsking = True
        if self.cancel:
            self.sigCancel.emit()
        super().closeEvent(event)

    def restart(self):
        self.cancel = False
        self.close()
        self.sigRestart.emit()
    
    def setDisabledNotExistingMeasurements(self, existing_colnames):
        self.existing_colnames = existing_colnames
        for chNameGroupbox in self.chNameGroupboxes:
            for checkBox in chNameGroupbox.checkBoxes:
                colname = checkBox.text()
                if colname in existing_colnames:
                    checkBox.setChecked(True)
                    continue
                
                checkBox.setChecked(False)
                checkBox.setDisabled(True)
                self.setNotExistingMeasurementTooltip(checkBox)
        
        for checkBox in self.sizeMetricsQGBox.checkBoxes:
            colname = checkBox.text()
            if colname in existing_colnames:
                checkBox.setChecked(True)
                continue
            checkBox.setChecked(False)
            checkBox.setDisabled(True)
            self.setNotExistingMeasurementTooltip(checkBox)
        
        for checkBox in self.regionPropsQGBox.checkBoxes:
            prop_name = checkBox.text()
            for existing_col in existing_colnames:
                if prop_name == existing_col:
                    checkBox.setChecked(True)
                    break
                m = re.match(fr'{prop_name}-\d', existing_col)
                if m is not None:
                    checkBox.setChecked(True)
                    break
            else:
                checkBox.setChecked(False)
                checkBox.setDisabled(True)
                self.setNotExistingMeasurementTooltip(checkBox)
        
        if self.mixedChannelsCombineMetricsQGBox is None:
            return
        
        for combCheckbox in self.mixedChannelsCombineMetricsQGBox.checkBoxes:
            colname = combCheckbox.text()
            if colname in existing_colnames:
                combCheckbox.setChecked(True)
                continue
            combCheckbox.setChecked(False)
            combCheckbox.setDisabled(True)
            self.setNotExistingMeasurementTooltip(combCheckbox)
        
    def addNonMeasurementColumns(self, colnames):
        additionalCols = measurements.get_non_measurements_cols(
            colnames, self.all_metrics
        )
        if not additionalCols:
            return
        self.nonMeasurementsGroupbox = CheckboxesGroupBox(
            additionalCols, title='Additional columns', checkable=True
        )
        self.groupsLayout.addWidget(
            self.nonMeasurementsGroupbox, 0, self.last_col+1, self.last_row+1, 1
        )
            
    
    def setNotExistingMeasurementTooltip(self, checkBox):
        checkBox.setToolTip(
            'Measurement is disabled because it is not present in selected '
            'acdc_output tables, hence it cannot be addded to concatenated '
            'table. '
        )

    def _updateCalcForEachZsliceRequested(self):
        for chNameGroupbox in self.chNameGroupboxes:
            chNameGroupbox.calcForEachZsliceRequested = (
                chNameGroupbox.isCalcForEachZsliceRequested()
            )

        self.sizeMetricsQGBox.calcForEachZsliceRequested = (
            self.sizeMetricsQGBox.isCalcForEachZsliceRequested()
        )

    def ok_cb(self):
        self._updateCalcForEachZsliceRequested()
                
        if self.allPos_acdc_df_cols is None:
            self.saveLastSelection()
            self.cancel = False
            self.doNotAskAgain = self.doNotAskAgainToggle.isChecked()
            self.close()
            self.sigClosed.emit()
            return

        self.okClicked = True
        unchecked_existing_colnames = self.uncheckedExistingMeasurements()
        self.existingUncheckedColnames = unchecked_existing_colnames
        # Kept for compatibility: regionprops columns (also the 
        # `<prop>-<i>` ones) are listed in `existingUncheckedColnames`
        self.existingUncheckedRps = []
        self.delExistingCols = False
        self.uncheckedExistingAsked = False
        if unchecked_existing_colnames:
            # The question is not modal --> no OK/Cancel meanwhile
            self._closedWhileAsking = False
            self.setDisabled(True)
            try:
                cancel, self.delExistingCols = (
                    self.warnUncheckedExistingMeasurements(
                        unchecked_existing_colnames, []
                    )
                )
            finally:
                self.setDisabled(False)
            if cancel or self._closedWhileAsking:
                return
            self.uncheckedExistingAsked = True
        
        self.saveLastSelection()
        self.cancel = False  
        self.doNotAskAgain = self.doNotAskAgainToggle.isChecked()
        self.close()
        self.sigClosed.emit()
    
    def measurementChannels(self):
        channels = [gb.chName for gb in self.chNameGroupboxes]
        for posData in (self.allPosData or []):
            for channel in (getattr(posData, 'chNames', None) or []):
                if channel not in channels:
                    channels.append(channel)
        return channels
    
    def selectionKernel(self):
        """Measurements kernel set with the current selection, as the one
        that computes and saves the measurements
        (`cli.ComputeMeasurementsKernel.set_metrics_from_set_measurements_dialog`).
        """
        from cellacdc import cli
        self._updateCalcForEachZsliceRequested()
        kernel = cli.ComputeMeasurementsKernel(None, '', False)
        kernel.init_args(self.measurementChannels(), '')
        kernel.isSegm3D = bool(self._isSegm3D)
        kernel.set_metrics_from_set_measurements_dialog(self)
        return kernel

    def _existingTablesSources(self, existing_colnames=None):
        """`[(posData, columns)]` of the existing tables to check"""
        if existing_colnames is not None:
            return [(self._posData, existing_colnames)]

        perPos = self.allPos_acdc_df_cols_per_pos
        allPosData = self.allPosData or []
        if perPos is not None and len(perPos) == len(allPosData):
            return list(zip(allPosData, perPos))

        return [(self._posData, self.allPos_acdc_df_cols or [])]

    def uncheckedExistingMeasurements(self, existing_colnames=None):
        """Measurement columns of the existing tables that the current
        selection would not save, as classified by the measurements kernel
        (`cli.ComputeMeasurementsKernel.unselected_existing_measurements`,
        each table with the combined measurements of its Position). Sets
        `existingUncheckedLocations` (`{colname: [Position folder names]}`).
        """
        kernel = self.selectionKernel()
        unchecked = []
        locations = {}
        for posData, colnames in self._existingTablesSources(existing_colnames):
            if posData is None:
                posData = SimpleNamespace(
                    chNames=self.measurementChannels(),
                    combineMetricsConfig=None
                )
            pos_foldername = getattr(posData, 'pos_foldername', None)
            unselected = kernel.unselected_existing_measurements(
                colnames, posData
            )
            for colname in unselected:
                if colname not in locations:
                    locations[colname] = []
                    unchecked.append(colname)
                if pos_foldername not in locations[colname]:
                    locations[colname].append(pos_foldername)
        self.existingUncheckedLocations = locations
        return unchecked

    def warnUncheckedExistingMeasurements(
            self, unchecked_existing_colnames, unchecked_existing_rps
        ):
        items = [*unchecked_existing_colnames, *unchecked_existing_rps]
        sources = self._existingTablesSources()
        exp_path = getattr(self._posData, 'exp_path', None)
        return askDeleteUncheckedExistingMeasurements(
            self, items, acdc_df_path=self.acdc_df_path,
            locations=self.existingUncheckedLocations,
            num_positions=len(sources),
            exp_foldername=os.path.basename(exp_path) if exp_path else None
        )

    def show(self, block=False):
        super().show(block=False)
        self.selectAllButton.setMinimumHeight(self.okButton.height())
        screenWidth = self.screen().size().width()
        screenHeight = self.screen().size().height()
        screenLeft = self.screen().geometry().x()
        screenTop = self.screen().geometry().y()
        h = screenHeight-200
        minColWith = screenWidth/5
        w = minColWith*(self.last_col+1)
        xLeft = int((screenWidth-w)/2)
        if w > screenWidth:
            self.move(screenLeft+10, screenTop+50)
            self.resize(screenWidth-20, h)
        else:
            self.move(screenLeft+xLeft, screenTop+50)
            self.resize(int(w), h)
        super().show(block=block)


class ComputeMetricsErrorsDialog(dialogs_base.QBaseDialog):
    def __init__(
            self, errorsDict, log_path='', parent=None, 
            log_type='custom_metrics'
        ):
        super().__init__(parent)

        self.errorsDict = errorsDict

        layout = QGridLayout()

        self.setWindowTitle('Errors summary')
        
        label = QLabel(self)
        standardIcon = getattr(QStyle, 'SP_MessageBoxWarning')
        icon = self.style().standardIcon(standardIcon)
        pixmap = icon.pixmap(60, 60)
        label.setPixmap(pixmap)
        layout.addWidget(label, 0, 0, alignment=Qt.AlignTop)

        if log_type == 'custom_metrics':
            infoText = ("""
                When computing <b>custom metrics</b> the following metrics 
                were <b>ignored</b> because they raised an <b>error</b>.<br><br>
            """)
        elif log_type == 'standard_metrics':
            infoText = ("""
                Some or all of the <b>standard metrics</b> were <b>NOT saved</b> 
                because Cell-ACDC encoutered the following errors.<br><br>
            """)
        elif log_type == 'region_props':
            rp_url = 'https://scikit-image.org/docs/0.18.x/api/skimage.measure.html#skimage.measure.regionprops'
            rp_href = f'<a href="{rp_url}">skimage.measure.regionprops</a>'
            infoText = (f"""
                <b>Region properties</b> were <b>NOT saved</b> because Cell-ACDC 
                encoutered the following errors.<br>
                Region properties are calculated using the <code>scikit-image</code> 
                function called <code>{rp_href}</code>.<br><br>
            """)
        elif log_type == 'missing_annot':
            infoText = ("""
                The following Positions were <b>SKIPPED</b> because they did 
                <b>not have cell cycle annotations</b>.<br><br>
                To add lineage tree information you first need to <b>do the 
                cell cycle analysis</b> in the GUI (launcher button
                "Correction: launch GUI...").<br><br>
            """)
        else:
            infoText = ("""
                Process raised the errors listed below.<br><br>
            """)

        github_issues_href = f'<a href={issues_url}>here</a>'   
        noteText = (f"""
            NOTE: If you <b>need help</b> understanding these errors you can 
            <b>open an issue</b> on our github page {github_issues_href}.
        """)
   
        infoLabel = QLabel(html_utils.paragraph(f'{infoText}{noteText}'))
        infoLabel.setOpenExternalLinks(True)
        layout.addWidget(infoLabel, 0, 1)

        scrollArea = QScrollArea()
        scrollAreaWidget = QWidget()  
        textLayout = QVBoxLayout()
        for func_name, traceback_format in errorsDict.items():
            nameLabel = QLabel(f'<b>{func_name}</b>: ')
            errorMessage = f'\n{traceback_format}'
            errorLabel = QLabel(errorMessage)
            errorLabel.setTextInteractionFlags(
                Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
            )
            # errorLabel.setStyleSheet("background-color: white")
            errorLabel.setFrameShape(QFrame.Shape.Panel)
            errorLabel.setFrameShadow(QFrame.Shadow.Sunken)
            textLayout.addWidget(nameLabel)
            textLayout.addWidget(errorLabel)
            textLayout.addStretch(1)
        
        scrollAreaWidget.setLayout(textLayout)
        scrollArea.setWidget(scrollAreaWidget)
        
        layout.addWidget(scrollArea, 1, 1)

        buttonsLayout = QHBoxLayout()
        showLogButton = widgets_push_buttons.showInFileManagerButton('Show log file...')
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(showLogButton)

        copyButton = widgets_push_buttons.copyPushButton('Copy error message')
        copyButton.clicked.connect(self.copyErrorMessage)
        buttonsLayout.addWidget(copyButton)
        self.copyButton = copyButton
        self.copyButton.text = 'Copy error message'
        self.copyButton.icon = self.copyButton.icon()
        
        okButton = widgets_push_buttons.okPushButton(' Ok ')
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        showLogButton.clicked.connect(partial(filesystem.showInExplorer, log_path))
        okButton.clicked.connect(self.close)
        layout.setVerticalSpacing(10)
        layout.addLayout(buttonsLayout, 2, 1)

        self.setLayout(layout)
        self.setFont(fonts.font)
    
    def copyErrorMessage(self):
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        copiedText = ''
        for _, traceback_format in self.errorsDict.items():
            errorBlock = f'{"="*30}\n{traceback_format}{"*"*30}'
            copiedText = f'{copiedText}{errorBlock}'
        cb.setText(copiedText, mode=cb.Clipboard)
        print('Error message copied.')
        self.copyButton.setIcon(QIcon(':okButton.svg'))
        self.copyButton.setText(' Copied to clipboard!')
        QTimer.singleShot(2000, self.restoreCopyButton)
    
    def restoreCopyButton(self):
        self.copyButton.setText(self.copyButton.text)
        self.copyButton.setIcon(self.copyButton.icon)
    
    def showEvent(self, a0) -> None:
        self.copyButton.setFixedWidth(self.copyButton.width())
        return super().showEvent(a0)
