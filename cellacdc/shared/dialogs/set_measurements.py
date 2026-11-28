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
