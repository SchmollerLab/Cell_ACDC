"""The "Columns to plot" list of the measurements viewer
(ViewerColumnsGuiElements): filtering, colors, y axes and conditional
formatting. Defines the default plotted columns. Spec:
measurements_viewer/__init__.py.
"""
import copy
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon, QPixmap, QPainter
from qtpy.QtWidgets import QAction, QMenu, QColorDialog
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc.apps_measurements_viewer import (
    ConditionalFormattingDialog, _FormattingConditionDialog
)
from cellacdc.measurements_viewer_utils import signal_color
from .. import viewer_utils


DEFAULT_MEASUREMENTS_COLUMNS = ('cell_area_pxl', 'cell_vol_vox', 'cell_area_um2')


class ViewerColumnsGuiElements:
    """The "Columns to plot" list: filling and filtering it, checked
    columns, colors and icons, the column context menu (y axes, color,
    conditional formatting) and renamed columns.
    """

    def updateColumnsList(self, renamed=None):
        """Repopulate the columns list keeping the checked columns.
        `renamed` ({old: new}) keeps renamed columns checked."""
        renamed = renamed or {}
        checkedColumns = {
            renamed.get(column, column) for column in self.checkedColumns()
        }
        isFirstInit = self.columnsModel.rowCount() == 0
        
        # Do not block the model signals: the search proxy and the view must 
        # be notified of the reset (otherwise they keep dangling indexes)
        self._isUpdatingColumnsList = True
        try:
            self._updateColumnsList(checkedColumns, isFirstInit)
        finally:
            self._isUpdatingColumnsList = False

    def _updateColumnsList(self, checkedColumns, isFirstInit):
        self.columnsModel.clear()
        if self.acdc_df is None:
            self.updateSignalAxes([])
            return
        
        numericColumns = [
            col for col in self.acdc_df.select_dtypes(
                include=['number', 'bool']
            ).columns
            if col not in viewer_utils.INDEX_COLUMNS
        ]
        if isFirstInit and self.measurementsModeCombobox.currentIndex() == 0:
            for col in DEFAULT_MEASUREMENTS_COLUMNS:
                if col in numericColumns:
                    checkedColumns = {col}
                    break
            else:
                checkedColumns = set(numericColumns[:1])
        
        channels = getattr(self.posData(), 'chNames', ()) if self.data else ()
        self.columnsModel.populate(
            self.acdc_df[numericColumns], channels, checkable=True
        )
        for item in self.columnsModel.measurementItems():
            col = item.text()
            if col in checkedColumns:
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)
        self.columnsListWidget.expandAll()
        self.updateSignalAxes(numericColumns)
        self.filterColumns(self.columnsSearchLineEdit.text())

    def updateSignalAxes(self, columns):
        for axis, col in self.measurementsAxes.items():
            if col not in columns:
                self.measurementsAxes[axis] = ''
        self._timeCourseColumns.intersection_update(columns)
        self.updateColumnsIcons()

    def checkedColumns(self):
        columns = []
        for item in self.columnsModel.measurementItems():
            if item.checkState() == Qt.Checked:
                columns.append(item.text())
        return columns

    def filterColumns(self, text):
        self.columnsSearchProxy.setFilterText(text)
        self.columnsHighlightDelegate.setFilterText(text)
        if text.strip():
            self.columnsListWidget.expandAll()
        self.columnsListWidget.viewport().update()

    def renameMeasurementColumnReferences(self, oldColumn, newColumn):
        """Move the plot selection, axes (comparison and time course y
        axis), color and conditional formatting of the renamed table column
        `oldColumn` to `newColumn` and update the columns list."""
        if oldColumn == newColumn:
            return
        if oldColumn in self._timeCourseColumns:
            self._timeCourseColumns.discard(oldColumn)
            self._timeCourseColumns.add(newColumn)
        for axis, column in self.measurementsAxes.items():
            if column == oldColumn:
                self.measurementsAxes[axis] = newColumn
        for mapping in (
                self.measurementsColors, self.conditionalFormatting,
                self._defaultSignalColorIndices, self.timeCourseYAxes):
            if oldColumn in mapping:
                mapping[newColumn] = mapping.pop(oldColumn)
        self.updateColumnsList(renamed={oldColumn: newColumn})

    def getSignalColor(self, col):
        """Get the plotting color of column `col`: the custom color if set
        by the user, otherwise its stable default color
        (`defaultSignalColorIndex`)."""
        color = self.measurementsColors.get(col)
        if color is not None:
            return color
        return signal_color(
            self.defaultSignalColorIndex(col),
            light_background=self.invertBwAction.isChecked()
        )

    def defaultSignalColorIndex(self, col):
        """Index of the default color of column `col` (see `signal_color`).

        A column keeps its color while it is plotted, and gets it back when
        plotted again unless another plotted column took it meanwhile. A
        newly plotted column takes the first color that no other plotted
        column uses. In comparison mode the columns plotted in the time
        course count as plotted, so that the comparison (Y column color)
        does not reassign their colors.
        """
        indices = self._defaultSignalColorIndices
        if self._measurementsMode == 0:
            plotted = set(self.checkedColumns())
        else:
            plotted = self._timeCourseColumns
        used = {
            index for column, index in indices.items()
            if column != col and column in plotted
        }
        index = indices.get(col)
        if index is None or index in used:
            index = next(i for i in range(len(used)+1) if i not in used)
            indices[col] = index
        return index

    def updateColumnsIcons(self):
        """Show color swatches or comparison-axis markers beside columns."""
        columns = self.checkedColumns()
        isComparison = self.measurementsModeCombobox.currentIndex() == 1
        self.columnsModel.blockSignals(True)
        for item in self.columnsModel.measurementItems():
            col = item.text()
            item.setCheckable(True)
            if isComparison:
                axes = '/'.join(
                    axis for axis, measurement in self.measurementsAxes.items()
                    if measurement == col
                )
                item.setData(axes, Qt.AccessibleDescriptionRole)
                if not axes:
                    item.setIcon(QIcon())
                    continue
                pixmap = QPixmap(24, 16)
                pixmap.fill(Qt.transparent)
                painter = QPainter(pixmap)
                painter.setPen(self.columnsListWidget.palette().text().color())
                font = self.columnsListWidget.font()
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(pixmap.rect(), Qt.AlignCenter, axes)
                painter.end()
                item.setIcon(QIcon(pixmap))
                continue
            item.setData('', Qt.AccessibleDescriptionRole)
            if col not in columns:
                item.setIcon(QIcon())
                continue
            pixmap = QPixmap(12, 12)
            pixmap.fill(self.getSignalColor(col))
            item.setIcon(QIcon(pixmap))
        self.columnsModel.blockSignals(False)
        # Signals were blocked to avoid itemChanged, so repaint explicitly
        self.columnsListWidget.viewport().update()

    def showColumnsContextMenu(self, pos):
        proxy_index = self.columnsListWidget.indexAt(pos)
        if not proxy_index.isValid():
            return
        source_index = self.columnsSearchProxy.mapToSource(proxy_index)
        item = self.columnsModel.itemFromIndex(source_index)
        if not item.isCheckable():
            return
        
        col = item.text()
        menu = QMenu(self.columnsListWidget)

        plotAction = QAction('Plot', menu)
        plotAction.setCheckable(True)
        plotAction.setChecked(item.checkState() == Qt.Checked)
        plotAction.toggled.connect(
            lambda checked, item=item: item.setCheckState(
                Qt.Checked if checked else Qt.Unchecked
            )
        )
        menu.addAction(plotAction)
        if self.measurementsModeCombobox.currentIndex() == 1:
            menu.popup(self.columnsListWidget.viewport().mapToGlobal(pos))
            return
        menu.addSeparator()

        self.addYAxisActions(menu, col, item.checkState() == Qt.Checked)
        menu.addSeparator()

        changeColorAction = QAction('Change color...', menu)
        changeColorAction.triggered.connect(
            lambda checked=False, col=col: self.changeSignalColor(col)
        )
        menu.addAction(changeColorAction)
        
        resetColorAction = QAction('Reset to default color', menu)
        resetColorAction.setEnabled(col in self.measurementsColors)
        resetColorAction.triggered.connect(
            lambda checked=False, col=col: self.resetSignalColor(col)
        )
        menu.addAction(resetColorAction)
        menu.addSeparator()

        formattingAction = QAction('Conditional formatting...', menu)
        formattingAction.triggered.connect(
            lambda checked=False, col=col: self.editConditionalFormatting(col)
        )
        menu.addAction(formattingAction)

        clearFormattingAction = QAction('Clear conditional formatting', menu)
        clearFormattingAction.setEnabled(col in self.conditionalFormatting)
        clearFormattingAction.triggered.connect(
            lambda checked=False, col=col: self.setConditionalFormatting(
                col, []
            )
        )
        menu.addAction(clearFormattingAction)

        menu.popup(self.columnsListWidget.viewport().mapToGlobal(pos))

    def addYAxisActions(self, menu, col, isPlotted):
        """Add the time course y axis actions of column `col` to its context
        `menu`: "Create new y axis for this" and the "Add to y-axis of..."
        submenu, which lists the other displayed y axes by their plotted
        columns. The column's own axis (remembered if it is not plotted) is
        not listed, so the submenu is empty (disabled) until a column is
        put on another axis."""
        yAxes = self.plottedYAxes()
        number = self.yAxisNumber(col)
        newAxisAction = QAction('Create new y axis for this', menu)
        # A plotted column alone on its axis already has its own axis
        newAxisAction.setEnabled(
            not isPlotted or len(yAxes.get(number, [])) > 1
        )
        newAxisAction.triggered.connect(
            lambda checked=False, col=col: self.plotOnNewYAxis(col)
        )
        menu.addAction(newAxisAction)

        addToAxisMenu = menu.addMenu('Add to y-axis of...')
        for otherNumber, columns in yAxes.items():
            if otherNumber == number:
                continue
            # '&' would be a mnemonic
            text = ', '.join(columns).replace('&', '&&')
            action = QAction(text, addToAxisMenu)
            action.triggered.connect(
                lambda checked=False, col=col, number=otherNumber:
                self.addToYAxisOf(col, number)
            )
            addToAxisMenu.addAction(action)
        addToAxisMenu.setEnabled(not addToAxisMenu.isEmpty())

    def changeSignalColor(self, col):
        initialColor = pg.mkColor(self.getSignalColor(col))
        color = QColorDialog.getColor(
            initialColor, self, f'Select color of "{col}"'
        )
        if not color.isValid():
            return
        
        self.logger.info(f'Changing color of "{col}" to {color.name()}')
        self.measurementsColors[col] = color
        self.plotMeasurements()

    def resetSignalColor(self, col):
        self.logger.info(f'Resetting color of "{col}" to the default')
        self.measurementsColors.pop(col, None)
        self.plotMeasurements()

    def cellConditionalFormatting(self, df_ID, col, rules):
        """Rule index per row of the selected cell for column `col` (-1: no
        rule). Errors are reported in the status bar, and the column is then
        plotted without formatting."""
        if not rules:
            return np.full(len(df_ID), -1, dtype=int)
        try:
            return self.evaluateConditionalFormatting(
                df_ID, rules, ID=self.selectedID,
                SizeT=getattr(self.posData(), 'SizeT', None),
                last_frame=self.analysisLastFrame()
            )
        except Exception as err:
            message = f'Conditional formatting of "{col}" failed: {err}'
            if message != getattr(self, '_lastFormattingError', None):
                self.logger.warning(message)
            self._lastFormattingError = message
            self.statusBar().showMessage(message, 10000)
            return np.full(len(df_ID), -1, dtype=int)

    def editFormattingCondition(self, condition):
        posData = self.posData() if self.data else None
        dialog = _FormattingConditionDialog(
            getattr(posData, 'chNames', []),
            getattr(posData, 'SizeZ', 1) > 1, bool(self.isSegm3D),
            acdc_df=self.dialogAnalysisDf(self.acdc_df, posData),
            SizeT=self.dialogAnalysisSizeT(posData),
            condition=condition, parent=self
        )
        dialog.exec_()
        dialog.deleteLater()
        if dialog.cancel:
            return None
        return dialog.equation

    def editConditionalFormatting(self, col):
        dialog = ConditionalFormattingDialog(
            col, self.conditionalFormatting.get(col, []),
            validateCondition=self.validateFormattingCondition,
            editCondition=self.editFormattingCondition, parent=self
        )
        dialog.exec_()
        if dialog.cancel:
            return
        self.setConditionalFormatting(col, dialog.rules)

    def setConditionalFormatting(self, col, rules):
        if rules:
            self.logger.info(
                f'Conditional formatting of "{col}" set: {len(rules)} rule(s)'
            )
            self.conditionalFormatting[col] = copy.deepcopy(rules)
        else:
            self.logger.info(f'Conditional formatting of "{col}" cleared')
            self.conditionalFormatting.pop(col, None)
        self._lastFormattingError = None
        self.plotMeasurements()
