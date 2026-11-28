"""The measurements plot of the viewer (ViewerPlottingGuiElements): time
courses, measurement vs measurement, extra y axes, autoscale, phase band
and point hovering/clicking. Defines the tool tip of the columns list in
Time course mode. Spec: measurements_viewer/__init__.py.
"""
import html
from functools import partial
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt, QTimer
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import qutils
from cellacdc import timepoint_predictors
from ..dialogs.conditional_formatting import formatting_color


# Tool tip of the columns list in Time course mode
TIME_COURSE_COLUMNS_TOOLTIP = (
    'Click a column to plot/unplot it. Right-click a column\n'
    'to change its color, y axis or conditional formatting.'
)


class ViewerPlottingGuiElements:
    """The measurements plot: time courses and measurement vs measurement
    plots, extra y axes, autoscale, unsupported plot options, the phase
    band, unanalyzed-frame shading, the current frame line and
    hovering/clicking points.
    """

    def rescalePlotToggled(self, checked):
        self.logger.info(f'Rescale plot {"on" if checked else "off"}')
        if checked:
            self.autoscaleMeasurementsPlot()
        self._set_setting_value('rescalePlot', checked)

    def comparisonPlotData(self):
        """X and Y columns of the plotted comparison scatter (copy), or None
        if not in comparison mode or nothing is plotted."""
        if self.measurementsModeCombobox.currentIndex() != 1:
            return
        if self.acdc_df is None or not self.measurementsPlot.listDataItems():
            return
        columns = [self.measurementsAxes['X'], self.measurementsAxes['Y']]
        if not all(col in self.acdc_df.columns for col in columns):
            return
        return self.acdc_df[columns].copy()

    def updateCurrentFrameLine(self):
        self.currentFrameLine.blockSignals(True)
        self.currentFrameLine.setValue(self.frame_i+1)
        self.currentFrameLine.blockSignals(False)
        self.phaseBandFrameLine.setValue(self.frame_i+1)

    def currentFrameLineMoved(self, line):
        frame_n = int(round(line.value()))
        self.setFrame(frame_n-1)
        self.updateCurrentFrameLine()
    
            # Per-cell constants values {name: {Cell_ID: value}}

    def plotReadsColumns(self, columns):
        """True if one of the table `columns` is plotted or is read by a
        conditional formatting condition of a plotted column."""
        columns = set(columns)
        if not columns:
            return False
        plotted = set(self.checkedColumns()) | set(self.measurementsAxes.values())
        if plotted & columns:
            return True
        Dependencies = timepoint_predictors.PredictorDependencies
        for formatted in plotted:
            for rule in self.conditionalFormatting.get(formatted, ()):
                inputs = Dependencies.inputs({
                    'type': timepoint_predictors.PREDICTOR_EQUATION,
                    'equation': rule.get('condition', '')
                })
                if columns.intersection(inputs):
                    return True
        return False

    def refreshPlotForColumns(self, columns):
        """After editing the table `columns` (time points or annotations),
        replot if the plot or its conditional formatting reads one of them,
        otherwise only redraw the time point lines."""
        if self.data and self.plotReadsColumns(columns):
            self.plotMeasurements()
        else:
            self.updateTimepointLines()

    def plotMeasurements(self, *args):
        plotItem = self.measurementsPlot.getPlotItem()
        for item in self.measurementsPlot.listDataItems():
            self.measurementsPlot.removeItem(item)
        plotItem.clearExtraYAxesItems()
        # Removed scatters get no hover exit event: drop their tip (it would
        # hide the tip of the points hovered next)
        self.clearMeasurementPointsTip()
        self.measurementsLegend.clear()
        self.updateCurrentFrameLine()
        self.updateColumnsIcons()

        isComparison = self.measurementsModeCombobox.currentIndex() == 1
        self.currentFrameLine.setVisible(not isComparison)
        self.updateTimepointLines()
        if not self.data or isComparison:
            plotItem.setExtraYAxesCount(0)
        self.updatePhaseBand()
        self.updateUnanalyzedRegions()
        if not self.data:
            self.measurementsPlot.setTitle('No data loaded')
            return
        if isComparison:
            self.plotMeasurementsComparison()
            return
        # Switching back to Measurement vs Measurement autoscales it
        self._comparisonRangeKey = None

        self.measurementsPlot.setLabel('bottom', 'Frame n.')
        isNormalized = self.normalizeCheckbox.isChecked()
        # Displayed y axes: the first one on the left, the others on the
        # right (`timeCourseYAxes`)
        yAxes = self.plottedYAxes()
        axisIndices = {number: i for i, number in enumerate(yAxes)}
        plotItem.setExtraYAxesCount(max(len(yAxes) - 1, 0))
        self.setTimeCourseYAxesLabels(yAxes, isNormalized)

        posData = self.posData()
        # Frame range 1..SizeT on the first time course plot of a position
        # (or after comparison mode), kept while nothing can be autoscaled
        # (no column plotted or no values for the selected cell)
        xRangeKey = (self.pos_i, posData.SizeT)
        isNewXRange = getattr(self, '_timecourseXRangeKey', None) != xRangeKey
        if isNewXRange:
            self.measurementsPlot.setXRange(
                1, max(posData.SizeT, 2), padding=0.02
            )
            self._timecourseXRangeKey = xRangeKey

        df_ID = self.getMeasurementsDf()
        columns = self.checkedColumns()
        if self.selectedID is None:
            self.measurementsPlot.setTitle('')
        else:
            self.measurementsPlot.setTitle(f'Cell ID {self.selectedID}')

        # Autoscale x and y (`autoscaleMeasurementsPlot`, as the plot's "A"
        # button) when the plotted curves change (column plotted or
        # removed, moved to another y axis, normalization toggled) or the
        # frame range was reset; the plot then follows the data until the
        # user zooms or pans by hand. The key is stored only once there was
        # data to fit (cell with data).
        rangeKey = (
            tuple(frozenset(axisColumns) for axisColumns in yAxes.values()),
            isNormalized
        )
        if isNewXRange or not columns:
            self._timecourseRangeKey = None
        autoscale = getattr(self, '_timecourseRangeKey', None) != rangeKey

        if df_ID is None or not columns:
            return

        plottedValues = []

        frames_n = df_ID.index.get_level_values(0).to_numpy() + 1
        phases = self.phaseValues(df_ID)
        for c, col in enumerate(columns):
            if col not in df_ID.columns:
                continue
            values = pd.to_numeric(df_ID[col], errors='coerce').to_numpy()
            rawValues = values.astype(float)
            values = rawValues
            if isNormalized:
                values = self._normalize(values)
            axisIndex = axisIndices[self.yAxisNumber(col)]
            color = self.getSignalColor(col)
            rules = self.conditionalFormatting.get(col, [])
            matched = self.cellConditionalFormatting(df_ID, col, rules)
            pointData = [
                {
                    'frame_i': int(frame_n-1),
                    'Cell_ID': self.selectedID,
                    'values': {col: (raw, value)},
                    'normalized': isNormalized,
                    'formatting': (
                        rules[m]['condition'] if m >= 0 else None
                    ),
                    'phase': phase,
                }
                for frame_n, raw, value, m, phase
                in zip(frames_n, rawValues, values, matched, phases)
            ]
            brushes = [
                pg.mkBrush(
                    formatting_color(rules[m]['color'], color) if m >= 0
                    else color
                )
                for m in matched
            ]
            # Segment i (point i to i+1) takes the format of point i. The
            # main curve draws only unformatted segments; each rule draws
            # its segments as separate pairs below the main curve's points.
            finite = np.isfinite(values)
            segments = finite[:-1] & finite[1:]
            connect = np.append(segments & (matched[:-1] < 0), False)
            for r, rule in enumerate(rules):
                idx = np.flatnonzero(segments & (matched[:-1] == r))
                if not len(idx):
                    continue
                pairs = np.column_stack((idx, idx + 1)).ravel()
                segmentsItem = self.plotOnTimeCourseAxis(
                    axisIndex, frames_n[pairs], values[pairs],
                    connect='pairs', pen=pg.mkPen(
                        color=formatting_color(rule['color'], color), width=2
                    )
                )
                segmentsItem.setZValue(-1)
            item = self.plotOnTimeCourseAxis(
                axisIndex, frames_n, values, name=col,
                pen=pg.mkPen(color=color, width=2), connect=connect,
                symbol='o', symbolSize=5, symbolBrush=brushes, symbolPen=None,
                data=pointData
            )
            self.connectSignalPoints(item)
            plottedValues.append(values)

        hasValues = any(np.isfinite(vals).any() for vals in plottedValues)
        if autoscale and hasValues:
            self.autoscaleMeasurementsPlot()
            self._timecourseRangeKey = rangeKey

    def autoscaleMeasurementsPlot(self):
        """Autoscale the measurements plot exactly like its "A" button
        (`PlotItem.autoBtnClicked`): enable pyqtgraph's auto range on both
        axes, so x and y fit the plotted data (time course or Measurement
        vs Measurement) and keep following it (e.g. another cell or a table
        edit) until the user zooms or pans by hand.

        The y auto range uses only the data in the visible x range
        (`ViewBox.setAutoVisible`), so it still follows the data when the
        user pans or zooms only the x axis. The extra y axes of the time
        course (`_TimeCoursePlotItem`) are autoscaled in the same way, as
        the "A" button does. pyqtgraph applies the range on the next paint
        (tests call ``viewBox.updateAutoRange()``).

        Every automatic rescaling of this plot must go through this method
        (column plotted, removed or moved to another y axis, normalization
        toggled, ``Rescale plot``, Measurement vs Measurement axes changed).
        """
        viewBox = self.measurementsPlot.getViewBox()
        viewBox.setAutoVisible(y=True)
        self.measurementsPlot.enableAutoRange()
        self.measurementsPlot.getPlotItem().enableExtraYAxesAutoRange()

    def disableUnsupportedPlotTransforms(self):
        """Disable the "Power Spectrum (FFT)", "dy/dx" and "Y vs. Y'"
        transforms of the measurements plot's options (context menu "Plot
        Options" > "Transforms"; log and "Subtract Mean" stay available).

        Their curves have fewer points than the data, but the time course
        curves and the comparison scatter pass one symbol brush and one
        hover data entry per point (the time course also one ``connect``
        flag), which pyqtgraph applies to the transformed points too: it
        raised "Number of brushes does not match number of points" (with
        FFT on, on every replot), on every y axis. Their x axis (frequency,
        value) would also not match the frame overlays (current frame line,
        time point lines, phase band, shaded unanalyzed frames) nor the
        clicked point's frame.
        """
        ctrl = self.measurementsPlot.getPlotItem().ctrl
        for name in self.UNSUPPORTED_PLOT_TRANSFORMS:
            check = getattr(ctrl, name, None)
            if check is None:
                continue
            check.setChecked(False)
            check.setEnabled(False)
            check.setToolTip(
                'Not available in the measurements viewer: each plotted '
                'point is a cell at\n'
                'one frame, and this transform changes the number of '
                'points or the x axis.'
            )

    def disableUnsupportedPlotDownsampling(self):
        """Uncheck and disable "Downsample" and "Clip to View" (with their
        factor, "Auto" and method options) in the "Downsample" menu of the
        measurements plot's options ("Max Traces" stays available).

        For the same reason as `disableUnsupportedPlotTransforms`: they
        plot fewer points than the data (one point every N frames, the mean
        or min/max of N points, only the points in the visible x range), and
        pyqtgraph applied the per-point symbol brushes and hover data of the
        time course and the comparison scatter to them: it raised "Number of
        brushes (or Length of meta data) does not match number of points" on
        every replot, and with "Clip to View" on every pan or zoom.
        """
        ctrl = self.measurementsPlot.getPlotItem().ctrl
        for name in self.UNSUPPORTED_PLOT_DOWNSAMPLING:
            widget = getattr(ctrl, name, None)
            if widget is None:
                continue
            if name.endswith('Check'):
                widget.setChecked(False)
            widget.setEnabled(False)
            widget.setToolTip(
                'Not available in the measurements viewer: each plotted '
                'point is a\n'
                'cell at one frame, and this option plots fewer points '
                'than the data.'
            )

    def plottedYAxes(self):
        """Plotted time course columns of each displayed y axis,
        ``{axis number: [columns]}`` sorted by axis number (the first axis
        is the left one, the others are on the right; see
        `timeCourseYAxes`)."""
        axes = {}
        for col in self.checkedColumns():
            axes.setdefault(self.yAxisNumber(col), []).append(col)
        return dict(sorted(axes.items()))

    def setTimeCourseYAxesLabels(self, yAxes, isNormalized):
        """Label the left axis 'Value' with a single y axis, otherwise every
        displayed axis with its plotted columns (`plottedYAxes`), escaped
        because pyqtgraph renders axis labels as HTML (a CSV column such as
        `size<5um` would be cut at the `<`)."""
        leftAxis = self.measurementsPlot.getAxis('left')
        if len(yAxes) <= 1:
            leftAxis.setLabel('Normalized value' if isNormalized else 'Value')
            return
        suffix = ' (normalized)' if isNormalized else ''
        plotItem = self.measurementsPlot.getPlotItem()
        axisItems = [leftAxis, *plotItem.shownExtraAxes()]
        for axisItem, columns in zip(axisItems, yAxes.values()):
            label = html.escape(', '.join(columns), quote=False)
            axisItem.setLabel(label + suffix)

    def plotOnTimeCourseAxis(self, index, *args, name=None, **kwargs):
        """Plot a `pg.PlotDataItem(*args, **kwargs)` on the displayed y axis
        `index` of the time course (0: left axis, else the extra axis
        `index - 1` on the right), with a legend entry if `name` is given.

        The item keeps the raw `name` (`PlotDataItem.name()`, e.g., the
        column headers of the CSV export), while the legend shows it
        escaped: pyqtgraph renders legend labels as HTML (a CSV column such
        as `size<5um` would be cut at the `<`).
        """
        item = pg.PlotDataItem(*args, **kwargs)
        if index == 0:
            self.measurementsPlot.addItem(item)
        else:
            plotItem = self.measurementsPlot.getPlotItem()
            plotItem.addExtraYAxisItem(index - 1, item)
        if name is not None:
            # Named once added: `PlotItem.addItem` would add a legend entry
            # with the unescaped name
            item.opts['name'] = name
            self.measurementsLegend.addItem(
                item, html.escape(name, quote=False)
            )
        return item

    def timeCourseDataItems(self):
        """Data items of the time course on every displayed y axis (left
        axis first)."""
        plotItem = self.measurementsPlot.getPlotItem()
        items = list(plotItem.listDataItems())
        for axisItems in plotItem.extraYAxesDataItems():
            items.extend(axisItems)
        return items

    def columnItem(self, col):
        """Item of column `col` in the columns list, or None."""
        for item in self.columnsModel.measurementItems():
            if item.text() == col:
                return item
        return None

    def plotOnNewYAxis(self, col):
        """Plot column `col` on a new y axis of the time course (context
        menu "Create new y axis for this")."""
        self.logger.info(f'Plotting "{col}" on a new y axis')
        self.plotOnYAxis(col, self.newYAxisNumber())

    def addToYAxisOf(self, col, number):
        """Plot column `col` on the y axis `number` (context menu "Add to
        y-axis of...", listing the axes by their plotted columns).

        Moving the last plotted column of the default axis 1 merges the
        axis `number` into axis 1 instead: columns plotted later (on axis 1
        by default) then join the merged axis rather than getting a new
        left axis of their own. Nothing merges when no column of axis 1 is
        plotted (the left axis is then another one)."""
        yAxes = self.plottedYAxes()
        self.logger.info(
            f'Plotting "{col}" on the y axis of '
            + ', '.join(f'"{column}"' for column in yAxes.get(number, []))
        )
        if number != 1 and yAxes.get(1) == [col]:
            self.mergeYAxisIntoDefault(number)
            number = 1
        self.plotOnYAxis(col, number)

    def plotOnYAxis(self, col, number):
        item = self.columnItem(col)
        if item is None:
            return
        self.setYAxisNumber(col, number)
        if item.checkState() == Qt.Checked:
            self.plotMeasurements()
        else:
            # Plots it (`measurementColumnChanged`)
            item.setCheckState(Qt.Checked)

    def measurementsModeChanged(self, *args):
        isComparison = self.measurementsModeCombobox.currentIndex() == 1
        self.logger.info(
            f'Plot mode: "{self.measurementsModeCombobox.currentText()}"'
        )
        if self._measurementsMode == 0:
            self._timeCourseColumns = set(self.checkedColumns())
        self._measurementsMode = self.measurementsModeCombobox.currentIndex()
        columns = (
            set(self.measurementsAxes.values()) if isComparison
            else self._timeCourseColumns
        )
        self.columnsModel.blockSignals(True)
        for item in self.columnsModel.measurementItems():
            item.setCheckState(
                Qt.Checked if item.text() in columns else Qt.Unchecked
            )
        self.columnsModel.blockSignals(False)
        self.columnsListWidget.setToolTip(
            'Click two columns to plot: first X, then Y.\n'
            'Click a selected column again to free its axis.'
            if isComparison else TIME_COURSE_COLUMNS_TOOLTIP
        )
        self.plotMeasurements()

    def measurementColumnChanged(self, item):
        if getattr(self, '_isUpdatingColumnsList', False):
            return
        if not item.isCheckable():
            return
        if self.measurementsModeCombobox.currentIndex() == 1:
            col = item.text()
            if item.checkState() != Qt.Checked:
                for axis, measurement in self.measurementsAxes.items():
                    if measurement == col:
                        self.measurementsAxes[axis] = ''
                        self.logger.info(f'Removing "{col}" from the {axis} axis')
            elif col not in self.measurementsAxes.values():
                axis = next(
                    (axis for axis, measurement in self.measurementsAxes.items()
                     if not measurement), None
                )
                if axis is None:
                    self.columnsModel.blockSignals(True)
                    item.setCheckState(Qt.Unchecked)
                    self.columnsModel.blockSignals(False)
                    self.statusBar().showMessage(
                        'Two measurements are already selected. Uncheck X or Y '
                        'before selecting another measurement.', 5000
                    )
                else:
                    self.measurementsAxes[axis] = col
                    self.logger.info(f'Plotting "{col}" on the {axis} axis')
        elif item.checkState() == Qt.Checked:
            self.logger.info(f'Plotting "{item.text()}"')
        else:
            self.logger.info(f'Removing "{item.text()}" from the plot')
        self.plotMeasurements()

    def plotMeasurementsComparison(self):
        xCol = self.measurementsAxes['X']
        yCol = self.measurementsAxes['Y']
        isNormalized = self.normalizeCheckbox.isChecked()
        suffix = ' (normalized)' if isNormalized else ''
        # Axis labels are HTML (`setTimeCourseYAxesLabels`)
        self.measurementsPlot.setLabel(
            'bottom', html.escape(xCol, quote=False) + suffix
        )
        self.measurementsPlot.setLabel(
            'left', html.escape(yCol, quote=False) + suffix
        )
        self.measurementsPlot.setTitle('All cells, all frames (current position)')
        # Autoscale only when the scatter's axes change (or a position is
        # loaded, or after the time course): redraws for other reasons (e.g.
        # phases evaluated when selecting a cell) keep the user's zoom
        rangeKey = (self.pos_i, xCol, yCol, isNormalized)
        if getattr(self, '_comparisonRangeKey', None) != rangeKey:
            self.autoscaleMeasurementsPlot()
            self._comparisonRangeKey = rangeKey
        self._timecourseXRangeKey = None
        if self.acdc_df is None:
            self.measurementsPlot.setTitle('No measurements available')
            return
        # A column removed from the table (e.g., time point identifier) is
        # an unset axis until the columns list is updated
        if (not xCol or not yCol or xCol not in self.acdc_df.columns
                or yCol not in self.acdc_df.columns):
            self.measurementsPlot.setTitle('Select X and Y measurements from the column list')
            return

        df = self.acdc_df
        frames = df.index.get_level_values('frame_i').to_numpy()
        IDs = df.index.get_level_values('Cell_ID').to_numpy()
        inRange = (frames >= 0) & (frames < self.posData().SizeT) & (IDs > 0)
        df = df.loc[inRange]
        frames, IDs = frames[inRange], IDs[inRange]
        rawX = pd.to_numeric(df[xCol], errors='coerce').to_numpy(dtype=float)
        rawY = pd.to_numeric(df[yCol], errors='coerce').to_numpy(dtype=float)
        phases = self.phaseValues(df)
        finite = np.isfinite(rawX) & np.isfinite(rawY)
        frames, IDs = frames[finite], IDs[finite]
        rawX, rawY = rawX[finite], rawY[finite]
        phases = phases[finite]
        if not len(rawX):
            self.measurementsPlot.setTitle('No finite paired measurements')
            return
        x = self._normalize(rawX) if isNormalized else rawX
        y = self._normalize(rawY) if isNormalized else rawY
        pointData = [
            {
                'frame_i': int(frame_i),
                'Cell_ID': int(ID),
                'values': {xCol: (rx, vx), yCol: (ry, vy)},
                'normalized': isNormalized,
                'phase': phase,
            }
            for frame_i, ID, rx, ry, vx, vy, phase
            in zip(frames, IDs, rawX, rawY, x, y, phases)
        ]
        brush = self.getSignalColor(yCol)
        if self.isShowingPhases():
            brush = self.comparisonPhaseBrushes(phases)
        item = self.measurementsPlot.plot(
            x, y, pen=None, symbol='o', symbolSize=7,
            symbolBrush=brush, symbolPen=None, data=pointData
        )
        self.connectSignalPoints(item)

    def isShowingPhases(self):
        return self.showPhasesCheckbox.isChecked() and self.hasPhaseColumn()

    def normalizeToggled(self, checked):
        self.logger.info(f'Normalize measurements {"on" if checked else "off"}')

    def showPhasesToggled(self, checked):
        self.logger.info(f'Show phases {"on" if checked else "off"}')
        self._set_setting_value(
            'measurements_viewer_show_phases', 'Yes' if checked else 'No'
        )
        self.plotMeasurements()

    def comparisonPhaseBrushes(self, phases):
        """Per-point brushes by phase; adds a legend entry per shown phase
        (names escaped: legend labels are HTML)."""
        colors = self.phaseColorMap()
        unassigned = pg.mkBrush(150, 150, 150, 120)
        brushes = {name: pg.mkBrush(color) for name, color in colors.items()}
        present = set(phases)
        for name, brush in brushes.items():
            if name in present:
                self.measurementsLegend.addItem(
                    pg.ScatterPlotItem(symbol='o', brush=brush, pen=None),
                    html.escape(name, quote=False)
                )
        if '' in present:
            self.measurementsLegend.addItem(
                pg.ScatterPlotItem(symbol='o', brush=unassigned, pen=None),
                'No phase'
            )
        return [brushes.get(phase, unassigned) for phase in phases]

    def syncPhaseBandAxisWidth(self):
        """Give the phase band the left axis width of the measurements plot
        and, with extra y axes, an empty right axis as wide as them, so that
        both plots show the frames at the same x positions."""
        width = self.measurementsPlot.getAxis('left').width()
        self.phaseBandPlot.getAxis('left').setWidth(int(width))
        plotItem = self.measurementsPlot.getPlotItem()
        rightWidth = sum(axis.width() for axis in plotItem.shownExtraAxes())
        if not rightWidth:
            self.phaseBandPlot.hideAxis('right')
            return
        rightAxis = self.phaseBandPlot.getAxis('right')
        rightAxis.setStyle(showValues=False)
        rightAxis.setTicks([[]])
        rightAxis.setWidth(int(round(rightWidth)))
        self.phaseBandPlot.showAxis('right')

    def updatePhaseBand(self):
        """Draw the selected cell's phases as colored bars below the time
        course. Hidden in comparison mode, without phases or when disabled."""
        for item in self.phaseBandItems:
            self.phaseBandPlot.removeItem(item)
        self.phaseBandItems = []
        self.phaseBandPlot.setToolTip('')
        self.showPhasesCheckbox.setEnabled(self.hasPhaseColumn())
        isComparison = self.measurementsModeCombobox.currentIndex() == 1
        df_ID = None
        if self.data and not isComparison and self.isShowingPhases():
            df_ID = self.getMeasurementsDf()
        runs = []
        if df_ID is not None:
            frames_n = df_ID.index.get_level_values(0).to_numpy() + 1
            runs = self.phaseRuns(frames_n, self.phaseValues(df_ID))
        self.phaseBandPlot.setVisible(bool(runs))
        if not runs:
            return
        colors = self.phaseColorMap()
        bars = pg.BarGraphItem(
            x0=[start - 0.5 for start, _, _ in runs],
            x1=[end + 0.5 for _, end, _ in runs],
            y0=0, height=1, pen=None,
            brushes=[pg.mkBrush(colors[phase]) for _, _, phase in runs]
        )
        self.phaseBandPlot.addItem(bars)
        self.phaseBandItems.append(bars)
        for start, end, phase in runs:
            color = colors[phase]
            textColor = 'k' if color.lightness() > 140 else 'w'
            text = pg.TextItem(phase, color=textColor, anchor=(0.5, 0.5))
            text.setPos((start + end) / 2, 0.5)
            self.phaseBandPlot.addItem(text, ignoreBounds=True)
            self.phaseBandItems.append(text)
        self.phaseBandPlot.setToolTip(qutils.wrap_tooltip('\n'.join(
            f'{phase}: frames {start}-{end}' for start, end, phase in runs
        )))
        self.phaseBandPlot.setYRange(0, 1, padding=0)
        self.syncPhaseBandAxisWidth()

    @staticmethod
    def createUnanalyzedRegion():
        """Gray, non-movable region over the frames ignored by the
        calculations (red hues are reserved for the frame lines). Below the
        data (of every y axis: `_TimeCoursePlotItem.addBackgroundItem`),
        time point and frame lines; added with ``ignoreBounds=True`` so that
        it never changes the auto range."""
        region = pg.LinearRegionItem(
            values=(0, 1), movable=False,
            brush=pg.mkBrush(128, 128, 128, 60), pen=pg.mkPen(None)
        )
        region.setZValue(-10)
        region.setVisible(False)
        return region

    def updateUnanalyzedRegions(self):
        """Shade the frames after the last analyzed frame of the displayed
        position (`analysisLastFrame`) in the time course and phase band
        (frame numbers are 1-based: frame index ``i`` is drawn at
        ``i + 1``). Hidden in comparison mode and when all frames are used.
        """
        regions = (self.unanalyzedRegion, self.phaseBandUnanalyzedRegion)
        isComparison = self.measurementsModeCombobox.currentIndex() == 1
        last = self.analysisLastFrame() if self.data else None
        if last is None or isComparison:
            for region in regions:
                region.setVisible(False)
            return
        SizeT = self.posData().SizeT
        # No tool tip: it would hide the points' hover info (shown with
        # the ViewBox tool tip) and the phase band tool tip on these frames
        for region in regions:
            region.setRegion((last + 1.5, SizeT + 0.5))
            region.setVisible(True)

    def connectSignalPoints(self, item):
        # PlotDataItem does not forward hover options to its scatter item.
        item.scatter.setData(
            x=item.xData, y=item.yData, data=item.opts['data'],
            brush=item.opts['symbolBrush'],
            hoverable=True, tip=self.measurementPointTip,
            hoverSize=11, hoverBrush=pg.mkBrush(255, 100, 100)
        )
        item.scatter.sigClicked.connect(self.measurementPointClicked)
        item.scatter.sigHovered.connect(self.measurementPointsHovered)

    def measurementPointsHovered(self, scatter, points, event):
        """Show the tip of the hovered points of every y axis on the main
        view box. Each scatter sets the tip of its own view box, and Qt
        shows only the topmost one (the main view box): points of
        different axes at the same pixel (common, as every axis fits its
        own data) would show only the left axis values. Emitted after each
        scatter's hover event (also on exit, with no points), so the last
        call of a mouse move has the points of every scatter."""
        if len(points):
            self._hoveredMeasurementPoints[scatter] = list(points)
        else:
            self._hoveredMeasurementPoints.pop(scatter, None)
        order = {
            item.scatter: i for i, item in enumerate(self.timeCourseDataItems())
        }
        hovered = sorted(
            self._hoveredMeasurementPoints.items(),
            key=lambda entry: order.get(entry[0], len(order))
        )
        points = [point for _, scatterPoints in hovered for point in scatterPoints]
        # As `ScatterPlotItem.hoverEvent`: at most 3 tips
        tips = [
            self.measurementPointTip(
                x=point.pos().x(), y=point.pos().y(), data=point.data()
            )
            for point in points[:3]
        ]
        if len(points) > 3:
            tips.append(f'({len(points) - 3} others...)')
        plotItem = self.measurementsPlot.getPlotItem()
        plotItem.vb.setToolTip('\n\n'.join(tips))
        for vb in plotItem.shownExtraViewBoxes():
            vb.setToolTip('')

    def clearMeasurementPointsTip(self):
        self._hoveredMeasurementPoints = {}
        plotItem = self.measurementsPlot.getPlotItem()
        plotItem.vb.setToolTip('')
        for vb, _ in plotItem.extraYAxes:
            vb.setToolTip('')

    @staticmethod
    def measurementPointTip(x, y, data):
        lines = [
            f'Cell ID: {data["Cell_ID"]}',
            f'Frame: {data["frame_i"]+1}',
        ]
        for col, (raw, value) in data['values'].items():
            lines.append(f'{col}: {raw:.6g}')
            if data['normalized']:
                lines.append(f'{col} (normalized): {value:.6g}')
        if data.get('phase'):
            lines.append(f'Phase: {data["phase"]}')
        if data.get('formatting'):
            lines.append(f'Formatting: {data["formatting"]}')
        # Long column names, phase names or formatting conditions
        return qutils.wrap_tooltip('\n'.join(lines))

    def measurementPointClicked(self, item, points, event):
        if event.button() != Qt.LeftButton or not len(points):
            return
        data = points[0].data()
        ID, frame_i = data['Cell_ID'], data['frame_i']
        if ID not in self.IDs or not 0 <= frame_i < self.posData().SizeT:
            self.statusBar().showMessage(
                f'Cell ID {ID} at frame {frame_i+1} is not available.', 5000
            )
            return
        # Navigating replots the measurements, deleting the clicked scatter item
        # while pyqtgraph is still dispatching the click to it --> defer
        QTimer.singleShot(
            100, partial(self.navigateToSignalPoint, ID, frame_i)
        )

    def navigateToSignalPoint(self, ID, frame_i):
        self.idScrollBar.setValue(self.IDs.index(ID))
        self.setFrame(frame_i)
        self.zoomToSelectedID(goToZslice=True, force=True)

    def recolorSignalsForBackground(self, checked=False):
        """Redraw the columns' default colors, which depend on the plot
        background (`signal_color`), after `Invert black/white` toggled."""
        if not self.isDataLoaded or self._isLoadingData:
            return
        self.updateColumnsIcons()
        self.plotMeasurements()
