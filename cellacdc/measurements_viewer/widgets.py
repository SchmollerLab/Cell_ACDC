"""Private widgets and pyqtgraph items of the measurements viewer: the time
course plot item and the view boxes of its extra y axes, time point
lines, the columns list delegate, the shortcut-aware action and
auto-shortcut text, stand-ins for main-GUI objects, and the measurements
tree model of the columns panel and the processing recipe dialog.
"""
from qtpy.QtCore import Qt
from qtpy.QtGui import QStandardItemModel, QStandardItem
from cellacdc import measurements
from cellacdc.shared.widgets import search_widgets as widgets_search_widgets
from qtpy.QtCore import Signal, QEvent, QRect
from qtpy.QtGui import QIcon, QKeySequence
from qtpy.QtWidgets import QAction, QStyleOptionViewItem
import pyqtgraph as pg
from cellacdc import qutils


class MeasurementTreeModel(QStandardItemModel):
    """Shared measurement hierarchy with selectable/checkable leaves only."""

    def populate(self, dataframe, channels=(), checkable=False):
        self.clear()
        nodes = {(): self.invisibleRootItem()}
        for path, columns in measurements.measurement_column_groups(dataframe, channels):
            for depth in range(1, len(path)+1):
                prefix = path[:depth]
                if prefix not in nodes:
                    group = QStandardItem(prefix[-1])
                    group.setFlags(Qt.ItemIsEnabled)
                    nodes[prefix[:-1]].appendRow(group)
                    nodes[prefix] = group
            for column in columns:
                item = QStandardItem(column)
                item.setEditable(False)
                item.setCheckable(checkable)
                item.setData(column, widgets_search_widgets.ButtonSearchCompleter.SEARCH_NAME_ROLE)
                nodes[path].appendRow(item)

    def measurementItems(self, parent=None):
        if parent is None:
            parent = self.invisibleRootItem()
        for row in range(parent.rowCount()):
            item = parent.child(row)
            if item.hasChildren():
                yield from self.measurementItems(item)
            else:
                yield item


# Old names of cellacdc.widgets that code reached through this module
# path before it existed (cellacdc/_compat.py).
from cellacdc import _compat as _acdc_compat  # noqa: E402

__getattr__, __dir__ = _acdc_compat.forward_module(
    __name__, 'cellacdc.widgets')


def auto_shortcut_text(shortcut):
    """Key sequence (text) of the ``Shift+<shortcut>`` action that
    re-evaluates the predictor of a time point identifier or custom
    annotation with `shortcut`.

    Empty when `shortcut` is empty or already contains Shift (it would
    collide with the shortcut itself).
    """
    if isinstance(shortcut, QKeySequence):
        shortcut = shortcut.toString()
    text = QKeySequence(shortcut or '').toString()
    if not text or 'Shift' in text:
        return ''
    return QKeySequence(f'Shift+{text}').toString()


class _ShortcutAwareAction(QAction):
    """QAction that tells whether `triggered` is emitted by its shortcut.

    `triggeredByShortcut` is True only while the action handles the
    ``QEvent.Shortcut`` sent by Qt's shortcut map (i.e., during the
    `triggered` slots of a key press), and False for a click on its toolbar
    button (also from the toolbar extension) or `trigger()`. The time point
    actions use it to add a time point at the hovered plot frame (shortcut)
    or at the current frame (click), independently of the cursor position.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.triggeredByShortcut = False

    def event(self, event):
        if event.type() != QEvent.Shortcut:
            return super().event(event)
        self.triggeredByShortcut = True
        try:
            return super().event(event)
        finally:
            self.triggeredByShortcut = False


class _TimepointLine(pg.InfiniteLine):
    """Draggable line of a time point of the selected cell."""
    sigRemoveRequested = Signal(object)

    def __init__(self, identifier, ID, frame_i):
        color = identifier.color
        super().__init__(
            pos=frame_i+1, angle=90, movable=True,
            pen=pg.mkPen(color=color, width=2, style=Qt.DashLine),
            hoverPen=pg.mkPen(color=color, width=4),
            labelOpts={
                'position': 0.95, 'color': color, 'movable': True,
                'fill': (0, 0, 0, 120)
            }
        )
        self.uid = identifier.uid
        self.ID = ID
        self.frame_i = frame_i
        self.setToolTip(qutils.wrap_tooltip(
            f'{identifier.name} (Cell ID {ID}, frame n. {frame_i+1})\n\n'
            'Drag to move. Right-click to remove.'
        ))
        # Above the current frame line (10), else a time point added at the
        # current frame (button click) is hidden under it. The dash gaps keep
        # the red line visible; dragging there moves the time point (user
        # choice over an invisible new time point).
        self.setZValue(11)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton and not self.moving:
            ev.accept()
            self.sigRemoveRequested.emit(self)
            return
        super().mouseClickEvent(ev)


class _TimeCourseViewBox(pg.ViewBox):
    """Main view box of the measurements plot (`_TimeCoursePlotItem`).

    Panning and zooming in the plot area (mouse drag, wheel, rectangle zoom)
    emits the y range before and after the change
    (`sigYRangeChangedInPlot`), so that the extra y axes of the time course
    move in the same way. It is emitted also when the y range did not change
    but its auto range was turned off (e.g. a horizontal pan), so that the
    extra axes stop autoscaling too. Drags and scrolls on the left axis
    (`axis` given by the `AxisItem`) change only this view box.
    """
    sigYRangeChangedInPlot = Signal(object, object)

    def wheelEvent(self, ev, axis=None):
        before, wasAuto = self.viewRange()[1], self.autoRangeEnabled()[1]
        super().wheelEvent(ev, axis=axis)
        self._emitYRangeChangedInPlot(before, wasAuto, axis)

    def mouseDragEvent(self, ev, axis=None):
        before, wasAuto = self.viewRange()[1], self.autoRangeEnabled()[1]
        super().mouseDragEvent(ev, axis=axis)
        self._emitYRangeChangedInPlot(before, wasAuto, axis)

    def _emitYRangeChangedInPlot(self, before, wasAuto, axis):
        if axis is not None:
            return
        after = self.viewRange()[1]
        isAutoTurnedOff = wasAuto and not self.autoRangeEnabled()[1]
        if list(after) != list(before) or isAutoTurnedOff:
            self.sigYRangeChangedInPlot.emit(before, after)


class _ExtraYViewBox(pg.ViewBox):
    """View box of an extra y axis of the time course (`_TimeCoursePlotItem`).
    A right-click on its axis opens the plot menu of the main view box, as
    on the left axis."""

    def __init__(self, mainViewBox):
        super().__init__(enableMenu=False)
        self._mainViewBox = mainViewBox

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            self._mainViewBox.mouseClickEvent(ev)
            return
        super().mouseClickEvent(ev)


class _TimeCoursePlotItem(pg.PlotItem):
    """Plot item of the measurements plot with extra y axes on the right for
    the time course (`setExtraYAxesCount`).

    Each extra axis has its own view box, a child of the plot item with the
    geometry of the main view box (`_TimeCourseViewBox`), x-linked to it and
    stacked under it: the main view box gets the mouse events of the plot
    area (pan, zoom, context menu, current frame and time point lines),
    while hover tips and clicks still reach the points of every axis
    (pyqtgraph sends hover events to every item under the cursor and clicks
    to the topmost item accepting them; a view box does not accept left
    clicks). Panning or zooming in the plot area moves every y axis in the
    same way (`moveExtraYAxesWithMain`); dragging or scrolling on an axis
    changes only that axis (a right-click on it opens the plot menu,
    `_ExtraYViewBox`). The "A" button autoscales every axis and is
    shown when one of them is not autoscaled; "View All" of the plot area
    menu fits every axis once (`fitExtraYAxes`). Curves of the extra axes
    are registered in `items`/`curves` (not `dataItems`, so
    `listDataItems` has only the left axis), so that the plot options
    (log mode, alpha, FFT, downsampling, ...) and the exporters of the
    context menu (CSV, SVG, ...) include them; a changed transform
    autoscales the y range of every axis, as pyqtgraph does for the main
    view box. The viewer disables the options that change the number of
    points (FFT, dy/dx, Y vs Y', downsampling, clip to view:
    `MeasurementsViewerWin.disableUnsupportedPlotTransforms` and
    `disableUnsupportedPlotDownsampling`), which fail on its per-point
    brushes. "Subtract Mean" exists only from pyqtgraph 0.14 (0.13.7 is
    still supported: check the API before using newer options). Background
    items (`addBackgroundItem`, e.g. shaded frames) stay in the bottom view
    box, under the curves of every axis. Extra view boxes are created on demand
    and kept (hidden, axis out of the layout) when not needed.
    """
    sigExtraYAxesChanged = Signal()
    # Read by `updateButtons` also while `PlotItem.__init__` runs
    extraYAxes = ()
    extraYAxesCount = 0

    def __init__(self, **kwargs):
        super().__init__(viewBox=_TimeCourseViewBox(), **kwargs)
        # [(ViewBox, AxisItem)]: the first `extraYAxesCount` are shown
        self.extraYAxes = []
        self.extraYAxesCount = 0
        # [(ViewBox, PlotDataItem)] added by `addExtraYAxisItem`
        self._extraItems = []
        self.backgroundItems = []
        self._backgroundViewBox = self.vb
        # Moves of the main view box (axis widths changed on both sides by
        # the same amount) do not resize it
        self.vb.geometryChanged.connect(self.updateExtraViewsGeometry)
        self.vb.sigYRangeChangedInPlot.connect(self.moveExtraYAxesWithMain)
        self.vb.menu.viewAll.triggered.connect(self.fitExtraYAxes)

    def shownExtraViewBoxes(self):
        return [vb for vb, _ in self.extraYAxes[:self.extraYAxesCount]]

    def shownExtraAxes(self):
        return [axis for _, axis in self.extraYAxes[:self.extraYAxesCount]]

    def _createExtraYAxis(self):
        vb = _ExtraYViewBox(self.vb)
        vb.setMouseEnabled(x=False, y=True)
        # Under the main view box (see class docstring; a ViewBox has z -100
        # and siblings with the same z are stacked in creation order, so
        # the first extra view box is the bottom one, `bottomViewBox`)
        vb.setParentItem(self)
        vb.setZValue(self.vb.zValue() - 1)
        vb.setXLink(self.vb)
        vb.setAutoVisible(y=True)
        vb.sigStateChanged.connect(lambda *args: self.updateButtons())
        axis = pg.AxisItem('right', parent=self)
        axis.setLogMode(*self.logModes())
        axis.linkToView(vb)
        axis.geometryChanged.connect(self.sigExtraYAxesChanged.emit)
        vb.hide()
        axis.hide()
        self.extraYAxes.append((vb, axis))

    def setExtraYAxesCount(self, count):
        """Show `count` extra y axes (right of the plot, in order)."""
        previous = self.extraYAxesCount
        if count == previous:
            return
        while len(self.extraYAxes) < count:
            self._createExtraYAxis()
        for i in range(previous, count):
            vb, axis = self.extraYAxes[i]
            # Right of the (hidden) default right axis in column 2
            self.layout.addItem(axis, 2, 3 + i)
            axis.show()
            vb.show()
            # A new axis fits its data
            vb.enableAutoRange(axis=pg.ViewBox.YAxis)
        self.extraYAxesCount = count
        self.updateExtraViewsGeometry()
        self._placeBackgroundItems()
        hidden = [vb for vb, _ in self.extraYAxes[count:previous]]
        for vb, axis in self.extraYAxes[count:previous]:
            self.layout.removeItem(axis)
            axis.hide()
            vb.hide()
        self._removeExtraItems(hidden)
        self.updateButtons()
        self.sigExtraYAxesChanged.emit()

    def bottomViewBox(self):
        """View box drawn under all the others: the first shown extra view
        box, else the main one."""
        viewBoxes = self.shownExtraViewBoxes()
        return viewBoxes[0] if viewBoxes else self.vb

    def addBackgroundItem(self, item):
        """Add `item` (ignored by the auto range) under the curves of every
        y axis (in `bottomViewBox`, moved when axes are shown or hidden)."""
        self.backgroundItems.append(item)
        self._backgroundViewBox.addItem(item, ignoreBounds=True)
        self._placeBackgroundItems()

    def _placeBackgroundItems(self):
        target = self.bottomViewBox()
        if target is self._backgroundViewBox:
            return
        for item in self.backgroundItems:
            self._backgroundViewBox.removeItem(item)
            target.addItem(item, ignoreBounds=True)
        self._backgroundViewBox = target

    def logModes(self):
        return (
            self.ctrl.logXCheck.isChecked(), self.ctrl.logYCheck.isChecked()
        )

    def addExtraYAxisItem(self, index, item):
        """Add the data `item` to the view box of the shown extra y axis
        `index`, with the plot options that `addItem` applies to the curves
        of the main view box, and register it for the exporters and later
        plot option changes (`items` and `curves`) once it was added."""
        vb = self.shownExtraViewBoxes()[index]
        item.setLogMode(*self.logModes())
        alpha, auto = self.alphaState()
        item.setAlpha(alpha, auto)
        # "Subtract Mean" exists from pyqtgraph 0.14 (0.13.7 is supported)
        subtractMeanCheck = getattr(self.ctrl, 'subtractMeanCheck', None)
        if (subtractMeanCheck is not None
                and hasattr(item, 'setSubtractMeanMode')):
            item.setSubtractMeanMode(subtractMeanCheck.isChecked())
        item.setFftMode(self.ctrl.fftCheck.isChecked())
        item.setDownsampling(*self.downsampleMode())
        item.setClipToView(self.clipToViewMode())
        vb.addItem(item)
        self.items.append(item)
        self.curves.append(item)
        self.itemMeta[item] = {}
        self._extraItems.append((vb, item))
        self.updateDecimation()

    def _removeExtraItems(self, viewBoxes=None):
        """Remove the data items of the extra view boxes (of `viewBoxes`
        only, if given)."""
        kept = []
        for vb, item in self._extraItems:
            if viewBoxes is not None and vb not in viewBoxes:
                kept.append((vb, item))
                continue
            for items in (self.items, self.curves):
                if item in items:
                    items.remove(item)
            self.itemMeta.pop(item, None)
            vb.removeItem(item)
        self._extraItems = kept

    def clearExtraYAxesItems(self):
        self._removeExtraItems()

    def extraYAxesDataItems(self):
        """Data items of each shown extra y axis (list per axis)."""
        return [
            [item for itemVb, item in self._extraItems if itemVb is vb]
            for vb in self.shownExtraViewBoxes()
        ]

    def updateLogMode(self, *args):
        """Apply the log mode of the plot options ("Transforms" of the
        context menu) also to the extra y axes (their curves are in
        `items`)."""
        super().updateLogMode()
        x, y = self.logModes()
        for vb, axis in self.extraYAxes:
            axis.setLogMode(x, y)
        self._autoRangeExtraYAxesAfterTransform()

    # The other transforms of the plot options apply to the curves of the
    # extra y axes (in `items`/`curves`), but pyqtgraph then autoscales only
    # the main view box (`enableAutoRange`): refit the extra y axes too,
    # else they keep the range of the untransformed data after a pan/zoom
    # (the viewer disables FFT, dy/dx and Y vs Y' for its own curves:
    # `MeasurementsViewerWin.disableUnsupportedPlotTransforms`)

    def updateSpectrumMode(self, b=None):
        super().updateSpectrumMode(b)
        self._autoRangeExtraYAxesAfterTransform()

    def updateSubtractMeanMode(self, *args):
        # pyqtgraph >= 0.14 only (connected to `ctrl.subtractMeanCheck`)
        update = getattr(super(), 'updateSubtractMeanMode', None)
        if update is not None:
            update()
        self._autoRangeExtraYAxesAfterTransform()

    def updateDerivativeMode(self, *args):
        super().updateDerivativeMode()
        self._autoRangeExtraYAxesAfterTransform()

    def updatePhasemapMode(self, *args):
        super().updatePhasemapMode()
        self._autoRangeExtraYAxesAfterTransform()

    def _autoRangeExtraYAxesAfterTransform(self):
        for vb, _ in self.extraYAxes:
            vb.enableAutoRange(axis=pg.ViewBox.YAxis)

    def updateExtraViewsGeometry(self, *args):
        # The extra view boxes are children of this item, like the main one
        geometry = self.vb.geometry()
        for vb in self.shownExtraViewBoxes():
            vb.setGeometry(geometry)
            vb.linkedViewChanged(self.vb, vb.XAxis)

    def moveExtraYAxesWithMain(self, before, after):
        """Apply the change of the main y range from `before` to `after`
        (pan/zoom in the plot area) to the extra y axes: the same relative
        shift and scaling, so that every curve moves with the mouse. With
        an unchanged range, the extra axes keep theirs and stop
        autoscaling, like the main y axis."""
        span = before[1] - before[0]
        if not span:
            return
        for vb in self.shownExtraViewBoxes():
            low, high = vb.viewRange()[1]
            scale = (high - low) / span
            vb.setYRange(
                low + (after[0] - before[0]) * scale,
                low + (after[1] - before[0]) * scale,
                padding=0
            )

    def fitExtraYAxes(self, *args):
        """Fit every shown extra y axis to its data in the visible x range
        once ("View All" of the plot area menu, which fits the main view
        box once with `autoRange`)."""
        for vb in self.shownExtraViewBoxes():
            vb.enableAutoRange(axis=pg.ViewBox.YAxis)
            vb.updateAutoRange()
            vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=False)

    def enableExtraYAxesAutoRange(self):
        """Autoscale the shown extra y axes like the main y axis of the "A"
        button: y auto range restricted to the visible x range."""
        for vb in self.shownExtraViewBoxes():
            vb.setAutoVisible(y=True)
            vb.enableAutoRange(axis=pg.ViewBox.YAxis)

    def isExtraYAxesAutoRange(self):
        return all(
            vb.autoRangeEnabled()[1] for vb in self.shownExtraViewBoxes()
        )

    def autoBtnClicked(self, *args):
        if self.autoBtn.mode == 'auto':
            self.enableExtraYAxesAutoRange()
        super().autoBtnClicked()

    def updateButtons(self):
        super().updateButtons()
        if self.isExtraYAxesAutoRange():
            return
        # An extra axis zoomed by hand while the main view box still
        # autoscales: show "A" as pyqtgraph does for the main view box
        try:
            if (getattr(self, '_exportOpts', False) is False
                    and getattr(self, 'mouseHovering', False)
                    and not getattr(self, 'buttonsHidden', False)):
                self.autoBtn.show()
        except RuntimeError:
            pass  # deleted plot


class _MeasurementColumnDelegate(widgets_search_widgets._ButtonSearchHighlightDelegate):
    indicatorWidth = 28

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if not index.flags() & Qt.ItemIsUserCheckable:
            return
        option.features &= ~(
            QStyleOptionViewItem.HasCheckIndicator
            | QStyleOptionViewItem.HasDecoration
        )
        option.icon = QIcon()
        option.rect.adjust(self.indicatorWidth, 0, 0, 0)

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        if not index.flags() & Qt.ItemIsUserCheckable:
            return
        rect = QRect(
            option.rect.left()+2, option.rect.center().y()-8, 24, 16
        )
        icon = index.data(Qt.DecorationRole)
        painter.save()
        if icon is not None and not icon.isNull():
            icon.paint(painter, rect, Qt.AlignCenter)
        else:
            painter.setPen(option.palette.mid().color())
            painter.drawRect(rect.adjusted(4, 2, -4, -2))
        painter.restore()

    def editorEvent(self, event, model, option, index):
        if not index.flags() & Qt.ItemIsUserCheckable:
            return False
        if event.type() in (
                QEvent.MouseButtonPress, QEvent.MouseButtonDblClick,
                QEvent.MouseButtonRelease):
            # The whole row (indicator and name) toggles the selection
            if event.button() != Qt.LeftButton:
                return False
            if not option.rect.contains(event.pos()):
                return False
            if event.type() != QEvent.MouseButtonRelease:
                return True
        elif event.type() == QEvent.KeyPress:
            if event.key() not in (Qt.Key_Space, Qt.Key_Select):
                return False
        else:
            return False
        state = (
            Qt.Unchecked
            if Qt.CheckState(index.data(Qt.CheckStateRole)) == Qt.Checked
            else Qt.Checked
        )
        return model.setData(index, state, Qt.CheckStateRole)

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setWidth(size.width()+self.indicatorWidth)
        size.setHeight(max(size.height(), 20))
        return size


class _NoOpItem:
    """Placeholder for main GUI items that do not exist in the viewer 
    (e.g., the right image LUT). Every method call is a no-op."""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

class _NoAutoPilotProfile:
    """The viewer does not record auto-pilot profiles like the main GUI"""
    def storeSelectedChannel(self, *args, **kwargs):
        pass
    
    def storeClickMessageBox(self, *args, **kwargs):
        pass
    
    def storeLoadedFluoChannels(self, *args, **kwargs):
        pass
