"""Widgets reached from the GUI mixins that the measurements viewer also
composes, so changing them affects both windows.

- Navigation (gui/mixins/navigation.py): navigateQScrollBar,
  linkedQScrollbar, SwitchPlaneCombobox.
- Overlay (gui/mixins/overlay.py): overlayLabelsGradientWidget,
  OverlayImageItem.
- Viewer tools and object info (gui/mixins/viewer_tools.py,
  gui/mixins/object_info.py): ScaleBar, TimestampItem, LabelItem, ROI,
  ZoomROI, the object properties tab guiTabControl with its group boxes
  (PixelSizeGroupbox, objPropsQGBox, objIntesityMeasurQGBox).
- Custom annotations (gui/mixins/custom_annotations.py):
  CustomAnnotToolButton, BaseScatterPlotItem,
  CustomAnnotationScatterPlotItem.
- Annotations (gui/mixins/annotations.py): CheckBox.
- Timestamp properties dialog (gui/mixins/dialogs.py): FontSizeWidget,
  TimeWidget and its sign_int_mapper.

Widgets used only by the main GUI window are in gui/widgets/.
"""
from collections import deque
import datetime
import numpy as np
from qtpy.QtCore import Signal, Qt, QPoint, QPoint, QRectF
from qtpy.QtWidgets import (
    QLabel, QHBoxLayout, QCheckBox, QWidget, QVBoxLayout, QGridLayout,
    QScrollArea, QComboBox, QScrollBar, QGroupBox, QAbstractSlider,
    QWidgetAction, QAction, QTabWidget, QStyle, QMenu, QActionGroup,
    QGraphicsObject, QStyleOptionSlider
)
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import measurements
from cellacdc import _core
from cellacdc.shared.widgets import toolbars as widgets_toolbars
from cellacdc.shared.widgets import line_edits as widgets_line_edits
from cellacdc.shared.widgets import push_buttons as widgets_push_buttons
from cellacdc.shared.widgets import basic_widgets as widgets_basic_widgets
from cellacdc.shared.widgets import numeric_inputs as widgets_numeric_inputs
from cellacdc.shared.widgets import plot_items as widgets_plot_items


sign_int_mapper = {
    '+': 1, '-': -1
}


class BaseScatterPlotItem(pg.ScatterPlotItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
    
    def tempClear(self):
        try:
            self._prevData = [d.copy() for d in self.getData()]
            self.setData([], [])
        except Exception as e:
            pass
    
    def restore(self):
        if self._prevData is not None:
            if self._prevData[0] is not None:
                self.setData(*self._prevData)


class CustomAnnotationScatterPlotItem(BaseScatterPlotItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)


class CheckBox(QCheckBox):
    sigToggled = Signal(bool, object)

    def __init__(self, *args, keyPressCallback=None,
                 # rightclick_menu_func=None
                 ):
        super().__init__(*args)
        self.keyPressCallback = keyPressCallback
        self.setFocusPolicy(Qt.NoFocus)
        self.toggled.connect(self.onToggled)
        self._exclusiveCheckboxes: list[QCheckBox] = []
        self._linkedCheckboxes: dict[str, QCheckBox] = {}
        self.exclusiveCheckboxesStates: dict[str, bool] = {}
        # if rightclick_menu_func is not None:
        #     self.rightclick_menu = rightclick_menu_func(self)
    
    def keyPressEvent(self, event) -> None:
        event.ignore()
        if self.keyPressCallback is None:
            return

        self.keyPressCallback()
    
    def setExclusiveOnCheckCheckbox(self, checkbox: QCheckBox):
        # Keep a list of checkboxes that needs to be unchecked when the `self`
        # checkbox is checked --> see `onToggled` method
        self._exclusiveCheckboxes.append(checkbox)
    
    def onToggled(self, checked: bool):
        for checkbox in self._exclusiveCheckboxes:
            name = checkbox.text()
            self.exclusiveCheckboxesStates[name] = checkbox.isChecked()

            if not checked:
                continue
            
            checkbox.setChecked(False)
        
        for checkbox in self._linkedCheckboxes.values():
            checkbox.setChecked(checked)
        
        self.sigToggled.emit(checked, self)
    
    def setCheckedNoSignal(self, checked: bool):
        self.blockSignals(True)
        self.setChecked(checked)
        self.blockSignals(False)

    def setCheckedAlwaysEmit(self, checked: bool):
        """Set the checked state and emit `sigToggled` even if the state
        does not change.

        `setChecked` emits `toggled` (and hence `sigToggled`) only when the
        state actually changes. Use this method when the slots connected to
        `sigToggled` must run regardless (e.g., to refresh what they display).
        Linked checkboxes are treated the same way.
        """
        if self.isChecked() != checked:
            self.setChecked(checked)
            return

        for checkbox in self._linkedCheckboxes.values():
            if checkbox.isChecked() != checked:
                checkbox.setChecked(checked)
            elif isinstance(checkbox, CheckBox):
                checkbox.sigToggled.emit(checked, checkbox)

        self.sigToggled.emit(checked, self)

    def isLinkedTo(self, checkbox: QCheckBox):
        return self._linkedCheckboxes.get(checkbox.text()) is checkbox

    def setLinkedCheckbox(self, checkbox: QCheckBox, linked: bool):
        if linked:
            self._linkedCheckboxes[checkbox.text()] = checkbox
        else:
            self._linkedCheckboxes.pop(checkbox.text(), None)


class CustomAnnotToolButton(widgets_toolbars.ToolButtonCustomColor):
    sigRemoveAction = Signal(object)
    sigKeepActiveAction = Signal(object)
    sigModifyAction = Signal(object)
    sigHideAction = Signal(object)

    def __init__(
            self, symbol, color, keepToolActive=True, parent=None,
            isHideChecked=True
        ):
        super().__init__(symbol, color=color, parent=parent)
        self.symbol = symbol
        self.keepToolActive = keepToolActive
        self.isHideChecked = isHideChecked
        # List of (text, callback) added on top of the context menu
        self.extraContextMenuActions = []
        self.sigRightClick.connect(self.showContextMenu)

    def showContextMenu(self, event):
        contextMenu = QMenu(self)
        for text, callback in self.extraContextMenuActions:
            extraAction = QAction(text, contextMenu)
            extraAction.triggered.connect(
                lambda checked=False, callback=callback: callback(self)
            )
            contextMenu.addAction(extraAction)
        contextMenu.addSeparator()

        removeAction = QAction('Remove annotation')
        removeAction.triggered.connect(self.removeAction)
        contextMenu.addAction(removeAction)

        editAction = QAction('Modify annotation parameters...')
        editAction.triggered.connect(self.modifyAction)
        contextMenu.addAction(editAction)

        hideAction = QAction('Hide annotations')
        hideAction.setCheckable(True)
        hideAction.setChecked(self.isHideChecked)
        hideAction.triggered.connect(self.hideAction)
        contextMenu.addAction(hideAction)

        keepActiveAction = QAction('Keep tool active after using it')
        keepActiveAction.setCheckable(True)
        keepActiveAction.setChecked(self.keepToolActive)
        keepActiveAction.triggered.connect(self.keepToolActiveActionToggled)
        contextMenu.addAction(keepActiveAction)

        contextMenu.exec(event.globalPos())

    def keepToolActiveActionToggled(self, checked):
        self.keepToolActive = checked
        self.sigKeepActiveAction.emit(self)

    def modifyAction(self):
        self.sigModifyAction.emit(self)

    def removeAction(self):
        self.sigRemoveAction.emit(self)

    def hideAction(self, checked):
        # The host window stores the option as a user setting
        # (`CustomAnnotationsGuiElements.customAnnotHide`)
        self.isHideChecked = checked
        self.sigHideAction.emit(self)


class PixelSizeGroupbox(QGroupBox):
    sigValueChanged = Signal(float, float, float)
    sigReset = Signal()
    
    def __init__(self, parent=None):
        super().__init__('Pixel size', parent)
        
        mainLayout = QGridLayout()
        
        row = 0
        label = QLabel('Pixel width (μm): ')
        self.pixelWidthWidget = widgets_line_edits.FloatLineEdit(initial=1.0)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.pixelWidthWidget, row, 1)
        
        row += 1
        label = QLabel('Pixel height (μm): ')
        self.pixelHeightWidget = widgets_line_edits.FloatLineEdit(initial=1.0)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.pixelHeightWidget, row, 1)
        
        row += 1
        label = QLabel('Voxel depth (μm): ')
        self.voxelDepthWidget = widgets_line_edits.FloatLineEdit(initial=1.0)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.voxelDepthWidget, row, 1)
        
        row += 1
        resetButton = widgets_push_buttons.reloadPushButton('Reset')
        mainLayout.addWidget(
            resetButton, row, 1, alignment=Qt.AlignRight
        )
        
        row += 1
        mainLayout.addWidget(widgets_basic_widgets.QHLine(), row, 0, 1, 2)
        
        mainLayout.setColumnStretch(0, 0)
        mainLayout.setColumnStretch(1, 1)

        self.setLayout(mainLayout)
        
        self.pixelWidthWidget.valueChanged.connect(self.emitValueChanged)
        self.pixelHeightWidget.valueChanged.connect(self.emitValueChanged)
        self.voxelDepthWidget.valueChanged.connect(self.emitValueChanged)
        resetButton.clicked.connect(self.emitReset)
    
    def emitReset(self):
        self.sigReset.emit()
    
    def emitValueChanged(self, value):
        PhysicalSizeX = self.pixelWidthWidget.value()
        PhysicalSizeY = self.pixelHeightWidget.value()
        PhysicalSizeZ = self.voxelDepthWidget.value()
        self.sigValueChanged.emit(PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ)

class objPropsQGBox(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, 'Properties', parent)

        mainLayout = QGridLayout()

        row = 0
        label = QLabel('Object ID: ')
        self.idSB = widgets_line_edits.IntLineEdit()
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.idSB, row, 1)

        row += 1
        mainLayout.addWidget(widgets_basic_widgets.QHLine(), row, 0, 1, 2)

        row += 1
        self.notExistingIDLabel = QLabel()
        self.notExistingIDLabel.setStyleSheet(
            'font-size:11px; color: rgb(255, 0, 0);'
        )
        mainLayout.addWidget(
            self.notExistingIDLabel, row, 0, 1, 2, alignment=Qt.AlignCenter
        )

        row += 1
        label = QLabel('Area (pixel): ')
        self.cellAreaPxlSB = widgets_line_edits.IntLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellAreaPxlSB, row, 1)

        row += 1
        label = QLabel('Area (<span>&#181;</span>m<sup>2</sup>): ')
        self.cellAreaUm2DSB = widgets_line_edits.FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellAreaUm2DSB, row, 1)

        row += 1
        mainLayout.addWidget(widgets_basic_widgets.QHLine(), row, 0, 1, 2)

        row += 1
        label = QLabel('Rotational volume (voxel): ')
        self.cellVolVoxSB = widgets_line_edits.IntLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolVoxSB, row, 1)

        row += 1
        label = QLabel('3D volume (voxel): ')
        self.cellVolVox3D_SB = widgets_line_edits.IntLineEdit(readOnly=True)
        self.cellVolVox3D_SB.label = label
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolVox3D_SB, row, 1)

        row += 1
        label = QLabel('Rotational volume (fl): ')
        self.cellVolFlDSB = widgets_line_edits.FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolFlDSB, row, 1)

        row += 1
        label = QLabel('3D volume (fl): ')
        self.cellVolFl3D_DSB = widgets_line_edits.FloatLineEdit(readOnly=True)
        self.cellVolFl3D_DSB.label = label
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolFl3D_DSB, row, 1)

        row += 1
        mainLayout.addWidget(widgets_basic_widgets.QHLine(), row, 0, 1, 2)

        row += 1
        label = QLabel('Solidity: ')
        self.solidityDSB = widgets_line_edits.FloatLineEdit(readOnly=True)
        self.solidityDSB.setMaximum(1)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.solidityDSB, row, 1)

        row += 1
        label = QLabel('Elongation: ')
        self.elongationDSB = widgets_line_edits.FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.elongationDSB, row, 1)

        row += 1
        mainLayout.addWidget(widgets_basic_widgets.QHLine(), row, 0, 1, 2)

        row += 1
        propsNames = measurements.get_props_names()[1:]
        self.additionalPropsCombobox = QComboBox()
        self.additionalPropsCombobox.addItems(propsNames)
        self.additionalPropsCombobox.indicator = widgets_line_edits.FloatLineEdit(readOnly=True)
        mainLayout.addWidget(self.additionalPropsCombobox, row, 0)
        mainLayout.addWidget(self.additionalPropsCombobox.indicator, row, 1)

        row += 1
        mainLayout.addWidget(widgets_basic_widgets.QHLine(), row, 0, 1, 2)
        
        mainLayout.setColumnStretch(0, 0)
        mainLayout.setColumnStretch(1, 1)

        self.setLayout(mainLayout)

class objIntesityMeasurQGBox(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, 'Intensity measurements', parent)

        mainLayout = QGridLayout()

        row = 0
        label = QLabel('Raw intensity measurements')

        row += 1
        label = QLabel('Channel: ')
        self.channelCombobox = QComboBox()
        self.channelCombobox.addItem('placeholderlong')
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.channelCombobox, row, 1)

        row += 1
        label = QLabel('Minimum: ')
        self.minimumDSB = widgets_line_edits.FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.minimumDSB, row, 1)

        row += 1
        label = QLabel('Maximum: ')
        self.maximumDSB = widgets_line_edits.FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.maximumDSB, row, 1)

        row += 1
        label = QLabel('Mean: ')
        self.meanDSB = widgets_line_edits.FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.meanDSB, row, 1)

        row += 1
        label = QLabel('Median: ')
        self.medianDSB = widgets_line_edits.FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.medianDSB, row, 1)

        row += 1
        metricsDesc = measurements._get_metrics_names()
        metricsFunc, _ = measurements.standard_metrics_func()
        items = list(set([metricsDesc[key] for key in metricsFunc.keys()]))
        items.append('Concentration')
        items.sort()
        nameFuncDict = {}
        for name, desc in metricsDesc.items():
            if name.find('_dataPrepBkgr')!=-1 or  name.find('_manualBkgr')!=-1:
                # Skip dataPrepBkgr and manualBkgr since in the dock widget 
                # we display only autoBkgr metrics
                continue
            if name.startswith('concentration_'):
                # We use amount function because dividing by volume is taken 
                # care in the GUI
                name = 'amount_autoBkgr'
            nameFuncDict[desc] = metricsFunc[name]

        funcionCombobox = QComboBox()
        funcionCombobox.addItems(items)
        self.additionalMeasCombobox = funcionCombobox
        self.additionalMeasCombobox.indicator = widgets_line_edits.FloatLineEdit(readOnly=True)
        self.additionalMeasCombobox.functions = nameFuncDict
        mainLayout.addWidget(funcionCombobox, row, 0)
        mainLayout.addWidget(self.additionalMeasCombobox.indicator, row, 1)

        self.setLayout(mainLayout)

    def addChannels(self, channels):
        self.channelCombobox.clear()
        self.channelCombobox.addItems(channels)

class guiTabControl(QTabWidget):
    def __init__(self, *args):
        super().__init__(args[0])

        self._defaultPixelSize = None
        
        self.propsTab = QScrollArea(self)

        container = QWidget()
        layout = QVBoxLayout()

        self.pixelSizeQGBox = PixelSizeGroupbox(parent=self.propsTab)
        self.propsQGBox = objPropsQGBox(parent=self.propsTab)
        self.intensMeasurQGBox = objIntesityMeasurQGBox(parent=self.propsTab)

        self.highlightCheckbox = QCheckBox('Highlight objects on mouse hover')
        self.highlightCheckbox.setChecked(False)
        
        self.highlightSearchedCheckbox = QCheckBox('Highlight searched object')
        self.highlightSearchedCheckbox.setChecked(True)

        highlightLayout = QHBoxLayout()
        highlightLayout.addWidget(self.highlightCheckbox)
        highlightLayout.addStretch(1)
        highlightLayout.addWidget(QLabel('|'))
        highlightLayout.addStretch(1)
        highlightLayout.addWidget(self.highlightSearchedCheckbox)
        
        layout.addLayout(highlightLayout)
        layout.addWidget(self.pixelSizeQGBox)
        layout.addWidget(self.propsQGBox)
        layout.addWidget(self.intensMeasurQGBox)  
        layout.addStretch(1)     
        container.setLayout(layout)

        self.propsTab.setWidgetResizable(True)
        self.propsTab.setWidget(container)
        self.addTab(self.propsTab, 'Measurements')
        
        self.pixelSizeQGBox.sigValueChanged.connect(self.pixelSizeChanged)
        self.pixelSizeQGBox.sigReset.connect(self.resetPixelSize)
    
    def addChannels(self, channels):
        self.intensMeasurQGBox.addChannels(channels)
    
    def resetPixelSize(self):
        if self._defaultPixelSize is None:
            return
        
        self.initPixelSize(*self._defaultPixelSize)
    
    def initPixelSize(self, PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ):
        self.pixelSizeQGBox.pixelWidthWidget.setValue(PhysicalSizeX)
        self.pixelSizeQGBox.pixelHeightWidget.setValue(PhysicalSizeY)
        self.pixelSizeQGBox.voxelDepthWidget.setValue(PhysicalSizeZ)
        self._defaultPixelSize = (PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ)
    
    def pixelSizeChanged(self, PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ):
        propsQGBox = self.propsQGBox
        yx_pxl_to_um2 = PhysicalSizeY*PhysicalSizeX
        vox_rot_to_fl = float(PhysicalSizeY)*pow(float(PhysicalSizeX), 2)
        vox_3D_to_fl = PhysicalSizeZ*PhysicalSizeY*PhysicalSizeX
        
        area_pxl = propsQGBox.cellAreaPxlSB.value()
        area_um2 = area_pxl*yx_pxl_to_um2
        propsQGBox.cellAreaUm2DSB.setValue(area_um2)
        
        vol_rot_vox = propsQGBox.cellVolVoxSB.value()
        vol_rot_fl = vol_rot_vox*vox_rot_to_fl
        propsQGBox.cellVolFlDSB.setValue(vol_rot_fl)
        
        vol_3D_vox = propsQGBox.cellVolVox3D_SB.value()
        vol_3D_fl = vol_3D_vox*vox_3D_to_fl
        propsQGBox.cellVolFl3D_DSB.setValue(vol_3D_fl)


class ROI(pg.ROI):
    def __init__(
            self, pos, size=pg.Point(1, 1), angle=0, invertible=False, 
            maxBounds=None, snapSize=1, scaleSnap=False, translateSnap=False, 
            rotateSnap=False, parent=None, pen=None, hoverPen=None, 
            handlePen=None, handleHoverPen=None, movable=True, rotatable=True, 
            resizable=True, removable=False, aspectLocked=False
        ):
        super().__init__(
            pos, size, angle, invertible, maxBounds, snapSize, scaleSnap, 
            translateSnap, rotateSnap, parent, pen, hoverPen, handlePen, 
            handleHoverPen, movable, rotatable, resizable, removable, 
            aspectLocked
        )
    
    def slice(self, zRange=None, tRange=None):
        x0, y0 = [int(round(c)) for c in self.pos()]
        w, h = [int(round(c)) for c in self.size()]
        xmin, xmax = x0, x0+w
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        ymin, ymax = y0, y0+h
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        if zRange is not None:
            zmin, zmax = zRange
            _slice = (slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax))
        else:
            _slice = (slice(ymin, ymax), slice(xmin, xmax))
        if tRange is not None:
            tmin, tmax = tRange
            _slice = (slice(tmin, tmax), *_slice)
        return _slice

    def bbox(self):
        x0, y0 = [int(round(c)) for c in self.pos()]
        w, h = [int(round(c)) for c in self.size()]
        xmin, xmax = x0, x0+w
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        ymin, ymax = y0, y0+h
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        
        return ymin, xmin, ymax, xmax


class ZoomROI(ROI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.viewRangesQueue = deque()
        
    def getLastRange(self):
        xRange, yRange = self.viewRangesQueue.pop()
        return xRange, yRange
    
    def storeLastRange(self, xRange, yRange):
        self.viewRangesQueue.append((xRange, yRange))


class navigateQScrollBar(widgets_numeric_inputs.ScrollBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._disableCustomPressEvent = False
        self.signal_slot_mapper = {}
        self._absoluteMaximum = None

    def disableCustomPressEvent(self):
        self._disableCustomPressEvent = True

    def enableCustomPressEvent(self):
        self._disableCustomPressEvent = False

    def setAbsoluteMaximum(self, absoluteMaximum):
        self._absoluteMaximum = absoluteMaximum

    def absoluteMaximum(self):
        return self._absoluteMaximum
    
    def setValueNoSignal(self, value):
        for signal_name, slot in self.signal_slot_mapper.items():
            signal = getattr(self, signal_name)
            try:
                signal.disconnect()
            except Exception as e:
                pass
        
        self.setSliderPosition(value)
        self.connectEvents(self.signal_slot_mapper)
    
    def handleMousePressEventFromAppFilter(self, event):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)

        control = self.style().hitTestComplexControl(
            QStyle.CC_ScrollBar,
            opt,
            event.position().toPoint(),
            self,
        )
        isSliderSingleStepAddAction = (
            event.button() == Qt.MouseButton.LeftButton 
            and control == QStyle.SC_ScrollBarAddLine
        )
        isRangeZero = (self.maximum() - self.minimum()) == 0
        if isSliderSingleStepAddAction and isRangeZero:
            # Force trigger action step add when range is 0
            self.triggerAction(QAbstractSlider.SliderSingleStepAdd)
    
    def connectEvents(self, signal_slot_mapper: dict):
        self.signal_slot_mapper = signal_slot_mapper
        for signal_name, slot in signal_slot_mapper.items():
            signal = getattr(self, signal_name)
            try:
                signal.disconnect()
            except Exception as e:
                pass
            signal.connect(slot)

class linkedQScrollbar(widgets_numeric_inputs.ScrollBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._linkedScrollBar = None

    def linkScrollBar(self, scrollbar):
        self._linkedScrollBar = scrollbar
        scrollbar.setSliderPosition(self.sliderPosition())

    def unlinkScrollBar(self):
        self._linkedScrollBar = None

    def setSliderPosition(self, position):
        QScrollBar.setSliderPosition(self, position)
        if self._linkedScrollBar is not None:
            self._linkedScrollBar.setSliderPosition(position)

    def setMaximum(self, max):
        QScrollBar.setMaximum(self, max)
        if self._linkedScrollBar is not None:
            self._linkedScrollBar.setMaximum(max)


class overlayLabelsGradientWidget(pg.GradientWidget):
    def __init__(
            self, imageItem, selectActionGroup, segmEndname, 
            parent=None, orientation='right'
        ):
        pg.GradientWidget.__init__(self, parent=parent, orientation=orientation)

        self.imageItem = imageItem
        self.selectActionGroup = selectActionGroup

        for action in self.menu.actions():
            if action.text() == 'HSV':
                HSV_action = action
            elif action.text() == 'RGB':
                RGB_ation = action
        self.menu.removeAction(HSV_action)
        self.menu.removeAction(RGB_ation)

        # Shuffle colors action
        self.shuffleCmapAction =  QAction(
            'Randomly shuffle colormap   (Shift+S)', self
        )
        self.menu.addAction(self.shuffleCmapAction)

        # Drawing mode
        drawModeMenu = QMenu('Drawing mode', self)
        self.drawModeActionGroup = QActionGroup(self)
        contoursDrawModeAction = QAction('Draw contours', drawModeMenu)
        contoursDrawModeAction.setCheckable(True)
        contoursDrawModeAction.setChecked(True)
        contoursDrawModeAction.segmEndname = segmEndname
        self.drawModeActionGroup.addAction(contoursDrawModeAction)
        drawModeMenu.addAction(contoursDrawModeAction)
        olDrawModeAction = QAction('Overlay labels', drawModeMenu)
        olDrawModeAction.setCheckable(True)
        olDrawModeAction.segmEndname = segmEndname
        self.drawModeActionGroup.addAction(olDrawModeAction)
        drawModeMenu.addAction(olDrawModeAction)
        self.menu.addMenu(drawModeMenu)

        self.labelsAlphaMenu = self.menu.addMenu(
            'Overlay labels alpha...'
        )
        hbox = QHBoxLayout()
        self.labelsAlphaSlider = widgets_numeric_inputs.sliderWithSpinBox(
            title='Alpha', title_loc='in_line', isFloat=True,
            normalize=True
        )
        self.labelsAlphaSlider.setMaximum(100)
        self.labelsAlphaSlider.setSingleStep(0.05)
        self.labelsAlphaSlider.setValue(0.3)
        hbox.addWidget(self.labelsAlphaSlider)
        widget = QWidget()
        widget.setLayout(hbox)
        act = QWidgetAction(self)
        act.setDefaultWidget(widget)
        self.labelsAlphaMenu.addSeparator()
        self.labelsAlphaMenu.addAction(act)

        self.menu.addSeparator()
        self.menu.addSection('Select segm. file to adjust:')
        for action in selectActionGroup.actions():
            self.menu.addAction(action)
        
        self.item.loadPreset('viridis')
        self.updateImageLut(None)
        self.updateImageOpacity(0.3)

        # Connect events
        self.sigGradientChangeFinished.connect(self.updateImageLut)
        self.labelsAlphaSlider.valueChanged.connect(self.updateImageOpacity)
        self.shuffleCmapAction.triggered.connect(self.shuffleCmap)
    
    def shuffleCmap(self):
        lut = self.imageItem.lut
        np.random.shuffle(lut)
        lut[0] = [0,0,0,0]
        self.imageItem.setLookupTable(lut)
        self.imageItem.update()
    
    def updateImageLut(self, gradientItem):
        lut = np.zeros((255, 4), dtype=np.uint8)
        lut[:,-1] = 255
        lut[:,:-1] = self.item.colorMap().getLookupTable(0,1,255)
        np.random.shuffle(lut)
        lut[0] = [0,0,0,0]
        self.imageItem.setLookupTable(lut)
        self.imageItem.setLevels([0, 255])
    
    def updateImageOpacity(self, value):
        self.imageItem.setOpacity(value)


class OverlayImageItem(pg.ImageItem):
    def __init__(
            self, image=None, **kargs
        ):
        super().__init__(image, **kargs)
        self.autoLevelsEnabled = None
    
    def setEnableAutoLevels(self, enabled: bool):
        self.autoLevelsEnabled = enabled
    
    def setImage(
            self, image=None, autoLevels=None, **kargs
        ):
        if autoLevels is None:
            autoLevels = self.autoLevelsEnabled
        
        super().setImage(image, autoLevels=autoLevels, **kargs)
        
    def setOpacity(self, value, **kwargs):
        if value == 0:
            value = 0.001
        
        if value == 1:
            value = 0.999
            
        super().setOpacity(value)


class LabelItem(pg.LabelItem):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def bbox(self):
        xl, yl = self.pos().x(), self.pos().y()
        wl, hl = self.itemRect().width(), self.itemRect().height()
        return yl, xl, yl+hl, xl+wl
    
    def setBold(self, bold=True):
        self.origPos = self.pos()
        self.setText(self.text, bold=bold)
        self.setPos(self.origPos)

class ScaleBar(QGraphicsObject): 
    sigEditProperties = Signal(object)
    sigRemove = Signal(object)
    
    def __init__(self, imageShape, viewRange, parent=None):
        super().__init__(parent)
        self.SizeY, self.SizeX = imageShape
        self.updateViewRange(viewRange)
        self.plotItem = widgets_plot_items.PlotCurveItem()
        self.labelItem = LabelItem()
        self._x_pad = 5
        self._y_pad = 3
        self._highlighted = False
        self._parent = parent
        self.clicked = False
        self.createContextMenu()
    
    def updateViewRange(self, viewRange):
        xRange, yRange = viewRange
        x0, x1 = xRange
        y0, y1 = yRange
        if x0 < 0:
            x0 = 0
        
        if x1 > self.SizeX:
            x1 = self.SizeX
        
        if y0 < 0:
            y0 = 0
        
        if y1 > self.SizeY:
            y1 = self.SizeY
        
        self.xmax = x1
        self.xmin = x0
        
        self.ymax = y1
        self.ymin = y0
    
    def createContextMenu(self):
        self.contextMenu = QMenu()
        action = QAction('Edit properties...', self.contextMenu)
        action.triggered.connect(self.emitEditProperties)
        self.contextMenu.addAction(action)
        self.contextMenu.addSeparator()
        action = QAction('Remove', self.contextMenu)
        action.triggered.connect(self.emitRemove)
        self.contextMenu.addAction(action)
    
    def emitEditProperties(self):
        self.setHighlighted(False)
        self.sigEditProperties.emit(self.properties())
    
    def emitRemove(self):
        self.sigRemove.emit(self)
    
    def isHighlighted(self):
        return self._highlighted
    
    def setHighlighted(self, highlighted):
        if self._highlighted and highlighted:
            return
        
        if not self._highlighted and not highlighted:
            return
        
        pen = self.highlightPen if highlighted else self.pen
        self.labelItem.setBold(bold=highlighted)
        self.plotItem.setPen(pen)
        
        self._highlighted = highlighted
    
    def showContextMenu(self, x, y):
        self.contextMenu.popup(QPoint(int(x), int(y)))
    
    def properties(self):
        properties = {
            'thickness': self._thickness,
            'length_pixel': self._length,
            'length_unit': self._length_unit,
            'is_text_visible': self._is_text_visible,
            'color': self._color,
            'loc': self._loc,
            'font_size': float(self._font_size[:-2]),
            'unit': self._unit,
            'num_decimals': self._num_decimals,
            'move_with_zoom': self._move_with_zoom,
        }
        return properties
    
    def move(self, xm, ym):
        self._loc = 'Custom'
        
        Dy = ym - self.yc
        Dx = xm - self.xc
        
        x0 = self.x0c + Dx
        x1 = x0 + self._length
        y0 = y1 = self.y0c + Dy
        self.plotItem.setData([x0, x1], [y0, y1])
        self.setTextPos()
    
    def paint(self, painter, option, widget):
        pass
    
    def boundingRect(self):
        ymin, xmin, ymax, xmax = self.bbox()
        return QRectF(xmin, ymin, xmax-xmin, ymax-ymin)
    
    def setLocationProperty(self, loc: str):
        self._loc = loc 
    
    def setMoveWithZoomProperty(self, move_with_zoom):
        self._move_with_zoom = move_with_zoom
    
    def setProperties(
            self, 
            length_pixel, 
            length_unit,
            thickness=3, 
            color='w',
            is_text_visible=True, 
            loc='top-left',
            font_size=12,
            unit='',
            num_decimals=0,
            move_with_zoom=False
        ):
        self._loc = loc
        self._color = color
        self._length = length_pixel
        self._length_unit = length_unit
        self._is_text_visible = is_text_visible
        self._font_size = f'{font_size}px'
        self._unit = unit
        self._num_decimals = num_decimals
        self._move_with_zoom = move_with_zoom
        self._thickness = thickness
        self.pen = pg.mkPen(width=thickness, color=color, cosmetic=False)
        self.highlightPen = pg.mkPen(
            width=thickness+2, color=color, cosmetic=False
        )
        self.pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        self.highlightPen.setCapStyle(Qt.PenCapStyle.FlatCap)
        self.plotItem.setPen(self.pen)
    
    def updatePhysicalLength(self, PhysicalSizeX):
        length_unit = self._length_unit
        unit = self._unit
        length_um = _core.convert_length(length_unit, unit, 'μm')
        length_pixel = length_um/PhysicalSizeX
        self._length = length_pixel
        self.update()
    
    def addToAxis(self, ax):
        # The bar is positioned relative to the view range: it must not
        # extend the bounds used by auto range
        ax.addItem(self.plotItem, ignoreBounds=True)
        ax.addItem(self.labelItem, ignoreBounds=True)

    def setText(self):
        if self._is_text_visible:
            number = round(self._length_unit, self._num_decimals)
            if self._num_decimals == 0:
                number = int(number)
            text = f'{number} {self._unit}'
        else:
            text = ''
        self.labelItem.setText(
            text, color=self._color, size=self._font_size
        )
    
    def setTextPos(self):
        xx, yy = self.plotItem.getData()
        x0 = xx[0]
        y0 = yy[0]
        xc = x0 + self._length/2
        wl = self.labelItem.itemRect().width()
        hl = self.labelItem.itemRect().height()
        xl = xc-wl/2
        yt = y0-hl    
        self.labelItem.setPos(xl, yt)
    
    def updatePosViewRangeChanged(self, viewRange):
        if self._loc == 'custom':
            xx, yy = self.plotItem.getData()
            x0p = xx[0]
            y0p = yy[0]
            xcp = x0p + self._length/2
            hl = self.labelItem.itemRect().height()
            ycp = y0p - hl/2  
            x0 = self.xmin
            y0 = self.ymin
            x_range = self.xmax - x0
            y_range = self.ymax - y0
            Dx_perc = (xcp - x0)/x_range
            Dy_perc = (ycp - y0)/y_range   
            
            self.updateViewRange(viewRange)
            
            X0 = self.xmin
            Y0 = self.ymin
            
            X_range = self.xmax - X0
            Y_range = self.ymax - Y0
            
            Xcp = X0 + (Dx_perc*X_range)
            Ycp = Y0 + (Dy_perc*Y_range)
            X0p = Xcp - (self._length/2)
            Y0p = Ycp + (hl/2)
            
            X1p = X0p + self._length
            Y1p = Y0p
            
            self.plotItem.setData([X0p, X1p], [Y0p, Y1p])
        else:
            self.updateViewRange(viewRange)
            self.update()
    
    def getStartXCoordFromLoc(self, loc):
        if loc == 'custom':
            xx, yy = self.plotItem.getData()
            x0 = xx[0]
            return x0
        
        self.setText()
        wl = self.labelItem.itemRect().width()
        if loc.find('left') != -1:
            x0 = self._x_pad + self.xmin
            xc = x0 + self._length/2
            xl = xc-wl/2
            if xl < x0:
                # Text is larger than line --> move line to the right
                x0 = self._x_pad + abs(xl-self._x_pad)
        else:
            x0 = self.xmax - self._length - self._x_pad
            xc = x0 + self._length/2
            x1 = x0 + self._length      
            xr = xc+wl/2
            if xr > x1:
                # Text is larger than line --> move line to the left
                delta_overshoot = xr - x1
                x0 = x0 - delta_overshoot
        return x0  
    
    def getStartYCoordFromLoc(self, loc):
        if loc == 'custom':
            xx, yy = self.plotItem.getData()
            y0 = yy[0]
            return y0
        
        self.setText()
        textHeight = self.labelItem.itemRect().height()
        if loc.find('top') != -1:
            return textHeight + self._y_pad + self.ymin
        else:
            return self.ymax - self._y_pad - self._thickness
    
    def update(self):
        x0 = self.getStartXCoordFromLoc(self._loc) # + self._thickness/2
        y0 = self.getStartYCoordFromLoc(self._loc)
        
        x1 = x0 + self._length # - self._thickness/2
        self.plotItem.setData([x0, x1], [y0, y0])
        
        self.setText()
        self.setTextPos()
    
    def draw(self, length_pixel, length_unit, **kwargs):
        self.setProperties(length_pixel, length_unit, **kwargs)
        self.update()
        
    def bbox(self):
        y_line_min, x_line_min, y_line_max, x_line_max = self.plotItem.bbox()
        y_lab_min, x_lab_min, y_lab_max, x_lab_max = self.labelItem.bbox()
        ymin = min(y_line_min, y_lab_min)
        xmin = min(x_line_min, x_lab_min)
        ymax = max(y_line_max, y_lab_max)
        xmax = max(x_line_max, x_lab_max)
        return ymin, xmin, ymax, xmax
    
    def mousePressed(self, x, y):
        self.clicked = True
        self.xc, self.yc = x, y
        xx, yy = self.plotItem.getData()
        self.x0c = xx[0]
        self.y0c = yy[0]
    
    def removeFromAxis(self, ax):
        ax.removeItem(self.labelItem)
        ax.removeItem(self.plotItem)


class SwitchPlaneCombobox(QComboBox):
    sigPlaneChanged = Signal(str, str)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addItems(['xy', 'zy', 'zx'])
        self._previousPlane = 'xy'
        self.currentTextChanged.connect(self.emitPlaneChanged)
    
    def emitPlaneChanged(self, plane):
        self.sigPlaneChanged.emit(self._previousPlane, plane)
        self._previousPlane = plane
    
    def setPlane(self, plane):
        self.setCurrentText(plane)
    
    def setCurrentText(self, text):
        self._previousPlane = self.plane()
        super().setCurrentText(text)
    
    def plane(self):
        return self.currentText()

    def depthAxes(self):
        plane = self.plane()
        for axes in 'xyz':
            if axes not in plane:
                return axes


class TimestampItem(LabelItem):
    sigEditProperties = Signal(object)
    sigRemove = Signal(object)
    
    def __init__(
            self, SizeY, SizeX, viewRange, 
            secondsPerFrame=1, 
            parent=None,
            start_timedelta=None
        ):
        self._secondsPerFrame = secondsPerFrame
        self._x_pad = 3
        self._y_pad = 2
        self.xmin, self.ymin = 0, 0
        self.SizeY = SizeY
        self.SizeX = SizeX
        self._highlighted = False
        self._parent = parent
        if start_timedelta is None:
            start_timedelta = datetime.timedelta(seconds=0)
        self._start_timedelta = start_timedelta
        self.clicked = False
        super().__init__(self)
        self.updateViewRange(viewRange)
        self.createContextMenu()
    
    def setSecondsPerFrame(self, secondsPerFrame):
        self._secondsPerFrame = secondsPerFrame
    
    def getBboxViewRange(self, viewRange):
        xRange, yRange = viewRange
        x0, x1 = xRange
        y0, y1 = yRange
        if x0 < 0:
            x0 = 0
        
        if x1 > self.SizeX:
            x1 = self.SizeX
        
        if y0 < 0:
            y0 = 0
        
        if y1 > self.SizeY:
            y1 = self.SizeY
        
        return x0, y0, x1, y1
   
    def updateViewRange(self, viewRange):
        x0, y0, x1, y1 = self.getBboxViewRange(viewRange)
        
        self.xmax = x1
        self.xmin = x0
        
        self.ymax = y1
        self.ymin = y0
    
    def createContextMenu(self):
        self.contextMenu = QMenu()
        action = QAction('Edit properties...', self.contextMenu)
        action.triggered.connect(self.emitEditProperties)
        self.contextMenu.addAction(action)
        self.contextMenu.addSeparator()
        action = QAction('Remove', self.contextMenu)
        action.triggered.connect(self.emitRemove)
        self.contextMenu.addAction(action)
    
    def emitRemove(self):
        self.sigRemove.emit(self)
    
    def mousePressed(self, x, y):
        self.clicked = True
    
    def emitEditProperties(self):
        self.setHighlighted(False)
        self.sigEditProperties.emit(self.properties())
    
    def isHighlighted(self):
        return self._highlighted
        
    def setHighlighted(self, highlighted):
        if self._highlighted and highlighted:
            return
        
        if not self._highlighted and not highlighted:
            return
        
        super().setText(self.text, bold=highlighted)
        
        self._highlighted = highlighted
    
    def showContextMenu(self, x, y):
        self.contextMenu.popup(QPoint(int(x), int(y)))
    
    def setLocationProperty(self, loc: str):
        self._loc = loc    
    
    def properties(self):
        properties = {
            'color': self._color,
            'loc': self._loc,
            'font_size': int(self._font_size[:-2]),
            'start_timedelta': self._start_timedelta,
            'move_with_zoom': self._move_with_zoom,
        }
        return properties

    def draw(self, frame_i, **kwargs):
        self.setProperties(**kwargs)
        self.update(frame_i)
    
    def update(self, frame_i):
        self.setPosFromLoc()
        self.setText(frame_i)
    
    def setMoveWithZoomProperty(self, move_with_zoom):
        self._move_with_zoom = move_with_zoom
    
    def updatePosViewRangeChanged(self, viewRange):
        if self._loc == 'custom':
            textHeight = self.itemRect().height()
            textWidth = self.itemRect().width()
            x0p = self.pos().x()
            y0p = self.pos().y()
            xcp = x0p + textWidth/2
            ycp = y0p + textHeight/2
            x0 = self.xmin
            y0 = self.ymin
            x_range = self.xmax - x0
            y_range = self.ymax - y0
            Dx_perc = (xcp - x0)/x_range
            Dy_perc = (ycp - y0)/y_range         
            
            self.updateViewRange(viewRange)
            
            X0 = self.xmin
            Y0 = self.ymin
            
            X_range = self.xmax - X0
            Y_range = self.ymax - Y0
            
            Xcp = X0 + (Dx_perc*X_range)
            Ycp = Y0 + (Dy_perc*Y_range)
            X0p = Xcp - (textWidth/2)
            Y0p = Ycp - (textHeight/2)
            
            y_pos_max = self.ymax - textHeight - self._y_pad
            if Y0p > y_pos_max:
                Y0p = y_pos_max
            
            x_pos_max = self.xmax - textWidth - self._x_pad
            if X0p > x_pos_max:
                X0p = x_pos_max
                
            self.setPos(X0p, Y0p)
        else:
            self.updateViewRange(viewRange)
            self.setPosFromLoc()
            
    
    def setPosFromLoc(self):
        textHeight = self.itemRect().height()
        textWidth = self.itemRect().width()
        if self._loc == 'custom':
            return
        
        if self._loc.find('top') != -1:
            y0 = self._y_pad + self.ymin
        else:
            y0 = self.ymax - textHeight - self._y_pad
        
        if self._loc.find('left') != -1:
            x0 = self._x_pad + self.xmin
        else:
            x0 = self.xmax - textWidth - self._x_pad

        self.setPos(x0, y0)
    
    def setProperties(
            self, 
            color=(255, 255, 255), 
            font_size='13px', 
            loc='top-left',
            start_timedelta=None,
            move_with_zoom=False
        ):
        if start_timedelta is not None:
            self._start_timedelta = start_timedelta
        self._color = color
        self._loc = loc
        self._font_size = font_size
        self._move_with_zoom = move_with_zoom

    def move(self, xm, ym):
        Dy = ym - self.yc
        Dx = xm - self.xc
        x0 = self.x0c + Dx
        y0 = self.y0c + Dy
        self.setPos(x0, y0)
    
    def mousePressed(self, x, y):
        self.clicked = True
        self.xc, self.yc = x, y
        self.x0c = self.pos().x()
        self.y0c = self.pos().y()
    
    def setText(self, frame_i):
        if not isinstance(frame_i, int):
            return
        
        seconds = frame_i*self._secondsPerFrame
        timedelta = datetime.timedelta(seconds=round(seconds))
            
        diff_seconds = (
            timedelta.total_seconds() 
            + self._start_timedelta.total_seconds()
        )
        if diff_seconds >= 0:
            timedelta = datetime.timedelta(seconds=round(diff_seconds))
            text = str(timedelta)
        else:
            abs_diff = abs(
                timedelta.total_seconds() 
                + self._start_timedelta.total_seconds()
            )
            abs_timedelta = datetime.timedelta(seconds=round(abs_diff))
            text = f'-{abs_timedelta}'
        
        # printl(timedelta)
        super().setText(
            text, color=self._color, size=self._font_size
        )
    
    def addToAxis(self, ax):
        ax.addItem(self)
    
    def removeFromAxis(self, ax):
        ax.removeItem(self)

class FontSizeWidget(QWidget):
    sigTextChanged = Signal(str)
    
    def __init__(self, parent=None, unit='px', initalVal=12):
        super().__init__(parent)
        
        layout = QHBoxLayout()
        
        self.spinbox = widgets_numeric_inputs.SpinBox()    
        self.spinbox.setValue(initalVal)    
        layout.addWidget(self.spinbox)
        
        self.unitLabel = QLabel(unit)
        layout.addWidget(self.unitLabel)
        
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)
        
        self.setLayout(layout)
        
        self.spinbox.valueChanged.connect(self.emitTextChanged)
    
    def emitTextChanged(self, value):
        self.sigTextChanged.emit(self.text())
    
    def setValue(self, value):
        if isinstance(value, str):
            value = int(value.replace(self.unitLabel.text(), '').strip())
        self.spinbox.setValue(value)
    
    def setText(self, text):
        value = int(text.replace(self.unitLabel.text(), '').strip())
        self.setValue(value)
    
    def text(self):
        return f'{self.spinbox.value()}{self.unitLabel.text()}'
    
    def value(self):
        return self.spinbox.value()


class TimeWidget(QGroupBox):
    sigValueChanged = Signal(object)
    
    def __init__(self, parent=None, orientation='vertical'):
        super().__init__(parent)
        
        mainLayout = QHBoxLayout()
        
        if orientation == 'vertical':
            spinboxesLayout = QVBoxLayout()
        elif orientation == 'horizontal':
            spinboxesLayout = QHBoxLayout()
        else:
            raise ValueError('orientation must be "vertical" or "horizontal"')
                
        self.signCombobox = QComboBox()
        self.signCombobox.addItems(('+', '-'))
        self.signCombobox.currentTextChanged.connect(self.emitValueChanged)
        
        mainLayout.addWidget(self.signCombobox)
        
        self.spinboxesMapper = {}
        units = ('days', 'hours', 'minutes', 'seconds')
        for unit in units:
            layout = QHBoxLayout()
            spinbox = widgets_numeric_inputs.SpinBox()
            spinbox.setMinimum(0)
            label = QLabel(unit)
            layout.addWidget(spinbox)
            layout.addWidget(label)
            spinbox.valueChanged.connect(self.emitValueChanged)
            self.spinboxesMapper[unit] = spinbox
            spinboxesLayout.addLayout(layout)
        
        mainLayout.addLayout(spinboxesLayout)
        
        self.setLayout(mainLayout)
        mainLayout.setContentsMargins(5, 5, 5, 5)
    
    def values(self):
        values = {}
        for unit, spinbox in self.spinboxesMapper.items():
            values[unit] = spinbox.value()
        
        signText = self.signCombobox.currentText()
        return values, sign_int_mapper[signText]
    
    def setValuesFromTimedelta(self, timedelta):
        total_seconds = timedelta.total_seconds()
        sign = 1 if total_seconds > 0 else -1
        days = timedelta.days
        hours, remainder = divmod(timedelta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        values = {
            'days': days, 
            'hours': hours, 
            'minutes': minutes, 
            'seconds': seconds
        }
        
        self.setValues(values, sign=sign)
    
    def timedelta(self):
        values, sign = self.values()
        return datetime.timedelta(**values)*sign
    
    def setValues(self, values: dict[str, int | float], sign=1):
        signText = '+' if sign > 0 else '-'
        self.signCombobox.setCurrentText(signText)
        for unit, value in values.items():
            spinbox = self.spinboxesMapper[unit]
            spinbox.setValue(value)
    
    def emitValueChanged(self, value):
        self.sigValueChanged.emit(self.values())
