"""Image display tools shared by the main GUI, the GUI mixins, data prep,
data structure, the utilities and the 3D volume renderer (from apps.py).

The imageViewer slideshow, the z-crop tool QCropZtool, the points-layer
appearance dialog and group box (also used by
volume_renderer/canvas.py), OverlayLabelsAppearanceDialog and
pgTestWindow (both unused). Sets pyqtgraph's row-major image axis order
at import, as apps.py did.
"""
import re
from typing import Literal
import matplotlib
import numpy as np
import pyqtgraph as pg
from qtpy.QtGui import QIcon, QFont, QCursor, QFont
from qtpy.QtCore import Qt, QSize, Signal
from qtpy.QtWidgets import (
    QMainWindow, QMenu, QLabel, QToolBar, QScrollBar, QWidget, QVBoxLayout,
    QHBoxLayout, QCheckBox, QGridLayout, QGroupBox, QAction, QActionGroup
)
from cellacdc import _window_stacking
from cellacdc import html_utils
from cellacdc.myutils import app_info, filesystem
from ..widgets import (
    basic_widgets as widgets_basic_widgets,
    combo_boxes as widgets_combo_boxes,
    item_views as widgets_item_views,
    line_edits as widgets_line_edits,
    lut_items as widgets_lut_items,
    message_box as widgets_message_box,
    numeric_inputs as widgets_numeric_inputs,
    plot_items as widgets_plot_items,
    push_buttons as widgets_push_buttons,
    toolbars as widgets_toolbars,
)
from cellacdc import fonts
from . import base as dialogs_base


pg.setConfigOption('imageAxisOrder', 'row-major')


class _PointsLayerAppearanceGroupbox(QGroupBox):
    sigValueChanged = Signal(object)
    
    def __init__(
            self, 
            *args, 
            backend: Literal['pyqtgraph', 'vispy']='pyqtgraph',
            is_3d=False,
            add_opacity_slider=False
        ):
        super().__init__(*args)

        self._backend = backend
        self._add_opacity_slider = add_opacity_slider
        self._is_3d = is_3d
        
        self.setTitle('Points appearance')

        layout = widgets_line_edits.FormLayout()

        '----------------------------------------------------------------------' 
        row = 0
        symbolInfoTxt = ("""
            <b>Symbol</b> used to draw the points.
        """)
        symbolInfoTxt = (f'{html_utils.paragraph(symbolInfoTxt)}')
        if backend == 'pyqtgraph':
            symbolCombobox = widgets_combo_boxes.pgScatterSymbolsCombobox()
        elif backend == 'vispy':
            symbolCombobox = widgets_combo_boxes.VisPyMarkersSymbolsCombobox()
            
        self.symbolWidget = widgets_line_edits.formWidget(
            symbolCombobox, addInfoButton=True,
            labelTextLeft='Symbol: ', parent=self, infoTxt=symbolInfoTxt,
            stretchWidget=False
        )
        layout.addFormWidget(self.symbolWidget, row=row)
        symbolCombobox.currentTextChanged.connect(self.emitSigValueChanged)
        '----------------------------------------------------------------------' 

        '----------------------------------------------------------------------' 
        row += 1
        self.colorButton = widgets_push_buttons.myColorButton(color=(255, 0, 0))
        self.colorWidget = widgets_line_edits.formWidget(
            self.colorButton, stretchWidget=True,
            labelTextLeft='Colour: ', parent=self
        )
        layout.addFormWidget(self.colorWidget, align=Qt.AlignLeft, row=row)
        self.colorButton.clicked.disconnect()
        self.colorButton.clicked.connect(self.selectColor)
        self.colorButton.sigColorChanging.connect(self.emitSigValueChanged)
        '----------------------------------------------------------------------' 

        '----------------------------------------------------------------------' 
        row += 1
        self.sizeSpinBox = widgets_numeric_inputs.SpinBox()
        self.sizeSpinBox.setValue(5)
        self.sizeWidget = widgets_line_edits.formWidget(
            self.sizeSpinBox, stretchWidget=True,
            labelTextLeft='Size: ', parent=self
        )
        layout.addFormWidget(self.sizeWidget, row=row)
        self.sizeSpinBox.valueChanged.connect(self.emitSigValueChanged)
        '----------------------------------------------------------------------' 
        
        '----------------------------------------------------------------------' 
        self.zHeightSpinBox = widgets_basic_widgets.DummyWidget()
        if not is_3d:
            row += 1
            self.addZheightSpinbox(row)
            layout.addFormWidget(self.zHeightWidget, row=row)
        '----------------------------------------------------------------------'

        '----------------------------------------------------------------------' 
        row += 1
        shortcutInfoTxt = ("""
        <b>Shortcut</b> that you can use to <b>hide/show</b> points.
        """)
        shortcutInfoTxt = (f'{html_utils.paragraph(shortcutInfoTxt)}')
        self.shortcutWidget = widgets_line_edits.formWidget(
            widgets_line_edits.ShortcutLineEdit(), addInfoButton=True,
            labelTextLeft='Shortcut: ', parent=self, infoTxt=shortcutInfoTxt
        )
        layout.addFormWidget(self.shortcutWidget, row=row)
        '----------------------------------------------------------------------'
        
        '----------------------------------------------------------------------' 
        self.opacitySlider = widgets_basic_widgets.DummyWidget()
        if add_opacity_slider:
            row += 1
            self.opacitySlider = widgets_numeric_inputs.sliderWithSpinBox(
                title_loc='in_line', 
                isFloat=True, 
                parent=self,
                normalize_factor=20
            )
            self.opacitySlider.setValue(0.3)
            self.opacitySlider.setRange(0.0, 1.0)
            self.opacitySlider.setSingleStep(0.05)
            self.opacitySlider.setDecimals(2)
            self.opacityWidget = widgets_line_edits.formWidget(
                self.opacitySlider, stretchWidget=True,
                labelTextLeft='Opacity: ', parent=self
            )
            layout.addFormWidget(self.opacityWidget, row=row)
            self.opacitySlider.valueChanged.connect(self.emitSigValueChanged)
        '----------------------------------------------------------------------' 

        self.setLayout(layout)
    
    def emitSigValueChanged(self, *args):
        self.sigValueChanged.emit(self.state())
    
    def addZheightSpinbox(self, row: int):
        zHeightTooltip = (
            'If "Z-depth" is greater than 1, the points will be annotated '
            'in all the z-slices in the range '
            '`z - (Z-depth/2) < z < z + (Z-depth/2)`\n'
            'where `z` is the center z-slice of the added point.'
        )
        self.zHeightSpinBox = widgets_numeric_inputs.OddSpinBox()
        self.zHeightSpinBox.setValue(1)
        self.zHeightSpinBox.setMinimum(1)
        self.zHeightWidget = widgets_line_edits.formWidget(
            self.zHeightSpinBox, stretchWidget=True,
            labelTextLeft='Z-depth: ', parent=self,
            toolTip=zHeightTooltip
        )
        self.zHeightSpinBox.valueChanged.connect(self.emitSigValueChanged)
    
    def restoreState(self, state: dict):
        key_widget_setter_mapper = {
            'shortcut': self.shortcutWidget.widget.setText,
            'color': self.colorButton.setColor,
            'symbol': self.symbolWidget.widget.setCurrentText,
            'pointSize': self.sizeSpinBox.setValue,
            'zHeight': self.zHeightSpinBox.setValue,
            'opacity': self.opacitySlider.setValue
        }
        for key, setter in key_widget_setter_mapper.items():
            value = state.get(key)
            if value is None:
                continue
            
            setter(value)
    
    def selectColor(self):
        color = self.colorButton.color()
        self.colorButton.origColor = color
        self.colorButton.colorDialog.setCurrentColor(color)
        _window_stacking.set_window_flags_on_top(
            self.colorButton.colorDialog, Qt.Window, parent=self
        )
        self.colorButton.colorDialog.open()    
        w = self.width()
        left = self.pos().x()
        colorDialogTop = self.colorButton.colorDialog.pos().y()
        self.colorButton.colorDialog.move(w+left+10, colorDialogTop)
    
    def state(self):
        r,g,b,a = self.colorButton.color().getRgb()
        _state = {
            'symbol': self.symbolWidget.widget.currentText(), 
            'color': (r,g,b),
            'pointSize': self.sizeSpinBox.value(),
            'shortcut': self.shortcutWidget.widget.text()
        }
        if not self._is_3d:
            _state['zHeight'] = self.zHeightSpinBox.value()
        
        if self._add_opacity_slider:
            _state['opacity'] = self.opacitySlider.value()
        
        return _state


class EditPointsLayerAppearanceDialog(dialogs_base.QBaseDialog):
    sigClosed = Signal()
    sigValueChanged = Signal(object)

    def __init__(
            self, 
            backend: Literal['pyqtgraph', 'vispy']='pyqtgraph', 
            is_3d=False,
            add_opacity_slider=False,
            hide_on_close=False,
            title='Points Layer Properties',
            parent=None
        ):
        self.cancel = True
        super().__init__(parent)

        self._parent = parent
        self._backend = backend
        self._is_3d = is_3d
        self._add_opacity_slider = add_opacity_slider
        self._hide_on_close = hide_on_close

        self.setWindowTitle(title)
        _window_stacking.set_window_flags_on_top(self, Qt.Window)

        mainLayout = QVBoxLayout()

        self.appearanceGroupbox = _PointsLayerAppearanceGroupbox(
            backend=backend, 
            is_3d=is_3d, 
            add_opacity_slider=add_opacity_slider
        )
        
        self.appearanceGroupbox.sigValueChanged.connect(self.emitValueChanged)

        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addWidget(self.appearanceGroupbox)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        self.setFont(fonts.font)
    
    def emitValueChanged(self, *args):
        self.sigValueChanged.emit(self.appearanceGroupbox.state())
    
    def restoreState(self, state):
        self.appearanceGroupbox.restoreState(state)
    
    def closeEvent(self, event):
        if self._hide_on_close:
            event.ignore()
            self.hide()
        else:
            super().closeEvent(event)
        self.sigClosed.emit()
    
    def force_close(self):
        self._hide_on_close = False
        self.close()
    
    def state(self):
        _state = self.appearanceGroupbox.state()
        return _state
    
    def ok_cb(self):
        self.cancel = False
        symbol = self.appearanceGroupbox.symbolWidget.widget.currentText()
        self.symbol = symbol
        if self._backend == 'pyqtgraph':
            self.symbol = re.findall(r"\'(.+)\'", symbol)[0]
        self.color = self.appearanceGroupbox.colorButton.color()
        self.pointSize = self.appearanceGroupbox.sizeSpinBox.value()
        
        self.zHeight = None
        if not self._is_3d:
            self.zHeight = self.appearanceGroupbox.zHeightSpinBox.value()
        
        self.opacity = 1.0
        if self._add_opacity_slider:
            self.opacity = self.appearanceGroupbox.opacitySlider.value()
        
        shortcutWidget = self.appearanceGroupbox.shortcutWidget
        self.shortcut = shortcutWidget.widget.text()
        self.keySequence = shortcutWidget.widget.keySequence
        if self._hide_on_close:
            self.hide()
        else:
            self.close()


class QCropZtool(dialogs_base.QBaseDialog):
    sigClose = Signal()
    sigZvalueChanged = Signal(str, int)
    sigReset = Signal()
    sigCrop = Signal(int, int)

    def __init__(
            self, SizeZ, cropButtonText='Apply crop', parent=None, 
            addDoNotShowAgain=False, title='Select z-slices'
        ):
        super().__init__(parent)

        self.cancel = True

        _window_stacking.set_window_flags_on_top(self, Qt.Tool)

        self.SizeZ = SizeZ
        self.numDigits = len(str(self.SizeZ))

        self.setWindowTitle(title)

        layout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        self.lowerZscrollbar = widgets_numeric_inputs.ScrollBarWithNumericControl()
        self.lowerZscrollbar.setMaximum(SizeZ)
        self.lowerZscrollbar.setMinimum(1)
        self.lowerZscrollbar.setValue(1)

        self.upperZscrollbar = widgets_numeric_inputs.ScrollBarWithNumericControl()
        self.upperZscrollbar.setMaximum(SizeZ)
        self.upperZscrollbar.setValue(SizeZ)

        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')
        cropButton = widgets_push_buttons.okPushButton(cropButtonText)
        buttonsLayout.addWidget(cropButton)
        buttonsLayout.addWidget(cancelButton)

        row = 0
        layout.addWidget(
            QLabel('Lower z-slice '), row, 0, alignment=Qt.AlignRight
        )
        layout.addWidget(self.lowerZscrollbar, row, 1)

        row += 1
        layout.setRowStretch(row, 5)

        row += 1
        layout.addWidget(
            QLabel('Upper z-slice '), row, 0, alignment=Qt.AlignRight
        )
        layout.addWidget(self.upperZscrollbar, row, 1)

        row += 1
        if addDoNotShowAgain:
            self.doNotShowAgainCheckbox = QCheckBox('Do not ask again')
            layout.addWidget(
                self.doNotShowAgainCheckbox, row, 1, alignment=Qt.AlignLeft
            )
            row += 1

        layout.addLayout(buttonsLayout, row, 1, alignment=Qt.AlignRight)

        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 10)

        self.setLayout(layout)

        # resetButton.clicked.connect(self.emitReset)
        cropButton.clicked.connect(self.emitCrop)
        cancelButton.clicked.connect(self.close)
        self.lowerZscrollbar.sigValueChanged.connect(self.ZvalueChanged)
        self.upperZscrollbar.sigValueChanged.connect(self.ZvalueChanged)

    def emitReset(self):
        self.sigReset.emit()

    def emitCrop(self):
        self.cancel = False
        low_z = self.lowerZscrollbar.value() - 1
        high_z = self.upperZscrollbar.value() - 1
        self.sigCrop.emit(low_z, high_z)
        self.close()

    def updateScrollbars(self, lower_z, upper_z):
        self.lowerZscrollbar.setValue(lower_z+1)
        self.upperZscrollbar.setValue(upper_z+1)

    def ZvalueChanged(self, value):
        which = 'lower' if self.sender() == self.lowerZscrollbar else 'upper'
        if which == 'lower' and value > self.upperZscrollbar.value()-1:
            self.lowerZscrollbar.setValue(self.upperZscrollbar.value()-1)
            return
        if which == 'upper' and value < self.lowerZscrollbar.value()+1:
            self.upperZscrollbar.setValue(self.lowerZscrollbar.value()+1)
            return

        z_slice_n = value - 1
        self.sigZvalueChanged.emit(which, z_slice_n)

    def showEvent(self, event):
        self.resize(int(self.width()*1.5), self.height())

    def closeEvent(self, event):
        super().closeEvent(event)
        self.sigClose.emit()


class imageViewer(QMainWindow):
    """Main Window."""
    sigClosed = Signal()
    sigHoveringImage = Signal(object, object)

    def __init__(
            self, parent=None, posData=None, button_toUncheck=None,
            spinBox=None, linkWindow=None, enableOverlay=False,
            isSigleFrame=False, enableMirroredCursor=False
        ):
        self.button_toUncheck = button_toUncheck
        self.parent = parent
        self.posData = posData
        self.spinBox = spinBox
        self.linkWindow = linkWindow
        self.enableMirroredCursor = enableMirroredCursor
        self.isSigleFrame = isSigleFrame
        self.minMaxValuesMapper = None
        """Initializer."""
        super().__init__(parent)

        if posData is None:
            posData = self.parent.data[self.parent.pos_i]
        self.posData = posData
        self.enableOverlay = enableOverlay

        self.gui_createActions()
        self.gui_createMenuBar()
        self.gui_createToolBars()

        self.gui_createStatusBar()

        self.gui_createGraphics()

        self.gui_connectImgActions()

        self.gui_createImgWidgets()
        self.gui_connectActions()

        self.gui_setSingleFrameMode(self.isSigleFrame)
        
        self.setupMirroredCursor()

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.graphLayout, 0, 0, 1, 1)
        mainLayout.addLayout(self.img_Widglayout, 1, 0)

        mainContainer.setLayout(mainLayout)

        self.frame_i = posData.frame_i
        self.num_frames = posData.SizeT
        
        version = app_info.read_version()
        self.setWindowTitle(f"Cell-ACDC v{version} - {posData.relPath}")

    def gui_createActions(self):
        # File actions
        self.exitAction = QAction("&Exit", self)

        # Toolbar actions
        self.prevAction = QAction("Previous frame", self)
        self.nextAction = QAction("Next Frame", self)
        self.jumpForwardAction = QAction("Jump to 10 frames ahead", self)
        self.jumpBackwardAction = QAction("Jump to 10 frames back", self)
        self.prevAction.setShortcut("left")
        self.nextAction.setShortcut("right")
        self.jumpForwardAction.setShortcut("up")
        self.jumpBackwardAction.setShortcut("down")
        self.addAction(self.nextAction)
        self.addAction(self.prevAction)
        self.addAction(self.jumpBackwardAction)
        self.addAction(self.jumpForwardAction)
        if self.enableOverlay:
            self.overlayButton = widgets_toolbars.rightClickToolButton(parent=self)
            self.overlayButton.setIcon(QIcon(":overlay.svg"))
            self.overlayButton.setCheckable(True)

    def gui_createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)
        # fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.exitAction)

    def gui_createToolBars(self):
        toolbarSize = 30

        editToolBar = QToolBar("Edit", self)
        editToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        self.addToolBar(editToolBar)

        self.editToolBar = editToolBar

        if self.enableOverlay:
            editToolBar.addWidget(self.overlayButton)

        if self.linkWindow:
            # Insert a spacing
            editToolBar.addWidget(QLabel('  '))
            self.linkWindowCheckbox = QCheckBox("Link to main GUI")
            self.linkWindowCheckbox.setChecked(True)
            editToolBar.addWidget(self.linkWindowCheckbox)
        
        if self.enableMirroredCursor:
            self.showMirroredCursorCheckbox = QCheckBox(
                'Show mirrored cursor from main window'
            )
            self.showMirroredCursorCheckbox.setChecked(True)
            editToolBar.addWidget(self.showMirroredCursorCheckbox)

    def setupMirroredCursor(self):
        self.cursor = pg.ScatterPlotItem(
            symbol='+', pxMode=True, pen=pg.mkPen('k', width=1),
            brush=pg.mkBrush('w'), size=16, tip=None
        )
        self.Plot.addItem(self.cursor)
    
    def gui_connectActions(self):
        self.exitAction.triggered.connect(self.close)
        self.prevAction.triggered.connect(self.prev_frame)
        self.nextAction.triggered.connect(self.next_frame)
        self.jumpForwardAction.triggered.connect(self.skip10ahead_frames)
        self.jumpBackwardAction.triggered.connect(self.skip10back_frames)
        if self.enableOverlay:
            self.overlayButton.toggled.connect(self.overlay_cb)
            self.overlayButton.sigRightClick.connect(self.showOverlayContextMenu)
    
    def gui_setSingleFrameMode(self, isSingleFrame: bool):
        if not isSingleFrame:
            return

        self.framesScrollBar.setDisabled(True)
        self.framesScrollBar.setVisible(False)
        self.frameLabel.hide()
        self.t_label.hide()
        self.prevAction.triggered.disconnect()
        self.nextAction.triggered.disconnect()
        self.jumpForwardAction.triggered.disconnect()
        self.jumpBackwardAction.triggered.disconnect()
        self.editToolBar.setVisible(False)

    def showOverlayContextMenu(self, event):
        if not self.overlayButton.isChecked():
            return

        if self.parent is not None:
            self.overlayContextMenu.exec_(QCursor.pos())

    def gui_createStatusBar(self):
        self.statusbar = self.statusBar()
        # Temporary message
        self.statusbar.showMessage("Ready", 3000)
        # Permanent widget
        self.wcLabel = QLabel(f"")
        self.statusbar.addPermanentWidget(self.wcLabel)

    def gui_createGraphics(self):
        self.graphLayout = pg.GraphicsLayoutWidget()

        # Plot Item container for image
        self.Plot = pg.PlotItem()
        self.Plot.invertY(True)
        self.Plot.setAspectLocked(True)
        self.Plot.hideAxis('bottom')
        self.Plot.hideAxis('left')
        self.graphLayout.addItem(self.Plot, row=1, col=1)

        # Image Item
        self.img = widgets_plot_items.BaseImageItem()
        self.img.setEnableAutoLevels(True)
        self.Plot.addItem(self.img)

        #Image histogram
        self.imgGrad = widgets_lut_items.myHistogramLUTitem(isViewer=True)
        self.imgGrad.gradient.showMenu = self.showLutItemOverlayContextMenu 
        self.imgGrad.vb.raiseContextMenu = lambda x: None
        self.imgGrad.setImageItem(self.img)
        self.graphLayout.addItem(self.imgGrad, row=1, col=0)

        # Current frame text
        self.frameLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        self.frameLabel.setText(' ')
        self.graphLayout.addItem(self.frameLabel, row=2, col=0, colspan=2)

        if not self.enableOverlay:
            return
    
    def gui_createOverlayItems(self):
        self.createOverlayChannelsActions()
        self.overlayLayersItems = {}
        for ch in self.posData.chNames:
            if ch == self.parent.user_ch_name:
                continue
            overlayItems = self.getOverlayItems(ch) 
            imageItem, lutItem, alphaScrollbar = overlayItems
            lutItem.vb.raiseContextMenu = lambda x: None
            lutItem.gradient.showMenu = self.showLutItemOverlayContextMenu    
            lutItem.overlayColorButton.sigColorChanging.connect(
                self.updateOlColors
            )
            self.addAlphaScrollbar(ch, imageItem, alphaScrollbar)
            self.overlayLayersItems[ch] = overlayItems
            self.Plot.addItem(imageItem)
    
    def createOverlayChannelsActions(self):
        self.overlayLutItemAdditionalActions = []
        separator = QAction(self)
        separator.setSeparator(True)
        self.overlayLutItemAdditionalActions.append(separator)
        section = self.imgGrad.gradient.menu.addSection(
            'Select channel to adjust: '
        )
        self.overlayLutItemAdditionalActions.append(section)
        self.imgGrad.gradient.menu.removeAction(section)
        
        self.overlayChNamesActionGroup = QActionGroup(self)
        self.overlayChNamesActionGroup.setExclusive(True)
        for chName in self.posData.chNames:
            action = QAction(chName, self)
            action.setCheckable(True)
            if chName == self.parent.user_ch_name:
                action.setChecked(True)
            self.overlayChNamesActionGroup.addAction(action)
        self.overlayChNamesActionGroup.triggered.connect(
            self.chNameGradientActionClicked
        )
    
    def chNameGradientActionClicked(self, action):
        # Action triggered from lutItem
        self.checkedOverlayChName = action.text()
        if action.text() == self.posData.user_ch_name:
            self.setOverlayItemsVisible('', False)
        else:
            self.setOverlayItemsVisible(action.text(), True)

    def showLutItemOverlayContextMenu(self, event):
        lutItem = self.currentLutItem
        
        for action in self.overlayLutItemAdditionalActions:
            try:
                lutItem.gradient.menu.removeAction(action)
            except Exception as e:
                pass

        for action in self.overlayChNamesActionGroup.actions():
            try:
                lutItem.gradient.menu.removeAction(action)
            except Exception as e:
                pass
        
        if self.overlayButton.isChecked():        
            for action in self.overlayLutItemAdditionalActions:
                lutItem.gradient.menu.addAction(action)
            
            for action in self.overlayChNamesActionGroup.actions():
                if action.text() == self.posData.user_ch_name:
                    lutItem.gradient.menu.addAction(action)
                    continue
                for filename in self.posData.ol_data:
                    if filename.endswith(action.text()):
                        lutItem.gradient.menu.addAction(action)
                        break
                    if filename.endswith(f'{action.text()}_aligned'):
                        lutItem.gradient.menu.addAction(action)
                        break

        try:
            # Convert QPointF to QPoint
            lutItem.gradient.menu.popup(event.screenPos().toPoint())
        except AttributeError:
            lutItem.gradient.menu.popup(event.screenPos())
        

    def gui_connectImgActions(self):
        self.img.hoverEvent = self.gui_hoverEventImg

    def gui_createImgWidgets(self):
        if self.posData is None:
            posData = self.parent.data[self.parent.pos_i]
        else:
            posData = self.posData
        self.img_Widglayout = QGridLayout()

        # Frames scrollbar
        self.framesScrollBar = QScrollBar(Qt.Horizontal)
        # self.framesScrollBar.setFixedHeight(20)
        self.framesScrollBar.setMinimum(1)
        self.framesScrollBar.setMaximum(posData.SizeT)
        t_label = QLabel('frame  ')
        _font = QFont()
        _font.setPixelSize(12)
        t_label.setFont(_font)
        self.img_Widglayout.addWidget(
                t_label, 0, 0, alignment=Qt.AlignRight)
        self.img_Widglayout.addWidget(
                self.framesScrollBar, 0, 1, 1, 20)
        self.t_label = t_label
        self.framesScrollBar.valueChanged.connect(self.framesScrollBarMoved)

        # z-slice scrollbar
        self.zSliceScrollBar = QScrollBar(Qt.Horizontal)
        # self.zSliceScrollBar.setFixedHeight(20)
        self.zSliceScrollBar.setMaximum(self.posData.SizeZ-1)
        _z_label = QLabel('z-slice  ')
        _font = QFont()
        _font.setPixelSize(12)
        _z_label.setFont(_font)
        self.z_label = _z_label
        self.img_Widglayout.addWidget(_z_label, 1, 0, alignment=Qt.AlignCenter)
        self.img_Widglayout.addWidget(self.zSliceScrollBar, 1, 1, 1, 20)

        if self.posData.SizeZ == 1:
            self.zSliceScrollBar.setDisabled(True)
            self.zSliceScrollBar.setVisible(False)
            _z_label.setVisible(False)
        
        self.img_Widglayout.setContentsMargins(100, 0, 50, 0)
        self.zSliceScrollBar.valueChanged.connect(self.update_z_slice)

        if self.enableOverlay:
            self.setOverlayColors()
            self.gui_createOverlayItems()
            self.createOverlayContextMenu()

            self.img.alphaScrollbar = self.addAlphaScrollbar(
                self.parent.user_ch_name, self.img
            )
    
    def getOverlayItems(self, channelName):
        imageItem = pg.ImageItem()
        imageItem.setOpacity(0.5)

        lutItem = widgets_lut_items.myHistogramLUTitem(isViewer=True)
        
        lutItem.setImageItem(imageItem)
        lutItem.vb.raiseContextMenu = lambda x: None
        initColor = self.overlayRGBs.pop(0)
        self.parent.initColormapOverlayLayerItem(initColor, lutItem)
        lutItem.addOverlayColorButton(initColor, channelName)
        lutItem.initColor = initColor
        lutItem.hide()

        alphaScrollBar = self.addAlphaScrollbar(channelName, imageItem)
        return imageItem, lutItem, alphaScrollBar
    
    def setMirroredCursorPos(self, x, y):
        if not self.enableMirroredCursor:
            return
        
        if not self.showMirroredCursorCheckbox.isChecked():
            return

        self.cursor.setData([x], [y])
        
    def setOverlayColors(self):
        self.overlayRGBs = [
            (255, 255, 0),
            (252, 72, 254),
            (49, 222, 134),
            (22, 108, 27)
        ]
        cmap = matplotlib.colormaps['gist_rainbow']
        self.overlayRGBs.extend(
            [tuple([round(c*255) for c in cmap(i)][:3]) 
            for i in np.linspace(0,1,8)]
        )

    def setOpacityOverlayLayersItems(self, value, imageItem=None):
        if imageItem is None:
            imageItem = self.sender().imageItem
            alpha = value/self.sender().maximum()
        else:
            alpha = value
        imageItem.setOpacity(alpha)
    
    def overlay_cb(self, checked):
        if checked:
            if self.posData.ol_data is None:
                selectedChannels = self.askSelectOverlayChannel()
                if selectedChannels is None:
                    self.overlayButton.toggled.disconnect()
                    self.overlayButton.setChecked(False)
                    self.overlayButton.toggled.connect(self.overlay_cb)
                    return
                success = self.parent.loadOverlayData(selectedChannels)         
                if not success:
                    return False
                lastChannel = selectedChannels[-1]
                self.checkedOverlayChName = lastChannel
                imageItem = self.overlayLayersItems[lastChannel][0]
                self.setOpacityOverlayLayersItems(0.5, imageItem=imageItem)
                self.img.setOpacity(0.5)
                self.setCheckedOverlayContextMenusActions(selectedChannels)
            else:
                self.checkedOverlayChName = (
                    self.parent.imgGrad.checkedChannelname
                )
                selectedChannels = self.parent.checkedOverlayChannels
                self.setCheckedOverlayContextMenusActions(selectedChannels)
            self.setOverlayItemsVisible(self.checkedOverlayChName, True)
        else:
            self.img.setOpacity(1.0)
            self.setOverlayItemsVisible('', False)
            for items in self.overlayLayersItems.values():
                imageItem = items[0]
                imageItem.clear()
        self.update_img()
    
    def createOverlayContextMenu(self):
        ch_names = [
            ch for ch in self.posData.chNames 
            if ch != self.posData.user_ch_name
        ]
        self.overlayContextMenu = QMenu()
        self.overlayContextMenu.addSeparator()
        self.checkedOverlayChannels = set()
        for chName in ch_names:
            action = QAction(chName, self.overlayContextMenu)
            action.setCheckable(True)
            action.toggled.connect(self.overlayChannelToggled)
            self.overlayContextMenu.addAction(action)
    
    def setCheckedOverlayContextMenusActions(self, channelNames):
        for action in self.overlayContextMenu.actions():
            if action.text() not in channelNames:
                continue
            action.setChecked(True)
            self.checkedOverlayChannels.add(action.text())

    def overlayChannelToggled(self, checked):
        # Action toggled from overlayButton context menu
        channelName = self.sender().text()
        if checked:
            posData = self.posData
            if channelName not in posData.loadedFluoChannels:
                self.parent.loadOverlayData([channelName], addToExisting=True)
            self.setOverlayItemsVisible(channelName, True)
            self.checkedOverlayChannels.add(channelName)    
            self.updateOlColors(None)
        else:
            self.checkedOverlayChannels.remove(channelName)
            imageItem = self.overlayLayersItems[channelName][0]
            imageItem.clear()
            try:
                channelToShow = next(iter(self.checkedOverlayChannels))
                self.setOverlayItemsVisible(channelToShow, True)
            except StopIteration:
                self.setOverlayItemsVisible('', False)
        self.update_img()
    
    def updateOlColors(self, button):
        lutItem = self.overlayLayersItems[self.checkedOverlayChName][1]
        rgb = lutItem.overlayColorButton.color().getRgb()[:3]
        self.parent.initColormapOverlayLayerItem(rgb, lutItem)
        lutItem.overlayColorButton.setColor(rgb)
    
    def addAlphaScrollbar(self, channelName, imageItem, alphaScrollBar=None):
        if alphaScrollBar is None:
            alphaScrollBar = QScrollBar(Qt.Horizontal)
        label = QLabel(f'Alpha {channelName}')
        label.setFont(fonts.font)
        label.hide()
        alphaScrollBar.imageItem = imageItem
        alphaScrollBar.label = label
        alphaScrollBar.setFixedHeight(self.parent.h)
        alphaScrollBar.hide()
        alphaScrollBar.setMinimum(0)
        alphaScrollBar.setMaximum(40)
        alphaScrollBar.setValue(20)
        alphaScrollBar.setToolTip(
            f'Control the alpha value of the overlaid channel {channelName}.\n'
            'alpha=0 results in NO overlay,\n'
            'alpha=1 results in only fluorescence data visible'
        )
        self.img_Widglayout.addWidget(
            alphaScrollBar.label, 2, 0, alignment=Qt.AlignRight
        )
        self.img_Widglayout.addWidget(alphaScrollBar, 2, 1, 1, 20)
        sp = alphaScrollBar.label.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        alphaScrollBar.label.setSizePolicy(sp)
        
        sp = alphaScrollBar.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        alphaScrollBar.setSizePolicy(sp)

        alphaScrollBar.valueChanged.connect(self.setOpacityOverlayLayersItems)
        return alphaScrollBar
    
    def setOverlayItemsVisible(self, channelName, visible):
        if visible:
            self.imgGrad.hide()
            self.img.alphaScrollbar.hide()
            self.img.alphaScrollbar.label.hide()
            try:
                self.graphLayout.removeItem(self.imgGrad)
            except Exception as e:
                pass
            itemsToShow = None
            for name, items in self.overlayLayersItems.items():
                _, lutItem, alphaSB = items
                if name == channelName:
                    itemsToShow = items
                else:
                    lutItem.hide()
                    alphaSB.hide()
                    alphaSB.label.hide()
                    try:
                        self.graphLayout.removeItem(lutItem)
                    except Exception as e:
                        pass
            
            if itemsToShow is None:
                self.graphLayout.addItem(self.imgGrad, row=1, col=0)
                self.imgGrad.show()
                self.currentLutItem = self.imgGrad
                self.img.alphaScrollbar.show()
                self.img.alphaScrollbar.label.show()
            else:
                _, lutItem, alphaSB = itemsToShow
                lutItem.show()
                alphaSB.show()
                alphaSB.label.show()
                self.currentLutItem = lutItem
                self.graphLayout.addItem(lutItem, row=1, col=0)
        else:
            if self.overlayButton.isChecked():
                self.img.alphaScrollbar.show()
                self.img.alphaScrollbar.label.show()
            else:
                self.img.alphaScrollbar.hide()
                self.img.alphaScrollbar.label.hide()
            for name, items in self.overlayLayersItems.items():
                _, lutItem, alphaSB = items
                lutItem.hide()
                alphaSB.hide()
                alphaSB.label.hide()
                try:
                    self.graphLayout.removeItem(lutItem)
                except Exception as e:
                    pass
            self.graphLayout.addItem(self.imgGrad, row=1, col=0)
            self.imgGrad.show()
            self.currentLutItem = self.imgGrad

    def framesScrollBarMoved(self, frame_n):
        self.frame_i = frame_n-1
        self.t_label.setText(
            f'frame n. {self.frame_i+1}/{self.num_frames}'
        )
        if self.spinBox is not None:
            self.spinBox.setValue(frame_n)
        self.update_img()

    def gui_hoverEventImg(self, event):
        # Update x, y, value label bottom right
        try:
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            _img = self.img.image
            Y, X = _img.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                val = _img[ydata, xdata]
                self.wcLabel.setText(f'(x={x:.2f}, y={y:.2f}, value={val:.2f})')
            else:
                self.wcLabel.setText(f'')
        except Exception as e:
            self.wcLabel.setText(f'')
        
        emitHovering = (
            self.enableMirroredCursor and 
            self.showMirroredCursorCheckbox.isChecked()
        )
        if emitHovering:
            if event.isExit():
                x, y = None, None
            else:
                x, y = event.pos()
            self.sigHoveringImage.emit(x, y)
            self.cursor.setData([], [])

    def next_frame(self):
        if self.frame_i < self.num_frames-1:
            self.frame_i += 1
        else:
            self.frame_i = 0
        self.update_img()

    def prev_frame(self):
        if self.frame_i > 0:
            self.frame_i -= 1
        else:
            self.frame_i = self.num_frames-1
        self.update_img()

    def skip10ahead_frames(self):
        if self.frame_i < self.num_frames-10:
            self.frame_i += 10
        else:
            self.frame_i = 0
        self.update_img()

    def skip10back_frames(self):
        if self.frame_i > 9:
            self.frame_i -= 10
        else:
            self.frame_i = self.num_frames-1
        self.update_img()

    def update_z_slice(self, z):
        if self.posData is None:
            posData = self.parent.data[self.parent.pos_i]
        else:
            posData = self.posData
            idx = (posData.filename, posData.frame_i)
            posData.segmInfo_df.at[idx, 'z_slice_used_gui'] = z
        
        self.z_label.setText(f'z-slice  {z+1:02}/{posData.SizeZ}')
        self.img.setCurrentZsliceIndex(z)
        self.update_img()

    def getImage(self):
        posData = self.posData
        frame_i = self.frame_i
        if posData.SizeZ > 1:
            idx = (posData.filename, frame_i)
            z = posData.segmInfo_df.at[idx, 'z_slice_used_gui']
            zProjHow = posData.segmInfo_df.at[idx, 'which_z_proj_gui']
            img = posData.img_data[frame_i]
            if zProjHow == 'single z-slice':
                self.zSliceScrollBar.setSliderPosition(z)
                self.z_label.setText(f'z-slice  {z+1:02}/{posData.SizeZ}')
                img = img[z].copy()
            elif zProjHow == 'max z-projection':
                img = img.max(axis=0).copy()
            elif zProjHow == 'mean z-projection':
                img = img.mean(axis=0).copy()
            elif zProjHow == 'median z-proj.':
                img = np.median(img, axis=0).copy()
        else:
            img = posData.img_data[frame_i].copy()
        return img

    def update_img(self):
        self.frameLabel.setText(
            f'Current frame = {self.frame_i+1}/{self.num_frames}'
        )
        if self.parent is None:
            img = self.getImage()
        else:
            img = self.parent.getImage(frame_i=self.frame_i, raw=True)
        
        self.img.setCurrentFrameIndex(self.frame_i)
        self.img.setImage(img)
        self.framesScrollBar.setSliderPosition(self.frame_i+1)
        
        if not self.enableOverlay:
            return

        if not self.overlayButton.isChecked():
            return
        
        self.setOverlayImages(frame_i=self.frame_i)
    
    def askSelectOverlayChannel(self):
        ch_names = [
            ch for ch in self.posData.chNames 
            if ch != self.posData.user_ch_name
        ]
        if not ch_names:
            msg = widgets_message_box.myMessageBox(wrapText=False)
            msg.information(
                self, 'No channels to overlay', html_utils.paragraph(
                    'There are no other channels to overlay.'
                )
            )
            return
        # Not empty: `loadOverlayData` needs at least one channel
        selectFluo = widgets_item_views.QDialogListbox(
            'Select channel',
            'Select channel names to overlay:\n',
            ch_names, multiSelection=True, parent=self,
            allowEmptySelection=False
        )
        selectFluo.exec_()
        if selectFluo.cancel:
            return

        return selectFluo.selectedItemsText
    
    def setOverlayImages(self, frame_i=None):
        posData = self.posData
        for filename in posData.ol_data:
            chName = filesystem.get_chname_from_basename(
                filename, posData.basename, remove_ext=False
            )
            if chName not in self.checkedOverlayChannels:
                continue
            
            imageItem = self.overlayLayersItems[chName][0]
            ol_img = self.parent.getOlImg(filename, frame_i=frame_i)
            imageItem.setImage(ol_img)

    def closeEvent(self, event):
        if self.button_toUncheck is not None:
            self.button_toUncheck.setChecked(False)
        self.sigClosed.emit()

    def show(self, left=None, top=None):
        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        QMainWindow.show(self)
        try:
            self.framesScrollBar.setFixedHeight(self.parent.h)
        except Exception as e:
            pass
        try:
            self.zSliceScrollBar.setFixedHeight(self.parent.h)
        except Exception as e:
            pass
        
        try:
            self.img.alphaScrollbar.setFixedHeight(self.parent.h)
        except Exception as e:
            pass
        if left is not None and top is not None:
            self.setGeometry(left, top, 850, 800)


class pgTestWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        self.graphLayout = pg.GraphicsLayoutWidget()
        self.ax1 = pg.PlotItem()
        self.ax1.setAspectLocked(True)
        self.graphLayout.addItem(self.ax1)

        layout.addWidget(self.graphLayout)

        self.setLayout(layout)


class OverlayLabelsAppearanceDialog(dialogs_base.QBaseDialog):
    sigValuesChanged = Signal(object)
    
    def __init__(self, scatterPlotItem: pg.ScatterPlotItem=None, parent=None):
        super().__init__(parent)
        
        self.cancel = True
        
        self.setWindowTitle('Overlay contours appearance properties')

        mainLayout = QVBoxLayout()
        
        formLayout = widgets_line_edits.FormLayout()
        
        row = -1
        
        row += 1
        self.colorButton = widgets_push_buttons.myColorButton(color=(255, 0, 0))
        self.colorButton.clicked.disconnect()
        self.colorButton.clicked.connect(self.selectColor)
        self.colorButton.setCursor(Qt.PointingHandCursor)
        self.colorWidget = widgets_line_edits.formWidget(
            self.colorButton, addInfoButton=False, stretchWidget=False,
            labelTextLeft='Symbol color: ', parent=self, 
            widgetAlignment='left'
        )
        if scatterPlotItem is not None:
            pen = scatterPlotItem.opts['pen']
            color = pen.color()
            self.colorButton.setColor(color)
        formLayout.addFormWidget(self.colorWidget, row=row)
        
        row += 1
        self.penWidthSpinBox = widgets_numeric_inputs.SpinBox()
        self.penWidthSpinBox.setMinimum(0)
        self.penWidthSpinBox.setValue(2)

        self.penWidthWidget = widgets_line_edits.formWidget(
            self.penWidthSpinBox, addInfoButton=False, stretchWidget=False,
            labelTextLeft='Symbol weight: ', parent=self, 
            widgetAlignment='left'
        )
        if scatterPlotItem is not None:
            pen = scatterPlotItem.opts['pen']
            width = pen.width()
            self.penWidthSpinBox.setValue(width)
        formLayout.addFormWidget(self.penWidthWidget, row=row)
        
        row += 1
        self.opacitySlider = widgets_numeric_inputs.sliderWithSpinBox(
            isFloat=True, normalize=True
        )
        self.opacitySlider.setMinimum(0)
        self.opacitySlider.setMaximum(100)
        self.opacitySlider.setValue(0.8)

        self.opacityWidget = widgets_line_edits.formWidget(
            self.opacitySlider, addInfoButton=False, stretchWidget=True,
            labelTextLeft='Symbol opacity: ', parent=self
        )
        if scatterPlotItem is not None:
            brush = scatterPlotItem.opts['brush']
            alpha = brush.color().alpha()
            opacity = alpha/255
            self.opacitySlider.setValue(opacity)
        formLayout.addFormWidget(self.opacityWidget, row=row)
        
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()
        
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addLayout(formLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
    
    def selectColor(self):
        color = self.colorButton.color()
        self.colorButton.origColor = color
        self.colorButton.colorDialog.setCurrentColor(color)
        _window_stacking.set_window_flags_on_top(
            self.colorButton.colorDialog, Qt.Window, parent=self
        )
        self.colorButton.colorDialog.open()
        w = self.width()
        left = self.pos().x()
        colorDialogTop = self.colorButton.colorDialog.pos().y()
        self.colorButton.colorDialog.move(w+left+10, colorDialogTop)
    
    def getBrush(self):
        r, g, b, _ = self.colorButton.color().getRgb()
        alpha = round(self.opacitySlider.value()*255)
        brushColor = (r, g, b, alpha)
        brush = pg.mkBrush(brushColor)
        return brush
    
    def getPen(self):
        color = self.colorButton.color()
        penWidth = self.penWidthSpinBox.value()
        if penWidth == 0:
            return
        
        pen = pg.mkPen(color, width=penWidth)
        return pen
    
    def ok_cb(self):
        self.cancel = False
        self.properties = {
            'brush': self.getBrush(),
            'pen': self.getPen()
        }
        self.close()
