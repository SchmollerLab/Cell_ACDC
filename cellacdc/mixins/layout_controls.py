"""Qt view adapter for layout-control workflows."""

from __future__ import annotations

from functools import partial

from natsort import natsorted
import re
from qtpy.QtCore import QTimer, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QAction,
    QActionGroup,
    QButtonGroup,
    QCheckBox,
    QDockWidget,
    QGridLayout,
    QLabel,
    QRadioButton,
    QSizePolicy,
    QWidget,
)

from cellacdc import myutils, widgets
from cellacdc.ui.modules.annotation.decorators import resetViewRange


class LayoutControlsView:
    """Qt-facing adapter around main layout and control surfaces."""

    """Headless decisions for GUI layout controls."""

    yes_value = 'Yes'
    no_value = 'No'

    def zoom_percentage_from_text(self, text: str) -> int:
        return int(re.findall(r'(\d+)%', text)[0])

    def zoom_factors(self, percentage: int) -> tuple[float, float] | None:
        if percentage == 100:
            return None
        factor = percentage / 100
        return factor, factor

    def checked_setting_value(self, checked: bool) -> str:
        return self.yes_value if checked else self.no_value

    def checked_from_setting_value(self, value) -> bool:
        return value == self.yes_value

    def should_retain_z_slider_space(
        self,
        *,
        checked: bool,
        z_slice_enabled: bool,
    ) -> bool:
        return checked and z_slice_enabled

    def tool_name_from_tooltip(self, tooltip: str) -> str:
        return re.findall(r'Name: (.*)', tooltip)[0]


    LEGACY_METHODS = (
        'zoomBottomLayoutActionTriggered',
        'retainSpaceSlidersToggled',
        'gui_createMainLayout',
        'gui_createRegionPropsDockWidget',
        'gui_createControlsToolbar',
        'gui_populateToolSettingsMenu',
        'useCenterBrushCursorHoverIDtoggled',
        'gui_createStatusBar',
        'gui_createTerminalWidget',
        'gui_terminalButtonClicked',
    )

    def __init__(self, host):
        object.__setattr__(self, 'host', host)
    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {'host'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def bind_legacy_methods(self):
        for name in self.LEGACY_METHODS:
            setattr(self.host, name, getattr(self, name))

    def zoomBottomLayoutActionTriggered(self, checked):
        if not checked:
            return
        perc = self.zoom_percentage_from_text(
            self.sender().text()
        )
        zoom_factors = self.zoom_factors(perc)
        if zoom_factors is not None:
            fontSizeFactor, heightFactor = zoom_factors
            self.resizeSlidersArea(
                fontSizeFactor=fontSizeFactor, heightFactor=heightFactor
            )
        else:
            self.image_controls_view.gui_resetBottomLayoutHeight()
        self.df_settings.at['bottom_sliders_zoom_perc', 'value'] = perc
        self.df_settings.to_csv(self.settings_csv_path)
        QTimer.singleShot(150, self.resizeGui)

    def retainSpaceSlidersToggled(self, checked):
        self.df_settings.at['retain_space_hidden_sliders', 'value'] = (
            self.checked_setting_value(checked)
        )
        self.df_settings.to_csv(self.settings_csv_path)
        retainSpaceZ = self.should_retain_z_slider_space(
            checked=checked,
            z_slice_enabled=self.zSliceScrollBar.isEnabled(),
        )
        myutils.setRetainSizePolicy(self.zSliceScrollBar, retain=retainSpaceZ)
        myutils.setRetainSizePolicy(self.zProjComboBox, retain=retainSpaceZ)
        myutils.setRetainSizePolicy(self.zSliceOverlay_SB, retain=retainSpaceZ)
        myutils.setRetainSizePolicy(self.zProjOverlay_CB, retain=retainSpaceZ)
        myutils.setRetainSizePolicy(self.overlay_z_label, retain=retainSpaceZ)

        QTimer.singleShot(200, self.resizeGui)

    def gui_createMainLayout(self):
        mainLayout = QGridLayout()
        row, col = 0, 1 # Leave column 1 for the overlay labels gradient editor
        mainLayout.addLayout(self.leftSideDocksLayout, row, col, 2, 1)

        row = 0
        col = 2
        mainLayout.addWidget(self.graphLayout, row, col, 1, 2)
        mainLayout.setRowStretch(row, 2)

        col = 4 # graphLayout spans two columns
        mainLayout.addWidget(self.labelsGrad, row, col)

        col = 5 
        mainLayout.addLayout(self.rightSideDocksLayout, row, col, 2, 1)

        col = 2
        row += 1
        self.resizeBottomLayoutLine = widgets.VerticalResizeHline()
        mainLayout.addWidget(self.resizeBottomLayoutLine, row, col, 1, 2)
        self.resizeBottomLayoutLine.dragged.connect(
            self.resizeBottomLayoutLineDragged
        )
        self.resizeBottomLayoutLine.clicked.connect(
            self.resizeBottomLayoutLineClicked
        )
        self.resizeBottomLayoutLine.released.connect(
            self.resizeBottomLayoutLineReleased
        )

        # row += 1
        # mainLayout.addItem(QSpacerItem(5,5), row+1, col, 1, 2)

        # row, col = 1, 2
        # mainLayout.addLayout(
        #     self.bottomLayout, row, col, 1, 2, alignment=Qt.AlignLeft
        # )

        row += 1
        mainLayout.addWidget(self.bottomScrollArea, row, col, 1, 2)
        mainLayout.setRowStretch(row, 0)

        # row, col = 2, 1
        # mainLayout.addWidget(self.terminal, row, col, 1, 4)
        # self.terminal.hide()

        return mainLayout

    def gui_createRegionPropsDockWidget(self, side=Qt.LeftDockWidgetArea):
        self.propsDockWidget = QDockWidget(
            'Cell-ACDC objects', self.host
        )
        self.guiTabControl = widgets.guiTabControl(self.propsDockWidget)

        # self.guiTabControl.setFont(_font)

        self.propsDockWidget.setWidget(self.guiTabControl)
        self.propsDockWidget.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable 
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.propsDockWidget.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )
        
        self.addDockWidget(side, self.propsDockWidget)
        self.propsDockWidget.hide()

    def gui_createControlsToolbar(self):
        self.controlToolBars = []
        self.addToolBarBreak()
        
        # Edit toolbar
        modeToolBar = widgets.ToolBar("Mode", self.host)
        self.addToolBar(modeToolBar)

        self.modeComboBox = widgets.ComboBox()
        self.modeComboBox.addItems(self.modeItems)
        self.modeComboBoxLabel = QLabel('    Mode: ')
        self.modeComboBoxLabel.setBuddy(self.modeComboBox)
        modeToolBar.addWidget(self.modeComboBoxLabel)
        modeToolBar.addWidget(self.modeComboBox)
        modeToolBar.setVisible(False)
        
        self.modeToolBar = modeToolBar
        
        self.overlayToolbar = widgets.OverlayToolbar(parent=self.host)
        self.addToolBar(Qt.TopToolBarArea, self.overlayToolbar)
        self.overlayToolbar.setVisible(False)
        self.overlayToolbar.sigSetTranspacency.connect(
            self.setOverlayTransparency
        )
        self.overlayToolbar.sigSetSingleChannel.connect(
            self.setOverlaySingleChannel
        )
        
        self.autoPilotZoomToObjToolbar = widgets.ToolBar(
            "Auto-zoom to objects", self
        )
        self.autoPilotZoomToObjToolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.autoPilotZoomToObjToolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, self.autoPilotZoomToObjToolbar)
        # self.autoPilotZoomToObjToolbar.setIconSize(QSize(16, 16))
        self.autoPilotZoomToObjToolbar.setVisible(False)
        self.autoPilotZoomToObjToolbar.keepVisibleWhenActive = True
        self.controlToolBars.append(self.autoPilotZoomToObjToolbar)
        
        # Highlighted ID or searched ID toolbar
        self.highlightIDToolbar = widgets.HighlightedIDToolbar(
            parent=self.host
        )
        self.addToolBar(Qt.TopToolBarArea, self.highlightIDToolbar)
        self.highlightIDToolbar.setVisible(False)
        self.highlightIDToolbar.keepVisibleWhenActive = True
        self.controlToolBars.append(self.highlightIDToolbar)
        
        self.highlightIDToolbar.sigIDChanged.connect(
            self.setHighlighedIDfromToolbar
        )
        
        # Widgets toolbar
        brushEraserToolBar = widgets.ToolBar("Widgets", self.host)
        self.addToolBar(Qt.TopToolBarArea, brushEraserToolBar)
        self.controlToolBars.append(brushEraserToolBar)

        self.editIDspinbox = widgets.SpinBox()
        # self.editIDspinbox.setMaximum(2**32-1)
        editIDLabel = QLabel('   ID: ')
        self.editIDLabelAction = brushEraserToolBar.addWidget(editIDLabel)
        self.editIDspinboxAction = brushEraserToolBar.addWidget(
            self.editIDspinbox
        )
        self.editIDLabelAction.setVisible(False)
        self.editIDspinboxAction.setVisible(False)
        self.editIDspinboxAction.setDisabled(True)
        self.editIDLabelAction.setDisabled(True)

        brushEraserToolBar.addWidget(QLabel(' '))
        self.autoIDcheckbox = QCheckBox('Auto-ID')
        self.autoIDcheckbox.setChecked(True)
        self.autoIDcheckboxAction = brushEraserToolBar.addWidget(self.autoIDcheckbox)
        self.autoIDcheckboxAction.setVisible(False)

        self.brushSizeSpinbox = widgets.SpinBox(
            disableKeyPress=True,
            allowNegative=False
        )
        self.brushSizeSpinbox.setValue(4)
        brushSizeLabel = QLabel('   Size: ')
        brushSizeLabel.setBuddy(self.brushSizeSpinbox)
        self.brushSizeLabelAction = brushEraserToolBar.addWidget(brushSizeLabel)
        self.brushSizeAction = brushEraserToolBar.addWidget(self.brushSizeSpinbox)
        self.brushSizeLabelAction.setVisible(False)
        self.brushSizeAction.setVisible(False)
        
        brushEraserToolBar.addWidget(QLabel('  '))
        self.brushAutoFillCheckbox = QCheckBox('Auto-fill holes')
        self.brushAutoFillAction = brushEraserToolBar.addWidget(
            self.brushAutoFillCheckbox
        )
        self.brushAutoFillAction.setVisible(False)
        if 'brushAutoFill' in self.df_settings.index:
            checked = self.df_settings.at['brushAutoFill', 'value'] == 'Yes'
            self.brushAutoFillCheckbox.setChecked(checked)
        
        brushEraserToolBar.addWidget(QLabel('  '))
        self.brushAutoHideCheckbox = QCheckBox('Hide objects when hovering')
        self.brushAutoHideAction = brushEraserToolBar.addWidget(
            self.brushAutoHideCheckbox
        )
        self.brushAutoHideCheckbox.setChecked(True)
        self.brushAutoHideAction.setVisible(False)
        if 'brushAutoHide' in self.df_settings.index:
            checked = self.df_settings.at['brushAutoHide', 'value'] == 'Yes'
            self.brushAutoHideCheckbox.setChecked(checked)
        
        brushEraserToolBar.setVisible(False)
        self.brushEraserToolBar = brushEraserToolBar

        self.wandControlsToolbar = widgets.WandControlsToolbar(
            parent=self.host
        )

        self.addToolBar(Qt.TopToolBarArea , self.wandControlsToolbar)
        self.wandControlsToolbar.setVisible(False)
        self.controlToolBars.append(self.wandControlsToolbar)

        separatorW = 5
        self.labelRoiToolbar = widgets.ToolBar(
            "Magic labeller controls", self.host
        )
        self.labelRoiToolbar.addWidget(QLabel('ROI n. of z-slices: '))
        self.labelRoiZdepthSpinbox = widgets.SpinBox(disableKeyPress=True)
        self.labelRoiToolbar.addWidget(self.labelRoiZdepthSpinbox)

        self.labelRoiToolbar.addWidget(widgets.QHWidgetSpacer(width=separatorW))
        self.labelRoiToolbar.addWidget(widgets.QVLine())
        self.labelRoiToolbar.addWidget(widgets.QHWidgetSpacer(width=separatorW))

        self.labelRoiReplaceExistingObjectsCheckbox = QCheckBox(
            'Remove objs. touched by new ones'
        )
        self.labelRoiToolbar.addWidget(self.labelRoiReplaceExistingObjectsCheckbox)
        self.labelRoiAutoClearBorderCheckbox = QCheckBox(
            'Clear ROI borders before adding new objs.'
        )
        self.labelRoiAutoClearBorderCheckbox.setChecked(True)
        self.labelRoiToolbar.addWidget(self.labelRoiAutoClearBorderCheckbox)
        
        self.labelRoiToolbar.addWidget(widgets.QHWidgetSpacer(width=separatorW))
        self.labelRoiToolbar.addWidget(widgets.QVLine())
        self.labelRoiToolbar.addWidget(widgets.QHWidgetSpacer(width=separatorW))

        group = QButtonGroup()
        group.setExclusive(True)
        self.labelRoiIsRectRadioButton = QRadioButton('Rect. ROI')
        self.labelRoiIsRectRadioButton.setChecked(True)
        self.labelRoiIsFreeHandRadioButton = QRadioButton('Freehand ROI')
        self.labelRoiIsCircularRadioButton = QRadioButton('Circular ROI')
        group.addButton(self.labelRoiIsRectRadioButton)
        group.addButton(self.labelRoiIsFreeHandRadioButton)
        group.addButton(self.labelRoiIsCircularRadioButton)
        self.labelRoiToolbar.addWidget(self.labelRoiIsRectRadioButton)
        self.labelRoiToolbar.addWidget(self.labelRoiIsFreeHandRadioButton)
        self.labelRoiToolbar.addWidget(self.labelRoiIsCircularRadioButton)
        self.labelRoiToolbar.addWidget(QLabel(' | Radius (pixel): '))
        self.labelRoiCircularRadiusSpinbox = widgets.SpinBox(disableKeyPress=True)
        self.labelRoiCircularRadiusSpinbox.setMinimum(1)
        self.labelRoiCircularRadiusSpinbox.setValue(11)
        self.labelRoiCircularRadiusSpinbox.setDisabled(True)
        self.labelRoiToolbar.addWidget(self.labelRoiCircularRadiusSpinbox)
        
        self.labelRoiToolbar.addWidget(widgets.QHWidgetSpacer(width=separatorW))
        self.labelRoiToolbar.addWidget(widgets.QVLine())
        self.labelRoiToolbar.addWidget(widgets.QHWidgetSpacer(width=separatorW))

        startFrameLabel = QLabel('Start frame n. ')
        startFrameLabel.setDisabled(True)
        self.labelRoiToolbar.addWidget(startFrameLabel)
        self.labelRoiStartFrameNoSpinbox = widgets.SpinBox(disableKeyPress=True)
        self.labelRoiStartFrameNoSpinbox.label = startFrameLabel
        self.labelRoiStartFrameNoSpinbox.setValue(1)
        self.labelRoiStartFrameNoSpinbox.setMinimum(1)
        self.labelRoiToolbar.addWidget(self.labelRoiStartFrameNoSpinbox)
        self.labelRoiStartFrameNoSpinbox.setDisabled(True)

        self.labelRoiFromCurrentFrameAction = QAction(self.host)
        self.labelRoiFromCurrentFrameAction.setText('Segment from current frame')
        self.labelRoiFromCurrentFrameAction.setIcon(QIcon(":frames_current.svg"))
        self.labelRoiToolbar.addAction(self.labelRoiFromCurrentFrameAction)
        self.labelRoiFromCurrentFrameAction.setDisabled(True)

        self.labelRoiToolbar.addWidget(widgets.QHWidgetSpacer(width=3))
        stopFrameLabel = QLabel(' Stop frame n. ')
        stopFrameLabel.setDisabled(True)
        self.labelRoiToolbar.addWidget(stopFrameLabel)
        self.labelRoiStopFrameNoSpinbox = widgets.SpinBox(disableKeyPress=True)
        self.labelRoiStopFrameNoSpinbox.label = stopFrameLabel
        self.labelRoiStopFrameNoSpinbox.setValue(1)
        self.labelRoiStopFrameNoSpinbox.setMinimum(1)
        self.labelRoiToolbar.addWidget(self.labelRoiStopFrameNoSpinbox)
        self.labelRoiStopFrameNoSpinbox.setDisabled(True)

        self.labelRoiToEndFramesAction = QAction(self.host)
        self.labelRoiToEndFramesAction.setText('Segment all remaining frames')
        self.labelRoiToEndFramesAction.setIcon(QIcon(":frames_end.svg"))
        self.labelRoiToolbar.addAction(self.labelRoiToEndFramesAction)
        self.labelRoiToEndFramesAction.setDisabled(True)

        self.labelRoiTrangeCheckbox = QCheckBox('Segment range of frames')
        self.labelRoiToolbar.addWidget(self.labelRoiTrangeCheckbox)

        self.labelRoiViewCurrentModelAction = QAction(self.host)
        self.labelRoiViewCurrentModelAction.setText(
            'View current model\'s parameters'
        )
        self.labelRoiViewCurrentModelAction.setIcon(QIcon(":view.svg"))
        self.labelRoiToolbar.addAction(self.labelRoiViewCurrentModelAction)
        self.labelRoiViewCurrentModelAction.setDisabled(True)

        self.addToolBar(Qt.TopToolBarArea, self.labelRoiToolbar)
        self.controlToolBars.append(self.labelRoiToolbar)
        self.labelRoiToolbar.setVisible(False)
        self.labelRoiTypesGroup = group

        self.loadLabelRoiLastParams()

        self.labelRoiTrangeCheckbox.toggled.connect(
            self.labelRoiTrangeCheckboxToggled
        )
        self.labelRoiReplaceExistingObjectsCheckbox.toggled.connect(
            self.storeLabelRoiParams
        )
        self.labelRoiIsCircularRadioButton.toggled.connect(
            self.labelRoiIsCircularRadioButtonToggled
        )
        self.labelRoiCircularRadiusSpinbox.valueChanged.connect(
            self.updateLabelRoiCircularSize
        )
        self.labelRoiCircularRadiusSpinbox.valueChanged.connect(
            self.storeLabelRoiParams
        )
        self.labelRoiZdepthSpinbox.valueChanged.connect(
            self.storeLabelRoiParams
        )
        self.labelRoiAutoClearBorderCheckbox.toggled.connect(
            self.storeLabelRoiParams
        )
        group.buttonToggled.connect(self.storeLabelRoiParams)

        self.labelRoiToEndFramesAction.triggered.connect(
            self.labelRoiToEndFramesTriggered
        )
        self.labelRoiFromCurrentFrameAction.triggered.connect(
            self.labelRoiFromCurrentFrameTriggered
        )
        self.labelRoiViewCurrentModelAction.triggered.connect(
            self.labelRoiViewCurrentModel
        )

        self.keepIDsToolbar = widgets.ToolBar(
            "Keep IDs controls", self.host
        )
        self.keepIDsConfirmAction = QAction()
        self.keepIDsConfirmAction.setIcon(QIcon(":greenTick.svg"))
        self.keepIDsConfirmAction.setToolTip('Apply "keep IDs" selection')
        self.keepIDsConfirmAction.setDisabled(True)
        self.keepIDsToolbar.addAction(self.keepIDsConfirmAction)
        self.keepIDsToolbar.addWidget(QLabel('  IDs to keep: '))
        instructionsText = (
            ' (Separate IDs by comma. Use a dash to denote a range of IDs)'
        )
        instructionsLabel = QLabel(instructionsText)
        self.keptIDsLineEdit = widgets.KeepIDsLineEdit(
            instructionsLabel, parent=self.host
        )
        self.keepIDsToolbar.addWidget(self.keptIDsLineEdit)
        self.keepIDsToolbar.addWidget(instructionsLabel)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.keepIDsToolbar.addWidget(spacer)
        self.addToolBar(Qt.TopToolBarArea, self.keepIDsToolbar)
        self.keepIDsToolbar.setVisible(False)
        self.controlToolBars.append(self.keepIDsToolbar)

        self.keptIDsLineEdit.sigEnterPressed.connect(self.applyKeepObjects)
        self.keptIDsLineEdit.sigIDsChanged.connect(self.updateKeepIDs)
        self.keepIDsConfirmAction.triggered.connect(self.applyKeepObjects)
        
        # closeToolbarAction = QAction(
        #     QIcon(":cancelButton.svg"), "Close toolbar...", self
        # )
        # closeToolbarAction.triggered.connect(self.closeToolbars)
        # self.autoPilotZoomToObjToolbar.addAction(closeToolbarAction)
        
        self.autoPilotZoomToObjToolbar.addWidget(widgets.QVLine())
        self.autoPilotZoomToObjToolbar.addWidget(
            widgets.QHWidgetSpacer(width=separatorW)
        )
        
        spinBox = widgets.SpinBox()
        spinBox.setMinimum(1)
        spinBox.label = QLabel('  Zoom to ID: ')
        spinBox.labelAction = self.autoPilotZoomToObjToolbar.addWidget(spinBox.label)
        spinBox.action = self.autoPilotZoomToObjToolbar.addWidget(spinBox)
        spinBox.editingFinished.connect(self.zoomToObj)
        spinBox.sigUpClicked.connect(self.autoZoomNextObj)
        spinBox.sigDownClicked.connect(self.autoZoomPrevObj)
        self.autoPilotZoomToObjSpinBox = spinBox
        toggle = widgets.Toggle()
        self.autoPilotZoomToObjToggle = toggle
        toggle.toggled.connect(self.autoPilotZoomToObjToggled)
        toggle.label = QLabel('  Auto-pilot: ')
        tooltip = (
            'When auto-pilot is active, you can use Up/Down arrows to '
            'automatically zoom to the next/previous object.\n\n'
            'Alternatively, you can type the ID of the object you want to '
            'zoom to.'
        )
        toggle.label.setToolTip(tooltip)
        toggle.setToolTip(tooltip)
        self.autoPilotZoomToObjToolbar.addWidget(toggle.label)
        self.autoPilotZoomToObjToolbar.addWidget(toggle)
        
        self.pointsLayersToolbars = []
        
        self.pointsLayersToolbar = widgets.PointsLayersToolbar(
            parent=self.host
        )
        self.pointsLayersToolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        
        self.pointsLayersToolbar.sigAddPointsLayer.connect(
            self.addPointsLayerTriggered
        )
        
        self.addToolBar(Qt.TopToolBarArea, self.pointsLayersToolbar)
        
        self.pointsLayersToolbar.setVisible(False)
        self.pointsLayersToolbar.keepVisibleWhenActive = True
        self.controlToolBars.append(self.pointsLayersToolbar)
        
        self.pointsLayersToolbars.append(
            self.pointsLayersToolbar
        )

        self.manualTrackingToolbar = widgets.ManualTrackingToolBar(
            "Manual tracking controls", self
        )
        self.manualTrackingToolbar.sigIDchanged.connect(self.initGhostObject)
        self.manualTrackingToolbar.sigDisableGhost.connect(self.clearGhost)
        self.manualTrackingToolbar.sigClearGhostContour.connect(
            self.clearGhostContour
        )
        self.manualTrackingToolbar.sigClearGhostMask.connect(
            self.clearGhostMask
        )
        self.manualTrackingToolbar.sigGhostOpacityChanged.connect(
            self.updateGhostMaskOpacity
        )

        self.addToolBar(Qt.TopToolBarArea, self.manualTrackingToolbar)
        self.manualTrackingToolbar.setVisible(False)
        self.controlToolBars.append(self.manualTrackingToolbar)
        
        self.manualBackgroundToolbar = widgets.ManualBackgroundToolBar(
            "Manual background controls", self
        )
        self.manualBackgroundToolbar.sigIDchanged.connect(
            self.initManualBackgroundObject
        )
        self.addToolBar(Qt.TopToolBarArea, self.manualBackgroundToolbar)
        self.manualBackgroundToolbar.setVisible(False)
        self.controlToolBars.append(self.manualBackgroundToolbar)
        
        # Copy lost object contour toolbar
        self.copyLostObjToolbar = widgets.CopyLostObjectToolbar(
            "Copy lost object controls", self
        )
        for name, action in self.copyLostObjToolbar.widgetsWithShortcut.items():
            self.widgetsWithShortcut[name] = action

        self.copyLostObjToolbar.sigCopyAllObjects.connect(
            self.copyAllLostObjects
        )
        
        self.addToolBar(Qt.TopToolBarArea, self.copyLostObjToolbar)
        self.copyLostObjToolbar.setVisible(False)
        # self.controlToolBars.append(self.copyLostObjToolbar)
        
        # Copy lost object contour toolbar
        self.drawClearRegionToolbar = widgets.DrawClearRegionToolbar(
            "Draw freehand region and clear objects controls", self
        )
        
        self.addToolBar(Qt.TopToolBarArea, self.drawClearRegionToolbar)
        self.drawClearRegionToolbar.setVisible(False)
        self.controlToolBars.append(self.drawClearRegionToolbar)

        try:
            addNewIDToggleState = self.df_settings.at['addNewIDsWhitelistToggle', 'value'] == 'Yes'
        except KeyError:
            addNewIDToggleState = True
        
        self.whitelistIDsToolbar = widgets.WhitelistIDsToolbar(
            addNewIDToggleState, self
        )
        for name, action in self.whitelistIDsToolbar.widgetsWithShortcut.items():
            self.widgetsWithShortcut[name] = action
        
        self.addToolBar(Qt.TopToolBarArea, self.whitelistIDsToolbar)
        self.whitelistIDsToolbar.setVisible(False)
        self.controlToolBars.append(self.whitelistIDsToolbar)
        
        self.magicPromptsToolbar = widgets.MagicPromptsToolbar(self.host)
        for name, action in self.magicPromptsToolbar.widgetsWithShortcut.items():
            self.widgetsWithShortcut[name] = action
        
        self.magicPromptsToolbar.sigComputeOnZoom.connect(
            self.magic_prompts_view.magicPromptsComputeOnZoomTriggered
        )
        self.magicPromptsToolbar.sigComputeOnImage.connect(
            self.magic_prompts_view.magicPromptsComputeOnImageTriggered
        )
        self.magicPromptsToolbar.sigInitSelectedModel.connect(
            self.magic_prompts_view.magicPromptsInitModel
        )
        self.magicPromptsToolbar.sigViewModelParams.connect(
            self.magic_prompts_view.viewSetMagicPromptModelParams
        )
        self.magicPromptsToolbar.sigClearPoints.connect(
            partial(
                self.magic_prompts_view.magicPromptsClearPoints,
                only_zoom=False,
            )
        )
        self.magicPromptsToolbar.sigClearPointsOnZmom.connect(
            partial(
                self.magic_prompts_view.magicPromptsClearPoints,
                only_zoom=True,
            )
        )
        self.magicPromptsToolbar.sigInterpolateZslice.connect(
            self.magic_prompts_view.magicPromptsInterpolateZsliceToggled
        )

        self.addToolBar(Qt.TopToolBarArea, self.magicPromptsToolbar)
        self.magicPromptsToolbar.setVisible(False)
        self.magicPromptsToolbar.keepVisibleWhenActive = True
        self.controlToolBars.append(self.magicPromptsToolbar)
        
        self.promptSegmentPointsLayerToolbar = (
            widgets.PromptableModelPointsLayerToolbar(parent=self.host)
        )
        self.promptSegmentPointsLayerToolbar.setContextMenuPolicy(
            Qt.PreventContextMenu
        )
        
        self.addToolBar(Qt.TopToolBarArea, self.promptSegmentPointsLayerToolbar)
        self.promptSegmentPointsLayerToolbar.setVisible(False)
        
        self.pointsLayersToolbars.append(
            self.promptSegmentPointsLayerToolbar
        )
        
        # Second level toolbar
        secondLevelToolbar = widgets.ToolBar(
            "Second level toolbar", self.host
        )
        self.addToolBar(Qt.TopToolBarArea, secondLevelToolbar)
        self.delObjToolAction = QAction(self.host)
        self.delObjToolAction.setIcon(QIcon(":del_obj_click.svg"))
        self.delObjToolAction.setCheckable(True)
        self.delObjToolAction.setToolTip(
            'Customisable delete object action\n\n'
            'Go to the `Settings --> Customise keyboard shortcuts...` menu '
            'on the top menubar\n'
            'to customise the action required to delete '
            'an object with a click.\n\n'
            'When working with 3D segmentations, to delete only the z-slice mask, hold "Shift" while clicking.'
        )
        secondLevelToolbar.addAction(self.delObjToolAction)
        secondLevelToolbar.setMovable(False)
        self.secondLevelToolbar = secondLevelToolbar
        self.secondLevelToolbar.setVisible(False)
        
    def gui_populateToolSettingsMenu(self):
        brushHoverModeActionGroup = QActionGroup(self.host)
        brushHoverModeActionGroup.setExclusive(True)
        self.brushHoverCenterModeAction = QAction()
        self.brushHoverCenterModeAction.setCheckable(True)
        self.brushHoverCenterModeAction.setText(
            'Use center of the brush/eraser cursor to determine hover ID'
        )
        self.brushHoverCircleModeAction = QAction()
        self.brushHoverCircleModeAction.setCheckable(True)
        self.brushHoverCircleModeAction.setText(
            'Use the entire circle of the brush/eraser cursor to determine hover ID'
        )
        brushHoverModeActionGroup.addAction(self.brushHoverCenterModeAction)
        brushHoverModeActionGroup.addAction(self.brushHoverCircleModeAction)
        brushHoverModeMenu = self.settingsMenu.addMenu(
            'Brush/eraser cursor hovering mode'
        )
        brushHoverModeMenu.addAction(self.brushHoverCenterModeAction)
        brushHoverModeMenu.addAction(self.brushHoverCircleModeAction)

        if 'useCenterBrushCursorHoverID' not in self.df_settings.index:
            self.df_settings.at['useCenterBrushCursorHoverID', 'value'] = 'Yes'

        useCenterBrushCursorHoverID = self.df_settings.at[
            'useCenterBrushCursorHoverID', 'value'
        ] == 'Yes'
        self.brushHoverCenterModeAction.setChecked(useCenterBrushCursorHoverID)
        self.brushHoverCircleModeAction.setChecked(not useCenterBrushCursorHoverID)

        self.brushHoverCenterModeAction.toggled.connect(
            self.useCenterBrushCursorHoverIDtoggled
        )

        self.settingsMenu.addSeparator()

        keepToolActiveNames = {
            'Segment range of frames': self.labelRoiTrangeCheckbox
        }
        for button in self.checkableQButtonsGroup.buttons():
            if button.toolTip() == "":
                toolName = "MISSING"
                continue
            else:
                toolName = self.tool_name_from_tooltip(
                    button.toolTip()
                )
            keepToolActiveNames[toolName] = button
        
        keepToolActiveNames = dict(natsorted(keepToolActiveNames.items()))
        
        applyToNewFrameNames = {
            'Segmenting for lost IDs': self.segForLostIDsButton,
            'Delete bordering objects': self.delBorderObjAction.button,
            'Delete newly segmented objects': self.delNewObjAction.button,
        }
        
        allToolsList = list(keepToolActiveNames.keys()) + list(applyToNewFrameNames.keys())
        allToolsList = natsorted(allToolsList)
        
        menus = {}
        
        for toolName in allToolsList:
            menuItemText = f'{toolName} tool'.replace('  ', ' ')
            menus[toolName] = self.settingsMenu.addMenu(menuItemText)
            
        self.keepToolActiveActions = dict()
        self.applyToolNewFrameActions = dict()
        self.applyToolNewFrameButtons = dict()
        all_checked = True
        
        for toolName, button in keepToolActiveNames.items():
            menu = menus[toolName]
            action = QAction(button)
            action.setText('Keep tool active after using it')
            action.setCheckable(True)
            if toolName in self.df_settings.index:
                action.setChecked(True)
            else:
                all_checked = False
            action.toggled.connect(self.keepToolActiveActionToggled)
            menu.addAction(action)
            self.keepToolActiveActions[toolName] = action
            
        for toolName, button in applyToNewFrameNames.items():
            menu = menus[toolName]
            action = QAction(button)
            action.setText('Apply when visitng new frame')            
            action.setCheckable(True)
            action.toggled.connect(self.applyToolNewFrameActionToggled)
            menu.addAction(action)
            self.applyToolNewFrameActions[toolName] = action
            self.applyToolNewFrameButtons[toolName] = button
        
        for toolName in self.applyToolNewFrameActions.keys():
            settingString = toolName.strip()
            settingString = toolName.replace(' ', '_')
            settingString = f'{settingString}_applyNewFrame'
            if settingString in self.df_settings.index:
                val = self.df_settings.at[settingString, 'value']
                if val == 'applyNewFrame':
                    self.applyToolNewFrameActions[toolName].setChecked(True)
        
        self.settingsMenu.addSeparator()

        self.keepAllToolsActiveToggle = QAction()
        self.keepAllToolsActiveToggle.setText(
            'Keep all tools active after using them'
        )
        self.keepAllToolsActiveToggle.setCheckable(True)
        self.keepAllToolsActiveToggle.setChecked(all_checked)
        self.keepAllToolsActiveToggle.toggled.connect(
            self.keepAllToolsActiveActionToggled
        )
        self.settingsMenu.addAction(self.keepAllToolsActiveToggle)
        self.settingsMenu.addSeparator()
        
        askHowFutureFramesMenu = self.settingsMenu.addMenu(
            'Ask how to propagate changes to future frames'
        )
        self.askHowFutureFramesActions = {}
        askHowFutureFramesActionsKeys = (
            'Delete ID', 
            'Exclude cell from analysis', 
            'Annotate cell as dead', 
            'Edit ID',
            'Keep ID'
        )
        for key in askHowFutureFramesActionsKeys:
            askHowFutureFramesAction = QAction()
            askHowFutureFramesAction.setText(f'Ask for "{key}" action')
            askHowFutureFramesAction.setCheckable(True)
            askHowFutureFramesAction.setChecked(True)
            askHowFutureFramesAction.setDisabled(True)
            askHowFutureFramesMenu.addAction(askHowFutureFramesAction)
            self.askHowFutureFramesActions[key] = askHowFutureFramesAction
        
        warningsMenu = self.settingsMenu.addMenu('Warnings and pop-ups')
        self.warnLostCellsAction = QAction()
        self.warnLostCellsAction.setText('Show pop-up warning for lost cells')
        self.warnLostCellsAction.setCheckable(True)
        self.warnLostCellsAction.setChecked(True)
        warningsMenu.addAction(self.warnLostCellsAction)

        warnEditingWithAnnotTexts = {
            'Delete ID': 'Show warning when deleting ID that has annotations',
            'Separate IDs': 'Show warning when separating IDs that have annotations',
            'Edit ID': 'Show warning when editing ID that has annotations',
            'Annotate ID as dead':
                'Show warning when annotating dead ID that has annotations',
            'Delete ID with eraser':
                'Show warning when erasing ID that has annotations',
            'Add new ID with brush tool':
                'Show warning when adding new ID (brush) that has annotations',
            'Merge IDs':
                'Show warning when merging IDs that have annotations',
            'Add new ID with curvature tool':
                'Show warning when adding new ID (curv. tool) that has annotations',
            'Add new ID with magic-wand':
                'Show warning when adding new ID (magic-wand) that has annotations',
            'Delete IDs using ROI':
                'Show warning when using ROIs to delete IDs that have annotations',
        }
        self.warnEditingWithAnnotActions = {}
        for key, desc in warnEditingWithAnnotTexts.items():
            action = QAction()
            action.setText(desc)
            action.setCheckable(True)
            action.setChecked(True)
            action.removeAnnot = False
            self.warnEditingWithAnnotActions[key] = action
            warningsMenu.addAction(action)

    def useCenterBrushCursorHoverIDtoggled(self, checked):
        if checked:
            self.df_settings.at['useCenterBrushCursorHoverID', 'value'] = 'Yes'
        else:
            self.df_settings.at['useCenterBrushCursorHoverID', 'value'] = 'No'
        self.df_settings.to_csv(self.settings_csv_path)

    def gui_createStatusBar(self):
        self.statusbar = self.statusBar()
        # Permanent widget
        self.wcLabel = QLabel('')
        self.statusbar.addPermanentWidget(self.wcLabel)

        # self.toggleTerminalButton = widgets.ToggleTerminalButton()
        # self.statusbar.addWidget(self.toggleTerminalButton)
        # self.toggleTerminalButton.sigClicked.connect(
        #     self.gui_terminalButtonClicked
        # )

        self.statusBarLabel = QLabel('')
        self.statusbar.addWidget(self.statusBarLabel)
    
    def gui_createTerminalWidget(self):
        self.terminal = widgets.QLog(logger=self.logger)
        self.terminal.connect()
        self.terminalDock = QDockWidget('Log', self.host)

        self.terminalDock.setWidget(self.terminal)
        self.terminalDock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.terminalDock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.terminalDock)
        # self.terminalDock.widget().layout().setContentsMargins(10,0,10,0)
        self.terminalDock.setVisible(False)
    
    @resetViewRange
    def gui_terminalButtonClicked(self, terminalVisible):
        self.terminalDock.setVisible(terminalVisible)