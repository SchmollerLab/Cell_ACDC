"""Layout of the main GUI window: main layout, bottom widgets and sliders
area, graphics plots and image items, two-images (right image) layout,
focus and resizing. Also TitleIDsPopupEventFilter (title IDs popup).
Mixins of guiWin (gui/main_window/window.py) only: the measurements
viewer never composes them.
"""
import re
import traceback
from collections import defaultdict
import numpy as np
from qtpy.QtCore import Qt, QRect, QRectF, QTimer, QEvent, QObject
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QAction, QLabel, QPushButton, QHBoxLayout, QMenu, QGroupBox, QGridLayout,
    QCheckBox, QActionGroup, QWidget, QGridLayout, QGraphicsProxyWidget,
    QVBoxLayout, QFormLayout
)
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import graphLayoutBkgrColor, darkBkgrColor
from cellacdc import apps
from cellacdc import core, myutils
from cellacdc import widgets
from cellacdc import colors
from . import constants as gui_main_constants


class TitleIDsPopupEventFilter(QObject):
    def __init__(self, enter_callback, leave_callback, parent=None):
        super().__init__(parent)
        self.enter_callback = enter_callback
        self.leave_callback = leave_callback

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter:
            self.enter_callback()
        elif event.type() == QEvent.Leave:
            self.leave_callback()
        return False


class LayoutGuiElements:
    """Layout of the main GUI window: main layout, quick settings, bottom
    widgets and sliders area, graphics plots and image items, the
    two-images (right image) layout, focus handling and resizing.
    resizeSlidersArea rebinds the font _font of
    gui/main_window/constants.py.
    """

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

        row += 1
        mainLayout.addWidget(self.bottomScrollArea, row, col, 1, 2)
        mainLayout.setRowStretch(row, 0)

        return mainLayout

    def gui_createQuickSettingsWidgets(self):
        self.quickSettingsLayout = QVBoxLayout()
        self.quickSettingsGroupbox = widgets.GroupBox()
        self.quickSettingsGroupbox.setTitle('Quick settings')

        layout = QFormLayout()
        layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint
        )
        layout.setFormAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        self.viewPreprocDataToggle = widgets.Toggle()
        viewPreprocDataToggleTooltip = (
            'View pre-processed data. See menu `Image --> Pre-processing...`\n'
            'on the top menubar.'
        )
        self.viewPreprocDataToggle.setChecked(False)
        self.viewPreprocDataToggle.setToolTip(viewPreprocDataToggleTooltip)
        viewPreprocDataToggleLabel = QLabel('View pre-processed image')
        viewPreprocDataToggleLabel.setToolTip(viewPreprocDataToggleTooltip)
        layout.addRow(viewPreprocDataToggleLabel, self.viewPreprocDataToggle)

        self.viewCombineChannelDataToggle = widgets.Toggle()
        viewCombineChannelDataToggleTooltip = (
            'View combined channel. See menu `Image --> combing channels...`\n'
            'on the top menubar.'
        )
        self.viewCombineChannelDataToggle.setChecked(False)
        self.viewCombineChannelDataToggle.setToolTip(
            viewCombineChannelDataToggleTooltip
        )
        viewCombineChannelDataToggleLabel = QLabel('View combined channels')
        viewCombineChannelDataToggleLabel.setToolTip(
            viewCombineChannelDataToggleTooltip
        )
        layout.addRow(
            viewCombineChannelDataToggleLabel, 
            self.viewCombineChannelDataToggle
        )

        self.autoSaveToggle = widgets.Toggle()
        autoSaveTooltip = (
            'Automatically store a copy of the segmentation data '
            'in the `.recovery` folder after every edit.'
        )
        self.autoSaveToggle.setChecked(True)
        self.autoSaveToggle.setToolTip(autoSaveTooltip)
        autoSaveLabel = QLabel('Autosave segmentation')
        autoSaveLabel.setToolTip(autoSaveTooltip)
        layout.addRow(autoSaveLabel, self.autoSaveToggle)
        
        self.autoSaveAnnotToggle = widgets.Toggle()
        autoSaveAnnotTooltip = (
            'Automatically store a copy of the annotations (acdc_output CSV file) '
            'in the `.recovery` folder after every edit.'
        )
        self.autoSaveAnnotToggle.setChecked(True)
        self.autoSaveAnnotToggle.setToolTip(autoSaveAnnotTooltip)
        autoSaveAnnotLabel = QLabel('Autosave annotations')
        autoSaveAnnotLabel.setToolTip(autoSaveAnnotTooltip)
        layout.addRow(autoSaveAnnotLabel, self.autoSaveAnnotToggle)
        
        self.autoSaveIntervalEditButton = widgets.editPushButton(
            flat=True, hoverable=True
        )
        self.autoSaveIntervalLabel = QLabel('Autosave interval')        
        self.autoSaveIntervalSetTooltip()
        layout.addRow(
            self.autoSaveIntervalLabel, self.autoSaveIntervalEditButton
        )
        
        self.autoSaveIntervalDialog = apps.AutoSaveIntervalDialog(parent=self)
        self.autoSaveIntervalDialog.setValues(*self.autoSaveIntevalValueUnit)
        
        self.ccaIntegrCheckerToggle = widgets.Toggle()
        ccaIntegrCheckerToggleTooltip = (
            'Toggle background cell cycle annotations integrity checker ON/OFF'
        )
        self.ccaIntegrCheckerToggle.setChecked(False)
        self.ccaIntegrCheckerToggle.setToolTip(ccaIntegrCheckerToggleTooltip)
        label = QLabel('Cc annot. checker')
        label.setToolTip(ccaIntegrCheckerToggleTooltip)
        layout.addRow(label, self.ccaIntegrCheckerToggle)
        if 'is_cca_integrity_checker_activated' in self.df_settings.index:
            idx = 'is_cca_integrity_checker_activated'
            val = int(self.df_settings.at[idx, 'value'])
            self.ccaIntegrCheckerToggle.setChecked(bool(val))
        
        self.annotLostObjsToggle = widgets.Toggle()
        annotLostObjsToggleTooltip = (
            'Toggle annotation of lost objects mode ON/OFF'
        )
        self.annotLostObjsToggle.setChecked(True)
        self.annotLostObjsToggle.setToolTip(annotLostObjsToggleTooltip)
        label = QLabel('Annot. lost objects')
        label.setToolTip(annotLostObjsToggleTooltip)
        layout.addRow(label, self.annotLostObjsToggle)

        self.realTimeTrackingToggle = widgets.Toggle()
        self.realTimeTrackingToggle.setChecked(True)
        self.realTimeTrackingToggle.setDisabled(True)
        label = QLabel('Real-time tracking')
        label.setDisabled(True)
        self.realTimeTrackingToggle.label = label
        layout.addRow(label, self.realTimeTrackingToggle)
        
        self.showAllContoursToggle = widgets.Toggle()
        showAllContoursTooltip = (
            'If active, all contours will be displayed, including inner contours'
            '(e.g. holes and sub-objects)'
        )
        self.showAllContoursToggle.setToolTip(showAllContoursTooltip)
        showAllContourLabel = QLabel('Show all contours')
        showAllContourLabel.setToolTip(showAllContoursTooltip)
        layout.addRow(showAllContourLabel, self.showAllContoursToggle)
        self.showAllContoursToggle.toggled.connect(
            self.showAllContoursToggled
        )

        # Font size
        self.fontSizeSpinBox = widgets.SpinBox()
        self.fontSizeSpinBox.setMinimum(1)
        self.fontSizeSpinBox.setMaximum(99)
        layout.addRow('Font size', self.fontSizeSpinBox) 
        savedFontSize = str(self.df_settings.at['fontSize', 'value'])
        if savedFontSize.find('pt') != -1:
            savedFontSize = savedFontSize[:-2]
        self.fontSize = int(savedFontSize)
        self.fontSizeSpinBox.setValue(self.fontSize)
        self.fontSizeSpinBox.editingFinished.connect(self.changeFontSize) 
        self.fontSizeSpinBox.sigUpClicked.connect(self.changeFontSize)
        self.fontSizeSpinBox.sigDownClicked.connect(self.changeFontSize)

        self.quickSettingsGroupbox.setLayout(layout)
        self.quickSettingsLayout.addWidget(self.quickSettingsGroupbox)
        self.quickSettingsLayout.addStretch(1)

    def gui_createBottomWidgets(self):
        self.annotOptionsCheckboxes = defaultdict(dict)
        self.gui_setupAnnotationOptionsLayout(0)
        self.gui_setupAnnotationOptionsLayout(1)

        # Frames scrollbar
        self.gui_createFramesNavigationWidgets()
        self.navigateScrollBar.setToolTip(
            'NOTE: The maximum frame number that can be visualized with this '
            'scrollbar\n'
            'is the last visited frame with the selected mode\n'
            '(see "Mode" selector on the top-right).\n\n'
            'If the scrollbar does not move it means that you never visited\n'
            'any frame with current mode.\n\n'
            'Note that the "Viewer" mode allows you to scroll ALL frames.'
        )

        # z-slice scrollbars
        self.gui_createZsliceNavigationWidgets()

        self.zSliceOverlay_SB = widgets.ScrollBar(Qt.Horizontal)
        _z_label = QLabel('Overlay z-slice  ')
        _z_label.setFont(gui_main_constants._font)
        _z_label.setDisabled(True)
        self.overlay_z_label = _z_label

        self.zProjOverlay_CB = widgets.ComboBox()
        self.zProjOverlay_CB.setFont(gui_main_constants._font)
        self.zProjOverlay_CB.addItems([
            'single z-slice', 'max z-projection', 'mean z-projection',
            'median z-proj.', 'same as above'
        ])
        self.zProjOverlay_CB.setCurrentIndex(4)
        self.zSliceOverlay_SB.setDisabled(True)

        self.img1BottomGroupbox = self.gui_createImg1BottomWidgets()

    def gui_createImg1BottomWidgets(self):
        bottomLeftLayout = QGridLayout()
        self.bottomLeftLayout = bottomLeftLayout
        container = QGroupBox('Navigate and annotate left image')

        row = 0
        bottomLeftLayout.addWidget(
            self.annotOptionsCheckboxes[0]['container'], row, 0, 1, 3
        )

        row += 1
        navWidgetsLayout = self.gui_createFramesNavigationLabelLayout()
        bottomLeftLayout.addLayout(
            navWidgetsLayout, row, 0, alignment=Qt.AlignRight
        )
        bottomLeftLayout.addWidget(self.navigateScrollBar, row, 1, 1, 2) 
        sp = self.navigateScrollBar.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.navigateScrollBar.setSizePolicy(sp)
        self.navSpinBox.connectValueChanged(self.navigateSpinboxValueChanged)
        self.navSpinBox.editingFinished.connect(
            self.navigateSpinboxEditingFinished
        )
        self.navSpinBox.sigUpClicked.connect(
            self.navigateSpinboxEditingFinished
        )
        self.navSpinBox.sigDownClicked.connect(
            self.navigateSpinboxEditingFinished
        )

        self.lastTrackedFrameLabel = QLabel()
        self.lastTrackedFrameLabel.setFont(gui_main_constants._font)
        bottomLeftLayout.addWidget(self.lastTrackedFrameLabel, row, 3)
        
        row += 1
        zSliceCheckboxLayout = self.gui_createZsliceCheckboxLayout()
        bottomLeftLayout.addLayout(
            zSliceCheckboxLayout, row, 0, alignment=Qt.AlignRight
        )
        bottomLeftLayout.addWidget(self.zSliceScrollBar, row, 1, 1, 2)
        bottomLeftLayout.addWidget(self.zProjComboBox, row, 3)
        bottomLeftLayout.addWidget(self.zProjLockViewButton, row, 4)
        bottomLeftLayout.addWidget(self.switchPlaneCombobox, row, 5)
        bottomLeftLayout.addWidget(self.launch3dViewerButton, row, 6)
        self.zSliceSpinbox.editingFinished.connect(self.zSliceScrollBarReleased)

        row += 1
        bottomLeftLayout.addWidget(
            self.overlay_z_label, row, 0, alignment=Qt.AlignRight
        )
        bottomLeftLayout.addWidget(self.zSliceOverlay_SB, row, 1, 1, 2)

        bottomLeftLayout.addWidget(self.zProjOverlay_CB, row, 3)

        row += 1
        self.alphaScrollbarRow = row

        bottomLeftLayout.setColumnStretch(0,0)
        bottomLeftLayout.setColumnStretch(1,3)
        bottomLeftLayout.setColumnStretch(2,0)

        container.setLayout(bottomLeftLayout)
        return container

    def gui_createImg2Widgets(self):
        bottomRightLayout = QVBoxLayout()
        self.rightBottomGroupbox = widgets.GroupBox(
            'Annotate right image independent of left image', 
            keyPressCallback=self.resetFocus
        )
        self.rightBottomGroupbox.setCheckable(True)
        self.rightBottomGroupbox.setChecked(False)
        self.rightBottomGroupbox.hide()

        self.rightImageFramesScrollbar = widgets.ScrollBarWithNumericControl(
            labelText='Frame n. '
        )
        self.rightImageFramesScrollbar.setVisible(False)
        
        bottomRightLayout.addWidget(self.annotOptionsCheckboxes[1]['container'])
        bottomRightLayout.addWidget(self.rightImageFramesScrollbar)
        bottomRightLayout.addStretch(1)

        self.rightBottomGroupbox.setLayout(bottomRightLayout)

        self.rightBottomGroupbox.toggled.connect(self.rightImageControlsToggled)

    def rightImageControlsToggled(self, checked):
        if self.isDataLoading:
            return

        linked = not checked
        for name, checkbox in self.annotOptionsCheckboxes[0].items():
            try:
                isChecked = checkbox.isChecked()
            except AttributeError:
                continue
            
            rightCheckbox = self.annotOptionsCheckboxes[1][name]
            if linked:
                rightCheckbox.setChecked(isChecked)
            
            checkbox.setLinkedCheckbox(rightCheckbox, linked)

    def setFocusGraphics(self):
        self.graphLayout.setFocus()

    def setFocusMain(self):
        # on macOS with Qt6 setFocus causes crashes. Disabled for now.
        return 

    def resetFocus(self):
        self.setFocusGraphics()
        self.setFocusMain()

    def gui_createBottomWidgetsToBottomLayout(self):
        # self.bottomDockWidget = QDockWidget(self)
        bottomScrollArea = widgets.ScrollArea(resizeVerticalOnShow=True)
        bottomScrollArea.sigLeaveEvent.connect(self.setFocusMain)
        bottomWidget = QWidget()
        bottomScrollAreaLayout = QVBoxLayout()
        self.bottomLayout = QHBoxLayout()
        self.bottomLayout.addLayout(self.quickSettingsLayout)
        self.bottomLayout.addStretch(1)
        self.bottomLayout.addWidget(self.img1BottomGroupbox)
        self.bottomLayout.addStretch(1)
        self.bottomLayout.addWidget(self.rightBottomGroupbox)
        self.bottomLayout.addStretch(1)   

        bottomScrollAreaLayout.addLayout(self.bottomLayout)
        bottomScrollAreaLayout.addStretch(1)

        bottomWidget.setLayout(bottomScrollAreaLayout)
        bottomScrollArea.setWidgetResizable(True)
        bottomScrollArea.setWidget(bottomWidget)
        self.bottomScrollArea = bottomScrollArea
        
        if 'bottom_sliders_zoom_perc' in self.df_settings.index:
            val = int(self.df_settings.at['bottom_sliders_zoom_perc', 'value'])
            zoom_perc = val
        else:
            zoom_perc = 100
        self.bottomLayoutContextMenu = QMenu('Bottom layout', self)
        zoomMenu = self.bottomLayoutContextMenu.addMenu('Zoom')
        actions = []
        self.bottomLayoutContextMenu.zoomActionGroup = QActionGroup(zoomMenu)
        for perc in np.arange(50, 151, 10):
            action = QAction(f'{perc}%', zoomMenu)
            action.setCheckable(True)
            if perc == zoom_perc:
                action.setChecked(True)
            action.toggled.connect(self.zoomBottomLayoutActionTriggered)
            actions.append(action)
            self.bottomLayoutContextMenu.zoomActionGroup.addAction(action)
        zoomMenu.addActions(actions)
        self.bottomLayoutContextMenu.addSeparator()

        self.editObjectTrackSettingsAction = (
                self.bottomLayoutContextMenu.addAction(
                    'Edit object track settings...'
            )
        )
        self.editObjectTrackSettingsAction.setCheckable(True)
        self.showFrameIndexAction = (
                self.bottomLayoutContextMenu.addAction(
                    'Show frame index (starts at 0)'
            )
        )
        self.showFrameIndexAction.setCheckable(True)
        self.showZsliceIndexAction = (
                self.bottomLayoutContextMenu.addAction(
                    'Show z-slice index (starts at 0)'
            )
        )
        self.showZsliceIndexAction.setCheckable(True)

        self.showFrameIndexAction.toggled.connect(self.showFrameIndexToggled)
        self.showZsliceIndexAction.toggled.connect(self.showZsliceIndexToggled)

        self.settingsMenu.insertAction(
            self.highLowResAction, self.showFrameIndexAction
        )
        self.settingsMenu.insertAction(
            self.highLowResAction, self.showZsliceIndexAction
        )

        self.bottomLayoutContextMenu.addSeparator()

        self.editObjectTrackSettingsAction.triggered.connect(
            self.setAnnotateObjTrackSettings 
        )
        resetAction = self.bottomLayoutContextMenu.addAction(
            'Reset default height'
        )
        resetAction.triggered.connect(self.resetBottomLayoutHeight)
        resetAnnotOptionsAction = self.bottomLayoutContextMenu.addAction(
            'Reset default annotation options'
        )
        resetAnnotOptionsAction.triggered.connect(
            self.resetDefaultAnnotationOptions
        )
        retainSpaceAction = self.bottomLayoutContextMenu.addAction(
            'Retain space of hidden sliders'
        )
        retainSpaceAction.setCheckable(True)
        if 'retain_space_hidden_sliders' in self.df_settings.index:
            retainSpaceChecked = (
                self.df_settings.at['retain_space_hidden_sliders', 'value']
                == 'Yes'
            )
        else:
            retainSpaceChecked = True
        retainSpaceAction.setChecked(retainSpaceChecked)
        retainSpaceAction.toggled.connect(self.retainSpaceSlidersToggled)
        self.retainSpaceSlidersAction = retainSpaceAction
        self.setBottomLayoutStretch()

    def gui_resetBottomLayoutHeight(self):
        self.h = self.defaultWidgetHeightBottomLayout
        self.checkBoxesHeight = 14
        self.fontPixelSize = 11
        self.resizeSlidersArea()

    def gui_createGraphicsPlots(self):
        self.graphLayout = pg.GraphicsLayoutWidget()
        # pyqtgraph ignores high-frequency mouse moves --> make sure the
        # last one is delivered (otherwise the hover position, e.g., of
        # the mirrored cursors, is stale when the mouse stops)
        self.mouseMoveTrailingEdgeFilter = widgets.MouseMoveTrailingEdgeFilter(
            self.graphLayout
        )
        if self.invertBwAction.isChecked():
            self.graphLayout.setBackground(graphLayoutBkgrColor)
            self.titleColor = 'black'
        else:
            self.graphLayout.setBackground(darkBkgrColor)
            self.titleColor = 'white'

        self.lutItemsLayout = self.graphLayout.addLayout(row=1, col=0)
        # self.lutItemsLayout.setBorder('w')

        # Left plot
        self.ax1 = widgets.MainPlotItem(showWelcomeText=True, ax_number=0)
        self.ax1.invertY(True)
        self.ax1.setAspectLocked(True)
        self.ax1.hideAxis('bottom')
        self.ax1.hideAxis('left')
        self.plotsCol = 1
        self.graphLayout.addItem(self.ax1, row=1, col=1)

        # Right plot
        self.ax2 = widgets.MainPlotItem(ax_number=1)
        self.ax2.setAspectLocked(True)
        self.ax2.invertY(True)
        self.ax2.hideAxis('bottom')
        self.ax2.hideAxis('left')
        # self.currentFrameLabelItem = pg.LabelItem(
        #     color=self.titleColor, size='13px'
        # )
        self.graphLayout.addItem(self.ax2, row=1, col=2)

    def gui_addGraphicsItems(self):
        # Auto image adjustment button
        proxy = QGraphicsProxyWidget()
        equalizeHistPushButton = QPushButton("Enhance contrast")
        widthHint = equalizeHistPushButton.sizeHint().width()
        equalizeHistPushButton.setMaximumWidth(widthHint)
        equalizeHistPushButton.setCheckable(True)
        if not self.invertBwAction.isChecked():
            equalizeHistPushButton.setStyleSheet(
                'QPushButton {background-color: #282828; color: #F0F0F0;}'
            )
        self.equalizeHistPushButton = equalizeHistPushButton
        proxy.setWidget(equalizeHistPushButton)
        self.graphLayout.addItem(proxy, row=0, col=0)
        self.equalizeHistPushButton = equalizeHistPushButton

        # Left image histogram
        self.imgGrad = widgets.myHistogramLUTitem(parent=self, name='image')
        self.imgGrad.restoreState(self.df_settings)
        self.lutItemsLayout.addItem(self.imgGrad, row=0, col=0)
        for action in self.imgGrad.rescaleActionGroup.actions():
            if action.text() == self.defaultRescaleIntensHow:
                action.setChecked(True)
            self.rescaleIntensMenu.addAction(action)
        
        # Colormap gradient widget
        self.labelsGrad = widgets.labelsGradientWidget(parent=self)
        try:
            stateFound = self.labelsGrad.restoreState(self.df_settings)
        except Exception as e:
            self.logger.exception(traceback.format_exc())
            print('======================================')
            self.logger.info(
                'Failed to restore previously used colormap. '
                'Using default colormap "viridis"'
            )
            self.labelsGrad.item.loadPreset('viridis')
        
        # Add actions to imgGrad gradient item
        self.imgGrad.gradient.menu.addAction(
            self.labelsGrad.showLabelsImgAction
        )
        self.imgGrad.gradient.menu.addAction(
            self.labelsGrad.showRightImgAction
        )
        self.imgGrad.gradient.menu.addAction(
            self.labelsGrad.showNextFrameAction
        )
        
        self.imgGrad.gradient.menu.addSeparator()
        
        self.imgGrad.gradient.menu.addMenu(self.exportMenu)            
        
        # Add actions to view menu
        self.viewMenu.addAction(self.labelsGrad.showLabelsImgAction)
        self.viewMenu.addAction(self.labelsGrad.showRightImgAction)
        
        self.viewMenu.addAction(self.launch3dViewerAction)
        
        # Right image histogram
        self.imgGradRight = widgets.baseHistogramLUTitem(
            name='image', parent=self, gradientPosition='left'
        )
        self.imgGradRight.gradient.menu.addAction(
            self.labelsGrad.showLabelsImgAction
        )
        self.imgGradRight.gradient.menu.addAction(
            self.labelsGrad.showRightImgAction
        )
        self.imgGradRight.gradient.menu.addAction(
            self.labelsGrad.showNextFrameAction
        )
        
        self.imgGrad.setChildLutItem(self.imgGradRight)

        # Title
        self.titleLabel = pg.LabelItem(
            justify='center', color=self.titleColor, size='14pt'
        )
        self.titleLabel.item.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.titleLabel.item.linkActivated.connect(
            self.titleLabelIDLinkActivated
        )
        self.titleLabel.item.linkHovered.connect(self.titleLabelIDLinkHovered)
        self.titleIDsPopup = QLabel(flags=Qt.Tool | Qt.FramelessWindowHint)
        self.titleIDsPopup.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.titleIDsPopup.linkActivated.connect(
            self.titleLabelIDLinkActivated
        )
        self.titleIDsPopup.linkHovered.connect(
            self.titleIDsPopupLinkHovered
        )
        self.titleIDsPopup.setStyleSheet(
            'QLabel { background-color: palette(window); '
            'border: 1px solid palette(mid); padding: 6px; }'
        )
        self.titleIDsPopupFilter = TitleIDsPopupEventFilter(
            self._keepTitleIDsPopupOpen,
            self._scheduleTitleIDsPopupHide,
            parent=self,
        )
        self.titleIDsPopup.installEventFilter(self.titleIDsPopupFilter)
        self.titleIDsPopupHideTimer = QTimer(self)
        self.titleIDsPopupHideTimer.setSingleShot(True)
        self.titleIDsPopupHideTimer.timeout.connect(self.titleIDsPopup.hide)
        self._titleIDsPopupHtmlByLink = {}
        self.fireworksOverlay = widgets.FireworksOverlay(self)
        self.graphLayout.addItem(self.titleLabel, row=0, col=1, colspan=2)        

    def gui_createPlotItems(self):
        if 'textIDsColor' in self.df_settings.index:
            rgbString = self.df_settings.at['textIDsColor', 'value']
            r, g, b = colors.rgb_str_to_values(rgbString)
            self.gui_createTextAnnotColors(r, g, b, custom=True)
            self.textIDsColorButton.setColor((r, g, b))
        else:
            self.gui_createTextAnnotColors(0,0,0, custom=False)

        if 'labels_text_color' in self.df_settings.index:
            rgbString = self.df_settings.at['labels_text_color', 'value']
            r, g, b = colors.rgb_str_to_values(rgbString)
            self.ax2_textColor = (r, g, b)
        else:
            self.ax2_textColor = (255, 0, 0)
        
        self.emptyLab = np.zeros((2,2), dtype=np.uint8)

        # Right image item linked to left
        self.rightImageItem = widgets.ChildImageItem(
            linkedScrollbar=self.rightImageFramesScrollbar
        )
        self.imgGradRight.setImageItem(self.rightImageItem)   
        self.ax2.addItem(self.rightImageItem)
        
        # Left image
        self.img1 = widgets.ParentImageItem(
            linkedImageItem=self.rightImageItem,
            activatingActions=(
                self.labelsGrad.showRightImgAction,
                self.labelsGrad.showNextFrameAction
            )
        )
        self.imgGrad.setImageItem(self.img1)
        self.img1.lutItem = self.imgGrad
        self.imgGrad.sigRescaleIntes.connect(self.rescaleIntensitiesLut)
        self.ax1.addBaseImageItem(self.img1)
        
        # RGBA image for true transparency mode
        self.rgbaImg1 = pg.ImageItem()
        
        # self.rgbaImg1.setImage(self.emptyLab)

        # Right image
        self.img2 = widgets.labImageItem()
        self.ax2.addItem(self.img2)

        self.topLayerItems = []
        self.topLayerItemsRight = []

        self.gui_createContourPens()
        self.gui_createMothBudLinePens()

        self.eraserCirclePen = pg.mkPen(width=1.5, color='r')
        
        # Temporary line item connecting bud to new mother
        self.BudMothTempLine = pg.PlotDataItem(pen=self.NewBudMoth_Pen)
        self.topLayerItems.append(self.BudMothTempLine)
        
        # Temporary line item connecting objects to merge
        self.mergeObjsTempLine = widgets.PlotCurveItem(pen=self.redDashLinePen)
        self.topLayerItems.append(self.mergeObjsTempLine)

        # Overlay segm. masks item
        self.labelsLayerImg1 = widgets.BaseLabelsImageItem()
        self.ax1.addItem(self.labelsLayerImg1)

        self.labelsLayerRightImg = widgets.BaseLabelsImageItem()
        self.ax2.addItem(self.labelsLayerRightImg)

        # Red/green border rect item
        self.GreenLinePen = pg.mkPen(color='g', width=2)
        self.RedLinePen = pg.mkPen(color='r', width=2)
        self.ax1BorderLine = pg.PlotDataItem()
        self.topLayerItems.append(self.ax1BorderLine)
        self.ax2BorderLine = pg.PlotDataItem(pen=pg.mkPen(color='r', width=2))
        self.topLayerItems.append(self.ax2BorderLine)

        # Brush/Eraser/Wand.. layer item
        # Labels image item --> `setImage` does not auto-level (the levels
        # are set by the left item, see `ParentImageItem.setLevels`)
        self.tempLayerRightImage = widgets.BaseLabelsImageItem()
        self.tempLayerImg1 = widgets.ParentImageItem(
            linkedImageItem=self.tempLayerRightImage,
            activatingActions=(self.labelsGrad.showRightImgAction, )
        )
        # Transparent until `initTempLayerBrush` sets the brush color (e.g.,
        # `clearTempBrushImage` in contours mode). Note that the linked right
        # image item does not inherit the lut
        transparentLut = np.zeros((2, 4), dtype=np.uint8)
        self.tempLayerImg1.setLookupTable(transparentLut)
        self.tempLayerRightImage.setLookupTable(transparentLut)
        self.topLayerItems.append(self.tempLayerImg1)
        self.topLayerItemsRight.append(self.tempLayerRightImage)

        # Brush strokes prompts overlay (1 = background, 2 = foreground)
        self.brushStrokesLayerImg1 = widgets.BaseLabelsImageItem()
        self.brushStrokesLayerImg1.setLookupTable(np.array(
            [[0, 0, 0, 0], [60, 160, 255, 255], [255, 80, 80, 255]],
            dtype=np.uint8
        ))
        self.brushStrokesLayerImg1.setLevels([0, 3])
        self.brushStrokesLayerImg1.setOpacity(0.55)
        self.brushStrokesLayerImg1.setVisible(False)
        self.topLayerItems.append(self.brushStrokesLayerImg1)

        # Cells selected to be added to the brush strokes (same look as the
        # "Keep IDs" selection, see `setBrushStrokesSelectedCellsLayerImage`)
        self.brushStrokesSelectedCellsLayerImg1 = pg.ImageItem()
        self.brushStrokesSelectedCellsLayerImg1.setOpacity(0.6)
        self.brushStrokesSelectedCellsLayerImg1.setVisible(False)
        self.topLayerItems.append(self.brushStrokesSelectedCellsLayerImg1)

        # Highlighted ID layer items
        self.highLightIDLayerImg1 = pg.ImageItem()
        self.topLayerItems.append(self.highLightIDLayerImg1)  

        # Highlighted ID layer items
        self.highLightIDLayerRightImage = pg.ImageItem()
        self.topLayerItemsRight.append(self.highLightIDLayerRightImage)

        # Keep IDs temp layers
        # Labels image item --> `setImage` does not auto-level (colors of
        # the IDs must not depend on the max kept ID)
        self.keepIDsTempLayerRight = widgets.BaseLabelsImageItem()
        self.keepIDsTempLayerLeft = widgets.ParentImageItem(
            linkedImageItem=self.keepIDsTempLayerRight,
            activatingActions=(self.labelsGrad.showRightImgAction, )
        )
        self.topLayerItems.append(self.keepIDsTempLayerLeft)
        self.topLayerItemsRight.append(self.keepIDsTempLayerRight)

        # Searched ID contour
        self.searchedIDitemRight = pg.ScatterPlotItem()
        self.searchedIDitemRight.setData(
            [], [], symbol='s', pxMode=False, size=1,
            brush=pg.mkBrush(color=(255,0,0,150)),
            pen=pg.mkPen(width=2, color='r'), tip=None
        )
        self.searchedIDitemLeft = pg.ScatterPlotItem()
        self.searchedIDitemLeft.setData(
            [], [], symbol='s', pxMode=False, size=1,
            brush=pg.mkBrush(color=(255,0,0,150)),
            pen=pg.mkPen(width=2, color='r'), tip=None
        )
        self.topLayerItems.append(self.searchedIDitemLeft)
        self.topLayerItemsRight.append(self.searchedIDitemRight)

        
        # Brush circle img1
        self.ax1_BrushCircle = pg.ScatterPlotItem()
        self.ax1_BrushCircle.setData(
            [], [], symbol='o', pxMode=False,
            brush=pg.mkBrush((255,255,255,50)),
            pen=pg.mkPen(width=2), tip=None
        )
        self.topLayerItems.append(self.ax1_BrushCircle)

        # Eraser circle img1
        self.ax1_EraserCircle = pg.ScatterPlotItem()
        self.ax1_EraserCircle.setData(
            [], [], symbol='o', pxMode=False,
            brush=None, pen=self.eraserCirclePen, tip=None
        )
        self.topLayerItems.append(self.ax1_EraserCircle)

        self.ax1_EraserX = pg.ScatterPlotItem()
        self.ax1_EraserX.setData(
            [], [], symbol='x', pxMode=False, size=3,
            brush=pg.mkBrush(color=(255,0,0,50)),
            pen=pg.mkPen(width=1, color='r'), tip=None
        )
        self.topLayerItems.append(self.ax1_EraserX)

        # Brush circle img1
        self.labelRoiCircItemLeft = widgets.LabelRoiCircularItem()
        self.labelRoiCircItemLeft.cleared = False
        self.labelRoiCircItemLeft.setData(
            [], [], symbol='o', pxMode=False,
            brush=pg.mkBrush(color=(255,0,0,0)),
            pen=pg.mkPen(color='r', width=2), tip=None
        )
        self.labelRoiCircItemRight = widgets.LabelRoiCircularItem()
        self.labelRoiCircItemRight.cleared = False
        self.labelRoiCircItemRight.setData(
            [], [], symbol='o', pxMode=False,
            brush=pg.mkBrush(color=(255,0,0,0)),
            pen=pg.mkPen(color='r', width=2), tip=None
        )
        self.topLayerItems.append(self.labelRoiCircItemLeft)
        self.topLayerItemsRight.append(self.labelRoiCircItemRight)
        
        self.ax1_binnedIDs_ScatterPlot = widgets.BaseScatterPlotItem()
        self.ax1_binnedIDs_ScatterPlot.setData(
            [], [], symbol='t', pxMode=False,
            brush=pg.mkBrush((255,0,0,50)), size=15,
            pen=pg.mkPen(width=3, color='r'), tip=None
        )
        self.topLayerItems.append(self.ax1_binnedIDs_ScatterPlot)
        
        self.ax1_ripIDs_ScatterPlot = widgets.BaseScatterPlotItem()
        self.ax1_ripIDs_ScatterPlot.setData(
            [], [], symbol='x', pxMode=False,
            brush=pg.mkBrush((255,0,0,50)), size=15,
            pen=pg.mkPen(width=2, color='r'), tip=None
        )
        self.topLayerItems.append(self.ax1_ripIDs_ScatterPlot)

        # Ruler plotItem and scatterItem
        rulerPen = pg.mkPen(color='r', style=Qt.DashLine, width=2)
        self.ax1_rulerPlotItem = widgets.RulerPlotItem(pen=rulerPen)
        self.ax1_rulerAnchorsItem = pg.ScatterPlotItem(
            symbol='o', size=9,
            brush=pg.mkBrush((255,0,0,50)),
            pen=pg.mkPen((255,0,0), width=2), tip=None
        )
        self.topLayerItems.append(self.ax1_rulerPlotItem)
        self.topLayerItems.append(self.ax1_rulerPlotItem.labelItem)
        self.topLayerItems.append(self.ax1_rulerAnchorsItem)

        # Start point of polyline roi
        self.ax1_point_ScatterPlot = pg.ScatterPlotItem()
        self.ax1_point_ScatterPlot.setData(
            [], [], symbol='o', pxMode=False, size=3,
            pen=pg.mkPen(width=2, color='r'),
            brush=pg.mkBrush((255,0,0,50)), tip=None
        )
        self.topLayerItems.append(self.ax1_point_ScatterPlot)

        # Experimental: scatter plot to add a point marker
        self.startPointPolyLineItem = pg.ScatterPlotItem()
        self.startPointPolyLineItem.setData(
            [], [], symbol='o', size=9,
            pen=pg.mkPen(width=2, color='r'),
            brush=pg.mkBrush((255,0,0,50)),
            hoverable=True, hoverBrush=pg.mkBrush((255,0,0,255)), tip=None
        )
        self.topLayerItems.append(self.startPointPolyLineItem)

        # Eraser circle img2
        self.ax2_EraserCircle = pg.ScatterPlotItem()
        self.ax2_EraserCircle.setData(
            [], [], symbol='o', pxMode=False, brush=None,
            pen=self.eraserCirclePen, tip=None
        )
        self.ax2.addItem(self.ax2_EraserCircle)
        self.ax2_EraserX = pg.ScatterPlotItem()
        self.ax2_EraserX.setData(
            [], [], symbol='x', pxMode=False, size=3,
            brush=pg.mkBrush(color=(255,0,0,50)),
            pen=pg.mkPen(width=1.5, color='r')
        )
        self.ax2.addItem(self.ax2_EraserX)

        # Brush circle img2
        self.ax2_BrushCirclePen = pg.mkPen(width=2)
        self.ax2_BrushCircleBrush = pg.mkBrush((255,255,255,50))
        self.ax2_BrushCircle = pg.ScatterPlotItem()
        self.ax2_BrushCircle.setData(
            [], [], symbol='o', pxMode=False,
            brush=self.ax2_BrushCircleBrush,
            pen=self.ax2_BrushCirclePen, tip=None
        )
        self.ax2.addItem(self.ax2_BrushCircle)

        # Annotated metadata markers (ScatterPlotItem)
        self.ax2_binnedIDs_ScatterPlot = widgets.BaseScatterPlotItem()
        self.ax2_binnedIDs_ScatterPlot.setData(
            [], [], symbol='t', pxMode=False,
            brush=pg.mkBrush((255,0,0,50)), size=15,
            pen=pg.mkPen(width=3, color='r'), tip=None
        )
        self.ax2.addItem(self.ax2_binnedIDs_ScatterPlot)
        
        self.ax2_ripIDs_ScatterPlot = widgets.BaseScatterPlotItem()
        self.ax2_ripIDs_ScatterPlot.setData(
            [], [], symbol='x', pxMode=False,
            brush=pg.mkBrush((255,0,0,50)), size=15,
            pen=pg.mkPen(width=2, color='r'), tip=None
        )
        self.ax2.addItem(self.ax2_ripIDs_ScatterPlot)

        self.freeRoiItem = widgets.PlotCurveItem(
            pen=pg.mkPen(color='r', width=2)
        )
        self.topLayerItems.append(self.freeRoiItem)
        
        self.warnPairingItem = widgets.PlotCurveItem(
            pen=pg.mkPen(color='r', width=5, style=Qt.DashLine),
            pxMode=False
        )
        self.topLayerItems.append(self.warnPairingItem)
        
        self.exportMaskImageItem = pg.ImageItem()

        self.ghostContourItemLeft = widgets.GhostContourItem(self.ax1)
        self.ghostContourItemRight = widgets.GhostContourItem(self.ax2)

        self.ghostMaskItemLeft = widgets.GhostMaskItem(self.ax1)
        self.ghostMaskItemRight = widgets.GhostMaskItem(self.ax2)
        
        self.manualBackgroundObjItem = widgets.GhostContourItem(
            self.ax1, penColor='r', textColor='r'
        )
        self.manualBackgroundImageItem = pg.ImageItem()

    def gui_createLabelRoiItem(self):
        Y, X = self.currentLab2D.shape
        # Label ROI rectangle
        pen = pg.mkPen('r', width=3)
        self.labelRoiItem = widgets.ROI(
            (0,0), (0,0),
            maxBounds=QRectF(QRect(0,0,X,Y)),
            scaleSnap=True,
            translateSnap=True,
            pen=pen, hoverPen=pen
        )

        posData = self.data[self.pos_i]
        if self.labelRoiZdepthSpinbox.value() == 0:
            self.labelRoiZdepthSpinbox.setValue(posData.SizeZ)
        self.labelRoiZdepthSpinbox.setMaximum(posData.SizeZ+1)

    def _gui_createGraphicsItems(self):
        for _posData in self.data:
            _posData.allData_li = [None]*_posData.SizeT
            
        posData = self.data[self.pos_i]

        allIDs, posData = core.count_objects_and_init_rps(
            posData, self.logger.info)

        self.logger.info(f'Creating graphical items...')

        self.ax1_contoursImageItem = pg.ImageItem()
        
        self.ax1_lostObjImageItem = pg.ImageItem()
        self.ax2_lostObjImageItem = pg.ImageItem()
        
        self.ax1_lostTrackedObjImageItem = pg.ImageItem()
        self.ax2_lostTrackedObjImageItem = pg.ImageItem()
        
        self.ax1_oldMothBudLinesItem = pg.ScatterPlotItem(
            symbol='s', pxMode=False, brush=self.oldMothBudLineBrush,
            size=self.mothBudLineWeight, pen=None
        )
        self.ax1_newMothBudLinesItem = pg.ScatterPlotItem(
            symbol='s', pxMode=False, brush=self.newMothBudLineBrush,
            size=self.mothBudLineWeight, pen=None
        )
        self.ax1_movementAgainstPrevLinesItem = pg.ScatterPlotItem(
            symbol='s', pxMode=False, brush=self.oldMothBudLineBrush,
            size=self.mothBudLineWeight, pen=None
        )
        self.ax1_trackerMovementAgainstPrevLinesItem = pg.ScatterPlotItem(
            symbol='s', pxMode=False, brush=self.oldMothBudLineBrush,
            size=self.mothBudLineWeight, pen=None
        )
        self.tracksPlotItems = dict()
        self.ax1_lostObjScatterItem = self.gui_getLostObjScatterItem()
        self.yellowContourScatterItem = self.gui_getLostObjScatterItem()
        self.ax1_movementAgainstPrevScatterItem = self.gui_getLostObjScatterItem()

        self.ax1_lostTrackedScatterItem = self.gui_getTrackedLostObjScatterItem()
        self.greenContourScatterItem = self.gui_getTrackedLostObjScatterItem()

        brush = pg.mkBrush((0,255,0,200))
        pen = pg.mkPen('g', width=1)
        self.ccaFailedScatterItem = pg.ScatterPlotItem(
            size=self.contLineWeight+1, pen=pen, 
            brush=brush, pxMode=False, symbol='s'
        )

        self.ax2_contoursImageItem = pg.ImageItem()
        self.ax2_oldMothBudLinesItem = pg.ScatterPlotItem(
            symbol='s', pxMode=False, brush=self.oldMothBudLineBrush,
            size=self.mothBudLineWeight, pen=None
        )
        self.ax2_newMothBudLinesItem = pg.ScatterPlotItem(
            symbol='s', pxMode=False, brush=self.newMothBudLineBrush,
            size=self.mothBudLineWeight, pen=None
        )
        self.ax2_movementAgainstPrevLinesItem = pg.ScatterPlotItem(
            symbol='s', pxMode=False, brush=self.oldMothBudLineBrush,
            size=self.mothBudLineWeight, pen=None
        )
        self.ax2_lostObjScatterItem = self.gui_getLostObjScatterItem()
        self.ax2_lostTrackedScatterItem = self.gui_getTrackedLostObjScatterItem()
        self.ax2_movementAgainstPrevScatterItem = self.gui_getLostObjScatterItem()
        
        self.gui_createTextAnnotItems(allIDs)
        self.gui_setTextAnnotColors()

        self.progressWin.mainPbar.setMaximum(0)
        self.gui_addOverlayLayerItems()
        self.gui_addTopLayerItems()

        self.gui_addCreatedAxesItems()
        self.gui_add_ax_cursors()
        
        self.progressWin.workerFinished = True
        self.progressWin.close()
        self.progressWin = None

        self.loadingDataCompleted()

    def gui_addTopLayerItems(self):
        for item in self.topLayerItems:
            self.ax1.addItem(item)
        
        for item in self.topLayerItemsRight:
            self.ax2.addItem(item)

        # self.ax2.addItem(self.currentFrameLabelItem)

    def gui_createGraphicsItems(self):
        # Create enough PlotDataItems and LabelItems to draw contours and IDs.
        self.progressWin = apps.QDialogWorkerProgress(
            title='Creating axes items', parent=self,
            pbarDesc='Creating axes items (see progress in the terminal)...'
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(0)

        QTimer.singleShot(50, self._gui_createGraphicsItems)

    def gui_connectGraphicsEvents(self):
        self.img1.hoverEvent = self.gui_hoverEventImg1
        self.img2.hoverEvent = self.gui_hoverEventImg2
        self.img1.mousePressEvent = self.gui_mousePressEventImg1
        self.img1.mouseMoveEvent = self.gui_mouseDragEventImg1
        self.img1.mouseReleaseEvent = self.gui_mouseReleaseEventImg1
        self.img2.mousePressEvent = self.gui_mousePressEventImg2
        self.img2.mouseMoveEvent = self.gui_mouseDragEventImg2
        self.img2.mouseReleaseEvent = self.gui_mouseReleaseEventImg2
        self.rightImageItem.mousePressEvent = self.gui_mousePressRightImage
        self.rightImageItem.mouseMoveEvent = self.gui_mouseDragRightImage
        self.rightImageItem.mouseReleaseEvent = self.gui_mouseReleaseRightImage
        self.rightImageItem.hoverEvent = self.gui_hoverEventRightImage
        # self.imgGrad.gradient.showMenu = self.gui_gradientContextMenuEvent
        self.imgGradRight.gradient.showMenu = self.gui_rightImageShowContextMenu
        # self.imgGrad.vb.contextMenuEvent = self.gui_gradientContextMenuEvent
        self.ax1.sigRangeChanged.connect(self.viewRangeChanged)

    def gui_initImg1BottomWidgets(self):
        self.zSliceScrollBar.hide()
        self.zProjComboBox.hide()
        self.zProjLockViewButton.hide()
        self.zSliceOverlay_SB.hide()
        self.zProjOverlay_CB.hide()
        self.overlay_z_label.hide()
        self.zSliceCheckbox.hide()
        self.zSliceSpinbox.hide()
        self.SizeZlabel.hide()

    def gui_addCreatedAxesItems(self):
        self.ax1.addItem(self.ax1_contoursImageItem)
        self.ax1.addItem(self.ax1_lostObjImageItem)
        self.ax1.addItem(self.ax1_lostTrackedObjImageItem)
        self.ax1.addItem(self.ax1_oldMothBudLinesItem)
        self.ax1.addItem(self.ax1_newMothBudLinesItem)
        self.ax1.addItem(self.ax1_lostObjScatterItem)
        self.ax1.addItem(self.ax1_lostTrackedScatterItem)
        self.ax1.addItem(self.ccaFailedScatterItem)
        self.ax1.addItem(self.yellowContourScatterItem)
        self.ax1.addItem(self.ax1_movementAgainstPrevLinesItem)
        self.ax1.addItem(self.ax1_trackerMovementAgainstPrevLinesItem)
        self.ax1_trackerMovementAgainstPrevLinesItem.setZValue(1) # make sure it is on top of the other lines
        self.ax1.addItem(self.ax1_movementAgainstPrevScatterItem)

        self.ax2.addItem(self.ax2_contoursImageItem)
        self.ax2.addItem(self.ax2_lostObjImageItem)
        self.ax2.addItem(self.ax2_lostTrackedObjImageItem)
        self.ax2.addItem(self.ax2_oldMothBudLinesItem)
        self.ax2.addItem(self.ax2_newMothBudLinesItem)
        self.ax2.addItem(self.ax2_lostObjScatterItem)
        self.ax2.addItem(self.ax2_movementAgainstPrevLinesItem)
        self.ax2.addItem(self.ax2_movementAgainstPrevScatterItem)

        self.textAnnot[0].addToPlotItem(self.ax1)
        self.textAnnot[1].addToPlotItem(self.ax2)
        
        self.ax1.addItem(self.exportMaskImageItem)
        self.ax1.exportMaskImageItem = self.exportMaskImageItem

    def gui_raiseBottomLayoutContextMenu(self, event):
        try:
            # Convert QPointF to QPoint
            self.bottomLayoutContextMenu.popup(event.globalPos().toPoint())
        except AttributeError:
            self.bottomLayoutContextMenu.popup(event.globalPos())

    def zoomBottomLayoutActionTriggered(self, checked):
        if not checked:
            return
        perc = int(re.findall(r'(\d+)%', self.sender().text())[0])
        if perc != 100:
            fontSizeFactor = perc/100
            heightFactor = perc/100
            self.resizeSlidersArea(
                fontSizeFactor=fontSizeFactor, heightFactor=heightFactor
            )
        else:
            self.gui_resetBottomLayoutHeight()
        self.df_settings.at['bottom_sliders_zoom_perc', 'value'] = perc
        self.df_settings.to_csv(self.settings_csv_path)
        QTimer.singleShot(150, self.resizeGui)

    def showFrameIndexToggled(self, checked):
        posData = self.data[self.pos_i]
        start_value = 0 if checked else 1
        self.navSpinBox.setStartValue(start_value)
        self.rightImageFramesScrollbar.setStartValue(start_value)

        correction = start_value - self.navSpinBox.minimum()
        self.navSizeLabel.setText(f'/{posData.SizeT + correction}')

        value = 'Yes' if checked else 'No'
        self.df_settings.at['showFrameIndex', 'value'] = value
        self.df_settings.to_csv(self.settings_csv_path)

        if start_value:
            self.t_label.setText('Frame n. ')
        else:
            self.t_label.setText('Frame idx')

    def showZsliceIndexToggled(self, checked):
        start_value = 0 if checked else 1
        self.zSliceSpinbox.setStartValue(start_value)
        self.editIDToolbar.setStartValue(start_value)

        correction = start_value - self.zSliceSpinbox.minimum()
        maximum = self.zSliceSpinbox.maximum()
        self.SizeZlabel.setText(f'/{maximum + correction}')

        value = 'Yes' if checked else 'No'
        self.df_settings.at['showZsliceIndex', 'value'] = value
        self.df_settings.to_csv(self.settings_csv_path)

        if not self.overlay_z_label.isVisible():
            return
        
        # 0-based slice drawn by the overlay (`getOlImg` keeps the overlay
        # scrollbar in sync with the main one in "same as above" mode),
        # displayed as `z + start_value` like the z-slice spinbox
        z = self.zSliceOverlay_SB.sliderPosition()
        posData = self.data[self.pos_i]
        self.overlay_z_label.setText(
            f'Overlay z-slice {z+start_value:02}/{posData.SizeZ+correction}'
        )

    def retainSpaceSlidersToggled(self, checked):
        if checked:
            self.df_settings.at['retain_space_hidden_sliders', 'value'] = 'Yes'
        else:
            self.df_settings.at['retain_space_hidden_sliders', 'value'] = 'No'
        self.df_settings.to_csv(self.settings_csv_path)
        if not self.zSliceScrollBar.isEnabled():
            retainSpaceZ = False
        else:
            retainSpaceZ = checked
        myutils.setRetainSizePolicy(self.zSliceScrollBar, retain=retainSpaceZ)
        myutils.setRetainSizePolicy(self.zProjComboBox, retain=retainSpaceZ)
        myutils.setRetainSizePolicy(self.zSliceOverlay_SB, retain=retainSpaceZ)
        myutils.setRetainSizePolicy(self.zProjOverlay_CB, retain=retainSpaceZ)
        myutils.setRetainSizePolicy(self.overlay_z_label, retain=retainSpaceZ)
        
        QTimer.singleShot(200, self.resizeGui)

    def resizeLeaveSpaceTerminalBelow(self):
        self.setWindowState(Qt.WindowMaximized)
        QTimer.singleShot(200, self._resizeLeaveSpaceTerminalBelow)

    def _resizeLeaveSpaceTerminalBelow(self):
        geometry = self.geometry()
        left = geometry.left()
        top = geometry.top()
        width = geometry.width()
        height = geometry.height()
        self.setGeometry(left, top+10, width, height-200)

    def resizeGui(self):
        self.ax1.vb.state['limits']['xRange'] = [None, None]
        self.ax1.vb.state['limits']['yRange'] = [None, None]
        self.autoRange()
        if self.ax1.getViewBox().state['limits']['xRange'][0] is not None:
            self.bottomScrollArea._resizeVertical()
            return
        (xmin, xmax), (ymin, ymax) = self.ax1.viewRange()
        maxYRange = int((ymax-ymin)*1.5)
        maxXRange = int((xmax-xmin)*1.5)
        self.ax1.setLimits(
            maxYRange=maxYRange, 
            maxXRange=maxXRange
        )
        self.bottomScrollArea._resizeVertical()
        self.setBottomLayoutHeight()

    def resetBottomLayoutHeight(self):
        self.bottomScrollArea._resizeVertical()
        self.resizeBottomLayoutLineReleased()
        self.resizeGui()

    def setVisible3DsegmWidgets(self):
        if not self.isSegm3D:
            self.interpZButton.setChecked(False)

        self.interpZButton.setVisible(self.isSegm3D)
        self.interpZButton.action.setVisible(self.isSegm3D)
        self.interpZConfirmAction.setVisible(self.isSegm3D)
        self.interpZButton.action.setDisabled(not self.isSegm3D)
        self.interpZButton.setDisabled(not self.isSegm3D)

        self.separateBudToolbar.setHowToPropagateZslicesVisible(self.isSegm3D)

    def resizeGuiAndAutoRange(self):
        self.resizeGui()
        QTimer.singleShot(200, self.autoRange)

    def setTwoImagesLayout(self, isTwoImages):
        self.isTwoImageLayout = isTwoImages
        if isTwoImages:
            self.graphLayout.removeItem(self.titleLabel)
            self.graphLayout.addItem(self.titleLabel, row=0, col=1, colspan=2)
            # self.mainLayout.setAlignment(self.bottomLayout, Qt.AlignLeft)
            self.ax2.show()
            self.ax2.vb.setYLink(self.ax1.vb)
            self.ax2.vb.setXLink(self.ax1.vb)
        else:
            self.graphLayout.removeItem(self.titleLabel)
            self.graphLayout.addItem(self.titleLabel, row=0, col=1)
            # self.mainLayout.setAlignment(self.bottomLayout, Qt.AlignCenter)  
            self.ax2.hide()
            oldLink = self.ax2.vb.linkedView(self.ax1.vb.YAxis)
            try:
                oldLink.sigYRangeChanged.disconnect()
                oldLink.sigXRangeChanged.disconnect()
            except TypeError:
                pass

    def showNextFrameImageItem(self, checked):
        self.rightImageFramesScrollbar.setVisible(checked)
        self.rightImageFramesScrollbar.setDisabled(not checked)
        self.setTwoImagesLayout(checked)
        if checked:
            self.df_settings.at['isNextFrameVisible', 'value'] = 'Yes'
            self.df_settings.at['isRightImageVisible', 'value'] = 'No'
            self.df_settings.at['isLabelsVisible', 'value'] = 'No'
            self.graphLayout.addItem(
                self.imgGradRight, row=1, col=self.plotsCol+2
            )
            self.storeCurrentAnnotOptions(1)
            self.rightBottomGroupbox.show()
            self.rightBottomGroupbox.setChecked(True)     
            self.annotDoNotAnntoateCheckbox(1).setChecked(True)   
            self.setDisabledAnnotCheckBoxes(1, True) 
            if not self.isDataLoading:
                self.updateAllImages()
        else:
            self.clearAx2Items()
            self.rightBottomGroupbox.hide()
            self.df_settings.at['isNextFrameVisible', 'value'] = 'No'
            try:
                self.graphLayout.removeItem(self.imgGradRight)
            except Exception:
                return
            self.rightImageItem.clear()
        
        self.df_settings.to_csv(self.settings_csv_path)
            
        QTimer.singleShot(300, self.resizeGui)

        self.setBottomLayoutStretch()    


    def showRightImageItem(self, checked):
        self.rightImageFramesScrollbar.setVisible(not checked)
        self.rightImageFramesScrollbar.setDisabled(checked)
        self.setTwoImagesLayout(checked)
        if checked:
            self.df_settings.at['isRightImageVisible', 'value'] = 'Yes'
            self.df_settings.at['isNextFrameVisible', 'value'] = 'No'
            self.df_settings.at['isLabelsVisible', 'value'] = 'No'
            self.graphLayout.addItem(
                self.imgGradRight, row=1, col=self.plotsCol+2
            )
            self.rightBottomGroupbox.show()
            self.setDisabledAnnotCheckBoxes(1, False)
            self.restoreAnnotOptions(1)
            if not self.isDataLoading:
                self.updateAllImages()
        else:
            self.clearAx2Items()
            self.rightBottomGroupbox.hide()
            self.df_settings.at['isRightImageVisible', 'value'] = 'No'
            try:
                self.graphLayout.removeItem(self.imgGradRight)
            except Exception:
                return
            self.rightImageItem.clear()
        
        self.df_settings.to_csv(self.settings_csv_path)
            
        QTimer.singleShot(300, self.resizeGui)

        self.setBottomLayoutStretch()    

    def showLabelImageItem(self, checked):
        self.rightImageFramesScrollbar.setVisible(not checked)
        self.rightImageFramesScrollbar.setDisabled(checked)
        self.setTwoImagesLayout(checked)
        self.setAnnotOptionsRightImageLabelsDisabled(checked)
        if checked:
            self.df_settings.at['isLabelsVisible', 'value'] = 'Yes'
            self.df_settings.at['isNextFrameVisible', 'value'] = 'No'
            self.df_settings.at['isRightImageVisible', 'value'] = 'No'
            self.rightBottomGroupbox.show()
            self.rightBottomGroupbox.setChecked(True)
            if not self.isDataLoading:
                self.updateAllImages()
        else:
            self.clearAx2Items()
            self.img2.clear()
            self.df_settings.at['isLabelsVisible', 'value'] = 'No'
            self.rightBottomGroupbox.hide()
            self.moveDelRoisToLeft()
        
        self.df_settings.to_csv(self.settings_csv_path)
        QTimer.singleShot(200, self.resizeGui)

        self.setBottomLayoutStretch()

    def setAnnotOptionsRightImageLabelsDisabled(self, disabled):
        self.annotContourCheckbox(1).setDisabled(disabled)
        self.annotOverlaySegmMaskCheckbox(1).setDisabled(disabled)
        if disabled:
            self.annotOverlaySegmMaskCheckbox(1).setChecked(False)
            self.annotIDsCheckbox(1).setChecked(True)

    def setBottomLayoutStretch(self):
        if (
            self.labelsGrad.showRightImgAction.isChecked()
            or self.labelsGrad.showNextFrameAction.isChecked()
        ):
            # Equally share space between the two control groupboxes
            self.bottomLayout.setStretch(1, 1)
            self.bottomLayout.setStretch(2, 7)
            self.bottomLayout.setStretch(3, 1)
            self.bottomLayout.setStretch(4, 7)
            self.bottomLayout.setStretch(5, 1)
        elif self.labelsGrad.showLabelsImgAction.isChecked():
            # Left control takes only left space
            self.bottomLayout.setStretch(1, 1)
            self.bottomLayout.setStretch(2, 7)
            self.bottomLayout.setStretch(3, 7)
            self.bottomLayout.setStretch(4, 1)
            self.bottomLayout.setStretch(5, 1)
        else:
            # Left control takes all the space
            self.bottomLayout.setStretch(1, 3)
            self.bottomLayout.setStretch(2, 14)
            self.bottomLayout.setStretch(3, 1)
            self.bottomLayout.setStretch(4, 1)
            self.bottomLayout.setStretch(5, 1)

    def resizeSlidersArea(self, fontSizeFactor=None, heightFactor=None):
        if heightFactor is None:
            self.newCheckBoxesHeight = self.checkBoxesHeight
            self.newHeight = self.h
        else:
            self.newHeight = round(self.h*heightFactor)
            self.newCheckBoxesHeight = round(self.checkBoxesHeight*heightFactor)
        
        if fontSizeFactor is None:
            newFontSize = self.fontPixelSize
        else:
            newFontSize = round(self.fontPixelSize*fontSizeFactor)
        newFont = QFont()
        newFont.setPixelSize(newFontSize)
        gui_main_constants._font = newFont
        # Read by mixin methods in other modules (e.g.
        # gui_overlay.addAlphaScrollbar) that cannot see gui.py's _font
        self.bottomLayoutFont = newFont
        self.zProjComboBox.setFont(newFont)
        self.t_label.setFont(newFont)
        self.zProjOverlay_CB.setFont(newFont)
        self.navSpinBox.setFont(newFont)
        self.zSliceSpinbox.setFont(newFont)
        self.SizeZlabel.setFont(newFont)
        self.navSizeLabel.setFont(newFont)
        self.overlay_z_label.setFont(newFont)
        self.img1BottomGroupbox.setFont(newFont)
        self.rightBottomGroupbox.setFont(newFont)
        try:
            self.img1.alphaScrollbar.label.setFont(newFont)
        except Exception as e:
            pass

        for ax in (0, 1):
            annotOptionsLayout = self.annotOptionsCheckboxes[ax]['layout']
            for i in range(annotOptionsLayout.count()):
                widget = annotOptionsLayout.itemAt(i).widget()
                widget.setFont(newFont)
        try:
            for channel, items in self.overlayLayersItems.items():
                alphaScrollbar = items[2]
                alphaScrollbar.label.setFont(newFont)
        except:
            pass
        QTimer.singleShot(100, self._resizeSlidersArea)

    def _resizeSlidersArea(self):
        self.navigateScrollBar.setFixedHeight(self.newHeight)
        self.zSliceScrollBar.setFixedHeight(self.newHeight)
        self.zSliceOverlay_SB.setFixedHeight(self.newHeight)
        self.zProjComboBox.setFixedHeight(self.newHeight)
        self.zProjOverlay_CB.setFixedHeight(self.newHeight)
        self.navSpinBox.setFixedHeight(self.newHeight)
        self.zSliceSpinbox.setFixedHeight(self.newHeight)
        try:
            self.img1.alphaScrollbar.setFixedHeight(self.newHeight)
        except Exception as e:
            pass
        try:
            for channel, items in self.overlayLayersItems.items():
                alphaScrollbar = items[2]
                alphaScrollbar.setFixedHeight(self.newHeight)
        except:
            pass
        checkBoxStyleSheet = (
            'QCheckBox::indicator {'
                f'width: {self.newCheckBoxesHeight}px;'
                f'height: {self.newCheckBoxesHeight}px'
            '}'
        )
        for ax in (0, 1):
            annotOptionsLayout = self.annotOptionsCheckboxes[ax]['layout']
            for i in range(annotOptionsLayout.count()):
                widget = annotOptionsLayout.itemAt(i).widget()
                if isinstance(widget, QCheckBox):
                    widget.setStyleSheet(checkBoxStyleSheet)

        self.zSliceCheckbox.setStyleSheet(checkBoxStyleSheet)
