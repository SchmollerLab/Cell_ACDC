"""Qt view adapter for image controls and bottom layout."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QAction,
    QActionGroup,
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QVBoxLayout,
    QWidget,
)

from cellacdc import widgets
from cellacdc.viewmodels.image_controls_viewmodel import ImageControlsViewModel

_font = QFont()
_font.setPixelSize(11)


class ImageControlsView:
    """Qt-facing adapter around image-control defaults and widgets."""

    def __init__(self, host, view_model: ImageControlsViewModel):
        object.__setattr__(self, 'host', host)
        object.__setattr__(self, 'view_model', view_model)

    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {'host', 'view_model'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def gui_createImg1Widgets(self):
        # Toggle contours/ID combobox
        self.drawIDsContComboBoxSegmItems = list(
            self.view_model.draw_ids_cont_combo_items()
        )
        self.drawIDsContComboBox = widgets.ComboBox()
        self.drawIDsContComboBox.setFont(_font)
        self.drawIDsContComboBox.addItems(self.drawIDsContComboBoxSegmItems)
        self.drawIDsContComboBox.setVisible(False)

        self.annotIDsCheckbox = widgets.CheckBox(
            'IDs', keyPressCallback=self.resetFocus)
        self.annotCcaInfoCheckbox = widgets.CheckBox(
            'Cell cycle info', keyPressCallback=self.resetFocus)
        self.annotNumZslicesCheckbox = widgets.CheckBox(
            'No. z-slices/object', keyPressCallback=self.resetFocus)

        self.annotContourCheckbox = widgets.CheckBox(
            'Contours', keyPressCallback=self.resetFocus)
        self.annotSegmMasksCheckbox = widgets.CheckBox(
            'Segm. masks', keyPressCallback=self.resetFocus)

        self.drawMothBudLinesCheckbox = widgets.CheckBox(
            'Only mother-daughter line', keyPressCallback=self.resetFocus
        )

        self.drawNothingCheckbox = widgets.CheckBox(
            'Do not annotate', keyPressCallback=self.resetFocus
        )

        self.annotOptionsWidget = QWidget()
        annotOptionsLayout = QHBoxLayout()

        # Show tree info checkbox
        self.showTreeInfoCheckbox = widgets.CheckBox(
            'Show tree info', keyPressCallback=self.resetFocus
        )
        self.showTreeInfoCheckbox.setFont(_font)
        sp = self.showTreeInfoCheckbox.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.showTreeInfoCheckbox.setSizePolicy(sp)
        self.showTreeInfoCheckbox.hide()

        annotOptionsLayout.addWidget(self.showTreeInfoCheckbox)
        annotOptionsLayout.addWidget(QLabel(' | '))
        annotOptionsLayout.addWidget(self.annotIDsCheckbox)
        annotOptionsLayout.addWidget(self.annotCcaInfoCheckbox)
        annotOptionsLayout.addWidget(self.drawMothBudLinesCheckbox)
        annotOptionsLayout.addWidget(self.annotNumZslicesCheckbox)
        annotOptionsLayout.addWidget(QLabel(' | '))
        annotOptionsLayout.addWidget(self.annotContourCheckbox)
        annotOptionsLayout.addWidget(self.annotSegmMasksCheckbox)
        annotOptionsLayout.addWidget(QLabel(' | '))
        annotOptionsLayout.addWidget(self.drawNothingCheckbox)
        annotOptionsLayout.addWidget(self.drawIDsContComboBox)
        self.annotOptionsLayout = annotOptionsLayout

        # Toggle highlight z+-1 objects combobox
        self.highlightZneighObjCheckbox = widgets.CheckBox(
            'Highlight objects in neighbouring z-slices',
            keyPressCallback=self.resetFocus
        )
        self.highlightZneighObjCheckbox.setFont(_font)
        self.highlightZneighObjCheckbox.hide()

        annotOptionsLayout.addWidget(self.highlightZneighObjCheckbox)
        self.annotOptionsWidget.setLayout(annotOptionsLayout)

        # Annotations options right image
        self.annotIDsCheckboxRight = widgets.CheckBox(
            'IDs', keyPressCallback=self.resetFocus)
        self.annotCcaInfoCheckboxRight = widgets.CheckBox(
            'Cell cycle info', keyPressCallback=self.resetFocus)
        self.annotNumZslicesCheckboxRight = widgets.CheckBox(
            'No. z-slices/object', keyPressCallback=self.resetFocus
        )

        self.annotContourCheckboxRight = widgets.CheckBox(
            'Contours', keyPressCallback=self.resetFocus)
        self.annotSegmMasksCheckboxRight = widgets.CheckBox(
            'Segm. masks', keyPressCallback=self.resetFocus)

        self.drawMothBudLinesCheckboxRight = widgets.CheckBox(
            'Only mother-daughter line', keyPressCallback=self.resetFocus
        )

        self.drawNothingCheckboxRight = widgets.CheckBox(
            'Do not annotate', keyPressCallback=self.resetFocus)

        self.annotOptionsWidgetRight = QWidget()
        annotOptionsLayoutRight = QHBoxLayout()

        annotOptionsLayoutRight.addWidget(QLabel('       '))
        annotOptionsLayoutRight.addWidget(QLabel(' | '))
        annotOptionsLayoutRight.addWidget(self.annotIDsCheckboxRight)
        annotOptionsLayoutRight.addWidget(self.annotCcaInfoCheckboxRight)
        annotOptionsLayoutRight.addWidget(self.drawMothBudLinesCheckboxRight)
        annotOptionsLayoutRight.addWidget(self.annotNumZslicesCheckboxRight)
        annotOptionsLayoutRight.addWidget(QLabel(' | '))
        annotOptionsLayoutRight.addWidget(self.annotContourCheckboxRight)
        annotOptionsLayoutRight.addWidget(self.annotSegmMasksCheckboxRight)
        annotOptionsLayoutRight.addWidget(QLabel(' | '))
        annotOptionsLayoutRight.addWidget(self.drawNothingCheckboxRight)
        self.annotOptionsLayoutRight = annotOptionsLayoutRight

        self.annotOptionsWidgetRight.setLayout(annotOptionsLayoutRight)

        # Frames scrollbar
        self.navigateScrollBar = widgets.navigateQScrollBar(Qt.Horizontal)
        self.navigateScrollBar.setDisabled(True)
        self.navigateScrollBar.setMinimum(1)
        self.navigateScrollBar.setMaximum(1)
        self.navigateScrollBar.setToolTip(
            'NOTE: The maximum frame number that can be visualized with this '
            'scrollbar\n'
            'is the last visited frame with the selected mode\n'
            '(see "Mode" selector on the top-right).\n\n'
            'If the scrollbar does not move it means that you never visited\n'
            'any frame with current mode.\n\n'
            'Note that the "Viewer" mode allows you to scroll ALL frames.'
        )
        t_label = QLabel('frame n.  ')
        t_label.setFont(_font)
        self.t_label = t_label

        # z-slice scrollbars
        self.zSliceScrollBar = widgets.linkedQScrollbar(Qt.Horizontal)

        self.zProjComboBox = widgets.ComboBox()
        self.zProjComboBox.setFont(_font)
        self.zProjComboBox.addItems(self.view_model.z_projection_options())
        self.zProjLockViewButton = widgets.LockPushButton()
        self.zProjLockViewButton.setCheckable(True)
        self.zProjLockViewButton.setToolTip(
            'If active, the selected z-slice view is applied to all frames'
        )
        self.zProjLockViewButton.hide()

        self.switchPlaneCombobox = widgets.SwitchPlaneCombobox()
        self.switchPlaneCombobox.setToolTip(
            'Switch viewed plane'
        )

        self.zSliceOverlay_SB = widgets.ScrollBar(Qt.Horizontal)
        _z_label = QLabel('Overlay z-slice  ')
        _z_label.setFont(_font)
        _z_label.setDisabled(True)
        self.overlay_z_label = _z_label

        self.zProjOverlay_CB = widgets.ComboBox()
        self.zProjOverlay_CB.setFont(_font)
        self.zProjOverlay_CB.addItems(
            self.view_model.overlay_z_projection_options()
        )
        self.zProjOverlay_CB.setCurrentIndex(4)
        self.zSliceOverlay_SB.setDisabled(True)

        self.img1BottomGroupbox = self.gui_getImg1BottomWidgets()

    def gui_getImg1BottomWidgets(self):
        bottomLeftLayout = QGridLayout()
        self.bottomLeftLayout = bottomLeftLayout
        container = QGroupBox('Navigate and annotate left image')

        row = 0
        bottomLeftLayout.addWidget(self.annotOptionsWidget, row, 0, 1, 4)
        # bottomLeftLayout.addWidget(
        #     self.drawIDsContComboBox, row, 1, 1, 2,
        #     alignment=Qt.AlignCenter
        # )

        # bottomLeftLayout.addWidget(
        #     self.showTreeInfoCheckbox, row, 0, 1, 1,
        #     alignment=Qt.AlignCenter
        # )

        row += 1
        navWidgetsLayout = QHBoxLayout()
        self.navSpinBox = widgets.SpinBox(disableKeyPress=True)
        self.navSpinBox.setMinimum(1)
        self.navSpinBox.setMaximum(100)
        self.navSizeLabel = QLabel('/ND')
        navWidgetsLayout.addWidget(self.t_label)
        navWidgetsLayout.addWidget(self.navSpinBox)
        navWidgetsLayout.addWidget(self.navSizeLabel)
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
        self.lastTrackedFrameLabel.setFont(_font)
        bottomLeftLayout.addWidget(self.lastTrackedFrameLabel, row, 3)

        row += 1
        zSliceCheckboxLayout = QHBoxLayout()
        self.zSliceCheckbox = QCheckBox('z-slice')
        self.zSliceSpinbox = widgets.SpinBox(disableKeyPress=True)
        self.zSliceSpinbox.setMinimum(1)
        self.SizeZlabel = QLabel('/ND')
        self.zSliceCheckbox.setToolTip(
            'Activate/deactivate control of the z-slices with keyboard arrows.\n\n'
            'SHORTCUT to toggle ON/OFF: "Z" key'
        )
        zSliceCheckboxLayout.addWidget(self.zSliceCheckbox)
        zSliceCheckboxLayout.addWidget(self.zSliceSpinbox)
        zSliceCheckboxLayout.addWidget(self.SizeZlabel)
        bottomLeftLayout.addLayout(
            zSliceCheckboxLayout, row, 0, alignment=Qt.AlignRight
        )
        bottomLeftLayout.addWidget(self.zSliceScrollBar, row, 1, 1, 2)
        bottomLeftLayout.addWidget(self.zProjComboBox, row, 3)
        bottomLeftLayout.addWidget(self.zProjLockViewButton, row, 4)
        bottomLeftLayout.addWidget(self.switchPlaneCombobox, row, 5)
        self.zSliceSpinbox.connectValueChanged(self.onZsliceSpinboxValueChange)
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

    def gui_createLabWidgets(self):
        bottomRightLayout = QVBoxLayout()
        self.rightBottomGroupbox = widgets.GroupBox(
            'Annotate right image independent of left image',
            keyPressCallback=self.resetFocus
        )
        self.rightBottomGroupbox.setCheckable(True)
        self.rightBottomGroupbox.setChecked(False)
        self.rightBottomGroupbox.hide()

        self.annotateRightHowCombobox = widgets.ComboBox()
        self.annotateRightHowCombobox.setFont(_font)
        self.annotateRightHowCombobox.addItems(self.drawIDsContComboBoxSegmItems)
        self.annotateRightHowCombobox.setCurrentIndex(
            self.drawIDsContComboBox.currentIndex()
        )
        self.annotateRightHowCombobox.setVisible(False)

        self.annotOptionsLayoutRight.addWidget(self.annotateRightHowCombobox)

        self.rightImageFramesScrollbar = widgets.ScrollBarWithNumericControl(
            labelText='Frame n. '
        )
        self.rightImageFramesScrollbar.setVisible(False)

        bottomRightLayout.addWidget(self.annotOptionsWidgetRight)
        bottomRightLayout.addWidget(self.rightImageFramesScrollbar)
        bottomRightLayout.addStretch(1)

        self.rightBottomGroupbox.setLayout(bottomRightLayout)

        self.rightBottomGroupbox.toggled.connect(self.rightImageControlsToggled)

    def rightImageControlsToggled(self, checked):
        if self.isDataLoading:
            return
        if checked:
            self.annotateRightHowCombobox.setCurrentText(
                self.drawIDsContComboBox.currentText()
            )
        self.updateAllImages()

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

        zoom_perc = self.view_model.bottom_layout_zoom_percent(
            self.df_settings
        )
        self.bottomLayoutContextMenu = QMenu('Bottom layout', self)
        zoomMenu = self.bottomLayoutContextMenu.addMenu('Zoom')
        actions = []
        self.bottomLayoutContextMenu.zoomActionGroup = QActionGroup(zoomMenu)
        for perc in self.view_model.bottom_layout_zoom_values():
            action = QAction(f'{perc}%', zoomMenu)
            action.setCheckable(True)
            if perc == zoom_perc:
                action.setChecked(True)
            action.toggled.connect(self.zoomBottomLayoutActionTriggered)
            actions.append(action)
            self.bottomLayoutContextMenu.zoomActionGroup.addAction(action)
        zoomMenu.addActions(actions)
        resetAction = self.bottomLayoutContextMenu.addAction(
            'Reset default height'
        )
        resetAction.triggered.connect(self.resizeGui)
        retainSpaceAction = self.bottomLayoutContextMenu.addAction(
            'Retain space of hidden sliders'
        )
        retainSpaceAction.setCheckable(True)
        retainSpaceChecked = self.view_model.retain_space_hidden_sliders(
            self.df_settings
        )
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
        from cellacdc.canvas.qt.dual_pane import create_dual_pane

        parts = create_dual_pane(invert_bw=self.invertBwAction.isChecked())
        self.graphLayout = parts['graphLayout']
        self.ax1 = parts['ax1']
        self.ax2 = parts['ax2']
        self.lutItemsLayout = parts['lutItemsLayout']
        self.titleColor = parts['titleColor']
        self.plotsCol = 1
