"""Toolbars of the main GUI, the controls toolbar, tool settings menus and
right-click menus of toolbar buttons. Also RightClickMenuFilter. Mixins
of guiWin (gui/main_window/window.py) only: the measurements viewer
never composes them.
"""
import re
from functools import partial
from natsort import natsorted
from qtpy.QtCore import Qt, QEvent, QObject
from qtpy.QtGui import QIcon, QCursor
from qtpy.QtWidgets import (
    QAction, QLabel, QSizePolicy, QMenu, QCheckBox, QToolButton, QButtonGroup,
    QActionGroup, QWidget, QRadioButton
)
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import widgets


class RightClickMenuFilter(QObject):
    """Install on any widget to show a custom QMenu on right-click,
    without altering the widget's normal left-click behavior."""
    
    def eventFilter(self, obj, event):
        if (event.type() == QEvent.MouseButtonPress 
                and event.button() == Qt.RightButton):
            menu = getattr(obj, 'rightClickMenu', None)
            if menu is not None:
                menu.exec_(QCursor.pos())
                return True  # swallow the event — don't let it fall through
                             # to the button's normal press handling
        return False  # everything else: pass through untouched


class ToolbarsGuiElements:
    """Toolbars of the main GUI (gui_createToolBars), the controls toolbar,
    the tool settings menu, right-click menus of toolbar buttons
    (RightClickMenuFilter) and closing the toolbars.
    """

    def gui_createToolBars(self):        
        # File toolbar
        fileToolBar = self.addToolBar("File")
        # fileToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        fileToolBar.setMovable(False)

        self.segmNdimIndicatorAction = fileToolBar.addWidget(
            self.segmNdimIndicator
        )
        self.segmNdimIndicatorAction.setVisible(False)
        fileToolBar.addAction(self.newAction)
        fileToolBar.addAction(self.openFolderAction)
        fileToolBar.addAction(self.openFileAction)
        fileToolBar.addAction(self.manageVersionsAction)
        fileToolBar.addAction(self.saveAction)
        fileToolBar.addAction(self.showInExplorerAction)
        # fileToolBar.addAction(self.reloadAction)
        fileToolBar.addAction(self.undoAction)
        fileToolBar.addAction(self.redoAction)
        self.fileToolBar = fileToolBar
        self.setEnabledFileToolbar(False)

        self.undoAction.setEnabled(False)
        self.redoAction.setEnabled(False)

        # Navigation toolbar
        navigateToolBar = widgets.ToolBar("Navigation", self)
        navigateToolBar.setContextMenuPolicy(Qt.PreventContextMenu)
        # navigateToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        self.addToolBar(navigateToolBar)
        navigateToolBar.addWidget(self.exportCurrentViewButton)
        self.widgetsWithShortcut['Setup exporting data from current view'] = (
            self.exportCurrentViewButton
        )
        
        navigateToolBar.addAction(self.searchAction)
        navigateToolBar.addWidget(self.zoomRectButton)

        self.slideshowButton = QToolButton(self)
        self.slideshowButton.setIcon(QIcon(":eye-plus.svg"))
        self.slideshowButton.setCheckable(True)
        self.slideshowButton.setShortcut('Ctrl+W')
        navigateToolBar.addWidget(self.slideshowButton)
        
        self.launch3dViewerAction = QAction(
            'Launch 3D viewer for the currently visualised volume', self
        )
        self.launch3dViewerAction.setIcon(QIcon(":3d.svg"))
        self.launch3dViewerAction.setCheckable(True)
        navigateToolBar.addAction(self.launch3dViewerAction)
        
        self.ImageMenu.addAction(self.launch3dViewerAction)
        
        navigateToolBar.addAction(self.autoPilotButton)
        
        # navigateToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        navigateToolBar.addAction(self.skipToNewIdAction)
        
        self.preprocessImageAction = QAction('Preprocess image', self)
        self.preprocessImageAction.setIcon(QIcon(":filter_image.svg"))
        navigateToolBar.addAction(self.preprocessImageAction)

        self.overlayButton = widgets.rightClickToolButton(parent=self)
        self.overlayButton.setIcon(QIcon(":overlay.svg"))
        self.overlayButton.setCheckable(True)

        self.overlayButtonAction = navigateToolBar.addWidget(self.overlayButton)
        # self.checkableButtons.append(self.overlayButton)
        # self.checkableQButtonsGroup.addButton(self.overlayButton)
        
        self.countObjsButton = QToolButton(self)
        self.countObjsButton.setIcon(QIcon(":count_objects.svg"))
        self.countObjsButton.setCheckable(True)
        self.countObjsButton.setShortcut('Ctrl+Shift+C')
        self.countObjsButtonAction = navigateToolBar.addWidget(
            self.countObjsButton
        )

        self.togglePointsLayerAction = QAction('Activate points layer', self)
        self.togglePointsLayerAction.setCheckable(True)
        self.togglePointsLayerAction.setIcon(QIcon(":pointsLayer.svg"))
        navigateToolBar.addAction(self.togglePointsLayerAction)

        self.overlayLabelsButton = widgets.rightClickToolButton(parent=self)
        self.overlayLabelsButton.setIcon(QIcon(":overlay_labels.svg"))
        self.overlayLabelsButton.setCheckable(True)
        # self.overlayLabelsButton.setVisible(False)
        self.overlayLabelsButtonAction = navigateToolBar.addWidget(
            self.overlayLabelsButton
        )
        self.overlayLabelsButtonAction.setVisible(False)

        self.rulerButton = QToolButton(self)
        self.rulerButton.setIcon(QIcon(":ruler.svg"))
        self.rulerButton.setCheckable(True)
        navigateToolBar.addWidget(self.rulerButton)
        self.checkableButtons.append(self.rulerButton)
        self.leftClickButtons.append(self.rulerButton)

        # fluorescence image color widget
        colorsToolBar = widgets.ToolBar("Colors", self)

        self.overlayColorButton = pg.ColorButton(self, color=(230,230,230))
        self.overlayColorButton.setDisabled(True)
        colorsToolBar.addWidget(self.overlayColorButton)

        self.textIDsColorButton = pg.ColorButton(self)
        colorsToolBar.addWidget(self.textIDsColorButton)

        self.addToolBar(colorsToolBar)
        colorsToolBar.setVisible(False)

        self.navigateToolBar = navigateToolBar

        # cca toolbar
        ccaToolBar = widgets.ToolBar("Cell cycle annotations", self)
        self.addToolBar(ccaToolBar)

        # Assign mother to bud button
        self.assignBudMothButton = QToolButton(self)
        self.assignBudMothButton.setIcon(QIcon(":assign-motherbud.svg"))
        self.assignBudMothButton.setCheckable(True)
        self.assignBudMothButton.setShortcut('A')
        self.assignBudMothButton.setVisible(False)
        self.assignBudMothButton.action = ccaToolBar.addWidget(
            self.assignBudMothButton
        )
        self.checkableButtons.append(self.assignBudMothButton)
        self.checkableQButtonsGroup.addButton(self.assignBudMothButton)
        self.functionsNotTested3D.append(self.assignBudMothButton)
        

        # Set is_history_known button
        self.setIsHistoryKnownButton = QToolButton(self)
        self.setIsHistoryKnownButton.setIcon(QIcon(":history.svg"))
        self.setIsHistoryKnownButton.setCheckable(True)
        self.setIsHistoryKnownButton.setShortcut('U')
        self.setIsHistoryKnownButton.setVisible(False)
        self.setIsHistoryKnownButton.action = ccaToolBar.addWidget(
            self.setIsHistoryKnownButton
        )
        self.checkableButtons.append(self.setIsHistoryKnownButton)
        self.checkableQButtonsGroup.addButton(self.setIsHistoryKnownButton)
        self.functionsNotTested3D.append(self.setIsHistoryKnownButton)
        
        self.annotateSingleMotherBudPairButton = QToolButton(self)
        self.annotateSingleMotherBudPairButton.setIcon(
            QIcon(":lock_id_annotate_future.svg")
        )
        self.annotateSingleMotherBudPairButton.setCheckable(True)
        self.annotateSingleMotherBudPairButton.setShortcut('Y')
        self.annotateSingleMotherBudPairButton.setVisible(False)
        self.annotateSingleMotherBudPairButton.action = ccaToolBar.addWidget(
            self.annotateSingleMotherBudPairButton
        )
        self.checkableButtons.append(self.annotateSingleMotherBudPairButton)
        self.widgetsWithShortcut[
            'Annotate one mother-bud pair at the time (cell cycle analysis)'
                ] = (
                self.annotateSingleMotherBudPairButton
            )
        self.checkableQButtonsGroup.addButton(
            self.annotateSingleMotherBudPairButton
        )
        self.functionsNotTested3D.append(self.annotateSingleMotherBudPairButton)
        
        ccaToolBar.addAction(self.assignBudMothAutoAction)
        ccaToolBar.addAction(self.editCcaToolAction)
        ccaToolBar.addAction(self.reInitCcaAction)
        ccaToolBar.setVisible(False)
        self.ccaToolBar = ccaToolBar
        self.functionsNotTested3D.append(self.assignBudMothAutoAction)
        self.functionsNotTested3D.append(self.reInitCcaAction)
        self.functionsNotTested3D.append(self.editCcaToolAction)

        # Edit toolbar
        editToolBar = widgets.ToolBar("Edit", self)
        editToolBar.setContextMenuPolicy(Qt.PreventContextMenu)
        
        self.addToolBar(editToolBar)
        
        self.manulAnnotToolButtons = set()

        self.brushButton = QToolButton(self)
        self.brushButton.setIcon(QIcon(":brush.svg"))
        self.brushButton.setCheckable(True)
        editToolBar.addWidget(self.brushButton)
        self.checkableButtons.append(self.brushButton)
        self.leftClickButtons.append(self.brushButton)
        self.brushButton.keyPressShortcut = Qt.Key_B
        self.widgetsWithShortcut['Brush'] = self.brushButton
        self.manulAnnotToolButtons.add(self.brushButton)
        self.toolsActiveInProj3Dsegm.add(self.brushButton)

        self.eraserButton = QToolButton(self)
        self.eraserButton.setIcon(QIcon(":eraser.svg"))
        self.eraserButton.setCheckable(True)
        editToolBar.addWidget(self.eraserButton)
        self.eraserButton.keyPressShortcut = Qt.Key_X
        self.widgetsWithShortcut['Eraser'] = self.eraserButton
        self.checkableButtons.append(self.eraserButton)
        self.leftClickButtons.append(self.eraserButton)
        self.manulAnnotToolButtons.add(self.eraserButton)

        self.curvToolButton = QToolButton(self)
        self.curvToolButton.setIcon(QIcon(":curvature-tool.svg"))
        self.curvToolButton.setCheckable(True)
        self.curvToolButton.setShortcut('C')
        self.curvToolButton.action = editToolBar.addWidget(self.curvToolButton)
        self.leftClickButtons.append(self.curvToolButton)
        # self.functionsNotTested3D.append(self.curvToolButton)
        self.widgetsWithShortcut['Curvature tool'] = self.curvToolButton
        # self.checkableButtons.append(self.curvToolButton)
        self.manulAnnotToolButtons.add(self.curvToolButton)

        self.wandToolButton = QToolButton(self)
        self.wandToolButton.setIcon(QIcon(":magic_wand.svg"))
        self.wandToolButton.setCheckable(True)
        self.wandToolButton.setShortcut('Ctrl+D')
        self.wandToolButton.action = editToolBar.addWidget(self.wandToolButton)
        self.leftClickButtons.append(self.wandToolButton)
        self.checkableButtons.append(self.eraserButton)
        self.widgetsWithShortcut['Magic wand'] = self.wandToolButton
        
        self.magicPromptsToolButton = QToolButton(self)
        self.magicPromptsToolButton.setIcon(QIcon(":magic-prompts.svg"))
        self.magicPromptsToolButton.setCheckable(True)
        self.magicPromptsToolButton.setShortcut('W')
        self.magicPromptsToolButton.action = editToolBar.addWidget(
            self.magicPromptsToolButton
        )
        self.widgetsWithShortcut['Magic prompts'] = self.magicPromptsToolButton
        
        self.clearFreehandRoiButton = QToolButton(self)
        self.clearFreehandRoiButton.setCheckable(True)
        self.clearFreehandRoiButton.setIcon(QIcon(":clear_freehand_region.svg"))
        self.widgetsWithShortcut['Clear freehand region'] = (
            self.clearFreehandRoiButton
        )
        self.toolsActiveInProj3Dsegm.add(self.clearFreehandRoiButton)
        
        self.checkableButtons.append(self.clearFreehandRoiButton)
        self.leftClickButtons.append(self.clearFreehandRoiButton)
        
        self.drawClearRegionAction = editToolBar.addWidget(
            self.clearFreehandRoiButton
        )

        self.widgetsWithShortcut[
            'Annotate mother/daughter pairing (cell cycle analysis)'] = (
            self.assignBudMothButton
        )
        self.widgetsWithShortcut[
            'Annotate unknown history (cell cycle analysis)'] = (
            self.setIsHistoryKnownButton
        )
        
        self.copyLostObjButton = QToolButton(self)
        self.copyLostObjButton.setIcon(QIcon(":copyContour.svg"))
        self.copyLostObjButton.setCheckable(True)
        self.copyLostObjButton.setShortcut('V')
        self.copyLostObjButton.action = editToolBar.addWidget(
            self.copyLostObjButton
        )
        self.checkableButtons.append(self.copyLostObjButton)
        self.checkableQButtonsGroup.addButton(self.copyLostObjButton)
        self.widgetsWithShortcut['Copy lost object contour'] = (
            self.copyLostObjButton
        )
        self.functionsNotTested3D.append(self.copyLostObjButton)
        
        self.labelRoiButton = widgets.rightClickToolButton(parent=self)
        self.labelRoiButton.setIcon(QIcon(":label_roi.svg"))
        self.labelRoiButton.setCheckable(True)
        self.labelRoiButton.setShortcut('L')
        self.labelRoiButton.action = editToolBar.addWidget(self.labelRoiButton)
        self.leftClickButtons.append(self.labelRoiButton)
        self.checkableButtons.append(self.labelRoiButton)
        self.checkableQButtonsGroup.addButton(self.labelRoiButton)
        self.widgetsWithShortcut['Label ROI'] = self.labelRoiButton
        # self.functionsNotTested3D.append(self.labelRoiButton)
        
        self.manualAnnotPastButton = QToolButton(self)
        self.manualAnnotPastButton.setIcon(QIcon(":lock_id_annotate_future.svg"))
        self.manualAnnotPastButton.setCheckable(True)
        self.manualAnnotPastButton.setShortcut('Y')
        self.manualAnnotPastButton.action = editToolBar.addWidget(
            self.manualAnnotPastButton
        )
        self.checkableButtons.append(self.manualAnnotPastButton)
        self.widgetsWithShortcut['Lock ID and annotate single object'] = (
            self.manualAnnotPastButton
        )
        self.functionsNotTested3D.append(self.manualAnnotPastButton)
        self.manulAnnotToolButtons.add(self.manualAnnotPastButton)

        self.segmentToolAction = QAction('Segment with last used model', self)
        self.segmentToolAction.setIcon(QIcon(":segment.svg"))
        self.segmentToolAction.setShortcut('R')
        self.widgetsWithShortcut['Repeat segmentation'] = self.segmentToolAction
        editToolBar.addAction(self.segmentToolAction)
        # `addAction` returns None on PyQt6
        widget = editToolBar.widgetForAction(self.segmentToolAction)
        self.widgetsForActions['Repeat segmentation'] = widget


        self.segForLostIDsButton = QToolButton(self)
        self.segForLostIDsButton.setIcon(QIcon(":segForLostIDs.svg"))
        self.segForLostIDsAction = editToolBar.addWidget(
            self.segForLostIDsButton
        )
        self.segForLostIDsButton.clicked.connect(
            self.segForLostIDsButtonClicked
        )
        self.customRightClickItems['Segment for lost IDs'] = (
            self.segForLostIDsButton, [self.editSegForLostIDsSetSettings]
            )

        # self.SegForLostIDsButton.setShortcut('U')
        # self.widgetsWithShortcut['Unknown lineage (lineage tree)'] = self.SegForLostIDsButton
        
        self.manualBackgroundButton = QToolButton(self)
        self.manualBackgroundButton.setIcon(QIcon(":manual_background.svg"))
        self.manualBackgroundButton.setCheckable(True)
        self.manualBackgroundButton.setShortcut('G')
        self.leftClickButtons.append(self.manualBackgroundButton)
        self.checkableButtons.append(self.manualBackgroundButton)
        self.checkableQButtonsGroup.addButton(self.manualBackgroundButton)
        self.widgetsWithShortcut['Manual background'] = self.manualBackgroundButton
        
        self.manualBackgroundAction = editToolBar.addWidget(
            self.manualBackgroundButton
        )
        
        self.delObjsOutSegmMaskAction = QAction(
            QIcon(":del_objs_out_segm.svg"), 
            'Select a segmentation file and delete all objects on the background', 
            self
        )
        self.delObjsOutSegmMaskAction.setShortcut('I')
        self.widgetsWithShortcut['Delete all objects outside segm'] = (
            self.delObjsOutSegmMaskAction
        )
        editToolBar.addAction(self.delObjsOutSegmMaskAction)
        # `addAction` returns None on PyQt6
        widget = editToolBar.widgetForAction(self.delObjsOutSegmMaskAction)
        self.widgetsForActions['Delete all objects outside segm'] = widget


        self.hullContToolButton = QToolButton(self)
        self.hullContToolButton.setIcon(QIcon(":hull.svg"))
        self.hullContToolButton.setCheckable(True)
        self.hullContToolButton.setShortcut('O')
        self.hullContToolButton.action = editToolBar.addWidget(self.hullContToolButton)
        self.checkableButtons.append(self.hullContToolButton)
        self.checkableQButtonsGroup.addButton(self.hullContToolButton)
        self.widgetsWithShortcut['Hull contour'] = self.hullContToolButton

        self.fillHolesToolButton = QToolButton(self)
        self.fillHolesToolButton.setIcon(QIcon(":fill_holes.svg"))
        self.fillHolesToolButton.setCheckable(True)
        self.fillHolesToolButton.setShortcut('F')
        self.fillHolesToolButton.action = editToolBar.addWidget(
            self.fillHolesToolButton
        )
        self.checkableButtons.append(self.fillHolesToolButton)
        self.checkableQButtonsGroup.addButton(self.fillHolesToolButton)
        self.widgetsWithShortcut['Fill holes'] = self.fillHolesToolButton

        self.moveLabelToolButton = QToolButton(self)
        self.moveLabelToolButton.setIcon(QIcon(":moveLabel.svg"))
        self.moveLabelToolButton.setCheckable(True)
        self.moveLabelToolButton.setShortcut('P')
        self.moveLabelToolButton.action = editToolBar.addWidget(self.moveLabelToolButton)
        self.checkableButtons.append(self.moveLabelToolButton)
        self.checkableQButtonsGroup.addButton(self.moveLabelToolButton)
        self.widgetsWithShortcut['Move label'] = self.moveLabelToolButton

        self.expandLabelToolButton = QToolButton(self)
        self.expandLabelToolButton.setIcon(QIcon(":expandLabel.svg"))
        self.expandLabelToolButton.setCheckable(True)
        self.expandLabelToolButton.setShortcut('E')
        self.expandLabelToolButton.action = editToolBar.addWidget(self.expandLabelToolButton)
        self.expandLabelToolButton.hide()
        self.checkableButtons.append(self.expandLabelToolButton)
        self.leftClickButtons.append(self.expandLabelToolButton)
        self.checkableQButtonsGroup.addButton(self.expandLabelToolButton)
        self.widgetsWithShortcut['Expand/shrink label'] = self.expandLabelToolButton

        self.editIDbutton = QToolButton(self)
        self.editIDbutton.setIcon(QIcon(":edit-id.svg"))
        self.editIDbutton.setCheckable(True)
        self.editIDbutton.setShortcut('N')
        editToolBar.addWidget(self.editIDbutton)
        self.checkableButtons.append(self.editIDbutton)
        self.checkableQButtonsGroup.addButton(self.editIDbutton)
        self.widgetsWithShortcut['Edit ID'] = self.editIDbutton

        self.separateBudButton = QToolButton(self)
        self.separateBudButton.setIcon(QIcon(":separate-bud.svg"))
        self.separateBudButton.setCheckable(True)
        self.separateBudButton.setShortcut('S')
        self.separateBudButton.action = editToolBar.addWidget(self.separateBudButton)
        self.checkableButtons.append(self.separateBudButton)
        self.checkableQButtonsGroup.addButton(self.separateBudButton)
        # self.functionsNotTested3D.append(self.separateBudButton)
        self.widgetsWithShortcut['Separate objects'] = self.separateBudButton

        self.mergeIDsButton = QToolButton(self)
        self.mergeIDsButton.setIcon(QIcon(":merge-multiple-IDs.svg"))
        self.mergeIDsButton.setCheckable(True)
        self.mergeIDsButton.setShortcut('M')
        self.mergeIDsButton.action = editToolBar.addWidget(
            self.mergeIDsButton
        )
        self.widgetsWithShortcut['Merge objects'] = (
            self.mergeIDsButton
        )
        self.toolsActiveInProj3Dsegm.add(self.mergeIDsButton)
        
        self.checkableQButtonsGroup.addButton(self.mergeIDsButton)
        self.checkableButtons.append(self.mergeIDsButton)
        self.leftClickButtons.append(self.mergeIDsButton)

        self.keepIDsButton = QToolButton(self)
        self.keepIDsButton.setIcon(QIcon(":keep_objects.svg"))
        self.keepIDsButton.setCheckable(True)
        self.keepIDsButton.action = editToolBar.addWidget(self.keepIDsButton)
        self.keepIDsButton.setShortcut('K')
        self.checkableButtons.append(self.keepIDsButton)
        self.checkableQButtonsGroup.addButton(self.keepIDsButton)
        # self.functionsNotTested3D.append(self.keepIDsButton)
        self.widgetsWithShortcut['Select objects to keep'] = self.keepIDsButton

        self.whitelistIDsButton = QToolButton(self)
        self.whitelistIDsButton.setIcon(QIcon(":whitelist.svg"))
        self.whitelistIDsButton.setCheckable(True)
        self.whitelistIDsButton.action = editToolBar.addWidget(
            self.whitelistIDsButton
        )
        self.whitelistIDsButton.setShortcut('Ctrl+Shift+W')
        self.checkableButtons.append(self.whitelistIDsButton)
        self.checkableQButtonsGroup.addButton(self.whitelistIDsButton)
        self.leftClickButtons.append(self.whitelistIDsButton)
        # self.functionsNotTested3D.append(self.whitelistIDsButton)
        self.widgetsWithShortcut['Select objects to add to a tracking whitelist'] = (
            self.whitelistIDsButton
        )

        self.binCellButton = QToolButton(self)
        self.binCellButton.setIcon(QIcon(":bin.svg"))
        self.binCellButton.setCheckable(True)
        self.binCellButton.action = editToolBar.addWidget(self.binCellButton)
        self.checkableButtons.append(self.binCellButton)
        self.checkableQButtonsGroup.addButton(self.binCellButton)
        self.binCellButton.setShortcut('Ctrl+E')
        self.widgetsWithShortcut['Exclude cell from analysis'] = self.binCellButton
        # self.functionsNotTested3D.append(self.binCellButton)

        self.manualTrackingButton = QToolButton(self)
        self.manualTrackingButton.setIcon(QIcon(":manual_tracking.svg"))
        self.manualTrackingButton.setCheckable(True)
        self.manualTrackingButton.setShortcut('T')
        self.checkableQButtonsGroup.addButton(self.manualTrackingButton)
        self.checkableButtons.append(self.manualTrackingButton)
        self.widgetsWithShortcut['Manual tracking'] = self.manualTrackingButton

        self.ripCellButton = QToolButton(self)
        self.ripCellButton.setIcon(QIcon(":rip.svg"))
        self.ripCellButton.setCheckable(True)
        self.ripCellButton.setShortcut('D')
        self.ripCellButton.action = editToolBar.addWidget(self.ripCellButton)
        self.checkableButtons.append(self.ripCellButton)
        self.checkableQButtonsGroup.addButton(self.ripCellButton)
        self.functionsNotTested3D.append(self.ripCellButton)
        self.widgetsWithShortcut['Annotate cell as dead'] = self.ripCellButton

        editToolBar.addAction(self.addDelRoiAction)
        # editToolBar.addAction(self.addDelPolyLineRoiAction)
        
        self.addDelPolyLineRoiAction = editToolBar.addWidget(
            self.addDelPolyLineRoiButton
        )
        self.addDelPolyLineRoiAction.roiType = 'polyline'
        
        editToolBar.addAction(self.delBorderObjAction)
        self.delBorderObjAction.button = editToolBar.widgetForAction(
            self.delBorderObjAction
        )
        editToolBar.addAction(self.delNewObjAction)
        self.delNewObjAction.button = editToolBar.widgetForAction(
            self.delNewObjAction
        )

        self.addDelRoiAction.toolbar = editToolBar
        self.functionsNotTested3D.append(self.addDelRoiAction)

        self.addDelPolyLineRoiAction.toolbar = editToolBar
        self.functionsNotTested3D.append(self.addDelPolyLineRoiAction)

        self.delBorderObjAction.toolbar = editToolBar
        self.functionsNotTested3D.append(self.delBorderObjAction)
        
        self.delNewObjAction.toolbar = editToolBar
        # self.functionsNotTested3D.append(self.delNewObjAction) so id this doesnt work in 3d i dont know anymore

        editToolBar.addAction(self.repeatTrackingAction)
        # `addAction` returns None on PyQt6
        widget = editToolBar.widgetForAction(self.repeatTrackingAction)
        self.widgetsForActions['Repeat Tracking'] = widget

        self.manualTrackingAction = editToolBar.addWidget(
            self.manualTrackingButton
        )

        # self.functionsNotTested3D.append(self.repeatTrackingAction)
        self.functionsNotTested3D.append(self.manualTrackingAction)

        self.reinitLastSegmFrameAction = QAction(self)
        self.reinitLastSegmFrameAction.setIcon(QIcon(":reinitLastSegm.svg"))
        self.reinitLastSegmFrameAction.setVisible(False)
        editToolBar.addAction(self.reinitLastSegmFrameAction)
        editToolBar.setVisible(False)
        self.reinitLastSegmFrameAction.toolbar = editToolBar
        self.functionsNotTested3D.append(self.reinitLastSegmFrameAction)


        self.editLin_TreeBar = widgets.ToolBar("Lin Tree Edit", self)
        self.editLin_TreeBar.setContextMenuPolicy(Qt.PreventContextMenu)
        
        self.addToolBar(self.editLin_TreeBar)
        self.editLin_TreeGroup = QButtonGroup()
        self.editLin_TreeGroup.setExclusive(True)

        self.findNextMotherButton = QToolButton(self)
        self.findNextMotherButton.setIcon(QIcon(":magnGlass.svg"))
        self.findNextMotherButton.setCheckable(True)
        self.editLin_TreeBar.addWidget(self.findNextMotherButton)
        self.editLin_TreeGroup.addButton(self.findNextMotherButton)
        self.findNextMotherButton.setShortcut('F')
        self.widgetsWithShortcut['Find next potential mother (lineage tree)'] = self.findNextMotherButton

        self.unknownLineageButton = QToolButton(self)
        self.unknownLineageButton.setIcon(QIcon(":history.svg"))
        self.unknownLineageButton.setCheckable(True)
        self.editLin_TreeBar.addWidget(self.unknownLineageButton)
        self.editLin_TreeGroup.addButton(self.unknownLineageButton)
        self.unknownLineageButton.setShortcut('U')
        self.widgetsWithShortcut['Unknown lineage (lineage tree)'] = self.unknownLineageButton

        self.noToolLinTreeButton = QToolButton(self)
        self.noToolLinTreeButton.setIcon(QIcon(":arrow_cursor.svg"))
        self.noToolLinTreeButton.setCheckable(True)
        self.editLin_TreeBar.addWidget(self.noToolLinTreeButton)
        self.editLin_TreeGroup.addButton(self.noToolLinTreeButton)
        self.noToolLinTreeButton.setShortcut('N')
        self.widgetsWithShortcut['No tool (lineage tree)'] = self.noToolLinTreeButton

        self.propagateLinTreeButton = QToolButton(self)
        self.propagateLinTreeButton.setIcon(QIcon(":compute.svg"))
        self.editLin_TreeBar.addWidget(self.propagateLinTreeButton)
        self.propagateLinTreeButton.setShortcut('P')
        self.widgetsWithShortcut['Propagate (lineage tree)'] = self.propagateLinTreeButton
        self.propagateLinTreeButton.clicked.connect(self.propagateLinTreeAction)

        self.viewLinTreeInfoButton = QToolButton(self)
        self.viewLinTreeInfoButton.setIcon(QIcon(":addCustomAnnotation.svg"))
        self.editLin_TreeBar.addWidget(self.viewLinTreeInfoButton)
        self.viewLinTreeInfoButton.setShortcut('S')
        self.widgetsWithShortcut['View Changes (lineage tree)'] = self.viewLinTreeInfoButton
        self.viewLinTreeInfoButton.clicked.connect(self.viewLinTreeInfoAction)
    

        modes_available = [
            'Segmentation and Tracking',
            'Cell cycle analysis',
            'Viewer',
            'Custom annotations',
            'Normal division: Lineage tree'
        ]
        self.modeItems = modes_available

        self.modeActionGroup = QActionGroup(self.modeMenu)
        for mode in self.modeItems:
            action = QAction(mode)
            action.setCheckable(True)
            self.modeActionGroup.addAction(action)
            self.modeMenu.addAction(action)
            if mode == 'Viewer':
                action.setChecked(True)

        self.editToolBar = editToolBar
        self.editToolBar.setVisible(False)
        self.navigateToolBar.setVisible(False)
        self.editLin_TreeBar.setVisible(False)

        self.gui_createAnnotateToolbar()

    def gui_editRightClickMenuButtons(self):
        NAMES_TO_IGNORE_ERROR = (
            "Toggle max z projection",
            "Previous",
            "Next"
        )
        
        ok_num_widgets = [1]
        for name, button in self.widgetsWithShortcut.items():
            if name in NAMES_TO_IGNORE_ERROR:
                continue
            res = self._setupRightClickMenuOnButton(button, name)
            if res[0] is False or res[1] not in ok_num_widgets and self.debug:
                print(f"Error setting up right click menu for: {name}")
                print(f"Number of associated widgets: {res[1]}")
            menu = button.rightClickMenu
            action = QAction('Set shortcut...', self)
            action.triggered.connect(
                lambda z, name=name: self.editShortcuts_cb(highlighted_shortcut=name)
            )
            menu.addAction(action)
            
        for name, button in self.keepToolActiveNames.items():
            if name in NAMES_TO_IGNORE_ERROR:
                continue
            res = self._setupRightClickMenuOnButton(button, name)
            if res[0] is False or res[1] not in ok_num_widgets:
                print(f"Error setting up right click menu for: {name}")
                print(f"Number of associated widgets: {res[1]}")
            menu = button.rightClickMenu
            action = self.keepToolActiveActions[name]
            menu.addAction(action)
            
        for name, button in self.applyToolNewFrameButtons.items():
            if name in NAMES_TO_IGNORE_ERROR:
                continue
            res = self._setupRightClickMenuOnButton(button, name)
            if res[0] is False or res[1] not in ok_num_widgets:
                print(f"Error setting up right click menu for: {name}")
                print(f"Number of associated widgets: {res[1]}")
            menu = button.rightClickMenu
            action = self.applyToolNewFrameActions[name]
            menu.addAction(action)
            
        # add custom right click items for some settings
        for name, (button, actions) in self.customRightClickItems.items():
            if name in NAMES_TO_IGNORE_ERROR:
                continue
            res = self._setupRightClickMenuOnButton(button, name)
            if res[0] is False or res[1] not in ok_num_widgets:
                print(f"Error setting up right click menu for: {name}")
                print(f"Number of associated widgets: {res[1]}")
            menu = button.rightClickMenu
            for action in actions:
                menu.addAction(action)

    def _setupRightClickMenuOnButton(self, target, name):
        if hasattr(target, 'rightClickMenu') and target.rightClickMenu is not None:
            return True, 1
        
        menu = QMenu(self)
        target.rightClickMenu = menu
        widgets = None
        if isinstance(target, QAction):
            if self.widgetsForActions.get(name) is not None:
                # Only the button of the action (on PyQt5,
                # `associatedWidgets` also returns the parent toolbar)
                widgets = [self.widgetsForActions[name]]
            elif hasattr(target, 'associatedWidgets'):
                widgets = target.associatedWidgets()
            elif hasattr(target, 'widgetsforAction'):
                widgets = target.widgetsforAction()
            elif hasattr(target, 'associatedObjects'):
                objects = target.associatedObjects()
                widgets = [
                    obj for obj in objects
                    if isinstance(obj, QToolButton) and obj is not self
                    and obj is not target.parent
                    ]

            if widgets is not None:
                widgets = [w for w in widgets if w is not None]
            if widgets is None or len(widgets) == 0:
                return False, 0
            else:    
                for w in widgets:
                    self._installRightClickFilter(w, menu)
        else:
            self._installRightClickFilter(target, menu)
        return True, len(widgets) if widgets is not None else 1

    def _installRightClickFilter(self, widget: QWidget, menu: QMenu):
        if getattr(widget, 'installedEventFilter', False):
            return
        
        if not hasattr(self, '_rightClickFilterObj'):
            self._rightClickFilterObj = RightClickMenuFilter()
        widget.installEventFilter(self._rightClickFilterObj)
        widget.installedEventFilter = True
        widget.rightClickMenu = menu

    def gui_createControlsToolbar(self):
        self.controlToolBars = []
        self.addToolBarBreak()
        
        # Edit toolbar
        modeToolBar = widgets.ToolBar("Mode", self)
        self.addToolBar(modeToolBar)

        self.modeComboBox = widgets.ComboBox()
        self.modeComboBox.addItems(self.modeItems)
        self.modeComboBoxLabel = QLabel('    Mode: ')
        self.modeComboBoxLabel.setBuddy(self.modeComboBox)
        modeToolBar.addWidget(self.modeComboBoxLabel)
        modeToolBar.addWidget(self.modeComboBox)
        modeToolBar.setVisible(False)
        
        self.modeToolBar = modeToolBar
        
        self.overlayToolbar = widgets.OverlayToolbar(parent=self)
        self.addToolBar(Qt.TopToolBarArea, self.overlayToolbar)
        self.overlayToolbar.setVisible(False)
        self.overlayToolbar.sigSetTranspacency.connect(
            self.setOverlayTransparency
        )
        self.overlayToolbar.sigSetSingleChannel.connect(
            self.setOverlaySingleChannel
        )
        # Overlay modes as the user left them (applied by `overlay_cb`)
        self.restoreOverlayModes()
        self.overlayToolbar.sigModesClicked.connect(self.saveOverlayModes)

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
            parent=self
        )
        self.addToolBar(Qt.TopToolBarArea, self.highlightIDToolbar)
        self.highlightIDToolbar.setVisible(False)
        self.highlightIDToolbar.keepVisibleWhenActive = True
        self.controlToolBars.append(self.highlightIDToolbar)
        
        self.highlightIDToolbar.sigIDChanged.connect(
            self.setHighlighedIDfromToolbar
        )
        
        # Widgets toolbar
        brushEraserToolBar = widgets.ToolBar("Widgets", self)
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

        brushEraserToolBar.addSeparator()

        self.interpZButton = QToolButton(self)
        self.interpZButton.setIcon(QIcon(":interpolate-Z.svg"))
        self.interpZButton.setCheckable(True)
        self.interpZButton.setShortcut('Shift+Q')
        self.interpZButton.action = brushEraserToolBar.addWidget(
            self.interpZButton
        )
        self.widgetsWithShortcut['Interpolate sparsely labelled Z volume'] = (
            self.interpZButton
        )

        self.interpZConfirmAction = QAction(self)
        self.interpZConfirmAction.setIcon(QIcon(":greenTick.svg"))
        self.interpZConfirmAction.setToolTip(
            'Interpolate the mask between annotated z-slices.\n\n'
            'Shortcut: "Enter"'
        )
        brushEraserToolBar.addAction(self.interpZConfirmAction)

        brushEraserToolBar.addSeparator()

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
        self.brushAutoFillCheckbox.setTristate(False)
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
            parent=self
        )

        self.addToolBar(Qt.TopToolBarArea , self.wandControlsToolbar)
        self.wandControlsToolbar.setVisible(False)
        self.controlToolBars.append(self.wandControlsToolbar)

        separatorW = 5
        self.labelRoiToolbar = widgets.ToolBar("Magic labeller controls", self)
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

        self.labelRoiFromCurrentFrameAction = QAction(self)
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

        self.labelRoiToEndFramesAction = QAction(self)
        self.labelRoiToEndFramesAction.setText('Segment all remaining frames')
        self.labelRoiToEndFramesAction.setIcon(QIcon(":frames_end.svg"))
        self.labelRoiToolbar.addAction(self.labelRoiToEndFramesAction)
        self.labelRoiToEndFramesAction.setDisabled(True)

        self.labelRoiTrangeCheckbox = QCheckBox('Segment range of frames')
        self.labelRoiToolbar.addWidget(self.labelRoiTrangeCheckbox)

        self.labelRoiViewCurrentModelAction = QAction(self)
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

        self.keepIDsToolbar = widgets.ToolBar("Keep IDs controls", self)
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
            instructionsLabel, parent=self
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

        self.mergeIDsToolbar = widgets.MergeIDsToolbar(
            "Merge multiple IDs controls", self, log_func=self.logger.info
        )
        self.addToolBar(Qt.TopToolBarArea, self.mergeIDsToolbar)
        self.mergeIDsToolbar.setVisible(False)
        self.controlToolBars.append(self.mergeIDsToolbar)
        
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
        
        self.pointsLayersToolbar = widgets.PointsLayersToolbar(parent=self)
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
        for name, widget in self.copyLostObjToolbar.widgetsForActions.items():
            self.widgetsForActions[name] = widget

        self.copyLostObjToolbar.sigCopyAllObjects.connect(
            self.copyAllLostObjects
        )
        
        self.addToolBar(Qt.TopToolBarArea, self.copyLostObjToolbar)
        self.copyLostObjToolbar.setVisible(False)
        # self.controlToolBars.append(self.copyLostObjToolbar)
        
        # Copy lost object contour toolbar
        self.clearFreehandRoiToolbar = widgets.ClearRegionToolbar(
            "Draw freehand region and clear objects controls", self
        )
        
        self.addToolBar(Qt.TopToolBarArea, self.clearFreehandRoiToolbar)
        self.clearFreehandRoiToolbar.setVisible(False)
        self.controlToolBars.append(self.clearFreehandRoiToolbar)

        try:
            addNewIDToggleState = self.df_settings.at['addNewIDsWhitelistToggle', 'value'] == 'Yes'
        except KeyError:
            addNewIDToggleState = True
        
        self.whitelistIDsToolbar = widgets.WhitelistIDsToolbar(
            addNewIDToggleState, self
        )
        for name, action in self.whitelistIDsToolbar.widgetsWithShortcut.items():
            self.widgetsWithShortcut[name] = action
        for name, widget in self.whitelistIDsToolbar.widgetsForActions.items():
            self.widgetsForActions[name] = widget
        
        self.addToolBar(Qt.TopToolBarArea, self.whitelistIDsToolbar)
        self.whitelistIDsToolbar.setVisible(False)
        self.controlToolBars.append(self.whitelistIDsToolbar)
        
        self.separateBudToolbar = widgets.SeparateBudToolbar(self)
        for name, action in self.separateBudToolbar.widgetsWithShortcut.items():
            self.widgetsWithShortcut[name] = action
        for name, widget in self.separateBudToolbar.widgetsForActions.items():
            self.widgetsForActions[name] = widget

        self.addToolBar(Qt.TopToolBarArea, self.separateBudToolbar)
        self.separateBudToolbar.setVisible(False)
        self.controlToolBars.append(self.separateBudToolbar)
        self.separateBudButton.toggled.connect(self.separateBudToolbar.setVisible)
        
        self.magicPromptsToolbar = widgets.MagicPromptsToolbar(self)
        for name, action in self.magicPromptsToolbar.widgetsWithShortcut.items():
            self.widgetsWithShortcut[name] = action
        for name, widget in self.magicPromptsToolbar.widgetsForActions.items():
            self.widgetsForActions[name] = widget

        self.magicPromptsToolbar.sigComputeOnZoom.connect(
            self.magicPromptsComputeOnZoomTriggered
        )
        self.magicPromptsToolbar.sigComputeOnImage.connect(
            self.magicPromptsComputeOnImageTriggered
        )
        self.magicPromptsToolbar.sigInitSelectedModel.connect(
            self.magicPromptsInitModel
        )
        self.magicPromptsToolbar.sigViewModelParams.connect(
            self.viewSetMagicPromptModelParams
        )
        self.magicPromptsToolbar.sigClearPoints.connect(
            partial(self.magicPromptsClearPoints, only_zoom=False)
        )
        self.magicPromptsToolbar.sigClearPointsOnZmom.connect(
            partial(self.magicPromptsClearPoints, only_zoom=True)
        )
        self.magicPromptsToolbar.sigInterpolateZslice.connect(
            self.magicPromptsInterpolateZsliceToggled
        )
        self.magicPromptsToolbar.sigPromptTypeChanged.connect(
            self.magicPromptsPromptTypeChanged
        )
        self.magicPromptsToolbar.sigTrain.connect(
            self.magicPromptsTrainTriggered
        )
        self.magicPromptsToolbar.sigSegmentImage.connect(
            self.magicPromptsSegmentImageTriggered
        )
        self.magicPromptsToolbar.sigSegmentZstack.connect(
            self.magicPromptsSegmentZstackTriggered
        )
        self.magicPromptsToolbar.sigSegmentStack.connect(
            self.magicPromptsSegmentStackTriggered
        )
        self.magicPromptsToolbar.sigClearStrokes.connect(
            partial(self.magicPromptsClearBrushStrokes, only_zoom=False)
        )
        self.magicPromptsToolbar.sigClearStrokesOnZoom.connect(
            partial(self.magicPromptsClearBrushStrokes, only_zoom=True)
        )
        self.magicPromptsToolbar.sigSaveModel.connect(
            self.magicPromptsSaveModelTriggered
        )
        self.magicPromptsToolbar.sigLoadModel.connect(
            self.magicPromptsLoadModelTriggered
        )
        self.magicPromptsToolbar.sigClearBoxes.connect(
            partial(self.magicPromptsClearBoxes, only_zoom=False)
        )
        self.magicPromptsToolbar.sigClearBoxesOnZoom.connect(
            partial(self.magicPromptsClearBoxes, only_zoom=True)
        )
        self.magicPromptsToolbar.sigSelectCellsToggled.connect(
            self.brushStrokesSelectCellsToggled
        )
        self.magicPromptsToolbar.sigSelectAllCells.connect(
            self.brushStrokesSelectAllCells
        )

        self.addToolBar(Qt.TopToolBarArea, self.magicPromptsToolbar)
        self.magicPromptsToolbar.setVisible(False)
        self.magicPromptsToolbar.keepVisibleWhenActive = True
        self.controlToolBars.append(self.magicPromptsToolbar)

        # Controls to add cells of the segmentation masks to the brush
        # strokes (shown only while selecting cells)
        self.brushStrokesSelectCellsToolbar = (
            widgets.BrushStrokesSelectCellsToolbar(
                'Select cells controls', self
            )
        )
        self.brushStrokesSelectCellsToolbar.setContextMenuPolicy(
            Qt.PreventContextMenu
        )
        self.brushStrokesSelectCellsToolbar.sigAddToTrainingData.connect(
            self.magicPromptsAddSelectedCellsToBrushStrokes
        )
        self.brushStrokesSelectCellsToolbar.sigSelectedIDsEdited.connect(
            self.brushStrokesSelectedIDsEdited
        )
        self.addToolBar(
            Qt.TopToolBarArea, self.brushStrokesSelectCellsToolbar
        )
        self.brushStrokesSelectCellsToolbar.setVisible(False)
        self.brushStrokesSelectCellsToolbar.keepVisibleWhenActive = True
        self.controlToolBars.append(self.brushStrokesSelectCellsToolbar)

        self.promptSegmentPointsLayerToolbar = (
            widgets.PromptableModelPointsLayerToolbar(parent=self)
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
        secondLevelToolbar = widgets.ToolBar("Second level toolbar", self)
        self.addToolBar(Qt.TopToolBarArea, secondLevelToolbar)
        self.delObjToolAction = QAction(self)
        self.delObjToolAction.setIcon(QIcon(":del_obj_click.svg"))
        self.delObjToolAction.setCheckable(True)
        self.delObjToolAction.setToolTip(
"""
Customisable delete object action

Go to the `Settings --> Customise keyboard shortcuts...` menu on the top menubar
to customise the action required to delete an object with a click.

When working with 3D segmentation masks:

- Shift+right-click to delete only the viewed z-slice.
- Ctrl+Shift to delete from the current Z-slice to the last Z-slice of the object's bounding box.
- Alt+Shift to delete from the current Z-slice to the first Z-slice of the object's bounding box.
"""
        )
        secondLevelToolbar.addAction(self.delObjToolAction)
        secondLevelToolbar.setMovable(False)
        self.secondLevelToolbar = secondLevelToolbar
        self.secondLevelToolbar.setVisible(False)

        self.exportCurrentViewToolbar = widgets.ExportCurrentViewToolbar(self)
        for name, action in self.exportCurrentViewToolbar.widgetsWithShortcut.items():
            self.widgetsWithShortcut[name] = action
        
        for name, widget in self.exportCurrentViewToolbar.widgetsForActions.items():
            self.widgetsForActions[name] = widget

        self.addToolBar(Qt.TopToolBarArea, self.exportCurrentViewToolbar)
        self.exportCurrentViewToolbar.setVisible(False)
        self.exportCurrentViewToolbar.keepVisibleWhenActive = True
        self.exportCurrentViewToolbar.sigExport.connect(
            self.exportCurrentView
        )
        self.controlToolBars.append(self.exportCurrentViewToolbar)

        self.editIDToolbar = widgets.EditIDToolbar(self)
        self.addToolBar(Qt.TopToolBarArea, self.editIDToolbar)
        self.editIDToolbar.setVisible(False)
        self.editIDToolbar.keepVisibleWhenActive = True
        self.controlToolBars.append(self.editIDToolbar)

        toolbar_targets = (
            (self.overlayToolbar, self.overlayButton),
            (self.autoPilotZoomToObjToolbar, self.autoPilotButton),
            (self.brushEraserToolBar, (self.brushButton, self.eraserButton)),
            (self.wandControlsToolbar, self.wandToolButton),
            (self.labelRoiToolbar, self.labelRoiButton),
            (self.keepIDsToolbar, self.keepIDsButton),
            (self.mergeIDsToolbar, self.mergeIDsButton),
            (self.pointsLayersToolbar, self.togglePointsLayerAction),
            (self.manualTrackingToolbar, self.manualTrackingAction),
            (self.manualBackgroundToolbar, self.manualBackgroundButton),
            (self.copyLostObjToolbar, self.copyLostObjButton),
            (self.clearFreehandRoiToolbar, self.clearFreehandRoiButton),
            (self.whitelistIDsToolbar, self.whitelistIDsButton),
            (self.magicPromptsToolbar, self.magicPromptsToolButton),
            (self.promptSegmentPointsLayerToolbar, self.magicPromptsToolButton),
            (self.exportCurrentViewToolbar, self.exportCurrentViewButton),
        )
        for toolbar, targets in toolbar_targets:
            self.searchWidget.registerToolbarTargets(toolbar, targets)

        self.gui_createToolCursorRegistry()

    def gui_populateToolSettingsMenu(self):
        brushHoverModeActionGroup = QActionGroup(self)
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
                toolName = re.findall(r'Name: (.*)', button.toolTip())[0]
            keepToolActiveNames[toolName] = button
        
        keepToolActiveNames = dict(natsorted(keepToolActiveNames.items()))
        self.keepToolActiveNames = keepToolActiveNames
        applyToNewFrameNames = {
            'Segmenting for lost IDs': self.segForLostIDsButton,
            'Delete bordering objects': self.delBorderObjAction.button,
            'Delete newly segmented objects': self.delNewObjAction.button,
        }
        apply_tools_on_new_frame_start_event_loop = (
            'Segmenting for lost IDs'
        )
        
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
            button.start_event_loop = toolName in apply_tools_on_new_frame_start_event_loop
        
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
            action.triggered.connect(
                partial(self.warnEditingWithAnnotActionTriggered, action)
            )
            self.warnEditingWithAnnotActions[key] = action
            warningsMenu.addAction(action)
            
        warningsMenu.addSeparator()
        
        checked= self._get_setting_value(
            "autoClearAssignedObjectsSecondStep",
            True,
            cast=bool
        )
        
        self.autoClearAssignedObjectsSecondStepAction = QAction()
        self.autoClearAssignedObjectsSecondStepAction.setText('Automatically clear assigned objects in second step')
        self.autoClearAssignedObjectsSecondStepAction.setCheckable(True)
        self.autoClearAssignedObjectsSecondStepAction.setChecked(checked)
        self.autoClearAssignedObjectsSecondStepAction.triggered.connect(
            lambda checked: self._set_setting_value("autoClearAssignedObjectsSecondStep", checked)
        )
        warningsMenu.addAction(self.autoClearAssignedObjectsSecondStepAction)

    def closeToolbars(self):
        for toolbar in self.sender().toolbars:
            toolbar.setVisible(False)
            for action in toolbar.actions():
                try:
                    action.button.setChecked(False)
                except Exception as e:
                    pass
