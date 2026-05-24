"""Qt view adapter for the main GUI toolbars."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QAction, QActionGroup, QButtonGroup, QToolButton

import pyqtgraph as pg

from cellacdc import widgets


class MainToolbarView:
    """Qt-facing adapter around top-level toolbar construction."""

    """Headless toolbar metadata used by the main toolbar view."""

    mode_items = (
        'Segmentation and Tracking',
        'Cell cycle analysis',
        'Viewer',
        'Custom annotations',
        'Normal division: Lineage tree',
    )

    def default_mode_items(self) -> tuple[str, ...]:
        return self.mode_items


    def __init__(self, host):
        object.__setattr__(self, 'host', host)
    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {'host'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def closeToolbars(self):
        for toolbar in self.sender().toolbars:
            toolbar.setVisible(False)
            for action in toolbar.actions():
                try:
                    action.button.setChecked(False)
                except Exception as e:
                    pass

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
        self.mode_controls_view.setEnabledFileToolbar(False)

        self.undoAction.setEnabled(False)
        self.redoAction.setEnabled(False)

        # Navigation toolbar
        navigateToolBar = widgets.ToolBar("Navigation", self.host)
        navigateToolBar.setContextMenuPolicy(Qt.PreventContextMenu)
        # navigateToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        self.addToolBar(navigateToolBar)
        navigateToolBar.addAction(self.findIdAction)
        
        navigateToolBar.addWidget(self.zoomRectButton)

        self.slideshowButton = QToolButton(self.host)
        self.slideshowButton.setIcon(QIcon(":eye-plus.svg"))
        self.slideshowButton.setCheckable(True)
        self.slideshowButton.setShortcut('Ctrl+W')
        navigateToolBar.addWidget(self.slideshowButton)
        
        navigateToolBar.addAction(self.autoPilotButton)
        
        # navigateToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        navigateToolBar.addAction(self.skipToNewIdAction)
        
        self.preprocessImageAction = QAction('Preprocess image', self.host)
        self.preprocessImageAction.setIcon(QIcon(":filter_image.svg"))
        navigateToolBar.addAction(self.preprocessImageAction)

        self.overlayButton = widgets.rightClickToolButton(parent=self.host)
        self.overlayButton.setIcon(QIcon(":overlay.svg"))
        self.overlayButton.setCheckable(True)

        self.overlayButtonAction = navigateToolBar.addWidget(self.overlayButton)
        # self.checkableButtons.append(self.overlayButton)
        # self.checkableQButtonsGroup.addButton(self.overlayButton)
        
        self.countObjsButton = QToolButton(self.host)
        self.countObjsButton.setIcon(QIcon(":count_objects.svg"))
        self.countObjsButton.setCheckable(True)
        self.countObjsButton.setShortcut('Ctrl+Shift+C')
        self.countObjsButtonAction = navigateToolBar.addWidget(
            self.countObjsButton
        )

        self.togglePointsLayerAction = QAction(
            'Activate points layer',
            self.host,
        )
        self.togglePointsLayerAction.setCheckable(True)
        self.togglePointsLayerAction.setIcon(QIcon(":pointsLayer.svg"))
        navigateToolBar.addAction(self.togglePointsLayerAction)

        self.overlayLabelsButton = widgets.rightClickToolButton(
            parent=self.host
        )
        self.overlayLabelsButton.setIcon(QIcon(":overlay_labels.svg"))
        self.overlayLabelsButton.setCheckable(True)
        # self.overlayLabelsButton.setVisible(False)
        self.overlayLabelsButtonAction = navigateToolBar.addWidget(
            self.overlayLabelsButton
        )
        self.overlayLabelsButtonAction.setVisible(False)

        self.rulerButton = QToolButton(self.host)
        self.rulerButton.setIcon(QIcon(":ruler.svg"))
        self.rulerButton.setCheckable(True)
        navigateToolBar.addWidget(self.rulerButton)
        self.checkableButtons.append(self.rulerButton)
        self.LeftClickButtons.append(self.rulerButton)

        # fluorescence image color widget
        colorsToolBar = widgets.ToolBar("Colors", self.host)

        self.overlayColorButton = pg.ColorButton(
            self.host, color=(230,230,230)
        )
        self.overlayColorButton.setDisabled(True)
        colorsToolBar.addWidget(self.overlayColorButton)

        self.textIDsColorButton = pg.ColorButton(self.host)
        colorsToolBar.addWidget(self.textIDsColorButton)

        self.addToolBar(colorsToolBar)
        colorsToolBar.setVisible(False)

        self.navigateToolBar = navigateToolBar

        # cca toolbar
        ccaToolBar = widgets.ToolBar("Cell cycle annotations", self.host)
        self.addToolBar(ccaToolBar)

        # Assign mother to bud button
        self.assignBudMothButton = QToolButton(self.host)
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
        self.setIsHistoryKnownButton = QToolButton(self.host)
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
        
        ccaToolBar.addAction(self.assignBudMothAutoAction)
        ccaToolBar.addAction(self.editCcaToolAction)
        ccaToolBar.addAction(self.reInitCcaAction)
        ccaToolBar.setVisible(False)
        self.ccaToolBar = ccaToolBar
        self.functionsNotTested3D.append(self.assignBudMothAutoAction)
        self.functionsNotTested3D.append(self.reInitCcaAction)
        self.functionsNotTested3D.append(self.editCcaToolAction)

        # Edit toolbar
        editToolBar = widgets.ToolBar("Edit", self.host)
        editToolBar.setContextMenuPolicy(Qt.PreventContextMenu)
        
        self.addToolBar(editToolBar)
        
        self.manulAnnotToolButtons = set()

        self.brushButton = QToolButton(self.host)
        self.brushButton.setIcon(QIcon(":brush.svg"))
        self.brushButton.setCheckable(True)
        editToolBar.addWidget(self.brushButton)
        self.checkableButtons.append(self.brushButton)
        self.LeftClickButtons.append(self.brushButton)
        self.brushButton.keyPressShortcut = Qt.Key_B
        self.widgetsWithShortcut['Brush'] = self.brushButton
        self.manulAnnotToolButtons.add(self.brushButton)

        self.eraserButton = QToolButton(self.host)
        self.eraserButton.setIcon(QIcon(":eraser.svg"))
        self.eraserButton.setCheckable(True)
        editToolBar.addWidget(self.eraserButton)
        self.eraserButton.keyPressShortcut = Qt.Key_X
        self.widgetsWithShortcut['Eraser'] = self.eraserButton
        self.checkableButtons.append(self.eraserButton)
        self.LeftClickButtons.append(self.eraserButton)
        self.manulAnnotToolButtons.add(self.eraserButton)

        self.curvToolButton = QToolButton(self.host)
        self.curvToolButton.setIcon(QIcon(":curvature-tool.svg"))
        self.curvToolButton.setCheckable(True)
        self.curvToolButton.setShortcut('C')
        self.curvToolButton.action = editToolBar.addWidget(self.curvToolButton)
        self.LeftClickButtons.append(self.curvToolButton)
        # self.functionsNotTested3D.append(self.curvToolButton)
        self.widgetsWithShortcut['Curvature tool'] = self.curvToolButton
        # self.checkableButtons.append(self.curvToolButton)
        self.manulAnnotToolButtons.add(self.curvToolButton)

        self.wandToolButton = QToolButton(self.host)
        self.wandToolButton.setIcon(QIcon(":magic_wand.svg"))
        self.wandToolButton.setCheckable(True)
        self.wandToolButton.setShortcut('Ctrl+D')
        self.wandToolButton.action = editToolBar.addWidget(self.wandToolButton)
        self.LeftClickButtons.append(self.wandToolButton)
        self.checkableButtons.append(self.eraserButton)
        self.widgetsWithShortcut['Magic wand'] = self.wandToolButton
        
        self.magicPromptsToolButton = QToolButton(self.host)
        self.magicPromptsToolButton.setIcon(QIcon(":magic-prompts.svg"))
        self.magicPromptsToolButton.setCheckable(True)
        self.magicPromptsToolButton.setShortcut('W')
        self.magicPromptsToolButton.action = editToolBar.addWidget(
            self.magicPromptsToolButton
        )
        self.widgetsWithShortcut['Magic prompts'] = self.magicPromptsToolButton
        
        self.drawClearRegionButton = QToolButton(self.host)
        self.drawClearRegionButton.setCheckable(True)
        self.drawClearRegionButton.setIcon(QIcon(":clear_freehand_region.svg"))
        self.widgetsWithShortcut['Clear freehand region'] = (
            self.drawClearRegionButton
        )
        self.toolsActiveInProj3Dsegm.add(self.drawClearRegionButton)
        
        self.checkableButtons.append(self.drawClearRegionButton)
        self.LeftClickButtons.append(self.drawClearRegionButton)
        
        self.drawClearRegionAction = editToolBar.addWidget(
            self.drawClearRegionButton
        )

        self.widgetsWithShortcut['Annotate mother/daughter pairing'] = (
            self.assignBudMothButton
        )
        self.widgetsWithShortcut['Annotate unknown history'] = (
            self.setIsHistoryKnownButton
        )
        
        self.copyLostObjButton = QToolButton(self.host)
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
        
        self.labelRoiButton = widgets.rightClickToolButton(parent=self.host)
        self.labelRoiButton.setIcon(QIcon(":label_roi.svg"))
        self.labelRoiButton.setCheckable(True)
        self.labelRoiButton.setShortcut('L')
        self.labelRoiButton.action = editToolBar.addWidget(self.labelRoiButton)
        self.LeftClickButtons.append(self.labelRoiButton)
        self.checkableButtons.append(self.labelRoiButton)
        self.checkableQButtonsGroup.addButton(self.labelRoiButton)
        self.widgetsWithShortcut['Label ROI'] = self.labelRoiButton
        # self.functionsNotTested3D.append(self.labelRoiButton)
        
        self.manualAnnotPastButton = QToolButton(self.host)
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

        self.segmentToolAction = QAction(
            'Segment with last used model',
            self.host,
        )
        self.segmentToolAction.setIcon(QIcon(":segment.svg"))
        self.segmentToolAction.setShortcut('R')
        self.widgetsWithShortcut['Repeat segmentation'] = self.segmentToolAction
        editToolBar.addAction(self.segmentToolAction)

        self.segForLostIDsButton = QToolButton(self.host)
        self.segForLostIDsButton.setIcon(QIcon(":segForLostIDs.svg"))
        self.segForLostIDsAction = editToolBar.addWidget(
            self.segForLostIDsButton
        )
        self.segForLostIDsButton.clicked.connect(
            self.seg_for_lost_ids_view.segForLostIDsButtonClicked
        )

        # self.SegForLostIDsButton.setShortcut('U')
        # self.widgetsWithShortcut['Unknown lineage (lineage tree)'] = self.SegForLostIDsButton
        
        self.manualBackgroundButton = QToolButton(self.host)
        self.manualBackgroundButton.setIcon(QIcon(":manual_background.svg"))
        self.manualBackgroundButton.setCheckable(True)
        self.manualBackgroundButton.setShortcut('G')
        self.LeftClickButtons.append(self.manualBackgroundButton)
        self.checkableButtons.append(self.manualBackgroundButton)
        self.checkableQButtonsGroup.addButton(self.manualBackgroundButton)
        self.widgetsWithShortcut['Manual background'] = self.manualBackgroundButton
        
        self.manualBackgroundAction = editToolBar.addWidget(
            self.manualBackgroundButton
        )
        
        self.delObjsOutSegmMaskAction = QAction(
            QIcon(":del_objs_out_segm.svg"), 
            'Select a segmentation file and delete all objects on the background', 
            self.host
        )
        self.delObjsOutSegmMaskAction.setShortcut('I')
        self.widgetsWithShortcut['Delete all objects outside segm'] = (
            self.delObjsOutSegmMaskAction
        )
        editToolBar.addAction(self.delObjsOutSegmMaskAction)

        self.hullContToolButton = QToolButton(self.host)
        self.hullContToolButton.setIcon(QIcon(":hull.svg"))
        self.hullContToolButton.setCheckable(True)
        self.hullContToolButton.setShortcut('O')
        self.hullContToolButton.action = editToolBar.addWidget(self.hullContToolButton)
        self.checkableButtons.append(self.hullContToolButton)
        self.checkableQButtonsGroup.addButton(self.hullContToolButton)
        self.functionsNotTested3D.append(self.hullContToolButton)
        self.widgetsWithShortcut['Hull contour'] = self.hullContToolButton

        self.fillHolesToolButton = QToolButton(self.host)
        self.fillHolesToolButton.setIcon(QIcon(":fill_holes.svg"))
        self.fillHolesToolButton.setCheckable(True)
        self.fillHolesToolButton.setShortcut('F')
        self.fillHolesToolButton.action = editToolBar.addWidget(
            self.fillHolesToolButton
        )
        self.checkableButtons.append(self.fillHolesToolButton)
        self.checkableQButtonsGroup.addButton(self.fillHolesToolButton)
        self.functionsNotTested3D.append(self.fillHolesToolButton)
        self.widgetsWithShortcut['Fill holes'] = self.fillHolesToolButton

        self.moveLabelToolButton = QToolButton(self.host)
        self.moveLabelToolButton.setIcon(QIcon(":moveLabel.svg"))
        self.moveLabelToolButton.setCheckable(True)
        self.moveLabelToolButton.setShortcut('P')
        self.moveLabelToolButton.action = editToolBar.addWidget(self.moveLabelToolButton)
        self.checkableButtons.append(self.moveLabelToolButton)
        self.checkableQButtonsGroup.addButton(self.moveLabelToolButton)
        self.widgetsWithShortcut['Move label'] = self.moveLabelToolButton

        self.expandLabelToolButton = QToolButton(self.host)
        self.expandLabelToolButton.setIcon(QIcon(":expandLabel.svg"))
        self.expandLabelToolButton.setCheckable(True)
        self.expandLabelToolButton.setShortcut('E')
        self.expandLabelToolButton.action = editToolBar.addWidget(self.expandLabelToolButton)
        self.expandLabelToolButton.hide()
        self.checkableButtons.append(self.expandLabelToolButton)
        self.LeftClickButtons.append(self.expandLabelToolButton)
        self.checkableQButtonsGroup.addButton(self.expandLabelToolButton)
        self.widgetsWithShortcut['Expand/shrink label'] = self.expandLabelToolButton

        self.editIDbutton = QToolButton(self.host)
        self.editIDbutton.setIcon(QIcon(":edit-id.svg"))
        self.editIDbutton.setCheckable(True)
        self.editIDbutton.setShortcut('N')
        editToolBar.addWidget(self.editIDbutton)
        self.checkableButtons.append(self.editIDbutton)
        self.checkableQButtonsGroup.addButton(self.editIDbutton)
        self.widgetsWithShortcut['Edit ID'] = self.editIDbutton

        self.separateBudButton = QToolButton(self.host)
        self.separateBudButton.setIcon(QIcon(":separate-bud.svg"))
        self.separateBudButton.setCheckable(True)
        self.separateBudButton.setShortcut('S')
        self.separateBudButton.action = editToolBar.addWidget(self.separateBudButton)
        self.checkableButtons.append(self.separateBudButton)
        self.checkableQButtonsGroup.addButton(self.separateBudButton)
        # self.functionsNotTested3D.append(self.separateBudButton)
        self.widgetsWithShortcut['Separate objects'] = self.separateBudButton

        self.mergeIDsButton = QToolButton(self.host)
        self.mergeIDsButton.setIcon(QIcon(":merge-IDs.svg"))
        self.mergeIDsButton.setCheckable(True)
        self.mergeIDsButton.setShortcut('M')
        self.mergeIDsButton.action = editToolBar.addWidget(self.mergeIDsButton)
        self.checkableButtons.append(self.mergeIDsButton)
        self.checkableQButtonsGroup.addButton(self.mergeIDsButton)
        # self.functionsNotTested3D.append(self.mergeIDsButton)
        self.widgetsWithShortcut['Merge objects'] = self.mergeIDsButton

        self.keepIDsButton = QToolButton(self.host)
        self.keepIDsButton.setIcon(QIcon(":keep_objects.svg"))
        self.keepIDsButton.setCheckable(True)
        self.keepIDsButton.action = editToolBar.addWidget(self.keepIDsButton)
        self.keepIDsButton.setShortcut('K')
        self.checkableButtons.append(self.keepIDsButton)
        self.checkableQButtonsGroup.addButton(self.keepIDsButton)
        # self.functionsNotTested3D.append(self.keepIDsButton)
        self.widgetsWithShortcut['Select objects to keep'] = self.keepIDsButton

        self.whitelistIDsButton = QToolButton(self.host)
        self.whitelistIDsButton.setIcon(QIcon(":whitelist.svg"))
        self.whitelistIDsButton.setCheckable(True)
        self.whitelistIDsButton.action = editToolBar.addWidget(
            self.whitelistIDsButton
        )
        self.whitelistIDsButton.setShortcut('Ctrl+K')
        self.checkableButtons.append(self.whitelistIDsButton)
        self.checkableQButtonsGroup.addButton(self.whitelistIDsButton)
        self.LeftClickButtons.append(self.whitelistIDsButton)
        # self.functionsNotTested3D.append(self.whitelistIDsButton)
        self.widgetsWithShortcut['Select objects to add to a tracking whitelist'] = (
            self.whitelistIDsButton
        )

        self.binCellButton = QToolButton(self.host)
        self.binCellButton.setIcon(QIcon(":bin.svg"))
        self.binCellButton.setCheckable(True)
        # self.binCellButton.setShortcut('R')
        self.binCellButton.action = editToolBar.addWidget(self.binCellButton)
        self.checkableButtons.append(self.binCellButton)
        self.checkableQButtonsGroup.addButton(self.binCellButton)
        # self.functionsNotTested3D.append(self.binCellButton)

        self.manualTrackingButton = QToolButton(self.host)
        self.manualTrackingButton.setIcon(QIcon(":manual_tracking.svg"))
        self.manualTrackingButton.setCheckable(True)
        self.manualTrackingButton.setShortcut('T')
        self.checkableQButtonsGroup.addButton(self.manualTrackingButton)
        self.checkableButtons.append(self.manualTrackingButton)
        self.widgetsWithShortcut['Manual tracking'] = self.manualTrackingButton

        self.ripCellButton = QToolButton(self.host)
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
        
        self.manualTrackingAction = editToolBar.addWidget(
            self.manualTrackingButton
        )

        self.functionsNotTested3D.append(self.repeatTrackingAction)
        self.functionsNotTested3D.append(self.manualTrackingAction)

        self.reinitLastSegmFrameAction = QAction(self.host)
        self.reinitLastSegmFrameAction.setIcon(QIcon(":reinitLastSegm.svg"))
        self.reinitLastSegmFrameAction.setVisible(False)
        editToolBar.addAction(self.reinitLastSegmFrameAction)
        editToolBar.setVisible(False)
        self.reinitLastSegmFrameAction.toolbar = editToolBar
        self.functionsNotTested3D.append(self.reinitLastSegmFrameAction)


        self.editLin_TreeBar = widgets.ToolBar("Lin Tree Edit", self.host)
        self.editLin_TreeBar.setContextMenuPolicy(Qt.PreventContextMenu)
        
        self.addToolBar(self.editLin_TreeBar)
        self.editLin_TreeGroup = QButtonGroup()
        self.editLin_TreeGroup.setExclusive(True)

        self.findNextMotherButton = QToolButton(self.host)
        self.findNextMotherButton.setIcon(QIcon(":magnGlass.svg"))
        self.findNextMotherButton.setCheckable(True)
        self.editLin_TreeBar.addWidget(self.findNextMotherButton)
        self.editLin_TreeGroup.addButton(self.findNextMotherButton)
        self.findNextMotherButton.setShortcut('F')
        self.widgetsWithShortcut['Find next potential mother (lineage tree)'] = self.findNextMotherButton

        self.unknownLineageButton = QToolButton(self.host)
        self.unknownLineageButton.setIcon(QIcon(":history.svg"))
        self.unknownLineageButton.setCheckable(True)
        self.editLin_TreeBar.addWidget(self.unknownLineageButton)
        self.editLin_TreeGroup.addButton(self.unknownLineageButton)
        self.unknownLineageButton.setShortcut('U')
        self.widgetsWithShortcut['Unknown lineage (lineage tree)'] = self.unknownLineageButton

        self.noToolLinTreeButton = QToolButton(self.host)
        self.noToolLinTreeButton.setIcon(QIcon(":arrow_cursor.svg"))
        self.noToolLinTreeButton.setCheckable(True)
        self.editLin_TreeBar.addWidget(self.noToolLinTreeButton)
        self.editLin_TreeGroup.addButton(self.noToolLinTreeButton)
        self.noToolLinTreeButton.setShortcut('N')
        self.widgetsWithShortcut['No tool (lineage tree)'] = self.noToolLinTreeButton

        self.propagateLinTreeButton = QToolButton(self.host)
        self.propagateLinTreeButton.setIcon(QIcon(":compute.svg"))
        self.editLin_TreeBar.addWidget(self.propagateLinTreeButton)
        self.propagateLinTreeButton.setShortcut('P')
        self.widgetsWithShortcut['Propagate (lineage tree)'] = self.propagateLinTreeButton
        self.propagateLinTreeButton.clicked.connect(self.propagateLinTreeAction)

        self.viewLinTreeInfoButton = QToolButton(self.host)
        self.viewLinTreeInfoButton.setIcon(QIcon(":addCustomAnnotation.svg"))
        self.editLin_TreeBar.addWidget(self.viewLinTreeInfoButton)
        self.viewLinTreeInfoButton.setShortcut('S')
        self.widgetsWithShortcut['View Changes (lineage tree)'] = self.viewLinTreeInfoButton
        self.viewLinTreeInfoButton.clicked.connect(self.viewLinTreeInfoAction)
    

        self.modeItems = list(self.mode_items())

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

    def gui_createAnnotateToolbar(self):
        # Edit toolbar
        self.annotateToolbar = widgets.ToolBar(
            "Custom annotations",
            self.host,
        )
        self.annotateToolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.addToolBar(Qt.LeftToolBarArea, self.annotateToolbar)
        self.annotateToolbar.addAction(self.loadCustomAnnotationsAction)
        self.annotateToolbar.addAction(self.addCustomAnnotationAction)
        self.annotateToolbar.addAction(self.viewAllCustomAnnotAction)
        self.annotateToolbar.setVisible(False)