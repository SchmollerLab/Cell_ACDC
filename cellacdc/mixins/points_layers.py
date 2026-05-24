"""Qt view adapter for points-layer workflows."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from functools import partial

import matplotlib
import numpy as np
import pyqtgraph as pg
import skimage.draw
import skimage.measure
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QLabel

from cellacdc import _warnings, apps, colors, exception_handler, html_utils, widgets


class PointsLayersMixin:
    """Qt-facing adapter around points-layer workflows."""

    """Headless decisions for points-layer GUI workflows."""

    recovery_tolerance_seconds = 15

    def addClickedPoint(self, action, x, y, id):
        x, y = round(x, 2), round(y, 2)
        posData = self.data[self.pos_i]
        pointsDataPos = action.pointsData.get(self.pos_i)
        if pointsDataPos is None:
            action.pointsData[self.pos_i] = {}

        framePointsData = action.pointsData[self.pos_i].get(posData.frame_i)
        if action.snapToMax:
            radius = round(action.pointSize / 2)
            rr, cc = skimage.draw.disk((round(y), round(x)), radius)
            idx_max = (self.img1.image[rr, cc]).argmax()
            y, x = rr[idx_max], cc[idx_max]

        if framePointsData is None:
            if posData.SizeZ > 1:
                zSlice = self.zSliceScrollBar.sliderPosition()
                action.pointsData[self.pos_i][posData.frame_i] = {
                    zSlice: {"x": [x], "y": [y], "id": [id]}
                }
            else:
                action.pointsData[self.pos_i][posData.frame_i] = {
                    "x": [x],
                    "y": [y],
                    "id": [id],
                }
        else:
            if posData.SizeZ > 1:
                zSlice = self.zSliceScrollBar.sliderPosition()
                z_data = framePointsData.get(zSlice)
                if z_data is None:
                    framePointsData[zSlice] = {"x": [x], "y": [y], "id": [id]}
                else:
                    framePointsData[zSlice]["x"].append(x)
                    framePointsData[zSlice]["y"].append(y)
                    framePointsData[zSlice]["id"].append(id)
                action.pointsData[self.pos_i][posData.frame_i] = framePointsData
            else:
                pointsDataPos = action.pointsData[self.pos_i]
                framePointsData = pointsDataPos[posData.frame_i]
                framePointsData["x"].append(x)
                framePointsData["y"].append(y)
                framePointsData["id"].append(id)

        self.markPointsLayerDirty(action=action)

    def addPointsByClickingButtonToggled(self, checked=True, sender=None):
        if sender is None:
            sender = self.sender()
        if not sender.isChecked():
            action = sender.action
            action.scatterItem.setVisible(False)
            return

        self.disconnectLeftClickButtons()
        self.uncheckLeftClickButtons(sender)
        self.connectLeftClickButtons()
        action = sender.action
        action.scatterItem.setVisible(True)
        self.ax1_BrushCircle.setBrush(action.brushColor)
        self.ax1_BrushCircle.setPen(action.penColor)

    def addPointsByClickingScatterItemHoverEntered(self, item, points, event):
        point = points[0]
        point_id = point.data()
        toolButton = item.action.button
        toolButton.rightClickIDSpinbox.prevId = toolButton.rightClickIDSpinbox.value()
        toolButton.rightClickIDSpinbox.setValue(point_id)

    @exception_handler
    def addPointsLayer(self, toolbar=None):
        proceed = self.checkLoadedTableIds(toolbar)

        if self.addPointsWin.cancel or not proceed:
            self.addPointsWin = None
            self.logger.info("Adding points layer cancelled.")
            return

        if toolbar is None:
            toolbar = self.pointsLayersToolbar

        symbol = self.addPointsWin.symbol
        color = self.addPointsWin.color
        pointSize = self.addPointsWin.pointSize
        zRadius = int((self.addPointsWin.zHeight - 1) / 2)
        r, g, b, a = color.getRgb()

        scatterItem = widgets.PointsScatterPlotItem(
            [],
            [],
            ax=self.ax1,
            symbol=symbol,
            pxMode=False,
            size=pointSize,
            brush=pg.mkBrush(color=(r, g, b, 100)),
            pen=pg.mkPen(width=2, color=(r, g, b)),
            hoverable=True,
            hoverBrush=pg.mkBrush((r, g, b, 200)),
            tip=None,
            show_data_as_tip=True,
        )
        self.ax1.addItem(scatterItem)

        toolButton = widgets.PointsLayerToolButton(symbol, color, parent=self)
        toolButton.actions = []
        toolButton.setCheckable(True)
        toolButton.setChecked(True)
        if self.addPointsWin.keySequence is not None:
            toolButton.setShortcut(self.addPointsWin.keySequence)
        toolButton.toggled.connect(self.pointLayerToolbuttonToggled)
        toolButton.sigEditAppearance.connect(self.editPointsLayerAppearance)
        toolButton.sigShowIdsToggled.connect(self.showPointsLayerIdsToggled)
        toolButton.sigRemove.connect(partial(self.removePointsLayer, toolbar=toolbar))

        action = toolbar.addWidget(toolButton)
        action.state = self.addPointsWin.state()

        toolButton.action = action
        action.brushColor = (r, g, b, 100)
        action.brushColorId0 = (
            *colors.hex_to_rgb(
                colors.lighten_color(np.array(action.brushColor) / 255, 0.3)
            ),
            100,
        )
        action.penColor = (r, g, b)
        action.penColorId0 = colors.lighten_color(np.array(action.penColor) / 255, 0.3)
        action.pointSize = pointSize
        action.zRadius = zRadius
        action.button = toolButton
        action.scatterItem = scatterItem
        scatterItem.action = action
        action.layerType = self.addPointsWin.layerType
        action.layerTypeIdx = self.addPointsWin.layerTypeIdx
        action.loadedDf = self.addPointsWin.loadedDf
        self.data[self.pos_i]
        action.pointsData = {}
        action.pointsData[self.pos_i] = self.addPointsWin.pointsData
        action.snapToMax = False
        action.loadedDfInfo = self.addPointsWin.loadedDfInfo
        self.setPointsLayerLoadedDfEndanme(action)

        if self.addPointsWin.layerType.startswith("Click to annotate point"):
            action.snapToMax = self.addPointsWin.snapToMaxToggle.isChecked()
            isLoadedDf = self.addPointsWin.clickEntryIsLoadedDf
            self.setupAddPointsByClicking(toolButton, isLoadedDf, toolbar=toolbar)
            if self.addPointsWin.autoPilotToggle.isChecked():
                self.autoPilotZoomToObjToggle.setChecked(True)

        weighingChannel = self.addPointsWin.weighingChannel
        self.loadPointsLayerWeighingData(action, weighingChannel)

        self.drawPointsLayers()

        if toolbar == self.promptSegmentPointsLayerToolbar:
            self.promptSegmentPointsLayerToolbar.isPointsLayerInit = True
            self.magicPromptsToolbar.clearPointsAction.setDisabled(False)
            self.magicPromptsToolbar.clearPointsActionOnZoom.setDisabled(False)
            QTimer.singleShot(200, self.magicPromptsToolbar.selectModelAction.trigger)

        self.addPointsWin = None

    def addPointsLayerTriggered(self, checked=False, toolbar=None):
        if toolbar is None:
            toolbar = self.pointsLayersToolbar

        if self.addPointsWin is not None:
            self.logger.info("Add points layer window is already open. Cannot add now.")
            return

        onlyMouseClicks = toolbar == self.promptSegmentPointsLayerToolbar
        posData = self.data[self.pos_i]
        self.addPointsWin = apps.AddPointsLayerDialog(
            channelNames=posData.chNames,
            imagesPath=posData.images_path,
            hideCentroidsSection=onlyMouseClicks,
            hideWeightedCentroidsSection=onlyMouseClicks,
            hideFromTableSection=onlyMouseClicks,
            hideManualEntrySection=onlyMouseClicks,
            hideWithMouseClicksSection=False,
            parent=self,
        )
        cmap = matplotlib.colormaps["gist_rainbow"]
        i = np.random.default_rng(seed=123).uniform()
        for action in toolbar.actions()[1:]:
            if not hasattr(action, "layerTypeIdx"):
                continue
            rgb = [round(c * 255) for c in cmap(i)][:3]
            self.addPointsWin.appearanceGroupbox.colorButton.setColor(rgb)
            break

        self.addPointsWin.sigCriticalReadTable.connect(self.logger.info)
        self.addPointsWin.sigLoadedTable.connect(self.logLoadedTablePointsLayer)
        self.addPointsWin.sigClosed.connect(
            partial(self.addPointsLayer, toolbar=toolbar)
        )
        self.addPointsWin.sigCheckClickEntryTableEndnameExists.connect(
            self.checkClickEntryTableEndnameExists
        )
        self.addPointsWin.show()
        if self.addPointsWin.clickEntryRadiobutton.isChecked():
            QTimer.singleShot(
                200,
                partial(
                    self.addPointsWin.sigCheckClickEntryTableEndnameExists.emit,
                    self.addPointsWin.clickEntryTableEndname.text(),
                    False,
                ),
            )

    def askLoadNewerRecoveryClickEntryDfs(self, tableEndName, newer_recovery_filepaths):
        if not newer_recovery_filepaths:
            return False

        num_tables = len(newer_recovery_filepaths)
        filepath, recovery_filepath = newer_recovery_filepaths[0]
        main_timestamp = datetime.fromtimestamp(os.path.getmtime(filepath)).strftime(
            "%a %d. %b. %y - %H:%M:%S"
        )
        recovery_timestamp = datetime.fromtimestamp(
            os.path.getmtime(recovery_filepath)
        ).strftime("%a %d. %b. %y - %H:%M:%S")

        if num_tables == 1:
            text = html_utils.paragraph(
                f"A newer recovery version of <code>{tableEndName}.csv</code> "
                "was found.<br><br>"
                f"Main table save date: <code>{main_timestamp}</code><br>"
                f"Recovery save date: <code>{recovery_timestamp}</code><br><br>"
                "Do you want to load the newer recovery version?"
            )
        else:
            text = html_utils.paragraph(
                f"Newer recovery versions of <code>{tableEndName}.csv</code> "
                f"were found for <b>{num_tables} positions</b>.<br><br>"
                f"Example main table save date: <code>{main_timestamp}</code><br>"
                f"Example recovery save date: <code>{recovery_timestamp}</code><br><br>"
                "Do you want to load the newer recovery version where available?"
            )

        msg = widgets.myMessageBox(wrapText=False)
        _, yesButton, _ = msg.warning(
            self.addPointsWin,
            "Newer recovery table found",
            text,
            buttonsTexts=("Cancel", "Yes, load newer recovery", "No, load main table"),
        )
        return msg.clickedButton == yesButton

    def askSaveAddedPoints(self):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("Do you want to <b>save the annotated points</b>?")
        _, noButton, yesButton = msg.question(
            self, "Save?", txt, buttonsTexts=("Cancel", "No", "Yes")
        )
        if msg.clickedButton != yesButton:
            return

        for toolbar in self.pointsLayersToolbars:
            for action in self.pointsLayersToolbar.actions():
                try:
                    if "Save annotated" in action.text():
                        action.trigger()
                except Exception:
                    pass

    def askSavePointsLayer(self, action):
        toolButton = action.button
        tableEndName = toolButton.clickEntryTableEndName
        saveAction = toolButton.saveAction

        txt = html_utils.paragraph(f"""
            Do you want to <b>save</b> the points you added
            (table called <code>{tableEndName}.csv</code>)?
        """)
        msg = widgets.myMessageBox(wrapText=False)
        _, _, saveButton = msg.question(
            self,
            "Save points layer?",
            txt,
            buttonsTexts=("Cancel", "No, do not save", "Yes, save points"),
        )
        if msg.clickedButton == saveButton:
            self.savePointsAddedByClicking(saveAction.saveToolbutton, None)

        return msg.cancel

    def autoPilotZoomToObjToggled(self, checked):
        if not checked:
            self.zoomOut()
            return

        posData = self.data[self.pos_i]
        if not posData.IDs:
            self.logger.info("There are no objects in current segmentation mask")
            return
        self.autoPilotZoomToObjSpinBox.setValue(posData.IDs[0])
        self.zoomToObj(posData.rp[0])

    def autoZoomNextObj(self):
        self.sender().setValue(self.sender().value() - 1)
        self.pointsLayerAutoPilot("next")
        self.setFocusMain()
        self.setFocusGraphics()

    def autoZoomPrevObj(self):
        self.sender().setValue(self.sender().value() + 1)
        self.pointsLayerAutoPilot("prev")
        self.setFocusMain()
        self.setFocusGraphics()

    def buttonAddPointsByClickingActive(self):
        for toolbar in self.pointsLayersToolbars:
            for action in toolbar.actions()[1:]:
                if not hasattr(action, "layerTypeIdx"):
                    continue
                if action.layerTypeIdx == 4 and action.button.isChecked():
                    return action.button

    def checkAskSavePointsLayers(self):
        for toolbar in self.pointsLayersToolbars:
            for action in toolbar.actions()[1:]:
                if not hasattr(action, "layerTypeIdx"):
                    continue
                if action.layerTypeIdx != 4:
                    continue

                scatterItem = action.scatterItem
                xx, yy = scatterItem.getData()

                if xx is None or len(xx) == 0:
                    toolButton = action.button
                    tableEndName = toolButton.clickEntryTableEndName
                    # Check in other loaded pos
                    are_there_points_to_save = False
                    for pos_i, _posData in enumerate(self.data):
                        if pos_i == self.pos_i:
                            continue

                        df = _posData.clickEntryPointsDfs.get(tableEndName)
                        if df is None:
                            continue

                        are_there_points_to_save = True
                        break

                    if not are_there_points_to_save:
                        continue

                cancel = self.askSavePointsLayer(action)
                if cancel:
                    return cancel

        return False

    def checkClickEntryTableEndnameExists(self, tableEndName, forceLoading):
        doesTableExists = False
        for posData in self.data:
            filepath, _ = self.getClickEntryTableFilepaths(posData, tableEndName)
            if os.path.exists(filepath):
                doesTableExists = True
                break

        if not doesTableExists:
            return

        if not forceLoading:
            msg = widgets.myMessageBox(wrapText=False)
            txt = html_utils.paragraph(
                f"The table <code>{tableEndName}.csv</code> already exists!<br><br>"
                "Do you want to load it?"
            )
            _, yesButton, _ = msg.warning(
                self.addPointsWin,
                "Table exists!",
                txt,
                buttonsTexts=("Cancel", "Yes, load it", "No, let me enter a new name"),
            )
            if msg.clickedButton != yesButton:
                return

        newer_recovery_filepaths = self.getClickEntryNewerRecoveryFilepaths(
            tableEndName
        )
        load_recovery_if_newer = self.askLoadNewerRecoveryClickEntryDfs(
            tableEndName, newer_recovery_filepaths
        )

        self.loadClickEntryDfs(tableEndName, loadRecoveryIfNewer=load_recovery_if_newer)

    def checkLoadedTableIds(self, toolbar):
        if toolbar != self.promptSegmentPointsLayerToolbar:
            return True

        for posData in self.data:
            for tableEndName, df in posData.clickEntryPointsDfs.items():
                for point_id in df["id"].values:
                    if point_id in posData.IDs_idxs:
                        proceed = self.warnAddingPointWithExistingId(
                            point_id, table_endname=tableEndName
                        )
                        return proceed

        return True

    def clearPointsLayers(self):
        for toolbar in self.pointsLayersToolbars:
            for action in toolbar.actions()[1:]:
                try:
                    action.scatterItem.clear()
                except Exception:
                    continue

    def click_entry_table_filename(
        self,
        basename: str,
        table_endname: str,
    ) -> str:
        table_basename = basename if basename.endswith("_") else f"{basename}_"
        filename = f"{table_basename}{table_endname}"
        if not filename.endswith(".csv"):
            filename = f"{filename}.csv"
        return filename

    def drawPointsLayers(self, computePointsLayers=True):
        posData = self.data[self.pos_i]
        for toolbar in self.pointsLayersToolbars:
            for action in toolbar.actions()[1:]:
                if not hasattr(action, "layerTypeIdx"):
                    continue

                if action.layerTypeIdx < 2 and computePointsLayers:
                    self.getCentroidsPointsData(action)

                if not action.button.isChecked():
                    continue

                frames = action.pointsData.get(self.pos_i, set())
                if posData.frame_i not in frames:
                    if action.layerTypeIdx != 4:
                        self.logger.info(
                            f"Frame number {posData.frame_i + 1} does not have any "
                            f'"{action.layerType}" point to display.'
                        )
                    continue

                framePointsData = action.pointsData[self.pos_i][posData.frame_i]

                if "x" not in framePointsData:
                    # 3D points
                    zProjHow = self.zProjComboBox.currentText()
                    isZslice = zProjHow == "single z-slice" and posData.SizeZ > 1
                    if isZslice:
                        xx, yy, ids, data = [], [], [], []
                        zSlice = self.zSliceScrollBar.sliderPosition()
                        zRadius = action.zRadius
                        zRange = range(zSlice - zRadius, zSlice + zRadius + 1)
                        for z in zRange:
                            z_data = framePointsData.get(z)
                            if z_data is None:
                                continue
                            xx.extend(z_data["x"])
                            yy.extend(z_data["y"])
                            ids.extend(z_data["id"])
                            try:
                                data.extend(z_data["data"])
                            except KeyError:
                                # data is needed only for loaded tables
                                pass
                    else:
                        xx, yy, ids, data = [], [], [], []
                        # z-projection --> draw all points
                        for z, z_data in framePointsData.items():
                            xx.extend(z_data["x"])
                            yy.extend(z_data["y"])
                            ids.extend(z_data["id"])
                            try:
                                data.extend(z_data["data"])
                            except KeyError:
                                # data is needed only for loaded tables
                                pass
                else:
                    # 2D segmentation
                    xx = framePointsData["x"]
                    yy = framePointsData["y"]
                    ids = framePointsData["id"]
                    try:
                        data = framePointsData["data"]
                    except KeyError:
                        # data is needed only for loaded tables
                        pass

                brushColors = [
                    action.brushColor if id != 0 else action.brushColorId0 for id in ids
                ]
                brushes = [pg.mkBrush(color) for color in brushColors]

                pensColor = [
                    action.penColor if id != 0 else action.penColorId0 for id in ids
                ]
                pens = [pg.mkPen(color) for color in pensColor]

                if action.layerTypeIdx == 2:
                    # For loaded table show the rest of the table as a tooltip
                    data = data
                    show_data_as_tip = True
                else:
                    data = ids
                    show_data_as_tip = False

                xx = np.array(xx)  # + 0.5
                yy = np.array(yy)  # + 0.5

                action.scatterItem.show_data_as_tip = show_data_as_tip
                action.scatterItem.setData(xx, yy, data=data, brush=brushes, pen=pens)

    def editPointsLayerAppearance(self, button):
        win = apps.EditPointsLayerAppearanceDialog(parent=self)
        win.restoreState(button.action.state)
        win.exec_()
        if win.cancel:
            return

        symbol = win.symbol
        color = win.color
        pointSize = win.pointSize
        zRadius = int((win.zHeight - 1) / 2)
        r, g, b, a = color.getRgb()

        scatterItem = button.action.scatterItem
        scatterItem.opts["hoverBrush"] = pg.mkBrush((r, g, b, 200))
        scatterItem.setSymbol(symbol, update=False)
        scatterItem.setBrush(pg.mkBrush(color=(r, g, b, 100)), update=False)
        scatterItem.setPen(pg.mkPen(width=2, color=(r, g, b)), update=False)
        scatterItem.setSize(pointSize, update=True)

        button.action.brushColor = (r, g, b, 100)
        button.action.penColor = (r, g, b)
        button.action.pointSize = pointSize
        button.action.zRadius = zRadius

        button.action.state = win.state()

    def flushDirtyPointsLayersAutosave(self):
        if not self.dirtyPointsLayerTableEndNames:
            return

        for tableEndName in tuple(
            self.dirtyPointsLayerTableEndNames
        ):  # avoid runtime error
            self.savePointsAddedByClickingFromEndname(tableEndName, recovery=True)

        self.dirtyPointsLayerTableEndNames.clear()

    def getAddedPointId(
        self,
        isMagicPrompts,
        addPointsByClickingButton,
        right_click,
        left_click,
        middle_click,
    ):
        action = addPointsByClickingButton.action
        if right_click:
            id = addPointsByClickingButton.rightClickIDSpinbox.value()
        elif left_click:
            id = addPointsByClickingButton.pointIdSpinbox.value()
            id = self.getClickedPointNewId(
                action,
                id,
                addPointsByClickingButton.pointIdSpinbox,
                isMagicPrompts=isMagicPrompts,
            )
            if isMagicPrompts:
                proceed = self.warnAddingPointWithExistingId(id)
                if not proceed:
                    return

            addPointsByClickingButton.pointIdSpinbox.setValue(id)
        elif middle_click:
            id = 0

        return id

    def getCentroidsPointsData(self, action):
        # Centroids (either weighted or not)
        # NOTE: if user requested to draw from table we load that in
        # apps.AddPointsLayerDialog.ok_cb()
        posData = self.data[self.pos_i]
        action.pointsData[self.pos_i] = {posData.frame_i: {}}
        if hasattr(action, "weighingData"):
            lab = posData.lab
            img = action.weighingData[self.pos_i][posData.frame_i]
            rp = skimage.measure.regionprops(lab, intensity_image=img)
            attr = "weighted_centroid"
        else:
            rp = posData.rp
            attr = "centroid"
        for i, obj in enumerate(rp):
            centroid = getattr(obj, attr)
            if len(centroid) == 3:
                zc, yc, xc = centroid
                z_int = round(zc)
                if z_int not in action.pointsData[self.pos_i][posData.frame_i]:
                    action.pointsData[self.pos_i][posData.frame_i][z_int] = {
                        "x": [xc],
                        "y": [yc],
                        "id": [obj.label],
                    }
                else:
                    z_data = action.pointsData[self.pos_i][posData.frame_i][z_int]
                    z_data["x"].append(xc)
                    z_data["y"].append(yc)
                    z_data["id"].append(obj.label)
            else:
                yc, xc = centroid
                if "y" not in action.pointsData[self.pos_i][posData.frame_i]:
                    action.pointsData[self.pos_i][posData.frame_i]["y"] = [yc]
                    action.pointsData[self.pos_i][posData.frame_i]["x"] = [xc]
                    action.pointsData[self.pos_i][posData.frame_i]["id"] = [obj.label]
                else:
                    action.pointsData[self.pos_i][posData.frame_i]["y"].append(yc)
                    action.pointsData[self.pos_i][posData.frame_i]["x"].append(xc)
                    action.pointsData[self.pos_i][posData.frame_i]["id"].append(
                        obj.label
                    )

    def getClickEntryNewerRecoveryFilepaths(self, tableEndName):
        newer_recovery_filepaths = []
        for posData in self.data:
            filepath, recovery_filepath = self.getClickEntryTableFilepaths(
                posData, tableEndName
            )
            if not os.path.exists(filepath) or not os.path.exists(recovery_filepath):
                continue

            if (
                os.path.getmtime(recovery_filepath) <= os.path.getmtime(filepath) + 15
            ):  # add a 15 second tolerance
                continue

            newer_recovery_filepaths.append((filepath, recovery_filepath))

        return newer_recovery_filepaths

    def getClickEntryTableFilepaths(self, posData, tableEndName):
        if posData.basename.endswith("_"):
            basename = posData.basename
        else:
            basename = f"{posData.basename}_"

        csv_filename = f"{basename}{tableEndName}"
        if not csv_filename.endswith(".csv"):
            csv_filename = f"{csv_filename}.csv"

        filepath = os.path.join(posData.images_path, csv_filename)
        recovery_filepath = os.path.join(posData.images_path, "recovery", csv_filename)
        return filepath, recovery_filepath

    def getClickedPointNewId(
        self, action, current_id, pointIdSpinbox, isMagicPrompts=False
    ):
        removed_id = getattr(pointIdSpinbox, "removedId", None)
        if removed_id is not None:
            pointIdSpinbox.removedId = None
            return removed_id

        posData = self.data[self.pos_i]
        if isMagicPrompts:
            is_already_new = self.isPointIdAlreadyNew(current_id, action)
            if is_already_new:
                return current_id

            new_ID = self.setBrushID(return_val=True)
            new_id = max(current_id, new_ID) + 1
            return new_id
        else:
            pointsDataPos = action.pointsData.get(self.pos_i)
            if pointsDataPos is None:
                return 1

            framePointsData = pointsDataPos.get(posData.frame_i)
            if framePointsData is None:
                return 1
            if posData.SizeZ > 1:
                new_id = 1
                for z_data in framePointsData.values():
                    max_id = max(z_data.get("id", 0), default=0) + 1
                    if max_id > new_id:
                        new_id = max_id
            else:
                new_id = max(framePointsData.get("id", 0), default=0) + 1
            if current_id >= new_id:
                return current_id
            return new_id

    def isPointIdAlreadyNew(self, point_id, action):
        posData = self.data[self.pos_i]
        if point_id in posData.IDs_idxs:
            return False

        is_ID = point_id in posData.IDs_idxs
        pointsDataPos = action.pointsData.get(self.pos_i)
        if pointsDataPos is None:
            return not is_ID

        framePointsData = pointsDataPos.get(posData.frame_i)
        if framePointsData is None:
            return not is_ID

        if "x" not in framePointsData:
            is_id_already_added = False
            for z, z_data in framePointsData.items():
                if point_id in z_data["id"]:
                    is_id_already_added = True
                    break
        else:
            is_id_already_added = point_id in framePointsData["id"]

        is_already_new = not is_ID and not is_id_already_added
        return is_already_new

    def loadClickEntryDfs(self, tableEndName, loadRecoveryIfNewer=False):
        for posData in self.data:
            filepath, recovery_filepath = self.getClickEntryTableFilepaths(
                posData, tableEndName
            )

            if loadRecoveryIfNewer:
                recovery_exists = os.path.exists(recovery_filepath)
                main_exists = os.path.exists(filepath)
                if recovery_exists and (
                    not main_exists
                    or os.path.getmtime(recovery_filepath)
                    > os.path.getmtime(filepath) + 15
                ):
                    filepath = recovery_filepath
                elif not main_exists:
                    continue

            if not os.path.exists(filepath):
                continue

            self.logger.info(f'Loading points from "{filepath}"...')
            df = pd.read_csv(filepath)
            if "id" not in df.columns:
                df["id"] = range(1, len(df) + 1)
            posData.clickEntryPointsDfs[tableEndName] = df

        try:
            self.addPointsWin.loadButton.confirmAction()
        except Exception:
            pass

    def loadPointsLayerWeighingData(self, action, weighingChannel):
        if not weighingChannel:
            return

        self.logger.info(f'Loading "{weighingChannel}" weighing data...')
        action.weighingData = []
        for p, posData in enumerate(self.data):
            if weighingChannel == posData.user_ch_name:
                wData = posData.img_data
                action.weighingData.append(wData)
                continue

            path, filename = self.getPathFromChName(weighingChannel, posData)
            if path is None:
                self.criticalFluoChannelNotFound(weighingChannel, posData)
                action.weighingData = []
                return

            if filename in posData.fluo_data_dict:
                # Weighing data already loaded as additional fluo channel
                wData = posData.fluo_data_dict[filename]
            else:
                # Weighing data never loaded --> load now
                wData, _ = self.load_fluo_data(path)
                if posData.SizeT == 1:
                    wData = wData[np.newaxis]
            action.weighingData.append(wData)

    def logLoadedTablePointsLayer(self, df, filename: str):
        separator = "-" * 100
        header = f'First 10 rows of loaded table - "{filename}":'
        footer = f"Number of points: {len(df)}"
        text = f"{separator}\n{header}\n\n{df.head(10)}\n\n{footer}\n{separator}"
        if filename:
            text = f"{text}\nFilename: {filename}"
        self.logger.info(text)

    def markPointsLayerDirty(self, tableEndName=None, action=None):
        if tableEndName is None and action is not None:
            tableEndName = getattr(action, "clickEntryTableEndName", None)

        if tableEndName is None:
            addPointsByClickingButton = self.buttonAddPointsByClickingActive()
            if addPointsByClickingButton is None:
                return
            tableEndName = addPointsByClickingButton.clickEntryTableEndName

        self.dirtyPointsLayerTableEndNames.add(tableEndName)

    def pointLayerToolbuttonToggled(self, checked):
        action = self.sender().action
        action.scatterItem.setVisible(checked)

    def pointsLayerAutoPilot(self, direction):
        if not self.autoPilotZoomToObjToggle.isChecked():
            return
        ID = self.autoPilotZoomToObjSpinBox.value()
        posData = self.data[self.pos_i]
        if not posData.IDs:
            return

        try:
            ID_idx = posData.IDs_idxs[ID]
            if direction == "next":
                nextID_idx = ID_idx + 1
            else:
                nextID_idx = ID_idx - 1
            obj = posData.rp[nextID_idx]
        except Exception:
            self.logger.info("Auto-pilot restarted from first ID")
            obj = posData.rp[0]

        self.autoPilotZoomToObjSpinBox.setValue(obj.label)
        self.zoomToObj(obj)

    def pointsLayerClicksDfsToData(self, posData, toolbar=None):
        if toolbar is None:
            toolbar = self.pointsLayersToolbar

        for action in toolbar.actions()[1:]:
            if not hasattr(action, "button"):
                continue

            if not hasattr(action.button, "clickEntryTableEndName"):
                continue
            tableEndName = action.button.clickEntryTableEndName
            action.pointsData[self.pos_i] = {}
            if posData.clickEntryPointsDfs.get(tableEndName) is None:
                continue

            df = posData.clickEntryPointsDfs[tableEndName]

            if posData.SizeZ > 1 and df["z"].isna().any():
                self.warnLoadedPointsTableIsNot3D(tableEndName)
                return

            for frame_i, df_frame in df.groupby("frame_i"):
                action.pointsData[self.pos_i][frame_i] = {}
                if posData.SizeZ > 1:
                    for z, df_zlice in df_frame.groupby("z"):
                        xx = df_zlice["x"].to_list()
                        yy = df_zlice["y"].to_list()
                        ids = df_zlice["id"].to_list()
                        action.pointsData[self.pos_i][frame_i][z] = {
                            "x": xx,
                            "y": yy,
                            "id": ids,
                        }
                else:
                    xx = df_frame["x"].to_list()
                    yy = df_frame["y"].to_list()
                    ids = df_frame["id"].to_list()
                    action.pointsData[self.pos_i][frame_i] = {
                        "x": xx,
                        "y": yy,
                        "id": ids,
                    }

    def pointsLayerDataToDf(self, posData, getOnlyActive=False, toolbar=None):
        df = None
        for toolbar in self.pointsLayersToolbars:
            for action in toolbar.actions()[1:]:
                if not hasattr(action, "button"):
                    continue
                if not hasattr(action.button, "clickEntryTableEndName"):
                    continue

                tableEndName = action.button.clickEntryTableEndName
                if getOnlyActive and not action.button.isChecked():
                    continue

                df = toolbar.fromActionToDataFrame(
                    action, posData, isSegm3D=self.isSegm3D
                )
                posData.clickEntryPointsDfs[tableEndName] = df
        return df

    def pointsLayerDfsToData(self, posData):
        self.pointsLayerClicksDfsToData(posData)

    def pointsLayerLoadedDfsToData(self):
        posData = self.data[self.pos_i]
        for toolbar in self.pointsLayersToolbars:
            for action in toolbar.actions()[1:]:
                if not hasattr(action, "loadedDfInfo"):
                    continue

                if action.loadedDfInfo is None:
                    continue

                endname = action.loadedDfInfo.get("endname")
                if endname is None:
                    continue

                filename = f"{posData.basename}{endname}"
                filepath = os.path.join(posData.images_path, filename)
                if not os.path.exists(filepath):
                    action.pointsData[self.pos_i] = {}

                df = load.load_df_points_layer(filepath)
                action.pointsData[self.pos_i] = load.loaded_df_to_points_data(
                    df,
                    action.loadedDfInfo["t"],
                    action.loadedDfInfo["z"],
                    action.loadedDfInfo["y"],
                    action.loadedDfInfo["x"],
                )
                self.logLoadedTablePointsLayer(df, filename=filename)

    def pointsLayerToggled(self, checked):
        if not checked:
            for action in self.pointsLayersToolbar.actions():
                try:
                    if "Save annotated" in action.text():
                        self.askSaveAddedPoints()
                        break
                except Exception:
                    pass
        self.pointsLayersToolbar.setVisible(checked)
        self.autoPilotZoomToObjToolbar.setVisible(checked)
        if self.pointsLayersNeverToggled:
            self.pointsLayersToolbar.sigAddPointsLayer.emit()
        self.pointsLayersNeverToggled = False
        QTimer.singleShot(200, self.autoRange)

    def reinitPointsLayers(self):
        for toolbar in self.pointsLayersToolbars:
            for action in toolbar.actions()[1:]:
                toolbar.removeAction(action)
            toolbar.setVisible(False)
            self.autoPilotZoomToObjToolbar.setVisible(False)

    def removeClickedPoints(self, action, points):
        posData = self.data[self.pos_i]
        framePointsData = action.pointsData[self.pos_i][posData.frame_i]
        if posData.SizeZ > 1:
            zProjHow = self.zProjComboBox.currentText()
            if zProjHow != "single z-slice":
                _warnings.warnCannotAddRemovePointsProjection()
                return
            zSlice = self.zSliceScrollBar.sliderPosition()
        else:
            zSlice = None

        removed_ids = []
        for point in points:
            pos = point.pos()
            x, y = pos.x(), pos.y()
            if zSlice is not None:
                zSliceRad = action.zRadius
                sliceFramePointsData = [
                    framePointsData[z]
                    for z in range(zSlice - zSliceRad, zSlice + zSliceRad + 1)
                    if z in framePointsData.keys()
                ]
            else:
                sliceFramePointsData = [framePointsData]

            for sliceFramePointsData in sliceFramePointsData:
                if point.data() in sliceFramePointsData["id"]:
                    sliceFramePointsData["x"].remove(x)
                    sliceFramePointsData["y"].remove(y)
                    sliceFramePointsData["id"].remove(point.data())
                    removed_ids.append(point.data())

        if removed_ids:
            self.markPointsLayerDirty(action=action)

        return removed_ids

    def removePointsLayer(self, button, toolbar=None):
        button.setChecked(False)
        button.action.scatterItem.setData([], [])
        button.action.loadedDfInfo = None
        self.ax1.removeItem(button.action.scatterItem)
        toolbar.removeAction(button.action)
        for action in button.actions:
            toolbar.removeAction(action)

        if toolbar == self.promptSegmentPointsLayerToolbar:
            self.promptSegmentPointsLayerToolbar.isPointsLayerInit = False

    def resizeRangeWelcomeText(self):
        xRange, yRange = self.ax1.viewRange()
        deltaX = xRange[1] - xRange[0]
        deltaY = yRange[1] - yRange[0]
        self.ax1.setXRange(0, deltaX)
        self.ax1.setYRange(0, deltaY)
        self.ax1.setLimits(xMin=0, xMax=deltaX, yMin=0, yMax=deltaY)

    def restartZoomAutoPilot(self):
        if not self.autoPilotZoomToObjToggle.isChecked():
            return

        posData = self.data[self.pos_i]
        if not posData.IDs:
            return

        self.autoPilotZoomToObjSpinBox.setValue(posData.IDs[0])
        self.zoomToObj(posData.rp[0])

    def restorePrevPointIdRightClick(self, addPointsByClickingButton):
        # Try to restore the id that was there before hovering
        # because the hovering was required only to delete the
        # point
        try:
            prevId = addPointsByClickingButton.rightClickIDSpinbox.prevId
            addPointsByClickingButton.rightClickIDSpinbox.setValue(prevId)
        except Exception:
            addPointsByClickingButton.rightClickIDSpinbox.prevId = None

    @exception_handler
    def savePointsAddedByClicking(self, button, event):
        sender = button.action
        toolButton = sender.toolButton
        tableEndName = toolButton.clickEntryTableEndName

        self.logger.info(f"Saving _{tableEndName}.csv table...")

        self.savePointsAddedByClickingFromEndname(tableEndName)

        self.logger.info(f"{tableEndName}.csv saved!")
        self.titleLabel.setText(f"{tableEndName}.csv saved!", color="g")

    def savePointsAddedByClickingFromEndname(self, tableEndName, recovery=False):
        self.pointsLayerDataToDf(self.data[self.pos_i])
        for posData in self.data:
            if not posData.basename.endswith("_"):
                basename = f"{posData.basename}_"
            else:
                basename = posData.basename
            tableFilename = f"{basename}{tableEndName}.csv"
            if recovery:
                tableFilepath = os.path.join(
                    posData.recoveryFolderpath(), tableFilename
                )
            else:
                tableFilepath = os.path.join(posData.images_path, tableFilename)
            df = posData.clickEntryPointsDfs.get(tableEndName)
            if df is None:
                continue
            df = df.sort_values(["frame_i", "Cell_ID"])
            df.to_csv(tableFilepath, index=False)

    def setHoverCircleAddPoint(self, x, y):
        addPointsByClickingButton = self.buttonAddPointsByClickingActive()
        if addPointsByClickingButton is None:
            return
        action = addPointsByClickingButton.action
        self.setHoverToolSymbolData(
            [x], [y], (self.ax1_BrushCircle,), size=action.pointSize
        )

    def setPointsLayerLoadedDfEndanme(self, action):
        if action.loadedDfInfo is None:
            return

        posData = self.data[self.pos_i]
        images_path = posData.images_path.replace("\\", "/")

        df_folderpath = os.path.dirname(
            action.loadedDfInfo["filepath"].replace("\\", "/")
        )

        if images_path != df_folderpath:
            return

        df_filename = os.path.basename(action.loadedDfInfo["filepath"])

        if not df_filename.startswith(posData.basename):
            return

        endname = df_filename[len(posData.basename) :]
        action.loadedDfInfo["endname"] = endname

        action.button.setToolTip(endname)

    def setupAddPointsByClicking(self, toolButton, isLoadedDf, toolbar):
        self.LeftClickButtons.append(toolButton)
        posData = self.data[self.pos_i]
        tableEndName = self.addPointsWin.clickEntryTableEndnameText
        if isLoadedDf is not None:
            posData = self.data[self.pos_i]
            tableEndName = tableEndName[len(posData.basename) :]
            self.loadClickEntryDfs(tableEndName)

        toolButton.toolbar = toolbar
        toolButton.clickEntryTableEndName = tableEndName
        self.checkableQButtonsGroup.addButton(toolButton)
        toolButton.toggled.connect(self.addPointsByClickingButtonToggled)

        self.addPointsByClickingButtonToggled(sender=toolButton)

        toolButton.setToolTip(tableEndName)

        pointIdSpinbox = widgets.SpinBox()
        pointIdSpinbox.setMinimum(0)
        pointIdSpinbox.setValue(1)
        pointIdSpinbox.label = QLabel(" Left-click ID: ")
        pointIdSpinbox.labelAction = toolbar.addWidget(pointIdSpinbox.label)
        if toolbar == self.promptSegmentPointsLayerToolbar:
            newID = self.setBrushID(return_val=True)
            pointIdSpinbox.setValue(newID)
            pointIdSpinbox.setReadOnly(True)
            pointIdSpinbox.setToolTip(
                "The ids added with left-click cannot be manually edited. "
                "They are always a new, non-existing id."
            )

        toolButton.actions.append(pointIdSpinbox.labelAction)
        pointIdSpinbox.action = toolbar.addWidget(pointIdSpinbox)
        toolButton.actions.append(pointIdSpinbox.action)
        pointIdSpinbox.toolButton = toolButton
        toolButton.pointIdSpinbox = pointIdSpinbox

        rightClickIDSpinbox = widgets.SpinBox()
        pointIdSpinbox.setLinkedValueWidget(rightClickIDSpinbox)
        rightClickIDSpinbox.setMaximumWidth(pointIdSpinbox.sizeHint().width())
        rightClickIDSpinbox.setValue(pointIdSpinbox.value())
        rightClickIDSpinbox.setMinimum(0)
        rightClickIDSpinbox.label = QLabel(" | Right-click ID: ")
        rightClickIDSpinbox.labelAction = toolbar.addWidget(rightClickIDSpinbox.label)
        toolButton.actions.append(rightClickIDSpinbox.labelAction)
        rightClickIDSpinbox.action = toolbar.addWidget(rightClickIDSpinbox)
        toolButton.actions.append(rightClickIDSpinbox.action)
        rightClickIDSpinbox.toolButton = toolButton
        toolButton.rightClickIDSpinbox = rightClickIDSpinbox

        saveToolbutton = widgets.SavePointsLayerButton(tableEndName, parent=self)
        saveToolbutton.sigRenameTableAction.connect(
            self.updatePointsLayerClickEntryTableEndname
        )
        saveToolbutton.sigLeftClick.connect(self.savePointsAddedByClicking)
        saveAction = toolbar.addWidget(saveToolbutton)
        saveToolbutton.action = saveAction
        saveAction.saveToolbutton = saveToolbutton
        saveAction.toolButton = toolButton
        toolButton.saveAction = saveAction
        toolButton.saveToolbutton = saveToolbutton

        toolButton.actions.append(saveAction)

        vlineAction = toolbar.addWidget(widgets.QVLine())
        spacerAction = toolbar.addWidget(widgets.QHWidgetSpacer(width=5))

        toolButton.actions.append(vlineAction)
        toolButton.actions.append(spacerAction)

        action = toolButton.action
        scatterItem = action.scatterItem
        scatterItem.sigHoverEntered.connect(
            self.addPointsByClickingScatterItemHoverEntered
        )

        self.pointsLayerClicksDfsToData(posData, toolbar=toolbar)

    def should_compute_points_layer(
        self,
        *,
        layer_type_index: int,
        compute_points_layers: bool,
    ) -> bool:
        return layer_type_index < 2 and compute_points_layers

    def should_load_recovery_table(
        self,
        *,
        recovery_exists: bool,
        main_exists: bool,
        recovery_mtime: float | None,
        main_mtime: float | None,
    ) -> bool:
        if not recovery_exists:
            return False
        if not main_exists:
            return True
        if recovery_mtime is None or main_mtime is None:
            return False
        return recovery_mtime > main_mtime + self.recovery_tolerance_seconds

    def should_log_missing_frame_points(self, layer_type_index: int) -> bool:
        return layer_type_index != 4

    def should_use_z_slice(
        self,
        *,
        z_projection_mode: str,
        size_z: int,
        frame_points_data: Mapping,
    ) -> bool:
        return (
            z_projection_mode == "single z-slice"
            and size_z > 1
            and "x" not in frame_points_data
        )

    def showPointsLayerIdsToggled(self, button, checked):
        button.action.scatterItem.drawIds = checked
        self.drawPointsLayers()

    def storeUndoAddPoint(self, action):
        if not hasattr(self, "undoAddPointQueueMapper"):
            self.undoAddPointQueueMapper = defaultdict(list)

        self.data[self.pos_i]
        pointsDataPos = action.pointsData.get(self.pos_i)
        if pointsDataPos is None:
            return

        state = deepcopy(pointsDataPos)
        self.undoAddPointQueueMapper[action].append(state)
        self.undoAction.setEnabled(True)

    def undoAddPoint(self, action):
        undoAddPointQueue = self.undoAddPointQueueMapper.get(action)
        if undoAddPointQueue is None:
            return False

        if len(undoAddPointQueue) == 0:
            return False

        self.data[self.pos_i]
        state = undoAddPointQueue.pop(-1)
        action.pointsData[self.pos_i] = state
        self.markPointsLayerDirty(action=action)

        self.drawPointsLayers(computePointsLayers=False)

        if len(self.undoAddPointQueueMapper[action]) == 0:
            self.undoAction.setEnabled(True)

        return True

    def updatePointsLayerClickEntryTableEndname(self, saveToolbutton, table_endname):
        saveAction = saveToolbutton.action
        toolButton = saveAction.toolButton
        toolButton.clickEntryTableEndName = table_endname

        self.logger.info(
            f'Done. Click entry table endname updated to "{table_endname}"'
        )

    def zoomToObj(self, obj=None):
        if not hasattr(self, "data"):
            return
        posData = self.data[self.pos_i]
        if obj is None:
            ID = self.sender().value()
            try:
                ID_idx = posData.IDs_idxs[ID]
                obj = obj = posData.rp[ID_idx]
            except Exception:
                self.logger.warning(f"ID {ID} does not exist (add points by clicking)")

        if obj is None:
            return

        self.goToZsliceSearchedID(obj)
        min_row, min_col, max_row, max_col = self.getObjBbox(obj.bbox)
        xRange = min_col - 5, max_col + 5
        yRange = max_row + 5, min_row - 5

        self.ax1.setRange(xRange=xRange, yRange=yRange)
