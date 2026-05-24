"""Qt view adapter for custom annotations."""

from __future__ import annotations

import json
import os
import re
import traceback
from collections import defaultdict

import pyqtgraph as pg
import pandas as pd
from qtpy.QtGui import QColor

from cellacdc import apps, html_utils, settings_folderpath, widgets


custom_annot_path = os.path.join(settings_folderpath, "custom_annotations.json")

from .annotation_display import AnnotationDisplay
from .object_properties import ObjectProperties


class CustomAnnotations(AnnotationDisplay, ObjectProperties):
    """Extracted from guiWin."""

    def addCustomAnnnotScatterPlot(self, symbolColor, symbol, toolButton):
        # Add scatter plot item
        symbolColorBrush = [0, 0, 0, 50]
        symbolColorBrush[:3] = symbolColor.getRgb()[:3]
        scatterPlotItem = widgets.CustomAnnotationScatterPlotItem()
        scatterPlotItem.setData(
            [],
            [],
            symbol=symbol,
            pxMode=False,
            brush=pg.mkBrush(symbolColorBrush),
            size=15,
            pen=pg.mkPen(width=3, color=symbolColor),
            hoverable=True,
            hoverBrush=pg.mkBrush(symbolColor),
            tip=None,
        )
        scatterPlotItem.sigHovered.connect(self.customAnnotHovered)
        scatterPlotItem.button = toolButton
        self.customAnnotDict[toolButton]["scatterPlotItem"] = scatterPlotItem
        self.ax1.addItem(scatterPlotItem)

    def addCustomAnnotButtonAllLoadedPos(self):
        allPosCustomAnnot = {}
        for pos_i, posData in enumerate(self.data):
            self.addCustomAnnotationSavedPos(pos_i=pos_i)
            allPosCustomAnnot = {**allPosCustomAnnot, **posData.customAnnot}
        for posData in self.data:
            posData.customAnnot = allPosCustomAnnot

    def addCustomAnnotation(self):
        self.readSavedCustomAnnot()

        self.addAnnotWin = apps.customAnnotationDialog(
            self.savedCustomAnnot, parent=self
        )
        self.addAnnotWin.sigDeleteSelecAnnot.connect(self.deleteSelectedAnnot)
        self.addAnnotWin.exec_()
        if self.addAnnotWin.cancel:
            self.logger.info("Custom annotation process cancelled.")
            return

        symbol = self.addAnnotWin.symbol
        symbolColor = self.addAnnotWin.state["symbolColor"]
        keySequence = self.addAnnotWin.shortcutWidget.widget.keySequence
        toolTip = self.addAnnotWin.toolTip
        name = self.addAnnotWin.state["name"]
        keepActive = self.addAnnotWin.state.get("keepActive", True)
        isHideChecked = self.addAnnotWin.state.get("isHideChecked", True)

        proceed = self.checkNameExists(name)
        if not proceed:
            self.logger.info("Custom annotation process cancelled.")
            return

        self.addCustomAnnotationItems(
            symbol,
            symbolColor,
            keySequence,
            toolTip,
            name,
            keepActive,
            isHideChecked,
            self.addAnnotWin.state,
        )
        self.saveCustomAnnot()
        self.doCustomAnnotation(0)

    def addCustomAnnotationButton(
        self,
        symbol,
        symbolColor,
        keySequence,
        toolTip,
        annotName,
        keepActive,
        isHideChecked,
    ):
        toolButton = widgets.customAnnotToolButton(
            symbol,
            symbolColor,
            parent=self,
            keepToolActive=keepActive,
            isHideChecked=isHideChecked,
        )
        toolButton.setCheckable(True)
        self.checkableQButtonsGroup.addButton(toolButton)
        if keySequence is not None:
            toolButton.setShortcut(keySequence)
        toolButton.setToolTip(toolTip)
        toolButton.name = annotName
        toolButton.toggled.connect(self.customAnnotButtonToggled)
        toolButton.sigRemoveAction.connect(self.removeCustomAnnotButton)
        toolButton.sigKeepActiveAction.connect(self.customAnnotKeepActive)
        toolButton.sigHideAction.connect(self.customAnnotHide)
        toolButton.sigModifyAction.connect(self.customAnnotModify)
        action = self.annotateToolbar.addWidget(toolButton)
        return toolButton, action

    def addCustomAnnotationItems(
        self,
        symbol,
        symbolColor,
        keySequence,
        toolTip,
        name,
        keepActive,
        isHideChecked,
        state,
    ):
        toolButton, action = self.addCustomAnnotationButton(
            symbol, symbolColor, keySequence, toolTip, name, keepActive, isHideChecked
        )

        self.customAnnotDict[toolButton] = {
            "action": action,
            "state": state,
            "annotatedIDs": [defaultdict(list) for _ in range(len(self.data))],
        }

        # Save custom annotation to cellacdc/temp/custom_annotations.json
        state_to_save = state.copy()
        state_to_save["symbolColor"] = tuple(symbolColor.getRgb())
        self.savedCustomAnnot[name] = state_to_save
        for posData in self.data:
            posData.customAnnot[name] = state_to_save

        # Add scatter plot item
        self.addCustomAnnnotScatterPlot(symbolColor, symbol, toolButton)

        customAnnotButton = self.customAnnotDict[toolButton]
        allPosAnnotatedIDs = customAnnotButton["annotatedIDs"]
        # Add 0s column to acdc_df
        for pos_i, posData in enumerate(self.data):
            for frame_i, data_dict in enumerate(posData.allData_li):
                acdc_df = data_dict["acdc_df"]
                if acdc_df is None:
                    continue
                if name not in acdc_df.columns:
                    acdc_df[name] = 0
                else:
                    acdc_df[name] = acdc_df[name].astype(int)
                    acdc_df_annot = acdc_df[acdc_df[name] == 1].reset_index()
                    annot_IDs = acdc_df_annot["Cell_ID"].to_list()
                    allPosAnnotatedIDs[pos_i][frame_i].extend(annot_IDs)

            if posData.acdc_df is not None:
                if name not in posData.acdc_df.columns:
                    posData.acdc_df[name] = 0
                else:
                    posData.acdc_df[name] = posData.acdc_df[name].astype(int)
                    acdc_df_annot = posData.acdc_df[
                        posData.acdc_df[name] == 1
                    ].reset_index()
                    annot_IDs = acdc_df_annot["Cell_ID"].to_list()
                    allPosAnnotatedIDs[pos_i][frame_i].extend(annot_IDs)

    def addCustomAnnotationSavedPos(self, pos_i=None):
        if pos_i is None:
            pos_i = self.pos_i

        posData = self.data[pos_i]
        for name, annotState in posData.customAnnot.items():
            # Check if button is already present and update only annotated IDs
            buttons = [b for b in self.customAnnotDict.keys() if b.name == name]
            if buttons:
                toolButton = buttons[0]
                allAnnotedIDs = self.customAnnotDict[toolButton]["annotatedIDs"]
                allAnnotedIDs[pos_i] = posData.customAnnotIDs.get(name, {})
                continue

            try:
                symbol = re.findall(r"\'(.+)\'", annotState["symbol"])[0]
            except Exception as e:
                self.logger.info(traceback.format_exc())
                symbol = "o"

            symbolColor = QColor(*annotState["symbolColor"])
            shortcut = annotState["shortcut"]
            if shortcut is not None:
                keySequence = widgets.macShortcutToWindows(shortcut)
                keySequence = widgets.KeySequenceFromText(keySequence)
            else:
                keySequence = None
            toolTip = myutils.getCustomAnnotTooltip(annotState)
            keepActive = annotState.get("keepActive", True)
            isHideChecked = annotState.get("isHideChecked", True)

            toolButton, action = self.addCustomAnnotationButton(
                symbol,
                symbolColor,
                keySequence,
                toolTip,
                name,
                keepActive,
                isHideChecked,
            )
            allPosAnnotIDs = [
                pos.customAnnotIDs.get(name, defaultdict(list)) for pos in self.data
            ]
            self.customAnnotDict[toolButton] = {
                "action": action,
                "state": annotState,
                "annotatedIDs": allPosAnnotIDs,
            }

            self.addCustomAnnnotScatterPlot(symbolColor, symbol, toolButton)

    def askCustomAnnotationNameExists(self, name):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(f"""
            The annotationa called <code>{name}</code> already exists in the 
            acdc_output CSV file.<br><br>
            If you continue, this column will be used to initialize 
            pre-annotated objects.<br><br>
            Do you want to continue?
        """)
        noButton, yesButton = msg.question(
            self,
            "Custom annotation name already exists",
            txt,
            buttonsTexts=("No, stop process", "Yes, use existing column"),
        )
        return msg.clickedButton == yesButton

    def checkNameExists(self, name):
        posData = self.data[self.pos_i]
        for frame_i, data_dict in enumerate(posData.allData_li):
            acdc_df = data_dict["acdc_df"]
            if acdc_df is None:
                continue
            if name in acdc_df.columns:
                return self.askCustomAnnotationNameExists(name)

        if posData.acdc_df is not None and name in posData.acdc_df.columns:
            return self.askCustomAnnotationNameExists(name)

        return True

    def clearCustomAnnot(self):
        for button in self.customAnnotDict.keys():
            scatterPlotItem = self.customAnnotDict[button]["scatterPlotItem"]
            scatterPlotItem.setData([], [])

    def clearScatterPlotCustomAnnotButton(self, button):
        scatterPlotItem = self.customAnnotDict[button]["scatterPlotItem"]
        scatterPlotItem.setData([], [])

    def customAnnotButtonToggled(self, checked):
        if checked:
            self.customAnnotButton = self.sender()
            # Uncheck the other buttons
            for button in self.customAnnotDict.keys():
                if button == self.sender():
                    continue

                button.toggled.disconnect()
                self.clearScatterPlotCustomAnnotButton(button)
                button.setChecked(False)
                button.toggled.connect(self.customAnnotButtonToggled)
            self.doCustomAnnotation(0)
        else:
            self.customAnnotButton = None
            button = self.sender()
            clearAnnotation = (
                button.isHideChecked or not self.viewAllCustomAnnotAction.isChecked()
            )
            if clearAnnotation:
                self.clearScatterPlotCustomAnnotButton(button)
            self.setHighlightID(False)
            self.resetCursor()

    def customAnnotHide(self, button):
        self.customAnnotDict[button]["state"]["isHideChecked"] = button.isHideChecked
        clearAnnot = (
            not button.isChecked()
            and button.isHideChecked
            and not self.viewAllCustomAnnotAction.isChecked()
        )
        if clearAnnot:
            # User checked hide annot with the button not active --> clear
            self.clearScatterPlotCustomAnnotButton(button)
        elif not button.isChecked():
            # User uncheked hide annot with the button not active --> show
            self.doCustomAnnotation(0)

    def customAnnotHovered(self, scatterPlotItem, points, event):
        # Show tool tip when hovering an annotation with annotation name and ID
        vb = scatterPlotItem.getViewBox()
        if vb is None:
            return
        if len(points) > 0:
            posData = self.data[self.pos_i]
            point = points[0]
            x, y = point.pos().x(), point.pos().y()
            xdata, ydata = int(x), int(y)
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]
            vb.setToolTip(f"Annotation name: {scatterPlotItem.button.name}\nID = {ID}")
        else:
            vb.setToolTip("")

    def customAnnotKeepActive(self, button):
        self.customAnnotDict[button]["state"]["keepActive"] = button.keepToolActive

    def customAnnotModify(self, button):
        state = self.customAnnotDict[button]["state"]
        self.addAnnotWin = apps.customAnnotationDialog(
            self.savedCustomAnnot, state=state
        )
        self.addAnnotWin.sigDeleteSelecAnnot.connect(self.deleteSelectedAnnot)
        self.addAnnotWin.exec_()
        if self.addAnnotWin.cancel:
            return

        # Rename column if existing
        posData = self.data[self.pos_i]
        acdc_df = posData.allData_li[posData.frame_i]["acdc_df"]
        if acdc_df is not None:
            old_name = self.customAnnotDict[button]["state"]["name"]
            new_name = self.addAnnotWin.state["name"]
            acdc_df = acdc_df.rename(columns={old_name: new_name})
            posData.allData_li[posData.frame_i]["acdc_df"] = acdc_df

        self.customAnnotDict[button]["state"] = self.addAnnotWin.state

        name = self.addAnnotWin.state["name"]
        state_to_save = self.addAnnotWin.state.copy()
        symbolColor = self.addAnnotWin.state["symbolColor"]
        state_to_save["symbolColor"] = tuple(symbolColor.getRgb())
        self.savedCustomAnnot[name] = self.addAnnotWin.state
        self.saveCustomAnnot()

        symbol = self.addAnnotWin.symbol
        symbolColor = self.customAnnotDict[button]["state"]["symbolColor"]
        button.setColor(symbolColor)
        button.update()
        symbolColorBrush = [0, 0, 0, 50]
        symbolColorBrush[:3] = symbolColor.getRgb()[:3]
        scatterPlotItem = self.customAnnotDict[button]["scatterPlotItem"]
        xx, yy = scatterPlotItem.getData()
        if xx is None:
            xx, yy = [], []
        scatterPlotItem.setData(
            xx,
            yy,
            symbol=symbol,
            pxMode=False,
            brush=pg.mkBrush(symbolColorBrush),
            size=15,
            pen=pg.mkPen(width=3, color=symbolColor),
        )

    def deleteSavedAnnotation(self):
        for item in self.selectAnnotWin.listBox.selectedItems():
            name = item.text()
            self.savedCustomAnnot.pop(name)
        self.deleteSelectedAnnot(self.selectAnnotWin.listBox.selectedItems())
        items = list(self.savedCustomAnnot.keys())
        self.selectAnnotWin.listBox.clear()
        self.selectAnnotWin.listBox.addItems(items)

    def deleteSelectedAnnot(self, itemsToDelete):
        self.saveCustomAnnot(only_temp=True)

    def doCustomAnnotation(self, ID):
        mode = self.modeComboBox.currentText()
        if not self.isSnapshot and mode != "Custom annotations":
            # Do not show annotations if timelapse and mode not annotations
            return

        if self.switchPlaneCombobox.depthAxes() != "z":
            return

        # NOTE: pass 0 for ID to not add
        posData = self.data[self.pos_i]
        if self.viewAllCustomAnnotAction.isChecked():
            # User requested to show all annotations --> iterate all buttons
            # Unless it actively clicked to annotate --> avoid annotating object
            # with all the annotations present
            buttons = list(self.customAnnotDict.keys())
        else:
            # Annotate if the button is active or isHideChecked is False
            buttons = [
                b
                for b in self.customAnnotDict.keys()
                if (b.isChecked() or not b.isHideChecked)
            ]
            if not buttons:
                return

        for button in buttons:
            annotatedIDs = self.customAnnotDict[button]["annotatedIDs"][self.pos_i]
            annotIDs_frame_i = annotatedIDs.get(posData.frame_i, [])
            state = self.customAnnotDict[button]["state"]
            acdc_df = posData.allData_li[posData.frame_i]["acdc_df"]

            if button.isChecked() and ID > 0:
                # Annotate only if existing ID and the button is checked
                if ID in annotIDs_frame_i:
                    annotIDs_frame_i.remove(ID)
                    acdc_df.at[ID, state["name"]] = 0
                elif ID != 0:
                    annotIDs_frame_i.append(ID)

            annotPerButton = self.customAnnotDict[button]
            allAnnotedIDs = annotPerButton["annotatedIDs"]
            posAnnotedIDs = allAnnotedIDs[self.pos_i]
            posAnnotedIDs[posData.frame_i] = annotIDs_frame_i

            if acdc_df is None:
                self.store_data(autosave=False)
            acdc_df = posData.allData_li[posData.frame_i]["acdc_df"]

            xx, yy = [], []
            for annotID in annotIDs_frame_i:
                if annotID not in posData.IDs_idxs:
                    continue

                obj_idx = posData.IDs_idxs[annotID]
                obj = posData.rp[obj_idx]
                acdc_df.at[annotID, state["name"]] = 1
                if not self.isObjVisible(obj.bbox):
                    continue
                y, x = self.getObjCentroid(obj.centroid)
                xx.append(x)
                yy.append(y)

            scatterPlotItem = self.customAnnotDict[button]["scatterPlotItem"]
            scatterPlotItem.setData(xx, yy)

            posData.allData_li[posData.frame_i]["acdc_df"] = acdc_df

        # if self.highlightedID != 0:
        #     self.highlightedID = 0
        #     self.setHighlightID(False)

        if buttons:
            return buttons[0]

    def loadCustomAnnotations(self):
        items = list(self.savedCustomAnnot.keys())
        if len(items) == 0:
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph("""
            There are no custom annotations saved.<br><br>
            Click on "Add custom annotation" button to start adding new 
            annotations.
            """)
            msg.warning(self, "No annotations saved", txt)
            return

        self.selectAnnotWin = widgets.QDialogListbox(
            "Load previously used custom annotation(s)",
            "Select annotations to load:",
            items,
            additionalButtons=("Delete selected annnotations",),
            parent=self,
            multiSelection=True,
        )
        for button in self.selectAnnotWin._additionalButtons:
            button.disconnect()
            button.clicked.connect(self.deleteSavedAnnotation)
        self.selectAnnotWin.exec_()
        if self.selectAnnotWin.cancel:
            return

        for selectedAnnotName in self.selectAnnotWin.selectedItemsText:
            selectedAnnot = self.savedCustomAnnot[selectedAnnotName]

            symbol = selectedAnnot["symbol"]
            symbol = re.findall(r"\'(.+)\'", symbol)[0]
            symbolColor = selectedAnnot["symbolColor"]
            symbolColor = pg.mkColor(symbolColor)
            keySequence = widgets.KeySequenceFromText(selectedAnnot["shortcut"])
            Type = selectedAnnot["type"]
            toolTip = (
                f"Name: {selectedAnnotName}\n\n"
                f"Type: {Type}\n\n"
                f"Usage: activate the button and RIGHT-CLICK on cell to annotate\n\n"
                f"Description: {selectedAnnot['description']}\n\n"
                f'Shortcut: "{keySequence}"'
            )
            keepActive = selectedAnnot["keepActive"]
            isHideChecked = selectedAnnot["isHideChecked"]
            state = {
                "type": Type,
                "name": selectedAnnotName,
                "symbol": selectedAnnot["symbol"],
                "shortcut": selectedAnnot["shortcut"],
                "description": selectedAnnot["description"],
                "keepActive": keepActive,
                "isHideChecked": isHideChecked,
                "symbolColor": symbolColor,
            }
            self.addCustomAnnotationItems(
                symbol,
                symbolColor,
                keySequence,
                toolTip,
                selectedAnnotName,
                keepActive,
                isHideChecked,
                state,
            )
            for pos_i, posData in enumerate(self.data):
                posData.customAnnot[selectedAnnotName] = selectedAnnot

        self.saveCustomAnnot()

    def readSavedCustomAnnot(self):
        tempAnnot = {}
        if os.path.exists(custom_annot_path):
            self.logger.info("Loading saved custom annotations...")
            tempAnnot = load.read_json(custom_annot_path, logger_func=self.logger.info)

        posData = self.data[self.pos_i]
        self.savedCustomAnnot = tempAnnot
        for pos_i, posData in enumerate(self.data):
            self.savedCustomAnnot = {**self.savedCustomAnnot, **posData.customAnnot}

    def reinitCustomAnnot(self):
        buttons = list(self.customAnnotDict.keys())
        for button in buttons:
            self.clearScatterPlotCustomAnnotButton(button)
            action = self.customAnnotDict[button]["action"]
            self.annotateToolbar.removeAction(action)
            self.checkableQButtonsGroup.removeButton(button)
            self.customAnnotDict.pop(button)
            # self.savedCustomAnnot.pop(name)

            self.saveCustomAnnot(only_temp=True)

    def removeCustomAnnotButton(self, button, askHow=True, save=True):
        if askHow:
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph("""
                Do you want to <b>remove also the column with annotations</b> or 
                only the annotation button?<br>
            """)
            _, removeOnlyButton, removeColButton = msg.question(
                self,
                "Remove only button?",
                txt,
                buttonsTexts=(
                    "Cancel",
                    "Remove only button",
                    " Remove also column with annotations ",
                ),
            )
            if msg.cancel:
                return
            removeOnlyButton = msg.clickedButton == removeOnlyButton
        else:
            removeOnlyButton = True

        name = self.customAnnotDict[button]["state"]["name"]
        # remove annotation from position
        for posData in self.data:
            try:
                posData.customAnnot.pop(name)
                posData.saveCustomAnnotationParams()
            except KeyError as e:
                # Current pos doesn't have any annotation button. Continue
                continue

            if posData.acdc_df is None:
                continue

            if removeOnlyButton:
                continue

            posData.acdc_df = posData.acdc_df.drop(columns=name, errors="ignore")
            for frame_i, data_dict in enumerate(posData.allData_li):
                acdc_df = data_dict["acdc_df"]
                if acdc_df is None:
                    continue
                acdc_df = acdc_df.drop(columns=name, errors="ignore")
                posData.allData_li[frame_i]["acdc_df"] = acdc_df

        self.clearScatterPlotCustomAnnotButton(button)

        action = self.customAnnotDict[button]["action"]
        self.annotateToolbar.removeAction(action)
        self.checkableQButtonsGroup.removeButton(button)
        self.customAnnotDict.pop(button)
        # self.savedCustomAnnot.pop(name)

        self.saveCustomAnnot(only_temp=True)

    def saveCustomAnnot(self, only_temp=False):
        if not hasattr(self, "savedCustomAnnot"):
            return

        if not self.savedCustomAnnot:
            return

        # Save to cell acdc temp path
        with open(custom_annot_path, mode="w") as file:
            json.dump(self.savedCustomAnnot, file, indent=2)

        if only_temp:
            return

        self.logger.info("Saving custom annotations parameters...")
        # Save to pos path
        for _posData in self.data:
            _posData.saveCustomAnnotationParams()

    def viewAllCustomAnnot(self, checked):
        if not checked:
            # Clear all annotations before showing only checked
            for button in self.customAnnotDict.keys():
                self.clearScatterPlotCustomAnnotButton(button)
        self.doCustomAnnotation(0)
