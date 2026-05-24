"""Qt view adapter for custom annotations."""

from __future__ import annotations

import json
import os
import re
import traceback
from collections import defaultdict

import pyqtgraph as pg
import os
import pandas as pd
from qtpy.QtGui import QColor

from cellacdc import apps, html_utils, settings_folderpath, widgets


custom_annot_path = os.path.join(settings_folderpath, 'custom_annotations.json')


class CustomAnnotationsView:
    """Qt-facing adapter around custom annotation buttons and dialogs."""

    def __init__(self, host):
        object.__setattr__(self, 'host', host)
    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {'host'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def readSavedCustomAnnot(self):
        if os.path.exists(custom_annot_path):
            self.logger.info('Loading saved custom annotations...')
        tempAnnot = self.read_saved_annotations(
            custom_annot_path, logger_func=self.logger.info
        )

        posData = self.data[self.pos_i]
        self.savedCustomAnnot = tempAnnot
        for pos_i, posData in enumerate(self.data):
            self.savedCustomAnnot = {
                **self.savedCustomAnnot, **posData.customAnnot
            }
    
    def addCustomAnnotButtonAllLoadedPos(self):
        allPosCustomAnnot = {}
        for pos_i, posData in enumerate(self.data):
            self.addCustomAnnotationSavedPos(pos_i=pos_i)
            allPosCustomAnnot = {**allPosCustomAnnot, **posData.customAnnot}
        for posData in self.data:
            posData.customAnnot = allPosCustomAnnot

    def addCustomAnnotationSavedPos(self, pos_i=None):
        if pos_i is None:
            pos_i = self.pos_i
        
        posData = self.data[pos_i]
        for name, annotState in posData.customAnnot.items():
            # Check if button is already present and update only annotated IDs
            buttons = [b for b in self.customAnnotDict.keys() if b.name==name]
            if buttons:
                toolButton = buttons[0]
                allAnnotedIDs = self.customAnnotDict[toolButton]['annotatedIDs']
                allAnnotedIDs[pos_i] = posData.customAnnotIDs.get(name, {})
                continue

            try:
                symbol = re.findall(r"\'(.+)\'", annotState['symbol'])[0]
            except Exception as e:
                self.logger.info(traceback.format_exc())
                symbol = 'o'
            
            symbolColor = QColor(*annotState['symbolColor'])
            shortcut = annotState['shortcut']
            if shortcut is not None:
                keySequence = widgets.macShortcutToWindows(shortcut)
                keySequence = widgets.KeySequenceFromText(keySequence)
            else:
                keySequence = None
            toolTip = self.tooltip(annotState)
            keepActive = annotState.get('keepActive', True)
            isHideChecked = annotState.get('isHideChecked', True)

            toolButton, action = self.addCustomAnnotationButton(
                symbol, symbolColor, keySequence, toolTip, name,
                keepActive, isHideChecked
            )
            allPosAnnotIDs = [
                pos.customAnnotIDs.get(name, defaultdict(list)) 
                for pos in self.data
            ]
            self.customAnnotDict[toolButton] = {
                'action': action,
                'state': annotState,
                'annotatedIDs': allPosAnnotIDs
            }

            self.addCustomAnnnotScatterPlot(symbolColor, symbol, toolButton)

    def addCustomAnnotationButton(
            self, symbol, symbolColor, keySequence, toolTip, annotName,
            keepActive, isHideChecked
        ):
        toolButton = widgets.customAnnotToolButton(
            symbol, symbolColor, parent=self.host, keepToolActive=keepActive,
            isHideChecked=isHideChecked
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

    def addCustomAnnnotScatterPlot(
            self, symbolColor, symbol, toolButton
        ):
        # Add scatter plot item
        symbolColorBrush = [0, 0, 0, 50]
        symbolColorBrush[:3] = symbolColor.getRgb()[:3]
        scatterPlotItem = widgets.CustomAnnotationScatterPlotItem()
        scatterPlotItem.setData(
            [], [], symbol=symbol, pxMode=False,
            brush=pg.mkBrush(symbolColorBrush), size=15,
            pen=pg.mkPen(width=3, color=symbolColor),
            hoverable=True, hoverBrush=pg.mkBrush(symbolColor),
            tip=None
        )
        scatterPlotItem.sigHovered.connect(self.customAnnotHovered)
        scatterPlotItem.button = toolButton
        self.customAnnotDict[toolButton]['scatterPlotItem'] = scatterPlotItem
        self.ax1.addItem(scatterPlotItem)
    
    def addCustomAnnotationItems(
            self, symbol, symbolColor, keySequence, toolTip, name,
            keepActive, isHideChecked, state
        ):
        toolButton, action = self.addCustomAnnotationButton(
            symbol, symbolColor, keySequence, toolTip, name,
            keepActive, isHideChecked
        )

        self.customAnnotDict[toolButton] = {
            'action': action,
            'state': state,
            'annotatedIDs': [defaultdict(list) for _ in range(len(self.data))]
        }

        # Save custom annotation to cellacdc/temp/custom_annotations.json
        state_to_save = state.copy()
        state_to_save['symbolColor'] = tuple(symbolColor.getRgb())
        self.savedCustomAnnot[name] = state_to_save
        for posData in self.data:
            posData.customAnnot[name] = state_to_save

        # Add scatter plot item
        self.addCustomAnnnotScatterPlot(symbolColor, symbol, toolButton)

        customAnnotButton = self.customAnnotDict[toolButton]
        allPosAnnotatedIDs = customAnnotButton['annotatedIDs']
        # Add 0s column to acdc_df
        for pos_i, posData in enumerate(self.data):
            for frame_i, data_dict in enumerate(posData.allData_li):
                acdc_df = data_dict['acdc_df']
                if acdc_df is None:
                    continue
                result = self.ensure_column(
                    acdc_df, name
                )
                data_dict['acdc_df'] = result.dataframe
                allPosAnnotatedIDs[pos_i][frame_i].extend(
                    result.annotated_ids
                )
                    
            if posData.acdc_df is not None:
                result = self.ensure_column(
                    posData.acdc_df,
                    name,
                )
                posData.acdc_df = result.dataframe
                allPosAnnotatedIDs[pos_i][frame_i].extend(
                    result.annotated_ids
                )

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
            vb.setToolTip(
                f'Annotation name: {scatterPlotItem.button.name}\n'
                f'ID = {ID}'
            )
        else:
            vb.setToolTip('')
    
    def loadCustomAnnotations(self):
        items = list(self.savedCustomAnnot.keys())
        if len(items) == 0:
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph("""

    """Headless custom annotation table updates."""

    def read_saved_annotations(
        self,
        annotations_path: str,
        *,
        logger_func=None,
    ) -> dict:
        if not os.path.exists(annotations_path):
            return {}
        return load.read_json(annotations_path, logger_func=logger_func)

    def tooltip(self, annotation_state: dict) -> str:
        return myutils.getCustomAnnotTooltip(annotation_state)

    def ensure_column(
        self,
        acdc_df: pd.DataFrame,
        annotation_name: str,
    ) -> CustomAnnotationColumnResult:
        return ensure_custom_annotation_column(acdc_df, annotation_name)

    def column_exists(
        self,
        frame_records,
        annotation_name: str,
        *,
        summary_acdc_df: pd.DataFrame | None = None,
    ) -> bool:
        return custom_annotation_column_exists(
            frame_records,
            annotation_name,
            summary_acdc_df=summary_acdc_df,
        )

    def drop_column(
        self,
        acdc_df: pd.DataFrame | None,
        annotation_name: str,
    ) -> pd.DataFrame | None:
        return drop_custom_annotation_column(acdc_df, annotation_name)

    def rename_column(
        self,
        acdc_df: pd.DataFrame | None,
        old_name: str,
        new_name: str,
    ) -> pd.DataFrame | None:
        return rename_custom_annotation_column(acdc_df, old_name, new_name)

    def remap_ids(self, annotated_ids_by_frame, old_ids, new_ids) -> dict:
        return remap_custom_annotation_ids(
            annotated_ids_by_frame,
            old_ids,
            new_ids,
        )

    def update_frame(
        self,
        acdc_df: pd.DataFrame,
        annotation_name: str,
        annotated_ids,
        *,
        clicked_id: int = 0,
        click_is_active: bool = False,
        existing_ids=None,
    ) -> CustomAnnotationFrameUpdate:
        return update_custom_annotation_frame(
            acdc_df,
            annotation_name,
            annotated_ids,
            clicked_id=clicked_id,
            click_is_active=click_is_active,
            existing_ids=existing_ids,
        )

            There are no custom annotations saved.<br><br>
            Click on "Add custom annotation" button to start adding new 
            annotations.
            """)
            msg.warning(self.host, 'No annotations saved', txt)
            return
        
        self.selectAnnotWin = widgets.QDialogListbox(
            'Load previously used custom annotation(s)',
            'Select annotations to load:', items,
            additionalButtons=('Delete selected annnotations', ),
            parent=self.host, multiSelection=True
        )
        for button in self.selectAnnotWin._additionalButtons:
            button.disconnect()
            button.clicked.connect(self.deleteSavedAnnotation)
        self.selectAnnotWin.exec_()
        if self.selectAnnotWin.cancel:
            return
        
        for selectedAnnotName in self.selectAnnotWin.selectedItemsText:
            selectedAnnot = self.savedCustomAnnot[selectedAnnotName]

            symbol = selectedAnnot['symbol']
            symbol = re.findall(r"\'(.+)\'", symbol)[0]
            symbolColor = selectedAnnot['symbolColor']
            symbolColor = pg.mkColor(symbolColor)
            keySequence = widgets.KeySequenceFromText(selectedAnnot['shortcut'])
            Type = selectedAnnot['type']
            toolTip = (
                f'Name: {selectedAnnotName}\n\n'
                f'Type: {Type}\n\n'
                f'Usage: activate the button and RIGHT-CLICK on cell to annotate\n\n'
                f'Description: {selectedAnnot["description"]}\n\n'
                f'Shortcut: "{keySequence}"'
            )
            keepActive = selectedAnnot['keepActive']
            isHideChecked = selectedAnnot['isHideChecked']
            state = {
                'type': Type,
                'name': selectedAnnotName,
                'symbol':  selectedAnnot['symbol'],
                'shortcut': selectedAnnot['shortcut'],
                'description': selectedAnnot["description"],
                'keepActive': keepActive,
                'isHideChecked': isHideChecked,
                'symbolColor': symbolColor
            }
            self.addCustomAnnotationItems(
                symbol, symbolColor, keySequence, toolTip, selectedAnnotName,
                keepActive, isHideChecked, state
            )
            for pos_i, posData in enumerate(self.data):
                posData.customAnnot[selectedAnnotName] = selectedAnnot
            
        self.saveCustomAnnot()
    
    def deleteSavedAnnotation(self):
        for item in self.selectAnnotWin.listBox.selectedItems():
            name = item.text()
            self.savedCustomAnnot.pop(name)
        self.deleteSelectedAnnot(self.selectAnnotWin.listBox.selectedItems())
        items = list(self.savedCustomAnnot.keys())
        self.selectAnnotWin.listBox.clear()
        self.selectAnnotWin.listBox.addItems(items)

    def addCustomAnnotation(self):
        self.readSavedCustomAnnot()

        self.addAnnotWin = apps.customAnnotationDialog(
            self.savedCustomAnnot, parent=self.host
        )
        self.addAnnotWin.sigDeleteSelecAnnot.connect(self.deleteSelectedAnnot)
        self.addAnnotWin.exec_()
        if self.addAnnotWin.cancel:
            self.logger.info('Custom annotation process cancelled.')
            return

        symbol = self.addAnnotWin.symbol
        symbolColor = self.addAnnotWin.state['symbolColor']
        keySequence = self.addAnnotWin.shortcutWidget.widget.keySequence
        toolTip = self.addAnnotWin.toolTip
        name = self.addAnnotWin.state['name']
        keepActive = self.addAnnotWin.state.get('keepActive', True)
        isHideChecked = self.addAnnotWin.state.get('isHideChecked', True)
        
        proceed = self.checkNameExists(name)
        if not proceed:
            self.logger.info('Custom annotation process cancelled.')
            return

        self.addCustomAnnotationItems(
            symbol, symbolColor, keySequence, toolTip, name,
            keepActive, isHideChecked, self.addAnnotWin.state
        )
        self.saveCustomAnnot()
        self.doCustomAnnotation(0)

    def askCustomAnnotationNameExists(self, name):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(f"""
            The annotationa called <code>{name}</code> already exists in the 
            acdc_output CSV file.<br><br>
            If you continue, this column will be used to initialize 
            pre-annotated objects.<br><br>
            Do you want to continue?
        """
        )
        noButton, yesButton = msg.question(
            self.host, 'Custom annotation name already exists', txt,
            buttonsTexts=('No, stop process', 'Yes, use existing column')
        )
        return msg.clickedButton == yesButton
        
    
    def checkNameExists(self, name):
        posData = self.data[self.pos_i]
        if self.column_exists(
                posData.allData_li,
                name,
                summary_acdc_df=posData.acdc_df,
        ):
            return self.askCustomAnnotationNameExists(name)
         
        return True
            
    
    def viewAllCustomAnnot(self, checked):
        if not checked:
            # Clear all annotations before showing only checked
            for button in self.customAnnotDict.keys():
                self.clearScatterPlotCustomAnnotButton(button)
        self.doCustomAnnotation(0)

    def clearScatterPlotCustomAnnotButton(self, button):
        scatterPlotItem = self.customAnnotDict[button]['scatterPlotItem']
        scatterPlotItem.setData([], [])

    def saveCustomAnnot(self, only_temp=False):
        if not hasattr(self, 'savedCustomAnnot'):
            return

        if not self.savedCustomAnnot:
            return

        # Save to cell acdc temp path
        with open(custom_annot_path, mode='w') as file:
            json.dump(self.savedCustomAnnot, file, indent=2)

        if only_temp:
            return
        
        self.logger.info('Saving custom annotations parameters...')
        # Save to pos path
        for _posData in self.data:
            _posData.saveCustomAnnotationParams()

    def customAnnotKeepActive(self, button):
        self.customAnnotDict[button]['state']['keepActive'] = button.keepToolActive

    def customAnnotHide(self, button):
        self.customAnnotDict[button]['state']['isHideChecked'] = button.isHideChecked
        clearAnnot = (
            not button.isChecked() and button.isHideChecked
            and not self.viewAllCustomAnnotAction.isChecked()
        )
        if clearAnnot:
            # User checked hide annot with the button not active --> clear
            self.clearScatterPlotCustomAnnotButton(button)
        elif not button.isChecked():
            # User uncheked hide annot with the button not active --> show
            self.doCustomAnnotation(0)

    def deleteSelectedAnnot(self, itemsToDelete):
        self.saveCustomAnnot(only_temp=True)

    def customAnnotModify(self, button):
        state = self.customAnnotDict[button]['state']
        self.addAnnotWin = apps.customAnnotationDialog(
            self.savedCustomAnnot, parent=self.host, state=state
        )
        self.addAnnotWin.sigDeleteSelecAnnot.connect(self.deleteSelectedAnnot)
        self.addAnnotWin.exec_()
        if self.addAnnotWin.cancel:
            return

        # Rename column if existing
        posData = self.data[self.pos_i]
        acdc_df = posData.allData_li[posData.frame_i]['acdc_df']
        if acdc_df is not None:
            old_name = self.customAnnotDict[button]['state']['name']
            new_name = self.addAnnotWin.state['name']
            posData.allData_li[posData.frame_i]['acdc_df'] = (
                self.rename_column(
                    acdc_df, old_name, new_name
                )
            )

        self.customAnnotDict[button]['state'] = self.addAnnotWin.state

        name = self.addAnnotWin.state['name']
        state_to_save = self.addAnnotWin.state.copy()
        symbolColor = self.addAnnotWin.state['symbolColor']
        state_to_save['symbolColor'] = tuple(symbolColor.getRgb())
        self.savedCustomAnnot[name] = self.addAnnotWin.state
        self.saveCustomAnnot()

        symbol = self.addAnnotWin.symbol
        symbolColor = self.customAnnotDict[button]['state']['symbolColor']
        button.setColor(symbolColor)
        button.update()
        symbolColorBrush = [0, 0, 0, 50]
        symbolColorBrush[:3] = symbolColor.getRgb()[:3]
        scatterPlotItem = self.customAnnotDict[button]['scatterPlotItem']
        xx, yy = scatterPlotItem.getData()
        if xx is None:
            xx, yy = [], []
        scatterPlotItem.setData(
            xx, yy, symbol=symbol, pxMode=False,
            brush=pg.mkBrush(symbolColorBrush), size=15,
            pen=pg.mkPen(width=3, color=symbolColor)
        )

    def doCustomAnnotation(self, ID):
        mode = self.modeComboBox.currentText()
        if not self.isSnapshot and mode != 'Custom annotations':
            # Do not show annotations if timelapse and mode not annotations
            return
        
        if self.switchPlaneCombobox.depthAxes() != 'z': 
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
                b for b in self.customAnnotDict.keys()
                if (b.isChecked() or not b.isHideChecked)
            ]
            if not buttons:
                return

        for button in buttons:
            annotatedIDs = (
                self.customAnnotDict[button]['annotatedIDs'][self.pos_i]
            )
            annotIDs_frame_i = annotatedIDs.get(posData.frame_i, [])
            state = self.customAnnotDict[button]['state']
            acdc_df = posData.allData_li[posData.frame_i]['acdc_df']
            if acdc_df is None:
                self.store_data(autosave=False)
            acdc_df = posData.allData_li[posData.frame_i]['acdc_df']

            result = self.update_frame(
                acdc_df,
                state['name'],
                annotIDs_frame_i,
                clicked_id=ID,
                click_is_active=button.isChecked(),
                existing_ids=posData.IDs_idxs,
            )

            annotPerButton = self.customAnnotDict[button]
            allAnnotedIDs = annotPerButton['annotatedIDs']
            posAnnotedIDs = allAnnotedIDs[self.pos_i]
            posAnnotedIDs[posData.frame_i] = result.annotated_ids
            acdc_df = result.dataframe
            
            xx, yy = [], []
            for annotID in result.present_annotated_ids:
                obj_idx = posData.IDs_idxs[annotID]
                obj = posData.rp[obj_idx]
                if not self.isObjVisible(obj.bbox):
                    continue
                y, x = self.getObjCentroid(obj.centroid)
                xx.append(x)
                yy.append(y)
                
            scatterPlotItem = self.customAnnotDict[button]['scatterPlotItem']
            scatterPlotItem.setData(xx, yy)

            posData.allData_li[posData.frame_i]['acdc_df'] = acdc_df
        
        # if self.highlightedID != 0:
        #     self.highlightedID = 0
        #     self.setHighlightID(False)

        if buttons:
            return buttons[0]

    def removeCustomAnnotButton(self, button, askHow=True, save=True):
        if askHow:
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph("""
                Do you want to <b>remove also the column with annotations</b> or 
                only the annotation button?<br>
            """)
            _, removeOnlyButton, removeColButton = msg.question(
                self.host, 'Remove only button?', txt, 
                buttonsTexts=(
                    'Cancel', 'Remove only button', 
                    ' Remove also column with annotations '
                )
            )
            if msg.cancel:
                return
            removeOnlyButton = msg.clickedButton == removeOnlyButton
        else:
            removeOnlyButton = True
        
        name = self.customAnnotDict[button]['state']['name']
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

            posData.acdc_df = self.drop_column(
                posData.acdc_df,
                name,
            )
            for frame_i, data_dict in enumerate(posData.allData_li):
                acdc_df = data_dict['acdc_df']
                if acdc_df is None:
                    continue
                posData.allData_li[frame_i]['acdc_df'] = (
                    self.drop_column(
                        acdc_df, name
                    )
                )

        self.clearScatterPlotCustomAnnotButton(button)

        action = self.customAnnotDict[button]['action']
        self.annotateToolbar.removeAction(action)
        self.checkableQButtonsGroup.removeButton(button)
        self.customAnnotDict.pop(button)
        # self.savedCustomAnnot.pop(name)

        self.saveCustomAnnot(only_temp=True)

    def reinitCustomAnnot(self):
        buttons = list(self.customAnnotDict.keys())
        for button in buttons:
            self.clearScatterPlotCustomAnnotButton(button)
            action = self.customAnnotDict[button]['action']
            self.annotateToolbar.removeAction(action)
            self.checkableQButtonsGroup.removeButton(button)
            self.customAnnotDict.pop(button)
            # self.savedCustomAnnot.pop(name)

            self.saveCustomAnnot(only_temp=True)

    def clearCustomAnnot(self):
        for button in self.customAnnotDict.keys():
            scatterPlotItem = self.customAnnotDict[button]['scatterPlotItem']
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
                button.isHideChecked 
                or not self.viewAllCustomAnnotAction.isChecked()
            )
            if clearAnnotation:    
                self.clearScatterPlotCustomAnnotButton(button)
            self.setHighlightID(False)
            self.resetCursor()