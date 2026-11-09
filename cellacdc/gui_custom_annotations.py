"""Mixin with the GUI elements and the logic of the custom annotations 
(user-defined per-object labels saved as a column of the acdc_output table).

It is shared by the main GUI (`gui.guiWin`) and the measurements viewer 
(`measurements_viewer.MeasurementsViewerWin`).

Each annotation has a checkable toolbar button (`widgets.customAnnotToolButton`) 
and a scatter plot item that draws a symbol on the annotated objects. 
`self.customAnnotDict` maps each button to 
`{'action', 'state', 'annotatedIDs', 'scatterPlotItem'}`.

Requirements on the inheriting class
------------------------------------
- `self.data`, `self.pos_i`, `self.logger`, `self.ax1`, `self.app`, 
  `self.isSnapshot`, `self.modeComboBox` and `self.switchPlaneCombobox`.
- `self.customAnnotDict = {}` and `self.customAnnotButton = None` set in 
  `__init__`. `self.checkableQButtonsGroup` is optional.
- The methods of `gui_annotations.AnnotationsGuiElements` (`get_2Dlab`, 
  `isObjInCurrSlice`, `getObjCentroid`), `setHighlightID` and `resetCursor`.

Hooks that the inheriting class can override
--------------------------------------------
- `customAnnotIsVisibleMode`: whether annotations are displayed.
- `customAnnotationDialogKwargs`: additional kwargs of 
  `apps.customAnnotationDialog` (e.g., predictor options).
- `customAnnotToolTip`: tool tip of the annotation button.
- `setCustomAnnotShortcut`: assign the keyboard shortcut.
- `initCustomAnnotColumn`, `renameCustomAnnotColumn`, 
  `removeCustomAnnotColumn`: storage of the annotations in the acdc_df.
- `saveCustomAnnotPositionParams`: persist the annotations of a position.
- `onCustomAnnotationChanged`, `onCustomAnnotationRemoved`: called after 
  an annotation is added/modified or removed.
"""
import os
import re
import json
import traceback
from collections import defaultdict

from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon, QColor, QKeySequence
from qtpy.QtWidgets import QAction

import pyqtgraph as pg

from . import widgets, apps, html_utils, myutils, load
from . import settings_folderpath

custom_annot_path = os.path.join(settings_folderpath, 'custom_annotations.json')


def custom_annot_state_to_save(state):
    """Copy of `state` that can be serialized to JSON (RGB(A) color)."""
    state_to_save = state.copy()
    symbolColor = state_to_save.get('symbolColor')
    if isinstance(symbolColor, QColor):
        state_to_save['symbolColor'] = tuple(symbolColor.getRgb())
    elif symbolColor is not None:
        state_to_save['symbolColor'] = tuple(symbolColor)
    return state_to_save


def custom_annot_symbol(state):
    """pyqtgraph symbol (e.g., 'o') from the combobox text of `state`."""
    try:
        return re.findall(r"\'(.+)\'", state['symbol'])[0]
    except Exception as err:
        return 'o'


class CustomAnnotationsGuiElements:
    def gui_createCustomAnnotationsActions(self):
        self.loadCustomAnnotationsAction = QAction(self)
        self.loadCustomAnnotationsAction.setIcon(QIcon(":load_annotation.svg"))
        self.loadCustomAnnotationsAction.setToolTip(
            'Load previously used custom annotations'
        )
    
        self.addCustomAnnotationAction = QAction(self)
        self.addCustomAnnotationAction.setIcon(QIcon(":addCustomAnnotation.svg"))
        self.addCustomAnnotationAction.setToolTip('Add custom annotation')

        self.viewAllCustomAnnotAction = QAction(self)
        self.viewAllCustomAnnotAction.setCheckable(True)
        self.viewAllCustomAnnotAction.setIcon(QIcon(":eye.svg"))
        self.viewAllCustomAnnotAction.setToolTip('Show all custom annotations')
    
    def gui_connectCustomAnnotationsActions(self):
        self.loadCustomAnnotationsAction.triggered.connect(
            self.loadCustomAnnotations
        )
        self.addCustomAnnotationAction.triggered.connect(
            self.addCustomAnnotation
        )
        self.viewAllCustomAnnotAction.toggled.connect(
            self.viewAllCustomAnnot
        )
    
    def gui_createAnnotateToolbar(
            self, area=Qt.LeftToolBarArea, visible=False
        ):
        self.annotateToolbar = widgets.ToolBar("Custom annotations", self)
        self.annotateToolbar.setContextMenuPolicy(Qt.PreventContextMenu)
        self.addToolBar(area, self.annotateToolbar)
        self.annotateToolbar.addAction(self.loadCustomAnnotationsAction)
        self.annotateToolbar.addAction(self.addCustomAnnotationAction)
        self.annotateToolbar.addAction(self.viewAllCustomAnnotAction)
        self.annotateToolbar.setVisible(visible)
    
    # --- Hooks ---------------------------------------------------------------
    def customAnnotIsVisibleMode(self):
        mode = self.modeComboBox.currentText()
        # Do not show annotations if timelapse and mode not annotations
        return self.isSnapshot or mode == 'Custom annotations'
    
    def customAnnotationDialogKwargs(self, state=None):
        return {}

    def canDeleteSavedCustomAnnot(self):
        """Whether the dialogs listing the saved (global) custom annotations
        offer to delete them. Hosts whose `saveCustomAnnot` does not write
        the global file (measurements viewer) return False, otherwise the
        deletion would not persist."""
        return True

    def customAnnotColumnRemovalNote(self, name):
        """Extra text for the removal question of custom annotation `name`
        (e.g. what reads its column). Empty in the main GUI."""
        return ''

    def customAnnotToolTip(self, state):
        return myutils.getCustomAnnotTooltip(state)
    
    def setCustomAnnotShortcut(self, button, shortcut):
        """Set the shortcut to activate the button. `shortcut` can be a 
        `QKeySequence`, a string or None (no shortcut)."""
        if not shortcut:
            shortcut = QKeySequence()
        elif isinstance(shortcut, str):
            shortcut = widgets.KeySequenceFromText(shortcut)
        button.setShortcut(shortcut)
    
    def saveCustomAnnotPositionParams(self, posData):
        posData.saveCustomAnnotationParams()
    
    def onCustomAnnotationChanged(self, button, oldState=None):
        pass
    
    def onCustomAnnotationRemoved(self, button, state):
        pass
    
    def initCustomAnnotColumn(self, button, name):
        """Add a column of 0s called `name` to the acdc_df of each frame 
        or initialize the annotated IDs from the existing column."""
        allPosAnnotatedIDs = self.customAnnotDict[button]['annotatedIDs']
        for pos_i, posData in enumerate(self.data):
            for frame_i, data_dict in enumerate(posData.allData_li):
                acdc_df = data_dict['acdc_df']
                if acdc_df is None:
                    continue
                if name not in acdc_df.columns:
                    acdc_df[name] = 0
                else:
                    acdc_df[name] = acdc_df[name].astype(int)
                    acdc_df_annot = acdc_df[acdc_df[name] == 1].reset_index()
                    annot_IDs = acdc_df_annot['Cell_ID'].to_list()
                    allPosAnnotatedIDs[pos_i][frame_i].extend(annot_IDs)
                    
            if posData.acdc_df is None:
                continue
            
            if name not in posData.acdc_df.columns:
                posData.acdc_df[name] = 0
                continue
            
            posData.acdc_df[name] = posData.acdc_df[name].astype(int)
            acdc_df_annot = (
                posData.acdc_df[posData.acdc_df[name] == 1].reset_index()
            )
            if 'frame_i' not in acdc_df_annot.columns:
                continue
            for frame_i, df in acdc_df_annot.groupby('frame_i'):
                annotIDs = allPosAnnotatedIDs[pos_i][frame_i]
                for ID in df['Cell_ID'].to_list():
                    if ID not in annotIDs:
                        annotIDs.append(ID)
    
    def renameCustomAnnotColumn(self, oldName, newName):
        for posData in self.data:
            for data_dict in posData.allData_li:
                acdc_df = data_dict['acdc_df']
                if acdc_df is None or oldName not in acdc_df.columns:
                    continue
                data_dict['acdc_df'] = acdc_df.rename(
                    columns={oldName: newName}
                )
            acdc_df = getattr(posData, 'acdc_df', None)
            if acdc_df is not None and oldName in acdc_df.columns:
                posData.acdc_df = acdc_df.rename(columns={oldName: newName})
    
    def removeCustomAnnotColumn(self, name):
        for posData in self.data:
            if posData.acdc_df is None:
                continue
            posData.acdc_df = posData.acdc_df.drop(
                columns=name, errors='ignore'
            )
            for frame_i, data_dict in enumerate(posData.allData_li):
                acdc_df = data_dict['acdc_df']
                if acdc_df is None:
                    continue
                acdc_df = acdc_df.drop(columns=name, errors='ignore')
                posData.allData_li[frame_i]['acdc_df'] = acdc_df
    
    # --- Shared logic --------------------------------------------------------
    def positionCustomAnnot(self, posData):
        """Custom annotations of `posData` {name: state} (created if 
        missing, e.g., when the position was loaded without them)."""
        customAnnot = getattr(posData, 'customAnnot', None)
        if not isinstance(customAnnot, dict):
            customAnnot = {}
            posData.customAnnot = customAnnot
        return customAnnot
    
    def customAnnotButtonByName(self, name):
        for button, annot in self.customAnnotDict.items():
            if annot['state']['name'] == name:
                return button
    
    def customAnnotButtonsToDraw(self):
        """Buttons whose annotations are displayed."""
        if self.viewAllCustomAnnotAction.isChecked():
            # User requested to show all annotations --> iterate all buttons
            return list(self.customAnnotDict.keys())
        # Annotate if the button is active or isHideChecked is False
        return [
            b for b in self.customAnnotDict.keys()
            if (b.isChecked() or not b.isHideChecked)
        ]
    
    def setUncheckedAllCustomAnnotButtons(self):
        for button in self.customAnnotDict.keys():
            button.setChecked(False)
    
    def undoCustomAnnotation(self):
        pass
    
    def readSavedCustomAnnot(self):
        tempAnnot = {}
        if os.path.exists(custom_annot_path):
            self.logger.info('Loading saved custom annotations...')
            tempAnnot = load.read_json(
                custom_annot_path, logger_func=self.logger.info
            )

        self.savedCustomAnnot = tempAnnot
        for posData in self.data:
            self.savedCustomAnnot = {
                **self.savedCustomAnnot, **self.positionCustomAnnot(posData)
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
            toolTip = self.customAnnotToolTip(annotState)
            keepActive = annotState.get('keepActive', True)
            isHideChecked = annotState.get('isHideChecked', True)

            toolButton, action = self.addCustomAnnotationButton(
                symbol, symbolColor, annotState['shortcut'], toolTip, name,
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
            self, symbol, symbolColor, shortcut, toolTip, annotName,
            keepActive, isHideChecked
        ):
        toolButton = widgets.customAnnotToolButton(
            symbol, symbolColor, parent=self, keepToolActive=keepActive,
            isHideChecked=isHideChecked
        )
        toolButton.setCheckable(True)
        buttonsGroup = getattr(self, 'checkableQButtonsGroup', None)
        if buttonsGroup is not None:
            buttonsGroup.addButton(toolButton)
        toolButton.setToolTip(toolTip)
        toolButton.name = annotName
        self.setCustomAnnotShortcut(toolButton, shortcut)
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
            self, symbol, symbolColor, shortcut, toolTip, name,
            keepActive, isHideChecked, state
        ):
        toolButton, action = self.addCustomAnnotationButton(
            symbol, symbolColor, shortcut, toolTip, name,
            keepActive, isHideChecked
        )

        self.customAnnotDict[toolButton] = {
            'action': action,
            'state': state,
            'annotatedIDs': [defaultdict(list) for _ in range(len(self.data))]
        }

        # Save custom annotation to cellacdc/temp/custom_annotations.json
        state_to_save = custom_annot_state_to_save(state)
        self.savedCustomAnnot[name] = state_to_save
        for posData in self.data:
            self.positionCustomAnnot(posData)[name] = state_to_save

        self.addCustomAnnnotScatterPlot(symbolColor, symbol, toolButton)
        self.initCustomAnnotColumn(toolButton, name)
        return toolButton
        
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
        self.readSavedCustomAnnot()
        items = list(self.savedCustomAnnot.keys())
        if len(items) == 0:
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph("""
            There are no custom annotations saved.<br><br>
            Click on "Add custom annotation" button to start adding new 
            annotations.
            """)
            msg.warning(self, 'No annotations saved', txt)
            return
        
        additionalButtons = (
            ('Delete selected annnotations', )
            if self.canDeleteSavedCustomAnnot() else ()
        )
        self.selectAnnotWin = widgets.QDialogListbox(
            'Load previously used custom annotation(s)',
            'Select annotations to load:', items,
            additionalButtons=additionalButtons,
            parent=self, multiSelection=True
        )
        for button in getattr(self.selectAnnotWin, '_additionalButtons', ()):
            button.disconnect()
            button.clicked.connect(self.deleteSavedAnnotation)
        self.selectAnnotWin.exec_()
        if self.selectAnnotWin.cancel:
            return
        
        for selectedAnnotName in self.selectAnnotWin.selectedItemsText:
            if self.customAnnotButtonByName(selectedAnnotName) is not None:
                self.logger.info(
                    f'Custom annotation "{selectedAnnotName}" already loaded.'
                )
                continue
            
            selectedAnnot = self.savedCustomAnnot[selectedAnnotName]
            symbol = custom_annot_symbol(selectedAnnot)
            symbolColor = pg.mkColor(selectedAnnot['symbolColor'])
            keepActive = selectedAnnot.get('keepActive', True)
            isHideChecked = selectedAnnot.get('isHideChecked', True)
            # Additional keys (e.g., predictor) are preserved
            state = {
                **selectedAnnot,
                'name': selectedAnnotName,
                'keepActive': keepActive,
                'isHideChecked': isHideChecked,
                'symbolColor': symbolColor
            }
            toolTip = self.customAnnotToolTip(state)
            button = self.addCustomAnnotationItems(
                symbol, symbolColor, selectedAnnot['shortcut'], toolTip, 
                selectedAnnotName, keepActive, isHideChecked, state
            )
            self.onCustomAnnotationChanged(button)
            
        self.saveCustomAnnot()
        self.doCustomAnnotation(0)
    
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
            self.savedCustomAnnot, parent=self,
            allowDeleteSaved=self.canDeleteSavedCustomAnnot(),
            **self.customAnnotationDialogKwargs()
        )
        self.addAnnotWin.sigDeleteSelecAnnot.connect(self.deleteSelectedAnnot)
        self.addAnnotWin.exec_()
        if self.addAnnotWin.cancel:
            self.logger.info('Custom annotation process cancelled.')
            return

        state = self.addAnnotWin.state
        symbol = self.addAnnotWin.symbol
        symbolColor = state['symbolColor']
        name = state['name']
        keepActive = state.get('keepActive', True)
        isHideChecked = state.get('isHideChecked', True)
        
        proceed = self.checkNameExists(name)
        if not proceed:
            self.logger.info('Custom annotation process cancelled.')
            return

        button = self.addCustomAnnotationItems(
            symbol, symbolColor, state['shortcut'], 
            self.customAnnotToolTip(state), name, keepActive, isHideChecked, 
            state
        )
        self.saveCustomAnnot()
        self.onCustomAnnotationChanged(button)
        self.doCustomAnnotation(0)
        return button

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
            self, 'Custom annotation name already exists', txt,
            buttonsTexts=('No, stop process', 'Yes, use existing column')
        )
        return msg.clickedButton == yesButton
    
    def checkNameExists(self, name):
        posData = self.data[self.pos_i]
        for frame_i, data_dict in enumerate(posData.allData_li):
            acdc_df = data_dict['acdc_df']
            if acdc_df is None:
                continue
            if name in acdc_df.columns:
                return self.askCustomAnnotationNameExists(name)
        
        acdc_df = getattr(posData, 'acdc_df', None)
        if acdc_df is not None and name in acdc_df.columns:
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
    
    def clearCustomAnnot(self):
        for button in self.customAnnotDict.keys():
            self.clearScatterPlotCustomAnnotButton(button)

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
        for _posData in self.data:
            self.saveCustomAnnotPositionParams(_posData)

    def customAnnotKeepActive(self, button):
        self.customAnnotDict[button]['state']['keepActive'] = button.keepToolActive

    def customAnnotHide(self, button):
        state = self.customAnnotDict[button]['state']
        state['isHideChecked'] = button.isHideChecked
        state_to_save = custom_annot_state_to_save(state)
        self.savedCustomAnnot[state['name']] = state_to_save
        for posData in self.data:
            self.positionCustomAnnot(posData)[state['name']] = state_to_save
        self.saveCustomAnnot()
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
        oldState = self.customAnnotDict[button]['state']
        self.addAnnotWin = apps.customAnnotationDialog(
            self.savedCustomAnnot, parent=self, state=oldState,
            allowDeleteSaved=self.canDeleteSavedCustomAnnot(),
            **self.customAnnotationDialogKwargs(state=oldState)
        )
        self.addAnnotWin.sigDeleteSelecAnnot.connect(self.deleteSelectedAnnot)
        self.addAnnotWin.exec_()
        if self.addAnnotWin.cancel:
            return

        state = self.addAnnotWin.state
        oldName = oldState['name']
        name = state['name']
        if name != oldName:
            self.renameCustomAnnotColumn(oldName, name)
            for posData in self.data:
                self.positionCustomAnnot(posData).pop(oldName, None)

        self.customAnnotDict[button]['state'] = state
        state_to_save = custom_annot_state_to_save(state)
        self.savedCustomAnnot[name] = state_to_save
        for posData in self.data:
            self.positionCustomAnnot(posData)[name] = state_to_save
        self.saveCustomAnnot()

        button.name = name
        button.keepToolActive = state.get('keepActive', True)
        button.isHideChecked = state.get('isHideChecked', True)
        button.setToolTip(self.customAnnotToolTip(state))
        self.setCustomAnnotShortcut(button, state['shortcut'])

        symbol = self.addAnnotWin.symbol
        symbolColor = pg.mkColor(state['symbolColor'])
        button.symbol = symbol
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
            pen=pg.mkPen(width=3, color=symbolColor),
            hoverBrush=pg.mkBrush(symbolColor)
        )
        self.onCustomAnnotationChanged(button, oldState=oldState)

    def doCustomAnnotation(self, ID):
        if not self.customAnnotIsVisibleMode():
            return
        
        if self.switchPlaneCombobox.depthAxes() != 'z': 
            return
        
        # NOTE: pass 0 for ID to not add
        posData = self.data[self.pos_i]
        buttons = self.customAnnotButtonsToDraw()
        if not buttons:
            return

        for button in buttons:
            annotatedIDs = (
                self.customAnnotDict[button]['annotatedIDs'][self.pos_i]
            )
            annotIDs_frame_i = annotatedIDs.get(posData.frame_i, [])
            state = self.customAnnotDict[button]['state']
            acdc_df = posData.allData_li[posData.frame_i]['acdc_df']
            
            if button.isChecked() and ID > 0:
                # Annotate only if existing ID and the button is checked
                if ID in annotIDs_frame_i:
                    annotIDs_frame_i.remove(ID)
                    acdc_df.at[ID, state['name']] = 0
                elif ID != 0:
                    annotIDs_frame_i.append(ID)
            
            annotPerButton = self.customAnnotDict[button]
            allAnnotedIDs = annotPerButton['annotatedIDs']
            posAnnotedIDs = allAnnotedIDs[self.pos_i]
            posAnnotedIDs[posData.frame_i] = annotIDs_frame_i
            
            if acdc_df is None:
                self.store_data(autosave=False)
            acdc_df = posData.allData_li[posData.frame_i]['acdc_df']
            
            xx, yy = [], []
            for annotID in annotIDs_frame_i:
                if annotID not in posData.rp.IDs:
                    continue
                obj = posData.rp.get_obj_from_ID(annotID)
                acdc_df.at[annotID, state['name']] = 1
                if not self.isObjInCurrSlice(obj.bbox):
                    continue
                y, x = self.getObjCentroid(
                    posData.rp.get_centroid(annotID, exact=True))
                xx.append(x)
                yy.append(y)
                
            scatterPlotItem = self.customAnnotDict[button]['scatterPlotItem']
            scatterPlotItem.setData(xx, yy)

            posData.allData_li[posData.frame_i]['acdc_df'] = acdc_df

        if buttons:
            return buttons[0]
    
    def drawCustomAnnotationIDs(self, button, IDs):
        """Draw the symbol of `button` on the objects `IDs` of the current 
        frame."""
        posData = self.data[self.pos_i]
        xx, yy = [], []
        for annotID in IDs:
            obj = posData.rp.get_obj_from_ID(annotID, warn=False)
            if obj is None:
                continue
            if not self.isObjInCurrSlice(obj.bbox):
                continue
            y, x = self.getObjCentroid(
                posData.rp.get_centroid(annotID, exact=True)
            )
            xx.append(x)
            yy.append(y)
        scatterPlotItem = self.customAnnotDict[button]['scatterPlotItem']
        scatterPlotItem.setData(xx, yy)

    def removeCustomAnnotButton(self, button, askHow=True, save=True):
        if askHow:
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph("""
                Do you want to <b>remove also the column with annotations</b> or
                only the annotation button?<br>
            """)
            note = self.customAnnotColumnRemovalNote(
                self.customAnnotDict[button]['state']['name']
            )
            if note:
                txt = f'{txt}{html_utils.paragraph(note)}'
            _, removeOnlyButton, removeColButton = msg.question(
                self, 'Remove only button?', txt, 
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
        
        state = self.customAnnotDict[button]['state']
        name = state['name']
        for posData in self.data:
            customAnnot = self.positionCustomAnnot(posData)
            if name not in customAnnot:
                continue
            customAnnot.pop(name)
            self.saveCustomAnnotPositionParams(posData)
        
        if not removeOnlyButton:
            self.removeCustomAnnotColumn(name)

        self.removeCustomAnnotItems(button)
        self.saveCustomAnnot(only_temp=True)
        self.onCustomAnnotationRemoved(button, state)
    
    def removeCustomAnnotItems(self, button):
        self.clearScatterPlotCustomAnnotButton(button)
        scatterPlotItem = self.customAnnotDict[button]['scatterPlotItem']
        self.ax1.removeItem(scatterPlotItem)
        if self.customAnnotButton is button:
            self.customAnnotButton = None
        action = self.customAnnotDict[button]['action']
        self.annotateToolbar.removeAction(action)
        buttonsGroup = getattr(self, 'checkableQButtonsGroup', None)
        if buttonsGroup is not None:
            buttonsGroup.removeButton(button)
        self.customAnnotDict.pop(button)
    
    def reinitCustomAnnot(self):
        buttons = list(self.customAnnotDict.keys())
        for button in buttons:
            state = self.customAnnotDict[button]['state']
            self.removeCustomAnnotItems(button)
            self.saveCustomAnnot(only_temp=True)
            self.onCustomAnnotationRemoved(button, state)

    def customAnnotButtonToggled(self, checked):
        if checked:
            self.customAnnotButton = self.sender()
            # Uncheck the other buttons
            for button in self.customAnnotDict.keys():
                if button == self.sender():
                    continue

                button.toggled.disconnect()
                button.setChecked(False)
                button.toggled.connect(self.customAnnotButtonToggled)
                if button not in self.customAnnotButtonsToDraw():
                    self.clearScatterPlotCustomAnnotButton(button)
            self.doCustomAnnotation(0)
        else:
            self.customAnnotButton = None
            button = self.sender()
            clearAnnotation = (
                button.isHideChecked
                and not self.viewAllCustomAnnotAction.isChecked()
            )
            if clearAnnotation:    
                self.clearScatterPlotCustomAnnotButton(button)
            self.setHighlightID(False)
            self.resetCursor()
