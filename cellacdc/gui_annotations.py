"""Mixin with the GUI elements and the logic to display annotations of the 
segmentation masks (contours, segm. masks overlay, IDs, cell cycle and lineage 
info, mother-daughter lines and object tracks) on the image axes.

It is shared by the main GUI (`gui.guiWin`) and the signals viewer 
(`signals_viewer.SignalsViewerWin`).

Requirements on the inheriting class
------------------------------------
- `self.data`, `self.pos_i` with `posData` having `allData_li`, `frame_i`, 
  `lab`, `rp`, `IDs`, `cca_df`, `lost_IDs`, `new_IDs`, `trackedLostIDs`, 
  `tracked_lost_centroids` and `whitelist`.
- `self.df_settings` and `self.settings_csv_path`.
- `self.imgGrad` (`widgets.myHistogramLUTitem`) and `self.labelsGrad` 
  (`widgets.labelsGradientWidget`).
- The z-slice navigation widgets (see `gui_navigation`).
- The method `annotatedAxesIndices` returns the indices of the axes with 
  annotations (main GUI: left and right image, i.e., (0, 1)).
"""
from functools import partial
from itertools import combinations

import numpy as np
import cv2

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QWidget, QGridLayout

import pyqtgraph as pg

from . import widgets, html_utils, annotate, core, colors, apps, printl

# (row, col, rowSpan, colSpan, slot)
CHECKBOX_OPTION_NAME_TO_LAYOUT_LOC_MAPPER = {
    'Contours': (0, 0, 1, 1, 'onContoursChecked'),
    'Segm. masks': (1, 0, 1, 1, 'onOverlaySegmMask'),
    'Separator_1': (0, 1, 2, 1, None),
    'IDs': (0, 2, 1, 1, 'onIDsChecked'),
    'Lineage info': (1, 2, 1, 1, 'onLineageInfo'),
    'Cell cycle info': (0, 3, 1, 1, 'onCellCycleInfo'),
    'Mother-daughter line': (1, 3, 1, 1, 'onMotherDaughterLine'),
    'Separator_2': (0, 4, 2, 1, None),
    'Object tracks': (0, 5, 1, 1, 'onObjectTracksChecked'),
    'Do not annotate': (1, 5, 1, 1, 'onDoNotAnnotateChecked')
}
CHECKBOX_OPTION_MUTUALLY_EXCLUSIVE_GROUPS = (
    ('Contours', 'Segm. masks'),
    ('IDs', 'Lineage info', 'Cell cycle info'),
)
CHECKBOX_OPTION_TOOLTIPS = {
    'Contours': '''Show contour outlines of segmentation masks. Customize 
    appearance by right clicking on the image. By default, "internal" 
    contours are not shown. This can be toggled by the option in the 
    bottom left menu.''',
    'Segm. masks': '''Show segmentation masks as a semi-transparent overlay. 
    Customize by right clicking on the image.''',
    'IDs': '''Show IDs of the current segmentation mask. Can be slow for many 
    objects in a single frame''',
    'Lineage info': '''Show information about the lineage for each displayed 
    object, linked to the "Normal Division: Lineage Tree" mode.''',
    'Cell cycle info': '''Show information about the lineage and cell cycle 
    stage for each displayed object. Linked to the "Cell cycle analysis" mode 
    for asymmetrically dividing cells.''',
    'Mother-daughter line': '''Show lines between mothers and daughters, for both 
    "Normal Division: Lineage Tree" and "Cell cycle analysis" mode. Customize 
    by right clicking on the image.''',
    'Object tracks': '''Show tracks of objects in current frame. Customize 
    by right clicking on the bottom part of the GUI or through "Settings" in 
    the top ribbon.'''
}

CHECKBOX_OPTION_DEFAULT_VALUES = {
    0: {'Contours', 'IDs'},
    1: {'Segm. masks', 'IDs'}
}


class AnnotationsGuiElements:
    def annotatedAxesIndices(self):
        """Indices of the axes where annotations are displayed (0 is the 
        left image and 1 is the right image)"""
        return (0, 1)

    def gui_setupAnnotationOptionsLayout(self, ax: int):  
        container = QWidget()
        layout = QGridLayout()
        container.setLayout(layout)
        for name, info in CHECKBOX_OPTION_NAME_TO_LAYOUT_LOC_MAPPER.items():
            row, col, rowSpan, colSpan, slot = info

            if name.startswith('Separator'):
                checkbox = None
                widget = widgets.QVLine()
            else:
                checkbox = widgets.CheckBox(
                    name, keyPressCallback=self.resetFocus
                )
                widget = checkbox

            layout.addWidget(widget, row, col, rowSpan, colSpan)

            if checkbox is None:
                continue

            if name in CHECKBOX_OPTION_TOOLTIPS:
                txt = CHECKBOX_OPTION_TOOLTIPS[name]
                txt = html_utils.paragraph(txt)
                checkbox.setToolTip(txt)

            if slot is None:
                continue

            self.annotOptionsCheckboxes[ax][name] = checkbox
            checkbox.sigToggled.connect(
                partial(getattr(self, slot), ax=ax)
            )

        for mutuallyExclusiveGroup in CHECKBOX_OPTION_MUTUALLY_EXCLUSIVE_GROUPS:
            for name1, name2 in combinations(mutuallyExclusiveGroup, 2):
                checkbox1 = self.annotOptionsCheckboxes[ax][name1]
                checkbox2 = self.annotOptionsCheckboxes[ax][name2]
                checkbox1.setExclusiveOnCheckCheckbox(checkbox2)
                checkbox2.setExclusiveOnCheckCheckbox(checkbox1)
        
        doNotAnnotateCheckbox = (
            self.annotOptionsCheckboxes[ax]['Do not annotate']
        )
        for checkbox in self.annotOptionsCheckboxes[ax].values():
            if checkbox == doNotAnnotateCheckbox:
                continue
            
            checkbox.setExclusiveOnCheckCheckbox(doNotAnnotateCheckbox)
            doNotAnnotateCheckbox.setExclusiveOnCheckCheckbox(checkbox)

        layout.setColumnStretch(layout.columnCount(), 1)
        layout.setContentsMargins(0, 0, 0, 5)

        self.annotOptionsCheckboxes[ax]['layout'] = layout
        self.annotOptionsCheckboxes[ax]['container'] = container

    def annotLineageInfoCheckbox(self, ax: int):
        return self.annotOptionsCheckboxes[ax]['Lineage info']

    def annotIDsCheckbox(self, ax: int):
        return self.annotOptionsCheckboxes[ax]['IDs']

    def annotContourCheckbox(self, ax: int):
        return self.annotOptionsCheckboxes[ax]['Contours']

    def annotObjectTracksCheckbox(self, ax: int):
        return self.annotOptionsCheckboxes[ax]['Object tracks']

    def annotCellCycleInfoCheckbox(self, ax: int):
        return self.annotOptionsCheckboxes[ax]['Cell cycle info']

    def annotOverlaySegmMaskCheckbox(self, ax: int):
        return self.annotOptionsCheckboxes[ax]['Segm. masks']

    def annotMotherDaughterLineCheckbox(self, ax: int):
        return self.annotOptionsCheckboxes[ax]['Mother-daughter line']

    def annotDoNotAnntoateCheckbox(self, ax: int):
        return self.annotOptionsCheckboxes[ax]['Do not annotate']

    def isOverlaySegmMaskChecked(self, ax: int):
        if ax == 1 and not self.labelsGrad.showRightImgAction.isChecked():
            return False
        
        return self.annotOverlaySegmMaskCheckbox(ax).isChecked()

    def isContoursChecked(self, ax: int):
        return self.annotContourCheckbox(ax).isChecked()

    def isObjectTracksChecked(self, ax: int):
        return self.annotObjectTracksCheckbox(ax).isChecked()

    def isCellCycleInfoChecked(self, ax: int):
        return self.annotCellCycleInfoCheckbox(ax).isChecked()

    def isMotherDaughterLineChecked(self, ax: int):
        return self.annotMotherDaughterLineCheckbox(ax).isChecked()

    def onContoursChecked(self, checked, checkbox, ax=0):
        if checked:
            self.updateContoursImage(ax=ax, delROIsIDs=None, compute=False)
        else:
            imageItem = self.getContoursImageItem(ax, force=True)
            if imageItem is None:
                return
            
            imageItem.clear()

    def onOverlaySegmMask(self, checked, checkbox, ax=0):
        if checked:
            self.setOverlaySegmMasks()
        else:
            if ax == 0:
                self.labelsLayerImg1.clear()
            else:
                self.labelsLayerRightImg.clear()

    def onIDsChecked(self, checked, checkbox, ax=0):
        self.textAnnot[ax].setLabelAnnot(checked)
        if checked:
            self.setAllTextAnnotations()
        else:
            self.textAnnot[ax].clear()
            self.updateViewBox(ax)

    def onLineageInfo(self, checked, checkbox, ax=0):
        self.textAnnot[ax].setLineageAnnot(checked)
        if checked:
            self.setAllTextAnnotations()
        else:
            self.textAnnot[ax].clear()
            self.updateViewBox(ax)

    def onCellCycleInfo(self, checked, checkbox, ax=0):
        self.textAnnot[ax].setCcaAnnot(checked)
        if checked:
            self.setAllTextAnnotations()
        else:
            self.textAnnot[ax].clear()
            self.updateViewBox(ax)

    def onMotherDaughterLine(self, checked, checkbox, ax=0):
        if checked:
            posData = self.data[self.pos_i]
            for obj in posData.rp:
                self.drawObjMothBudLines(obj, posData, ax=ax)
        else:
            self.getMothBudLineScatterItem(ax, True).clear()
            self.getMothBudLineScatterItem(ax, False).clear()
            self.updateViewBox(ax)

    def onObjectTracksChecked(self, checked, checkbox, ax=0):
        self.annotateAllObjectTracks()

    def onDoNotAnnotateChecked(self, checked, checkbox, ax=0):
        if checked:
            for name, isChecked in checkbox.exclusiveCheckboxesStates.items():
                self.annotOptionsToRestore[ax][name] = isChecked
        else:
            self.restoreAnnotOptions(ax)

    def storeCurrentAnnotationsOptions(self):
        for ax in self.annotatedAxesIndices():
            self.storeCurrentAnnotOptions(ax)

    def storeCurrentAnnotOptions(self, ax: int):
        if not hasattr(self, 'annotOptionsToRestore'):
            return
        
        for name, checkbox in self.annotOptionsCheckboxes[ax].items():
            try:
                isChecked = checkbox.isChecked()
            except AttributeError:
                continue

            self.annotOptionsToRestore[ax][name] = isChecked

    def restoreAnnotOptions(self, ax: int, options=None):
        if options is None and not hasattr(self, 'annotOptionsToRestore'):
            return

        if options is None:
            options = self.annotOptionsToRestore
        
        if options is None:
            return

        for name, checked in options[ax].items():
            checkbox = self.annotOptionsCheckboxes[ax][name]
            checkbox.setChecked(checked)
        
        self.annotOptionsToRestore[ax] = {}

    def restoreAnnotationsOptions(self):
        for ax in self.annotatedAxesIndices():
            self.restoreAnnotOptions(ax)

    def resetDefaultAnnotationOptions(self, *args):
        index_to_drop = []
        for ax, checkboxes in self.annotOptionsCheckboxes.items():
            for name, checkbox in checkboxes.items():
                try:
                    checkbox.setChecked(False)
                except Exception as err:
                    continue

                self.df_settings.at[f'{name};;{ax}', 'value'] = (
                    'checked' if name in CHECKBOX_OPTION_DEFAULT_VALUES[ax] 
                    else 'not_checked'
                )
        
        self.df_settings.to_csv(self.settings_csv_path)
        self.restoreSavedAnnotationOptions()

    def restoreSavedAnnotationOptions(self):
        for ax, checkboxes in self.annotOptionsCheckboxes.items():
            for name, checkbox in checkboxes.items():
                try:
                    checkbox.setChecked(False)
                except Exception as err:
                    continue

                try:
                    checked = (
                        self.df_settings.at[f'{name};;{ax}', 'value'] 
                        == 'checked'
                    )
                    checkbox.setChecked(checked)
                    continue
                except Exception as err:
                    pass

                try:
                    checked = name in CHECKBOX_OPTION_DEFAULT_VALUES[ax]
                    checkbox.setChecked(checked)
                    continue
                except Exception as err:
                    pass

    def saveAnnotationOptions(self):
        if not hasattr(self, 'annotOptionsToRestore'):
            return
        
        self.storeCurrentAnnotationsOptions()
        for ax, states in self.annotOptionsToRestore.items():
            for name, checked in states.items():
                self.df_settings.at[f'{name};;{ax}', 'value'] = (
                    'checked' if checked else 'not_checked'
                )
                self.df_settings.to_csv(self.settings_csv_path)

    def areContoursRequested(self, ax):
        if ax == 0 and self.isContoursChecked(0):
            return True

        if ax == 1:
            if not self.labelsGrad.showRightImgAction.isChecked():
                return False

            isRightDifferentAnnot = self.rightBottomGroupbox.isChecked()
            areContRequestedRight = self.isContoursChecked(1)           
            if isRightDifferentAnnot and areContRequestedRight:
                return True
            
            areContRequestedLeft = self.isContoursChecked(0)
            if not isRightDifferentAnnot and areContRequestedLeft:
                return True
            
        return False

    def areMothBudLinesRequested(self, ax):
        if ax == 0:
            if self.isMotherDaughterLineChecked(0):
                return True
        else:
            if not self.labelsGrad.showRightImgAction.isChecked():
                return False
            
            isRightDifferentAnnot = self.rightBottomGroupbox.isChecked()
            areLinesRequestedRight = self.isMotherDaughterLineChecked(1)
            
            if isRightDifferentAnnot and areLinesRequestedRight:
                return True
        
            areLinesRequestedLeft = self.isMotherDaughterLineChecked(0)
            if not isRightDifferentAnnot and areLinesRequestedLeft:
                return True
        return False

    def getMothBudLineScatterItem(self, ax, new):
        if ax == 0:
            if new:
                return self.ax1_newMothBudLinesItem
            else:
                return self.ax1_oldMothBudLinesItem
        else:
            if new:
                return self.ax2_newMothBudLinesItem
            else:
                return self.ax2_oldMothBudLinesItem

    def gui_createTextAnnotColors(self, r, g, b, custom=False):
        if custom:
            self.objLabelAnnotRgb = (int(r), int(g), int(b))
            self.SphaseAnnotRgb = (int(r*0.9), int(r*0.9), int(b*0.9))
            self.G1phaseAnnotRgba = (int(r*0.8), int(g*0.8), int(b*0.8), 220)
        else:
            self.objLabelAnnotRgb = (255, 255, 255) # white
            self.SphaseAnnotRgb = (229, 229, 229)
            self.G1phaseAnnotRgba = (204, 204, 204, 220)
        self.dividedAnnotRgb = (245, 188, 1) # orange

        self.emptyBrush = pg.mkBrush((0,0,0,0))
        self.emptyPen = pg.mkPen((0,0,0,0))

    def gui_setTextAnnotColors(self):
        for ax in self.annotatedAxesIndices():
            self.textAnnot[ax].setColors(
                self.objLabelAnnotRgb, self.dividedAnnotRgb, 
                self.SphaseAnnotRgb, self.G1phaseAnnotRgba, 
                self.objLostAnnotRgb, self.objLostTrackedAnnotRgb
            )

    def gui_createTextAnnotItems(self, allIDs):
        self.textAnnot = {}
        isHighResolution = self.highLowResAction.isChecked()
        # TextAnnotations' internal scaling flag counter-scales fonts to keep
        # their screen size fixed, so it is the inverse of the UI toggle.
        scalingMode = self.scalingLabelsAction.isChecked()
        for ax in self.annotatedAxesIndices():
            ax_textAnnot = annotate.TextAnnotations()
            ax_textAnnot.initFonts(self.fontSize)
            ax_textAnnot.createItems(
                isHighResolution, allIDs, scalingMode=scalingMode
            )
            ax_textAnnot.setZValue(1)
            self.textAnnot[ax] = ax_textAnnot

    def gui_createMothBudLinePens(self):
        if 'mothBudLineSize' in self.df_settings.index:
            val = self.df_settings.at['mothBudLineSize', 'value']
            self.mothBudLineWeight = int(val)
        else:
            self.mothBudLineWeight = 2

        self.newMothBudlineColor = (255, 0, 0)            
        if 'mothBudLineColor' in self.df_settings.index:
            val = self.df_settings.at['mothBudLineColor', 'value']
            rgba = colors.rgba_str_to_values(val)
            self.mothBudLineColor = rgba[0:3]
        else:
            self.mothBudLineColor = (255,165,0)

        try:
            self.imgGrad.mothBudLineColorButton.sigColorChanging.disconnect()
            self.imgGrad.mothBudLineColorButton.sigColorChanged.disconnect()
        except Exception as e:
            pass
        try:
            for act in self.imgGrad.mothBudLineWightActionGroup.actions():
                act.toggled.disconnect()
        except Exception as e:
            pass
        for act in self.imgGrad.mothBudLineWightActionGroup.actions():
            if act.lineWeight == self.mothBudLineWeight:
                act.setChecked(True)
            else:
                act.setChecked(False)
        self.imgGrad.mothBudLineColorButton.setColor(self.mothBudLineColor[:3])

        self.imgGrad.mothBudLineColorButton.sigColorChanging.connect(
            self.updateMothBudLineColour
        )
        self.imgGrad.mothBudLineColorButton.sigColorChanged.connect(
            self.saveMothBudLineColour
        )
        for act in self.imgGrad.mothBudLineWightActionGroup.actions():
            act.toggled.connect(self.mothBudLineWeightToggled)

        # MOther-bud lines brushes
        self.NewBudMoth_Pen = pg.mkPen(
            color=self.newMothBudlineColor, width=self.mothBudLineWeight+1, 
            style=Qt.DashLine
        )
        self.OldBudMoth_Pen = pg.mkPen(
            color=self.mothBudLineColor, width=self.mothBudLineWeight, 
            style=Qt.DashLine
        )
        
        self.redDashLinePen = pg.mkPen(
            color='r', width=2, style=Qt.DashLine
        )

        self.oldMothBudLineBrush = pg.mkBrush(self.mothBudLineColor)
        self.newMothBudLineBrush = pg.mkBrush(self.newMothBudlineColor)

    def gui_createContourPens(self):
        if 'contLineWeight' in self.df_settings.index:
            val = self.df_settings.at['contLineWeight', 'value']
            self.contLineWeight = int(val)
        else:
            self.contLineWeight = 1
        if 'contLineColor' in self.df_settings.index:
            val = self.df_settings.at['contLineColor', 'value']
            rgba = colors.rgba_str_to_values(val)
            self.contLineColor = rgba
            self.newIDlineColor = [min(255, v+50) for v in self.contLineColor]
        else:
            self.contLineColor = (255, 0, 0, 200)
            self.newIDlineColor = (255, 0, 0, 255)

        try:
            self.imgGrad.contoursColorButton.sigColorChanging.disconnect()
            self.imgGrad.contoursColorButton.sigColorChanged.disconnect()
        except Exception as e:
            pass
        try:
            for act in self.imgGrad.contLineWightActionGroup.actions():
                act.toggled.disconnect()
        except Exception as e:
            pass
        for act in self.imgGrad.contLineWightActionGroup.actions():
            if act.lineWeight == self.contLineWeight:
                act.setChecked(True)
        self.imgGrad.contoursColorButton.setColor(self.contLineColor[:3])

        self.imgGrad.contoursColorButton.sigColorChanging.connect(
            self.updateContColour
        )
        self.imgGrad.contoursColorButton.sigColorChanged.connect(
            self.saveContColour
        )
        for act in self.imgGrad.contLineWightActionGroup.actions():
            act.toggled.connect(self.contLineWeightToggled)

        # Contours pens
        self.oldIDs_cpen = pg.mkPen(
            color=self.contLineColor, width=self.contLineWeight
        )
        self.newIDs_cpen = pg.mkPen(
            color=self.newIDlineColor, width=self.contLineWeight+1
        )
        self.tempNewIDs_cpen = pg.mkPen(
            color='g', width=self.contLineWeight+1
        )

    def gui_getLostObjScatterItem(self):
        self.objLostAnnotRgb = (245, 184, 0)
        brush = pg.mkBrush((*self.objLostAnnotRgb, 150))
        pen = pg.mkPen(self.objLostAnnotRgb, width=1)
        lostObjScatterItem = pg.ScatterPlotItem(
            size=self.contLineWeight+1, pen=pen, 
            brush=brush, pxMode=False, symbol='s'
        )
        return lostObjScatterItem

    def gui_getTrackedLostObjScatterItem(self):
        self.objLostTrackedAnnotRgb = (0, 255, 0)
        brush = pg.mkBrush((*self.objLostTrackedAnnotRgb, 150))
        pen = pg.mkPen(self.objLostTrackedAnnotRgb, width=1)
        lostObjScatterItem = pg.ScatterPlotItem(
            size=self.contLineWeight+1, pen=pen, 
            brush=brush, pxMode=False, symbol='s'
        )
        return lostObjScatterItem

    def updateContColour(self, colorButton):
        color = colorButton.color().getRgb()
        self.df_settings.at['contLineColor', 'value'] = str(color)
        self._updateContColour(color)
        self.updateAllImages()

    def _updateContColour(self, color):
        self.gui_createContourPens()
        for items in self.overlayLayersItems.values():
            lutItem = items[1]
            lutItem.contoursColorButton.setColor(color)

    def saveContColour(self, colorButton):
        self.df_settings.to_csv(self.settings_csv_path)

    def updateMothBudLineColour(self, colorButton):
        color = colorButton.color().getRgb()
        self.df_settings.at['mothBudLineColor', 'value'] = str(color)
        self._updateMothBudLineColour(color)
        self.updateAllImages()

    def _updateMothBudLineColour(self, color):
        self.gui_createMothBudLinePens()
        for ax in self.annotatedAxesIndices():
            self.getMothBudLineScatterItem(ax, True).setBrush(
                self.newMothBudLineBrush
            )
            self.getMothBudLineScatterItem(ax, False).setBrush(
                self.oldMothBudLineBrush
            )
        for items in self.overlayLayersItems.values():
            lutItem = items[1]
            lutItem.mothBudLineColorButton.setColor(color)

    def saveMothBudLineColour(self, colorButton):
        self.df_settings.to_csv(self.settings_csv_path)

    def contLineWeightToggled(self, checked=True):
        if not checked:
            return
        self.imgGrad.uncheckContLineWeightActions()
        w = self.sender().lineWeight
        self.df_settings.at['contLineWeight', 'value'] = w
        self.df_settings.to_csv(self.settings_csv_path)
        self._updateContLineThickness()
        self.updateAllImages()

    def _updateContLineThickness(self):
        self.gui_createContourPens()
        for act in self.imgGrad.contLineWightActionGroup.actions():
            if act == self.sender():
                act.setChecked(True)
            act.toggled.connect(self.contLineWeightToggled)

    def mothBudLineWeightToggled(self, checked=True):
        if not checked:
            return
        self.imgGrad.uncheckContLineWeightActions()
        w = self.sender().lineWeight
        self.df_settings.at['mothBudLineSize', 'value'] = w
        self.df_settings.to_csv(self.settings_csv_path)
        self._updateMothBudLineSize(w)
        self.updateAllImages()

    def _updateMothBudLineSize(self, size):
        self.gui_createMothBudLinePens()
        
        for act in self.imgGrad.mothBudLineWightActionGroup.actions():
            if act == self.sender():
                act.setChecked(True)
            act.toggled.connect(self.mothBudLineWeightToggled)
        
        for ax in self.annotatedAxesIndices():
            self.getMothBudLineScatterItem(ax, False).setSize(size)
            self.getMothBudLineScatterItem(ax, True).setSize(size)

    def toggleTextIDsColorButton(self, checked=True):
        self.textIDsColorButton.selectColor()

    def updateTextAnnotColor(self, button):
        r, g, b = np.array(self.textIDsColorButton.color().getRgb()[:3])
        self.imgGrad.textColorButton.setColor((r, g, b))
        for items in self.overlayLayersItems.values():
            lutItem = items[1]
            lutItem.textColorButton.setColor((r, g, b))
        self.gui_createTextAnnotColors(r,g,b, custom=True)
        self.gui_setTextAnnotColors()
        self.updateAllImages()

    def saveTextIDsColors(self, button):
        self.df_settings.at['textIDsColor', 'value'] = self.objLabelAnnotRgb
        self.df_settings.to_csv(self.settings_csv_path)

    def setLut(self, shuffle=True):
        self.lut = self.labelsGrad.item.colorMap().getLookupTable(0,1,255)     
        if shuffle:
            np.random.shuffle(self.lut)
        
        # Insert background color
        if 'labels_bkgrColor' in self.df_settings.index:
            rgbString = self.df_settings.at['labels_bkgrColor', 'value']
            try:
                r, g, b = rgbString
            except Exception as e:
                r, g, b = colors.rgb_str_to_values(rgbString)
        else:
            r, g, b = 25, 25, 25
            self.df_settings.at['labels_bkgrColor', 'value'] = (r, g, b)

        self.lut = np.insert(self.lut, 0, [r, g, b], axis=0)

    def extendLabelsLUT(self, lenNewLut):
        posData = self.data[self.pos_i]
        # Build a new lut to include IDs > than original len of lut
        if lenNewLut > len(self.lut):
            numNewColors = lenNewLut-len(self.lut)
            # Index original lut
            _lut = np.zeros((lenNewLut, 3), np.uint8)
            _lut[:len(self.lut)] = self.lut
            # Pick random colors and append them at the end to recycle them
            randomIdx = np.random.randint(0,len(self.lut),size=numNewColors)
            for i, idx in enumerate(randomIdx):
                rgb = self.lut[idx]
                _lut[len(self.lut)+i] = rgb
            self.lut = _lut
            self.initLabelsImageItems()
            return True
        return False

    def getLabelsImageLut(self):
        lut = np.zeros((len(self.lut), 4), dtype=np.uint8)
        lut[:,-1] = 255
        lut[:,:-1] = self.lut
        lut[0] = [0,0,0,0]
        return lut

    def initContoursImage(self):
        posData = self.data[self.pos_i]
        z_slice = self.z_lab()
        img = posData.img_data[posData.frame_i]
        Y, X = img[z_slice].shape[-2:]
            
        self.contoursImage = np.zeros((Y, X, 4), dtype=np.uint8)

    def getContoursImageItem(self, ax, force=False):
        if not self.areContoursRequested(ax) and not force:
            return
                
        if ax == 0:
            return self.ax1_contoursImageItem
        else:
            return self.ax2_contoursImageItem

    def updateContoursImage(self, ax, delROIsIDs=None, compute=True):
        if delROIsIDs is None:
            delROIsIDs = []
        imageItem = self.getContoursImageItem(ax)
        if imageItem is None:
            return
        
        if not hasattr(self, 'contoursImage'):
            self.initContoursImage()
        else:
            self.contoursImage[:] = 0
        
        contours = []
        display_rp = self.get2DRP()

        for obj in display_rp:
            if delROIsIDs and obj.label in delROIsIDs:
                continue
            contours_obj = self.getObjContours(
                obj,
                all_external=True,
                include_internal=self.showAllContoursToggle.isChecked(),
            )
            contours.extend(contours_obj)
            
        thickness = self.contLineWeight
        color = self.contLineColor
        self.setContoursImage(imageItem, contours, thickness, color)

    def setContoursImage(self, imageItem, contours, thickness, color):
        cv2.drawContours(self.contoursImage, contours, -1, color, thickness)
        imageItem.setImage(self.contoursImage)

    def getObjContours(
            self, obj, all_external=False, local=False,
            include_internal=False, rp=None
        ):        
        try:
            contours = core.get_obj_contours(
                obj=obj,
                local=local,
                all_external=all_external,
                all=include_internal
            )
                
        except Exception as e:
            if all_external:
                contours = []
            else:
                contours = None
            self.logger.warning(
                f'Object ID {obj.label} contours drawing failed. '
                f'(bounding box = {obj.bbox})'
                f'Error: {e}'
            )
        return contours

    def setAllContoursImages(self, delROIsIDs=None, compute=True):
        for ax in self.annotatedAxesIndices():
            self.updateContoursImage(
                ax=ax, delROIsIDs=delROIsIDs, compute=compute
            )

    # @exec_time
    def setAllTextAnnotations(self, labelsToSkip=None):
        self.setLostNewOldPrevIDs()
        posData = self.data[self.pos_i]
        
        for ax in self.annotatedAxesIndices():
            kwargs = {}
            if ax == 0:
                kwargs['getCurrentZfunc'] = self.z_lab
            self.textAnnot[ax].setAnnotations(
                posData=posData, 
                labelsToSkip=labelsToSkip, 
                isVisibleCheckFunc=self.isObjInCurrSlice,
                highlightedID=self.highlightedID, 
                annotateLost=self.annotLostObjsToggle.isChecked(), 
                getObjCentroidFunc=self.getObjCentroid,
                rp_func=self.get2DRP,
                rp3D=posData.rp,
                **kwargs
            )
        for ax in self.annotatedAxesIndices():
            self.textAnnot[ax].update()

    def initTextAnnot(self, force=False):
        posData = self.data[self.pos_i]
        if hasattr(posData, 'lab'):
            Y, X = posData.lab.shape[-2:]
        else:
            Y, X = posData.img_data.shape[-2:]
        for ax in self.annotatedAxesIndices():
            self.textAnnot[ax].initItem((Y, X))

    def clearAnnotItems(self):
        for ax in self.annotatedAxesIndices():
            self.textAnnot[ax].clear()

    def setOverlaySegmMasks(self, force=False, forceIfNotActive=False):
        if not hasattr(self, 'currentLab2D'):
            return

        isOverlaySegmLeftActive = self.isOverlaySegmMaskChecked(0)
        isOverlaySegmRightActive = (
            1 in self.annotatedAxesIndices() 
            and self.isOverlaySegmMaskChecked(1)
        )

        isOverlaySegmActive = (
            isOverlaySegmLeftActive or isOverlaySegmRightActive
            or force
        )
        if not isOverlaySegmActive and not forceIfNotActive:
            return 

        alpha = self.imgGrad.labelsAlphaSlider.value()
        if alpha == 0:
            return

        posData = self.data[self.pos_i]
        maxID = max(posData.IDs, default=0)

        if maxID >= len(self.lut):
            self.extendLabelsLUT(maxID+10)

        currentLab2D = self.currentLab2D
        if (
            self.isSegm3D
            and self.zProjComboBox.currentText() != 'single z-slice'
            and getattr(posData, 'rp', None) is not None
            and posData.rp.is3D
        ):
            slicing = self.switchPlaneCombobox.depthAxes()
            currentLab2D = posData.rp.get_projection_lab_sorted(slicing=slicing)
            self.currentLab2D = currentLab2D

        if isOverlaySegmLeftActive:
            self.labelsLayerImg1.setImage(currentLab2D, autoLevels=False)

        if isOverlaySegmRightActive: 
            self.labelsLayerRightImg.setImage(currentLab2D, autoLevels=False)

    def isObjInCurrSlice(self, obj_bbox, debug=False, z_slice=None):
        if z_slice is None:
            z_slice = self.z_lab()
            
        if self.isSegm3D:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            if not isZslice:
                # required a projection --> all obj are visible
                return True
            
            depthAxes = self.switchPlaneCombobox.depthAxes()
            
            min_z, min_y, min_x, max_z, max_y, max_x = obj_bbox
            if depthAxes == 'z':
                min_val, max_val = min_z, max_z
                val = z_slice
            elif depthAxes == 'y':
                min_val, max_val = min_y, max_y
                val = z_slice[-1]
            else:
                min_val, max_val = min_x, max_x
                val = z_slice[-1]
            
            if val >= min_val and val < max_val:
                return True
            else:
                return False
        else:
            return True

    def z_lab(self, checkIfProj=False):
        if checkIfProj and self.zProjComboBox.currentText() != 'single z-slice':
            return
        
        if not self.isSegm3D:
            return 
        
        posData = self.data[self.pos_i]

        idx = self.zSliceScrollBar.sliderPosition()
        
        # ensure idx doesnt exceed the number of z-slices of the position
        idx_z = min(idx, posData.SizeZ-1)
         
        if not self.switchPlaneCombobox.isEnabled():
            return idx_z
        
        depthAxes = self.switchPlaneCombobox.depthAxes()
        if depthAxes == 'z':
            return idx_z
        elif depthAxes == 'y':
            idx_y = min(idx, posData.SizeY-1)
            return (slice(None), idx_y)
        else:
            idx_x = min(idx, posData.SizeX-1)
            return (slice(None), slice(None), idx_x)

    def get_2Dlab(self, lab=None, force_z=True, slice_i=None):
        if lab is None:
            posData = self.data[self.pos_i]
            lab = posData.lab
        if self.isSegm3D:
            if force_z:
                return lab[self.z_lab() if slice_i is None else slice_i]
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            if isZslice:
                return lab[self.z_lab() if slice_i is None else slice_i]
            else:
                if self.switchPlaneCombobox.isEnabled():
                    slicing = self.switchPlaneCombobox.depthAxes()
                else:
                    slicing = 'z'

                posData = self.data[self.pos_i]
                rp = getattr(posData, 'rp', None)
                if rp is not None and rp.is3D:
                    return rp.get_projection_lab_sorted(slicing=slicing)

                rp = self._acdcRegionProps(lab, precache_centroids=False)
                return rp.get_projection_lab_sorted(slicing=slicing)
        else:
            return lab

    def getObjCentroid(self, obj_centroid):
        if self.isSegm3D:
            depthAxes = self.switchPlaneCombobox.depthAxes()
            zc, yc, xc = obj_centroid
            if depthAxes == 'z':
                return yc, xc 
            elif depthAxes == 'y':
                return zc, xc 
            else:
                return zc, yc 
        else:
            return obj_centroid

    def drawAllMothBudLines(self):
        posData = self.data[self.pos_i]
        for obj in posData.rp:
            for ax in self.annotatedAxesIndices():
                self.drawObjMothBudLines(obj, posData, ax=ax)

    def drawObjMothBudLines(self, obj, posData, ax=0):
        areMothBudLinesRequested = self.areMothBudLinesRequested(ax)
        if not areMothBudLinesRequested:
            return
        
        if posData.cca_df is None:
            return 

        mode = str(self.modeComboBox.currentText())
        if mode == 'Normal division: Lineage Tree':
            return

        ID = obj.label
        try:
            cca_df_ID = posData.cca_df.loc[ID]
        except KeyError:
            return        
        
        if not self.isObjInCurrSlice(obj.bbox):
            return
        
        ccs_ID = cca_df_ID['cell_cycle_stage']
        if ccs_ID == 'G1':
            return

        relationship = cca_df_ID['relationship']
        if relationship != 'bud':
            return

        emerg_frame_i = cca_df_ID['emerg_frame_i']
        isNew = emerg_frame_i == posData.frame_i
        scatterItem = self.getMothBudLineScatterItem(ax, isNew)
        relative_ID = cca_df_ID['relative_ID']

        relative_ID_obj = posData.rp.get_obj_from_ID(relative_ID)
        obj_centroid = posData.rp.get_centroid(ID)
        rel_obj_centroid = posData.rp.get_centroid(relative_ID)
        y1, x1 = self.getObjCentroid(obj_centroid)
        y2, x2 = self.getObjCentroid(rel_obj_centroid)
        xx, yy = core.get_line(y1, x1, y2, x2, dashed=True)
        scatterItem.addPoints(xx, yy)

    def drawAllLineageTreeLines(self):
        """
        Draw all lineage tree lines on the GUI.

        This method retrieves the lineage tree data and draws the lineage tree lines
        connecting cells and their respective mothers when the mother has split.
        """
        if self.lineage_tree is None:
            return

        if len(self.lineage_tree.frames_for_dfs) < 2:
            return

        self.clearAllCellToCellLines()
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i

        if frame_i == 0:
            return 

        lin_tree_df = posData.allData_li[frame_i]['acdc_df']
        lin_tree_df_prev = posData.allData_li[frame_i-1]['acdc_df']
        rp = posData.rp
        prev_rp = posData.allData_li[frame_i-1]['regionprops']

        self.setTitleText()

        new_cells = lin_tree_df.index.difference(lin_tree_df_prev.index) # I could use this for the if already but this is probably faster for frames where nothing changes
        if new_cells.shape[0] == 0:
            return
        
        for ax in (0, 1):
            if not self.areMothBudLinesRequested(ax):
                continue

            for ID in new_cells:
                curr_obj = rp.get_obj_from_ID(ID)
                lin_tree_df_ID = lin_tree_df.loc[ID]

                # lin_tree_df_mother_ID = lin_tree_df_prev.loc[lin_tree_df_ID["parent_ID_tree"]]
                if lin_tree_df_ID['parent_ID_tree'] == -1: # make sure that new obj where the parents are not known get skipped
                    continue
                
                mother_obj = prev_rp.get_obj_from_ID(lin_tree_df_ID['parent_ID_tree'])

                emerg_frame_i = lin_tree_df_ID['emerg_frame_i']
                isNew = False

                self.drawObjLin_TreeMothBudLines(ax, curr_obj, mother_obj, isNew, ID=ID)

    def drawObjLin_TreeMothBudLines(self, ax, obj, mother_obj, isNew, ID=None):
        """
        Draw moth-bud lines between an object and its mother object.

        Parameters
        ----------
        ax : cellacdc.widgets.MainPlotItem
            The Cell-ACDC GUI axes object to draw on.
        obj : Object
            The object for which to draw the moth-bud lines.
        mother_obj : Object
            The mother object to connect with.
        isNew : bool
            Indicates whether the object is new or not.
        ID : int, optional
            The ID of the object, by default None.
        """
        if not self.areMothBudLinesRequested(ax):
            return

        if not ID:
            ID = obj.label
                
        if not self.isObjInCurrSlice(obj.bbox):
            return

        scatterItem = self.getMothBudLineScatterItem(ax, isNew)
        
        posData = self.data[self.pos_i]
        prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
        rp = posData.rp
        if ID is None:
            ID = obj.label
        ID_mother = mother_obj.label
        y1, x1 = self.getObjCentroid(rp.get_centroid(ID))
        y2, x2 = self.getObjCentroid(prev_rp.get_centroid(ID_mother))
        xx, yy = core.get_line(y1, x1, y2, x2, dashed=True)
        scatterItem.addPoints(xx, yy)

    def clearAllCellToCellLines(self):
        for ax in self.annotatedAxesIndices():
            self.getMothBudLineScatterItem(ax, True).setData([], [])
            self.getMothBudLineScatterItem(ax, False).setData([], [])

    def updateViewBox(self, ax: int):
        if ax == 0:
            axesItem = self.ax1
        else:
            axesItem = self.ax2
        axesItem.getViewBox().update()

    def showAllContoursToggled(self):
        if not self.isDataLoaded:
            return

        self.updateAllImages()

    def annotLostObjsToggled(self, checked):
        if not self.isDataLoaded:
            return
        self.updateAllImages()

    def initannotateObjTrackSettings(self):
        if hasattr(self, 'annotateObjTrackSettings'):
            return

        default_settings = {
            'length': 5,
            'against_prev': False,
            'color': (255, 100, 0),
            'n_fade': 15,
            'max_width': 4,
            'min_width': 1,
            'max_alpha_perc': 100,
            'min_alpha_perc': 10,
            'against_prev_line_color': (255, 165, 0),
            'against_prev_line_width': 2,
            'against_prev_contour_color': (245, 184, 0),
            'against_prev_contour_width': 2,
            '3D_show_only_visible': False
        }

        def _parse_color_setting(value, default):
            if value is None:
                return tuple(default)
            if isinstance(value, str):
                parts = value.lstrip('(').rstrip(')').split(',')
                parts = [int(p.strip()) for p in parts if p.strip()]
                if len(parts) >= 3:
                    return tuple(parts[:3])
                return tuple(default)

            try:
                return tuple(int(v) for v in value[:3])
            except Exception:
                return tuple(default)

        if (self.df_settings is not None
            and 'annotateObjTrackLength' in self.df_settings.index):
            self.annotateObjTrackSettings = {
                'length': self._get_setting_value(
                    'annotateObjTrackLength', default_settings['length']
                ),
                'against_prev': self._get_setting_value(
                    'annotateObjTrackAgainstPrev',
                    default_settings['against_prev'],
                    cast=bool
                ),
                'n_fade': self._get_setting_value(
                    'annotateObjTrackNFade', default_settings['n_fade']
                ),
                'max_width': self._get_setting_value(
                    'annotateObjTrackMaxWidth', default_settings['max_width']
                ),
                'min_width': self._get_setting_value(
                    'annotateObjTrackMinWidth', default_settings['min_width']
                ),
                'max_alpha_perc': self._get_setting_value(
                    'annotateObjTrackMaxAlphaPerc',
                    default_settings['max_alpha_perc']
                ),
                'min_alpha_perc': self._get_setting_value(
                    'annotateObjTrackMinAlphaPerc',
                    default_settings['min_alpha_perc']
                ),
                'against_prev_line_width': self._get_setting_value(
                    'annotateObjTrackAgainstPrevLineWidth',
                    default_settings['against_prev_line_width']
                ),
                'against_prev_contour_width': self._get_setting_value(
                    'annotateObjTrackAgainstPrevContourWidth',
                    default_settings['against_prev_contour_width']
                ),
            }
            color = self.df_settings.at['annotateObjTrackColor', 'value']
            self.annotateObjTrackSettings['color'] = _parse_color_setting(
                color, default=default_settings['color']
            )
            against_prev_line_color = self._get_setting_value(
                'annotateObjTrackAgainstPrevLineColor',
                default_settings['against_prev_line_color'],
                cast=str
            )
            self.annotateObjTrackSettings['against_prev_line_color'] = (
                _parse_color_setting(
                    against_prev_line_color,
                    default=default_settings['against_prev_line_color']
                )
            )

            against_prev_contour_color = self._get_setting_value(
                'annotateObjTrackAgainstPrevContourColor',
                default_settings['against_prev_contour_color'],
                cast=str
            )
            self.annotateObjTrackSettings['against_prev_contour_color'] = (
                _parse_color_setting(
                    against_prev_contour_color,
                    default=default_settings['against_prev_contour_color']
                )
            )
            self.annotateObjTrackSettings['3D_show_only_visible'] = (
                self._get_setting_value(
                    'annotateObjTrack3DShowOnlyVisible',
                    default_settings['3D_show_only_visible'],
                    cast=bool
                )
            )
            return

        self.annotateObjTrackSettings = dict(default_settings)

    def setAnnotateObjTrackSettings(self, *args):
        self.initannotateObjTrackSettings()
        win = apps.AnnotateObjTrackSettingsDialog(
            settings=self.annotateObjTrackSettings,
            parent=self,
        )
        win.sigValuesChanged.connect(
            self.onSetAnnotateObjTrackSettingsSigValuesChanged
            )
        win.show(block=True)
        if win.cancel:
            self.annotateAllObjectTracks() # restore original
            return

        self.annotateObjTrackSettings = win.settings
        self.annotateAllObjectTracks()
        
        self.df_settings.at[
            'annotateObjTrackLength', 'value'
            ] = self.annotateObjTrackSettings['length']
        self.df_settings.at[
            'annotateObjTrackAgainstPrev', 'value'
            ] = self.annotateObjTrackSettings['against_prev']
        self.df_settings.at[
            'annotateObjTrackColor', 'value'
            ] = str(self.annotateObjTrackSettings['color'])
        self.df_settings.at[
            'annotateObjTrackNFade', 'value'
            ] = self.annotateObjTrackSettings['n_fade']
        self.df_settings.at[
            'annotateObjTrackMaxWidth', 'value'
            ] = self.annotateObjTrackSettings['max_width']
        self.df_settings.at[
            'annotateObjTrackMinWidth', 'value'
            ] = self.annotateObjTrackSettings['min_width']
        self.df_settings.at[
            'annotateObjTrackMaxAlphaPerc', 'value'
            ] = self.annotateObjTrackSettings['max_alpha_perc']
        self.df_settings.at[
            'annotateObjTrackMinAlphaPerc', 'value'
            ] = self.annotateObjTrackSettings['min_alpha_perc']
        self.df_settings.loc[
            'annotateObjTrackAgainstPrevLineColor', 'value'
            ] = str(self.annotateObjTrackSettings['against_prev_line_color'])
        self.df_settings.loc[
            'annotateObjTrackAgainstPrevLineWidth', 'value'
            ] = self.annotateObjTrackSettings['against_prev_line_width']
        self.df_settings.loc[
            'annotateObjTrackAgainstPrevContourColor', 'value'
            ] = str(self.annotateObjTrackSettings['against_prev_contour_color'])
        self.df_settings.loc[
            'annotateObjTrackAgainstPrevContourWidth', 'value'
            ] = self.annotateObjTrackSettings['against_prev_contour_width']
        self.df_settings.loc[
            'annotateObjTrack3DShowOnlyVisible', 'value'
            ] = self.annotateObjTrackSettings['3D_show_only_visible']
        self.df_settings.to_csv(self.settings_csv_path)

    def onSetAnnotateObjTrackSettingsSigValuesChanged(self, settings):
        self.annotateAllObjectTracks(settings=settings)

    def getMovementAgainstPrevLinesItem(self, ax: int):
        itemName = f'ax{ax+1}_movementAgainstPrevLinesItem'
        return getattr(self, itemName)

    def getMovementAgainstPrevScatterItem(self, ax: int):
        itemName = f'ax{ax+1}_movementAgainstPrevScatterItem'
        return getattr(self, itemName)

    def _setAgainstPrevAppearance(self, ax: int, settings: dict):
        line_item = self.getMovementAgainstPrevLinesItem(ax)
        contour_item = self.getMovementAgainstPrevScatterItem(ax)

        line_color = tuple(settings['against_prev_line_color'])
        contour_color = tuple(settings['against_prev_contour_color'])
        line_width = max(1, int(settings['against_prev_line_width']))
        contour_width = max(1, int(settings['against_prev_contour_width']))

        appearance_key = (
            line_color,
            line_width,
            contour_color,
            contour_width,
        )
        cache = getattr(self, '_againstPrevAppearanceCache', {})
        if cache.get(ax) == appearance_key:
            return

        line_item.setBrush(pg.mkBrush((*line_color, 255)))
        line_item.setSize(line_width)

        contour_item.setBrush(pg.mkBrush((*contour_color, 150)))
        contour_item.setPen(pg.mkPen(contour_color, width=1))
        contour_item.setSize(contour_width)
        cache[ax] = appearance_key
        self._againstPrevAppearanceCache = cache

    def annotateAllObjectTracks(self, settings=None):
        if settings is None:
            self.initannotateObjTrackSettings()
            settings = self.annotateObjTrackSettings

        to_annotate_ax = []
        for ax in self.annotatedAxesIndices():
            movementAgainstPrevLinesItem = self.getMovementAgainstPrevLinesItem(ax)
            movementAgainstPrevScatterItem = (
                self.getMovementAgainstPrevScatterItem(ax)
            )
            movementAgainstPrevLinesItem.setData([], [])
            movementAgainstPrevScatterItem.clear()

            if not self.isObjectTracksChecked(ax):
                continue
            to_annotate_ax.append(ax)
            
        self.clearObjTracks()
        if not to_annotate_ax:
            return

        if settings['against_prev'] is True:
            for ax in to_annotate_ax:
                self.annotateObjTrackAgainstPrev(ax, settings=settings)
            return
                
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        
        length_requested = settings['length']
        length = min(length_requested, frame_i)
        centroids = dict()
        rp = posData.rp
        for ID in rp.IDs:
            obj = rp.get_obj_from_ID(ID)
            if settings['3D_show_only_visible'] and self.isSegm3D:
                if not self.isObjInCurrSlice(obj.bbox):
                    continue
            centroids[ID] = {frame_i: self.getObjCentroid(
                rp.get_centroid(ID, as_ints=True, exact=True)
                )}
            
        for idx in range(frame_i-1, frame_i-length-1, -1):
            _rp = posData.allData_li[idx]['regionprops']
            if _rp is None:
                lab = posData.allData_li[idx]['labels']
                if lab is None:
                    lab = posData.segm_data[idx]
                _rp = self._acdcRegionProps(lab)
                posData.allData_li[idx]['regionprops'] = _rp
                
            for ID, centroid_dict in centroids.items():
                obj = _rp.get_obj_from_ID(ID, warn=False)
                if obj is None:
                    continue
                
                centroid_dict[idx] = self.getObjCentroid(
                    _rp.get_centroid(ID, as_ints=True, exact=True)
                )                    
        
        color = settings['color']
        max_width = settings['max_width']
        min_width = settings['min_width']
        max_alpha = (settings['max_alpha_perc'] / 100) * 255
        min_alpha= (settings['min_alpha_perc'] / 100) * 255
        n_fade = settings['n_fade']

        tracks = []
        for track in centroids.values():
            frames = sorted(track.keys())
            points = [(track[f][1], track[f][0]) for f in frames]
            tracks.append((points, frames))
            
            
        for ax in to_annotate_ax:
            if ax in self.tracksPlotItems:
                tracksPlotItem = self.tracksPlotItems[ax]
                tracksPlotItem.setAppearance(
                    color=color, n_fade=n_fade, max_width=max_width,
                    min_width=min_width, max_alpha=max_alpha, min_alpha=min_alpha
                )
            else:
                tracksPlotItem = annotate.FadingTracksItem(
                    color=color, n_fade=n_fade, max_width=max_width,
                    min_width=min_width, max_alpha=max_alpha, min_alpha=min_alpha
                )
                self.tracksPlotItems[ax] = tracksPlotItem
                if ax == 0:
                    self.ax1.addItem(tracksPlotItem)
                else:
                    self.ax2.addItem(tracksPlotItem)
            
            tracksPlotItem.setTracks(tracks)
            tracksPlotItem.show()

    def clearObjTracks(self):
        for tracksPlotItem in self.tracksPlotItems.values():
            # tracksPlotItem.clear()
            tracksPlotItem.hide()

    def annotateObjTrackAgainstPrev(self, ax: int, settings: dict=None):
        self.initannotateObjTrackSettings()
        settings = settings or self.annotateObjTrackSettings
        self._setAgainstPrevAppearance(ax, settings)

        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        
        if frame_i == 0:
            return        

        movementAgainstPrevScatterItem = (
            self.getMovementAgainstPrevScatterItem(ax)
        )
        movementAgainstPrevLinesItem = self.getMovementAgainstPrevLinesItem(ax)
        rp_2D = self.get2DRP()
        prev_rp_2D = self.get2DRP(frame_i=frame_i-1)
        rp_3D = posData.rp
        if rp_2D is None or prev_rp_2D is None:
            self.logger.warning(
                f"[WARNING] rp or prev rp could not be retrieved"
            )
            return

        for ID in rp_2D.IDs:
            obj = rp_2D.get_obj_from_ID(ID)    
            obj_prev = prev_rp_2D.get_obj_from_ID(ID, warn=False)
            if obj_prev is None:
                continue

            allContours = self.getObjContours(
                obj_prev,
                all_external=True,
                include_internal=self.showAllContoursToggle.isChecked()
            )
            obj_3D = rp_3D.get_obj_from_ID(ID, warn=False)
            for objContours in allContours:
                if not self.isObjInCurrSlice(obj_3D.bbox):
                    continue
                xx = objContours[:,0] + 0.5
                yy = objContours[:,1] + 0.5
                movementAgainstPrevScatterItem.addPoints(xx, yy)
                
            y1, x1 = obj_prev.centroid
            y2, x2 = obj.centroid
            xx, yy = core.get_line(y1, x1, y2, x2, dashed=False)
            movementAgainstPrevLinesItem.addPoints(xx, yy)

    def getPrevFrameIDs(self, current_frame_i=None):
        posData = self.data[self.pos_i]
        if current_frame_i is None:
            current_frame_i = posData.frame_i
        
        if current_frame_i is None:
            return []
        
        if current_frame_i == 0:
            return []
        
        prev_frame_i = current_frame_i - 1
        if posData.allData_li[prev_frame_i]['regionprops'] is not None:
            prevIDs = posData.allData_li[prev_frame_i]['regionprops'].IDs
            return prevIDs
        
        # IDs in previous frame were not stored --> load prev lab from HDD
        prev_lab = self.get_labels_array(
            from_store=False, 
            frame_i=prev_frame_i,
            return_copy=False
        )
        rp = self._acdcRegionProps(
            prev_lab, precache_centroids=False
        )
        posData.allData_li[prev_frame_i]['regionprops'] = rp
        prevIDs = rp.IDs
        return prevIDs

    # @exec_time
    def setLostNewOldPrevIDs(self):
        posData = self.data[self.pos_i]
        if posData.frame_i == 0:
            posData.lost_IDs = []
            posData.new_IDs = []
            posData.old_IDs = []
            # posData.multiContIDs = set()
            self.setLookingGoodTitle()
            return []
        
        # elif self.modeComboBox.currentText() == 'Viewer':
        #     pass
        
        out = self.updateLostNewCurrentIDs()
        lost_IDs, new_IDs, IDs_with_holes, tracked_lost_IDs = (
            out
        )
        self.setTitleText(
            lost_IDs, new_IDs, IDs_with_holes, tracked_lost_IDs
        )
        return

    def updateLostNewCurrentIDs(self):
        posData = self.data[self.pos_i]
        
        prev_IDs = self.getPrevFrameIDs()  
        tracked_lost_IDs = self.getTrackedLostIDs()
        curr_IDs = posData.IDs
        lost_IDs = [
            ID for ID in prev_IDs if ID not in curr_IDs
            and ID not in tracked_lost_IDs
        ]
        new_IDs = [
            ID for ID in curr_IDs if ID not in prev_IDs
        ]
        IDs_with_holes = []
        posData.lost_IDs = lost_IDs
        posData.new_IDs = new_IDs
        posData.old_IDs = prev_IDs
        posData.IDs = curr_IDs
        
        out = (
            lost_IDs, new_IDs, IDs_with_holes, tracked_lost_IDs
        )
        return out

    def getTrackedLostIDs(self, prev_lab=None, IDs_in_frames=None, frame_i=None):
        trackedLostIDs = set()
        posData = self.data[self.pos_i]
        if self.isExportingVideo:
            posData.trackedLostIDs = trackedLostIDs
            return trackedLostIDs
        
        retrackedLostcent = set()
        if frame_i is None:
            frame_i = posData.frame_i
            
        if prev_lab is None:
            prev_lab = self.get_labels_array(
                from_store=True, 
                frame_i=posData.frame_i-1, 
                return_existing=False,
                return_copy=False
            )

        if IDs_in_frames is None:
            IDs_in_frames = posData.IDs

        try:
            tracked_lost_centroids = posData.tracked_lost_centroids[frame_i]
        except KeyError or TypeError:
            tracked_lost_centroids = set()

        for centroid in tracked_lost_centroids:
            if len(centroid) < 3 and prev_lab.ndim == 3:
                # Ignore wrongly stored centroids
                continue

            ID = prev_lab[centroid]
            if ID == 0:
                continue

            if ID in IDs_in_frames:
                retrackedLostcent.add(centroid)
                continue

            trackedLostIDs.add(ID)

        posData.tracked_lost_centroids[frame_i] = (
            tracked_lost_centroids - retrackedLostcent
        )
        posData.trackedLostIDs = trackedLostIDs

        return trackedLostIDs
