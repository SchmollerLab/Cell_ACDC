"""Histogram LUT items and colormap widgets: baseHistogramLUTitem and
myHistogramLUTitem (image and overlay LUTs with the colormap menu, see
the LUT spec in gui/mixins/navigation.py), the labels colormap widgets
(BaseLabelsGradientWidget, labelsGradientWidget), the gradient editor
items that swap in the colormap registries of style.py
(BaseGradientEditorItemImage, BaseGradientEditorItemLabels), the custom
colormap menu action (CustomGradientMenuAction), FilterObject and
highlightableQWidgetAction.
"""
import os
import operator
import re
import numpy as np
import traceback
from qtpy.QtCore import Signal, Qt, QRect, QPointF, QRect, QObject
from qtpy.QtGui import (
    QColor, QBrush, QPainter, QPixmap, QLinearGradient, QLinearGradient
)
from qtpy.QtWidgets import (
    QLabel, QHBoxLayout, QWidget, QWidgetAction, QAction, QMenu, QActionGroup
)
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import _window_stacking
from cellacdc import is_mac
from cellacdc import printl
from cellacdc import colors, config
from cellacdc import apps
from cellacdc import QtScoped
from . import push_buttons as widgets_push_buttons
from . import style as widgets_style
from . import combo_boxes as widgets_combo_boxes
from . import numeric_inputs as widgets_numeric_inputs


def CustomGradientMenuAction(gradient: QLinearGradient, name: str, parent):
    pixmap = QPixmap(100, 15)
    painter = QPainter(pixmap)
    brush = QBrush(gradient)
    painter.fillRect(QRect(0, 0, 100, 15), brush)
    painter.end()
    label = QLabel()
    label.setPixmap(pixmap)
    label.setContentsMargins(1, 1, 1, 1)
    labelName = QLabel(name)
    hbox = QHBoxLayout()
    delButton = widgets_push_buttons.delPushButton()
    hbox.addWidget(labelName)
    hbox.addStretch(1)
    hbox.addWidget(label)
    hbox.addWidget(delButton)
    widget = QWidget()
    widget.setLayout(hbox)
    action = QWidgetAction(parent)
    action.name = name
    action.setDefaultWidget(widget)
    action.delButton = delButton
    delButton.action = action
    return action


class FilterObject(QObject):
    sigFilteredEvent = Signal(object, object)

    def __init__(self) -> None:
        super().__init__()
    
    def eventFilter(self, object, event):
        self.sigFilteredEvent.emit(object, event)
        return super().eventFilter(object, event)


class BaseGradientEditorItemImage(pg.GradientEditorItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
    
    def restoreState(self, state):
        pg.graphicsItems.GradientEditorItem.Gradients = widgets_style.GradientsImage
        return super().restoreState(state)


class BaseGradientEditorItemLabels(pg.GradientEditorItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
    
    def restoreState(self, state):
        pg.graphicsItems.GradientEditorItem.Gradients = widgets_style.GradientsLabels
        return super().restoreState(state)

class baseHistogramLUTitem(pg.HistogramLUTItem):
    sigAddColormap = Signal(object, str)
    sigRescaleIntes = Signal(object)
    sigGradientChanged = Signal(object)

    def __init__(self, name='image', axisLabel='', parent=None, **kwargs):
        pg.GradientEditorItem = BaseGradientEditorItemLabels

        super().__init__(**kwargs)

        self._defaultState = super().saveState()
        self.labelStyle = {'color': '#ffffff', 'font-size': '11px'}

        if axisLabel:
            self.setAxisLabel(axisLabel)

        self.cmaps = widgets_style.cmaps
        self._parent = parent
        self.name = name

        _window_stacking.set_window_flags_on_top(
            self.gradient.colorDialog, Qt.Dialog, parent=parent
        )
        self.gradient.colorDialog.accepted.disconnect()
        self.gradient.colorDialog.accepted.connect(self.tickColorAccepted)

        self.isInverted = False
        self.lastGradientName = 'grey'
        self.lastGradient = widgets_style.Gradients['grey']

        # Colormaps picked in the LUT menu (pyqtgraph `ColorMapMenu`, also
        # in pyqtgraph 0.13.7) go through `colorMapMenuClicked`
        colorMapTriggered = getattr(
            self.gradient.menu, 'sigColorMapTriggered', None
        )
        if colorMapTriggered is not None:
            try:
                colorMapTriggered.disconnect(self.gradient.colorMapMenuClicked)
            except (TypeError, RuntimeError):
                # Not connected: keep pyqtgraph's own handling
                pass
            else:
                colorMapTriggered.connect(self.colorMapMenuClicked)

        for action in self.gradient.menu.actions():
            if action.text() == 'HSV':
                HSV_action = action
            elif action.text() == 'RGB':
                RGB_ation = action
        self.gradient.menu.removeAction(HSV_action)
        self.gradient.menu.removeAction(RGB_ation)
        
        # Rescale intensities (LUT)
        rescaleIntensMenu = self.gradient.menu.addMenu(
            'Rescale intensities (LUT)'
        )
        rescaleActionGroup = QActionGroup(self)
        rescaleActionGroup.setExclusive(True)
        
        self.rescaleEach2DimgAction = QAction(
            'Rescale each 2D image', rescaleIntensMenu
        )
        self.rescaleEach2DimgAction.setCheckable(True)
        self.rescaleEach2DimgAction.setChecked(True)
        rescaleActionGroup.addAction(self.rescaleEach2DimgAction)
        rescaleIntensMenu.addAction(self.rescaleEach2DimgAction)
        
        self.rescaleAcrossZstackAction = QAction(
            'Rescale across z-stack', rescaleIntensMenu
        )
        self.rescaleAcrossZstackAction.setCheckable(True)
        self.rescaleAcrossZstackAction.setChecked(False)
        rescaleActionGroup.addAction(self.rescaleAcrossZstackAction)
        rescaleIntensMenu.addAction(self.rescaleAcrossZstackAction)
        
        self.rescaleAcrossTimeAction = QAction(
            'Rescale across time frames', rescaleIntensMenu
        )
        self.rescaleAcrossTimeAction.setCheckable(True)
        self.rescaleAcrossTimeAction.setChecked(False)
        rescaleActionGroup.addAction(self.rescaleAcrossTimeAction)
        rescaleIntensMenu.addAction(self.rescaleAcrossTimeAction)
        
        self.customRescaleAction = QAction(
            'Choose custom levels...', rescaleIntensMenu
        )
        self.customRescaleAction.setCheckable(True)
        rescaleActionGroup.addAction(self.customRescaleAction)
        rescaleIntensMenu.addAction(self.customRescaleAction)
        
        self.doNotRescaleAction = QAction(
            'Do no rescale, display raw image', rescaleIntensMenu
        )
        self.doNotRescaleAction.setCheckable(True)
        rescaleActionGroup.addAction(self.doNotRescaleAction)
        rescaleIntensMenu.addAction(self.doNotRescaleAction)
        
        self.rescaleActionGroup = rescaleActionGroup
        rescaleActionGroup.triggered.connect(self.rescaleActionTriggered)

        # Add custom colormap action
        self.customCmapsMenu = self.gradient.menu.addMenu('Custom colormaps')
        self.customCmapsMenu.aboutToShow.connect(self.onShowCustomCmapsMenu)
        self.customCmapsMenu.triggered.connect(self.customCmapsMenuTriggered)
        
        self.saveColormapAction = QAction(
            'Save current colormap...', self
        )
        self.gradient.menu.addAction(self.saveColormapAction)
        self.saveColormapAction.triggered.connect(
            self.saveColormap
        )

        self.addCustomGradients()

        # Set inverted gradients for invert bw action
        self.addInvertedColorMaps()

        self.gradient.menu.addSeparator()

        # hide histogram tool
        self.vb.hide()
        
        # Disable moving the axis up and down
        self.axis.unlinkFromView()

        # Disable histogram default context Menu event
        self.vb.raiseContextMenu = lambda x: None
    
    def rescaleActionTriggered(self, action):
        self.sigRescaleIntes.emit(action)
    
    def onShowCustomCmapsMenu(self):
        self.customCmapsMenu.show()
        
    def gradientChanged(self):
        super().gradientChanged()
        self.sigGradientChanged.emit(self)
    
    def customCmapsMenuTriggered(self, action):
        cmap = action.cmap
        self.gradient.colorMapMenuClicked(cmap)
        self.gradient.showTicks(True)
        self.allowAddingShownTicks()

    def colorMapMenuClicked(self, cmap):
        """Colormap `cmap` picked in the LUT menu
        (`ColorMapMenu.sigColorMapTriggered`, connected instead of the
        gradient's own `colorMapMenuClicked`). For colormaps that are not
        preset gradients pyqtgraph hides the ticks (one per colormap
        position) only after `setColorMap` emitted
        `sigGradientChangeFinished`: emit it again once they are hidden, so
        that the slots saving the LUT save hidden ticks."""
        self.gradient.colorMapMenuClicked(cmap)
        if str(cmap.name).startswith('preset-gradient:'):
            self.allowAddingShownTicks()
            return
        self.gradient.sigGradientChangeFinished.emit(self.gradient)

    def allowAddingShownTicks(self):
        """Allow adding ticks (clicking the gradient) when its ticks are
        shown. pyqtgraph's `showTicks(False)` backs up `allowAdd` once per
        tick (i.e. False from the second tick on), so its `showTicks(True)`
        never allows adding ticks again after a colormap with hidden
        ticks."""
        ticks = list(self.gradient.ticks)
        if ticks and ticks[0].isVisibleTo(self.gradient):
            self.gradient.allowAdd = True
    
    def setAxisLabel(self, text):
        self.labelText = text
        self.axis.setLabel(text, **self.labelStyle)
    
    def updateAxisLabel(self):
        text = self.axis.label.toPlainText()
        if not text:
            return
        self.setAxisLabel(text)
    
    def setGradient(self, gradient):
        self.gradient.restoreState(gradient)
        self.lastGradient = gradient
        self.allowAddingShownTicks()
    
    def colormapClicked(self, checked=False, name=None):
        name = self.sender().name
        self.lastGradientName = name
        if self.isInverted:
            self.setGradient(self.invertedGradients[name])
        else:
            self.setGradient(widgets_style.Gradients[name])

    def sortTicks(self, ticks):
        sortedTicks = sorted(ticks, key=operator.itemgetter(0))
        return sortedTicks
    
    def resetState(self):
        self.restoreState(self._defaultState)
    
    def getInvertedGradients(self):
        invertedGradients = {}
        for name, gradient in widgets_style.Gradients.items():
            ticks = gradient['ticks']
            sortedTicks = self.sortTicks(ticks)
            if name in widgets_style.nonInvertibleCmaps:
                invertedColors = sortedTicks
            else:
                invertedColors = [
                    (t[0], ti[1]) 
                    for t, ti in zip(sortedTicks, sortedTicks[::-1])
                ]
            invertedGradient = {}
            invertedGradient['ticks'] = invertedColors
            invertedGradient['mode'] = gradient['mode']
            invertedGradients[name] = invertedGradient
        return invertedGradients
    
    def addInvertedColorMaps(self):
        self.invertedGradients = self.getInvertedGradients()
        for action in self.gradient.menu.actions():
            if not hasattr(action, 'name'):
                continue
            
            if action.name not in self.cmaps:
                continue
            
            action.triggered.disconnect()
            action.triggered.connect(self.colormapClicked)

            px = QPixmap(100, 15)
            p = QPainter(px)
            invertedGradient = self.invertedGradients[action.name]
            qtGradient = QLinearGradient(QPointF(0,0), QPointF(100,0))
            ticks = self.sortTicks(invertedGradient['ticks'])
            qtGradient.setStops([(x, QColor(*color)) for x,color in ticks])
            brush = QBrush(qtGradient)
            p.fillRect(QRect(0, 0, 100, 15), brush)
            p.end()
            widget = action.defaultWidget()
            hbox = widget.layout()
            rectLabelWidget = QLabel()
            rectLabelWidget.setPixmap(px)
            hbox.addWidget(rectLabelWidget)
            rectLabelWidget.hide()
    
    def setInvertedColorMaps(self, inverted):
        if inverted:
            showIdx = 2
            hideIdx = 1
            self.labelStyle['color'] = '#000000'
        else:
            showIdx = 1
            hideIdx = 2
            self.labelStyle['color'] = '#ffffff'
        
        for action in self.gradient.menu.actions():
            if not hasattr(action, 'name'):
                continue
            
            if action.name not in self.cmaps:
                continue

            widget = action.defaultWidget()
            hbox = widget.layout()
            hideCmapRect = hbox.itemAt(hideIdx).widget()
            showCmapRect = hbox.itemAt(showIdx).widget()
            hideCmapRect.hide()
            showCmapRect.show()
        
        self.updateAxisLabel()
        self.isInverted = inverted
    
    def invertGradient(self, gradient):
        ticks = gradient['ticks']
        sortedTicks = self.sortTicks(ticks)
        invertedColors = [
            (t[0], ti[1]) 
            for t, ti in zip(sortedTicks, sortedTicks[::-1])
        ]
        invertedGradient = {}
        invertedGradient['ticks'] = invertedColors
        invertedGradient['mode'] = gradient['mode']
        if 'ticksVisible' in gradient:
            # E.g., colormaps with hidden ticks (`colorMapMenuClicked`)
            invertedGradient['ticksVisible'] = gradient['ticksVisible']
        return invertedGradient
    
    def invertCurrentColormap(self, inverted, debug=False):
        self.setGradient(self.invertGradient(self.lastGradient))
    
    def addCustomGradient(self, gradient_name, gradient_ticks, restore=True):
        self.originalLength = self.gradient.length
        self.gradient.length = 100
        if restore:
            self.gradient.restoreState(gradient_ticks)
        gradient = self.gradient.getGradient()
        action = CustomGradientMenuAction(gradient, gradient_name, self.gradient)
        # action.triggered.connect(self.gradient.contextMenuClicked)
        action.delButton.clicked.connect(self.removeCustomGradient)
        action.cmap = colors.pg_ticks_to_colormap(gradient_ticks['ticks'])
        # self.gradient.menu.insertAction(self.saveColormapAction, action)
        self.customCmapsMenu.addAction(action)
        self.gradient.length = self.originalLength
        widgets_style.GradientsImage[gradient_name] = gradient_ticks
    
    def removeCustomGradient(self):
        button = self.sender()
        action = button.action
        self.customCmapsMenu.removeAction(action)
        cp = config.ConfigParser()
        cp.read(widgets_style.custom_cmaps_filepath)
        cp.remove_section(f'image.{action.name}')
        with open(widgets_style.custom_cmaps_filepath, mode='w') as file:
            cp.write(file)
    
    def addCustomGradients(self):
        try:
            CustomGradients = widgets_style.getCustomGradients(name='image')
            if not CustomGradients:
                return
            for gradient_name, gradient_ticks in CustomGradients.items():
                self.addCustomGradient(gradient_name, gradient_ticks)
        except Exception as e:
            printl(traceback.format_exc())
            pass
    
    def _askNameColormap(self):
        inputWin = apps.QInput(parent=self._parent, title='Colormap name')
        inputWin.askText('Insert a name for the colormap: ', allowEmpty=False)
        if inputWin.cancel:
            return
        cmapName = inputWin.answer
        return cmapName
    
    def saveColormap(self):
        cmapName = self._askNameColormap()
        if cmapName is None:
            return
        
        cp = config.ConfigParser()
        if os.path.exists(widgets_style.custom_cmaps_filepath):
            cp.read(widgets_style.custom_cmaps_filepath)
        
        SECTION = f'{self.name}.{cmapName}'
        cp[SECTION] = {}

        # gradient_ticks = []
        state = self.gradient.saveState()
        for key, value in state.items():
            if key != 'ticks':
                continue
            for t, tick in enumerate(value):
                pos, rgb = tick
                # gradient_ticks.append((pos, rgb))
                rgb = ','.join([str(c) for c in rgb])
                val = f'{pos},{rgb}'
                cp[SECTION][f'tick_{t}_pos_rgb'] = val
        
        with open(widgets_style.custom_cmaps_filepath, mode='w') as file:
            cp.write(file)
        
        self.addCustomGradient(cmapName, state, restore=False)
    
    def tickColorAccepted(self):
        self.gradient.currentColorAccepted()
        # self.sigTickColorAccepted.emit(self.gradient.colorDialog.color().getRgb())
    
    def setRescaleIntensitiesHow(self, how):
        for action in self.rescaleActionGroup.actions():
            if action.text() == how:
                action.setChecked(True)
                return
    
    def setColormap(
            self, 
            cmap: colors.AcdcColorMap
        ) -> None:
        if isinstance(cmap, str):
            self.gradient.loadPreset(cmap)
            return
        
        lut = []
        for color in cmap:
            qcolor = pg.mkColor(color)
            rgba_float = qcolor.getRgbF()
            rgba_uint = [int(c*255) for c in rgba_float]
            lut.append(rgba_uint)
        
        ncolors = len(lut)
        positions = np.linspace(0, 1, len(lut))
        state = {
            "mode": "rgb",
            "ticks": [
                (float(pos), tuple(color))
                for pos, color in zip(positions, lut)
            ]
        }
        
        self.gradient.restoreState(state)


class myHistogramLUTitem(baseHistogramLUTitem):
    sigGradientMenuEvent = Signal(object)
    sigTickColorAccepted = Signal(object)
    sigAddScaleBar = Signal(bool)
    sigAddTimestamp = Signal(bool)
    sigFontSizeChanged = Signal(int)

    def __init__(
            self, parent=None, name='image', axisLabel='', isViewer=False, 
            **kwargs
        ):
        super().__init__(
            parent=parent, name=name, axisLabel=axisLabel, **kwargs
        )

        self.name = name
        self._parent = parent
        
        self.childLutItem = None

        self.isViewer = isViewer
        if isViewer:
            # In the viewer we don't allow additional settings from the menu
            return
        
        # Add scale bar action
        self.addScaleBarAction = QAction('Add scale bar', self)
        self.addScaleBarAction.setCheckable(True)
        self.addScaleBarAction.triggered.connect(self.emitAddScaleBar)
        self.gradient.menu.addAction(self.addScaleBarAction)
        
        # Add timestamp action
        self.addTimestampAction = QAction('Add timestamp', self)
        self.addTimestampAction.setCheckable(True)
        self.addTimestampAction.triggered.connect(self.emitAddTimestamp)
        self.gradient.menu.addAction(self.addTimestampAction)

        # Invert bw action
        self.invertBwAction = QAction('Invert black/white', self)
        self.invertBwAction.setCheckable(True)
        self.gradient.menu.addAction(self.invertBwAction)

        # Font size menu action
        self.fontSizeMenu = widgets_combo_boxes.FontSizeQMenu(prefix='Text ')
        self.fontSizeMenu.sigFontSizeChanged.connect(
            self.sigFontSizeChanged.emit
        )
        self.gradient.menu.addMenu(self.fontSizeMenu) 

        # Text color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Text color: '))
        self.textColorButton = widgets_push_buttons.myColorButton(color=(255,255,255))
        hbox.addStretch(1)
        hbox.addWidget(self.textColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.textColorButton.click)
        self.gradient.menu.addAction(act)

        # Contours line weight
        contLineWeightMenu = QMenu('Contours line weight', self.gradient.menu)
        self.contLineWightActionGroup = QActionGroup(self)
        self.contLineWightActionGroup.setExclusionPolicy(
            QActionGroup.ExclusionPolicy.Exclusive
        )
        for w in range(1, 11):
            action = QAction(str(w))
            action.setCheckable(True)
            if w == 2:
                action.setChecked(True)
            action.lineWeight = w
            self.contLineWightActionGroup.addAction(action)
            action = contLineWeightMenu.addAction(action)
        self.gradient.menu.addMenu(contLineWeightMenu)

        # Contours color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Contours color: '))
        self.contoursColorButton = widgets_push_buttons.myColorButton(color=(25,25,25))
        hbox.addStretch(1)
        hbox.addWidget(self.contoursColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.contoursColorButton.click)
        self.gradient.menu.addAction(act)

        # Mother-bud line weight
        mothBudLineWeightMenu = QMenu('Mother-bud line weight', self.gradient.menu)
        self.mothBudLineWightActionGroup = QActionGroup(self)
        self.mothBudLineWightActionGroup.setExclusionPolicy(
            QActionGroup.ExclusionPolicy.Exclusive
        )
        for w in range(1, 11):
            action = QAction(str(w))
            action.setCheckable(True)
            if w == 2:
                action.setChecked(True)
            action.lineWeight = w
            self.mothBudLineWightActionGroup.addAction(action)
            action = mothBudLineWeightMenu.addAction(action)
        self.gradient.menu.addMenu(mothBudLineWeightMenu)

        # Mother-bud line color
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Mother-bud line color: '))
        self.mothBudLineColorButton = widgets_push_buttons.myColorButton(color=(255,0,0))
        hbox.addStretch(1)
        hbox.addWidget(self.mothBudLineColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.mothBudLineColorButton.click)
        self.gradient.menu.addAction(act)

        self.labelsAlphaMenu = self.gradient.menu.addMenu(
            'Segm. masks overlay alpha...'
        )
        # self.labelsAlphaMenu.setDisabled(True)
        hbox = QHBoxLayout()
        self.labelsAlphaSlider = widgets_numeric_inputs.sliderWithSpinBox(
            title='Alpha', title_loc='in_line', isFloat=True,
            normalize=True
        )
        self.labelsAlphaSlider.setMaximum(100)
        self.labelsAlphaSlider.setSingleStep(0.05)
        self.labelsAlphaSlider.setValue(0.3)
        hbox.addWidget(self.labelsAlphaSlider)
        shortCutText = 'Command+Up/Down' if is_mac else 'Ctrl+Up/Down'
        hbox.addWidget(QLabel(f'({shortCutText})'))
        widget = QWidget()
        widget.setLayout(hbox)
        act = QWidgetAction(self)
        act.setDefaultWidget(widget)
        self.labelsAlphaMenu.addSeparator()
        self.labelsAlphaMenu.addAction(act)

        # Default settings
        self.defaultSettingsAction = QAction('Restore default settings...', self)
        self.gradient.menu.addAction(self.defaultSettingsAction)

        self.filterObject = FilterObject()
        self.filterObject.sigFilteredEvent.connect(self.gradientMenuEventFilter)
        self.gradient.menu.installEventFilter(self.filterObject)
        self.highlightedAction = None
        self.lastHoveredAction = None
    
    def setFontSize(self, font_size: int, blockSignal=False):
        if blockSignal:
            self.fontSizeMenu.blockSignals(True)
        
        self.fontSizeMenu.setFontSize(font_size)

        if blockSignal:
            self.fontSizeMenu.blockSignals(False)
    
    def setChildLutItem(self, childLutItem):
        self.childLutItem = childLutItem
    
    def removeAddScaleBarAction(self):
        self.gradient.menu.removeAction(self.addScaleBarAction)
    
    def removeAddTimestampAction(self):
        self.gradient.menu.removeAction(self.addTimestampAction)
    
    def emitAddScaleBar(self):
        self.sigAddScaleBar.emit(self.addScaleBarAction.isChecked())
    
    def emitAddTimestamp(self):
        self.sigAddTimestamp.emit(self.addTimestampAction.isChecked())
    
    def gradientMenuEventFilter(self, object, event):
        if event.type() == QtScoped.QEventTypeAttribute('MouseMove'):
            hoveredAction = self.gradient.menu.actionAt(event.pos())
            isActionEntered = (
                hoveredAction != self.lastHoveredAction
            )
            if isActionEntered:
                if isinstance(hoveredAction, highlightableQWidgetAction):
                    # print('Entered a custom action')
                    pass
                isActionLeft = (
                    self.highlightedAction is not None
                    and self.highlightedAction != hoveredAction
                ) 
                if isActionLeft:
                    if isinstance(
                            self.highlightedAction, highlightableQWidgetAction
                        ):
                        # print('Left a custom action')
                        pass
                self.highlightedAction = hoveredAction

            self.lastHoveredAction = hoveredAction
    
    def addOverlayColorButton(self, rgbColor, channelName):
        # Overlay color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Overlay color: '))
        self.overlayColorButton = widgets_push_buttons.myColorButton(color=rgbColor)
        self.overlayColorButton.channel = channelName
        hbox.addStretch(1)
        hbox.addWidget(self.overlayColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.overlayColorButton.click)
        self.gradient.menu.addAction(act)

    def uncheckContLineWeightActions(self):
        for act in self.contLineWightActionGroup.actions():
            try:
                act.toggled.disconnect()
            except Exception as e:
                pass
            act.setChecked(False)

    def uncheckMothBudLineLineWeightActions(self):
        for act in self.mothBudLineWightActionGroup.actions():
            try:
                act.toggled.disconnect()
            except Exception as e:
                pass
            act.setChecked(False)

    def restoreState(self, df):
        if 'textIDsColor' in df.index:
            rgbString = df.at['textIDsColor', 'value']
            r, g, b = colors.rgb_str_to_values(rgbString)
            self.textColorButton.setColor((r, g, b))

        if 'contLineColor' in df.index:
            rgba_str = df.at['contLineColor', 'value']
            rgb = colors.rgba_str_to_values(rgba_str)[:3]
            self.contoursColorButton.setColor(rgb)
        
        if 'contLineWeight' in df.index:
            w = df.at['contLineWeight', 'value']
            w = int(w)
            for action in self.contLineWightActionGroup.actions():
                if action.lineWeight == w:
                    action.setChecked(True)
                    break
        
        if 'mothBudLineWeight' in df.index:
            w = df.at['mothBudLineWeight', 'value']
            w = int(w)
            for action in self.mothBudLineWightActionGroup.actions():
                if action.lineWeight == w:
                    action.setChecked(True)
                    break

        if 'overlaySegmMasksAlpha' in df.index:
            alpha = df.at['overlaySegmMasksAlpha', 'value']
            self.labelsAlphaSlider.setValue(float(alpha))
        
        if 'mothBudLineColor' in df.index:
            rgba_str = df.at['mothBudLineColor', 'value']
            rgb = colors.rgba_str_to_values(rgba_str)[:3]
            self.mothBudLineColorButton.setColor(rgb)
        
        checked = df.at['is_bw_inverted', 'value'] == 'Yes'
        self.invertBwAction.setChecked(checked)

        self.restoreColormap(df)
    
    def super_saveState(self):
        return super().saveState()
    
    def saveState(self, df):
        # remove previous state
        df = df[~df.index.str.contains('img_cmap')].copy()

        state = self.gradient.saveState()
        for key, value in state.items():
            if key == 'ticks':
                for t, tick in enumerate(value):
                    pos, rgb = tick
                    df.at[f'img_cmap_tick{t}_rgb', 'value'] = rgb
                    df.at[f'img_cmap_tick{t}_pos', 'value'] = pos
            else:
                if isinstance(value, bool):
                    value = 'Yes' if value else 'No'
                df.at[f'img_cmap_{key}', 'value'] = value
        return df
    
    def restoreColormap(self, df):
        state = {'mode': 'rgb', 'ticksVisible': True, 'ticks': []}
        ticks_pos = {}
        ticks_rgb = {}
        stateFound = False
        for setting, value in df.itertuples():
            idx = setting.find('img_cmap_')
            if idx == -1:
                continue

            stateFound = True
            m = re.findall(r'tick(\d+)_(\w+)', setting)
            if m:
                tick_idx, tick_type = m[0]
                if tick_type == 'pos':
                    ticks_pos[int(tick_idx)] = float(value)
                elif tick_type == 'rgb':
                    ticks_rgb[int(tick_idx)] = colors.rgba_str_to_values(value)
            else:
                key = setting[9:]
                if value == 'Yes':
                    value = True
                elif value == 'No':
                    value = False
                state[key] = value

        if stateFound:
            ticks = [(0, 0)]*len(ticks_pos)
            for idx, val in ticks_pos.items():
                pos = val
                rgb = ticks_rgb[idx]
                ticks[idx] = (pos, rgb)

            state['ticks'] = ticks
            self.gradient.restoreState(state)

    def regionChanged(self):
        super().regionChanged()
        if self.childLutItem is None:
            return
        
        imageItem = self.imageItem()
        try:
            mn, mx = imageItem.quickMinMax(targetSize=65536)
            # mn and mx can still be NaN if the data is all-NaN
            if mn == mx or imageItem._xp.isnan(mn) or imageItem._xp.isnan(mx):
                mn = 0
                mx = 255
        except AttributeError as err:
            mn, mx = self.getLevels() 
        
        self.childLutItem.setLevels(min=mn, max=mx)


class highlightableQWidgetAction(QWidgetAction):
    def __init__(self, parent) -> None:
        super().__init__(parent)


class BaseLabelsGradientWidget(pg.GradientWidget):
    sigShuffleCmap = Signal()
    sigGreeedyShuffleCmap = Signal()
    
    def __init__( self, *args, parent=None, orientation='right', **kargs):
        pg.GradientEditorItem = BaseGradientEditorItemLabels
        
        pg.GradientWidget.__init__(
            self, *args, parent=parent, orientation=orientation, **kargs
        )

        self._parent = parent
        self.name = 'labels'

        for action in self.menu.actions():
            if action.text() == 'HSV':
                HSV_action = action
            elif action.text() == 'RGB':
                RGB_ation = action
        self.menu.removeAction(HSV_action)
        self.menu.removeAction(RGB_ation)

        # Add custom colormap action
        self.customCmapsMenu = self.menu.addMenu('Custom colormaps')
        self.customCmapsMenu.aboutToShow.connect(self.onShowCustomCmapsMenu)
        self.customCmapsMenu.triggered.connect(self.customCmapsMenuTriggered)
        
        self.saveColormapAction = QAction(
            'Save current colormap...', self
        )
        self.menu.addAction(self.saveColormapAction)
        self.saveColormapAction.triggered.connect(
            self.saveColormap
        )
        
        self.menu.addSeparator()
        
        # Shuffle colors action
        self.shuffleCmapAction =  QAction(
            'Randomly shuffle colormap (Shift+S)', self
        )
        self.menu.addAction(self.shuffleCmapAction)

        self.greedyShuffleCmapAction = QAction(
            'Greedily shuffle colormap (Alt+Shift+S)', self
        )
        self.menu.addAction(self.greedyShuffleCmapAction)

        self.addCustomGradients()
        
        # Background color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Background color: '))
        self.colorButton = widgets_push_buttons.myColorButton(color=(25,25,25))
        hbox.addStretch(1)
        hbox.addWidget(self.colorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.colorButton.click)
        self.menu.addAction(act)
        
        self.shuffleCmapAction.triggered.connect(self.sigShuffleCmap.emit)
        self.greedyShuffleCmapAction.triggered.connect(
            self.sigGreeedyShuffleCmap.emit
        )
    
    def onShowCustomCmapsMenu(self):
        self.customCmapsMenu.show()
        
    def customCmapsMenuTriggered(self, action):
        cmap = action.cmap
        self.item.colorMapMenuClicked(cmap)
        self.item.showTicks(True)
    
    def saveColormap(self):
        cmapName = self._askNameColormap()
        if cmapName is None:
            return
        
        cp = config.ConfigParser()
        if os.path.exists(widgets_style.custom_cmaps_filepath):
            cp.read(widgets_style.custom_cmaps_filepath)
        
        SECTION = f'{self.name}.{cmapName}'
        cp[SECTION] = {}

        state = self.item.saveState()
        for key, value in state.items():
            if key != 'ticks':
                continue
            for t, tick in enumerate(value):
                pos, rgb = tick
                rgb = ','.join([str(c) for c in rgb])
                val = f'{pos},{rgb}'
                cp[SECTION][f'tick_{t}_pos_rgb'] = val
        
        with open(widgets_style.custom_cmaps_filepath, mode='w') as file:
            cp.write(file)
        
        self.addCustomGradient(cmapName, state, restore=False)
    
    def addCustomGradients(self):
        try:
            CustomGradients = widgets_style.getCustomGradients(name='labels')
            if not CustomGradients:
                return
            for gradient_name, gradient_ticks in CustomGradients.items():
                self.addCustomGradient(gradient_name, gradient_ticks)
        except Exception as e:
            printl(traceback.format_exc())
            pass
    
    def _askNameColormap(self):
        inputWin = apps.QInput(parent=self._parent, title='Colormap name')
        inputWin.askText('Insert a name for the colormap: ', allowEmpty=False)
        if inputWin.cancel:
            return
        cmapName = inputWin.answer
        return cmapName

    def addCustomGradient(self, gradient_name, gradient_ticks, restore=True):
        currentState = self.item.saveState()
        self.originalLength = self.item.length
        self.item.length = 100
        if restore:
            self.item.restoreState(gradient_ticks)
        gradient = self.item.getGradient()
        action = CustomGradientMenuAction(gradient, gradient_name, self.item)
        # action.triggered.connect(self.item.contextMenuClicked)
        action.delButton.clicked.connect(self.removeCustomGradient)
        action.cmap = colors.pg_ticks_to_colormap(gradient_ticks['ticks'])
        # self.item.menu.insertAction(self.saveColormapAction, action)
        self.customCmapsMenu.addAction(action)
        self.item.length = self.originalLength
        self.item.restoreState(currentState)
        widgets_style.GradientsLabels[gradient_name] = gradient_ticks

class labelsGradientWidget(BaseLabelsGradientWidget):
    sigShowRightImgToggled = Signal(bool)
    sigShowLabelsImgToggled = Signal(bool)
    sigShowNextFrameToggled = Signal(bool)
    sigFontSizeChanged = Signal(int)

    def __init__( self, *args, parent=None, orientation='right', **kargs):
        super().__init__(
            *args, parent=parent, orientation=orientation, **kargs)

        # Font size menu action
        self.fontSizeMenu = widgets_combo_boxes.FontSizeQMenu(prefix='Text ')
        self.fontSizeMenu.sigFontSizeChanged.connect(
            self.sigFontSizeChanged.emit
        )
        self.menu.addMenu(self.fontSizeMenu)   

        # IDs color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Text color: '))
        self.textColorButton = widgets_push_buttons.myColorButton(color=(25,25,25))
        hbox.addStretch(1)
        hbox.addWidget(self.textColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.textColorButton.click)
        self.menu.addAction(act)   
        self.menu.addSeparator()  
        
        self.permanentGreedyCmapAction = QAction(
            'Always use greedy colormap', self
        )
        self.permanentGreedyCmapAction.setCheckable(True)
        self.menu.addAction(self.permanentGreedyCmapAction)

        # Invert bw action
        self.invertBwAction = QAction('Invert black/white', self)
        self.invertBwAction.setCheckable(True)
        self.menu.addAction(self.invertBwAction)

        # Show labels action
        self.showLabelsImgAction = QAction('Show segmentation image', self)
        self.showLabelsImgAction.setCheckable(True)
        self.menu.addAction(self.showLabelsImgAction)

        # Show right image action
        self.showRightImgAction = QAction('Show duplicated left image', self)
        self.showRightImgAction.setCheckable(True)
        self.menu.addAction(self.showRightImgAction)
        
        # Show next frame action
        self.showNextFrameAction = QAction('Show next frame', self)
        self.showNextFrameAction.setCheckable(True)
        self.menu.addAction(self.showNextFrameAction)

        # Default settings
        self.defaultSettingsAction = QAction(
            'Restore default settings...', self)
        self.menu.addAction(self.defaultSettingsAction)

        self.menu.addSeparator()

        self.showRightImgAction.toggled.connect(self.showRightImageToggled)
        self.showLabelsImgAction.toggled.connect(self.showLabelsImageToggled)
        self.showNextFrameAction.toggled.connect(self.showNextFrameToggled)
    
    def setFontSize(self, font_size: int, blockSignal=False):
        if blockSignal:
            self.fontSizeMenu.blockSignals(True)
        
        self.fontSizeMenu.setFontSize(font_size)

        if blockSignal:
            self.fontSizeMenu.blockSignals(False)
    
    def removeCustomGradient(self):
        button = self.sender()
        action = button.action
        self.customCmapsMenu.removeAction(action)
        cp = config.ConfigParser()
        cp.read(widgets_style.custom_cmaps_filepath)
        cp.remove_section(f'labels.{action.name}')
        with open(widgets_style.custom_cmaps_filepath, mode='w') as file:
            cp.write(file)
    
    def isRightImageVisible(self):
        return (
            self.showLabelsImgAction.isChecked() 
            or self.showNextFrameAction.isChecked() 
        )

    def showRightImageToggled(self, checked):
        if checked and self.isRightImageVisible():
            # Hide the right labels image before showing right image
            self.showLabelsImgAction.setChecked(False)
            self.showNextFrameAction.setChecked(False)
            self.sigShowLabelsImgToggled.emit(False)
            self.sigShowNextFrameToggled.emit(checked)
        self.sigShowRightImgToggled.emit(checked)
    
    def showLabelsImageToggled(self, checked):
        if checked and self.isRightImageVisible():
            # Hide the right image before showing labels image
            self.showRightImgAction.setChecked(False)
            self.showNextFrameAction.setChecked(False)
            self.sigShowRightImgToggled.emit(False)
            self.sigShowNextFrameToggled.emit(False)
        self.sigShowLabelsImgToggled.emit(checked)
    
    def showNextFrameToggled(self, checked):
        if checked and self.isRightImageVisible():
            # Hide the right image before showing labels image
            self.showRightImgAction.setChecked(False)
            self.showLabelsImgAction.setChecked(False)
            self.sigShowRightImgToggled.emit(False)
            self.sigShowLabelsImgToggled.emit(False)
        self.sigShowNextFrameToggled.emit(checked)

    def saveState(self, df):
        # remove previous state
        df = df[~df.index.str.contains('lab_cmap')].copy()

        state = self.item.saveState()
        for key, value in state.items():
            if key == 'ticks':
                for t, tick in enumerate(value):
                    pos, rgb = tick
                    df.at[f'lab_cmap_tick{t}_rgb', 'value'] = rgb
                    df.at[f'lab_cmap_tick{t}_pos', 'value'] = pos
            else:
                if isinstance(value, bool):
                    value = 'Yes' if value else 'No'
                df.at[f'lab_cmap_{key}', 'value'] = value
        return df

    def restoreState(self, df, loadCmap=True):
        # Insert background color
        if 'labels_bkgrColor' in df.index:
            rgbString = df.at['labels_bkgrColor', 'value']
            r, g, b = colors.rgb_str_to_values(rgbString)
            self.colorButton.setColor((r, g, b))

        if 'labels_text_color' in df.index:
            rgbString = df.at['labels_text_color', 'value']
            r, g, b = colors.rgb_str_to_values(rgbString)
            self.textColorButton.setColor((r, g, b))
        else:
            self.textColorButton.setColor((255, 0, 0))

        checked = df.at['is_bw_inverted', 'value'] == 'Yes'
        self.invertBwAction.setChecked(checked)

        if not loadCmap:
            return

        state = {'mode': 'rgb', 'ticksVisible': True, 'ticks': []}
        ticks_pos = {}
        ticks_rgb = {}
        stateFound = False
        for setting, value in df.itertuples():
            idx = setting.find('lab_cmap_')
            if idx == -1:
                continue

            stateFound = True
            m = re.findall(r'tick(\d+)_(\w+)', setting)
            if m:
                tick_idx, tick_type = m[0]
                if tick_type == 'pos':
                    ticks_pos[int(tick_idx)] = float(value)
                elif tick_type == 'rgb':
                    ticks_rgb[int(tick_idx)] = colors.rgba_str_to_values(value)
            else:
                key = setting[9:]
                if value == 'Yes':
                    value = True
                elif value == 'No':
                    value = False
                state[key] = value

        if stateFound:
            ticks = [(0, 0)]*len(ticks_pos)
            for idx, val in ticks_pos.items():
                pos = val
                rgb = ticks_rgb[idx]
                ticks[idx] = (pos, rgb)

            state['ticks'] = ticks
            self.item.restoreState(state)
        else:
            self.item.loadPreset('viridis')

        return stateFound

    def showMenu(self, ev):
        try:
            # Convert QPointF to QPoint
            self.menu.popup(ev.screenPos().toPoint())
        except AttributeError:
            self.menu.popup(ev.screenPos())
