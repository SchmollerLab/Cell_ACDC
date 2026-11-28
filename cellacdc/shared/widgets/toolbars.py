"""ToolBar with the overflow shortcuts of buttons that a too narrow toolbar
hides (_ToolBarOverflowShortcut, _isHiddenByOverflow,
updateToolbarsOverflowShortcuts), ToolBarSeparator, OverlayToolbar,
CheckableAction and the shared tool buttons (rightClickToolButton,
ToolButtonCustomColor, GradientToolButton, PointsLayerToolButton,
OverlayChannelToolButton).
"""
import traceback
from qtpy.QtCore import Signal, Qt, QPointF, QEvent, QObject
from qtpy.QtGui import (
    QColor, QPainter, QIcon, QLinearGradient, QLinearGradient
)
from qtpy.QtWidgets import (
    QLabel, QToolButton, QCheckBox, QWidget, QRadioButton, QSizePolicy,
    QWidgetAction, QAction, QToolBar, QMenu, QButtonGroup, QAbstractButton,
    QShortcut
)
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import html_utils
from cellacdc import plot
from . import basic_widgets as widgets_basic_widgets
from . import numeric_inputs as widgets_numeric_inputs
from . import combo_boxes as widgets_combo_boxes
from . import lut_items as widgets_lut_items


class ToolBarSeparator:
    def __init__(self, width=5, toolbar: QToolBar=None):
        self._parts = (
            widgets_basic_widgets.QHWidgetSpacer(width=width), 
            widgets_basic_widgets.QVLine(),
            widgets_basic_widgets.QHWidgetSpacer(width=width)
        )
        self._actions = []
        self._toolbar = None
        if toolbar is not None:
            self.addToToolbar(toolbar)
    
    def addToToolbar(self, toolbar):
        self._toolbar = toolbar
        for part in self._parts:
            action = toolbar.addWidget(part)
            self._actions.append(action)
    
    def removeFromToolbar(self):
        if self._toolbar is None:
            return
        
        for action in self._actions:
            self._toolbar.removeAction(action)

def _isHiddenByOverflow(toolbar: QToolBar, action: QAction, widget: QWidget):
    """True if `widget` (added to `toolbar` with `addWidget`, returning
    `action`) is hidden only because it does not fit in the toolbar: it
    would be visible in a wider toolbar (action visible, widget enabled)
    and the extension button ">>" is shown."""
    if widget.isVisible():
        return False
    if not action.isVisible() or not widget.isEnabled():
        return False
    extendButton = getattr(toolbar, 'extendButton', None)
    if extendButton is None:
        return True
    # A widget hidden while the toolbar does not overflow is hidden on
    # purpose (e.g., with `widget.hide()`): treat it as hidden
    return extendButton.isVisible()

class _ToolBarOverflowShortcut(QObject):
    """Shortcut of a button added with `QToolBar.addWidget` that works while
    the toolbar overflows and hides the button (see `ToolBar`).

    Qt triggers the shortcut of a widget (`QAbstractButton.setShortcut`)
    only while the widget is visible, and `QToolBar` hides the widgets that
    do not fit (shown again only when the extension button ">>" is
    expanded). This object owns a `QShortcut` of the toolbar (active while
    the toolbar is visible) with the same key as the button, enabled only
    while the button is hidden by the overflow: action visible, button
    enabled, extension button shown. When the button is visible its own
    shortcut works and this one must be disabled, otherwise Qt would find
    two shortcuts for the same key and trigger neither of them.

    The key is copied from the button whenever its visibility or enabled
    state changes; after changing the shortcut of a button that might be
    hidden, call `ToolBar.updateOverflowShortcuts`
    (see `updateToolbarsOverflowShortcuts`).
    """
    _UPDATE_EVENTS = (
        QEvent.Type.Show, QEvent.Type.Hide, QEvent.Type.EnabledChange
    )

    def __init__(self, toolbar: QToolBar, action: QAction, button):
        super().__init__(toolbar)
        self._toolbar = toolbar
        self._action = action
        self._button = button
        self._extendButton = getattr(toolbar, 'extendButton', None)
        self.shortcut = QShortcut(toolbar)
        self.shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        # Holding the key must not toggle the button repeatedly
        self.shortcut.setAutoRepeat(False)
        self.shortcut.setEnabled(False)
        self.shortcut.activated.connect(self._activated)
        button.installEventFilter(self)
        if self._extendButton is not None:
            self._extendButton.installEventFilter(self)
        action.changed.connect(self.update)
        self.update()

    def eventFilter(self, watched, event):
        if event.type() in self._UPDATE_EVENTS:
            self.update()
        return False

    def isButtonHiddenByOverflow(self):
        return _isHiddenByOverflow(self._toolbar, self._action, self._button)

    def update(self):
        try:
            key = self._button.shortcut()
            enabled = not key.isEmpty() and self.isButtonHiddenByOverflow()
        except RuntimeError:
            # Button already deleted
            self.shortcut.setEnabled(False)
            return
        if self.shortcut.key() != key:
            self.shortcut.setKey(key)
        if self.shortcut.isEnabled() != enabled:
            self.shortcut.setEnabled(enabled)

    def _activated(self):
        if self._button.shortcut() != self.shortcut.key():
            # Shortcut of the button changed without an update: the new
            # key is set now, the old one must not click the button
            self.update()
            return
        if not self.isButtonHiddenByOverflow():
            return
        self._button.click()

    def remove(self):
        self.shortcut.setEnabled(False)
        try:
            self._action.changed.disconnect(self.update)
        except (TypeError, RuntimeError):
            pass
        for widget in (self._button, self._extendButton):
            if widget is None:
                continue
            try:
                widget.removeEventFilter(self)
            except RuntimeError:
                pass
        self.shortcut.deleteLater()
        self.deleteLater()

def updateToolbarsOverflowShortcuts(window: QWidget):
    """Copy the current shortcuts of the buttons of every `ToolBar` of
    `window` to the shortcuts that work while the toolbars overflow. Call
    it after changing the shortcuts of toolbar buttons (see
    `_ToolBarOverflowShortcut`)."""
    for toolbar in window.findChildren(ToolBar):
        toolbar.updateOverflowShortcuts()

class ToolBar(QToolBar):
    """`QToolBar` with helpers to add widgets.

    Buttons added with `addWidget` keep their shortcut
    (`QAbstractButton.setShortcut`) while the toolbar overflows and hides
    them, as `QAction`s added with `addAction` do (see
    `_ToolBarOverflowShortcut`). Hide such buttons through the action
    returned by `addWidget` (`action.setVisible(False)`, as Qt requires
    for toolbar widgets), not with `button.hide()`.
    """
    def __init__(self, *args, **kwargs) -> None:
        self._log_func = kwargs.pop('log_func', None)

        super().__init__(*args, **kwargs)

        self.widgetsWithShortcut = {}
        self.widgetsForActions = {}
        self._overflowShortcuts = {}

        for child in self.children():
            if child.objectName() == 'qt_toolbar_ext_button':
                self.extendButton = child
                self.extendButton.setIcon(QIcon(":expand.svg"))
                break

        self._logLabelAdded = False

    def actionEvent(self, event):
        super().actionEvent(event)
        overflowShortcuts = getattr(self, '_overflowShortcuts', None)
        if overflowShortcuts is None:
            return

        action = event.action()
        if event.type() == QEvent.Type.ActionRemoved:
            overflowShortcut = overflowShortcuts.pop(action, None)
            if overflowShortcut is not None:
                overflowShortcut.remove()
            return

        if event.type() != QEvent.Type.ActionAdded:
            return

        if not isinstance(action, QWidgetAction):
            # Buttons of QActions: the action shortcut works already
            return

        button = self.widgetForAction(action)
        if not isinstance(button, QAbstractButton):
            return

        overflowShortcuts[action] = _ToolBarOverflowShortcut(
            self, action, button
        )

    def updateOverflowShortcuts(self):
        """Copy the current button shortcuts to the shortcuts that work
        while the toolbar overflows (see `_ToolBarOverflowShortcut`)."""
        for overflowShortcut in self._overflowShortcuts.values():
            overflowShortcut.update()

    def isWidgetHiddenByOverflow(self, widget: QWidget):
        """True if `widget` (added with `addWidget`) is hidden only because
        the toolbar overflows, i.e. it would be visible if the toolbar were
        wider. Key handlers that require a visible tool button should
        accept it too (e.g., the brush/eraser keys of the main GUI)."""
        for action in self.actions():
            if self.widgetForAction(action) is widget:
                return _isHiddenByOverflow(self, action, widget)
        return False

    def showEvent(self, event):
        if self._log_func is None:
            return super().showEvent(event)
        
        if self._logLabelAdded:
            return super().showEvent(event)
        
        self.addSeparator()
        self._logLabel = self.addLabel()
        self._logLabelAdded = True
        return super().showEvent(event)

    def log(self, msg):
        try:
            self._logLabel.setText(html_utils.span(msg, color='r'))   
        except Exception as err:
            pass
            
        try:
            self._log_func(msg)
        except Exception as err:
            print(msg)
                 
    def addSeparator(self, width=5):
        separator = ToolBarSeparator(width=width, toolbar=self)
        return separator
    
    def removeSeparator(self, separator):
        separator.removeFromToolbar()
    
    def addSpinBox(self, label='', fixedWidth=False):
        spinbox = widgets_numeric_inputs.SpinBox(disableKeyPress=True)
        if label:
            spinbox.label = QLabel(label)
            spinbox.labelAction = self.addWidget(spinbox.label)
        
        if fixedWidth:
            spinbox.setFixedWidth(spinbox.sizeHint().width())
        spinbox.action = self.addWidget(spinbox)
        return spinbox
    
    def addButton(
            self, icon_str: str, text='', checkable=False, ret_widget=False
        ):
        action = QAction(QIcon(icon_str), text, self)
        action.setCheckable(checkable)
        widget = self.addAction(action)
        if widget is None:
            widget = self.widgetForAction(action)
        if ret_widget:
            return action, widget
        return action

    def addComboBox(self, items=None, label=''):
        combobox = widgets_combo_boxes.ComboBox()
        
        if items is not None:
            combobox.addItems(items)
            
        if label:
            combobox.label = QLabel(label)
            combobox.labelAction = self.addWidget(combobox.label)
        
        combobox.action = self.addWidget(combobox)
        return combobox

    def addLabel(self, text=''):
        label = QLabel(text)
        label.action = self.addWidget(label)
        return label
    
    def addCheckBox(self, text='', checked=False):
        checkbox = QCheckBox(text)
        checkbox.setChecked(checked)
        checkbox.action = self.addWidget(checkbox)
        return checkbox
    
    def addSpacing(self, width=5):
        spacer = QWidget()
        spacer.setFixedWidth(width)
        spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Maximum)
        action = self.addWidget(spacer)
        return action
    
    def addStretch(self, stretch=1):
        stretchItem = QWidget()
        stretchItem.setMinimumWidth(stretch)
        stretchItem.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        action = self.addWidget(stretchItem)
        return action
    
    def addRadioButton(self, text: str, checked=False, buttonsGroup=None):
        if buttonsGroup is None:
            buttonsGroup = QButtonGroup()
            buttonsGroup.setExclusive(True)
        radioButton = QRadioButton(text)
        radioButton.setChecked(checked)
        radioButton.action = self.addWidget(radioButton)
        buttonsGroup.addButton(radioButton)
        return radioButton, buttonsGroup


class rightClickToolButton(QToolButton):
    sigRightClick = Signal(object)
    sigLeftClick = Signal(object, object)

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            self.sigLeftClick.emit(self, event)
        elif event.button() == Qt.MouseButton.RightButton:
            self.sigRightClick.emit(event)


class ToolButtonCustomColor(rightClickToolButton):
    def __init__(self, symbol, color='r', parent=None):
        super().__init__(parent=parent)
        if not isinstance(color, QColor):
            color = pg.mkColor(color)

        self.symbol = symbol
        self.setColor(color)

    def setColor(self, color):
        self.penColor = color
        self.brushColor = [0, 0, 0, 100]
        self.brushColor[:3] = color.getRgb()[:3]
    
    def updateSymbol(self, symbol, update=True):
        self.symbol = symbol
        if not update:
            return
        self.update()
    
    def updateColor(self, color, update=True):
        self.setColor(color)
        if not update:
            return
        self.update()
    
    def updateIcon(self, symbol, color):
        self.updateSymbol(symbol)
        self.updateColor(color)
        self.update()

    def paintEvent(self, event):
        QToolButton.paintEvent(self, event)
        try:
            p = QPainter(self)
            w, h = self.width(), self.height()
            sf = 0.6
            p.scale(w*sf, h*sf)
            p.translate(0.5/sf, 0.5/sf)
            symbol = plot.PyQtGraphScatterPlotSymbolPathMatter[self.symbol]
            pen = pg.mkPen(color=self.penColor, width=2)
            brush = pg.mkBrush(color=self.brushColor)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(pen)
            p.setBrush(brush)
            p.drawPath(symbol)
        except Exception as e:
            traceback.print_exc()
        finally:
            p.end()

class GradientToolButton(rightClickToolButton):
    def __init__(self, colors=((255, 0, 0),), parent=None):
        super().__init__(parent=parent)
        self._qcolors = [pg.mkColor(c) for c in colors]
        if len(self._qcolors) < 2:
            self._qcolors.append(self._qcolors[0])
    
    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        pen = pg.mkPen(color=self._qcolors[-1], width=2)

        pad = 7
        
        rect = self.rect().adjusted(pad, pad, -pad, -pad)  # A little padding

        # Gradient: bottom to top
        gradient = QLinearGradient(
            QPointF(rect.bottomLeft()), QPointF(rect.topLeft())
        )
        
         # Set color stops evenly distributed
        num_colors = len(self._qcolors)
        for i, color in enumerate(self._qcolors):
            gradient.setColorAt(i / (num_colors - 1), color)
        
        if not self.isChecked():
            painter.setOpacity(0.4)

        painter.setBrush(gradient)
        painter.setPen(pen)
        painter.drawRect(rect)
        
        painter.end()

class PointsLayerToolButton(ToolButtonCustomColor):
    sigEditAppearance = Signal(object)
    sigShowIdsToggled = Signal(object, bool)
    sigRemove = Signal(object)

    def __init__(self, symbol, color='r', parent=None):
        super().__init__(symbol, color=color, parent=parent)
        self.sigRightClick.connect(self.showContextMenu)
        self.initContextMenu()
    
    def initContextMenu(self):
        contextMenu = QMenu(self)
        contextMenu.addSeparator()

        editAction = QAction('Edit points appearance...')
        editAction.triggered.connect(self.editAppearance)
        contextMenu.addAction(editAction)
        
        removeAction = QAction('Remove points')
        removeAction.triggered.connect(self.emitRemove)
        contextMenu.addAction(removeAction)
        
        showIdsAction = QAction('Show point ids')
        showIdsAction.setCheckable(True)
        showIdsAction.setChecked(True)
        contextMenu.addAction(showIdsAction)
        showIdsAction.toggled.connect(self.emitShowIdsToggled)
        
        self._contextMenu = contextMenu
    
    def showContextMenu(self, event):
        self._contextMenu.exec(event.globalPos())
    
    def emitRemove(self):
        self.sigRemove.emit(self)
    
    def emitShowIdsToggled(self, checked):
        self.sigShowIdsToggled.emit(self, checked)
    
    def editAppearance(self):
        self.sigEditAppearance.emit(self)


class CheckableAction(QAction):
    clicked = Signal(bool)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.setCheckable(True)
        self.toggled.connect(self.emitClicked)
    
    def emitClicked(self, checked):
        self.clicked.emit(checked)
    
    def setChecked(self, checked):
        self.toggled.disconnect()
        super().setChecked(checked)
        self.toggled.connect(self.emitClicked)


class OverlayToolbar(ToolBar):
    sigSetTranspacency = Signal(bool)
    sigSetSingleChannel = Signal(bool)
    # The user clicked an overlay mode (true transparency or alpha-encoded
    # intensity), emitted after both check states were updated. Not emitted
    # when code changes them (e.g., `setTransparent` to export images).
    sigModesClicked = Signal()

    def __init__(self, name='Overlay tools', parent=None):
        
        super().__init__(name, parent)
        
        self.guiWin = parent
        
        self.setContextMenuPolicy(Qt.PreventContextMenu)
        
        self.addSeparator()
        
        self.transparencyCheckbox = self.addCheckBox(
            text='True transparency (RGBA composite)'
        )
        
        self.transparencyCheckbox.setToolTip(
            'Activate to hierarchically blend channels using\n'
            'relative weights derived from the scrollbars.\n\n'
            'Each scrollbar controls the balance between adjacent channels, allowing\n'
            'you to fine-tune their contributions to the final blended image.'
        )

        self.alphaEncodedIntensityCheckbox = self.addCheckBox(
            text='Alpha-encoded intensity (RGBA composite)'
        )
        self.alphaEncodedIntensityCheckbox.setToolTip(
            'Activate to encode pixel intensity as alpha transparency: pixels\n'
            'with zero intensity are fully transparent, 50% intensity\n'
            'corresponds to 50% opacity, and maximum intensity is fully opaque.\n\n'
            'This allows low-intensity regions to appear transparent and\n'
            'creates the perception of seeing through these regions.'
        )
        
        self.addSeparator()
        
        self.singleChannelCheckbox = self.addCheckBox(
            text='Single channel'
        )
        
        self.singleChannelCheckbox.setToolTip(
            'When single channel mode is activated, selecting a\n'
            'channel will display only that channel in the overlay.'
        )
        
        self.transparencyCheckbox.toggled.connect(self.transparencyToggled)
        self.singleChannelCheckbox.toggled.connect(
            self.sigSetSingleChannel.emit
        )
        self.alphaEncodedIntensityCheckbox.toggled.connect(
            self.alphaEncodedIntensityToggled)
        # `clicked` comes after `toggled` (partner checkbox already updated)
        self.transparencyCheckbox.clicked.connect(self.emitModesClicked)
        self.alphaEncodedIntensityCheckbox.clicked.connect(
            self.emitModesClicked
        )

    def emitModesClicked(self, checked=False):
        self.sigModesClicked.emit()

    def setModesChecked(self, transparent=False, alphaEncoded=False):
        """Check the overlay modes without emitting any signal, e.g., to
        restore the check states saved by the user before the overlay is
        on (the host applies them when the overlay is turned on).
        Alpha-encoded intensity implies true transparency, as when clicked.
        """
        states = (
            (self.transparencyCheckbox, transparent or alphaEncoded),
            (self.alphaEncodedIntensityCheckbox, alphaEncoded),
        )
        for checkbox, checked in states:
            wasBlocked = checkbox.blockSignals(True)
            checkbox.setChecked(bool(checked))
            checkbox.blockSignals(wasBlocked)

    def alphaEncodedIntensityToggled(self, checked):
        if checked:
            self.transparencyCheckbox.blockSignals(True)
            self.transparencyCheckbox.setChecked(True)
            self.transparencyCheckbox.blockSignals(False)
        transp_checked = self.transparencyCheckbox.isChecked()
        self.sigSetTranspacency.emit(transp_checked)
        
    def transparencyToggled(self, checked):
        if not checked:
            self.alphaEncodedIntensityCheckbox.blockSignals(True)
            self.alphaEncodedIntensityCheckbox.setChecked(False)
            self.alphaEncodedIntensityCheckbox.blockSignals(False)
        
        self.sigSetTranspacency.emit(checked)
    
    def setTransparent(self, transparent: bool):
        self.transparencyCheckbox.setChecked(transparent)
    
    def isTransparent(self):
        return self.transparencyCheckbox.isChecked()
    
    def isSingleChannel(self):
        return self.singleChannelCheckbox.isChecked()
    
    def isAlphaEncodedIntensity(self):
        return self.alphaEncodedIntensityCheckbox.isChecked()

class OverlayChannelToolButton(GradientToolButton):
    def __init__(
            self, 
            channel_name: str, 
            lut_item: widgets_lut_items.myHistogramLUTitem, 
            shortcut='0',
            parent=None,
        ):
        super().__init__(
            colors=lut_item.gradient.getLookupTable(256), 
            parent=parent
        )
        self._channel_name = channel_name
        
        lut_item.sigGradientChanged.connect(self.updateColors)
        
        self.setToolTip(
            f'Show/hide "{channel_name}" channel\n\n'
            f'Shortcut: {shortcut}'
        )
        
        self.setCheckable(True)
    
    def channelName(self):
        return self._channel_name
    
    def updateColors(self, lut_item):
        colors = lut_item.gradient.getLookupTable(256)
        self._qcolors = [pg.mkColor(c) for c in colors]
        self.update()
    
    def setVisible(self, visible: bool):
        super().setVisible(visible)
        if not hasattr(self, 'action'):
            return
        
        self.action.setVisible(visible)
