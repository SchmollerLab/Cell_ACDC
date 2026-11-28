"""Keyboard handling helpers and the shortcut registry of the main GUI.
keyPressEvent, keyReleaseEvent and eventFilter stay on guiWin in
gui/main_window/window.py. Mixins of guiWin (gui/main_window/window.py)
only: the measurements viewer never composes them.
"""
import re
from functools import partial
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QKeySequence, QKeyEvent
from qtpy.QtWidgets import (
    QAction, QMenu, QAbstractSlider, QWidget, QApplication, QAbstractButton
)
from cellacdc import widgets
from cellacdc.plot import imshow


class KeyboardGuiElements:
    """Keyboard helpers of keyPressEvent, keyReleaseEvent and eventFilter
    (which stay on guiWin): key callbacks, persistent shortcuts, key
    modifiers of the 3D tools, reserved and hard-coded shortcuts and
    copy/paste. customAnnotationDialogKwargs overrides the hook of the
    shared CustomAnnotationsGuiElements.
    """

    def onCtrlPressedFirstTime(self):
        x, y = self.xHoverImg, self.yHoverImg
        if x is None:
            self.xyOnCtrlPressedFirstTime = None
            return

        xdata, ydata = int(x), int(y)
        Y, X = self.currentLab2D.shape

        if not (xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y):
            self.xyOnCtrlPressedFirstTime = None
            return
        
        ID = self.currentLab2D[ydata, xdata]
        if ID == 0:
            self.xyOnCtrlPressedFirstTime = None
            return 
        
        self.xyOnCtrlPressedFirstTime = (xdata, ydata)

    def onCtrlReleased(self):
        self.xyOnCtrlPressedFirstTime = None

    def checkSetDelObjActionActive(self, event, checked=True):
        if self.delObjAction is None and self.is_win:
            return
        
        if self.delObjAction is None:
            # On mac we check for Key_Control
            if event.key() == Qt.Key_Control:
                self.delObjToolAction.setChecked(checked)
                return True

            return False

        delObjKeySequence, delObjQtButton = self.delObjAction
        keySequenceText = widgets.QKeyEventToString(event).rstrip('+')

        if delObjKeySequence is None:
            return False

        delObjKeySequenceText = widgets.macShortcutToWindows(
            delObjKeySequence.toString()
        )
        keySequenceText = widgets.macShortcutToWindows(keySequenceText)

        if keySequenceText.endswith(delObjKeySequenceText):
            self.delObjToolAction.setChecked(checked)
            return True
        
        return False

    def checkTriggerKeyPressShortcuts(self, event: QKeyEvent):
        isBrushKey = event.key() == self.brushButton.keyPressShortcut
        isEraserKey = event.key() == self.eraserButton.keyPressShortcut
        if isBrushKey or isEraserKey:
            return isBrushKey, isEraserKey
        
        modifierText = widgets.modifierKeyToText(event.modifiers())
        for widget in self.widgetsWithShortcut.values():
            if not hasattr(widget, 'keyPressShortcut'):
                continue
            
            if event.key() == widget.keyPressShortcut:
                if widget.isCheckable():
                    widget.setChecked(True)
                else:
                    widget.trigger()       
                continue
            
            shortcutText = widget.keyPressShortcut.toString()
            try:
                mod, key = shortcutText.split('+')
                if modifierText == mod and event.key() == QKeySequence(key):
                    widget.trigger()
                    
            except Exception as e:
                pass
        
        return isBrushKey, isEraserKey

    def _temp_debug(self, id=None):
        posData = self.data[self.pos_i]
        imshow(posData.lab, annotate_labels_idxs=[0])

    def keyPressCheckSetSpinboxValue(self, event, spinbox):
        """Check if the key pressed is a digit and set the spinbox value 
        accordingly."""
        try:
            n = int(event.text())
            if self.typingEditID:
                value = int(f'{spinbox.value()}{n}')
            else:
                value = n
                self.typingEditID = True
            spinbox.setValue(value)
            
            try:
                spinbox.timer.stop()
            except Exception as err:
                pass
            
            spinbox.timer = QTimer(spinbox)
            spinbox.timer.timeout.connect(
                self.editingSpinboxValueTimerCallback
            )
            spinbox.timer.start(2000)
            spinbox.timer.setSingleShot(True)
            success = True
        except Exception as e:
            # printl(traceback.format_exc())
            success = False        
        return success

    def editingSpinboxValueTimerCallback(self):
        self.typingEditID = False

    def ownedWindowsOutsideParentChain(self):
        """Windows of this GUI whose parent chain does not reach it, so that
        `eventFilter` still treats their key presses as the GUI's own (e.g.,
        Left/Right stepping the frames). The 3D viewer is a child of the
        launcher (`mainWin`, for its lifetime), but its frames follow the
        GUI ("Sync frames with parent GUI"). None items are ignored.

        Progress windows parented to the launcher are not included: the GUI
        is disabled while they run.
        """
        return (getattr(self, '_volume_renderer', None),)

    def getFocusedInputWidget(self):
        """Return the widget with keyboard focus if it is a text or number
        input box (e.g., line edit or spin box), None otherwise.

        The key presses that these widgets do not handle (e.g., Enter) are
        passed on to the main window, but they are not meant as shortcuts.
        """
        widgetWithFocus = QApplication.focusWidget()
        if widgetWithFocus is None:
            return

        widgetClassesFilteringEvent = (
            'QLineEdit', 'QTextEdit', 'QPlainTextEdit', 'QAbstractSpinBox'
        )
        for cls in widgetClassesFilteringEvent:
            if widgetWithFocus.inherits(cls):
                return widgetWithFocus

    def isEnterConfirmingTypedValue(self):
        """Return True if Enter (passed on by Qt to the main window)
        confirms the value typed in the input box with the focus, instead
        of applying the active tool
        """
        inputWidget = self.getFocusedInputWidget()
        if inputWidget is None:
            return False

        isValueTyped = getattr(inputWidget, 'isValueTyped', None)
        if isValueTyped is None:
            # E.g., line edits (they get the focus to type in them)
            return True

        # Spin boxes: not if the value was changed only with the arrows
        return isValueTyped()

    def triggerPersistentShortcut(self, key):
        """Trigger (or click) the action with the persistent shortcut `key`
        (`widgetsPersistentShortcut`: Left/Right for the previous/next
        frame), from `eventFilter` and `keyPressEvent`.

        Not when it is disabled on purpose (e.g., `prevAction`/`nextAction`
        while the lost cells prompt is open, `setFrameNavigationDisabled`):
        Qt5's `QAction.trigger()` would still emit `triggered`, and the key
        goes on to its target instead (e.g., to move the focus between the
        prompt's buttons). So while frame navigation is disabled, Left/Right
        do not step the z-slices either (z-slice checkbox or Z held, as
        with Qt6 before); Up/Down still do.

        Returns
        -------
        bool
            Whether an action was triggered (or clicked).
        """
        for name, persistentKey in self.widgetsPersistentShortcut.items():
            if persistentKey != key:
                continue
            action = self.widgetsWithShortcut[name]
            if not action.isEnabled():
                return False
            if hasattr(action, 'click'):
                action.click()
            elif hasattr(action, 'trigger'):
                action.trigger()
            return True
        return False

    def get3DActionKeyboardModifier(self, modifiers):
        if not self.isSegm3D:
            return
        
        is_shift = modifiers == Qt.KeyboardModifier.ShiftModifier
        is_control_shift = modifiers == (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.ShiftModifier
        )
        is_alt_shift = modifiers == (
            Qt.KeyboardModifier.AltModifier
            | Qt.KeyboardModifier.ShiftModifier
        )

        return is_shift, is_control_shift, is_alt_shift

    def get3DZrangeFromKeyModifiers(self, modifiers):
        z_start, z_end = None, None

        if not self.isSegm3D:
            return z_start, z_end

        is_shift, is_control_shift, is_alt_shift = (
            self.get3DActionKeyboardModifier(modifiers)
        )
        if is_control_shift:
            z_start = self.zSliceScrollBar.sliderPosition()
        elif is_alt_shift:
            z_end = self.zSliceScrollBar.sliderPosition()
        elif is_shift:
            z_start = self.zSliceScrollBar.sliderPosition()
            z_end = self.zSliceScrollBar.sliderPosition()
        
        return z_start, z_end

    def applyEditIDKeyModifiers(self, modifiers):
        if not self.editIDbutton.isChecked():
            return False
        
        self._applyKeymodifiersToolbarSegm3D(
            modifiers, self.editIDToolbar
        )

    def applySplitIDsKeyModifiers(self, modifiers):
        if not self.separateBudButton.isChecked():
            return False
        
        self._applyKeymodifiersToolbarSegm3D(
            modifiers, self.separateBudToolbar
        )

    def _applyKeymodifiersToolbarSegm3D(self, modifiers, toolbar):    
        if not self.isSegm3D:
            return False

        is_shift, is_control_shift, is_alt_shift = (
            self.get3DActionKeyboardModifier(modifiers)
        )

        if is_shift:
            toolbar.howToPropagateRadioCurrent.setChecked(True)
            return True
        
        elif is_control_shift:
            toolbar.howToPropagateRadioToEnd.setChecked(True)
            return True
        
        elif is_alt_shift:
            toolbar.howToPropagateRadioToStart.setChecked(True)
            return True
        
        return False

    def resetSplitIDsOptions(self):
        if not self.separateBudButton.isChecked():
            return
        
        if not self.isSegm3D:
            return
        
        self.separateBudToolbar.howToPropagateRadioEntireObj.setChecked(True)

    def resetEditIDOptions(self):
        if not self.editIDbutton.isChecked():
            return
        
        if not self.isSegm3D:
            return
        
        self.editIDToolbar.howToPropagateRadioEntireObj.setChecked(True)

    def doubleKeyTimerCallBack(self):
        if self.isKeyDoublePress:
            self.doubleKeyTimeElapsed = False
            return
        self.doubleKeyTimeElapsed = True
        self.countKeyPress = 0
        if self.Button is None:
            return

        isBrushChecked = self.Button.isChecked()
        if isBrushChecked and self.uncheck:
            self.Button.setChecked(False)
        c = self.defaultToolBarButtonColor
        self.Button.setStyleSheet(f'background-color: {c}')

    def updateBrushCursorOnShiftRelease(self):        
        xdata, ydata = int(self.xHoverImg), int(self.yHoverImg)
        self.setHoverToolSymbolColor(
            xdata, ydata, self.ax2_BrushCirclePen,
            (self.ax2_BrushCircle, self.ax1_BrushCircle),
            self.brushButton, brush=self.ax2_BrushCircleBrush,
            byPassShiftCheck=True
        )
        if self.isSegm3D:
            self.changeBrushID()

    def onShiftReleased(self):
        if self.brushButton.isChecked() and self.xHoverImg is not None:
            self.updateBrushCursorOnShiftRelease()

    def getHardcodedShortcuts(self):
        """Shortcuts displayed (not editable) in the shortcuts editor"""
        hard_shortcuts = {
            'Ctrl+Shift+N': ('New Window', self.newWindowAction),
            'Ctrl+N': ('New Segmentation File', self.newAction),
            'Ctrl+O': ('Load Folder', self.openFolderAction),
            'Shift+P': ('Load Different Position', self.loadPosAction),
            'Ctrl+Shift+S': ('Save as', self.saveAsAction),
            'Ctrl+Shift+V': ('Export to Video', self.exportToVideoAction),
            'Ctrl+Shift+I': ('Export to Image', self.exportToImageAction),
            'Ctrl+Alt+S': ('Save', self.saveAction),
            'Ctrl+S': ('Save Only Segmentation Masks', self.quickSaveAction),
            'Ctrl+Z': ('Undo', self.undoAction),
            'Ctrl+Y': ('Redo', self.redoAction),
            'Ctrl+Shift+A': ('Autopilot', self.autoPilotButton),
            'Ctrl+F': ('Search', self.focusOnSearchAction),
            'Ctrl+T': (
                'Track current frame with real-time tracker', 
                self.repeatTrackingMenuAction
            ),
            'Alt+Shift+T': (
                'Track multiple frames with selected tracker', 
                self.repeatTrackingVideoAction
            ),
            'Ctrl+K': (
                'Customize keyboard shortcuts', self.editShortcutsAction
            ),
            'Ctrl+M': (
                'Show mirrored cursor on images', self.showMirroredCursorAction
            ),
            'Ctrl+Shift+P': (
                'Edit cell cycle annotations', self.manuallyEditCcaAction
            ),
            'Ctrl+P': ('View cell cycle annotations', self.viewCcaTableAction),
            'Shift+S': ('Randomly shuffle colormap', self.shuffleCmapAction),
            'Alt+Shift+S': (
                'Greedily shuffle colormap', self.greedyShuffleCmapAction
            ),
            'Alt+Shift+P': ('Pre-processing', self.preprocessAction),
            'Alt+Shift+C': (
                'Combine channels and segmentation files', 
                self.combineChannelsAction
            ),
            'Ctrl+L': (
                'Relabel IDs sequentially', self.relabelSequentialAction
            ),
            'Left': 'Go to previous frame',
            'Right': 'Go to next frame',
        }
        # Keys of custom annotations and points layers (edited in their own
        # dialogs), so that the editor flags a tool key that collides
        for name, button in self.userKeyButtons():
            shortcut = button.shortcut().toString()
            if shortcut and shortcut not in hard_shortcuts:
                hard_shortcuts[shortcut] = (name, button)
        return hard_shortcuts

    def userKeyButtons(self):
        """[(name, button)] of the buttons whose key is chosen in their own
        dialogs (custom annotations and points layers)."""
        buttons = [
            (f'Annotation "{annot["state"]["name"]}"', button)
            for button, annot in self.customAnnotDict.items()
        ]
        for toolbar in getattr(self, 'pointsLayersToolbars', []):
            for action in toolbar.actions():
                button = toolbar.widgetForAction(action)
                if isinstance(button, widgets.PointsLayerToolButton):
                    name = getattr(action, 'layerType', '') or 'Points layer'
                    buttons.append((f'Points layer "{name}"', button))
        return buttons

    def reservedShortcuts(self, excludeWidget=None):
        """Key sequences already used in the main window {sequence: used by},
        refused as keys of new custom annotations and points layers.

        Includes the keys of every action and tool button of the window in
        any mode (e.g., the edit tools of snapshots are shown next to the
        custom annotations, and two buttons with the same key trigger
        neither), the brush/eraser keys, the key of the "Delete object"
        action (`delObjAction`) and the keys handled in `keyPressEvent`
        (including the overlay digits). The key of `excludeWidget` (e.g.,
        the custom annotation being edited) is not included.

        The dialogs call it on OK (they are not modal), so that keys
        assigned while they are open are refused too."""
        reserved = dict(self.KEY_EVENT_KEYS)
        zoomOutKey = QKeySequence(self.zoomOutKeyValue).toString()
        reserved[zoomOutKey] = 'Zoom out'
        delObjAction = getattr(self, 'delObjAction', None)
        delObjKeySequence = None if delObjAction is None else delObjAction[0]
        if delObjKeySequence is not None:
            # Plain QKeySequence text (`KeySequenceFromText.toString` gives
            # macOS key names on macOS)
            delObjKey = QKeySequence(delObjKeySequence).toString()
            if delObjKey:
                reserved[delObjKey] = 'Delete object'
        for name, widget in self.widgetsWithShortcut.items():
            shortcut = getattr(widget, 'keyPressShortcut', None)
            if shortcut is None:
                continue
            shortcut = QKeySequence(shortcut).toString()
            if shortcut:
                reserved[shortcut] = name
        widgets_ = self.findChildren(QAction) + self.findChildren(QAbstractButton)
        for widget in widgets_:
            if widget is excludeWidget:
                continue
            shortcut = widget.shortcut().toString()
            if not shortcut or shortcut in reserved:
                continue
            owner = widget.parent() if isinstance(widget, QAction) else widget
            if isinstance(owner, QWidget):
                ownerWindow = owner.window()
                if ownerWindow is not self and not isinstance(ownerWindow, QMenu):
                    # e.g., buttons of a dialog parented to this window
                    continue
            text = widget.text() or widget.toolTip().split('\n')[0]
            text = re.sub(r'<[^>]+>', '', text).replace('&', '').strip()
            text = re.sub(r'^Name:\s*', '', text)
            if text.endswith('...'):
                text = text[:-3]
            reserved[shortcut] = text or shortcut
        for name, button in self.userKeyButtons():
            if button is excludeWidget:
                continue
            shortcut = button.shortcut().toString()
            if shortcut:
                reserved[shortcut] = name
        return reserved

    def customAnnotationDialogKwargs(self, state=None):
        button = (
            None if state is None
            else self.customAnnotButtonByName(state['name'])
        )
        # Called on OK (the dialog does not block the window), still without
        # the key of the edited annotation
        return {'reservedShortcuts': partial(self.reservedShortcuts, button)}

    def onKeyHome(self):
        self.zSliceScrollBar.triggerAction(
            QAbstractSlider.SliderAction.SliderSingleStepAdd
        )

    def onKeyEnd(self):
        self.zSliceScrollBar.triggerAction(
            QAbstractSlider.SliderAction.SliderSingleStepSub
        )

    def onKeyPageUp(self):
        isAutoPilotActive = (
            self.autoPilotZoomToObjToggle.isChecked()
            and self.autoPilotZoomToObjToolbar.isVisible()
        )
        if isAutoPilotActive:
            self.pointsLayerAutoPilot('next')
        elif self.zSliceScrollBar.isVisible():
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepAdd
            )

    def onKeyPageDown(self):
        isAutoPilotActive = (
            self.autoPilotZoomToObjToggle.isChecked()
            and self.autoPilotZoomToObjToolbar.isVisible()
        )
        if isAutoPilotActive:
            self.pointsLayerAutoPilot('prev')
        elif self.zSliceScrollBar.isVisible():
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepAdd
            )

    def keyUpCallback(
            self, isBrushActive, isWandActive, isExpandLabelActive, 
            isLabelRoiCircActive
        ):
        isAutoPilotActive = (
            self.autoPilotZoomToObjToggle.isChecked()
            and self.autoPilotZoomToObjToolbar.isVisible()
        )
        if isBrushActive:
            brushSize = self.brushSizeSpinbox.value()
            self.brushSizeSpinbox.setValue(brushSize+1)
        elif isWandActive:
            wandTolerance = self.wandControlsToolbar.toleranceSpinbox.value()
            self.wandControlsToolbar.toleranceSpinbox.setValue(wandTolerance+1)
        elif isExpandLabelActive:
            self.expandLabel(dilation=True)
            self.expandFootprintSize += 1
        elif isLabelRoiCircActive:
            val = self.labelRoiCircularRadiusSpinbox.value()
            self.labelRoiCircularRadiusSpinbox.setValue(val+1)
        elif isAutoPilotActive:
            self.pointsLayerAutoPilot('next')
        else:
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepAdd
            )

    def keyDownCallback(
            self, isBrushActive, isWandActive, isExpandLabelActive, 
            isLabelRoiCircActive
        ):
        isAutoPilotActive = (
            self.autoPilotZoomToObjToggle.isChecked()
            and self.autoPilotZoomToObjToolbar.isVisible()
        )
        if isBrushActive:
            brushSize = self.brushSizeSpinbox.value()
            self.brushSizeSpinbox.setValue(brushSize-1)
        elif isWandActive:
            wandTolerance = self.wandControlsToolbar.toleranceSpinbox.value()
            self.wandControlsToolbar.toleranceSpinbox.setValue(wandTolerance-1)
        elif isExpandLabelActive:
            self.expandLabel(dilation=False)
            self.expandFootprintSize += 1
        elif isLabelRoiCircActive:
            val = self.labelRoiCircularRadiusSpinbox.value()
            self.labelRoiCircularRadiusSpinbox.setValue(val-1)
        elif isAutoPilotActive:
            self.pointsLayerAutoPilot('prev')
        elif self.isNavigateActionOnNextFrame():
            posData = self.data[self.pos_i]
            self.rightImageFramesScrollbar.setValue(posData.frame_i+2)
        else:
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepSub
            )

    def copyContent(self):
        pass

    def pasteContent(self):
        pass

    def cutContent(self):
        pass

    def onEscape(
            self, 
            isTypingIDFunctionChecked=False, 
            buttonsToNotUncheck=None,
            doAutoRange=True    
        ):
        if buttonsToNotUncheck is None:
            buttonsToNotUncheck = set()
            
        if self.keepIDsButton.isChecked() and self.keptObjectsIDs:
            self.keptObjectsIDs = widgets.KeptObjectIDsList(
                self.keptIDsLineEdit, self.keepIDsConfirmAction
            )
            self.highlightHoverIDsKeptObj(0, 0, hoverID=0)
            QTimer.singleShot(300, self.autoRange)
            return

        if self.brushButton.isChecked() and self.typingEditID:
            self.autoIDcheckbox.setChecked(True)
            self.typingEditID = False
            QTimer.singleShot(300, self.autoRange)
            return
        
        if isTypingIDFunctionChecked and self.typingEditID:
            self.typingEditID = False
            QTimer.singleShot(300, self.autoRange)
            return
        
        if self.labelRoiButton.isChecked() and self.isMouseDragImg1:
            self.isMouseDragImg1 = False
            self.labelRoiItem.setPos((0,0))
            self.labelRoiItem.setSize((0,0))
            self.freeRoiItem.clear()
            QTimer.singleShot(300, self.autoRange)
            return
        
        if self.zoomRectButton.isChecked():
            self.zoomRectCancelled()
            QTimer.singleShot(300, self.autoRange)
            return
        
        self.setUncheckedAllButtons(buttonsToNotUncheck=buttonsToNotUncheck)
        self.setUncheckedAllCustomAnnotButtons()
        self.setUncheckedPointsLayers()
        self.clearTempBrushImage()
        self.isMouseDragImg1 = False
        self.typingEditID = False
        self.clearHighlightedID()
        try:
            # check if polyLineRoi is closed
            state = self.polyLineRoi.getState()
            if not ('closed' in state and state['closed'] is True):
                self.polyLineRoi.clearPoints()
        except Exception as e:
            pass
        
        if doAutoRange:
            QTimer.singleShot(11, self.autoRange)
