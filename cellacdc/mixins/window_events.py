"""Qt view adapter for main-window and pointer events."""

from __future__ import annotations

import gc
import os
import traceback
import time

from qtpy.QtCore import Qt, QSettings, QTimer
from qtpy.QtGui import QCursor, QFont, QKeyEvent, QKeySequence, QPixmap
from qtpy.QtWidgets import QAbstractSlider, QCheckBox, QMainWindow

from cellacdc import (
    apps,
    exception_handler,
    html_utils,
    is_mac,
    printl,
    qutils,
    widgets,
)
from cellacdc.plot import imshow


_font = QFont()
_font.setPixelSize(11)

from .app_shell import AppShell
from .frame_navigation import FrameNavigation


class WindowEvents(AppShell, FrameNavigation):
    """Extracted from guiWin."""

    def _resizeLeaveSpaceTerminalBelow(self):
        geometry = self.geometry()
        left = geometry.left()
        top = geometry.top()
        width = geometry.width()
        height = geometry.height()
        self.setGeometry(left, top + 10, width, height - 200)

    def _resizeSlidersArea(self):
        self.navigateScrollBar.setFixedHeight(self.newHeight)
        self.zSliceScrollBar.setFixedHeight(self.newHeight)
        self.zSliceOverlay_SB.setFixedHeight(self.newHeight)
        self.zProjComboBox.setFixedHeight(self.newHeight)
        self.zProjOverlay_CB.setFixedHeight(self.newHeight)
        self.navSpinBox.setFixedHeight(self.newHeight)
        self.zSliceSpinbox.setFixedHeight(self.newHeight)
        try:
            self.img1.alphaScrollbar.setFixedHeight(self.newHeight)
        except Exception as e:
            pass
        try:
            for channel, items in self.overlayLayersItems.items():
                alphaScrollbar = items[2]
                alphaScrollbar.setFixedHeight(self.newHeight)
        except:
            pass
        checkBoxStyleSheet = (
            "QCheckBox::indicator {"
            f"width: {self.newCheckBoxesHeight}px;"
            f"height: {self.newCheckBoxesHeight}px"
            "}"
        )
        for i in range(self.annotOptionsLayout.count()):
            widget = self.annotOptionsLayout.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                widget.setStyleSheet(checkBoxStyleSheet)
        for i in range(self.annotOptionsLayoutRight.count()):
            widget = self.annotOptionsLayoutRight.itemAt(i).widget()
            if isinstance(widget, QCheckBox):
                widget.setStyleSheet(checkBoxStyleSheet)
        self.zSliceCheckbox.setStyleSheet(checkBoxStyleSheet)

    def _temp_debug(self, id=None):
        posData = self.data[self.pos_i]
        imshow(posData.lab, annotate_labels_idxs=[0])

    def askCloseAllWindows(self):
        txt = html_utils.paragraph("""
            There are other open windows that were created from this window.
            <br><br>
            If you proceed, the <b>other windows will be closed too.<br>
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, "Open windows", txt, buttonsTexts=("Cancel", "Ok, close now"))
        return msg.cancel

    def changeEvent(self, event):
        try:
            self.delObjToolAction.setChecked(False)
        except Exception as err:
            return

    def changeRightClickToLeftOnMac(self, mouseEvent):
        button = mouseEvent.button()
        if not is_mac:
            return button

        delObjKeySequence, delObjQtButton = self.delObjAction
        if delObjKeySequence is None:
            return button

        if not delObjKeySequence.toString() == "Control":
            return button

        if button != Qt.MouseButton.RightButton:
            return button

        if delObjQtButton == Qt.MouseButton.LeftButton:
            # On mac, pressing "Control" and clicking with left button changes
            # it to a right click button --> here, left click is required for
            # delete object --> force return of left click
            return Qt.MouseButton.LeftButton

        return button

    def checkOverlayToolbuttonClicked(self, event):
        success = False
        try:
            n = int(event.text())
            toolbutton = self.allOverlayToolbuttonsByIdx.get(n, None)
            toolbutton.click()
            success = True
        except Exception as e:
            # printl(traceback.format_exc())
            success = False
        return success

    def checkSetDelObjActionActive(self, event):
        if self.delObjAction is None and self.is_win:
            return

        if self.delObjAction is None:
            # On mac we check for Key_Control
            if event.key() == Qt.Key_Control:
                self.delObjToolAction.setChecked(True)
            return

        delObjKeySequence, delObjQtButton = self.delObjAction
        keySequenceText = widgets.QKeyEventToString(event).rstrip("+")

        if delObjKeySequence is None:
            # self.delObjToolAction.setChecked(True)
            return

        delObjKeySequenceText = widgets.macShortcutToWindows(
            delObjKeySequence.toString()
        )
        keySequenceText = widgets.macShortcutToWindows(keySequenceText)

        # printl(
        #     delObjKeySequence.toString(),
        #     keySequenceText,
        #     delObjKeySequenceText
        # )

        if keySequenceText == delObjKeySequenceText:
            self.delObjToolAction.setChecked(True)

    def checkTriggerKeyPressShortcuts(self, event: QKeyEvent):
        isBrushKey = event.key() == self.brushButton.keyPressShortcut
        isEraserKey = event.key() == self.eraserButton.keyPressShortcut
        if isBrushKey or isEraserKey:
            return isBrushKey, isEraserKey

        modifierText = widgets.modifierKeyToText(event.modifiers())
        for widget in self.widgetsWithShortcut.values():
            if not hasattr(widget, "keyPressShortcut"):
                continue

            if event.key() == widget.keyPressShortcut:
                if widget.isCheckable():
                    widget.setChecked(True)
                else:
                    widget.trigger()
                continue

            shortcutText = widget.keyPressShortcut.toString()
            try:
                mod, key = shortcutText.split("+")
                if modifierText == mod and event.key() == QKeySequence(key):
                    widget.trigger()

            except Exception as e:
                pass

        return isBrushKey, isEraserKey

    def clearMemory(self):
        if not hasattr(self, "data"):
            return
        self.logger.info("Clearing memory...")
        for posData in self.data:
            try:
                del posData.img_data
            except Exception as e:
                pass
            try:
                del posData.segm_data
            except Exception as e:
                pass
            try:
                del posData.ol_data_dict
            except Exception as e:
                pass
            try:
                del posData.fluo_data_dict
            except Exception as e:
                pass
            try:
                del posData.ol_data
            except Exception as e:
                pass
        del self.data

    def closeEvent(self, event):
        self.setDisabled(False)
        cancel = self.checkAskSavePointsLayers()
        if cancel:
            event.ignore()
            return

        self.onEscape()
        self.saveWindowGeometry()

        if self.newWindows:
            cancel = self.askCloseAllWindows()
            if cancel:
                event.ignore()
                return

            for window in self.newWindows:
                window.close()

        if self.slideshowWin is not None:
            self.slideshowWin.close()
        if self.ccaTableWin is not None:
            self.ccaTableWin.close()

        proceed = self.askSaveOnClosing(event)
        if not proceed:
            event.ignore()
            return

        self.autoSaveClose()

        if self.autoSaveActiveWorkers:
            progressWin = apps.QDialogWorkerProgress(
                title="Closing autosaving worker",
                parent=self,
                pbarDesc="Closing autosaving worker...",
            )
            progressWin.show(self.app)
            progressWin.mainPbar.setMaximum(0)
            self.waitCloseAutoSaveWorkerLoop = qutils.QWhileLoop(
                self._waitCloseAutoSaveWorker, period=250
            )
            self.waitCloseAutoSaveWorkerLoop.exec_()
            progressWin.workerFinished = True
            progressWin.close()

        self.stopPreprocWorker()
        self.stopCombineWorker()
        self.stopCcaIntegrityCheckerWorker()

        # Close the inifinte loop of the thread
        if self.lazyLoader is not None:
            self.lazyLoader.exit = True
            self.lazyLoaderWaitCond.wakeAll()
            self.waitReadH5cond.wakeAll()

        if self.storeStateWorker is not None:
            # Close storeStateWorker
            self.storeStateWorker._stop()
            while self.storeStateWorker.isFinished:
                time.sleep(0.05)

        # Block main thread while separate threads closes
        time.sleep(0.1)

        self.clearMemory()

        self.logger.info("Closing GUI logger...")
        self.logger.close()

        if self.lazyLoader is None:
            self.sigClosed.emit(self)

        gc.collect()

    def doubleKeySpacebarTimerCallback(self):
        if self.isKeyDoublePress:
            self.doubleKeyTimeElapsed = False
            return
        self.doubleKeyTimeElapsed = True
        self.countKeyPress = 0

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
        self.Button.setStyleSheet(f"background-color: {c}")

    def doubleRightClickTimerCallBack(self):
        if self.isDoubleRightClick:
            self.doubleRightClickTimeElapsed = False
            return
        self.doubleRightClickTimeElapsed = True
        self.countRightClicks = 0

        # Time to double right click on img1 expired --> single right-click
        self.gui_imgGradShowContextMenu(*self._img1_click_xy)

    def dragEnterEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        if os.path.isdir(file_path):
            exp_path = file_path
            basename = os.path.basename(file_path)
            if basename.find("Position_") != -1 or basename == "Images":
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.acceptProposedAction()

    def dropEvent(self, event):
        event.setDropAction(Qt.CopyAction)
        file_path = event.mimeData().urls()[0].toLocalFile()
        self.logger.info(f'Dragged and dropped path "{file_path}"')
        basename = os.path.basename(file_path)
        if os.path.isdir(file_path):
            exp_path = file_path
            self.openFolder(exp_path=exp_path)
        else:
            self.openFile(file_path=file_path)

    def editingSpinboxValueTimerCallback(self):
        self.typingEditID = False

    def enterEvent(self, event):
        event.accept()
        if self.slideshowWin is not None:
            posData = self.data[self.pos_i]
            mainWinGeometry = self.geometry()
            mainWinLeft = mainWinGeometry.left()
            mainWinTop = mainWinGeometry.top()
            mainWinWidth = mainWinGeometry.width()
            mainWinHeight = mainWinGeometry.height()
            mainWinRight = mainWinLeft + mainWinWidth
            mainWinBottom = mainWinTop + mainWinHeight

            slideshowWinGeometry = self.slideshowWin.geometry()
            slideshowWinLeft = slideshowWinGeometry.left()
            slideshowWinTop = slideshowWinGeometry.top()
            slideshowWinWidth = slideshowWinGeometry.width()
            slideshowWinHeight = slideshowWinGeometry.height()

            # Determine if overlap
            overlap = (slideshowWinTop < mainWinBottom) and (
                slideshowWinLeft < mainWinRight
            )

            autoActivate = (
                self.isDataLoaded
                and not overlap
                and not posData.disableAutoActivateViewerWindow
            )

            if autoActivate:
                # self.setFocus()
                self.activateWindow()

    def gui_createCursors(self):
        pixmap = QPixmap(":wand_cursor.svg")
        self.wandCursor = QCursor(pixmap, 16, 16)

        pixmap = QPixmap(":curv_cursor.svg")
        self.curvCursor = QCursor(pixmap, 16, 16)

        pixmap = QPixmap(":addDelPolyLineRoi_cursor.svg")
        self.polyLineRoiCursor = QCursor(pixmap, 16, 16)

        pixmap = QPixmap(":cross_cursor.svg")
        self.addPointsCursor = QCursor(pixmap, 16, 16)

    def keyDownCallback(
        self, isBrushActive, isWandActive, isExpandLabelActive, isLabelRoiCircActive
    ):
        isAutoPilotActive = (
            self.autoPilotZoomToObjToggle.isChecked()
            and self.autoPilotZoomToObjToolbar.isVisible()
        )
        if isBrushActive:
            brushSize = self.brushSizeSpinbox.value()
            self.brushSizeSpinbox.setValue(brushSize - 1)
        elif isWandActive:
            wandTolerance = self.wandControlsToolbar.toleranceSpinbox.value()
            self.wandControlsToolbar.toleranceSpinbox.setValue(wandTolerance - 1)
        elif isExpandLabelActive:
            self.expandLabel(dilation=False)
            self.expandFootprintSize += 1
        elif isLabelRoiCircActive:
            val = self.labelRoiCircularRadiusSpinbox.value()
            self.labelRoiCircularRadiusSpinbox.setValue(val - 1)
        elif isAutoPilotActive:
            self.pointsLayerAutoPilot("prev")
        elif self.isNavigateActionOnNextFrame():
            posData = self.data[self.pos_i]
            self.rightImageFramesScrollbar.setValue(posData.frame_i + 2)
        else:
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepSub
            )

    def keyPressCheckSetSpinboxValue(self, event, spinbox):
        """Check if the key pressed is a digit and set the spinbox value
        accordingly."""
        try:
            n = int(event.text())
            if self.typingEditID:
                value = int(f"{spinbox.value()}{n}")
            else:
                value = n
                self.typingEditID = True
            spinbox.setValue(value)

            try:
                spinbox.timer.stop()
            except Exception as err:
                pass

            spinbox.timer = QTimer(spinbox)
            spinbox.timer.timeout.connect(self.editingSpinboxValueTimerCallback)
            spinbox.timer.start(2000)
            spinbox.timer.setSingleShot(True)
            success = True
        except Exception as e:
            # printl(traceback.format_exc())
            success = False
        return success

    def keyPressEvent(self, ev):
        ctrl = ev.modifiers() == Qt.ControlModifier
        if ctrl and ev.key() == Qt.Key_D:
            self.resizeLeaveSpaceTerminalBelow()
            return

        if ev.key() == Qt.Key_Q and self.debug:
            try:
                from . import _q_debug

                _q_debug.q_debug(self)
            except Exception as err:
                printl(traceback.format_exc())
                printl('[ERROR]: Error with "_qdebug" module. See Traceback above.')
                pass

        if not self.isDataLoaded:
            self.logger.warning(
                "Data not loaded yet. Key pressing events are not connected."
            )
            return

        if ev.key() == Qt.Key_Control:
            if not ctrl:
                self.wasCtrlPressedFirstTime = True
                self.onCtrlPressedFirstTime()

        if ev.key() == Qt.Key_PageDown:
            self.onKeyPageDown()

        if ev.key() == Qt.Key_PageUp:
            self.onKeyPageUp()

        if ev.key() == Qt.Key_Home:
            self.onKeyHome()

        if ev.key() == Qt.Key_End:
            self.onKeyEnd()

        modifiers = ev.modifiers()
        isAltModifier = modifiers == Qt.AltModifier
        isCtrlModifier = modifiers == Qt.ControlModifier
        isShiftModifier = modifiers == Qt.ShiftModifier

        self.checkSetDelObjActionActive(ev)

        self.isZmodifier = (
            ev.key() == Qt.Key_Z
            and not isAltModifier
            and not isCtrlModifier
            and not isShiftModifier
        )
        if isShiftModifier:
            if self.brushButton.isChecked():
                # Force default brush symbol with shift down
                self.setHoverToolSymbolColor(
                    1,
                    1,
                    self.ax2_BrushCirclePen,
                    (self.ax2_BrushCircle, self.ax1_BrushCircle),
                    self.brushButton,
                    brush=self.ax2_BrushCircleBrush,
                    ID=0,
                )
            if self.isSegm3D:
                self.changeBrushID()

        isAnyModifier = isAltModifier or isCtrlModifier or isShiftModifier
        if not isAnyModifier and self.overlayButton.isChecked():
            isButtonClicked = self.checkOverlayToolbuttonClicked(ev)
            if isButtonClicked:
                return

        isBrushActive = self.brushButton.isChecked() or self.eraserButton.isChecked()
        isManualTrackingActive = self.manualTrackingButton.isChecked()
        isManualBackgroundActive = self.manualBackgroundButton.isChecked()
        isTypingIDFunctionChecked = False
        if self.brushButton.isChecked() and not self.autoIDcheckbox.isChecked():
            success = self.keyPressCheckSetSpinboxValue(ev, self.editIDspinbox)
            isTypingIDFunctionChecked = True

        if isManualTrackingActive:
            isTypingIDFunctionChecked = self.keyPressCheckSetSpinboxValue(
                ev, self.manualTrackingToolbar.spinboxID
            )

        elif isManualBackgroundActive:
            isTypingIDFunctionChecked = self.keyPressCheckSetSpinboxValue(
                ev, self.manualBackgroundToolbar.spinboxID
            )

        addPointsByClickingButton = self.buttonAddPointsByClickingActive()
        if (
            addPointsByClickingButton is not None
            and addPointsByClickingButton.toolbar.isVisible()
        ):
            isTypingIDFunctionChecked = self.keyPressCheckSetSpinboxValue(
                ev, addPointsByClickingButton.rightClickIDSpinbox
            )

        isBrushKey, isEraserKey = self.checkTriggerKeyPressShortcuts(ev)
        isExpandLabelActive = self.expandLabelToolButton.isChecked()
        isWandActive = self.wandToolButton.isChecked()
        isLabelRoiCircActive = (
            self.labelRoiButton.isChecked()
            and self.labelRoiIsCircularRadioButton.isChecked()
        )
        how = self.drawIDsContComboBox.currentText()
        isOverlaySegm = how.find("overlay segm. masks") != -1
        if ev.key() == Qt.Key_Up and not isCtrlModifier:
            self.keyUpCallback(
                isBrushActive, isWandActive, isExpandLabelActive, isLabelRoiCircActive
            )
        elif ev.key() == Qt.Key_Down and not isCtrlModifier:
            self.keyDownCallback(
                isBrushActive, isWandActive, isExpandLabelActive, isLabelRoiCircActive
            )
        elif ev.key() == Qt.Key_Enter or ev.key() == Qt.Key_Return:
            if isTypingIDFunctionChecked:
                self.typingEditID = False
            elif self.keepIDsButton.isChecked():
                self.keepIDsConfirmAction.trigger()
        elif ev.key() == Qt.Key_Escape:
            self.onEscape(isTypingIDFunctionChecked=isTypingIDFunctionChecked)
        elif isAltModifier:
            isCursorSizeAll = self.app.overrideCursor() == Qt.SizeAllCursor
            # Alt is pressed while cursor is on images --> set SizeAllCursor
            if self.xHoverImg is not None and not isCursorSizeAll:
                self.app.setOverrideCursor(Qt.SizeAllCursor)
        elif isCtrlModifier and isOverlaySegm:
            if ev.key() == Qt.Key_Up:
                val = self.imgGrad.labelsAlphaSlider.value()
                delta = 5 / self.imgGrad.labelsAlphaSlider.maximum()
                val = val + delta
                self.imgGrad.labelsAlphaSlider.setValue(val, emitSignal=True)
            elif ev.key() == Qt.Key_Down:
                val = self.imgGrad.labelsAlphaSlider.value()
                delta = 5 / self.imgGrad.labelsAlphaSlider.maximum()
                val = val - delta
                self.imgGrad.labelsAlphaSlider.setValue(val, emitSignal=True)
        elif ev.key() == self.zoomOutKeyValue:
            self.zoomToCells(enforce=True)
            if self.countKeyPress == 0:
                self.isKeyDoublePress = False
                self.countKeyPress = 1
                self.doubleKeyTimeElapsed = False
                self.Button = None
                QTimer.singleShot(400, self.doubleKeyTimerCallBack)
            elif self.countKeyPress == 1 and not self.doubleKeyTimeElapsed:
                self.ax1.autoRange()
                self.isKeyDoublePress = True
                self.countKeyPress = 0
        elif ev.key() == Qt.Key_Space:
            if self.countKeyPress == 0:
                # Single press --> wait that it's not double press
                self.isKeyDoublePress = False
                self.countKeyPress = 1
                self.doubleKeyTimeElapsed = False
                QTimer.singleShot(300, self.doubleKeySpacebarTimerCallback)
            elif self.countKeyPress == 1 and not self.doubleKeyTimeElapsed:
                self.isKeyDoublePress = True
                # Double press --> toggle draw nothing
                self.onDoubleSpaceBar()
                self.countKeyPress = 0
        elif isBrushKey or isEraserKey:
            if isBrushKey:
                self.Button = self.brushButton
            else:
                self.Button = self.eraserButton

            if not self.Button.isVisible():
                return

            if self.countKeyPress == 0:
                # If first time clicking B activate brush and start timer
                # to catch double press of B
                if not self.Button.isChecked():
                    self.uncheck = False
                    self.Button.setChecked(True)
                else:
                    self.uncheck = True
                self.countKeyPress = 1
                self.isKeyDoublePress = False
                self.doubleKeyTimeElapsed = False

                QTimer.singleShot(400, self.doubleKeyTimerCallBack)
            elif self.countKeyPress == 1 and not self.doubleKeyTimeElapsed:
                self.isKeyDoublePress = True
                color = self.Button.palette().button().color().name()
                if color == self.doublePressKeyButtonColor:
                    c = self.defaultToolBarButtonColor
                else:
                    c = self.doublePressKeyButtonColor
                self.Button.setStyleSheet(f"background-color: {c}")
                self.countKeyPress = 0
                if self.xHoverImg is not None:
                    xdata, ydata = int(self.xHoverImg), int(self.yHoverImg)
                    if isBrushKey:
                        self.setHoverToolSymbolColor(
                            xdata,
                            ydata,
                            self.ax2_BrushCirclePen,
                            (self.ax2_BrushCircle, self.ax1_BrushCircle),
                            self.brushButton,
                            brush=self.ax2_BrushCircleBrush,
                        )
                    elif isEraserKey:
                        self.setHoverToolSymbolColor(
                            xdata,
                            ydata,
                            self.eraserCirclePen,
                            (self.ax2_EraserCircle, self.ax1_EraserCircle),
                            self.eraserButton,
                        )

    def keyReleaseEvent(self, ev):
        if self.app.overrideCursor() == Qt.SizeAllCursor:
            self.app.restoreOverrideCursor()
        if ev.key() == Qt.Key_Control:
            self.onCtrlReleased()
        elif ev.key() == Qt.Key_Shift:
            self.onShiftReleased()

        canRepeat = (
            ev.key() == Qt.Key_Left
            or ev.key() == Qt.Key_Right
            or ev.key() == Qt.Key_Up
            or ev.key() == Qt.Key_Down
            or ev.key() == Qt.Key_Control
            or ev.key() == Qt.Key_Backspace
            or self.delObjToolAction.isChecked()
        )

        if canRepeat and ev.isAutoRepeat():
            return

        self.delObjToolAction.setChecked(False)

        if ev.isAutoRepeat() and not ev.key() == Qt.Key_Z:
            if self.warnKeyPressedMsg is not None:
                return
            self.warnKeyPressedMsg = widgets.myMessageBox(
                showCentered=False, wrapText=False
            )
            txt = html_utils.paragraph(f"""
            Please, <b>do not keep the key "{ev.text().upper()}" 
            pressed.</b><br><br>
            It confuses me :)<br><br>
            Thanks!
            """)
            self.warnKeyPressedMsg.warning(self, "Release the key, please", txt)
            self.warnKeyPressedMsg = None
        elif ev.isAutoRepeat() and ev.key() == Qt.Key_Z and self.isZmodifier:
            self.zKeptDown = True
        elif ev.key() == Qt.Key_Z and self.isZmodifier:
            posData = self.data[self.pos_i]
            self.isZmodifier = False
            if not self.zKeptDown and posData.SizeZ > 1:
                self.zSliceCheckbox.setChecked(not self.zSliceCheckbox.isChecked())
            self.zKeptDown = False

    def keyUpCallback(
        self, isBrushActive, isWandActive, isExpandLabelActive, isLabelRoiCircActive
    ):
        isAutoPilotActive = (
            self.autoPilotZoomToObjToggle.isChecked()
            and self.autoPilotZoomToObjToolbar.isVisible()
        )
        if isBrushActive:
            brushSize = self.brushSizeSpinbox.value()
            self.brushSizeSpinbox.setValue(brushSize + 1)
        elif isWandActive:
            wandTolerance = self.wandControlsToolbar.toleranceSpinbox.value()
            self.wandControlsToolbar.toleranceSpinbox.setValue(wandTolerance + 1)
        elif isExpandLabelActive:
            self.expandLabel(dilation=True)
            self.expandFootprintSize += 1
        elif isLabelRoiCircActive:
            val = self.labelRoiCircularRadiusSpinbox.value()
            self.labelRoiCircularRadiusSpinbox.setValue(val + 1)
        elif isAutoPilotActive:
            self.pointsLayerAutoPilot("next")
        else:
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepAdd
            )

    def leaveEvent(self, event):
        if self.slideshowWin is not None:
            posData = self.data[self.pos_i]
            mainWinGeometry = self.geometry()
            mainWinLeft = mainWinGeometry.left()
            mainWinTop = mainWinGeometry.top()
            mainWinWidth = mainWinGeometry.width()
            mainWinHeight = mainWinGeometry.height()
            mainWinRight = mainWinLeft + mainWinWidth
            mainWinBottom = mainWinTop + mainWinHeight

            slideshowWinGeometry = self.slideshowWin.geometry()
            slideshowWinLeft = slideshowWinGeometry.left()
            slideshowWinTop = slideshowWinGeometry.top()
            slideshowWinWidth = slideshowWinGeometry.width()
            slideshowWinHeight = slideshowWinGeometry.height()

            # Determine if overlap
            overlap = (slideshowWinTop < mainWinBottom) and (
                slideshowWinLeft < mainWinRight
            )

            autoActivate = (
                self.isDataLoaded
                and not overlap
                and not posData.disableAutoActivateViewerWindow
            )

            if autoActivate:
                self.slideshowWin.setFocus()
                self.slideshowWin.activateWindow()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton:
            pos = self.resizeBottomLayoutLine.mapFromGlobal(event.globalPos())
            if pos.y() >= 0:
                self.gui_raiseBottomLayoutContextMenu(event)
        return super().mousePressEvent(event)

    def onKeyEnd(self):
        self.zSliceScrollBar.triggerAction(
            QAbstractSlider.SliderAction.SliderSingleStepSub
        )

    def onKeyHome(self):
        self.zSliceScrollBar.triggerAction(
            QAbstractSlider.SliderAction.SliderSingleStepAdd
        )

    def onKeyPageDown(self):
        isAutoPilotActive = (
            self.autoPilotZoomToObjToggle.isChecked()
            and self.autoPilotZoomToObjToolbar.isVisible()
        )
        if isAutoPilotActive:
            self.pointsLayerAutoPilot("prev")
        elif self.zSliceScrollBar.isVisible():
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepAdd
            )

    def onKeyPageUp(self):
        isAutoPilotActive = (
            self.autoPilotZoomToObjToggle.isChecked()
            and self.autoPilotZoomToObjToolbar.isVisible()
        )
        if isAutoPilotActive:
            self.pointsLayerAutoPilot("next")
        elif self.zSliceScrollBar.isVisible():
            self.zSliceScrollBar.triggerAction(
                QAbstractSlider.SliderAction.SliderSingleStepAdd
            )

    def onShiftReleased(self):
        if self.brushButton.isChecked() and self.xHoverImg is not None:
            self.updateBrushCursorOnShiftRelease()

    def readSettings(self):
        settings = QSettings("schmollerlab", "acdc_gui")
        if settings.value("geometry") is not None:
            self.restoreGeometry(settings.value("geometry"))

    def resizeBottomLayoutLineClicked(self, event):
        pass

    def resizeBottomLayoutLineDragged(self, event):
        if not self.img1BottomGroupbox.isVisible():
            return
        newBottomLayoutHeight = self.bottomScrollArea.minimumHeight() - event.y()
        self.bottomScrollArea.setFixedHeight(newBottomLayoutHeight)

    def resizeBottomLayoutLineReleased(self):
        QTimer.singleShot(100, self.autoRange)

    def resizeEvent(self, event):
        if hasattr(self, "ax1"):
            self.ax1.autoRange()

    def resizeLeaveSpaceTerminalBelow(self):
        self.setWindowState(Qt.WindowMaximized)
        QTimer.singleShot(200, self._resizeLeaveSpaceTerminalBelow)

    def resizeSlidersArea(self, fontSizeFactor=None, heightFactor=None):
        global _font
        if heightFactor is None:
            self.newCheckBoxesHeight = self.checkBoxesHeight
            self.newHeight = self.h
        else:
            self.newHeight = round(self.h * heightFactor)
            self.newCheckBoxesHeight = round(self.checkBoxesHeight * heightFactor)

        if fontSizeFactor is None:
            newFontSize = self.fontPixelSize
        else:
            newFontSize = round(self.fontPixelSize * fontSizeFactor)
        newFont = QFont()
        newFont.setPixelSize(newFontSize)
        _font = newFont
        self.zProjComboBox.setFont(newFont)
        self.t_label.setFont(newFont)
        self.zProjOverlay_CB.setFont(newFont)
        self.annotateRightHowCombobox.setFont(newFont)
        self.drawIDsContComboBox.setFont(newFont)
        self.showTreeInfoCheckbox.setFont(newFont)
        self.highlightZneighObjCheckbox.setFont(newFont)
        self.navSpinBox.setFont(newFont)
        self.zSliceSpinbox.setFont(newFont)
        self.SizeZlabel.setFont(newFont)
        self.navSizeLabel.setFont(newFont)
        self.overlay_z_label.setFont(newFont)
        self.img1BottomGroupbox.setFont(newFont)
        self.rightBottomGroupbox.setFont(newFont)
        try:
            self.img1.alphaScrollbar.label.setFont(newFont)
        except Exception as e:
            pass
        for i in range(self.annotOptionsLayout.count()):
            widget = self.annotOptionsLayout.itemAt(i).widget()
            widget.setFont(newFont)
        for i in range(self.annotOptionsLayoutRight.count()):
            widget = self.annotOptionsLayoutRight.itemAt(i).widget()
            widget.setFont(newFont)
        try:
            for channel, items in self.overlayLayersItems.items():
                alphaScrollbar = items[2]
                alphaScrollbar.label.setFont(newFont)
        except:
            pass
        QTimer.singleShot(100, self._resizeSlidersArea)

    def saveWindowGeometry(self):
        settings = QSettings("schmollerlab", "acdc_gui")
        settings.setValue("geometry", self.saveGeometry())

    def show(self):
        self.setFont(_font)
        QMainWindow.show(self)

        self.setWindowState(Qt.WindowNoState)
        self.setWindowState(Qt.WindowActive)
        self.raise_()

        self.readSettings()
        self.storeDefaultAndCustomColors()

        self.h = self.navSpinBox.size().height()
        fontSizeFactor = None
        heightFactor = None
        if "bottom_sliders_zoom_perc" in self.df_settings.index:
            val = int(self.df_settings.at["bottom_sliders_zoom_perc", "value"])
            if val != 100:
                fontSizeFactor = val / 100
                heightFactor = val / 100

        self.defaultWidgetHeightBottomLayout = self.h
        self.checkBoxesHeight = 14
        self.fontPixelSize = 11
        self.defaultBottomLayoutHeight = self.img1BottomGroupbox.height()

        self.bottomLayout.setStretch(0, 0)
        self.bottomLayout.addSpacing(self.quickSettingsGroupbox.width())
        self.resizeSlidersArea(fontSizeFactor=fontSizeFactor, heightFactor=heightFactor)
        self.bottomScrollArea.hide()

        self.gui_initImg1BottomWidgets()
        self.img1BottomGroupbox.hide()

        w = self.showPropsDockButton.width()
        h = self.showPropsDockButton.height()

        self.showPropsDockButton.setMaximumWidth(15)
        self.showPropsDockButton.setMaximumHeight(120)

        for toolbar in self.controlToolBars:
            toolbar.setMinimumHeight(self.secondLevelToolbar.sizeHint().height())

        self.graphLayout.setFocus()

    def showEvent(self, event):
        if self.mainWin is not None:
            if not self.mainWin.isMinimized():
                return
            self.mainWin.showAllWindows()
        # self.setFocus()
        self.activateWindow()

    def stopPreprocWorker(self):
        self.logger.info("Closing pre-processing worker...")
        try:
            self.preprocWorker.stop()
        except Exception as err:
            pass

    def storeDefaultAndCustomColors(self):
        c = self.overlayButton.palette().button().color().name()
        self.defaultToolBarButtonColor = c
        self.doublePressKeyButtonColor = "#fa693b"

    def super_show(self):
        super().show()

    def updateBrushCursorOnShiftRelease(self):
        xdata, ydata = int(self.xHoverImg), int(self.yHoverImg)
        self.setHoverToolSymbolColor(
            xdata,
            ydata,
            self.ax2_BrushCirclePen,
            (self.ax2_BrushCircle, self.ax1_BrushCircle),
            self.brushButton,
            brush=self.ax2_BrushCircleBrush,
            byPassShiftCheck=True,
        )
        if self.isSegm3D:
            self.changeBrushID()
