"""Editing toolbars of the main GUI window (never the measurements viewer):
manual tracking, copy lost object, clear region, manual background,
whitelist IDs, merge IDs, brush strokes cell selection, highlighted ID,
magic wand controls, separate bud, export current view and edit ID
toolbars, with the ID line edits (KeepIDsLineEdit, IDsLineEdit),
KeptObjectIDsList and ToolButtonTextIcon.
"""
import os
import re
from qtpy.QtCore import Signal, Qt, QRegularExpression, QRectF
from qtpy.QtGui import QPainter, QRegularExpressionValidator
from qtpy.QtWidgets import (
    QLabel, QToolButton, QCheckBox, QWidget, QRadioButton, QSizePolicy,
    QComboBox, QDoubleSpinBox, QButtonGroup
)
import qtpy.compat
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import myutils, html_utils
from cellacdc import _palettes
from cellacdc import apps
from cellacdc.shared.widgets import line_edits as widgets_line_edits
from cellacdc.shared.widgets import toolbars as widgets_toolbars
from cellacdc.shared.widgets import message_box as widgets_message_box
from ..mixins import widgets as gui_mixin_widgets


class KeepIDsLineEdit(widgets_line_edits.ValidLineEdit):
    sigIDsChanged = Signal(list)
    sigSort = Signal()
    sigEnterPressed = Signal()

    def __init__(self, instructionsLabel, parent=None):
        super().__init__(parent)

        self.validPattern = '^[0-9-, ]+$'
        regExpr = QRegularExpression(self.validPattern)
        self.setValidator(QRegularExpressionValidator(regExpr))

        self.textChanged.connect(self.onTextChanged)
        self.editingFinished.connect(self.onEditingFinished)

        self.instructionsText = instructionsLabel.text()
        self._label = instructionsLabel
    
    def keyPressEvent(self, event) -> None:
        super().keyPressEvent(event)
        if event.text() == ',':
            self.sigSort.emit()
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            # QLineEdit ignores Enter --> the event would be propagated to
            # the parent widgets (e.g., `guiWin.keyPressEvent`) and Enter
            # would be handled twice. Note that checking the focus in the
            # parent is not enough because the slots connected to
            # `sigEnterPressed` can open dialogs that take the focus.
            event.accept()
            self.sigEnterPressed.emit()
    
    def values(self) -> set:
        text = self.text()
        IDs = []
        rangesMatch = re.findall(r'(\d+-\d+)', text)
        if rangesMatch:
            for rangeText in rangesMatch:
                start, stop = rangeText.split('-')
                start, stop = int(start), int(stop)
                IDs.extend(range(start, stop+1))
            text = re.sub(r'(\d+)-(\d+)', '', text)
        IDsMatch = re.findall(r'(\d+)', text)
        if IDsMatch:
            for ID in IDsMatch:
                IDs.append(int(ID))
        return set(IDs)
    
    def onTextChanged(self, text):
        IDs = self.values()
        self.IDs = sorted(list(set(IDs)))
        self.sigIDsChanged.emit(self.IDs)
    
    def onEditingFinished(self):
        self.sigSort.emit()
    
    def warnNotExistingID(self):
        self.setInvalidStyleSheet()
        self._label.setText(
            '  Some of the IDs are not existing --> they will be IGNORED'
        )
        self._label.setStyleSheet('color: red')

    def warnNoIDsSelected(self):
        self.setInvalidStyleSheet()
        self._label.setText(
            '  No existing IDs selected --> nothing to apply'
        )
        self._label.setStyleSheet('color: red')

    def setInstructionsText(self):
        self.setValidStyleSheet()
        self._label.setText(self.instructionsText)
        self._label.setStyleSheet('')


class KeptObjectIDsList(list):
    def __init__(self, lineEdit, confirmSelectionAction, *args):
        self.lineEdit = lineEdit
        self.lineEdit.setText('')
        self.confirmSelectionAction = confirmSelectionAction
        confirmSelectionAction.setDisabled(True)
        super().__init__(*args)
    
    def setText(self):
        text  = myutils.format_IDs(self)
        
        self.lineEdit.setText(text)
    
    def append(self, element, editText=True):
        super().append(element)
        if editText:
            self.setText()
        # `setText` triggers the slots connected to the line edit (e.g.,
        # `guiWin.updateKeepIDs`) that can remove the element again
        self.confirmSelectionAction.setEnabled(len(self) > 0)

    def remove(self, element, editText=True):
        super().remove(element)
        if editText:
            self.setText()
        self.confirmSelectionAction.setEnabled(len(self) > 0)


class ManualTrackingToolBar(widgets_toolbars.ToolBar):
    sigIDchanged = Signal(int)
    sigDisableGhost = Signal()
    sigClearGhostContour = Signal()
    sigClearGhostMask = Signal()
    sigGhostOpacityChanged = Signal(int)

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.spinboxID = self.addSpinBox(label='ID to track: ')
        self.spinboxID.setMinimum(1)

        self.addSeparator()

        self.showGhostCheckbox = QCheckBox('Show ghost object')
        self.showGhostCheckbox.setChecked(True)
        self.addWidget(self.showGhostCheckbox)

        self.ghostContourRadiobutton = QRadioButton('Contour')
        self.ghostMaskRadiobutton = QRadioButton('Mask ; ')
        self.ghostMaskRadiobutton.setChecked(True)
        self.addWidget(self.ghostContourRadiobutton)
        self.addWidget(self.ghostMaskRadiobutton)

        self.ghostMaskOpacitySpinbox = self.addSpinBox('Mask opacity:  ')
        self.ghostMaskOpacitySpinbox.setMaximum(100)
        self.ghostMaskOpacitySpinbox.setValue(30)

        self.showGhostCheckbox.toggled.connect(self.showGhostCheckboxToggled)
        self.ghostContourRadiobutton.toggled.connect(
            self.ghostContourRadiobuttonToggled
        )
        self.spinboxID.valueChanged.connect(self.IDchanged)

        self.ghostMaskOpacitySpinbox.valueChanged.connect(
            self.ghostOpacityValueChanged
        )

        self.addSeparator()

        self.infoLabel = QLabel('')
        self.addWidget(self.infoLabel)
    
    def showInfo(self, text):
        text = html_utils.paragraph(text, font_color='black')
        self.infoLabel.setText(text)

    def showWarning(self, text):
        text = html_utils.paragraph(f'WARNING: {text}', font_color='red')
        self.infoLabel.setText(text)
    
    def clearInfoText(self):
        self.infoLabel.setText('')
    
    def IDchanged(self, value):
        self.sigIDchanged.emit(value)
    
    def showGhostCheckboxToggled(self, checked):
        disabled = not checked
        self.ghostContourRadiobutton.setDisabled(disabled)
        self.ghostMaskRadiobutton.setDisabled(disabled)
        self.ghostMaskOpacitySpinbox.setDisabled(disabled)
        self.ghostMaskOpacitySpinbox.label.setDisabled(disabled)
        if disabled:
            self.sigDisableGhost.emit()
    
    def ghostContourRadiobuttonToggled(self, checked):
        self.ghostMaskOpacitySpinbox.setDisabled(checked)
        self.ghostMaskOpacitySpinbox.label.setDisabled(checked)
        if checked:
            self.sigClearGhostMask.emit()      
        else:
            self.sigClearGhostContour.emit()
    
    def ghostOpacityValueChanged(self, value):
        self.sigGhostOpacityChanged.emit(value)

class CopyLostObjectToolbar(widgets_toolbars.ToolBar):
    sigCopyAllObjects = Signal(int, int)
    
    def __init__(self, *args) -> None:
        super().__init__(*args)
        
        action, widget = self.addButton(':copyContour_all.svg', ret_widget=True)
        # action.setShortcut('Alt+C')
        action.keyPressShortcut = widgets_line_edits.KeySequenceFromText('Alt+C')
        action.setToolTip(
            'Copy all lost objects\n\n'
            'Shortcut: Alt+C'
        )
        self.widgetsWithShortcut['Copy all lost objects'] = action
        self.widgetsForActions['Copy all lost objects'] = widget
        
        action.triggered.connect(self.emitSigCopyAllObjects)
        
        self.addSeparator()
        
        self.maxOverlapNumberControl = self.addSpinBox(
            label='Maximum overlap to accept lost object [%]: '
        )
        self.maxOverlapNumberControl.setMinimum(0)
        self.maxOverlapNumberControl.setValue(10)
        tooltip = (
            'Maximum overlap to accept lost object [%]\n\n'
            'If the overlap between the lost object and an object already '
            'existing is greater than this value,\n'
            'the lost object will not be added.'
        )
        self.maxOverlapNumberControl.setToolTip(tooltip)
        self.maxOverlapNumberControl.label.setToolTip(tooltip)
        
        self.addSeparator()
        
        self.untilFrameNumberControl = self.addSpinBox(
            label='Copy lost object(s) for the next number of frames: '
        )
        self.untilFrameNumberControl.setMinimum(0)
        self.untilFrameNumberControl.setValue(0)

    def emitSigCopyAllObjects(self):
        self.sigCopyAllObjects.emit(
            self.untilFrameNumberControl.value(), 
            self.maxOverlapNumberControl.value()
        )

class ClearRegionToolbar(widgets_toolbars.ToolBar):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        
        group = QButtonGroup()
        group.setExclusive(True)
        self.clearTouchingObjsRadioButton = QRadioButton(
            'Clear all touching objects'
        )
        self.clearOnlyEnclosedObjsRadioButton = QRadioButton(
            'Clear only fully enclosed objects'
        )
        self.clearOnlyEnclosedObjsRadioButton.setChecked(True)
        group.addButton(self.clearTouchingObjsRadioButton)
        group.addButton(self.clearOnlyEnclosedObjsRadioButton)
        
        self.addWidget(self.clearTouchingObjsRadioButton)
        self.addWidget(self.clearOnlyEnclosedObjsRadioButton)
        
        self.addSeparator()
        
        self.numZslicesUpSpinbox = self.addSpinBox(
            label='Num. of z-slices to clear upwards: '
        )
        self.numZslicesUpSpinbox.setMinimum(0)
        self.numZslicesUpSpinbox.setValue(0)
        
        self.numZslicesDownSpinbox = self.addSpinBox(
            label='Num. of z-slices to clear downwards: '
        )
        self.numZslicesDownSpinbox.setMinimum(0)
        self.numZslicesDownSpinbox.setValue(0)
    
    def setZslicesControlEnabled(self, enabled, SizeZ=None):
        self.numZslicesUpSpinbox.labelAction.setVisible(enabled)
        self.numZslicesUpSpinbox.action.setVisible(enabled)
        
        self.numZslicesDownSpinbox.labelAction.setVisible(enabled)
        self.numZslicesDownSpinbox.action.setVisible(enabled)
        
        if SizeZ is None:
            return
        
        self.numZslicesUpSpinbox.setMaximum(SizeZ)
        self.numZslicesDownSpinbox.setMaximum(SizeZ)
    
    def zRange(self, z_slice, SizeZ):
        if z_slice is None:
            zRange = (0, SizeZ)
            return zRange
        
        numZslicesUp = self.numZslicesUpSpinbox.value()
        numZslicesDown = self.numZslicesDownSpinbox.value()
        
        zmin = z_slice - numZslicesDown
        zmax = z_slice + numZslicesDown + 1
        
        zmin = zmin if zmin >= 0 else 0
        zmax = zmax if zmax <= SizeZ else SizeZ
        
        return (zmin, zmax)

class ManualBackgroundToolBar(widgets_toolbars.ToolBar):
    sigIDchanged = Signal(int)

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.spinboxID = self.addSpinBox(label='Set background of ID ')
        self.spinboxID.setMinimum(1)
        self.spinboxID.valueChanged.connect(self.IDchanged)
        
        self.infoLabel = QLabel('')
        self.addWidget(self.infoLabel)
    
    def IDchanged(self, value):
        self.sigIDchanged.emit(value)
    
    def showWarning(self, text):
        text = html_utils.paragraph(f'WARNING: {text}', font_color='red')
        self.infoLabel.setText(text)
    
    def clearInfoText(self):
        self.infoLabel.setText('')
    
    def doesIDexist(self):
        return not bool(self.infoLabel.text())


class ToolButtonTextIcon(widgets_toolbars.rightClickToolButton):
    def __init__(self, text='', parent=None):
        super().__init__(parent=parent)
        self._text = text
        self._penColor = _palettes.text_pen_color()
    
    def setText(self, text):
        self._text = text
        self.update()
    
    def text(self):
        return self._text
    
    def paintEvent(self, event):
        QToolButton.paintEvent(self, event)
        p = QPainter(self)
        
        pen = pg.mkPen(color=self._penColor, width=2)
        p.setPen(pen)
        
        w, h = self.width(), self.height()
        sf = 0.7
        rect_w = w*sf
        rect_h = h*sf
        x = (w-rect_w)/2
        y = (h-rect_h)/2
        rect = QRectF(x, y, rect_w, rect_h) 
        
        font = p.font()
        font.setBold(True)
        font.setPixelSize(int(h/len(self._text)))
        p.setFont(font)
        
        p.drawText(rect, Qt.AlignCenter, self._text)
        p.end()


class IDsLineEdit(KeepIDsLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setText(self, IDs):
        if not isinstance(IDs, set) and not isinstance(IDs, list):
            raise TypeError('IDs must be a set or list')
        
        formatted_text = myutils.format_IDs(IDs)
        super().setText(formatted_text)

class WhitelistIDsToolbar(widgets_toolbars.ToolBar):
    sigWhitelistChanged = Signal(list)
    sigViewOGIDs = Signal(bool)
    sigWhitelistAccepted = Signal(list)
    sigAddNewIDs = Signal(bool)
    sigLoadOGLabs = Signal()
    sigTrackOGagainstPreviousFrame = Signal(bool)
    
    def __init__(self, addNewIDToggleState, *args) -> None:
        super().__init__(*args)

        whitelistLineEditLabel = QLabel('Whitelist IDs: ')
        self.addWidget(whitelistLineEditLabel)
        
        self.whitelistLineEdit = IDsLineEdit(
            whitelistLineEditLabel, parent=self
        )
        self.whitelistLineEdit.sigEnterPressed.connect(self.accept)
        self.whitelistLineEdit.sigIDsChanged.connect(self.emitWhitelistChanged)
        self.addWidget(self.whitelistLineEdit)

        # accept button
        self.acceptButton = self.addButton(':greenTick.svg')
        self.acceptButton.triggered.connect(self.accept)
        
        self.addSeparator()
        
        self.roiToggle = self.addButton(':ROI.svg', checkable=True)
        self.roiToggle.setChecked(False)
        self.roiToggle.setToolTip(
            'Select IDs by drawing a freehand ROI.\n\n'
            'Draw with the mouse left button to include all objects '
            'fully enclosed in the ROI.\n\n'
            'Draw with the mouse right button to include all objects '
            'touched by the ROI.'
        )

        # add a view OG toggle
        self.viewOGToggle, viewOGWidget = self.addButton(':eye.svg', checkable=True, ret_widget=True)
        viewOGTooltip = (
            'View the non-whitelisted segmentation mask.\n\n'
            'You can activate this to add new IDs to the whitelist,\n'
            'correct tracking errors, etc.'
        )
        self.viewOGToggle.setChecked(True)
        self.viewOGToggle.setToolTip(viewOGTooltip)
        self.viewOGToggle.setShortcut('Shift+K')
        key = 'View the non-whitelisted segmentation mask'
        self.widgetsWithShortcut[key] = self.viewOGToggle
        self.widgetsForActions[key] = viewOGWidget
        
        self.viewOGToggle.toggled.connect(self.emitViewOGIDs)
        self.emitViewOGIDs(True)

        # add a Toggle to add new IDs
        self.addNewIDToggle = QCheckBox(
            'Automatically add new IDs to whitelist'
        )
        self.addNewIDToggle.setChecked(addNewIDToggleState)
        self.addWidget(self.addNewIDToggle)
        self.addNewIDToggle.toggled.connect(self.emitAddNewIDs)
        self.emitAddNewIDs(addNewIDToggleState)
        
        self.addSeparator()

        # add a button to load og df
        self.loadOGButton = self.addButton(':open_file.svg')
        self.loadOGButton.triggered.connect(self.sigLoadOGLabs.emit)
        self.loadOGButton.setToolTip(
            'Select which segmentation mask file to load '
            'as the non-whitelisted masks'
        )

        self.TrackOGagainstPreviousFrameButton = self.addButton(':segment.svg')
        self.TrackOGagainstPreviousFrameButton.triggered.connect(
            self.sigTrackOGagainstPreviousFrame.emit
        )
        self.TrackOGagainstPreviousFrameButton.setToolTip(
            'Track the non-whitelisted segmentation masks against the previous frame and copy over successfull tacks'
        )

        self.addSeparator()

        # add an info button
        self.infoButton = self.addButton(':info.svg')
        self.infoButton.triggered.connect(self.showInfo)
        
        # add a spacer to the toolbar
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.addWidget(spacer)

    def isRoiToggled(self):
        return self.roiToggle.isChecked()
    
    def setIDs(self, IDs):
        if not IDs:
            return
        
        self.whitelistLineEdit.setText(IDs)

    def emitWhitelistChanged(self, whitelist):
        self.sigWhitelistChanged.emit(whitelist)

    def emitViewOGIDs(self, checked):
        self.sigViewOGIDs.emit(checked)
    
    def accept(self):
        try:
            whitelist = self.whitelistLineEdit.IDs
        except AttributeError as e:
            if "has no attribute 'IDs'" in str(e):
                whitelist = list()
        self.viewOGToggle.toggled.disconnect()            
        self.viewOGToggle.setChecked(False)
        self.viewOGToggle.toggled.connect(self.emitViewOGIDs)
        self.sigWhitelistAccepted.emit(whitelist)
    
    def emitAddNewIDs(self, checked):
        self.sigAddNewIDs.emit(checked)

    def showInfo(self):
        msg = widgets_message_box.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            This function is used to track a subset of segmented objects.<br><br>
            
            To add new IDs to the white list, click with left mouse button on the 
            object to add.<br>
            You can also write directly into the <code>Whitelist IDs</code> widget<br>
            and separate the IDs by commas.<br><br>

            You can also use the ROI button to collect all IDs overlapping a region.<br>
            Toggle the ROI button off to keep editing the IDs manually.<br><br>

            On 3D data, the ROI collects IDs across the entire volume.<br><br>
            
            After adding the IDs, click on the "Accept" button to remove the 
            non-whitelisted objects.<br>
            Every time you visit a new frame, the non-whitelisted objects will 
            be removed automatically.<br><br>
            Use the "Eye" button to view the non-whitelisted segmentation masks.<br>
            This will allow you to correct tracking errors, add new IDs to the 
            white list, etc.<br><br>
            
            If you previously saved the whitelisted masks, you can load the 
            non-whitelisted file<br>
            by clicking on the "Load file" button to restart from where you 
            left last time.
        """
        )
        msg.information(self, 'White list IDs', txt)

class MergeIDsToolbar(widgets_toolbars.ToolBar):
    sigIDsChanged = Signal(list)
    sigAccept = Signal(list)
    sigRoiToggled = Signal(bool)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        mergeLineEditLabel = self.addLabel('IDs to merge: ')

        self.mergeLineEdit = IDsLineEdit(
            mergeLineEditLabel, parent=self
        )
        self.mergeLineEdit.sigEnterPressed.connect(self.accept)
        self.mergeLineEdit.sigIDsChanged.connect(self.sigIDsChanged.emit)
        self.addWidget(self.mergeLineEdit)

        self.acceptButton = self.addButton(':greenTick.svg')
        self.acceptButton.setToolTip(
            'Merge all listed IDs into the first ID in the list'
        )
        self.acceptButton.triggered.connect(self.accept)
        
        self.addSeparator()

        self.onlyCurrentZsliceCheckbox = QCheckBox('Only current z-slice')
        self.onlyCurrentZsliceCheckbox.setToolTip(
            'When active on 3D data, collect and merge IDs only on the '
            'currently displayed z-slice instead of across the entire volume'
        )
        self.addWidget(self.onlyCurrentZsliceCheckbox)

        self.infoButton = self.addButton(':info.svg')
        self.infoButton.triggered.connect(self.showInfo)
        
        self.addSeparator()

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.addWidget(spacer)
    
    def setIDs(self, IDs):
        if not IDs:
            return
        
        self.mergeLineEdit.setText(IDs)

    def setOnlyCurrentZsliceEnabled(self, enabled):
        self.onlyCurrentZsliceCheckbox.setVisible(enabled)
        self.onlyCurrentZsliceCheckbox.setEnabled(enabled)
        if not enabled:
            self.onlyCurrentZsliceCheckbox.setChecked(False)

    def isOnlyCurrentZslice(self):
        return self.onlyCurrentZsliceCheckbox.isChecked()

    def accept(self):
        try:
            IDs = self.mergeLineEdit.IDs
        except AttributeError as e:
            if "has no attribute 'IDs'" in str(e):
                IDs = list()
            else:
                raise
        self.sigAccept.emit(IDs)

    def showInfo(self):
        msg = widgets_message_box.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            With this tool, you can merge multiple objects by drawing a freehand 
            ROI.<br><br>
            By drawing with the mouse <b>right button</b>, all objects touched by the 
            ROI will be merged.<br>
            In this mode, the resulting ID will be the last you released the 
            mouse button on.<br><br>
            By drawing with the mouse <b>left button</b>, all objects fully enclosed by the ROI will be merged.<br>
            In this mode, the resulting ID will be the smallest in the selection.<br><br>
            When working with 3D segmentation masks, you can choose whether to 
            merge across the entire volume<br>
            or only on the displayed z-slice.
        """)
        msg.information(self, 'Merge multiple IDs', txt)

class BrushStrokesSelectCellsToolbar(widgets_toolbars.ToolBar):
    """Controls to select cells of the segmentation masks and add them to
    the brush strokes (i.e., the training data) of the promptable models
    trained from brush strokes (e.g., ConvPaint).

    The selection works like the "Keep IDs" controls (same IDs list and
    line edit).
    """
    sigAddToTrainingData = Signal(object)
    sigSelectedIDsEdited = Signal(object, list)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.addToTrainingDataAction = self.addButton(':greenTick.svg')
        self.addToTrainingDataAction.setToolTip(
            'Add the selected cells to the brush strokes (training data) of '
            'the current frame\n\n'
            'The selected cells are added as foreground and the pixels '
            'without segmentation masks as background (pixels that already '
            'have brush strokes are not modified).\n\n'
            'Shortcut: Enter'
        )

        self.addLabel('  Selected IDs: ')
        instructionsText = (
            '  (Click on the cells to select them or press Ctrl+A to select '
            'all. Separate IDs by comma. Use a dash to denote a range of IDs)'
        )
        instructionsLabel = QLabel(instructionsText)
        self.selectedIDsLineEdit = KeepIDsLineEdit(
            instructionsLabel, parent=self
        )
        self.selectedIDsLineEdit.setToolTip(
            'IDs of the selected cells (you can also type them)'
        )
        self.addWidget(self.selectedIDsLineEdit)

        self.addSeparator()

        self.onlyZoomCheckbox = self.addCheckBox(
            'Only in zoomed area', checked=False
        )
        self.onlyZoomCheckbox.setToolTip(
            'If checked, only the cutout of the image that is currently '
            'displayed (zoomed area) is added to the brush strokes.\n\n'
            'This is useful to add only a region where all the cells are '
            'correctly segmented, and to keep the training fast with large '
            'images.'
        )

        self.addWidget(instructionsLabel)
        spacer = QWidget()
        spacer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.addWidget(spacer)

        self.selectedIDs = KeptObjectIDsList(
            self.selectedIDsLineEdit, self.addToTrainingDataAction
        )

        self.addToTrainingDataAction.triggered.connect(
            self.emitSigAddToTrainingData
        )
        self.selectedIDsLineEdit.sigEnterPressed.connect(
            self.emitSigAddToTrainingData
        )
        self.selectedIDsLineEdit.sigIDsChanged.connect(
            self.emitSigSelectedIDsEdited
        )

    def emitSigAddToTrainingData(self):
        if not self.addToTrainingDataAction.isEnabled():
            # No cells selected
            return
        self.sigAddToTrainingData.emit(self)

    def emitSigSelectedIDsEdited(self, IDs):
        self.sigSelectedIDsEdited.emit(self, IDs)

    def setSelectedIDs(self, IDs=None):
        """Replace the selection with `IDs` (empty selection if None)
        without emitting `sigSelectedIDsEdited`."""
        self.selectedIDsLineEdit.blockSignals(True)
        try:
            self.selectedIDs = KeptObjectIDsList(
                self.selectedIDsLineEdit, self.addToTrainingDataAction
            )
            if IDs is not None:
                for ID in IDs:
                    if ID in self.selectedIDs:
                        continue
                    self.selectedIDs.append(int(ID), editText=False)
                self.selectedIDs.setText()
        finally:
            self.selectedIDsLineEdit.blockSignals(False)
        self.selectedIDsLineEdit.setInstructionsText()

    def resetSelection(self):
        self.setSelectedIDs(None)

    def toggleSelectedID(self, ID):
        """Select `ID` or unselect it if it is already selected. Returns
        True if `ID` is selected."""
        ID = int(ID)
        self.selectedIDsLineEdit.blockSignals(True)
        try:
            if ID in self.selectedIDs:
                self.selectedIDs.remove(ID)
                isSelected = False
            else:
                self.selectedIDs.append(ID)
                isSelected = True
        finally:
            self.selectedIDsLineEdit.blockSignals(False)
        self.selectedIDsLineEdit.setInstructionsText()
        return isSelected

    def syncSelectedIDs(self, IDs):
        """Update the selection to `IDs` (e.g., typed in the line edit)
        without modifying the text of the line edit."""
        for ID in list(self.selectedIDs):
            if ID not in IDs:
                self.selectedIDs.remove(ID, editText=False)
        for ID in IDs:
            if ID not in self.selectedIDs:
                self.selectedIDs.append(ID, editText=False)


class HighlightedIDToolbar(widgets_toolbars.ToolBar):
    sigIDChanged = Signal(int)
    
    def __init__(self, name='Highlighted ID', parent=None):
        
        super().__init__(name, parent)
        
        self.spinbox = self.addSpinBox('Highlighted ID: ')
        self.spinbox.valueChanged.connect(self.emitSigIDChanged)
        
        self.addSeparator()
    
    def emitSigIDChanged(self, *args, **kwargs):
        self.sigIDChanged.emit(self.spinbox.value())
    
    def setIDNoSignals(self, ID: int):
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(ID)
        self.spinbox.blockSignals(False)


class WandControlsToolbar(widgets_toolbars.ToolBar):    
    def __init__(self, name='Magic wand controls', parent=None):
        super().__init__(name, parent)
        
        self.toleranceSpinbox = self.addSpinBox('Tolerance [%]: ')
        self.toleranceSpinbox.setMinimum(0)
        self.toleranceSpinbox.setMaximum(100)
        self.toleranceSpinbox.setValue(5)
        self.toleranceSpinbox.setToolTip(
            'The tolerance is calculated as a percentage of the minimum-maximum '
            'pixel values range of the loaded dataset.\n\n'
            'If tolerance is greater than 0, the pixels adjacent to the added '
            'pixels with value within +- tolerance will be considered part of '
            'the object.'
        )
        self.addLabel(r'% of min-max intensity range ')
        
        self.addSeparator()
        
        self.autoFillHolesCheckbox = self.addCheckBox(
            'Auto-fill holes'
        )
        
        self.addSeparator()
        
        self.useConvexHullCheckbox = self.addCheckBox(
            'Use convex hull mask'
        )
        
        self.addSeparator()


class SeparateBudToolbar(widgets_toolbars.ToolBar):
    def __init__(self, parent):
        super().__init__(parent)
        self.seg_3D = False
        self.parent = parent

        self.howToPropagateRadioEntireObj, self.howToPropagateButtonsGroup = (
            self.addRadioButton(
                'Entire object'
            )
        )
        self.howToPropagateRadioEntireObj.setChecked(True)
        self.howToPropagateRadioEntireObj.setToolTip(
            'Change the object ID in the entire object.'
        )

        self.howToPropagateRadioCurrent, _ = (
            self.addRadioButton(
                'Current z-slice', 
                buttonsGroup=self.howToPropagateButtonsGroup
            )
        )
        self.howToPropagateRadioCurrent.setToolTip(
            'Change the object ID only in the current z-slice.\n\n'
            'Shortcut: Shift'
        )

        self.howToPropagateRadioToEnd, _ = (
            self.addRadioButton(
                'From current to last z-slice', 
                buttonsGroup=self.howToPropagateButtonsGroup
            )
        )
        self.howToPropagateRadioToEnd.setToolTip(
            'Change the object ID from the current Z-slice to the last '
            "Z-slice of the object's bounding box (inclusive).\n\n"
            'Shortcut: Shift + Ctrl'
        )

        self.howToPropagateRadioToStart, _ = (
            self.addRadioButton(
                'From current to first z-slice', 
                buttonsGroup=self.howToPropagateButtonsGroup
            )
        )
        self.howToPropagateRadioToStart.setToolTip(
            'Change the object ID from the first Z-slice of the '
            "object's bounding box to the current Z-slice (inclusive).\n\n"
            'Shortcut: Alt + Ctrl'
        )

        self.addSeparator()
        
        sepModeSelector = QComboBox(self)
        sepModeSelector.addItems(['Slice by slice', '3D Convexity defects'])
        self.sepModeSelectorAction = self.addWidget(sepModeSelector)
        sepModeSelector.currentIndexChanged.connect(self.onSepModeChanged)
        self.sepModeSelector = sepModeSelector
        
        separator2D = self.addSeparator()

        self.convexity2DParamWidgets = {}
        self.convexity2DControlActions = list(separator2D._actions)
        float_2d_params = (
            ('eps_percent', 'Epsilon fraction:', 0.01, 0.0, 1.0, 0.01,
             'Contour approximation fraction used by 2D convexity splitting.'),
        )
        for name, label_text, default, minimum, maximum, step, tooltip in float_2d_params:
            label = QLabel(label_text, self)
            selector = QDoubleSpinBox(self)
            selector.setRange(minimum, maximum)
            selector.setSingleStep(step)
            selector.setValue(default)
            label.setToolTip(tooltip)
            selector.setToolTip(tooltip)
            self.convexity2DControlActions.append(self.addWidget(label))
            self.convexity2DControlActions.append(self.addWidget(selector))
            self.convexity2DParamWidgets[name] = (selector, default, float)
            self.convexity2DControlActions.append(self.addSpacing())

        splitDisconnected = gui_mixin_widgets.CheckBox(
            'Split disconnected regions'
        )
        splitDisconnected.setToolTip(
            'Assign separate IDs to disconnected parts before applying '
            'slice-by-slice convexity splitting or 2D splitting.'
        )
        splitDisconnected.setChecked(True)
        self.convexity2DControlActions.append(self.addWidget(splitDisconnected))
        self.convexity2DParamWidgets['split_disconnected2D'] = (
            splitDisconnected, True, bool
        )
        self.convexity2DControlActions.append(self.addSpacing())

        separator3D = self.addSeparator()
        
        float_params = (
            ('smooth_sigma', 'EDT smoothing sigma:', 1.0, 0.0, 1000.0, 0.1,
             'Gaussian smoothing applied to the distance transform in '
             'physical units. Larger values suppress more boundary noise.'),
            ('h_maxima_frac', 'Seed prominence:', 0.25, 0.0, 1.0, 0.01,
             'Minimum seed prominence as a fraction of the maximum distance. '
             'Larger values produce fewer candidate seeds.'),
            ('compactness', 'Compactness:', 0.01, 0.0, 1000.0, 0.01,
             'Watershed compactness weight. Small positive values favor '
             'rounder regions; zero disables the compactness bias.'),
            ('min_volume_frac', 'Min. volume fraction:', 0.1, 0.0, 0.5, 0.01,
             'Reject splits where the smaller child is below this fraction '
             'of the parent volume.'),
            ('min_solidity', 'Min. solidity:', 0.90, 0.0, 1.0, 0.01,
             'Objects at or above this solidity with fewer than two seeds '
             'are not split when the solidity gate is enabled.'),
            ('defect_depth_frac', 'Defect depth fraction:', 0.2, 0.0, 1.0, 0.01,
             'Surface vertices deeper than this fraction of the deepest '
             'convexity defect are used to fit the defect ring.'),
            ('max_ringness', 'Max. ringness:', 0.35, 0.0, 1.0, 0.01,
             'Maximum defect-cloud ringness accepted for plane fitting. '
             'Lower values require a more planar ring.'),
        )
        self.convexity3DParamWidgets = {}
        self.convexityControlActions = []

        self.convexityControlActions.extend(separator3D._actions)

        for name, label_text, default, minimum, maximum, step, tooltip in float_params:
            label = QLabel(label_text, self)
            selector = QDoubleSpinBox(self)
            selector.setRange(minimum, maximum)
            selector.setSingleStep(step)
            label.setToolTip(tooltip)
            selector.setToolTip(tooltip)
            self.convexityControlActions.append(self.addWidget(label))
            self.convexityControlActions.append(self.addWidget(selector))
            self.convexity3DParamWidgets[name] = (selector, default, float)
            spacer = self.addSpacing()
            self.convexityControlActions.append(spacer)

        bool_params = (
            ('split_disconnected3D', 'Split disconnected regions', True,
             'Assign separate IDs to disconnected 3D components before '
             'applying convexity splitting.'),
            ('require_improvement', 'Prefer solidity improvement', True,
             'Prefer candidates where both children are more solid than the '
             'parent. If none improve solidity, rank all valid candidates.'),
            ('gate_on_solidity', 'Gate on solidity', True,
             'Skip objects above the minimum solidity unless the distance '
             'transform contains at least two seeds.'),
        )
        for name, text, default, tooltip in bool_params:
            checkbox = gui_mixin_widgets.CheckBox(text)
            checkbox.setToolTip(tooltip)
            self.convexityControlActions.append(self.addWidget(checkbox))
            self.convexity3DParamWidgets[name] = (checkbox, default, bool)
            spacer = self.addSpacing()
            self.convexityControlActions.append(spacer)

        self.goToSepSliceSeparator = self.addSeparator()
        
        goToSepSliceCheckBox = gui_mixin_widgets.CheckBox('Go to separated slice')
        self.goToSepSliceCheckBoxAction = self.addWidget(
            goToSepSliceCheckBox
        )
        self.goToSepSliceCheckBox = goToSepSliceCheckBox
        
        self.setSegm3D(self.seg_3D)
        self.setSettings()

        goToSepSliceCheckBox.stateChanged.connect(self.saveSettings)
        for widget, _, value_type in self.convexity3DParamWidgets.values():
            if value_type is bool:
                widget.stateChanged.connect(self.saveSettings)
            else:
                widget.valueChanged.connect(self.saveSettings)
        for widget, _, value_type in self.convexity2DParamWidgets.values():
            if value_type is bool:
                widget.stateChanged.connect(self.saveSettings)
            else:
                widget.valueChanged.connect(self.saveSettings)
        sepModeSelector.currentIndexChanged.connect(self.saveSettings)

            
    def setSettings(self):
        parent = self.parent 
        mode = parent._get_setting_value(
            'sep_IDs_mode', '3D Convexity defects', str
        )
        self.sepModeSelector.setCurrentText(mode)
        
        go_to_sep_slice = parent._get_setting_value(
            'go_to_sep_slice', False, bool
        )
        self.goToSepSliceCheckBox.setChecked(go_to_sep_slice)

        for name, (widget, default, value_type) in self.convexity3DParamWidgets.items():
            value = parent._get_setting_value(
                name, default, value_type
            )
            if value_type is bool:
                widget.setChecked(value)
            else:
                widget.setValue(value)

        for name, (widget, default, value_type) in self.convexity2DParamWidgets.items():
            value = parent._get_setting_value(
                name, default, value_type
            )
            if value_type is bool:
                widget.setChecked(value)
            else:
                widget.setValue(value)
        
    def saveSettings(self):
        settings_df = self.parent.df_settings
        
        settings_df.loc[
            'sep_IDs_mode', 'value'
            ] = self.sepModeSelector.currentText()
        settings_df.loc[
            'go_to_sep_slice', 'value'
            ] = self.goToSepSliceCheckBox.isChecked()
        for name, (widget, _, value_type) in self.convexity3DParamWidgets.items():
            value = widget.isChecked() if value_type is bool else widget.value()
            settings_df.loc[name, 'value'] = value
        for name, (widget, _, value_type) in self.convexity2DParamWidgets.items():
            value = widget.isChecked() if value_type is bool else widget.value()
            settings_df.loc[name, 'value'] = value
        
        settings_df.to_csv(self.parent.settings_csv_path)
        
    def onSepModeChanged(self, index):
        mode = self.sepModeSelector.currentText()
        self.setConvexity2DControlsVisible(
            not self.seg_3D or mode == 'Slice by slice'
        )
        self.setConvexityControlsVisible(
            self.seg_3D and mode == '3D Convexity defects'
        )

    def setSegm3D(self, seg_3D=False):
        self.seg_3D = seg_3D
        self.sepModeSelectorAction.setVisible(seg_3D)
        self.goToSepSliceCheckBoxAction.setVisible(seg_3D)
        for action in self.goToSepSliceSeparator._actions:
            action.setVisible(seg_3D)

        self.onSepModeChanged(self.sepModeSelector.currentIndex())

    def setConvexityControlsVisible(self, visible):
        for action in self.convexityControlActions:
            action.setVisible(visible)

    def setConvexity2DControlsVisible(self, visible):
        for action in self.convexity2DControlActions:
            action.setVisible(visible)
            
    def goToSepSliceChecked(self):
        return self.goToSepSliceCheckBox.isChecked()

    def separationMode(self):
        return self.sepModeSelector.currentText()
    
    def setSeparationMode(self, mode):
        self.sepModeSelector.setCurrentText(mode)

    def splitConv3DKwargs(self):
        kwargs = {}
        for name, (widget, _, value_type) in self.convexity3DParamWidgets.items():
            if name == 'split_disconnected3D':
                kwargs['split_disconnected'] = widget.isChecked() if value_type is bool else widget.value()
                continue
            kwargs[name] = (
                widget.isChecked() if value_type is bool else widget.value()
            )
        return kwargs

    def splitConv2DKwargs(self):
        kwargs = {}
        for name, (widget, _, value_type) in self.convexity2DParamWidgets.items():
            if name == 'split_disconnected2D':
                kwargs['split_disconnected'] = widget.isChecked() if value_type is bool else widget.value()
                continue
            kwargs[name] = (
                widget.isChecked() if value_type is bool else widget.value()
            )
        return kwargs
    
    def getHowToPropagateZslices(self, current_z: int):
        how = self.howToPropagateButtonsGroup.checkedButton().text()
        if how == 'Entire object':
            return None, None
        
        if how == 'Current z-slice':
            return None, None
        
        if how == 'From current to last z-slice':
            return current_z, None
        
        return None, current_z

    def setHowToPropagateZslicesVisible(self, isSegm3D: bool):
        self.howToPropagateRadioEntireObj.setVisible(isSegm3D)
        self.howToPropagateRadioCurrent.setVisible(isSegm3D)
        self.howToPropagateRadioToEnd.setVisible(isSegm3D)
        self.howToPropagateRadioToStart.setVisible(isSegm3D)

        self.howToPropagateRadioEntireObj.action.setVisible(isSegm3D)
        self.howToPropagateRadioCurrent.action.setVisible(isSegm3D)
        self.howToPropagateRadioToEnd.action.setVisible(isSegm3D)
        self.howToPropagateRadioToStart.action.setVisible(isSegm3D)


class ExportCurrentViewToolbar(widgets_toolbars.ToolBar):    
    sigExport = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        models = ['Cellpose']

        self.modelCombobox = self.addComboBox(
            models,
            'For the model: '
        )
        self.addSeparator()

        startFolder = myutils.get_export_current_view_gui_most_recent_path()

        self.addWidget(QLabel('Destination folder: '))
        self.dstFolderPathControl = widgets_line_edits.ElidingLineEdit()
        self.dstFolderPathControl.setToolTip(
            'Select destination folder where to export the current view data'
        )
        self.dstFolderPathControl.action = self.addWidget(
            self.dstFolderPathControl
        )
        self.dstFolderPathControl.setText(startFolder)

        self.dstBrowseButton = self.addButton(
            ':folder-open.svg', 
            text='Select destination folder...',
        )
        self.dstBrowseButton.triggered.connect(
            self.selectDstFolder
        )

        self.selectDstFromRecentPathsAction = self.addButton(
            ':open_file.svg', 
            text='Select destination folder from recent paths...',
        )
        self.selectDstFromRecentPathsAction.triggered.connect(
            self.selectDstFromRecentPaths
        )

        self.showInFileManagerAction = self.addButton(
            ':drawer.svg', 
            text='Open folder in file manager',
        )
        self.showInFileManagerAction.triggered.connect(
            self.showInFileManager
        )

        self.addSeparator()

        self.exportAction, button = self.addButton(
            ':file-save.svg', 
            text='Export - Save current view data',
            ret_widget=True
        )
        self.exportAction.setToolTip(
            'Save current view data\n\n'
            'Shortcut: Shift+E'
        )
        self.exportAction.triggered.connect(self.emitExport)
        self.exportAction.keyPressShortcut = widgets_line_edits.KeySequenceFromText('Shift+E')

        self.widgetsWithShortcut['Export - Save current view data'] = (
            self.exportAction
        )
        self.widgetsForActions['Export - Save current view data'] = button

        self.addSpacing(5)

        self.autoExportOnNextCheckbox = self.addCheckBox(
            text='Automatically export when going to next frame/Position'
        )

        self.addSeparator()

        self.infoTextLabel = self.addLabel()
        self.infoTextLabel.setMinimumWidth(50)

        self.addStretch(10)
    
    def selectDstFromRecentPaths(self):
        recent_paths, dates = myutils.get_export_current_view_gui_recent_paths(
            self
        )
        if not recent_paths:
            return
        
        searchLineEditText = (
            'Search recent paths (Ctrl+click to select multiple)...'
        )
        self.recentPathsDialog = apps.SearchableListboxDialog(
            recent_paths,
            title='Select destination folder from recent paths',
            searchLineEditText=searchLineEditText,
            parent=self.parent(),
            multiSelection=False,
            dates=dates,
        )
        self.recentPathsDialog.sigOk.connect(
            self.dstFolderPathControl.setText
        )
        self.recentPathsDialog.show()

    def selectDstFolder(self):
        startFolder = self.dstFolderPathControl.text()
        folder_path = qtpy.compat.getexistingdirectory(
            caption='Select destination folder', 
            basedir=startFolder
        )
        if not folder_path:
            return
        
        self.dstFolderPathControl.setText(folder_path)
        myutils.add_to_export_current_view_gui_recent_paths(folder_path)

    def showInFileManager(self):
        dstFolderPath = self.dstFolderPathControl.text()
        if not os.path.exists(dstFolderPath):
            self.setInfoText(
                html_utils.span('WARNING: Destination folder does not exist')
            )
            return
        
        myutils.showInExplorer(dstFolderPath)

    def isAutoExportOnNext(self):
        return self.autoExportOnNextCheckbox.isChecked()
    
    def dstFolderPath(self):
        return self.dstFolderPathControl.text()
    
    def dstFolderPathSelected(self, folderPath):
        self.dstFolderPathControl.setText(folderPath)
        myutils.addToRecentPaths(folderPath)
    
    def emitExport(self):
        self.sigExport.emit()
    
    def selectedModelName(self):
        return self.modelCombobox.currentText()
    
    def setInfoText(self, text):
        self.infoTextLabel.setText(text)

class EditIDToolbar(widgets_toolbars.ToolBar):    
    def __init__(self, parent=None):
        super().__init__(parent)

        self.howToPropagateRadioEntireObj, self.howToPropagateButtonsGroup = (
            self.addRadioButton(
                'Entire object'
            )
        )
        self.howToPropagateRadioEntireObj.setChecked(True)
        self.howToPropagateRadioEntireObj.setToolTip(
            'Change the object ID in the entire object.'
        )

        self.howToPropagateRadioCurrent, _ = (
            self.addRadioButton(
                'Current z-slice', 
                buttonsGroup=self.howToPropagateButtonsGroup
            )
        )
        self.howToPropagateRadioCurrent.setToolTip(
            'Change the object ID only in the current z-slice.\n\n'
            'Shortcut: Shift'
        )

        self.howToPropagateRadioToEnd, _ = (
            self.addRadioButton(
                'From current to last z-slice', 
                buttonsGroup=self.howToPropagateButtonsGroup
            )
        )
        self.howToPropagateRadioToEnd.setToolTip(
            'Change the object ID from the current Z-slice to the last '
            "Z-slice of the object's bounding box (inclusive).\n\n"
            'Shortcut: Shift + Ctrl'
        )

        self.howToPropagateRadioToStart, _ = (
            self.addRadioButton(
                'From current to first z-slice', 
                buttonsGroup=self.howToPropagateButtonsGroup
            )
        )
        self.howToPropagateRadioToStart.setToolTip(
            'Change the object ID from the first Z-slice of the '
            "object's bounding box to the current Z-slice (inclusive).\n\n"
            'Shortcut: Alt + Ctrl'
        )

        self.addSeparator()

        self.howToPropagateRadioRange, _ = (
            self.addRadioButton(
                'Custom range', 
                buttonsGroup=self.howToPropagateButtonsGroup
            )
        )
        self.howToPropagateRadioRange.setToolTip(
            'Change the object ID within a defined range of Z-slices (last included).'
        )

        self.addSpacing(5)

        self.howToPropagateRangeStartZSpinbox = self.addSpinBox(
            'Start z', fixedWidth=True
        )
        self.howToPropagateRangeStartZSpinbox.setMinimum(0)
        self.howToPropagateRangeStartZSpinbox.setStartValue(1)
        self.howToPropagateRangeStartZSpinbox.setDisabled(True)

        self.addSpacing(5)
            
        self.howToPropagateRangeEndZSpinbox = self.addSpinBox(
            'End z', fixedWidth=True
        )
        self.howToPropagateRangeEndZSpinbox.setMinimum(0)
        self.howToPropagateRangeEndZSpinbox.setStartValue(1)
        self.howToPropagateRangeEndZSpinbox.setDisabled(True)

        self.howToPropagateRadioRange.toggled.connect(
            self.howToPropagateRadioRangeToggled
        )
        self.howToPropagateRangeStartZSpinbox.sigValueChanged.connect(
            self.howToPropagateRangeStartZSpinboxValueChanged
        )
    
    def setStartValue(self, start_value):
        self.howToPropagateRangeStartZSpinbox.setStartValue(start_value)
        self.howToPropagateRangeEndZSpinbox.setStartValue(start_value)
    
    def getStartValue(self):
        return self.howToPropagateRangeStartZSpinbox.getStartValue()
    
    def getHowToPropagateZslices(self):
        how = self.howToPropagateButtonsGroup.checkedButton().text()
        z_range = (
            self.howToPropagateRangeStartZSpinbox.value(), 
            self.howToPropagateRangeEndZSpinbox.value()
        )
        return how, z_range 
    
    def setCurrentZslice(self, currentZslice, SizeZ):
        self.howToPropagateRangeStartZSpinbox.setValue(currentZslice)
        self.howToPropagateRangeStartZSpinbox.setMaximum(SizeZ)
        self.howToPropagateRangeEndZSpinbox.setValue(currentZslice)
        self.howToPropagateRangeEndZSpinbox.setMaximum(SizeZ)
    
    def howToPropagateRangeStartZSpinboxValueChanged(self, value):
        self.howToPropagateRangeEndZSpinbox.setMinimum(value)
        if value > self.howToPropagateRangeEndZSpinbox.value():
            self.howToPropagateRangeEndZSpinbox.setValue(value)

    def howToPropagateRadioRangeToggled(self, checked: bool):
        self.howToPropagateRangeStartZSpinbox.setDisabled(not checked)
        self.howToPropagateRangeEndZSpinbox.setDisabled(not checked)
        start_z = self.howToPropagateRangeStartZSpinbox.value()
        end_z = self.howToPropagateRangeEndZSpinbox.value()
        if start_z > end_z:
            self.howToPropagateRangeEndZSpinbox.setValue(start_z)
